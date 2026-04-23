#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from robotarm_executor.chair_grasp_moveit_diffusion_policy import (
    ARM_JOINT_NAMES,
    BASE_DEFAULT_PARAMS,
    OpenVLADatasetCollector,
    apply_gripper,
    copy_joint_state,
    gripper_width_from_joint_state,
)


THIS_DIR = Path(__file__).resolve().parent


def resolve_robotarm_project_root() -> Path:
    env_root = os.environ.get("ROBOTARM_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "pi0_lite" / "pi0_lite.py").exists():
            return candidate

    search_roots = [THIS_DIR, Path.cwd().resolve()]
    for root in search_roots:
        for candidate in [root, *root.parents]:
            direct = candidate / "pi0_lite" / "pi0_lite.py"
            src_layout = candidate / "src" / "robotarm_project" / "pi0_lite" / "pi0_lite.py"
            if direct.exists():
                return candidate
            if src_layout.exists():
                return candidate / "src" / "robotarm_project"

    raise RuntimeError(
        "Could not locate robotarm_project root. "
        "Set ROBOTARM_PROJECT_ROOT to the directory containing pi0_lite."
    )


PROJECT_ROOT = resolve_robotarm_project_root()
PI0_LITE_DIR = PROJECT_ROOT / "pi0_lite"
DEFAULT_PI0_LITE_CHECKPOINT = str(PI0_LITE_DIR / "pi0_lite_output" / "pi0_lite_final.pt")

if str(PI0_LITE_DIR) not in sys.path:
    sys.path.insert(0, str(PI0_LITE_DIR))

from pi0_lite import predict_single  # noqa: E402


PI0_LITE_EXTRA_DEFAULTS = {
    "pi0_lite_checkpoint_path": DEFAULT_PI0_LITE_CHECKPOINT,
    "pi0_lite_device": "cuda",
    "pi0_lite_steps": 32,
    "pi0_lite_image_size": 224,
    "policy_max_steps": 20,
    "policy_timeout_sec": 120.0,
    "policy_action_duration": 0.5,
    "policy_action_rate_hz": 30.0,
    "max_joint_delta_rad": 0.08,
    "gripper_delta_close_threshold": -0.01,
    "gripper_delta_open_threshold": 0.002,
    "gripper_close_min_ee_z": 0.30,
    "gripper_close_confirmation_steps": 2,
    "use_terminate_signal": False,
    "terminate_threshold": 0.5,
}


class Pi0LitePolicyExecutor(OpenVLADatasetCollector):
    def __init__(self):
        param_overrides = dict(BASE_DEFAULT_PARAMS)
        param_overrides.update(PI0_LITE_EXTRA_DEFAULTS)
        super().__init__(
            node_name="chair_grasp_moveit_pi0_lite_policy",
            param_overrides=param_overrides,
        )

        self.args.pi0_lite_checkpoint_path = str(self.get_parameter("pi0_lite_checkpoint_path").value)
        self.args.pi0_lite_device = str(self.get_parameter("pi0_lite_device").value)
        self.args.pi0_lite_steps = int(self.get_parameter("pi0_lite_steps").value)
        self.args.pi0_lite_image_size = int(self.get_parameter("pi0_lite_image_size").value)
        self.args.policy_max_steps = int(self.get_parameter("policy_max_steps").value)
        self.args.policy_timeout_sec = float(self.get_parameter("policy_timeout_sec").value)
        self.args.policy_action_duration = float(self.get_parameter("policy_action_duration").value)
        self.args.policy_action_rate_hz = float(self.get_parameter("policy_action_rate_hz").value)
        self.args.max_joint_delta_rad = float(self.get_parameter("max_joint_delta_rad").value)
        self.args.gripper_delta_close_threshold = float(self.get_parameter("gripper_delta_close_threshold").value)
        self.args.gripper_delta_open_threshold = float(self.get_parameter("gripper_delta_open_threshold").value)
        self.args.gripper_close_min_ee_z = float(self.get_parameter("gripper_close_min_ee_z").value)
        self.args.gripper_close_confirmation_steps = max(
            1, int(self.get_parameter("gripper_close_confirmation_steps").value)
        )
        self.args.use_terminate_signal = bool(self.get_parameter("use_terminate_signal").value)
        self.args.terminate_threshold = float(self.get_parameter("terminate_threshold").value)

        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None

        self.policy_log_path = os.path.join(self.run_dir, "policy_log.jsonl")

        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": "pre_grasp_then_pi0_lite_policy",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                    "pi0_lite_checkpoint_path": self.args.pi0_lite_checkpoint_path,
                    "pi0_lite_device": self.args.pi0_lite_device,
                    "pi0_lite_steps": self.args.pi0_lite_steps,
                    "pi0_lite_image_size": self.args.pi0_lite_image_size,
                    "gripper_delta_close_threshold": self.args.gripper_delta_close_threshold,
                    "gripper_close_min_ee_z": self.args.gripper_close_min_ee_z,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.get_logger().info(
            f"Pi0-Lite policy ready. run_dir={self.run_dir}, checkpoint={self.args.pi0_lite_checkpoint_path}"
        )

    def _append_policy_log(self, payload):
        with open(self.policy_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _build_policy_state(self, joint_state):
        ee_pose = self.ee_pose_from_fk(joint_state=joint_state)
        gripper_width = gripper_width_from_joint_state(joint_state)
        if gripper_width is None:
            raise RuntimeError("No gripper width available from joint state")
        return self._arm_joint_vector(joint_state) + [
            float(gripper_width),
            float(ee_pose.position[0]),
            float(ee_pose.position[1]),
            float(ee_pose.position[2]),
            float(ee_pose.orientation[0]),
            float(ee_pose.orientation[1]),
            float(ee_pose.orientation[2]),
            float(ee_pose.orientation[3]),
        ]

    def _capture_policy_frame(self):
        now = time.time()
        if self.args.wait_for_rgb_before_start:
            self._wait_for_fresh_rgb(require_new_frame=self._last_saved_rgb_receive_time is not None)
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available for policy inference")
        if self._rgb_is_stale(now):
            raise RuntimeError("RGB image is stale for policy inference")
        image_path = self._save_image()
        return {
            "image_path": image_path,
            "rgb_timestamp": self._latest_rgb_time,
            "receive_time": self._latest_rgb_receive_time,
            "capture_time": now,
            "sample_index": self._sample_index - 1,
        }

    def _run_pi0_lite_inference(self, frame_item, joint_state):
        checkpoint_path = self.args.pi0_lite_checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Pi0-Lite checkpoint not found: {checkpoint_path}")

        image_size = self.args.pi0_lite_image_size
        image_size = None if image_size <= 0 else image_size

        state_vec = self._build_policy_state(joint_state)
        start = time.time()
        pred = predict_single(
            checkpoint_path=checkpoint_path,
            image_path=frame_item["image_path"],
            instruction=self.args.instruction_text,
            state=state_vec,
            image_size=image_size,
            steps=self.args.pi0_lite_steps,
            device=self.args.pi0_lite_device,
        )
        latency_sec = time.time() - start

        first = pred["first_action"]
        action_format = pred.get("action_format", "")
        if action_format != "joint_delta":
            raise RuntimeError(
                f"Pi0-Lite executor expects joint_delta model output, but got action_format={action_format}"
            )

        action = np.asarray(
            list(first["joint_delta"]) + list(first["gripper_delta"]) + [float(first["terminate_episode"])],
            dtype=np.float32,
        )
        return action, latency_sec, pred, state_vec

    def _apply_joint_delta(self, joint_state, action):
        joint_delta = np.asarray(action[:7], dtype=np.float32)
        gripper_delta = float(action[7])

        if self.args.max_joint_delta_rad > 0.0:
            joint_delta = np.clip(joint_delta, -self.args.max_joint_delta_rad, self.args.max_joint_delta_rad)

        out = copy_joint_state(joint_state)
        joint_name_to_index = {name: idx for idx, name in enumerate(out.name)}
        for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
            if joint_name not in joint_name_to_index:
                raise RuntimeError(f"Missing arm joint in JointState: {joint_name}")
            state_index = joint_name_to_index[joint_name]
            out.position[state_index] = float(out.position[state_index] + joint_delta[joint_index])

        close_requested = False
        if gripper_delta <= self.args.gripper_delta_close_threshold:
            apply_gripper(out, self.args.close_finger)
            close_requested = True
        elif gripper_delta >= self.args.gripper_delta_open_threshold:
            apply_gripper(out, self.args.open_finger)

        return out, joint_delta, gripper_delta, close_requested

    def run_pi0_lite_policy_loop(self):
        start_time = time.time()
        step_index = 0
        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
        self.get_logger().info("Pi0-Lite policy loop started from pre-grasp pose")

        while rclpy.ok():
            if step_index >= self.args.policy_max_steps:
                self.get_logger().info(f"Policy loop stopped at max steps: {step_index}")
                break
            if (time.time() - start_time) >= self.args.policy_timeout_sec:
                self.get_logger().warn("Policy loop timed out")
                break

            frame_item = self._capture_policy_frame()
            current_js = copy_joint_state(self._latest_js)
            current_pose = self.ee_pose_from_fk(joint_state=current_js)

            self._phase = "policy_infer"
            action, latency_sec, pred, state_vec = self._run_pi0_lite_inference(frame_item, current_js)
            q_next, clipped_joint_delta, gripper_delta, close_signal_model = self._apply_joint_delta(
                current_js, action
            )

            next_pose = self.ee_pose_from_fk(joint_state=q_next)
            close_allowed_by_height = min(
                float(current_pose.position[2]), float(next_pose.position[2])
            ) <= self.args.gripper_close_min_ee_z
            if close_signal_model:
                self._close_signal_streak += 1
            else:
                self._close_signal_streak = 0
            close_requested = (
                close_signal_model
                and close_allowed_by_height
                and self._close_signal_streak >= self.args.gripper_close_confirmation_steps
            )
            if close_signal_model and not close_requested:
                q_next = copy_joint_state(q_next)
                apply_gripper(q_next, self.args.open_finger)
                next_pose = self.ee_pose_from_fk(joint_state=q_next)

            self._phase = "policy_apply"
            self.move_smooth(
                current_js,
                q_next,
                duration=self.args.policy_action_duration,
                rate_hz=self.args.policy_action_rate_hz,
                phase="policy_apply",
            )
            self._latest_js = copy_joint_state(q_next)
            self._policy_requested_close = self._policy_requested_close or close_requested

            terminate_value = float(action[8])
            terminate_requested = self.args.use_terminate_signal and (
                terminate_value >= self.args.terminate_threshold
            )
            margin_to_threshold = float(gripper_delta - self.args.gripper_delta_close_threshold)
            self._policy_gripper_deltas.append(float(gripper_delta))
            if close_requested and self._policy_close_step is None:
                self._policy_close_step = step_index

            self._append_policy_log(
                {
                    "step_index": step_index,
                    "phase": self._phase,
                    "image_path": frame_item["image_path"],
                    "sample_index": frame_item["sample_index"],
                    "rgb_timestamp": frame_item["rgb_timestamp"],
                    "receive_time": frame_item["receive_time"],
                    "capture_time": frame_item["capture_time"],
                    "policy_state": state_vec,
                    "raw_action": action.tolist(),
                    "raw_prediction": pred,
                    "inference_latency_sec": latency_sec,
                    "applied_joint_delta": clipped_joint_delta.tolist(),
                    "gripper_delta": gripper_delta,
                    "gripper_action": float(action[7]),
                    "gripper_delta_close_threshold": float(self.args.gripper_delta_close_threshold),
                    "gripper_delta_margin_to_close_threshold": margin_to_threshold,
                    "close_signal_model": bool(close_signal_model),
                    "close_allowed_by_height": bool(close_allowed_by_height),
                    "gripper_close_min_ee_z": float(self.args.gripper_close_min_ee_z),
                    "close_signal_streak": self._close_signal_streak,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                    "close_requested": bool(close_requested),
                    "terminate_value": terminate_value,
                    "terminate_requested": bool(terminate_requested),
                    "current_ee_pose": list(current_pose.position) + list(current_pose.orientation),
                    "target_ee_pose": list(next_pose.position) + list(next_pose.orientation),
                    "joint_target": list(q_next.position),
                    "timestamp": time.time(),
                }
            )

            if close_requested:
                self.get_logger().info("Pi0-Lite policy requested gripper close; ending policy loop")
                break
            if terminate_requested:
                self.get_logger().info(
                    f"Pi0-Lite policy requested termination with value={terminate_value:.4f}"
                )
                break

            step_index += 1

    def try_execute(self):
        if self._executed or self._execution_in_progress or self._latest_js is None:
            return

        detection = self._locked_detection or self._latest_detection
        if detection is None:
            return

        if self.args.wait_for_rgb_before_start and self._latest_rgb is None:
            self.get_logger().warn(f"Waiting for RGB image on topic {self.args.rgb_topic} before starting")
            return

        detection_age_sec = self._extract_detection_age_sec(detection)
        if (
            (self._locked_detection is None or not self.args.ignore_stale_when_locked)
            and detection_age_sec is not None
            and detection_age_sec > self.args.detection_stale_sec
        ):
            self.get_logger().warn(
                f"Detection is stale: age={detection_age_sec:.3f}s, limit={self.args.detection_stale_sec:.3f}s"
            )
            return

        self._execution_in_progress = True
        try:
            self._stop_continuous_recording()
            point_world = self.camera_point_to_world(detection)

            raw_goal_z = float(point_world[2] + self.args.grasp_offset)
            goal_pos = (
                float(point_world[0]),
                float(point_world[1]),
                max(raw_goal_z, self.args.min_goal_z),
            )
            pre_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.approach_offset)
            lift_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.lift_offset)

            current_pose = self.ee_pose_from_fk()
            goal_quat = self._resolve_goal_orientation(current_pose.orientation)

            q_pre = self.compute_ik(pre_pos, goal_quat)
            apply_gripper(q_pre, self.args.open_finger)
            current_js = copy_joint_state(self._latest_js)

            self._phase = "pre_grasp"
            self.move_smooth(current_js, q_pre, duration=self.args.duration, phase="pre_grasp_move")
            self._latest_js = copy_joint_state(q_pre)
            self._pause_for_observation(self.args.pre_grasp_pause_sec, hold_joint_state=q_pre)

            self._phase = "policy"
            self.run_pi0_lite_policy_loop()

            if self._policy_requested_close:
                self._phase = "lift"
                q_lift = self.compute_ik(lift_pos, goal_quat)
                apply_gripper(q_lift, self.args.close_finger)
                self.move_smooth(
                    copy_joint_state(self._latest_js),
                    q_lift,
                    duration=self.args.duration,
                    phase="lift",
                )
                self._latest_js = copy_joint_state(q_lift)

            self._executed = True
            self.get_logger().info(f"Pi0-Lite policy run finished. log_path={self.policy_log_path}")
        except Exception as exc:
            self.get_logger().error(f"Pi0-Lite policy failed at phase={self._phase}: {exc}")
        finally:
            self._execution_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = Pi0LitePolicyExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


## 실행예시
# ros2 run robotarm_executor chair_grasp_moveit_pi0_lite_policy