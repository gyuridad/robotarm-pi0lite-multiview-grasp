#!/usr/bin/env python3

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from robotarm_executor.chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone import (
    DEFAULT_PI0_LITE_EXTERNAL_CHECKPOINT,
    ExternalPi0LitePolicyExecutorCartesianStandalone,
    apply_gripper,
    copy_joint_state,
    predict_single,
)


THIS_DIR = Path(__file__).resolve().parent


def resolve_robotarm_project_root() -> Path:
    env_root = os.environ.get("ROBOTARM_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "pi0_lite" / "pi0_lite_phase_aux_external_joint_delta.py").exists():
            return candidate

    search_roots = [THIS_DIR, Path.cwd().resolve()]
    for root in search_roots:
        for candidate in [root, *root.parents]:
            direct = candidate / "pi0_lite" / "pi0_lite_phase_aux_external_joint_delta.py"
            src_layout = (
                candidate
                / "src"
                / "robotarm_project"
                / "pi0_lite"
                / "pi0_lite_phase_aux_external_joint_delta.py"
            )
            if direct.exists():
                return candidate
            if src_layout.exists():
                return candidate / "src" / "robotarm_project"

    raise RuntimeError(
        "Could not locate robotarm_project root. "
        "Set ROBOTARM_PROJECT_ROOT to the directory containing pi0_lite."
    )


PROJECT_ROOT = resolve_robotarm_project_root()
DEFAULT_JOINT_DELTA_CHECKPOINT = str(
    PROJECT_ROOT
    / "pi0_lite"
    / "pi0_lite_output"
    / "run_udp_aux_external_joint_delta"
    / "pi0_lite_phase_aux_external_final.pt"
)


class ExternalPi0LitePolicyExecutorJointDeltaFromCartesianStandalone(
    ExternalPi0LitePolicyExecutorCartesianStandalone
):
    def __init__(self):
        super().__init__()

        extra_defaults = {
            "joint_delta_scale": 1.0,
            "max_joint_delta_rad": 0.08,
            "gripper_delta_close_threshold": -0.01,
            "gripper_delta_open_threshold": 0.002,
        }
        for name, value in extra_defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

        self.args.joint_delta_scale = max(0.0, float(self.get_parameter("joint_delta_scale").value))
        self.args.max_joint_delta_rad = float(self.get_parameter("max_joint_delta_rad").value)
        self.args.gripper_delta_close_threshold = float(
            self.get_parameter("gripper_delta_close_threshold").value
        )
        self.args.gripper_delta_open_threshold = float(
            self.get_parameter("gripper_delta_open_threshold").value
        )

        current_checkpoint = str(self.args.pi0_lite_checkpoint_path or "").strip()
        if not current_checkpoint or current_checkpoint == DEFAULT_PI0_LITE_EXTERNAL_CHECKPOINT:
            self.args.pi0_lite_checkpoint_path = DEFAULT_JOINT_DELTA_CHECKPOINT

        self._rewrite_meta_for_joint_delta()
        self.get_logger().info(
            "Pi0-Lite external joint-delta policy executor via cartesian standalone enabled. "
            f"checkpoint={self.args.pi0_lite_checkpoint_path}, "
            f"joint_delta_scale={self.args.joint_delta_scale:.2f}, "
            f"max_joint_delta_rad={self.args.max_joint_delta_rad:.3f}"
        )

    def _rewrite_meta_for_joint_delta(self):
        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["mode"] = "pre_grasp_then_pi0_lite_policy_external_joint_delta_from_cartesian_standalone"
            meta["pi0_lite_checkpoint_path"] = self.args.pi0_lite_checkpoint_path
            meta["joint_delta_scale"] = self.args.joint_delta_scale
            meta["max_joint_delta_rad"] = self.args.max_joint_delta_rad
            meta["gripper_delta_close_threshold"] = self.args.gripper_delta_close_threshold
            meta["gripper_delta_open_threshold"] = self.args.gripper_delta_open_threshold
            with open(self.meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.get_logger().warn(f"Failed to rewrite joint-delta meta checkpoint path: {exc}")

    def _run_pi0_lite_inference(self, frame_item, joint_state):
        checkpoint_path = self.args.pi0_lite_checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Pi0-Lite checkpoint not found: {checkpoint_path}")

        image_size = self.args.pi0_lite_image_size
        image_size = None if image_size <= 0 else image_size
        state_vec = self._build_policy_state(joint_state)

        pred = predict_single(
            checkpoint_path=checkpoint_path,
            image_path=frame_item["image_path"],
            external_image_path=frame_item["external_image_path"],
            instruction=self.args.instruction_text,
            state=state_vec,
            image_size=image_size,
            steps=self.args.pi0_lite_steps,
            device=self.args.pi0_lite_device,
        )

        first = pred["first_action"]
        action_format = pred.get("action_format", "")
        if action_format != "joint_delta":
            raise RuntimeError(
                "Pi0-Lite joint-delta executor via cartesian standalone expects "
                f"joint_delta output, got action_format={action_format}"
            )

        action = np.asarray(
            list(first["joint_delta"]) + list(first["gripper_delta"]) + [float(first["terminate_episode"])],
            dtype=np.float32,
        )
        return action, pred, state_vec

    def _apply_joint_delta(self, joint_state, action):
        joint_delta = np.asarray(action[:7], dtype=np.float32) * float(self.args.joint_delta_scale)
        gripper_delta = float(action[7])

        if self.args.max_joint_delta_rad > 0.0:
            joint_delta = np.clip(joint_delta, -self.args.max_joint_delta_rad, self.args.max_joint_delta_rad)

        out = copy_joint_state(joint_state)
        joint_name_to_index = {name: idx for idx, name in enumerate(out.name)}
        for joint_index, joint_name in enumerate([f"panda_joint{i}" for i in range(1, 8)]):
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
        start_time = rclpy.clock.Clock().now().nanoseconds / 1e9
        step_index = 0
        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
        self.get_logger().info(
            "Pi0-Lite external joint-delta policy loop via cartesian standalone started from pre-grasp pose"
        )

        while rclpy.ok():
            now = rclpy.clock.Clock().now().nanoseconds / 1e9
            if step_index >= self.args.policy_max_steps:
                self.get_logger().info(f"Policy loop stopped at max steps: {step_index}")
                break
            if (now - start_time) >= self.args.policy_timeout_sec:
                self.get_logger().warn("Policy loop timed out")
                break

            frame_item = self._capture_policy_frame()
            current_js = copy_joint_state(self._latest_js)
            current_pose = self.ee_pose_from_fk(joint_state=current_js)

            self._phase = "policy_infer"
            infer_start = rclpy.clock.Clock().now().nanoseconds / 1e9
            action, pred, state_vec = self._run_pi0_lite_inference(frame_item, current_js)
            latency_sec = (rclpy.clock.Clock().now().nanoseconds / 1e9) - infer_start

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
                    "external_image_path": frame_item["external_image_path"],
                    "external_image_source": frame_item["external_image_source"],
                    "sample_index": frame_item["sample_index"],
                    "rgb_timestamp": frame_item["rgb_timestamp"],
                    "external_rgb_timestamp": frame_item["external_rgb_timestamp"],
                    "camera_sync_delta_sec": frame_item["camera_sync_delta_sec"],
                    "external_match_delta_sec": frame_item["external_match_delta_sec"],
                    "external_match_basis": frame_item["external_match_basis"],
                    "receive_time": frame_item["receive_time"],
                    "external_receive_time": frame_item["external_receive_time"],
                    "capture_time": frame_item["capture_time"],
                    "policy_state": [float(v) for v in state_vec],
                    "raw_action": [float(v) for v in action.tolist()],
                    "raw_prediction": pred,
                    "inference_latency_sec": float(latency_sec),
                    "applied_joint_delta": [float(v) for v in clipped_joint_delta],
                    "gripper_delta": float(gripper_delta),
                    "gripper_action": float(action[7]),
                    "gripper_delta_close_threshold": float(self.args.gripper_delta_close_threshold),
                    "gripper_delta_margin_to_close_threshold": float(margin_to_threshold),
                    "close_signal_model": bool(close_signal_model),
                    "close_allowed_by_height": bool(close_allowed_by_height),
                    "gripper_close_min_ee_z": float(self.args.gripper_close_min_ee_z),
                    "close_signal_streak": self._close_signal_streak,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                    "close_requested": bool(close_requested),
                    "terminate_value": float(terminate_value),
                    "terminate_requested": bool(terminate_requested),
                    "current_ee_pose": [float(v) for v in current_pose.position]
                    + [float(v) for v in current_pose.orientation],
                    "target_ee_pose": [float(v) for v in next_pose.position]
                    + [float(v) for v in next_pose.orientation],
                    "joint_target": [float(v) for v in q_next.position],
                    "timestamp": float(rclpy.clock.Clock().now().nanoseconds / 1e9),
                }
            )

            self.get_logger().info(
                "Joint-delta policy step summary via cartesian standalone. "
                f"step={step_index}, latency_sec={float(latency_sec):.3f}, "
                f"external_source={frame_item['external_image_source']}, "
                f"phase_pred={pred.get('predicted_phase')}, "
                f"gripper_delta={float(gripper_delta):.6f}, "
                f"terminate={float(terminate_value):.6f}"
            )

            if close_requested:
                self.get_logger().info(
                    "Pi0-Lite external joint-delta policy via cartesian standalone requested gripper close; "
                    "ending policy loop"
                )
                break
            if terminate_requested:
                self.get_logger().info(
                    "Pi0-Lite external joint-delta policy via cartesian standalone requested termination "
                    f"with value={terminate_value:.4f}"
                )
                break
            step_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = ExternalPi0LitePolicyExecutorJointDeltaFromCartesianStandalone()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("stopped by user")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# ros2 run robotarm_executor chair_grasp_moveit_pi0_lite_policy_external_joint_delta_from_cartesian_standalone --ros-args \
#   -p use_external_camera:=true \
#   -p policy_timeout_sec:=300.0 \
#   -p policy_max_steps:=50 \
#   -p joint_delta_scale:=6.0 \
#   -p max_joint_delta_rad:=0.2


