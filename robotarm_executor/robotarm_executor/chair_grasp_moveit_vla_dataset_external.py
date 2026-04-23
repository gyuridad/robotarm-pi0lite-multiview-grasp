#!/usr/bin/env python3

from collections import deque
import json
import math
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from moveit_msgs.msg import Constraints, OrientationConstraint, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from robotarm_common.chair_grasp_common import UdpChunkAssembler, camera_to_world, parse_frame_payload


THIS_DIR = Path(__file__).resolve().parent


def resolve_robotarm_project_root() -> Path:
    env_root = os.environ.get("ROBOTARM_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "robotarm_executor").exists() or (candidate / "robotarm_common").exists():
            return candidate

    for root in [THIS_DIR, Path.cwd().resolve()]:
        for candidate in [root, *root.parents]:
            if (candidate / "robotarm_executor").exists() and (candidate / "robotarm_common").exists():
                return candidate
            src_layout = candidate / "src" / "robotarm_project"
            if (src_layout / "robotarm_executor").exists() and (src_layout / "robotarm_common").exists():
                return src_layout

    raise RuntimeError(
        "Could not locate robotarm_project root. "
        "Set ROBOTARM_PROJECT_ROOT to /.../src/robotarm_project."
    )


PROJECT_ROOT = resolve_robotarm_project_root()
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "debug_runs" / "chair_grasp_openvla")

DEFAULT_PARAMS = {
    "detection_topic": "/chair_detection_json",
    "joint_state_topic": "/joint_states",
    "joint_command_topic": "/joint_command",
    "rgb_topic": "/hand_Camera_rgb",
    "ik_service": "/compute_ik",
    "fk_service": "/compute_fk",
    "camera_frame": "",
    "tf_base_frame": "World",
    "moveit_base_frame": "world",
    "tip_link": "panda_hand",
    "tf_timeout_sec": 1.0,
    "approach_offset": 0.20,
    "grasp_offset": 0.03,
    "min_goal_z": 0.17,
    "lift_offset": 0.08,
    "retreat_offset": 0.05,
    "open_finger": 0.04,
    "close_finger": 0.0,
    "duration": 1.5,
    "gripper_motion_duration": 0.8,
    "gripper_settle_sec": 0.5,
    "pre_grasp_pause_sec": 0.8,
    "frame_sample_period_sec": 0.10,
    "fk_timeout_sec": 2.0,
    "preferred_camera_frame": "",
    "lock_detection_once": True,
    "ignore_stale_when_locked": True,
    "detection_stale_sec": 2.0,
    "orientation_mode": "top_down",
    "fixed_grasp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "top_down_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "orientation_tolerance_rad": 0.12,
    "dataset_root": DEFAULT_DATASET_ROOT,
    "instruction_text": "의자를 집어",
    "image_format": "jpg",
    "jpeg_quality": 95,
    "wait_for_rgb_before_start": True,
    "rgb_wait_timeout_sec": 0.3,
    "max_rgb_staleness_sec": 0.5,
    "enable_retreat": False,
}

GROUP_NAME = "panda_arm"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]


def quat_normalize(quat):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = quat_normalize(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def copy_joint_state(msg: JointState) -> JointState:
    out = JointState()
    out.header = msg.header
    out.name = list(msg.name)
    out.position = list(msg.position)
    out.velocity = list(msg.velocity)
    out.effort = list(msg.effort)
    return out


def quat_from_param(value, default):
    if value is None or len(value) != 4:
        return quat_normalize(default)
    return quat_normalize([float(component) for component in value])


def apply_gripper(js: JointState, open_width: float):
    for name in FINGER_JOINT_NAMES:
        if name in js.name:
            js.position[js.name.index(name)] = open_width


def gripper_width_from_joint_state(js: JointState):
    if js is None:
        return None
    finger_positions = []
    for name in FINGER_JOINT_NAMES:
        if name in js.name:
            finger_positions.append(float(js.position[js.name.index(name)]))
    if not finger_positions:
        return None
    return float(sum(finger_positions))


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ros_image_to_rgb(msg: Image):
    enc = (msg.encoding or "").lower()
    if enc == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if enc == "rgba8":
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    if enc == "bgra8":
        bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    if enc == "mono8":
        mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    raise RuntimeError(f"Unsupported RGB encoding: {msg.encoding}")


class OpenVLADatasetCollector(Node):
    def __init__(self):
        super().__init__("chair_grasp_moveit_openvla_dataset")
        for name, value in DEFAULT_PARAMS.items():
            self.declare_parameter(name, value)
        self.args = SimpleNamespace(
            detection_topic=str(self.get_parameter("detection_topic").value),
            joint_state_topic=str(self.get_parameter("joint_state_topic").value),
            joint_command_topic=str(self.get_parameter("joint_command_topic").value),
            rgb_topic=str(self.get_parameter("rgb_topic").value),
            ik_service=str(self.get_parameter("ik_service").value),
            fk_service=str(self.get_parameter("fk_service").value),
            camera_frame=str(self.get_parameter("camera_frame").value),
            tf_base_frame=str(self.get_parameter("tf_base_frame").value),
            moveit_base_frame=str(self.get_parameter("moveit_base_frame").value),
            tip_link=str(self.get_parameter("tip_link").value),
            tf_timeout_sec=float(self.get_parameter("tf_timeout_sec").value),
            approach_offset=float(self.get_parameter("approach_offset").value),
            grasp_offset=float(self.get_parameter("grasp_offset").value),
            min_goal_z=float(self.get_parameter("min_goal_z").value),
            lift_offset=float(self.get_parameter("lift_offset").value),
            retreat_offset=float(self.get_parameter("retreat_offset").value),
            open_finger=float(self.get_parameter("open_finger").value),
            close_finger=float(self.get_parameter("close_finger").value),
            duration=float(self.get_parameter("duration").value),
            gripper_motion_duration=float(self.get_parameter("gripper_motion_duration").value),
            gripper_settle_sec=float(self.get_parameter("gripper_settle_sec").value),
            pre_grasp_pause_sec=max(0.0, float(self.get_parameter("pre_grasp_pause_sec").value)),
            frame_sample_period_sec=max(0.01, float(self.get_parameter("frame_sample_period_sec").value)),
            fk_timeout_sec=float(self.get_parameter("fk_timeout_sec").value),
            preferred_camera_frame=str(self.get_parameter("preferred_camera_frame").value),
            lock_detection_once=bool(self.get_parameter("lock_detection_once").value),
            ignore_stale_when_locked=bool(self.get_parameter("ignore_stale_when_locked").value),
            detection_stale_sec=float(self.get_parameter("detection_stale_sec").value),
            orientation_mode=str(self.get_parameter("orientation_mode").value),
            fixed_grasp_quat_xyzw=[float(v) for v in self.get_parameter("fixed_grasp_quat_xyzw").value],
            top_down_quat_xyzw=[float(v) for v in self.get_parameter("top_down_quat_xyzw").value],
            orientation_tolerance_rad=float(self.get_parameter("orientation_tolerance_rad").value),
            dataset_root=str(self.get_parameter("dataset_root").value),
            instruction_text=str(self.get_parameter("instruction_text").value),
            image_format=str(self.get_parameter("image_format").value).lower(),
            jpeg_quality=int(self.get_parameter("jpeg_quality").value),
            wait_for_rgb_before_start=bool(self.get_parameter("wait_for_rgb_before_start").value),
            rgb_wait_timeout_sec=max(0.0, float(self.get_parameter("rgb_wait_timeout_sec").value)),
            max_rgb_staleness_sec=max(0.0, float(self.get_parameter("max_rgb_staleness_sec").value)),
            enable_retreat=bool(self.get_parameter("enable_retreat").value),
        )

        self.service_cb_group = ReentrantCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self.ik_cli = self.create_client(GetPositionIK, self.args.ik_service, callback_group=self.service_cb_group)
        self.fk_cli = self.create_client(GetPositionFK, self.args.fk_service, callback_group=self.service_cb_group)
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service not available: {self.args.ik_service}")
        if not self.fk_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"FK service not available: {self.args.fk_service}")

        self._latest_js = None
        self._latest_rgb = None
        self._latest_rgb_time = None
        self._latest_rgb_receive_time = None
        self._last_saved_rgb_receive_time = None
        self._latest_detection = None
        self._locked_detection = None
        self._executed = False
        self._execution_in_progress = False
        self._phase = "idle"
        self._sample_index = 0
        self._recording_active = False
        self._next_record_time = 0.0

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.args.dataset_root, run_id)
        self.images_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.samples_path = os.path.join(self.run_dir, "samples_openvla.jsonl")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_type": "openvla_stream",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            JointState, self.args.joint_state_topic, self.on_joint_state, 10, callback_group=self.io_cb_group
        )
        self.create_subscription(
            String, self.args.detection_topic, self.on_detection, 10, callback_group=self.io_cb_group
        )
        self.create_subscription(Image, self.args.rgb_topic, self.on_rgb, 10, callback_group=self.io_cb_group)
        self.pub = self.create_publisher(JointState, self.args.joint_command_topic, 10)
        self.create_timer(0.1, self.try_execute, callback_group=self.io_cb_group)

        self.get_logger().info(
            f"OpenVLA dataset collection ready. run_dir={self.run_dir}, rgb_topic={self.args.rgb_topic}"
        )

    def on_joint_state(self, msg: JointState):
        self._latest_js = copy_joint_state(msg)

    def on_detection(self, msg: String):
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to parse detection JSON: {exc}")
            return
        self._latest_detection = result
        if self.args.lock_detection_once and self._locked_detection is None:
            self._locked_detection = result

    def on_rgb(self, msg: Image):
        try:
            self._latest_rgb = ros_image_to_rgb(msg)
            self._latest_rgb_time = stamp_to_sec(msg.header.stamp)
            self._latest_rgb_receive_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode RGB image: {exc}")

    def _extract_detection_age_sec(self, detection):
        stamp = detection.get("stamp") if isinstance(detection, dict) else None
        if not isinstance(stamp, dict):
            return None
        sec = stamp.get("sec")
        nanosec = stamp.get("nanosec")
        if sec is None or nanosec is None:
            return None
        return time.time() - (float(sec) + float(nanosec) * 1e-9)

    def _resolve_goal_orientation(self, current_orientation):
        mode = (self.args.orientation_mode or "keep_current").strip().lower()
        if mode == "fixed":
            return quat_from_param(self.args.fixed_grasp_quat_xyzw, current_orientation)
        if mode == "top_down":
            return quat_from_param(self.args.top_down_quat_xyzw, current_orientation)
        return quat_normalize(current_orientation)

    def _joint_indices(self, js):
        indices = []
        for name in ARM_JOINT_NAMES:
            if name not in js.name:
                raise RuntimeError(f"Missing arm joint in JointState: {name}")
            indices.append(js.name.index(name))
        return indices

    def _arm_joint_vector(self, js):
        indices = self._joint_indices(js)
        return [float(js.position[index]) for index in indices]

    def camera_point_to_world(self, detection):
        xyz_camera = detection["detection"].get("xyz_camera")
        if xyz_camera is None:
            raise RuntimeError("Detection does not contain 3D camera coordinates")

        point_camera = np.asarray(xyz_camera, dtype=np.float32)
        t_world_camera = detection.get("t_world_camera")
        if t_world_camera is not None:
            return camera_to_world(point_camera, np.asarray(t_world_camera, dtype=np.float32))

        camera_info = detection.get("camera_info") or {}
        raw_frame = camera_info.get("frame_id") or ""
        candidates = []
        for candidate in [raw_frame, raw_frame.lstrip("/"), self.args.preferred_camera_frame, self.args.camera_frame]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError("Detection does not contain T_world_camera or camera frame_id")

        deadline = time.time() + max(0.0, self.args.tf_timeout_sec)
        last_exc = None
        while time.time() < deadline:
            for camera_frame in candidates:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.args.tf_base_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=self.args.tf_timeout_sec),
                    )
                    rotation = transform.transform.rotation
                    translation = transform.transform.translation
                    rotmat = quat_xyzw_to_rotmat((rotation.x, rotation.y, rotation.z, rotation.w))
                    trans = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
                    return (rotmat @ point_camera) + trans
                except TransformException as exc:
                    last_exc = exc
            rclpy.spin_once(self, timeout_sec=0.02)
        raise RuntimeError(f"TF lookup failed for camera frame candidates {candidates}: {last_exc}")

    def ee_pose_from_fk(self, base_frame=None, tip_link=None, timeout=None, joint_state=None):
        joint_state = self._latest_js if joint_state is None else joint_state
        if joint_state is None:
            raise RuntimeError("No joint state available for FK")

        req = GetPositionFK.Request()
        req.header.frame_id = self.args.moveit_base_frame if base_frame is None else base_frame
        req.fk_link_names = [self.args.tip_link if tip_link is None else tip_link]
        req.robot_state.joint_state = joint_state

        future = self.fk_cli.call_async(req)
        timeout = self.args.fk_timeout_sec if timeout is None else timeout
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done():
            if time.time() >= end_time:
                raise RuntimeError("FK request timed out")
            rclpy.spin_once(self, timeout_sec=0.02)

        resp = future.result()
        if resp is None or len(resp.pose_stamped) == 0:
            raise RuntimeError("FK returned empty pose")

        pose = resp.pose_stamped[0].pose
        return SimpleNamespace(
            position=(pose.position.x, pose.position.y, pose.position.z),
            orientation=quat_normalize((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)),
        )

    def compute_ik(self, pos, quat_xyzw, timeout=2.0, avoid_collisions=False, ik_link_name=None, seed_joint_state=None):
        px, py, pz = [float(v) for v in np.asarray(pos).ravel()[:3]]
        qx, qy, qz, qw = quat_normalize([float(v) for v in np.asarray(quat_xyzw).ravel()[:4]])

        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout = Duration(seconds=float(timeout)).to_msg()
        req.ik_request.ik_link_name = self.args.tip_link if ik_link_name is None else ik_link_name
        req.ik_request.pose_stamped.header.frame_id = self.args.moveit_base_frame
        req.ik_request.pose_stamped.pose.position.x = px
        req.ik_request.pose_stamped.pose.position.y = py
        req.ik_request.pose_stamped.pose.position.z = pz
        req.ik_request.pose_stamped.pose.orientation.x = qx
        req.ik_request.pose_stamped.pose.orientation.y = qy
        req.ik_request.pose_stamped.pose.orientation.z = qz
        req.ik_request.pose_stamped.pose.orientation.w = qw

        constraint = OrientationConstraint()
        constraint.header.frame_id = self.args.moveit_base_frame
        constraint.link_name = req.ik_request.ik_link_name
        constraint.orientation = req.ik_request.pose_stamped.pose.orientation
        constraint.absolute_x_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_y_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_z_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.weight = 1.0
        constraints = Constraints()
        constraints.orientation_constraints = [constraint]
        req.ik_request.constraints = constraints

        seed_source = self._latest_js if seed_joint_state is None else seed_joint_state
        if seed_source is None:
            raise RuntimeError("No joint state available for IK seed")
        seed = RobotState()
        seed.joint_state = seed_source
        req.ik_request.robot_state = seed

        future = self.ik_cli.call_async(req)
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done():
            if time.time() >= end_time:
                raise RuntimeError("IK request timed out")
            rclpy.spin_once(self, timeout_sec=0.02)

        resp = future.result()
        if resp is None:
            raise RuntimeError("IK returned no response")
        if getattr(resp.error_code, "val", 0) != 1:
            raise RuntimeError(f"IK failed with error code {resp.error_code.val}")

        solution = resp.solution.joint_state
        solution_map = dict(zip(solution.name, solution.position))
        isaac_order = list(seed_source.name)
        base_map = dict(zip(seed_source.name, seed_source.position))
        base_map.update(solution_map)

        out = JointState()
        out.name = isaac_order
        out.position = [float(base_map.get(name, 0.0)) for name in isaac_order]
        return out

    def _save_image(self):
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available")

        ext = "png" if self.args.image_format == "png" else "jpg"
        image_name = f"sample_{self._sample_index:06d}.{ext}"
        image_path = os.path.join(self.images_dir, image_name)
        bgr = cv2.cvtColor(self._latest_rgb, cv2.COLOR_RGB2BGR)
        if ext == "png":
            ok = cv2.imwrite(image_path, bgr)
        else:
            ok = cv2.imwrite(image_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to save image: {image_path}")
        self._last_saved_rgb_receive_time = self._latest_rgb_receive_time
        return image_path

    def _wait_for_fresh_rgb(self, require_new_frame=False):
        baseline = self._last_saved_rgb_receive_time if require_new_frame else None
        if self._latest_rgb is not None:
            if baseline is None:
                return True
            if self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline:
                return True
        start = time.time()
        while time.time() - start < self.args.rgb_wait_timeout_sec:
            if self._latest_rgb is not None:
                if baseline is None:
                    return True
                if self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline:
                    return True
            rclpy.spin_once(self, timeout_sec=0.02)
        if self._latest_rgb is None:
            return False
        if baseline is None:
            return True
        return self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline

    def _rgb_is_stale(self, sample_time):
        rgb_reference_time = self._latest_rgb_receive_time
        if rgb_reference_time is None:
            return True
        if self.args.max_rgb_staleness_sec <= 0.0:
            return False
        return abs(sample_time - rgb_reference_time) > self.args.max_rgb_staleness_sec

    def _save_openvla_sample(self, phase, joint_state, note):
        now = time.time()
        if self.args.wait_for_rgb_before_start:
            self._wait_for_fresh_rgb(require_new_frame=self._last_saved_rgb_receive_time is not None)
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available for sample")
        if self._rgb_is_stale(now):
            raise RuntimeError("RGB image is stale for sample")

        image_path = self._save_image()
        ee_pose = self.ee_pose_from_fk(joint_state=joint_state)
        sample = {
            "image": os.path.relpath(image_path, self.run_dir),
            "instruction": self.args.instruction_text,
            "phase": phase,
            "note": note,
            "action": {
                "arm_joint_position": self._arm_joint_vector(joint_state),
                "gripper_width": gripper_width_from_joint_state(joint_state),
                "ee_pose": [
                    float(ee_pose.position[0]),
                    float(ee_pose.position[1]),
                    float(ee_pose.position[2]),
                    float(ee_pose.orientation[0]),
                    float(ee_pose.orientation[1]),
                    float(ee_pose.orientation[2]),
                    float(ee_pose.orientation[3]),
                ],
            },
            "timestamp": now,
            "rgb_timestamp": self._latest_rgb_time,
        }
        with open(self.samples_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._sample_index += 1

    def _start_continuous_recording(self):
        self._recording_active = True
        self._next_record_time = time.time()

    def _stop_continuous_recording(self):
        self._recording_active = False

    def _maybe_record_frame(self, phase, joint_state, note="", force=False):
        if not self._recording_active and not force:
            return
        now = time.time()
        if not force and now < self._next_record_time:
            return
        self._save_openvla_sample(phase, joint_state, note)
        self._next_record_time = now + self.args.frame_sample_period_sec

    def _settle_and_capture_joint_state(self, settle_sec=None):
        settle_sec = self.args.gripper_settle_sec if settle_sec is None else settle_sec
        deadline = time.time() + max(0.0, settle_sec)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.time())))
        if self._latest_js is None:
            raise RuntimeError("No joint state available after settle")
        return copy_joint_state(self._latest_js)

    def _pause_for_observation(self, pause_sec, hold_joint_state=None, rate_hz=20.0):
        deadline = time.time() + max(0.0, pause_sec)
        while time.time() < deadline:
            if hold_joint_state is not None:
                hold_msg = JointState()
                hold_msg.name = list(hold_joint_state.name)
                hold_msg.position = list(hold_joint_state.position)
                self.pub.publish(hold_msg)
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.time())))
            if rate_hz > 0.0:
                time.sleep(1.0 / rate_hz)

    def move_smooth(self, from_js, to_js, duration=1.5, rate_hz=100, phase=None):
        names = list(from_js.name)
        if names != list(to_js.name):
            raise RuntimeError("Joint name order mismatch")
        start = np.asarray(from_js.position, dtype=float)
        goal = np.asarray(to_js.position, dtype=float)
        steps = max(1, int(duration * rate_hz))
        active_phase = self._phase if phase is None else phase
        for index in range(steps + 1):
            alpha = index / steps
            point = (1.0 - alpha) * start + alpha * goal
            msg = JointState()
            msg.name = names
            msg.position = point.tolist()
            self.pub.publish(msg)
            self._latest_js = copy_joint_state(msg)
            self._maybe_record_frame(active_phase, self._latest_js, note="stream")
            time.sleep(1.0 / rate_hz)

    def try_execute(self):
        return


class ExternalVLADatasetCollector(OpenVLADatasetCollector):
    """
    Diffusion/VLA dataset collector with external camera support.

    Base behavior stays aligned with the diffusion-style collector:
    - approach is split into far / mid / near / contact_ready
    - approach phases are sampled more densely
    - target pose can be jittered for data diversity

    Additional external-camera behavior:
    - receive wrist/hand RGB and external RGB through ROS topics or UDP
    - match wrist and external frames by nearest timestamp before saving
    - save external RGB alongside the internal wrist RGB
    - write external_image into each JSONL sample
    - optionally fall back to the internal RGB if external frames are unavailable
    """

    def __init__(self):
        super().__init__()

        extra_defaults = {
            "approach_mid_offset": 0.14,
            "approach_near_offset": 0.08,
            "contact_ready_offset": 0.04,
            "approach_pause_sec": 0.15,
            "approach_sample_period_sec": 0.05,
            "stage_pause_sec": 0.10,
            "goal_xy_jitter_m": 0.0,
            "goal_z_jitter_m": 0.0,
            "hand_input_mode": "ros_topic",
            "hand_udp_listen_host": "0.0.0.0",
            "hand_udp_listen_port": 9005,
            "hand_udp_poll_period_sec": 0.01,
            "hand_udp_stale_after_sec": 10.0,
            "hand_udp_max_frame_age_sec": 0.0,
            "external_rgb_topic": "/external_rgb",
            "external_input_mode": "ros_topic",
            "use_external_camera": True,
            "external_camera_required": False,
            "external_camera_fallback_to_internal": True,
            "external_rgb_wait_timeout_sec": 0.3,
            "hand_rgb_wait_timeout_sec": 0.3,
            "sync_tolerance_sec": 0.10,
            "external_match_mode": "prefer_receive_time",
            "hand_frame_buffer_size": 30,
            "external_frame_buffer_size": 30,
            "max_external_rgb_staleness_sec": 0.5,
            "external_udp_listen_host": "0.0.0.0",
            "external_udp_listen_port": 9002,
            "external_udp_poll_period_sec": 0.01,
            "external_udp_stale_after_sec": 10.0,
            "external_udp_max_frame_age_sec": 0.0,
        }
        for name, value in extra_defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

        self.args.approach_mid_offset = float(self.get_parameter("approach_mid_offset").value)
        self.args.approach_near_offset = float(self.get_parameter("approach_near_offset").value)
        self.args.contact_ready_offset = float(self.get_parameter("contact_ready_offset").value)
        self.args.approach_pause_sec = max(0.0, float(self.get_parameter("approach_pause_sec").value))
        self.args.approach_sample_period_sec = max(
            0.01, float(self.get_parameter("approach_sample_period_sec").value)
        )
        self.args.stage_pause_sec = max(0.0, float(self.get_parameter("stage_pause_sec").value))
        self.args.goal_xy_jitter_m = max(0.0, float(self.get_parameter("goal_xy_jitter_m").value))
        self.args.goal_z_jitter_m = max(0.0, float(self.get_parameter("goal_z_jitter_m").value))
        self.args.hand_input_mode = str(self.get_parameter("hand_input_mode").value).strip().lower()
        self.args.hand_udp_listen_host = str(self.get_parameter("hand_udp_listen_host").value)
        self.args.hand_udp_listen_port = int(self.get_parameter("hand_udp_listen_port").value)
        self.args.hand_udp_poll_period_sec = max(
            0.001, float(self.get_parameter("hand_udp_poll_period_sec").value)
        )
        self.args.hand_udp_stale_after_sec = max(
            0.1, float(self.get_parameter("hand_udp_stale_after_sec").value)
        )
        self.args.hand_udp_max_frame_age_sec = max(
            0.0, float(self.get_parameter("hand_udp_max_frame_age_sec").value)
        )
        self.args.external_rgb_topic = str(self.get_parameter("external_rgb_topic").value)
        self.args.external_input_mode = str(self.get_parameter("external_input_mode").value).strip().lower()
        self.args.use_external_camera = bool(self.get_parameter("use_external_camera").value)
        self.args.external_camera_required = bool(self.get_parameter("external_camera_required").value)
        self.args.external_camera_fallback_to_internal = bool(
            self.get_parameter("external_camera_fallback_to_internal").value
        )
        self.args.hand_rgb_wait_timeout_sec = max(
            0.0, float(self.get_parameter("hand_rgb_wait_timeout_sec").value)
        )
        self.args.sync_tolerance_sec = max(0.0, float(self.get_parameter("sync_tolerance_sec").value))
        self.args.external_match_mode = str(
            self.get_parameter("external_match_mode").value
        ).strip().lower()
        self.args.hand_frame_buffer_size = max(2, int(self.get_parameter("hand_frame_buffer_size").value))
        self.args.external_frame_buffer_size = max(
            2, int(self.get_parameter("external_frame_buffer_size").value)
        )
        self.args.external_rgb_wait_timeout_sec = max(
            0.0, float(self.get_parameter("external_rgb_wait_timeout_sec").value)
        )
        self.args.max_external_rgb_staleness_sec = max(
            0.0, float(self.get_parameter("max_external_rgb_staleness_sec").value)
        )
        self.args.external_udp_listen_host = str(self.get_parameter("external_udp_listen_host").value)
        self.args.external_udp_listen_port = int(self.get_parameter("external_udp_listen_port").value)
        self.args.external_udp_poll_period_sec = max(
            0.001, float(self.get_parameter("external_udp_poll_period_sec").value)
        )
        self.args.external_udp_stale_after_sec = max(
            0.1, float(self.get_parameter("external_udp_stale_after_sec").value)
        )
        self.args.external_udp_max_frame_age_sec = max(
            0.0, float(self.get_parameter("external_udp_max_frame_age_sec").value)
        )

        self.external_images_dir = os.path.join(self.run_dir, "external_images")
        os.makedirs(self.external_images_dir, exist_ok=True)
        self._hand_frame_buffer = deque(maxlen=self.args.hand_frame_buffer_size)
        self._external_frame_buffer = deque(maxlen=self.args.external_frame_buffer_size)
        self._hand_udp_assembler = None
        self._hand_udp_sock = None
        self._hand_udp_timer = None
        self._latest_external_rgb = None
        self._latest_external_rgb_time = None
        self._latest_external_rgb_receive_time = None
        self._last_saved_external_rgb_receive_time = None
        self._external_udp_assembler = None
        self._external_udp_sock = None
        self._external_udp_timer = None
        self.args.hand_runtime_rgb_topic = self.args.rgb_topic
        self.args.external_runtime_rgb_topic = self.args.external_rgb_topic

        if self.args.hand_input_mode == "udp":
            self._setup_hand_udp_receiver()

        if self.args.use_external_camera:
            if self.args.external_input_mode == "udp":
                self._setup_external_udp_receiver()
            else:
                self.create_subscription(
                    Image,
                    self.args.external_rgb_topic,
                    self.on_external_rgb,
                    10,
                    callback_group=self.io_cb_group,
                )

        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_type": "diffusion_vla_stream_external",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "hand_input_mode": self.args.hand_input_mode,
                    "hand_udp_listen_host": self.args.hand_udp_listen_host,
                    "hand_udp_listen_port": self.args.hand_udp_listen_port,
                    "external_rgb_topic": self.args.external_rgb_topic,
                    "external_input_mode": self.args.external_input_mode,
                    "use_external_camera": self.args.use_external_camera,
                    "external_camera_required": self.args.external_camera_required,
                    "external_camera_fallback_to_internal": self.args.external_camera_fallback_to_internal,
                    "sync_tolerance_sec": self.args.sync_tolerance_sec,
                    "external_match_mode": self.args.external_match_mode,
                    "external_udp_listen_host": self.args.external_udp_listen_host,
                    "external_udp_listen_port": self.args.external_udp_listen_port,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                    "phase_layout": [
                        "pre_grasp",
                        "approach_far",
                        "approach_mid",
                        "approach_near",
                        "contact_ready",
                        "close",
                        "lift",
                    ],
                    "approach_offset": self.args.approach_offset,
                    "approach_mid_offset": self.args.approach_mid_offset,
                    "approach_near_offset": self.args.approach_near_offset,
                    "contact_ready_offset": self.args.contact_ready_offset,
                    "goal_xy_jitter_m": self.args.goal_xy_jitter_m,
                    "goal_z_jitter_m": self.args.goal_z_jitter_m,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.get_logger().info(
            "External VLA dataset collection ready. "
            f"run_dir={self.run_dir}, rgb_topic={self.args.hand_runtime_rgb_topic}, "
            f"hand_input_mode={self.args.hand_input_mode}, "
            f"external_input_mode={self.args.external_input_mode}, "
            f"external_rgb_topic={self.args.external_runtime_rgb_topic}, "
            f"external_match_mode={self.args.external_match_mode}, "
            f"use_external_camera={self.args.use_external_camera}"
        )

    def _format_frame_debug_status(self):
        hand_latest = (
            f"{self._latest_rgb_time:.3f}/{self._latest_rgb_receive_time:.3f}"
            if self._latest_rgb_time is not None and self._latest_rgb_receive_time is not None
            else "none"
        )
        external_latest = (
            f"{self._latest_external_rgb_time:.3f}/{self._latest_external_rgb_receive_time:.3f}"
            if self._latest_external_rgb_time is not None and self._latest_external_rgb_receive_time is not None
            else "none"
        )
        return (
            f"hand_topic={self.args.hand_runtime_rgb_topic}, "
            f"hand_buffer={len(self._hand_frame_buffer)}, hand_latest_stamp/recv={hand_latest}, "
            f"external_topic={self.args.external_runtime_rgb_topic}, "
            f"external_buffer={len(self._external_frame_buffer)}, "
            f"external_latest_stamp/recv={external_latest}"
        )

    def _extract_packet_rgb_stamp(self, packet, receive_time):
        camera_info = packet.get("camera_info") or {}
        rgb_stamp = camera_info.get("rgb_stamp")
        if rgb_stamp is not None:
            try:
                return float(rgb_stamp)
            except (TypeError, ValueError):
                pass
        stamp = packet.get("stamp")
        if stamp is not None:
            try:
                return float(stamp)
            except (TypeError, ValueError):
                pass
        return float(receive_time)

    def _append_hand_frame(self, rgb, stamp, receive_time, source):
        frame = {
            "rgb": rgb,
            "stamp": float(stamp) if stamp is not None else None,
            "receive_time": float(receive_time),
            "source": source,
        }
        self._hand_frame_buffer.append(frame)
        self._latest_rgb = rgb
        self._latest_rgb_time = frame["stamp"]
        self._latest_rgb_receive_time = frame["receive_time"]

    def _append_external_frame(self, rgb, stamp, receive_time, source):
        frame = {
            "rgb": rgb,
            "stamp": float(stamp) if stamp is not None else None,
            "receive_time": float(receive_time),
            "source": source,
        }
        self._external_frame_buffer.append(frame)
        self._latest_external_rgb = rgb
        self._latest_external_rgb_time = frame["stamp"]
        self._latest_external_rgb_receive_time = frame["receive_time"]

    def _setup_hand_udp_receiver(self):
        self._hand_udp_assembler = UdpChunkAssembler(stale_after_sec=self.args.hand_udp_stale_after_sec)
        self._hand_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._hand_udp_sock.bind((self.args.hand_udp_listen_host, self.args.hand_udp_listen_port))
        self._hand_udp_sock.setblocking(False)
        self._hand_udp_timer = self.create_timer(
            self.args.hand_udp_poll_period_sec,
            self._poll_hand_udp,
            callback_group=self.io_cb_group,
        )
        self.get_logger().info(
            "Hand UDP receiver enabled. "
            f"listen={self.args.hand_udp_listen_host}:{self.args.hand_udp_listen_port}"
        )

    def _setup_external_udp_receiver(self):
        self._external_udp_assembler = UdpChunkAssembler(
            stale_after_sec=self.args.external_udp_stale_after_sec
        )
        self._external_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._external_udp_sock.bind(
            (self.args.external_udp_listen_host, self.args.external_udp_listen_port)
        )
        self._external_udp_sock.setblocking(False)
        self._external_udp_timer = self.create_timer(
            self.args.external_udp_poll_period_sec,
            self._poll_external_udp,
            callback_group=self.io_cb_group,
        )
        self.get_logger().info(
            "External UDP receiver enabled. "
            f"listen={self.args.external_udp_listen_host}:{self.args.external_udp_listen_port}"
        )

    def on_rgb(self, msg: Image):
        if getattr(self.args, "hand_input_mode", "ros_topic") == "udp":
            return
        try:
            rgb = ros_image_to_rgb(msg)
            stamp = stamp_to_sec(msg.header.stamp)
            receive_time = time.time()
            self._append_hand_frame(rgb, stamp, receive_time, "hand_ros")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode RGB image: {exc}")

    def on_external_rgb(self, msg: Image):
        try:
            rgb = ros_image_to_rgb(msg)
            stamp = stamp_to_sec(msg.header.stamp)
            receive_time = time.time()
            self._append_external_frame(rgb, stamp, receive_time, "external_ros")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode external RGB image: {exc}")

    def _poll_hand_udp(self):
        if self._hand_udp_sock is None or self._hand_udp_assembler is None:
            return

        while True:
            try:
                datagram, _ = self._hand_udp_sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().warn(f"Hand UDP receive failed: {exc}")
                break

            payload = self._hand_udp_assembler.push(datagram)
            if payload is None:
                continue

            try:
                packet = parse_frame_payload(payload)
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse hand UDP payload: {exc}")
                continue

            now = time.time()
            packet_stamp = packet.get("stamp")
            if self.args.hand_udp_max_frame_age_sec > 0.0 and packet_stamp is not None:
                try:
                    frame_age = max(0.0, now - float(packet_stamp))
                except (TypeError, ValueError):
                    frame_age = None
                if frame_age is not None and frame_age > self.args.hand_udp_max_frame_age_sec:
                    continue

            rgb = packet.get("rgb")
            if rgb is None:
                continue
            rgb_stamp = self._extract_packet_rgb_stamp(packet, now)
            self._append_hand_frame(rgb, rgb_stamp, now, "hand_udp")

    def _poll_external_udp(self):
        if self._external_udp_sock is None or self._external_udp_assembler is None:
            return

        while True:
            try:
                datagram, _ = self._external_udp_sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().warn(f"External UDP receive failed: {exc}")
                break

            payload = self._external_udp_assembler.push(datagram)
            if payload is None:
                continue

            try:
                packet = parse_frame_payload(payload)
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse external UDP payload: {exc}")
                continue

            now = time.time()
            packet_stamp = packet.get("stamp")
            if (
                self.args.external_udp_max_frame_age_sec > 0.0
                and packet_stamp is not None
            ):
                try:
                    frame_age = max(0.0, now - float(packet_stamp))
                except (TypeError, ValueError):
                    frame_age = None
                if frame_age is not None and frame_age > self.args.external_udp_max_frame_age_sec:
                    continue

            rgb = packet.get("rgb")
            if rgb is None:
                continue
            rgb_stamp = self._extract_packet_rgb_stamp(packet, now)
            self._append_external_frame(rgb, rgb_stamp, now, "external_udp")

    def _phase_sample_period(self, phase: str) -> float:
        if phase in {"approach_far", "approach_mid", "approach_near", "contact_ready"}:
            return self.args.approach_sample_period_sec
        return self.args.frame_sample_period_sec

    def _maybe_record_frame(self, phase, joint_state, note="", force=False):
        if not self._recording_active and not force:
            return
        now = time.time()
        if not force and now < self._next_record_time:
            return
        self._save_openvla_sample(phase, joint_state, note)
        self._next_record_time = now + self._phase_sample_period(phase)

    def _jittered_goal_components(self, point_world):
        px = float(point_world[0])
        py = float(point_world[1])
        pz = float(point_world[2])
        if self.args.goal_xy_jitter_m > 0.0:
            px += random.uniform(-self.args.goal_xy_jitter_m, self.args.goal_xy_jitter_m)
            py += random.uniform(-self.args.goal_xy_jitter_m, self.args.goal_xy_jitter_m)
        if self.args.goal_z_jitter_m > 0.0:
            pz += random.uniform(-self.args.goal_z_jitter_m, self.args.goal_z_jitter_m)
        return px, py, pz

    def _capture_stage(self, phase, joint_state, note, pause_sec=None):
        pause_sec = self.args.stage_pause_sec if pause_sec is None else pause_sec
        if pause_sec > 0.0:
            self._pause_for_observation(pause_sec, hold_joint_state=joint_state)
        snapshot = self._settle_and_capture_joint_state(settle_sec=0.0)
        self._latest_js = copy_joint_state(snapshot)
        self._maybe_record_frame(phase, snapshot, note=note, force=True)
        return snapshot

    def _move_vertical_axis_locked(
        self,
        from_js,
        fixed_x,
        fixed_y,
        start_z,
        target_z,
        quat_xyzw,
        *,
        duration,
        rate_hz=40.0,
        phase=None,
        gripper_width=None,
    ):
        steps = max(1, int(duration * rate_hz))
        active_phase = self._phase if phase is None else phase
        current_js = copy_joint_state(from_js)

        for index in range(1, steps + 1):
            alpha = index / steps
            target_pos = (
                float(fixed_x),
                float(fixed_y),
                float((1.0 - alpha) * start_z + alpha * target_z),
            )
            solved_js = self.compute_ik(target_pos, quat_xyzw, seed_joint_state=current_js)
            if gripper_width is not None:
                apply_gripper(solved_js, gripper_width)
            self.pub.publish(solved_js)
            current_js = copy_joint_state(solved_js)
            self._latest_js = copy_joint_state(solved_js)
            self._maybe_record_frame(active_phase, self._latest_js, note="stream")
            time.sleep(1.0 / rate_hz)

        return current_js

    def _save_rgb_image(self, rgb, image_dir):
        ext = "png" if self.args.image_format == "png" else "jpg"
        image_name = f"sample_{self._sample_index:06d}.{ext}"
        image_path = os.path.join(image_dir, image_name)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ext == "png":
            ok = cv2.imwrite(image_path, bgr)
        else:
            ok = cv2.imwrite(image_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to save image: {image_path}")
        return image_path

    def _save_external_image(self, frame):
        image_path = self._save_rgb_image(frame["rgb"], self.external_images_dir)
        self._last_saved_external_rgb_receive_time = frame["receive_time"]
        return image_path

    def _save_hand_image(self, frame):
        image_path = self._save_rgb_image(frame["rgb"], self.images_dir)
        self._last_saved_rgb_receive_time = frame["receive_time"]
        return image_path

    def _frame_is_stale(self, frame, sample_time, max_staleness_sec):
        if frame is None:
            return True
        if max_staleness_sec <= 0.0:
            return False
        receive_time = frame.get("receive_time")
        if receive_time is None:
            return True
        return abs(sample_time - receive_time) > max_staleness_sec

    def _latest_frame_after(self, buffer_obj, baseline):
        if not buffer_obj:
            return None
        for frame in reversed(buffer_obj):
            receive_time = frame.get("receive_time")
            if baseline is None or (receive_time is not None and receive_time > baseline):
                return frame
        return None

    def _wait_for_buffered_frame(self, buffer_obj, wait_timeout_sec, baseline, poll_udp_fn=None):
        if self._latest_frame_after(buffer_obj, baseline) is not None:
            return True
        start = time.time()
        while time.time() - start < wait_timeout_sec:
            if poll_udp_fn is not None:
                poll_udp_fn()
            candidate = self._latest_frame_after(buffer_obj, baseline)
            if candidate is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.02)
        return self._latest_frame_after(buffer_obj, baseline) is not None

    def _select_hand_frame(self, sample_time):
        if self.args.hand_input_mode == "udp":
            self._poll_hand_udp()
        for frame in reversed(self._hand_frame_buffer):
            if not self._frame_is_stale(frame, sample_time, self.args.max_rgb_staleness_sec):
                return frame
        return None

    def _match_external_frame(self, hand_frame, sample_time):
        if not self.args.use_external_camera:
            return None, "disabled", None, "disabled"

        if self.args.external_input_mode == "udp":
            self._poll_external_udp()

        best_frame = None
        best_delta = None
        hand_stamp = hand_frame.get("stamp")
        hand_receive_time = hand_frame.get("receive_time")
        match_mode = getattr(self.args, "external_match_mode", "prefer_receive_time")
        for frame in self._external_frame_buffer:
            if self._frame_is_stale(frame, sample_time, self.args.max_external_rgb_staleness_sec):
                continue
            frame_stamp = frame.get("stamp")
            frame_receive_time = frame.get("receive_time")
            if (
                match_mode == "prefer_receive_time"
                and hand_receive_time is not None
                and frame_receive_time is not None
            ):
                delta = abs(hand_receive_time - frame_receive_time)
                delta_basis = "receive_time"
            elif hand_stamp is None or frame_stamp is None:
                delta = abs(sample_time - frame["receive_time"])
                delta_basis = "sample_time"
            else:
                delta = abs(hand_stamp - frame_stamp)
                delta_basis = "stamp"
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_frame = frame
                best_delta_basis = delta_basis

        if best_frame is not None:
            if self.args.sync_tolerance_sec <= 0.0 or best_delta <= self.args.sync_tolerance_sec:
                return best_frame, best_frame.get("source", "external"), best_delta, best_delta_basis

        if self.args.external_camera_required:
            raise RuntimeError("External camera frame unavailable or out of sync")
        return None, "missing", best_delta, locals().get("best_delta_basis", "unavailable")

    def _save_openvla_sample(self, phase, joint_state, note):
        now = time.time()
        if self.args.wait_for_rgb_before_start:
            self._wait_for_buffered_frame(
                self._hand_frame_buffer,
                self.args.hand_rgb_wait_timeout_sec,
                self._last_saved_rgb_receive_time,
                self._poll_hand_udp if self.args.hand_input_mode == "udp" else None,
            )
        if self.args.use_external_camera and self.args.external_camera_required:
            self._wait_for_buffered_frame(
                self._external_frame_buffer,
                self.args.external_rgb_wait_timeout_sec,
                self._last_saved_external_rgb_receive_time,
                self._poll_external_udp if self.args.external_input_mode == "udp" else None,
            )

        hand_frame = self._select_hand_frame(now)
        if hand_frame is None:
            raise RuntimeError("No fresh hand RGB image available for sample")

        image_path = self._save_hand_image(hand_frame)
        external_frame, external_image_source, external_match_delta_sec, external_match_basis = (
            self._match_external_frame(hand_frame, now)
        )
        external_image_path = None
        external_rgb_timestamp = None
        if external_frame is not None and external_image_source != "disabled":
            external_image_path = self._save_external_image(external_frame)
            external_rgb_timestamp = external_frame.get("stamp")
        ee_pose = self.ee_pose_from_fk(joint_state=joint_state)
        sample = {
            "image": os.path.relpath(image_path, self.run_dir),
            "instruction": self.args.instruction_text,
            "phase": phase,
            "note": note,
            "action": {
                "arm_joint_position": self._arm_joint_vector(joint_state),
                "gripper_width": gripper_width_from_joint_state(joint_state),
                "ee_pose": [
                    float(ee_pose.position[0]),
                    float(ee_pose.position[1]),
                    float(ee_pose.position[2]),
                    float(ee_pose.orientation[0]),
                    float(ee_pose.orientation[1]),
                    float(ee_pose.orientation[2]),
                    float(ee_pose.orientation[3]),
                ],
            },
            "timestamp": now,
            "rgb_timestamp": hand_frame.get("stamp"),
            "external_image_source": external_image_source,
            "external_match_basis": external_match_basis,
        }
        if external_match_delta_sec is not None:
            sample["external_match_delta_sec"] = external_match_delta_sec

        if external_image_path is not None:
            sample["external_image"] = os.path.relpath(external_image_path, self.run_dir)
            sample["external_rgb_timestamp"] = external_rgb_timestamp
            if hand_frame.get("stamp") is not None and external_rgb_timestamp is not None:
                sample["camera_sync_delta_sec"] = abs(hand_frame["stamp"] - external_rgb_timestamp)
            obs = sample.setdefault("observation", {})
            obs["external_image"] = sample["external_image"]

        with open(self.samples_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._sample_index += 1

    def try_execute(self):
        if self._executed or self._execution_in_progress or self._latest_js is None:
            return

        detection = self._locked_detection or self._latest_detection
        if detection is None:
            return

        if self.args.hand_input_mode == "udp":
            self._poll_hand_udp()
        if self.args.external_input_mode == "udp":
            self._poll_external_udp()

        if self.args.wait_for_rgb_before_start and not self._hand_frame_buffer:
            self.get_logger().warn(
                "Waiting for hand RGB image before starting. "
                f"source={self.args.hand_input_mode}, {self._format_frame_debug_status()}"
            )
            return

        if (
            self.args.use_external_camera
            and self.args.external_camera_required
            and not self._external_frame_buffer
        ):
            self.get_logger().warn(
                "Waiting for external RGB image before starting. "
                f"source={self.args.external_input_mode}, {self._format_frame_debug_status()}"
            )
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
            px, py, pz = self._jittered_goal_components(point_world)

            raw_goal_z = float(pz + self.args.grasp_offset)
            goal_z = max(raw_goal_z, self.args.min_goal_z)
            goal_pos = (px, py, goal_z)
            pre_pos = (px, py, goal_z + self.args.approach_offset)
            approach_mid_pos = (px, py, goal_z + self.args.approach_mid_offset)
            approach_near_pos = (px, py, goal_z + self.args.approach_near_offset)
            contact_ready_pos = (px, py, goal_z + self.args.contact_ready_offset)
            lift_pos = (px, py, goal_z + self.args.lift_offset)
            retreat_pos = (px, py, lift_pos[2] + self.args.retreat_offset)

            current_pose = self.ee_pose_from_fk()
            goal_quat = self._resolve_goal_orientation(current_pose.orientation)

            q_pre = self.compute_ik(pre_pos, goal_quat)
            q_mid = self.compute_ik(approach_mid_pos, goal_quat, seed_joint_state=q_pre)
            q_near = self.compute_ik(approach_near_pos, goal_quat, seed_joint_state=q_mid)
            q_ready = self.compute_ik(contact_ready_pos, goal_quat, seed_joint_state=q_near)
            q_goal_open = self.compute_ik(goal_pos, goal_quat, seed_joint_state=q_ready)

            for js in (q_pre, q_mid, q_near, q_ready, q_goal_open):
                apply_gripper(js, self.args.open_finger)

            current_js = copy_joint_state(self._latest_js)

            self._phase = "pre_grasp"
            self.move_smooth(current_js, q_pre, duration=self.args.duration, phase="pre_grasp_move")
            pre_js = self._capture_stage(
                "pre_grasp", q_pre, note="after_pre_grasp_pause", pause_sec=self.args.pre_grasp_pause_sec
            )

            self._start_continuous_recording()
            self._maybe_record_frame("pre_grasp", pre_js, note="pre_grasp_anchor", force=True)

            self._phase = "approach_far"
            q_mid = self._move_vertical_axis_locked(
                q_pre,
                px,
                py,
                pre_pos[2],
                approach_mid_pos[2],
                goal_quat,
                duration=self.args.duration * 0.75,
                phase="approach_far",
                gripper_width=self.args.open_finger,
            )
            mid_js = self._capture_stage("approach_far", q_mid, note="after_approach_far")

            self._phase = "approach_mid"
            q_near = self._move_vertical_axis_locked(
                mid_js,
                px,
                py,
                approach_mid_pos[2],
                approach_near_pos[2],
                goal_quat,
                duration=self.args.duration * 0.65,
                phase="approach_mid",
                gripper_width=self.args.open_finger,
            )
            near_js = self._capture_stage("approach_mid", q_near, note="after_approach_mid")

            self._phase = "approach_near"
            q_ready = self._move_vertical_axis_locked(
                near_js,
                px,
                py,
                approach_near_pos[2],
                contact_ready_pos[2],
                goal_quat,
                duration=self.args.duration * 0.55,
                phase="approach_near",
                gripper_width=self.args.open_finger,
            )
            ready_js = self._capture_stage("approach_near", q_ready, note="after_approach_near")

            self._phase = "contact_ready"
            q_goal_open = self._move_vertical_axis_locked(
                ready_js,
                px,
                py,
                contact_ready_pos[2],
                goal_pos[2],
                goal_quat,
                duration=self.args.duration * 0.45,
                phase="contact_ready",
                gripper_width=self.args.open_finger,
            )
            goal_open_js = self._capture_stage("contact_ready", q_goal_open, note="pre_close_ready")
            q_close = copy_joint_state(goal_open_js)
            apply_gripper(q_close, self.args.close_finger)

            self._phase = "close"
            self.move_smooth(goal_open_js, q_close, duration=self.args.gripper_motion_duration, phase="close")
            close_snapshot_js = self._settle_and_capture_joint_state()
            self._latest_js = copy_joint_state(close_snapshot_js)
            self._maybe_record_frame("close", close_snapshot_js, note="post_close", force=True)

            self._phase = "lift"
            q_lift = self.compute_ik(lift_pos, goal_quat)
            apply_gripper(q_lift, self.args.close_finger)
            self.move_smooth(q_close, q_lift, duration=self.args.duration, phase="lift")
            self._maybe_record_frame("lift", q_lift, note="post_lift", force=True)

            if self.args.enable_retreat:
                q_retreat = self.compute_ik(retreat_pos, goal_quat)
                apply_gripper(q_retreat, self.args.close_finger)
                self.move_smooth(q_lift, q_retreat, duration=self.args.duration, phase="retreat")
                self._maybe_record_frame("lift", q_retreat, note="post_retreat", force=True)

            self._executed = True
            self._stop_continuous_recording()
            self.get_logger().info(
                f"External VLA dataset saved to {self.samples_path} with {self._sample_index} samples"
            )
        except Exception as exc:
            self._stop_continuous_recording()
            self.get_logger().error(
                f"External VLA dataset collection failed at phase={self._phase}: {exc}. "
                f"{self._format_frame_debug_status()}"
            )
        finally:
            self._execution_in_progress = False

    def destroy_node(self):
        if self._hand_udp_sock is not None:
            try:
                self._hand_udp_sock.close()
            except OSError:
                pass
            self._hand_udp_sock = None
        if self._external_udp_sock is not None:
            try:
                self._external_udp_sock.close()
            except OSError:
                pass
            self._external_udp_sock = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExternalVLADatasetCollector()
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


# 객체탐지용 RGB-D 외부 sender: udp_camera_sender_debug_multi.py
# 외부카메라 이미지 저장용 RGB-only sender: udp_external_rgb_sender_debug.py
# 손목카메라 이미지 저장용 sender: udp_hand_camera_sender_debug.py

## 실행예시
# ros2 run robotarm_executor chair_grasp_moveit_vla_dataset_external --ros-args \
#   -p dataset_root:=/home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   -p hand_input_mode:=udp \
#   -p hand_udp_listen_host:=0.0.0.0 \
#   -p hand_udp_listen_port:=9005 \
#   -p external_input_mode:=udp \
#   -p external_udp_listen_host:=0.0.0.0 \
#   -p external_udp_listen_port:=9003 \
#   -p use_external_camera:=true \
#   -p external_camera_required:=false \
#   -p sync_tolerance_sec:=3.0
