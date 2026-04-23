#!/usr/bin/env python3

from collections import deque
import json
import math
import os
import socket
import sys
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
        if (candidate / "pi0_lite" / "pi0_lite_phase_aux_external.py").exists():
            return candidate

    search_roots = [THIS_DIR, Path.cwd().resolve()]
    for root in search_roots:
        for candidate in [root, *root.parents]:
            direct = candidate / "pi0_lite" / "pi0_lite_phase_aux_external.py"
            src_layout = candidate / "src" / "robotarm_project" / "pi0_lite" / "pi0_lite_phase_aux_external.py"
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
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "debug_runs" / "chair_grasp_openvla")
DEFAULT_PI0_LITE_EXTERNAL_CHECKPOINT = str(
    PI0_LITE_DIR / "pi0_lite_output" / "run_udp_aux_external" / "pi0_lite_phase_aux_external_final.pt"
)

if str(PI0_LITE_DIR) not in sys.path:
    sys.path.insert(0, str(PI0_LITE_DIR))

from pi0_lite_phase_aux_external import predict_single  # noqa: E402


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
    "rgb_wait_timeout_sec": 2.0,
    "max_rgb_staleness_sec": 2.0,
    "enable_retreat": False,
}

GROUP_NAME = "panda_arm"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

PI0_LITE_EXTERNAL_EXTRA_DEFAULTS = {
    "pi0_lite_checkpoint_path": DEFAULT_PI0_LITE_EXTERNAL_CHECKPOINT,
    "pi0_lite_device": "cuda",
    "pi0_lite_steps": 32,
    "pi0_lite_image_size": 224,
    "policy_max_steps": 20,
    "policy_timeout_sec": 120.0,
    "policy_action_duration": 0.5,
    "policy_action_rate_hz": 30.0,
    "gripper_close_min_ee_z": 0.30,
    "gripper_close_confirmation_steps": 2,
    "use_terminate_signal": False,
    "terminate_threshold": 0.5,
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
    "hand_rgb_wait_timeout_sec": 2.0,
    "external_rgb_wait_timeout_sec": 0.5,
    "sync_tolerance_sec": 100.0,
    "hand_frame_buffer_size": 30,
    "external_frame_buffer_size": 30,
    "max_external_rgb_staleness_sec": 2.0,
    "external_match_mode": "prefer_receive_time",
    "external_resize_scale": 1.0,
    "allow_recent_external_frame_reuse": True,
    "external_udp_listen_host": "0.0.0.0",
    "external_udp_listen_port": 9002,
    "external_udp_poll_period_sec": 0.01,
    "external_udp_stale_after_sec": 10.0,
    "external_udp_max_frame_age_sec": 0.0,
    "translation_scale": 1.0,
    "rotation_scale": 1.0,
    "max_delta_translation_m": 0.01,
    "max_delta_rotation_rad": 0.08,
    "gripper_close_threshold": 0.01,
    "gripper_open_threshold": -0.01,
}


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


def quat_normalize(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = quat_normalize(q1)
    x2, y2, z2, w2 = quat_normalize(q2)
    return quat_normalize(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def euler_xyz_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return quat_normalize(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


class OpenVLADatasetCollector(Node):
    def __init__(self, node_name="chair_grasp_moveit_openvla_dataset", param_overrides=None):
        super().__init__(node_name)

        defaults = dict(DEFAULT_PARAMS)
        if param_overrides:
            defaults.update(param_overrides)
        for name, value in defaults.items():
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

    def compute_ik(self, pos, quat_xyzw, timeout=2.0, avoid_collisions=False, ik_link_name=None):
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

        seed = RobotState()
        seed.joint_state = self._latest_js
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
        isaac_order = list(self._latest_js.name)
        base_map = dict(zip(self._latest_js.name, self._latest_js.position))
        base_map.update(solution_map)

        out = JointState()
        out.name = isaac_order
        out.position = [float(base_map.get(name, 0.0)) for name in isaac_order]
        return out

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

    def _start_continuous_recording(self):
        self._recording_active = True
        self._next_record_time = time.time()

    def _stop_continuous_recording(self):
        self._recording_active = False

    def _maybe_record_frame(self, phase, joint_state, note="", force=False):
        return

    def _settle_and_capture_joint_state(self, settle_sec=None):
        settle_sec = self.args.gripper_settle_sec if settle_sec is None else settle_sec
        deadline = time.time() + max(0.0, settle_sec)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.time())))
        if self._latest_js is None:
            raise RuntimeError("No joint state available after settle")
        return copy_joint_state(self._latest_js)

    def move_smooth(self, from_js, to_js, duration=1.5, rate_hz=100, phase=None):
        names = list(from_js.name)
        if names != list(to_js.name):
            raise RuntimeError("Joint name order mismatch")
        start = np.asarray(from_js.position, dtype=float)
        goal = np.asarray(to_js.position, dtype=float)
        steps = max(1, int(duration * rate_hz))
        for index in range(steps + 1):
            alpha = index / steps
            point = (1.0 - alpha) * start + alpha * goal
            msg = JointState()
            msg.name = names
            msg.position = point.tolist()
            self.pub.publish(msg)
            self._latest_js = copy_joint_state(msg)
            time.sleep(1.0 / rate_hz)

    def try_execute(self):
        return


class ExternalPi0LitePolicyExecutorCartesianStandalone(OpenVLADatasetCollector):
    def __init__(self):
        param_overrides = dict(DEFAULT_PARAMS)
        param_overrides.update(PI0_LITE_EXTERNAL_EXTRA_DEFAULTS)
        super().__init__(
            node_name="chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone",
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
        self.args.gripper_close_min_ee_z = float(self.get_parameter("gripper_close_min_ee_z").value)
        self.args.gripper_close_confirmation_steps = max(
            1, int(self.get_parameter("gripper_close_confirmation_steps").value)
        )
        self.args.use_terminate_signal = bool(self.get_parameter("use_terminate_signal").value)
        self.args.terminate_threshold = float(self.get_parameter("terminate_threshold").value)
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
        self.args.external_rgb_wait_timeout_sec = max(
            0.0, float(self.get_parameter("external_rgb_wait_timeout_sec").value)
        )
        self.args.sync_tolerance_sec = max(0.0, float(self.get_parameter("sync_tolerance_sec").value))
        self.args.hand_frame_buffer_size = max(2, int(self.get_parameter("hand_frame_buffer_size").value))
        self.args.external_frame_buffer_size = max(
            2, int(self.get_parameter("external_frame_buffer_size").value)
        )
        self.args.max_external_rgb_staleness_sec = max(
            0.0, float(self.get_parameter("max_external_rgb_staleness_sec").value)
        )
        self.args.external_match_mode = str(self.get_parameter("external_match_mode").value).strip().lower()
        self.args.external_resize_scale = max(0.01, float(self.get_parameter("external_resize_scale").value))
        self.args.allow_recent_external_frame_reuse = bool(
            self.get_parameter("allow_recent_external_frame_reuse").value
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
        self.args.translation_scale = float(self.get_parameter("translation_scale").value)
        self.args.rotation_scale = float(self.get_parameter("rotation_scale").value)
        self.args.max_delta_translation_m = float(self.get_parameter("max_delta_translation_m").value)
        self.args.max_delta_rotation_rad = float(self.get_parameter("max_delta_rotation_rad").value)
        self.args.gripper_close_threshold = float(self.get_parameter("gripper_close_threshold").value)
        self.args.gripper_open_threshold = float(self.get_parameter("gripper_open_threshold").value)

        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
        self._policy_capture_start_time = None
        self._policy_hand_start_stamp = None
        self._policy_external_start_stamp = None
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
        self.policy_images_dir = os.path.join(self.run_dir, "policy_images")
        os.makedirs(self.policy_images_dir, exist_ok=True)
        self.external_policy_images_dir = os.path.join(self.run_dir, "external_policy_images")
        os.makedirs(self.external_policy_images_dir, exist_ok=True)

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

        self.policy_log_path = os.path.join(self.run_dir, "policy_log.jsonl")
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": "pre_grasp_then_pi0_lite_policy_external_cartesian_standalone",
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
                    "external_resize_scale": self.args.external_resize_scale,
                    "allow_recent_external_frame_reuse": self.args.allow_recent_external_frame_reuse,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                    "pi0_lite_checkpoint_path": self.args.pi0_lite_checkpoint_path,
                    "pi0_lite_device": self.args.pi0_lite_device,
                    "pi0_lite_steps": self.args.pi0_lite_steps,
                    "pi0_lite_image_size": self.args.pi0_lite_image_size,
                    "gripper_close_min_ee_z": self.args.gripper_close_min_ee_z,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                    "translation_scale": self.args.translation_scale,
                    "rotation_scale": self.args.rotation_scale,
                    "max_delta_translation_m": self.args.max_delta_translation_m,
                    "max_delta_rotation_rad": self.args.max_delta_rotation_rad,
                    "gripper_close_threshold": self.args.gripper_close_threshold,
                    "gripper_open_threshold": self.args.gripper_open_threshold,
                    "external_udp_listen_host": self.args.external_udp_listen_host,
                    "external_udp_listen_port": self.args.external_udp_listen_port,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.get_logger().info(
            "Pi0-Lite external cartesian standalone policy ready. "
            f"run_dir={self.run_dir}, checkpoint={self.args.pi0_lite_checkpoint_path}, "
            f"hand_input_mode={self.args.hand_input_mode}, external_input_mode={self.args.external_input_mode}, "
            f"hand_rgb_topic={self.args.rgb_topic}, external_rgb_topic={self.args.external_rgb_topic}, "
            f"translation_scale={self.args.translation_scale:.3f}, rotation_scale={self.args.rotation_scale:.3f}, "
            f"external_match_mode={self.args.external_match_mode}, "
            f"external_resize_scale={self.args.external_resize_scale:.2f}"
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
            if self.args.external_resize_scale != 1.0:
                target_w = max(1, int(round(rgb.shape[1] * self.args.external_resize_scale)))
                target_h = max(1, int(round(rgb.shape[0] * self.args.external_resize_scale)))
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            stamp = stamp_to_sec(msg.header.stamp)
            receive_time = time.time()
            self._append_external_frame(rgb, stamp, receive_time, "external_ros_direct")
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
            if self.args.external_udp_max_frame_age_sec > 0.0 and packet_stamp is not None:
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

    def _save_policy_rgb(self, rgb, image_dir):
        ext = "png" if self.args.image_format == "png" else "jpg"
        image_name = f"policy_{self._sample_index:06d}.{ext}"
        image_path = os.path.join(image_dir, image_name)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ext == "png":
            ok = cv2.imwrite(image_path, bgr)
        else:
            ok = cv2.imwrite(image_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to save policy image: {image_path}")
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

    def _frame_is_after_baseline(self, frame, receive_baseline=None, stamp_baseline=None):
        if frame is None:
            return False
        receive_time = frame.get("receive_time")
        if receive_baseline is not None and (receive_time is None or receive_time <= receive_baseline):
            return False
        if stamp_baseline is not None:
            stamp = frame.get("stamp")
            if stamp is None or stamp <= stamp_baseline:
                return False
        return True

    def _latest_frame_after(self, buffer_obj, receive_baseline=None, stamp_baseline=None):
        if not buffer_obj:
            return None
        for frame in reversed(buffer_obj):
            if self._frame_is_after_baseline(
                frame,
                receive_baseline=receive_baseline,
                stamp_baseline=stamp_baseline,
            ):
                return frame
        return None

    def _wait_for_buffered_frame(
        self,
        buffer_obj,
        wait_timeout_sec,
        receive_baseline,
        stamp_baseline=None,
        poll_udp_fn=None,
    ):
        if self._latest_frame_after(
            buffer_obj,
            receive_baseline=receive_baseline,
            stamp_baseline=stamp_baseline,
        ) is not None:
            return True
        start = time.time()
        while time.time() - start < wait_timeout_sec:
            if poll_udp_fn is not None:
                poll_udp_fn()
            candidate = self._latest_frame_after(
                buffer_obj,
                receive_baseline=receive_baseline,
                stamp_baseline=stamp_baseline,
            )
            if candidate is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.02)
        return self._latest_frame_after(
            buffer_obj,
            receive_baseline=receive_baseline,
            stamp_baseline=stamp_baseline,
        ) is not None

    def _prune_buffer_before(self, buffer_obj, receive_baseline=None, stamp_baseline=None):
        if (receive_baseline is None and stamp_baseline is None) or not buffer_obj:
            return
        kept = [
            frame
            for frame in buffer_obj
            if self._frame_is_after_baseline(
                frame,
                receive_baseline=receive_baseline,
                stamp_baseline=stamp_baseline,
            )
        ]
        buffer_obj.clear()
        buffer_obj.extend(kept)

    def _frame_match_delta(self, hand_frame, frame, sample_time):
        match_mode = getattr(self.args, "external_match_mode", "prefer_receive_time")
        hand_stamp = hand_frame.get("stamp")
        frame_stamp = frame.get("stamp")
        hand_receive_time = hand_frame.get("receive_time")
        frame_receive_time = frame.get("receive_time")

        if match_mode == "prefer_receive_time":
            if hand_receive_time is not None and frame_receive_time is not None:
                return abs(hand_receive_time - frame_receive_time), "receive_time"
            if hand_stamp is not None and frame_stamp is not None:
                return abs(hand_stamp - frame_stamp), "stamp"
        elif match_mode == "stamp":
            if hand_stamp is not None and frame_stamp is not None:
                return abs(hand_stamp - frame_stamp), "stamp"
            if hand_receive_time is not None and frame_receive_time is not None:
                return abs(hand_receive_time - frame_receive_time), "receive_time"

        if hand_stamp is not None and frame_stamp is not None:
            return abs(hand_stamp - frame_stamp), "stamp"
        if hand_receive_time is not None and frame_receive_time is not None:
            return abs(hand_receive_time - frame_receive_time), "receive_time"
        return abs(sample_time - frame["receive_time"]), "sample_time"

    def _select_hand_frame(self, sample_time, receive_baseline=None, stamp_baseline=None):
        if self.args.hand_input_mode == "udp":
            self._poll_hand_udp()
        for frame in reversed(self._hand_frame_buffer):
            if not self._frame_is_after_baseline(
                frame,
                receive_baseline=receive_baseline,
                stamp_baseline=stamp_baseline,
            ):
                continue
            if not self._frame_is_stale(frame, sample_time, self.args.max_rgb_staleness_sec):
                return frame
        return None

    def _match_external_frame(self, hand_frame, sample_time, receive_baseline=None, stamp_baseline=None):
        if not self.args.use_external_camera:
            return None, "disabled", None, None
        if self.args.external_input_mode == "udp":
            self._poll_external_udp()

        def choose_best_frame(ignore_baseline=False):
            best_frame = None
            best_delta = None
            best_basis = None
            for frame in self._external_frame_buffer:
                if not ignore_baseline and not self._frame_is_after_baseline(
                    frame,
                    receive_baseline=receive_baseline,
                    stamp_baseline=stamp_baseline,
                ):
                    continue
                if self._frame_is_stale(frame, sample_time, self.args.max_external_rgb_staleness_sec):
                    continue
                delta, basis = self._frame_match_delta(hand_frame, frame, sample_time)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_frame = frame
                    best_basis = basis
            return best_frame, best_delta, best_basis

        best_frame, best_delta, best_basis = choose_best_frame(ignore_baseline=False)
        best_source = best_frame.get("source", "external") if best_frame is not None else None
        if best_frame is None and self.args.allow_recent_external_frame_reuse and self._external_frame_buffer:
            best_frame, best_delta, best_basis = choose_best_frame(ignore_baseline=True)
            if best_frame is not None:
                best_source = f"{best_frame.get('source', 'external')}_reused"

        if best_frame is not None:
            if self.args.sync_tolerance_sec <= 0.0 or best_delta <= self.args.sync_tolerance_sec:
                return best_frame, best_source, best_delta, best_basis

        if self.args.external_camera_fallback_to_internal:
            return None, "internal_fallback", None, None
        if self.args.external_camera_required:
            raise RuntimeError("External image is required for policy inference but no synced frame was found")
        return None, "missing", None, None

    def _capture_policy_frame(self):
        now = time.time()
        receive_baseline = self._policy_capture_start_time
        hand_stamp_baseline = self._policy_hand_start_stamp
        external_stamp_baseline = self._policy_external_start_stamp
        if self.args.wait_for_rgb_before_start:
            self._wait_for_buffered_frame(
                self._hand_frame_buffer,
                self.args.hand_rgb_wait_timeout_sec,
                max(
                    value
                    for value in [self._last_saved_rgb_receive_time, receive_baseline]
                    if value is not None
                )
                if any(value is not None for value in [self._last_saved_rgb_receive_time, receive_baseline])
                else None,
                hand_stamp_baseline,
                self._poll_hand_udp if self.args.hand_input_mode == "udp" else None,
            )
        if self.args.use_external_camera and self.args.external_camera_required:
            self._wait_for_buffered_frame(
                self._external_frame_buffer,
                self.args.external_rgb_wait_timeout_sec,
                max(
                    value
                    for value in [self._last_saved_external_rgb_receive_time, receive_baseline]
                    if value is not None
                )
                if any(
                    value is not None
                    for value in [self._last_saved_external_rgb_receive_time, receive_baseline]
                )
                else None,
                external_stamp_baseline,
                self._poll_external_udp if self.args.external_input_mode == "udp" else None,
            )

        hand_frame = self._select_hand_frame(
            now,
            receive_baseline=receive_baseline,
            stamp_baseline=hand_stamp_baseline,
        )
        if hand_frame is None:
            raise RuntimeError("No fresh hand RGB image available for policy inference")
        external_frame, external_source, external_match_delta_sec, external_match_basis = self._match_external_frame(
            hand_frame,
            now,
            receive_baseline=receive_baseline,
            stamp_baseline=external_stamp_baseline,
        )

        image_path = self._save_policy_rgb(hand_frame["rgb"], self.policy_images_dir)
        self._last_saved_rgb_receive_time = hand_frame.get("receive_time")
        self._sample_index += 1

        external_image_path = None
        external_timestamp = None
        external_receive_time = None
        camera_sync_delta_sec = None
        if external_frame is not None and external_source != "disabled":
            external_image_path = self._save_policy_rgb(external_frame["rgb"], self.external_policy_images_dir)
            external_timestamp = external_frame.get("stamp")
            external_receive_time = external_frame.get("receive_time")
            self._last_saved_external_rgb_receive_time = external_receive_time
            if hand_frame.get("stamp") is not None and external_timestamp is not None:
                camera_sync_delta_sec = abs(hand_frame["stamp"] - external_timestamp)
        elif external_source == "internal_fallback":
            external_image_path = image_path

        frame_item = {
            "image_path": image_path,
            "external_image_path": external_image_path,
            "rgb_timestamp": hand_frame.get("stamp"),
            "external_rgb_timestamp": external_timestamp,
            "receive_time": hand_frame.get("receive_time"),
            "external_receive_time": external_receive_time,
            "capture_time": now,
            "sample_index": self._sample_index - 1,
            "external_image_source": external_source,
            "camera_sync_delta_sec": camera_sync_delta_sec,
            "external_match_delta_sec": external_match_delta_sec,
            "external_match_basis": external_match_basis,
        }
        if self.args.use_external_camera and frame_item["external_image_source"] == "missing":
            self.get_logger().warn(
                "Policy inference is running without an external frame. "
                f"sample_index={frame_item['sample_index']}, {self._format_policy_buffer_status()}"
            )
        return frame_item

    def _wait_for_policy_start_frames(self):
        deadline = time.time() + max(
            self.args.hand_rgb_wait_timeout_sec,
            self.args.external_rgb_wait_timeout_sec,
            0.5,
        )
        while time.time() < deadline:
            now = time.time()
            hand_frame = self._select_hand_frame(
                now,
                receive_baseline=self._policy_capture_start_time,
                stamp_baseline=self._policy_hand_start_stamp,
            )
            if hand_frame is None:
                rclpy.spin_once(self, timeout_sec=0.02)
                continue

            external_frame, external_source, _, _ = self._match_external_frame(
                hand_frame,
                now,
                receive_baseline=self._policy_capture_start_time,
                stamp_baseline=self._policy_external_start_stamp,
            )
            if not self.args.use_external_camera or external_source == "disabled":
                self._last_saved_rgb_receive_time = self._policy_capture_start_time
                self._last_saved_external_rgb_receive_time = self._policy_capture_start_time
                return
            if external_frame is not None:
                self._last_saved_rgb_receive_time = self._policy_capture_start_time
                self._last_saved_external_rgb_receive_time = self._policy_capture_start_time
                return
            rclpy.spin_once(self, timeout_sec=0.02)

        if self.args.use_external_camera and self.args.external_camera_required:
            raise RuntimeError("No synced post-pre-grasp external frame available before policy start")

    def _format_policy_buffer_status(self):
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
            f"hand_buffer={len(self._hand_frame_buffer)}, hand_latest_stamp/recv={hand_latest}, "
            f"external_buffer={len(self._external_frame_buffer)}, external_latest_stamp/recv={external_latest}"
        )

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
            external_image_path=frame_item["external_image_path"],
            instruction=self.args.instruction_text,
            state=state_vec,
            image_size=image_size,
            steps=self.args.pi0_lite_steps,
            device=self.args.pi0_lite_device,
        )
        latency_sec = time.time() - start

        first = pred["first_action"]
        action_format = pred.get("action_format", "")
        if action_format != "cartesian_delta":
            raise RuntimeError(
                f"Pi0-Lite external cartesian executor expects cartesian_delta output, got action_format={action_format}"
            )

        action = np.asarray(
            list(first["world_vector"])
            + list(first["rotation_delta"])
            + list(first["gripper_closedness_action"])
            + [float(first["terminate_episode"])],
            dtype=np.float32,
        )
        return action, latency_sec, pred, state_vec

    def _apply_cartesian_delta(self, current_pose, action):
        translation = np.asarray(action[:3], dtype=np.float32) * float(self.args.translation_scale)
        rotation = np.asarray(action[3:6], dtype=np.float32) * float(self.args.rotation_scale)
        gripper_action = float(action[6])

        if self.args.max_delta_translation_m > 0.0:
            translation = np.clip(
                translation,
                -float(self.args.max_delta_translation_m),
                float(self.args.max_delta_translation_m),
            )
        if self.args.max_delta_rotation_rad > 0.0:
            rotation = np.clip(
                rotation,
                -float(self.args.max_delta_rotation_rad),
                float(self.args.max_delta_rotation_rad),
            )

        current_pos = np.asarray(current_pose.position, dtype=np.float32)
        target_pos = current_pos + translation
        delta_quat = euler_xyz_to_quat(rotation[0], rotation[1], rotation[2])
        target_quat = quat_multiply(delta_quat, current_pose.orientation)
        q_next = self.compute_ik(target_pos, target_quat)

        close_requested = False
        if gripper_action > float(self.args.gripper_close_threshold):
            apply_gripper(q_next, self.args.close_finger)
            close_requested = True
        elif gripper_action < float(self.args.gripper_open_threshold):
            apply_gripper(q_next, self.args.open_finger)

        return q_next, target_pos, target_quat, translation, rotation, gripper_action, close_requested

    def run_pi0_lite_policy_loop(self):
        start_time = time.time()
        step_index = 0
        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
        self.get_logger().info("Pi0-Lite external cartesian standalone policy loop started from pre-grasp pose")

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
            (
                q_next,
                target_pos,
                target_quat,
                clipped_translation,
                clipped_rotation,
                gripper_action,
                close_signal_model,
            ) = self._apply_cartesian_delta(current_pose, action)

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

            terminate_value = float(action[7])
            terminate_requested = self.args.use_terminate_signal and (
                terminate_value >= self.args.terminate_threshold
            )
            margin_to_threshold = float(gripper_action - self.args.gripper_close_threshold)
            self._policy_gripper_deltas.append(float(gripper_action))
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
                    "applied_world_vector": [float(v) for v in clipped_translation],
                    "applied_rotation_delta": [float(v) for v in clipped_rotation],
                    "gripper_closedness_action": float(gripper_action),
                    "gripper_close_threshold": float(self.args.gripper_close_threshold),
                    "gripper_action_margin_to_close_threshold": float(margin_to_threshold),
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
                    "target_ee_pose": [float(v) for v in target_pos] + [float(v) for v in target_quat],
                    "resolved_target_ee_pose": [float(v) for v in next_pose.position]
                    + [float(v) for v in next_pose.orientation],
                    "joint_target": [float(v) for v in q_next.position],
                    "timestamp": float(time.time()),
                }
            )

            self.get_logger().info(
                "Cartesian standalone policy step summary. "
                f"step={step_index}, latency_sec={float(latency_sec):.3f}, "
                f"external_source={frame_item['external_image_source']}, "
                f"phase_pred={pred.get('predicted_phase')}, "
                f"dz={float(clipped_translation[2]):.6f}, "
                f"gripper_action={float(gripper_action):.6f}, "
                f"terminate={float(terminate_value):.6f}"
            )

            if close_requested:
                self.get_logger().info(
                    "Pi0-Lite external cartesian standalone policy requested gripper close; ending policy loop"
                )
                break
            if terminate_requested:
                self.get_logger().info(
                    "Pi0-Lite external cartesian standalone policy requested termination "
                    f"with value={terminate_value:.4f}"
                )
                break
            step_index += 1

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
                f"Waiting for hand RGB image from {self.args.hand_input_mode} source before starting"
            )
            return
        if (
            self.args.use_external_camera
            and self.args.external_camera_required
            and not self._external_frame_buffer
        ):
            self.get_logger().warn("Waiting for external camera input before starting")
            return

        detection_age_sec = self._extract_detection_age_sec(detection)
        if (
            (self._locked_detection is None or not self.args.ignore_stale_when_locked)
            and detection_age_sec is not None
            and detection_age_sec > self.args.detection_stale_sec
        ):
            self.get_logger().warn(
                f"Detection is stale: age={detection_age_sec:.3f}s, "
                f"limit={self.args.detection_stale_sec:.3f}s"
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
            self._policy_capture_start_time = time.time()
            self._policy_hand_start_stamp = self._latest_rgb_time
            self._policy_external_start_stamp = self._latest_external_rgb_time
            self._prune_buffer_before(
                self._hand_frame_buffer,
                receive_baseline=self._policy_capture_start_time,
                stamp_baseline=self._policy_hand_start_stamp,
            )
            self._prune_buffer_before(
                self._external_frame_buffer,
                receive_baseline=self._policy_capture_start_time,
                stamp_baseline=self._policy_external_start_stamp,
            )
            self._last_saved_rgb_receive_time = self._policy_capture_start_time
            self._last_saved_external_rgb_receive_time = self._policy_capture_start_time
            self._wait_for_policy_start_frames()

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
            self.get_logger().info(f"Pi0-Lite external cartesian standalone run finished. log_path={self.policy_log_path}")
        except Exception as exc:
            self.get_logger().error(
                f"Pi0-Lite external cartesian standalone policy failed at phase={self._phase}: {exc}"
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
    node = ExternalPi0LitePolicyExecutorCartesianStandalone()
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


# ros2 run robotarm_executor chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone --ros-args \
#   -p use_external_camera:=true \
#   -p policy_timeout_sec:=300.0 \
#   -p policy_max_steps:=50 \
#   -p translation_scale:=6.0
