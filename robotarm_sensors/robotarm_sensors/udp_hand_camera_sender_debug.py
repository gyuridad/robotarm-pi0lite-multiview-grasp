#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import socket
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from robotarm_common.chair_grasp_common import make_frame_payload, send_udp_chunks
from robotarm_common.lda_debug import DEFAULT_DEBUG_BASE_DIR, LDADebugger, ReasonCode


def resize_camera_info(msg: CameraInfo, scale: float) -> CameraInfo:
    resized = CameraInfo()
    resized.header = msg.header
    resized.height = max(1, int(round(msg.height * scale)))
    resized.width = max(1, int(round(msg.width * scale)))
    resized.distortion_model = msg.distortion_model
    resized.d = list(msg.d)
    resized.k = list(msg.k)
    resized.r = list(msg.r)
    resized.p = list(msg.p)

    resized.k[0] *= scale
    resized.k[2] *= scale
    resized.k[4] *= scale
    resized.k[5] *= scale
    resized.p[0] *= scale
    resized.p[2] *= scale
    resized.p[5] *= scale
    resized.p[6] *= scale
    return resized


def parse_target_csv(raw: str, default_ip: str, default_port: int) -> List[Tuple[str, int]]:
    text = (raw or "").strip()
    if not text:
        return [(default_ip, int(default_port))]

    targets: List[Tuple[str, int]] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid target entry '{item}'. Expected ip:port")
        host, port_text = item.rsplit(":", 1)
        targets.append((host.strip(), int(port_text.strip())))

    if not targets:
        raise ValueError("No valid UDP targets parsed from dest_targets_csv")
    return targets


class IsaacHandCameraUdpSenderDebug(Node):
    def __init__(self):
        super().__init__("isaac_hand_camera_udp_sender_debug")

        self.declare_parameter("rgb_topic", "/hand_Camera_rgb")
        self.declare_parameter("depth_topic", "/hand_Camera_depth")
        self.declare_parameter("camera_info_topic", "/hand_Camera_info")
        self.declare_parameter("allow_rgb_only", True)

        self.declare_parameter("dest_ip", "127.0.0.1")
        self.declare_parameter("dest_port", 9005)
        self.declare_parameter("dest_targets_csv", "127.0.0.1:9005")
        self.declare_parameter("send_hz", 10.0)
        self.declare_parameter("max_chunk_payload", 12000)
        self.declare_parameter("resize_scale", 1.0)
        self.declare_parameter("camera_frame_override", "")

        self.declare_parameter("qos_reliable", True)
        self.declare_parameter("qos_depth", 20)

        self.declare_parameter("robot_name", "robotarm")
        self.declare_parameter("lda_debug_dir", DEFAULT_DEBUG_BASE_DIR)
        self.declare_parameter("heartbeat_sec", 5.0)
        self.declare_parameter("summary_period_sec", 1.0)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.allow_rgb_only = bool(self.get_parameter("allow_rgb_only").value)

        self.dest_ip = str(self.get_parameter("dest_ip").value)
        self.dest_port = int(self.get_parameter("dest_port").value)
        self.dest_targets_csv = str(self.get_parameter("dest_targets_csv").value)
        self.dest_targets = parse_target_csv(self.dest_targets_csv, self.dest_ip, self.dest_port)
        self.send_hz = float(self.get_parameter("send_hz").value)
        self.max_chunk_payload = int(self.get_parameter("max_chunk_payload").value)
        self.resize_scale = float(self.get_parameter("resize_scale").value)
        self.camera_frame_override = str(self.get_parameter("camera_frame_override").value)
        self.qos_reliable = bool(self.get_parameter("qos_reliable").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)

        self.robot_name = str(self.get_parameter("robot_name").value)
        self.lda_debug_dir = str(self.get_parameter("lda_debug_dir").value)
        self.heartbeat_sec = float(self.get_parameter("heartbeat_sec").value)
        self.summary_period_sec = float(self.get_parameter("summary_period_sec").value)

        self.dbg = LDADebugger(
            node_name="isaac_hand_camera_udp_sender_debug",
            robot_name=self.robot_name,
            namespace=self.get_namespace(),
            base_dir=self.lda_debug_dir,
            also_print=False,
        )

        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if self.qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
        )
        qos_caminfo = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if self.qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))

        self.bridge = CvBridge()
        self.rgb_queue = deque(maxlen=1)
        self.depth_queue = deque(maxlen=1)
        self.latest_camera_info: Optional[CameraInfo] = None

        self.frame_id = 0
        self.last_heartbeat_time = 0.0
        self.last_summary_time = time.time()
        self.frames_sent_since_summary = 0
        self.bytes_sent_since_summary = 0
        self.last_encode_latency_sec: Optional[float] = None
        self.last_send_latency_sec: Optional[float] = None

        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, qos_img)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_img)
        self.sub_camera_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.cb_camera_info, qos_caminfo
        )
        self.timer = self.create_timer(1.0 / max(1e-6, self.send_hz), self.on_timer_send)

        self.dbg.write_interface_snapshot(
            publishes=[],
            subscribes=[
                {"name": self.rgb_topic, "msg_type": "sensor_msgs/msg/Image", "direction": "subscribe", "required": True},
                {"name": self.depth_topic, "msg_type": "sensor_msgs/msg/Image", "direction": "subscribe", "required": True},
                {"name": self.camera_info_topic, "msg_type": "sensor_msgs/msg/CameraInfo", "direction": "subscribe", "required": True},
            ],
            frames={},
            params={
                "rgb_topic": self.rgb_topic,
                "depth_topic": self.depth_topic,
                "camera_info_topic": self.camera_info_topic,
                "allow_rgb_only": self.allow_rgb_only,
                "dest_targets_csv": self.dest_targets_csv,
                "send_hz": self.send_hz,
                "max_chunk_payload": self.max_chunk_payload,
                "resize_scale": self.resize_scale,
                "camera_frame_override": self.camera_frame_override,
                "qos_reliable": self.qos_reliable,
                "qos_depth": self.qos_depth,
                "robot_name": self.robot_name,
                "lda_debug_dir": self.lda_debug_dir,
            },
        )

        self.dbg.emit_event(
            "INFO",
            "node_started",
            ReasonCode.COMMON.NODE_STARTED,
            {
                "rgb_topic": self.rgb_topic,
                "depth_topic": self.depth_topic,
                "camera_info_topic": self.camera_info_topic,
                "allow_rgb_only": self.allow_rgb_only,
                "udp_targets": [f"{host}:{port}" for host, port in self.dest_targets],
                "send_hz": self.send_hz,
                "max_chunk_payload": self.max_chunk_payload,
                "resize_scale": self.resize_scale,
            },
        )

        self.get_logger().info(
            f"[IsaacHandCameraUdpSenderDebug] rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"camera_info={self.camera_info_topic}, "
            f"allow_rgb_only={self.allow_rgb_only}, "
            f"udp_targets={','.join(f'{host}:{port}' for host, port in self.dest_targets)}, "
            f"send_hz={self.send_hz:.2f}, chunk={self.max_chunk_payload}, "
            f"resize_scale={self.resize_scale:.2f}, "
            f"qos={'RELIABLE' if self.qos_reliable else 'BEST_EFFORT'}"
        )

    def cb_rgb(self, msg: Image):
        try:
            enc = msg.encoding.lower()
            callback_stamp = time.time()

            if enc == "rgb8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                rgb = img.copy()
            elif enc == "bgr8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif enc == "rgba8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif enc == "bgra8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.rgb_queue.append((rgb, stamp_sec))
            self.dbg.emit_event(
                "DEBUG",
                "rgb_received",
                ReasonCode.VISION.IMAGE_RECEIVED,
                {
                    "encoding": enc,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "queue_size": len(self.rgb_queue),
                    "rgb_stamp": stamp_sec,
                    "callback_stamp": callback_stamp,
                },
            )
        except Exception as exc:
            self.dbg.emit_event(
                "WARN",
                "rgb_decode_failed",
                ReasonCode.VISION.POSE_ESTIMATION_FAILED,
                {"error": repr(exc)},
            )
            self.get_logger().warn(f"cb_rgb error: {exc}")

    def cb_depth(self, msg: Image):
        try:
            callback_stamp = time.time()

            if msg.encoding.upper() == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            elif msg.encoding.upper() == "16UC1":
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
                depth = raw.astype(np.float32) / 1000.0
            else:
                converted = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth = converted.astype(np.float32)

            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.depth_queue.append((depth, stamp_sec, msg.encoding))
            self.dbg.emit_event(
                "DEBUG",
                "depth_received",
                ReasonCode.VISION.IMAGE_RECEIVED,
                {
                    "encoding": msg.encoding,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "queue_size": len(self.depth_queue),
                    "depth_stamp": stamp_sec,
                    "callback_stamp": callback_stamp,
                },
            )
        except Exception as exc:
            self.dbg.emit_event(
                "WARN",
                "depth_decode_failed",
                ReasonCode.VISION.POSE_ESTIMATION_FAILED,
                {"error": repr(exc)},
            )
            self.get_logger().warn(f"cb_depth error: {exc}")

    def cb_camera_info(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def _make_fallback_camera_info(self, width: int, height: int) -> CameraInfo:
        msg = CameraInfo()
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = max(1.0, float(width))
        fy = max(1.0, float(height))
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def on_timer_send(self):
        now = time.time()

        if not self.rgb_queue:
            self.dbg.emit_event(
                "WARN",
                "frame_insufficient",
                ReasonCode.VISION.SOURCE_IMAGE_STALE,
                {
                    "rgb_queue_size": len(self.rgb_queue),
                    "depth_queue_size": len(self.depth_queue),
                    "camera_info_ready": self.latest_camera_info is not None,
                    "allow_rgb_only": self.allow_rgb_only,
                },
            )
            return

        try:
            rgb, rgb_stamp = self.rgb_queue[-1]
            if self.depth_queue:
                depth, depth_stamp, depth_encoding = self.depth_queue[-1]
            elif self.allow_rgb_only:
                depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                depth_stamp = rgb_stamp
                depth_encoding = "32FC1"
            else:
                self.dbg.emit_event(
                    "WARN",
                    "frame_insufficient",
                    ReasonCode.VISION.SOURCE_IMAGE_STALE,
                    {
                        "rgb_queue_size": len(self.rgb_queue),
                        "depth_queue_size": len(self.depth_queue),
                        "camera_info_ready": self.latest_camera_info is not None,
                        "allow_rgb_only": self.allow_rgb_only,
                    },
                )
                return

            if self.latest_camera_info is not None:
                camera_info_msg = self.latest_camera_info
            elif self.allow_rgb_only:
                camera_info_msg = self._make_fallback_camera_info(rgb.shape[1], rgb.shape[0])
            else:
                self.dbg.emit_event(
                    "WARN",
                    "frame_insufficient",
                    ReasonCode.VISION.SOURCE_IMAGE_STALE,
                    {
                        "rgb_queue_size": len(self.rgb_queue),
                        "depth_queue_size": len(self.depth_queue),
                        "camera_info_ready": self.latest_camera_info is not None,
                        "allow_rgb_only": self.allow_rgb_only,
                    },
                )
                return

            if self.resize_scale != 1.0:
                target_w = max(1, int(round(rgb.shape[1] * self.resize_scale)))
                target_h = max(1, int(round(rgb.shape[0] * self.resize_scale)))
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                camera_info_msg = resize_camera_info(camera_info_msg, self.resize_scale)

            encode_start = time.time()
            payload = make_frame_payload(
                rgb=rgb,
                depth=depth,
                camera_info={
                    "k": list(camera_info_msg.k),
                    "p": list(camera_info_msg.p),
                    "d": list(camera_info_msg.d),
                    "width": int(camera_info_msg.width),
                    "height": int(camera_info_msg.height),
                    "frame_id": self.camera_frame_override or camera_info_msg.header.frame_id,
                    "depth_encoding": depth_encoding,
                    "rgb_stamp": rgb_stamp,
                    "depth_stamp": depth_stamp,
                },
                t_world_camera=None,
                stamp=self.get_clock().now().nanoseconds * 1e-9,
            )
            encode_end = time.time()

            send_start = time.time()
            for target in self.dest_targets:
                send_udp_chunks(
                    sock=self.sock,
                    target=target,
                    payload=payload,
                    frame_id=self.frame_id,
                    max_payload=self.max_chunk_payload,
                )
            send_end = time.time()

            self.last_encode_latency_sec = max(0.0, encode_end - encode_start)
            self.last_send_latency_sec = max(0.0, send_end - send_start)

            target_count = len(self.dest_targets)
            total_sent_bytes = len(payload) * target_count
            self.frames_sent_since_summary += target_count
            self.bytes_sent_since_summary += total_sent_bytes

            self.dbg.emit_event(
                "INFO",
                "udp_frame_sent_hand",
                ReasonCode.VISION.UDP_FRAME_SENT,
                {
                    "frame_id": self.frame_id,
                    "frame_bytes": len(payload),
                    "target_count": target_count,
                    "total_sent_bytes": total_sent_bytes,
                    "targets": [f"{host}:{port}" for host, port in self.dest_targets],
                    "rgb_stamp": rgb_stamp,
                    "depth_stamp": depth_stamp,
                    "stamp_sync_diff_sec": abs(rgb_stamp - depth_stamp),
                    "encode_latency_sec": self.last_encode_latency_sec,
                    "send_latency_sec": self.last_send_latency_sec,
                },
            )

            if now - self.last_summary_time >= self.summary_period_sec:
                self.last_summary_time = now
                self.dbg.emit_periodic_summary(
                    event="udp_frame_send_summary_hand",
                    reason_code=ReasonCode.VISION.UDP_FRAME_SENT,
                    data={
                        "frames_sent_in_window": self.frames_sent_since_summary,
                        "total_bytes_in_window": self.bytes_sent_since_summary,
                        "effective_send_fps": self.frames_sent_since_summary / self.summary_period_sec,
                        "target_count": target_count,
                        "last_encode_latency_sec": self.last_encode_latency_sec,
                        "last_send_latency_sec": self.last_send_latency_sec,
                        "rgb_age_sec": now - rgb_stamp if rgb_stamp else None,
                        "depth_age_sec": now - depth_stamp if depth_stamp else None,
                    },
                    every_sec=self.summary_period_sec,
                    level="INFO",
                )

                self.dbg.write_snapshot(
                    {
                        "frame_id": self.frame_id,
                        "payload_bytes": len(payload),
                        "target_count": target_count,
                        "targets": [f"{host}:{port}" for host, port in self.dest_targets],
                        "total_chunks": math.ceil(len(payload) / self.max_chunk_payload),
                        "frames_sent_in_window": self.frames_sent_since_summary,
                        "total_bytes_in_window": self.bytes_sent_since_summary,
                        "rgb_stamp": rgb_stamp,
                        "depth_stamp": depth_stamp,
                    }
                )

                self.frames_sent_since_summary = 0
                self.bytes_sent_since_summary = 0

            self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF

        except Exception as exc:
            self.dbg.emit_event(
                "ERROR",
                "on_timer_send_exception_hand",
                ReasonCode.COMMON.EXCEPTION,
                {"error": repr(exc)},
            )
            self.get_logger().warn(f"on_timer_send hand error: {exc}")

        if now - self.last_heartbeat_time >= self.heartbeat_sec:
            self.last_heartbeat_time = now
            self.dbg.heartbeat(
                {
                    "queue_ready": len(self.rgb_queue) > 0 and len(self.depth_queue) > 0 and self.latest_camera_info is not None,
                    "frame_id": self.frame_id,
                    "rgb_queue_size": len(self.rgb_queue),
                    "depth_queue_size": len(self.depth_queue),
                    "target_count": len(self.dest_targets),
                }
            )

    def destroy_node(self):
        try:
            self.dbg.emit_event("INFO", "node_stopped", ReasonCode.COMMON.NODE_STOPPED, {})
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IsaacHandCameraUdpSenderDebug()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


## 실행예시
# ros2 run robotarm_sensors udp_hand_camera_sender_debug --ros-args \
#   -p dest_targets_csv:="127.0.0.1:9005"

# ros2 run robotarm_sensors udp_hand_camera_sender_debug --ros-args \
#   -p rgb_topic:=/hand_Camera_rgb \
#   -p dest_targets_csv:="127.0.0.1:9005"
