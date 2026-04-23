#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum
import json
import os
import re
import shutil
import socket
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


class ReasonCode:
    class COMMON:
        NODE_STARTED = "COMMON.NODE_STARTED"
        NODE_STOPPED = "COMMON.NODE_STOPPED"
        EXCEPTION = "COMMON.EXCEPTION"
        TIMEOUT = "COMMON.TIMEOUT"
        HEARTBEAT = "COMMON.HEARTBEAT"
        INVALID_PARAM = "COMMON.INVALID_PARAM"
        CONFIG_SNAPSHOT_WRITTEN = "COMMON.CONFIG_SNAPSHOT_WRITTEN"
        OLD_RUNS_DELETED = "COMMON.OLD_RUNS_DELETED"
        STORAGE_BUDGET_EXCEEDED = "COMMON.STORAGE_BUDGET_EXCEEDED"
        STORAGE_CLEANUP_FAILED = "COMMON.STORAGE_CLEANUP_FAILED"

    class INTERFACE:
        TOPIC_NAME_MISMATCH = "INTERFACE.TOPIC_NAME_MISMATCH"
        TOPIC_NOT_FOUND = "INTERFACE.TOPIC_NOT_FOUND"
        TOPIC_NO_PUBLISHER = "INTERFACE.TOPIC_NO_PUBLISHER"
        TOPIC_NO_SUBSCRIBER = "INTERFACE.TOPIC_NO_SUBSCRIBER"
        MESSAGE_TYPE_MISMATCH = "INTERFACE.MESSAGE_TYPE_MISMATCH"
        NAMESPACE_MISMATCH = "INTERFACE.NAMESPACE_MISMATCH"
        REMAP_MISSING = "INTERFACE.REMAP_MISSING"
        FRAME_ID_MISMATCH = "INTERFACE.FRAME_ID_MISMATCH"
        ROBOT_PREFIX_MISMATCH = "INTERFACE.ROBOT_PREFIX_MISMATCH"
        QOS_MISMATCH_SUSPECTED = "INTERFACE.QOS_MISMATCH_SUSPECTED"
        ACTION_NAME_MISMATCH = "INTERFACE.ACTION_NAME_MISMATCH"
        SERVICE_NAME_MISMATCH = "INTERFACE.SERVICE_NAME_MISMATCH"
        PARAM_DRIVEN_NAME_MISMATCH = "INTERFACE.PARAM_DRIVEN_NAME_MISMATCH"
        INTERFACE_OK = "INTERFACE.INTERFACE_OK"

    class VISION:
        IMAGE_RECEIVED = "VISION.IMAGE_RECEIVED"
        DETECTION_FAILED = "VISION.DETECTION_FAILED"
        TARGET_DETECTION_STALE = "VISION.TARGET_DETECTION_STALE"
        POSE_ESTIMATION_FAILED = "VISION.POSE_ESTIMATION_FAILED"
        FRAME_RX = "VISION.FRAME_RX"
        FRAME_ENCODED = "VISION.FRAME_ENCODED"
        UDP_FRAME_SENT = "VISION.UDP_FRAME_SENT"
        UDP_FRAME_REASSEMBLED = "VISION.UDP_FRAME_REASSEMBLED"
        SOURCE_IMAGE_STALE = "VISION.SOURCE_IMAGE_STALE"
        PIPELINE_LATENCY_HIGH = "VISION.PIPELINE_LATENCY_HIGH"
        DETECTION_OK = "VISION.DETECTION_OK"

    class MANIPULATION:
        GOAL_SET = "MANIPULATION.GOAL_SET"
        TRAJECTORY_PLANNED = "MANIPULATION.TRAJECTORY_PLANNED"
        TRAJECTORY_EXECUTION_STARTED = "MANIPULATION.TRAJECTORY_EXECUTION_STARTED"
        TRAJECTORY_EXECUTION_SUCCESS = "MANIPULATION.TRAJECTORY_EXECUTION_SUCCESS"
        TRAJECTORY_EXECUTION_FAILED = "MANIPULATION.TRAJECTORY_EXECUTION_FAILED"
        TRAJECTORY_PLANNING_FAILED = "MANIPULATION.TRAJECTORY_PLANNING_FAILED"
        TF_LOOKUP_FAILED = "MANIPULATION.TF_LOOKUP_FAILED"
        JOINT_STATE_MISSING = "MANIPULATION.JOINT_STATE_MISSING"
        JOINT_STATE_STALE = "MANIPULATION.JOINT_STATE_STALE"
        IK_SOLVER_FAILED = "MANIPULATION.IK_SOLVER_FAILED"
        TARGET_POSE_UNREACHABLE = "MANIPULATION.TARGET_POSE_UNREACHABLE"

    class GRIPPER:
        GRIPPER_COMMAND_SENT = "GRIPPER.GRIPPER_COMMAND_SENT"
        GRIPPER_STATE_OK = "GRIPPER.GRIPPER_STATE_OK"
        GRIPPER_OPEN_SUCCESS = "GRIPPER.GRIPPER_OPEN_SUCCESS"
        GRIPPER_CLOSE_SUCCESS = "GRIPPER.GRIPPER_CLOSE_SUCCESS"
        GRIPPER_STALLED = "GRIPPER.GRIPPER_STALLED"
        GRIPPER_NOT_RESPONDING = "GRIPPER.GRIPPER_NOT_RESPONDING"
        OBJECT_DROPPED = "GRIPPER.OBJECT_DROPPED"
        GRIPPER_FORCE_TOO_HIGH = "GRIPPER.GRIPPER_FORCE_TOO_HIGH"

    class CONTROL:
        CMD_PUBLISHED = "CONTROL.CMD_PUBLISHED"
        FEEDBACK_RECEIVED = "CONTROL.FEEDBACK_RECEIVED"
        FEEDBACK_STALE = "CONTROL.FEEDBACK_STALE"
        CONTROL_ERROR_HIGH = "CONTROL.CONTROL_ERROR_HIGH"
        CONTROL_OK = "CONTROL.CONTROL_OK"


@dataclass
class InterfaceSnapshot:
    node_name: str
    robot_name: str
    namespace: str = ""
    publishes: List[Dict[str, Any]] = field(default_factory=list)
    subscribes: List[Dict[str, Any]] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    frames: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfaceCheckResult:
    ok: bool
    reason_code: str
    summary: str
    expected_name: Optional[str] = None
    actual_name: Optional[str] = None
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunInfo:
    run_id: str
    path: str
    mtime: float
    size_bytes: int


DEFAULT_DEBUG_BASE_DIR = os.path.expanduser(
    "~/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs"
)


def now_sec() -> float:
    return time.time()


RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def canonicalize_name(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"/+", "/", str(name).strip())
    if not s:
        return ""
    if len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    if not s.startswith("/"):
        s = "/" + s
    return s


normalize_name = canonicalize_name


def split_name_tokens(name: str) -> List[str]:
    name = canonicalize_name(name)
    return [x for x in name.split("/") if x] if name else []


def extract_robot_prefix(name: str) -> Optional[str]:
    toks = split_name_tokens(name)
    return toks[0] if toks else None


def same_base_name(a: str, b: str) -> bool:
    ta = split_name_tokens(a)
    tb = split_name_tokens(b)
    return bool(ta and tb and ta[-1] == tb[-1])


def strip_robot_prefix(name: str, robot_prefixes: Optional[List[str]] = None) -> str:
    name = canonicalize_name(name)
    toks = split_name_tokens(name)
    if not toks:
        return name
    if robot_prefixes and toks[0] in robot_prefixes:
        return "/" + "/".join(toks[1:]) if len(toks) > 1 else "/"
    return name


def likely_namespace_mismatch(expected: str, actual: str) -> bool:
    e = canonicalize_name(expected)
    a = canonicalize_name(actual)
    return e != a and same_base_name(e, a)


def likely_robot_prefix_mismatch(
    expected: str,
    actual: str,
    robot_prefixes: Optional[List[str]] = None,
) -> bool:
    e = canonicalize_name(expected)
    a = canonicalize_name(actual)
    if e == a:
        return False
    e0 = extract_robot_prefix(e)
    a0 = extract_robot_prefix(a)
    e_stripped = strip_robot_prefix(e, robot_prefixes or [])
    a_stripped = strip_robot_prefix(a, robot_prefixes or [])
    return e_stripped == a_stripped and e0 != a0


def to_builtin_jsonable(obj: Any, _seen=None, _depth: int = 0, _max_depth: int = 8) -> Any:
    """
    JSON 직렬화 안전 변환.
    - 순환 참조 방지
    - Enum 안전 처리
    - __dict__ 무한 재귀 방지
    - 깊이 제한
    """
    if _seen is None:
        _seen = set()

    if _depth > _max_depth:
        return f"<max_depth:{type(obj).__name__}>"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, enum.Enum):
        return {
            "__enum__": f"{obj.__class__.__name__}.{obj.name}",
            "value": obj.value,
        }

    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"

    obj_id = id(obj)
    track_seen = isinstance(obj, (dict, list, tuple, set)) or hasattr(obj, "__dict__") or is_dataclass(obj)
    if track_seen:
        if obj_id in _seen:
            return f"<circular_ref:{type(obj).__name__}>"
        _seen.add(obj_id)

    if isinstance(obj, list):
        return [to_builtin_jsonable(x, _seen, _depth + 1, _max_depth) for x in obj]

    if isinstance(obj, tuple):
        return [to_builtin_jsonable(x, _seen, _depth + 1, _max_depth) for x in obj]

    if isinstance(obj, set):
        return [to_builtin_jsonable(x, _seen, _depth + 1, _max_depth) for x in obj]

    if isinstance(obj, dict):
        return {
            str(k): to_builtin_jsonable(v, _seen, _depth + 1, _max_depth)
            for k, v in obj.items()
        }

    if is_dataclass(obj):
        return to_builtin_jsonable(asdict(obj), _seen, _depth + 1, _max_depth)

    if hasattr(obj, "__dict__"):
        safe_dict = {}
        for k, v in vars(obj).items():
            if str(k).startswith("_"):
                continue
            safe_dict[str(k)] = to_builtin_jsonable(v, _seen, _depth + 1, _max_depth)
        return safe_dict

    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _safe_rmtree(path: str) -> bool:
    try:
        shutil.rmtree(path)
        return True
    except Exception:
        return False


class LDADebugger:
    def __init__(
        self,
        node_name: str,
        robot_name: str = "robotarm",
        namespace: str = "",
        base_dir: str = DEFAULT_DEBUG_BASE_DIR,
        run_id: Optional[str] = None,
        robot_prefixes: Optional[List[str]] = None,
        also_print: bool = True,
        retention_days: int = 30,
        min_keep_runs: int = 20,
        max_total_size_mb: int = 2048,
        cleanup_on_start: bool = True,
    ):
        self.node_name = node_name
        self.robot_name = robot_name
        self.namespace = canonicalize_name(namespace) if namespace else "/"
        self.base_dir = os.path.expanduser(base_dir)
        self.also_print = also_print
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.start_time = now_sec()
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.robot_prefixes = robot_prefixes or ["robotarm"]

        self.retention_days = max(0, int(retention_days))
        self.min_keep_runs = max(1, int(min_keep_runs))
        self.max_total_size_bytes = max(1, int(max_total_size_mb)) * 1024 * 1024

        self.event_throttle_sec: Dict[str, float] = {}

        os.makedirs(self.base_dir, exist_ok=True)
        if cleanup_on_start:
            self._cleanup_old_runs()

        self.run_dir = os.path.join(self.base_dir, self.run_id)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.snapshot_dir = os.path.join(self.run_dir, "snapshots")
        self.meta_dir = os.path.join(self.run_dir, "meta")
        self.interface_dir = os.path.join(self.run_dir, "interfaces")
        for d in [self.log_dir, self.snapshot_dir, self.meta_dir, self.interface_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, f"{self.node_name}.jsonl")
        self.snapshot_path = os.path.join(self.snapshot_dir, f"{self.node_name}.json")
        self.meta_path = os.path.join(self.meta_dir, f"{self.node_name}.json")
        self.interface_path = os.path.join(self.interface_dir, f"{self.node_name}.json")
        self.write_meta()

    def _list_run_infos(self) -> List[RunInfo]:
        runs: List[RunInfo] = []
        try:
            for name in os.listdir(self.base_dir):
                path = os.path.join(self.base_dir, name)
                if not os.path.isdir(path):
                    continue
                if not RUN_ID_PATTERN.match(name):
                    continue
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0.0
                runs.append(RunInfo(name, path, mtime, _dir_size_bytes(path)))
        except Exception:
            return []
        runs.sort(key=lambda x: x.mtime, reverse=True)
        return runs

    def _cleanup_old_runs(self):
        try:
            runs = self._list_run_infos()
            if not runs:
                return

            deleted: List[Dict[str, Any]] = []
            keep_set = set(r.run_id for r in runs[: self.min_keep_runs])
            now = now_sec()
            age_limit_sec = self.retention_days * 86400

            if self.retention_days > 0:
                for run in runs[self.min_keep_runs:]:
                    age_sec = max(0.0, now - run.mtime)
                    if age_sec > age_limit_sec and _safe_rmtree(run.path):
                        deleted.append(
                            {
                                "run_id": run.run_id,
                                "reason": "retention_days",
                                "size_bytes": run.size_bytes,
                                "age_sec": age_sec,
                            }
                        )

            runs = self._list_run_infos()
            total_size = sum(r.size_bytes for r in runs)

            if total_size > self.max_total_size_bytes:
                for run in reversed(runs):
                    if total_size <= self.max_total_size_bytes:
                        break
                    if run.run_id in keep_set:
                        continue
                    if _safe_rmtree(run.path):
                        total_size -= run.size_bytes
                        deleted.append(
                            {
                                "run_id": run.run_id,
                                "reason": "max_total_size_mb",
                                "size_bytes": run.size_bytes,
                            }
                        )

            if deleted:
                cleanup_path = os.path.join(self.base_dir, "cleanup_history.jsonl")
                record = {
                    "ts": now_sec(),
                    "event": "cleanup_old_runs",
                    "reason_code": ReasonCode.COMMON.OLD_RUNS_DELETED,
                    "deleted_runs": deleted,
                    "retention_days": self.retention_days,
                    "min_keep_runs": self.min_keep_runs,
                    "max_total_size_bytes": self.max_total_size_bytes,
                }
                with open(cleanup_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(to_builtin_jsonable(record), ensure_ascii=False) + "\n")
        except Exception:
            cleanup_path = os.path.join(self.base_dir, "cleanup_history.jsonl")
            record = {
                "ts": now_sec(),
                "event": "cleanup_old_runs_failed",
                "reason_code": ReasonCode.COMMON.STORAGE_CLEANUP_FAILED,
                "traceback": traceback.format_exc(),
            }
            try:
                with open(cleanup_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(to_builtin_jsonable(record), ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _append_jsonl(self, path: str, record: Dict[str, Any]):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_builtin_jsonable(record), ensure_ascii=False) + "\n")

    def _write_json(self, path: str, payload: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_builtin_jsonable(payload), f, ensure_ascii=False, indent=2)

    def write_meta(self):
        self._write_json(
            self.meta_path,
            {
                "node_name": self.node_name,
                "robot_name": self.robot_name,
                "namespace": self.namespace,
                "hostname": self.hostname,
                "pid": self.pid,
                "run_id": self.run_id,
                "start_time": self.start_time,
                "robot_prefixes": self.robot_prefixes,
                "base_dir": self.base_dir,
                "retention_days": self.retention_days,
                "min_keep_runs": self.min_keep_runs,
                "max_total_size_bytes": self.max_total_size_bytes,
            },
        )

    def emit_event(
        self,
        level: str,
        event: str,
        reason_code: str,
        data: Optional[Dict[str, Any]] = None,
        throttle_key: Optional[str] = None,
        throttle_sec: Optional[float] = None,
    ):
        if throttle_key and throttle_sec is not None and throttle_sec > 0:
            last_ts = self.event_throttle_sec.get(throttle_key, 0.0)
            if now_sec() - last_ts < throttle_sec:
                return
            self.event_throttle_sec[throttle_key] = now_sec()

        record = {
            "ts": now_sec(),
            "node": self.node_name,
            "robot": self.robot_name,
            "namespace": self.namespace,
            "run_id": self.run_id,
            "level": level,
            "event": event,
            "reason_code": reason_code,
            "data": data or {},
        }
        self._append_jsonl(self.log_path, record)
        if self.also_print:
            print(
                f"[{level}] [{self.node_name}] event={event} "
                f"reason={reason_code} data={record['data']}"
            )

    def emit_periodic_summary(
        self,
        event: str,
        reason_code: str,
        data: Optional[Dict[str, Any]] = None,
        every_sec: float = 1.0,
        level: str = "INFO",
    ):
        self.emit_event(
            level=level,
            event=event,
            reason_code=reason_code,
            data=data,
            throttle_key=f"summary:{event}",
            throttle_sec=every_sec,
        )

    def write_snapshot(self, snapshot: Any):
        self._write_json(
            self.snapshot_path,
            {
                "ts": now_sec(),
                "node": self.node_name,
                "robot": self.robot_name,
                "namespace": self.namespace,
                "run_id": self.run_id,
                "uptime_sec": now_sec() - self.start_time,
                "snapshot": snapshot,
            },
        )

    def emit_exception(
        self,
        where: str,
        exc: Exception,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        data = {
            "where": where,
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "traceback": traceback.format_exc(),
        }
        if extra_data:
            data.update(extra_data)
        self.emit_event("ERROR", "exception", ReasonCode.COMMON.EXCEPTION, data)

    def write_interface_snapshot(
        self,
        publishes: Optional[List[Dict[str, Any]]] = None,
        subscribes: Optional[List[Dict[str, Any]]] = None,
        services: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        frames: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        snap = InterfaceSnapshot(
            node_name=self.node_name,
            robot_name=self.robot_name,
            namespace=self.namespace,
            publishes=publishes_or_abs_names(publishes or []),
            subscribes=publishes_or_abs_names(subscribes or []),
            services=publishes_or_abs_names(services or []),
            actions=publishes_or_abs_names(actions or []),
            frames=frames or {},
            params=params or {},
            extra=extra or {},
        )

        self._write_json(
            self.interface_path,
            {
                "ts": now_sec(),
                "run_id": self.run_id,
                "interface_snapshot": asdict(snap),
            },
        )
        self.emit_event(
            "INFO",
            "interface_snapshot_written",
            ReasonCode.COMMON.CONFIG_SNAPSHOT_WRITTEN,
            {
                "interface_path": self.interface_path,
                "publish_count": len(snap.publishes),
                "subscribe_count": len(snap.subscribes),
                "service_count": len(snap.services),
                "action_count": len(snap.actions),
            },
        )

    def check_topic_name_match(
        self,
        expected_name: str,
        actual_name: str,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        expected = canonicalize_name(expected_name)
        actual = canonicalize_name(actual_name)
        details = dict(context or {})
        details.update(
            {
                "expected_name": expected,
                "actual_name": actual,
                "expected_type": expected_type,
                "actual_type": actual_type,
            }
        )

        if expected == actual:
            if expected_type and actual_type and expected_type != actual_type:
                return InterfaceCheckResult(
                    False,
                    ReasonCode.INTERFACE.MESSAGE_TYPE_MISMATCH,
                    "topic name is same but message type differs",
                    expected,
                    actual,
                    expected_type,
                    actual_type,
                    details,
                )
            return InterfaceCheckResult(
                True,
                ReasonCode.INTERFACE.INTERFACE_OK,
                "topic name matches",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )

        if likely_robot_prefix_mismatch(expected, actual, self.robot_prefixes):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.ROBOT_PREFIX_MISMATCH,
                "topic base is same but robot prefix differs",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        if likely_namespace_mismatch(expected, actual):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.NAMESPACE_MISMATCH,
                "topic base is same but namespace differs",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        if same_base_name(expected, actual):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.TOPIC_NAME_MISMATCH,
                "topic suffix matches partly but full name differs",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        return InterfaceCheckResult(
            False,
            ReasonCode.INTERFACE.TOPIC_NAME_MISMATCH,
            "topic name differs",
            expected,
            actual,
            expected_type,
            actual_type,
            details,
        )

    def check_service_name_match(
        self,
        expected_name: str,
        actual_name: str,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        expected = canonicalize_name(expected_name)
        actual = canonicalize_name(actual_name)
        details = dict(context or {})
        details.update(
            {
                "expected_name": expected,
                "actual_name": actual,
                "expected_type": expected_type,
                "actual_type": actual_type,
            }
        )
        if expected == actual:
            if expected_type and actual_type and expected_type != actual_type:
                return InterfaceCheckResult(
                    False,
                    ReasonCode.INTERFACE.MESSAGE_TYPE_MISMATCH,
                    "service name is same but service type differs",
                    expected,
                    actual,
                    expected_type,
                    actual_type,
                    details,
                )
            return InterfaceCheckResult(
                True,
                ReasonCode.INTERFACE.INTERFACE_OK,
                "service name matches",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        if likely_robot_prefix_mismatch(expected, actual, self.robot_prefixes):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.ROBOT_PREFIX_MISMATCH,
                "service base is same but robot prefix differs",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        if likely_namespace_mismatch(expected, actual):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.NAMESPACE_MISMATCH,
                "service base is same but namespace differs",
                expected,
                actual,
                expected_type,
                actual_type,
                details,
            )
        return InterfaceCheckResult(
            False,
            ReasonCode.INTERFACE.SERVICE_NAME_MISMATCH,
            "service name differs",
            expected,
            actual,
            expected_type,
            actual_type,
            details,
        )

    def check_action_name_match(
        self,
        expected_name: str,
        actual_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        expected = canonicalize_name(expected_name)
        actual = canonicalize_name(actual_name)
        details = dict(context or {})
        details.update({"expected_name": expected, "actual_name": actual})
        if expected == actual:
            return InterfaceCheckResult(
                True,
                ReasonCode.INTERFACE.INTERFACE_OK,
                "action name matches",
                expected,
                actual,
                details=details,
            )
        if likely_robot_prefix_mismatch(expected, actual, self.robot_prefixes):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.ROBOT_PREFIX_MISMATCH,
                "action base is same but robot prefix differs",
                expected,
                actual,
                details=details,
            )
        if likely_namespace_mismatch(expected, actual):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.NAMESPACE_MISMATCH,
                "action base is same but namespace differs",
                expected,
                actual,
                details=details,
            )
        return InterfaceCheckResult(
            False,
            ReasonCode.INTERFACE.ACTION_NAME_MISMATCH,
            "action name differs",
            expected,
            actual,
            details=details,
        )

    def check_frame_id_match(
        self,
        expected_frame: str,
        actual_frame: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        expected = canonicalize_name(expected_frame)
        actual = canonicalize_name(actual_frame)
        details = dict(context or {})
        details.update({"expected_frame": expected, "actual_frame": actual})
        if expected == actual:
            return InterfaceCheckResult(
                True,
                ReasonCode.INTERFACE.INTERFACE_OK,
                "frame id matches",
                expected,
                actual,
                details=details,
            )
        if likely_robot_prefix_mismatch(expected, actual, self.robot_prefixes):
            return InterfaceCheckResult(
                False,
                ReasonCode.INTERFACE.ROBOT_PREFIX_MISMATCH,
                "frame base is same but robot prefix differs",
                expected,
                actual,
                details=details,
            )
        return InterfaceCheckResult(
            False,
            ReasonCode.INTERFACE.FRAME_ID_MISMATCH,
            "frame id differs",
            expected,
            actual,
            details=details,
        )

    def emit_interface_check(self, result: InterfaceCheckResult, event: str = "interface_check"):
        self.emit_event(
            "INFO" if result.ok else "WARN",
            event,
            result.reason_code,
            {
                "summary": result.summary,
                "expected_name": result.expected_name,
                "actual_name": result.actual_name,
                "expected_type": result.expected_type,
                "actual_type": result.actual_type,
                **result.details,
            },
        )

    def diagnose_expected_vs_candidates(
        self,
        expected_name: str,
        candidate_names: List[str],
        expected_type: Optional[str] = None,
        candidate_type_map: Optional[Dict[str, str]] = None,
        interface_kind: str = "topic",
        context: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        candidate_type_map = candidate_type_map or {}
        context = context or {}
        expected_name = canonicalize_name(expected_name)
        candidate_names = [canonicalize_name(x) for x in candidate_names]
        candidate_type_map = {
            canonicalize_name(k): v for k, v in candidate_type_map.items()
        }

        if not candidate_names:
            reason = (
                ReasonCode.INTERFACE.TOPIC_NOT_FOUND
                if interface_kind == "topic"
                else ReasonCode.INTERFACE.SERVICE_NAME_MISMATCH
                if interface_kind == "service"
                else ReasonCode.INTERFACE.ACTION_NAME_MISMATCH
            )
            return InterfaceCheckResult(
                False,
                reason,
                f"no candidate {interface_kind}s found",
                expected_name=expected_name,
                expected_type=expected_type,
                details=context,
            )

        checks = []
        for cand in candidate_names:
            actual_type = candidate_type_map.get(cand)
            if interface_kind == "topic":
                checks.append(
                    self.check_topic_name_match(
                        expected_name, cand, expected_type, actual_type, context
                    )
                )
            elif interface_kind == "service":
                checks.append(
                    self.check_service_name_match(
                        expected_name, cand, expected_type, actual_type, context
                    )
                )
            else:
                checks.append(self.check_action_name_match(expected_name, cand, context))

        priority = {
            ReasonCode.INTERFACE.INTERFACE_OK: 0,
            ReasonCode.INTERFACE.MESSAGE_TYPE_MISMATCH: 1,
            ReasonCode.INTERFACE.ROBOT_PREFIX_MISMATCH: 2,
            ReasonCode.INTERFACE.NAMESPACE_MISMATCH: 3,
            ReasonCode.INTERFACE.TOPIC_NAME_MISMATCH: 4,
            ReasonCode.INTERFACE.SERVICE_NAME_MISMATCH: 4,
            ReasonCode.INTERFACE.ACTION_NAME_MISMATCH: 4,
        }
        return sorted(checks, key=lambda x: priority.get(x.reason_code, 999))[0]

    def diagnose_param_topic(
        self,
        param_name: str,
        param_value: str,
        observed_topics: List[str],
        expected_type: Optional[str] = None,
        observed_type_map: Optional[Dict[str, str]] = None,
        event: str = "param_topic_diagnosis",
        extra: Optional[Dict[str, Any]] = None,
    ) -> InterfaceCheckResult:
        result = self.diagnose_expected_vs_candidates(
            expected_name=param_value,
            candidate_names=observed_topics,
            expected_type=expected_type,
            candidate_type_map=observed_type_map,
            interface_kind="topic",
            context={"param_name": param_name, **(extra or {})},
        )
        if result.ok:
            self.emit_event(
                "INFO",
                event,
                ReasonCode.INTERFACE.INTERFACE_OK,
                {
                    "param_name": param_name,
                    "configured_topic": canonicalize_name(param_value),
                    "summary": result.summary,
                },
            )
        else:
            self.emit_event(
                "WARN",
                event,
                ReasonCode.INTERFACE.PARAM_DRIVEN_NAME_MISMATCH,
                {
                    "param_name": param_name,
                    "configured_topic": canonicalize_name(param_value),
                    "best_match_reason": result.reason_code,
                    "best_match_summary": result.summary,
                    "best_actual_name": result.actual_name,
                    "expected_type": expected_type,
                    "actual_type": result.actual_type,
                    **result.details,
                },
            )
        return result

    def heartbeat(self, data: Optional[Dict[str, Any]] = None):
        self.emit_event("DEBUG", "heartbeat", ReasonCode.COMMON.HEARTBEAT, data or {})

    def snapshot_ros_interfaces(self, ros_node, include_hidden: bool = True) -> Dict[str, Any]:
        topic_types = {
            canonicalize_name(name): list(types)
            for name, types in ros_node.get_topic_names_and_types(no_demangle=include_hidden)
        }
        services = {
            canonicalize_name(name): list(types)
            for name, types in ros_node.get_service_names_and_types()
        }
        return {"topics": topic_types, "services": services}

    def infer_qos_suspect(
        self,
        pub_count: int,
        sub_count: int,
        topic_name: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        if pub_count > 0 and sub_count > 0:
            self.emit_event(
                "WARN",
                "qos_mismatch_suspected",
                ReasonCode.INTERFACE.QOS_MISMATCH_SUSPECTED,
                {
                    "topic_name": canonicalize_name(topic_name),
                    "publisher_count": pub_count,
                    "subscriber_count": sub_count,
                    **(extra or {}),
                },
            )


def publishes_or_abs_names(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        copied = dict(item)
        if "name" in copied and copied["name"] is not None:
            copied["name"] = canonicalize_name(copied["name"])
        if "resolved_name" in copied and copied["resolved_name"] is not None:
            copied["resolved_name"] = canonicalize_name(copied["resolved_name"])
        out.append(copied)
    return out
