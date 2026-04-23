#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSON decode error at line {line_num}: {exc}")
    return items


def save_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_nested(data: Dict[str, Any], keys: List[str], default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_episode_base_key(sample: Dict[str, Any]) -> str:
    if "episode_id" in sample:
        return str(sample["episode_id"])

    instruction = str(sample.get("instruction", ""))
    image_path = str(sample.get("image", ""))
    parent_dir = str(Path(image_path).parent)
    return f"stream||{instruction}||{parent_dir}"


def infer_episode_suffix(input_path: str) -> str:
    parent_name = Path(input_path).resolve().parent.name
    digits = "".join(ch for ch in parent_name if ch.isdigit())
    if digits:
        return f"ep{int(digits):03d}"
    return "ep001"


def assign_episode_keys(samples: List[Dict[str, Any]], input_path: str) -> None:
    episode_suffix = infer_episode_suffix(input_path)

    for sample in samples:
        if "episode_id" in sample:
            episode_key = str(sample["episode_id"])
            sample["_episode_base_key"] = episode_key
            sample["_episode_key"] = episode_key
            continue

        base_key = infer_episode_base_key(sample)
        sample["_episode_base_key"] = base_key
        sample["_episode_key"] = f"{base_key}||{episode_suffix}"


def get_sort_timestamp(sample: Dict[str, Any]) -> float:
    for key in ["timestamp", "rgb_timestamp"]:
        if key in sample:
            try:
                return float(sample[key])
            except Exception:
                pass

    image_path = str(sample.get("image", ""))
    stem = Path(image_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return float(digits)
        except Exception:
            pass
    return 0.0


def validate_ee_pose(sample: Dict[str, Any]) -> Tuple[bool, Optional[List[float]]]:
    ee_pose = get_nested(sample, ["action", "ee_pose"], default=None)
    if ee_pose is None or not isinstance(ee_pose, list):
        return False, None
    try:
        ee_pose = [float(x) for x in ee_pose]
    except Exception:
        return False, None
    if len(ee_pose) != 7:
        return False, None
    return True, ee_pose


def validate_joint(sample: Dict[str, Any], expected_dim: int = 7) -> Tuple[bool, Optional[List[float]]]:
    joint = get_nested(sample, ["action", "arm_joint_position"], default=None)
    if joint is None or not isinstance(joint, list):
        return False, None
    try:
        joint = [float(x) for x in joint]
    except Exception:
        return False, None
    if len(joint) != expected_dim:
        return False, None
    return True, joint


def quat_normalize(q: List[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def quat_conjugate(q: List[float]) -> List[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_multiply(q1: List[float], q2: List[float]) -> List[float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quat_to_euler_xyz(q: List[float]) -> List[float]:
    x, y, z, w = quat_normalize(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def rotation_delta_from_quat(q_cur: List[float], q_next: List[float]) -> List[float]:
    q_delta = quat_multiply(quat_normalize(q_next), quat_conjugate(quat_normalize(q_cur)))
    q_delta = quat_normalize(q_delta)
    if q_delta[3] < 0:
        q_delta = [-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3]]
    return quat_to_euler_xyz(q_delta)


def external_image_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    obs = sample.get("observation", {})
    value = obs.get("external_image") or sample.get("external_image")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_pi0_aux_external_sample(
    current_sample: Dict[str, Any],
    next_sample: Dict[str, Any],
    current_joint: Optional[List[float]],
    current_ee_pose: List[float],
    next_ee_pose: List[float],
    terminate_episode: float,
) -> Dict[str, Any]:
    cur_pos = current_ee_pose[:3]
    nxt_pos = next_ee_pose[:3]
    world_vector = [n - c for c, n in zip(cur_pos, nxt_pos)]

    cur_quat = current_ee_pose[3:]
    nxt_quat = next_ee_pose[3:]
    rotation_delta = rotation_delta_from_quat(cur_quat, nxt_quat)

    current_gripper = get_nested(current_sample, ["action", "gripper_width"], default=None)
    next_gripper = get_nested(next_sample, ["action", "gripper_width"], default=None)
    gripper_closedness_action = [0.0]
    if current_gripper is not None and next_gripper is not None:
        try:
            current_gripper = float(current_gripper)
            next_gripper = float(next_gripper)
            gripper_closedness_action = [current_gripper - next_gripper]
        except Exception:
            gripper_closedness_action = [0.0]

    external_image = external_image_from_sample(current_sample)
    next_external_image = external_image_from_sample(next_sample)

    observation = {
        "image": current_sample.get("image"),
        "natural_language_instruction": current_sample.get("instruction"),
    }
    if external_image:
        observation["external_image"] = external_image

    output = {
        "image": current_sample.get("image"),
        "instruction": current_sample.get("instruction"),
        "observation": observation,
        "action": {
            "world_vector": world_vector,
            "rotation_delta": rotation_delta,
            "gripper_closedness_action": gripper_closedness_action,
            "terminate_episode": float(terminate_episode),
        },
        "state": {
            "arm_joint_position": current_joint,
            "gripper_width": current_gripper,
            "ee_pose": current_ee_pose,
        },
        "meta": {
            "phase": current_sample.get("phase"),
            "note": current_sample.get("note"),
            "timestamp": current_sample.get("timestamp"),
            "rgb_timestamp": current_sample.get("rgb_timestamp"),
            "external_image_source": current_sample.get("external_image_source"),
            "external_rgb_timestamp": current_sample.get("external_rgb_timestamp"),
            "camera_sync_delta_sec": current_sample.get("camera_sync_delta_sec"),
            "external_match_basis": current_sample.get("external_match_basis"),
            "external_match_delta_sec": current_sample.get("external_match_delta_sec"),
            "external_image": external_image,
            "next_image": next_sample.get("image"),
            "next_external_image": next_external_image,
            "next_timestamp": next_sample.get("timestamp"),
            "next_rgb_timestamp": next_sample.get("rgb_timestamp"),
            "next_external_rgb_timestamp": next_sample.get("external_rgb_timestamp"),
        },
    }

    if external_image:
        output["external_image"] = external_image

    return output


def convert_jsonl_to_pi0_aux_external(
    input_path: str,
    output_path: str,
    expected_joint_dim: int = 7,
    min_group_size: int = 2,
) -> None:
    samples = load_jsonl(input_path)
    assign_episode_keys(samples, input_path)
    print(f"[INFO] Loaded {len(samples)} samples")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    valid_count = 0
    invalid_count = 0

    for sample in samples:
        ok_ee, ee_pose = validate_ee_pose(sample)
        if not ok_ee:
            invalid_count += 1
            continue

        ok_joint, joint = validate_joint(sample, expected_dim=expected_joint_dim)
        if not ok_joint:
            joint = None

        sample["_ee_pose_cache"] = ee_pose
        sample["_joint_cache"] = joint
        key = str(sample.get("_episode_key", infer_episode_base_key(sample)))
        groups.setdefault(key, []).append(sample)
        valid_count += 1

    print(f"[INFO] Valid samples: {valid_count}")
    print(f"[INFO] Invalid samples skipped: {invalid_count}")
    print(f"[INFO] Number of groups: {len(groups)}")

    converted_items: List[Dict[str, Any]] = []
    skipped_small_groups = 0

    for group_key, group_samples in groups.items():
        if len(group_samples) < min_group_size:
            skipped_small_groups += 1
            continue

        group_samples.sort(key=get_sort_timestamp)
        for idx in range(len(group_samples) - 1):
            cur = group_samples[idx]
            nxt = group_samples[idx + 1]
            terminate_episode = 1.0 if idx == len(group_samples) - 2 else 0.0

            item = build_pi0_aux_external_sample(
                current_sample=cur,
                next_sample=nxt,
                current_joint=cur.get("_joint_cache"),
                current_ee_pose=cur["_ee_pose_cache"],
                next_ee_pose=nxt["_ee_pose_cache"],
                terminate_episode=terminate_episode,
            )
            item["episode_key"] = group_key
            item["frame_index"] = idx
            item["meta"]["episode_key"] = group_key
            item["meta"]["episode_base_key"] = cur.get("_episode_base_key")
            item["meta"]["step_index"] = idx
            converted_items.append(item)

    print(f"[INFO] Small groups skipped: {skipped_small_groups}")
    print(f"[INFO] Converted samples created: {len(converted_items)}")
    save_jsonl(output_path, converted_items)
    print(f"[INFO] Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw OpenVLA jsonl into pi0_lite_phase_aux_external training jsonl."
    )
    parser.add_argument("--input", type=str, required=True, help="Input raw samples_openvla.jsonl path")
    parser.add_argument("--output", type=str, required=True, help="Output pi0-lite delta jsonl path")
    parser.add_argument("--expected_joint_dim", type=int, default=7)
    parser.add_argument("--min_group_size", type=int, default=2)
    args = parser.parse_args()

    convert_jsonl_to_pi0_aux_external(
        input_path=args.input,
        output_path=args.output,
        expected_joint_dim=args.expected_joint_dim,
        min_group_size=args.min_group_size,
    )


if __name__ == "__main__":
    main()


## 실행예시
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/prepare_pi0_lite_phase_aux_external_dataset.py \
#   --input /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/udp_image_merge_episode_001/samples_openvla.jsonl \
#   --output /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/udp_image_merge_episode_001/samples_pi0_aux_external_delta.jsonl