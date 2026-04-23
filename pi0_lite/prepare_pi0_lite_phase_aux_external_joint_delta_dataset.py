#!/usr/bin/env python3

import argparse
import json
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


def external_image_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    obs = sample.get("observation", {})
    value = obs.get("external_image") or sample.get("external_image")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_pi0_aux_external_joint_delta_sample(
    current_sample: Dict[str, Any],
    next_sample: Dict[str, Any],
    current_joint: List[float],
    next_joint: List[float],
    current_ee_pose: List[float],
    terminate_episode: float,
) -> Dict[str, Any]:
    joint_delta = [n - c for c, n in zip(current_joint, next_joint)]

    current_gripper = get_nested(current_sample, ["action", "gripper_width"], default=None)
    next_gripper = get_nested(next_sample, ["action", "gripper_width"], default=None)
    gripper_delta = [0.0]
    if current_gripper is not None and next_gripper is not None:
        try:
            current_gripper = float(current_gripper)
            next_gripper = float(next_gripper)
            # Width decrease should mean "close", so use a negative delta.
            gripper_delta = [next_gripper - current_gripper]
        except Exception:
            gripper_delta = [0.0]

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
            "joint_delta": joint_delta,
            "gripper_delta": gripper_delta,
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


def convert_jsonl_to_pi0_aux_external_joint_delta(
    input_path: str,
    output_path: str,
    expected_joint_dim: int = 7,
    min_group_size: int = 2,
) -> None:
    samples = load_jsonl(input_path)
    if not samples:
        raise ValueError(f"No samples found in {input_path}")

    assign_episode_keys(samples, input_path)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        episode_key = sample["_episode_key"]
        grouped.setdefault(episode_key, []).append(sample)

    output_items: List[Dict[str, Any]] = []
    skipped_groups = 0
    skipped_pairs = 0

    for episode_key, group_samples in grouped.items():
        group_samples.sort(key=get_sort_timestamp)
        if len(group_samples) < min_group_size:
            skipped_groups += 1
            continue

        frame_index = 0
        for idx in range(len(group_samples) - 1):
            current_sample = group_samples[idx]
            next_sample = group_samples[idx + 1]

            ok_cur_joint, current_joint = validate_joint(current_sample, expected_dim=expected_joint_dim)
            ok_next_joint, next_joint = validate_joint(next_sample, expected_dim=expected_joint_dim)
            ok_cur_pose, current_ee_pose = validate_ee_pose(current_sample)

            if not (ok_cur_joint and ok_next_joint and ok_cur_pose):
                skipped_pairs += 1
                continue

            terminate_episode = 1.0 if idx == len(group_samples) - 2 else 0.0
            item = build_pi0_aux_external_joint_delta_sample(
                current_sample=current_sample,
                next_sample=next_sample,
                current_joint=current_joint,
                next_joint=next_joint,
                current_ee_pose=current_ee_pose,
                terminate_episode=terminate_episode,
            )
            item["episode_key"] = episode_key
            item["frame_index"] = frame_index
            frame_index += 1
            output_items.append(item)

    save_jsonl(output_path, output_items)
    print(
        f"[INFO] saved {len(output_items)} samples to {output_path} "
        f"(groups={len(grouped)}, skipped_groups={skipped_groups}, skipped_pairs={skipped_pairs})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw OpenVLA external-camera JSONL to pi0_lite joint_delta format."
    )
    parser.add_argument("--input", required=True, help="Path to input samples_openvla.jsonl")
    parser.add_argument("--output", required=True, help="Path to output joint-delta JSONL")
    parser.add_argument("--expected_joint_dim", type=int, default=7)
    parser.add_argument("--min_group_size", type=int, default=2)
    args = parser.parse_args()

    convert_jsonl_to_pi0_aux_external_joint_delta(
        input_path=args.input,
        output_path=args.output,
        expected_joint_dim=args.expected_joint_dim,
        min_group_size=args.min_group_size,
    )


if __name__ == "__main__":
    main()


# Example:
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/prepare_pi0_lite_phase_aux_external_joint_delta_dataset.py \
#   --input /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/udp_image_merge_episode_001/samples_openvla.jsonl \
#   --output /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/udp_image_merge_episode_001/samples_pi0_aux_external_joint_delta.jsonl
