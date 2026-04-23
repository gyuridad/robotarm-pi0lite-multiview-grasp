#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _normalize_rel_path(dir_name: str, path_value: str) -> str:
    if not path_value:
        return path_value
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(Path(dir_name) / path_obj).replace("\\", "/")


def _read_jsonl_rows(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSON parse failed, skip: {jsonl_path} line {line_num} | {exc}")
                continue
            row["_line_num"] = line_num
            rows.append(row)
    return rows


def _rewrite_row_for_episode(row: Dict, episode_dir_name: str) -> Dict:
    merged = dict(row)

    merged["episode_key"] = episode_dir_name
    if "frame_index" not in merged:
        merged["frame_index"] = int(merged.get("_line_num", 1) - 1)

    image_path = _normalize_rel_path(episode_dir_name, str(merged.get("image", "")))
    if image_path:
        merged["image"] = image_path

    external_image = _normalize_rel_path(episode_dir_name, str(merged.get("external_image", "")))
    if external_image:
        merged["external_image"] = external_image

    observation = merged.get("observation", {})
    if isinstance(observation, dict):
        obs = dict(observation)
        if "image" in obs:
            obs["image"] = _normalize_rel_path(episode_dir_name, str(obs.get("image", "")))
        if "external_image" in obs:
            obs["external_image"] = _normalize_rel_path(episode_dir_name, str(obs.get("external_image", "")))
        merged["observation"] = obs

    meta = merged.get("meta", {})
    if isinstance(meta, dict):
        meta_out = dict(meta)
        meta_out["episode_key"] = episode_dir_name
        if "external_image" in meta_out and meta_out["external_image"]:
            meta_out["external_image"] = _normalize_rel_path(episode_dir_name, str(meta_out["external_image"]))
        if "next_image" in meta_out and meta_out["next_image"]:
            meta_out["next_image"] = _normalize_rel_path(episode_dir_name, str(meta_out["next_image"]))
        if "next_external_image" in meta_out and meta_out["next_external_image"]:
            meta_out["next_external_image"] = _normalize_rel_path(
                episode_dir_name, str(meta_out["next_external_image"])
            )
        merged["meta"] = meta_out

    merged.pop("_line_num", None)
    return merged


def merge_pi0_lite_phase_aux_external_jsonl(
    root_dir: str,
    output_jsonl: str = "merged_samples_pi0_aux_external_delta.jsonl",
    input_jsonl_name: str = "samples_pi0_aux_external_delta.jsonl",
    episode_glob: str = "udp_image_merge_episode_*",
) -> None:
    root = Path(root_dir).resolve()
    output_path = root / output_jsonl

    episode_dirs = sorted([p for p in root.glob(episode_glob) if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories matched '{episode_glob}' under: {root}")

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for ep_dir in episode_dirs:
            ep_name = ep_dir.name
            jsonl_path = ep_dir / input_jsonl_name
            if not jsonl_path.exists():
                print(f"[WARN] Missing input file, skip: {jsonl_path}")
                continue

            print(f"[INFO] Reading: {jsonl_path}")
            rows = _read_jsonl_rows(jsonl_path)
            if not rows:
                print(f"[WARN] Empty file, skip: {jsonl_path}")
                continue

            for row in rows:
                merged_row = _rewrite_row_for_episode(row, ep_name)
                fout.write(json.dumps(merged_row, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"[DONE] Merge complete: {output_path}")
    print(f"[DONE] Total merged samples: {total_written}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge pi0_lite_phase_aux_external delta jsonl files from multiple episode directories."
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="merged_samples_pi0_aux_external_delta.jsonl")
    parser.add_argument("--input_jsonl_name", type=str, default="samples_pi0_aux_external_delta.jsonl")
    parser.add_argument("--episode_glob", type=str, default="udp_image_merge_episode_*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge_pi0_lite_phase_aux_external_jsonl(
        root_dir=args.root_dir,
        output_jsonl=args.output_jsonl,
        input_jsonl_name=args.input_jsonl_name,
        episode_glob=args.episode_glob,
    )


if __name__ == "__main__":
    main()


## 사용 예시
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/merge_pi0_lite_phase_aux_external_jsonl.py \
#   --root_dir /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_jsonl merged_samples_pi0_aux_external_delta.jsonl \
#   --episode_glob "udp_image_merge_episode_*"