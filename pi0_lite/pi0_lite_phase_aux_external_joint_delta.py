#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

from pi0_lite_phase_aux_external import build_parser as build_base_parser
from pi0_lite_phase_aux_external import predict_single, train


THIS_DIR = Path(__file__).resolve().parent


def resolve_robotarm_project_root() -> Path:
    env_root = os.environ.get("ROBOTARM_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "pi0_lite" / "pi0_lite.py").exists():
            return candidate

    for root in [THIS_DIR, Path.cwd().resolve()]:
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
DEFAULT_DATA_ROOT = PROJECT_ROOT / "debug_runs" / "chair_grasp_openvla"
DEFAULT_JSONL_PATH = str(DEFAULT_DATA_ROOT / "merged_samples_pi0_aux_external_joint_delta.jsonl")
DEFAULT_IMAGE_ROOT = str(DEFAULT_DATA_ROOT)
DEFAULT_EXTERNAL_IMAGE_ROOT = str(DEFAULT_DATA_ROOT)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "pi0_lite" / "pi0_lite_output" / "run_udp_aux_external_joint_delta")


def _set_action_default(parser: argparse.ArgumentParser, dest: str, *, default=None, required=None) -> None:
    for action in parser._actions:
        if action.dest != dest:
            continue
        if default is not None:
            action.default = default
        if required is not None:
            action.required = required
        return
    raise ValueError(f"Could not find parser action: {dest}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = (
        "pi0_lite_phase_aux_external_joint_delta: joint-delta training wrapper "
        "for merged external-image dataset"
    )

    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    train_parser = subparsers_action.choices["train"]

    _set_action_default(train_parser, "jsonl_path", default=DEFAULT_JSONL_PATH, required=False)
    _set_action_default(train_parser, "image_root", default=DEFAULT_IMAGE_ROOT, required=False)
    _set_action_default(
        train_parser,
        "external_image_root",
        default=DEFAULT_EXTERNAL_IMAGE_ROOT,
        required=False,
    )
    _set_action_default(train_parser, "output_dir", default=DEFAULT_OUTPUT_DIR, required=False)
    _set_action_default(train_parser, "action_format", default="joint_delta", required=False)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        state = json.loads(args.state_json)
        out = predict_single(
            checkpoint_path=args.checkpoint,
            image_path=args.image_path,
            external_image_path=args.external_image_path,
            instruction=args.instruction,
            state=state,
            steps=args.steps,
            device=args.device,
            image_size=args.image_size,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


## 사용 예시
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_phase_aux_external_joint_delta.py train
#
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_phase_aux_external_joint_delta.py train \
#   --epochs 300 \
#   --save_every_epochs 100
#
# 위 기본값은 아래를 사용합니다.
#   --jsonl_path /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_pi0_aux_external_joint_delta.jsonl
#   --image_root /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla
#   --external_image_root /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla
#   --output_dir /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_output/run_udp_aux_external_joint_delta
#   --action_format joint_delta
