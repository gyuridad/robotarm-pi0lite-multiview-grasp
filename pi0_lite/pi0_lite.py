
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import CLIPTextModel, CLIPTokenizer

try:
    from torchvision import models
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
    _HAS_TORCHVISION = True
except Exception:
    models = None
    ResNet18_Weights = None
    ResNet34_Weights = None
    ResNet50_Weights = None
    _HAS_TORCHVISION = False


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# 역할은 해당 모듈을 “고정(freeze)”해서 학습 중 가중치가 업데이트되지 않게 만드는 것
def freeze_module(module: nn.Module) -> None:
    module.eval()
        # 이 모듈을 평가 모드(eval mode) 로 바꿉니다.
        #     이렇게 하면 대표적으로
        #             Dropout 비활성화
        #             BatchNorm이 학습 통계 갱신 안 함
        #     상태가 됩니다.
    for p in module.parameters():
        p.requires_grad = False


# 이 함수는 모듈을
#     학습 가능하게 만들 수도 있고
#     학습 불가능하게 만들 수도 있는
# 일반형 함수입니다.
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag
            # 각 파라미터의 requires_grad 값을 flag로 바꿉니다.
            # 즉:
            #     flag=True 이면
            #             → gradient 계산함
            #             → 학습 가능 상태
            #     flag=False 이면
            #             → gradient 계산 안 함
            #             → 고정된 상태
            ### 이 코드에서 실제 의미
            #     업로드한 코드에서는 enable_flow_only_training() 안에서 먼저 전체 파라미터를 다 꺼두고,
            #     그다음 일부 모듈만 다시 켭니다.
            # 즉 흐름이 대략:
            #     전부 requires_grad=False
            #     action 관련 모듈만 requires_grad=True
            # 이렇게 되어 원하는 부분만 선택적으로 학습하게 만듭니다.


# 이 함수는 모델 안의 파라미터 개수를 세어서, 전체 개수와 현재 학습 가능한 개수를 반환하는 역할
def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# Text
"""
두 클래스의 관계를 먼저 한 줄로 말하면,

    CLIPTextProcessor 는 텍스트를 토큰으로 바꾸는 전처리 담당
    CLIPTextTokenEncoder 는 그 토큰을 받아 의미 있는 임베딩 벡터(토큰 특징) 로 바꾸는 인코더 담당

입니다. 
즉 앞단 전처리기 + 뒷단 신경망 인코더 관계입니다.
"""
# ============================================================

class CLIPTextProcessor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 48):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        out = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"].squeeze(0),
            "attention_mask": out["attention_mask"].squeeze(0),
        }


class CLIPTextTokenEncoder(nn.Module):
    """
    Keep CLIP text as token sequence.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        out_dim: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        self.clip_text = CLIPTextModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.clip_text.config.hidden_size, out_dim)
            # self.clip_text.config.hidden_size는?
            #     CLIP text model이 각 토큰마다 뽑아주는 원래 feature 차원 수입니다.
            #         예를 들어 CLIP text encoder가 토큰 하나를 표현할 때
            #             512차원 벡터일 수도 있고
            #             768차원 벡터일 수도 있고
            #         모델 종류에 따라 달라질 수 있습니다.
            #     즉 이 값은
            #         “CLIP이 원래 내보내는 토큰 벡터 크기” 입니다.
            # out_dim은?
            #     이 코드 안에서 최종적으로 맞추고 싶은 차원입니다.

        self.norm = nn.LayerNorm(out_dim)
        if freeze:
            freeze_module(self.clip_text)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.clip_text(input_ids=input_ids, attention_mask=attention_mask)
            # 실제 흐름 예시
            #     예를 들어 instruction이
            #             "의자를 집어"
            #     라고 하면 먼저 tokenizer가 input_ids, attention_mask를 만듭니다.

            #     그다음
            #             outputs = self.clip_text(input_ids=input_ids, attention_mask=attention_mask)
            #     를 하면 CLIP text encoder가 각 토큰에 대해 hidden vector를 계산합니다.

            #     예를 들어 결과가:
            #             outputs.last_hidden_state.shape = (1, 48, 512)
            #     라면 의미는
            #             배치 1개
            #             최대 길이 48토큰
            #             각 토큰마다 512차원 특징 벡터
            #     입니다.

        hidden = self.norm(self.proj(outputs.last_hidden_state))    # hidden.shape = (1, 48, 256)
        return hidden, attention_mask.bool()


# ============================================================
# Dataset
# ============================================================

def _flatten_float_list(x) -> List[float]:
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            if isinstance(v, list):
                out.extend(_flatten_float_list(v))
            else:
                out.append(float(v))
        return out
    return [float(x)]


# 역할은 json 한 줄(row) 안에서 로봇의 현재 상태(state) 정보를 꺼내서, 
# 모델이 바로 사용할 수 있는 1차원 숫자 벡터로 만드는 것
def build_state_vector(row: Dict) -> List[float]:
        # row 구조 예시:
        #         row = {
        #             "observation": { ... },   # 이게 obs
        #             "state": { ... },
        #             "action": { ... },
        #             "episode_key": ...,
        #             "frame_index": ...
        #         }

    state = row.get("state", {})
    obs = row.get("observation", {})
        # obs 구조 예시:
        #         obs = {
        #             "image": ...,
        #             "natural_language_instruction": ...,
        #             # 또는
        #             "instruction": ...,

        #             # 경우에 따라 fallback state
        #             "state": ...,
        #             "EEF_state": ...,
        #             "gripper_state": ...
        #         }

    state_vec = []
    if "arm_joint_position" in state:
        state_vec.extend(_flatten_float_list(state["arm_joint_position"]))
    if "gripper_width" in state:
        state_vec.extend(_flatten_float_list(state["gripper_width"]))
    if "ee_pose" in state:
        state_vec.extend(_flatten_float_list(state["ee_pose"]))

    if not state_vec and isinstance(obs, dict):
        for k in ["state", "EEF_state", "gripper_state"]:
            if k in obs:
                state_vec.extend(_flatten_float_list(obs[k]))

    return state_vec


# 역할은 json의 action 딕셔너리에서 실제 학습용 action 숫자 벡터를 만들어내는 것
def build_action_vector(action: Dict, action_format: str = "auto") -> Tuple[List[float], str]:
    resolved_format = action_format
    if resolved_format == "auto":
        if any(k in action for k in ["world_vector", "rotation_delta"]):
            resolved_format = "cartesian_delta"
        elif any(k in action for k in ["joint_delta", "arm_joint_delta"]):
            resolved_format = "joint_delta"
        else:
            raise ValueError(f"Could not infer action format from keys: {sorted(action.keys())}")

    action_vec = []
    if resolved_format == "cartesian_delta":   # end-effector를 공간상에서 얼마나 움직일지 표현하는 형식
        if "world_vector" in action:
            action_vec.extend(_flatten_float_list(action["world_vector"]))
        if "rotation_delta" in action:
            action_vec.extend(_flatten_float_list(action["rotation_delta"]))
        if "gripper_closedness_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_closedness_action"]))
    elif resolved_format == "joint_delta":     # 각 관절을 얼마나 바꿀지 표현하는 형식
        if "joint_delta" in action:
            action_vec.extend(_flatten_float_list(action["joint_delta"]))
        elif "arm_joint_delta" in action:
            action_vec.extend(_flatten_float_list(action["arm_joint_delta"]))
        else:
            raise ValueError("joint_delta action format requires 'joint_delta' or 'arm_joint_delta'")

        if "gripper_delta" in action:
            action_vec.extend(_flatten_float_list(action["gripper_delta"]))
        elif "gripper_width_delta" in action:
            action_vec.extend(_flatten_float_list(action["gripper_width_delta"]))
        elif "gripper_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_action"]))
        elif "gripper_closedness_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_closedness_action"]))
        else:
            raise ValueError("joint_delta action format requires gripper delta/action field")
    else:
        raise ValueError(f"Unsupported action format: {resolved_format}")

    terminate = action.get("terminate_episode", 0.0)
    action_vec.extend(_flatten_float_list(terminate))
    return action_vec, resolved_format


# build_action_vector함수와 반대 역할
# 역할은 모델이 예측한 1차원 action 벡터를 다시 사람이 보기 쉬운 action 딕셔너리 형태로 복원하는 것
def format_output_action(action_vec: Sequence[float], action_format: str) -> Dict:
    values = [float(v) for v in action_vec]
    if action_format == "joint_delta":
        if len(values) < 9:
            raise ValueError(f"joint_delta output expects at least 9 dims, got {len(values)}")
        return {
            "joint_delta": values[:7],
            "gripper_delta": [values[7]],
            "terminate_episode": float(values[8]),
        }

    # cartesian_delta 형식이면:
    if len(values) < 8:
        raise ValueError(f"cartesian_delta output expects at least 8 dims, got {len(values)}")
    return {
        "world_vector": values[:3],
        "rotation_delta": values[3:6],
        "gripper_closedness_action": [values[6]],
        "terminate_episode": float(values[7]),
    }


@dataclass
class RobotFrame:
    image_path: str
    instruction: str
    state: List[float]
    action: List[float]
    episode_key: str
    frame_index: int


class OpenVLADeltaDataset(Dataset):
    """
    역할은 jsonl에 저장된 로봇 데이터(이미지, 텍스트 지시문, 상태, 액션)를 읽어서,
    학습용 샘플 단위로 정리해 주는 데이터셋 클래스입니다.

    즉 PyTorch 모델이 바로 먹을 수 있도록 전처리 + 시퀀스 구성 + 정규화 + 텐서 변환까지 맡고 있습니다.

    쉽게 말하면 이 클래스는:

        raw jsonl 데이터를 읽고
        한 프레임씩 정리하고
        미래 horizon 길이의 action sequence를 만들고
        이미지/텍스트/state/action을 tensor로 바꿔서
        DataLoader가 꺼내 쓸 수 있게 해주는 역할

    입니다.
    """
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        text_processor: CLIPTextProcessor,
        image_size: int = 224,
        horizon: int = 4,
        action_dim: int = 0,
            # action_dim을 0으로 둔 이유는
            # 처음에는 action 차원을 비워 두고, 실제 데이터의 첫 action 벡터 길이를 보고 자동으로 결정하기 위해서
            # 이 코드의 action 형식은 두 가지일 수 있습니다.
            #     cartesian_delta
            #     joint_delta
            # 각 형식에 따라 action 길이가 달라질 수 있습니다.
        normalize: bool = True,
        action_format: str = "auto",
    ):
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.text_processor = text_processor
        self.image_size = image_size
        self.horizon = horizon
        self.action_dim = action_dim
        self.normalize = normalize
        self.action_format = action_format

        self.frames = self._load_frames(jsonl_path)
            # 역할은 jsonl 파일을 한 줄씩 읽어서, 각 줄을 모델이 쓰기 쉬운 RobotFrame 객체로 정리하는 것
        self.samples = self._build_sequence_indices(self.frames, horizon)
        self.state_dim = len(self.frames[0].state)
            # 앞에서 _load_frames()에서 각 row를 RobotFrame으로 바꿨죠.
            # 첫 번째 프레임의 state 벡터
            # state 벡터의 길이 = 차원 수
        self.action_stats = self._compute_action_stats() if normalize else None
        self.state_stats = self._compute_state_stats() if normalize else None

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def _load_frames(self, path: str) -> List[RobotFrame]:
        frames: List[RobotFrame] = []
        inferred_action_format: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("observation", {})
                action = row.get("action", {})

                image_path = self._resolve_image_path(obs.get("image", ""))
                instruction = obs.get("natural_language_instruction") or obs.get("instruction") or "do the task"
                state_vec = build_state_vector(row)
                action_vec, row_action_format = build_action_vector(action, self.action_format)

                if inferred_action_format is None:
                    inferred_action_format = row_action_format
                elif row_action_format != inferred_action_format:
                    raise ValueError(
                        f"Mixed action formats detected: {inferred_action_format} vs {row_action_format} at line {line_idx + 1}"
                    )

                if self.action_dim <= 0:
                    self.action_dim = len(action_vec)
                if len(action_vec) != self.action_dim:
                    raise ValueError(
                        f"Expected action_dim={self.action_dim}, got {len(action_vec)} at line {line_idx + 1}"
                    )
                if len(state_vec) == 0:
                    raise ValueError(f"No state found at line {line_idx + 1}")

                episode_key = str(row.get("episode_key", row.get("episode_id", "episode_0")))
                frame_index = int(row.get("frame_index", line_idx))
                frames.append(RobotFrame(image_path, instruction, state_vec, action_vec, episode_key, frame_index))

        if not frames:
            raise ValueError(f"No frames found in {path}")
        if inferred_action_format is not None:
            self.action_format = inferred_action_format
        return frames

    # 각 episode 안에서 “현재 프레임 1개 + 앞으로 horizon개 프레임 인덱스”를 만드는 함수
    def _build_sequence_indices(self, frames: List[RobotFrame], horizon: int) -> List[Tuple[int, List[int]]]:
        """
        예제 : episode가 1개이고 horizon=4

            프레임이 이렇게 있다고 해봅시다.

            frames 인덱스:     0   1   2   3   4
            frame_index:       0   1   2   3   4
            episode_key:       A   A   A   A   A

            즉 같은 episode A에 프레임 5개가 있습니다.
        """
        # 1단계: episode별로 묶기
        by_episode: Dict[str, List[int]] = {}
        for idx, fr in enumerate(frames):
            by_episode.setdefault(fr.episode_key, []).append(idx)
                # by_episode = {
                #     "A": [0, 1, 2, 3, 4]
                # }

        samples: List[Tuple[int, List[int]]] = []

        # 2단계: 각 시작점(start_i)마다 미래 horizon개 뽑기
        for _, inds in by_episode.items():
            inds = sorted(inds, key=lambda i: frames[i].frame_index)
            """
            inds는 episode별로 모은 프레임 인덱스인데,
            파일에서 읽은 순서가 항상 시간 순서라는 보장이 없습니다.

            정렬 기준은 리스트 값이 아니라 “frame_index”
            sorted(inds)           ❌ 그냥 숫자 정렬
            sorted(inds, key=...)  ✅ 실제 시간 기준 정렬
            """
            for start_i in range(len(inds)):
                future = inds[start_i:start_i + horizon]
                if len(future) < horizon:
                    future = future + [future[-1]] * (horizon - len(future))
                samples.append((inds[start_i], future))
                    # start_i = 0
                    #         future = [0, 1, 2, 3]
                    #         samples.append((0, [0, 1, 2, 3]))
                    # start_i = 1
                    #         future = [1, 2, 3, 4]
                    #         samples.append((1, [1, 2, 3, 4]))
        return samples

    def _compute_action_stats(self) -> Dict[str, torch.Tensor]:
        actions = np.stack([self.frames[idx].action for idx, _ in self.samples], axis=0)
        return {
            "mean": torch.tensor(actions.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(actions.std(axis=0) + 1e-6, dtype=torch.float32),
        }

    def _compute_state_stats(self) -> Dict[str, torch.Tensor]:
        """
        전체를 한 숫자로 평균내는 게 아니라,
        각 열(column)마다 평균을 구하는 것입니다.

        예를 들어

        첫 번째 state 성분 평균
        두 번째 state 성분 평균
        세 번째 state 성분 평균

        이렇게 구합니다.

        __getitem__()에서 실제 정규화에 사용됩니다.
        """
        states = np.stack([f.state for f in self.frames], axis=0)
        return {
            "mean": torch.tensor(states.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(states.std(axis=0) + 1e-6, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = x * 2.0 - 1.0
        """
        학습 데이터로더의 -1 ~ 1 정규화
                / 255.0 으로 먼저 픽셀값을 0 ~ 1로 바꾸고
                x = x * 2.0 - 1.0 으로 -1 ~ 1 범위로 다시 바꿉니다.
        """
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_idx, future_indices = self.samples[idx]
        frame = self.frames[cur_idx]
        image = self._load_image(frame.image_path)
        text_tokens = self.text_processor.encode(frame.instruction)
        input_ids = text_tokens["input_ids"].long()
        attention_mask = text_tokens["attention_mask"].long()
        state = torch.tensor(frame.state, dtype=torch.float32)
        actions = torch.tensor([self.frames[j].action for j in future_indices], dtype=torch.float32)

        if self.normalize:
            state = (state - self.state_stats["mean"]) / self.state_stats["std"]
            actions = (actions - self.action_stats["mean"]) / self.action_stats["std"]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "instruction": frame.instruction,
        }


# ============================================================
# Vision / state tokens
# ============================================================

class PatchVisionEncoder(nn.Module):
    """
    Lightweight vision token extractor.

    왜 이름이 PatchVisionEncoder인가

        입력 이미지를 한 번에 벡터 1개로 압축하지 않고,
        공간 위치별 feature를 여러 개 남겨서 token처럼 쓰기 때문입니다.

    즉 결과는 보통 이런 느낌입니다.

        이미지 전체 → 벡터 1개
        가 아니라
        이미지 여러 위치 → 토큰 여러 개

    그래서 Transformer가 이미지의 여러 부분을 따로 볼 수 있습니다.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # 예를 들어 입력이:
        #                 (B, 3, 224, 224)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
                # 채널: 3 → 64
                # 가로세로 크기: 대략 절반
                # 그다음 Conv들도 stride=2라서 계속 크기를 줄입니다.
                # 왜 7을 쓰냐 (핵심 이유)
                #     ✅ 이유 1: 초반에 넓은 영역을 보기 위해
                #     ✅ 이유 2: ResNet 설계에서 온 관례
                #             이 구조는 사실 여기서 온 것입니다:
                #             👉 ResNet-18 / ResNet-50
                #     ✅ 이유 3: stride=2와 궁합이 좋음
                #             👉 출력 크기가 깔끔하게 절반으로 줄어듭니다.
                #                     공식:

                #                         output= (W − K + 2P) / S + 1

                #                     예:
                #                         (224 − 7 + 2 × 3) / 2 + 1 = 112
                #             👉 딱 절반 (224 → 112)
                #             즉:
                #                 정보 많이 보면서
                #                 해상도 줄이고
                #                 계산량 줄이는
                #             효율적인 시작입니다.

            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv2d(256, out_dim, 3, stride=2, padding=1),
            nn.GroupNorm(16, out_dim),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        feat = feat.flatten(2).transpose(1, 2)
            # 역할은 CNN이 만든 feature map을 Transformer가 읽기 좋은 “토큰 시퀀스 형태”로 바꾸는 것
            # CNN을 통과한 뒤 feat는 보통 이런 형태입니다.
            #     (2, 256, 14, 14)
            #     의미는:
            #             이미지 2장
            #             채널 256개
            #             공간 위치 14×14
            # (B, C, H, W) 형태를 (B, N, C) 형태로 바꾸는 과정입니다.
            #     여기서
            #             B: 배치 크기
            #             C: 채널 수
            #             H, W: 공간 크기
            #             N = H × W: 토큰 개수
            #     입니다.
            # feat.flatten(2)는?
            #     이건 2번 차원부터 끝까지를 한 줄로 펴라는 뜻
            #     즉 (B, C, H, W) → (B, C, H×W)
            #         (2, 256, 14, 14)
            #         →
            #         (2, 256, 196)
        return self.norm(feat)


class ResNetTokenEncoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for pretrained vision encoder")

        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        """
        ResNet 단순화 구조:

        input
            → conv1
            → bn1
            → relu
            → maxpool
            → layer1
            → layer2
            → layer3
            → layer4
            → avgpool
            → fc
        """

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.proj = nn.Linear(feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return self.norm(x)


class StateTokenizer(nn.Module):
    """
    쉬운 숫자 예시
        예를 들어:

                state_dim = 15
                d_model = 256
                batch size B = 2

        입력:
                state.shape = (2, 15)

        state_proj 후
                (2, 15) → (2, 256)

        unsqueeze(1) 후
                (2, 256) → (2, 1, 256)

        summary_token.expand(...) 후
                summary.shape = (2, 1, 256)

        cat 후 최종 출력
                (2, 2, 256)

        즉 각 샘플마다 state 관련 토큰이 2개 생깁니다.


    왜 state token이 2개인가?
        최종 구조는:
                [state_summary_token, state_token]
        실제 state 임베딩 1개와 학습 가능한 summary token 1개를 만들어 반환하는 구조입니다.
        summary token은 학습 가능한 파라미터로, 모델이 state에서 중요한 정보를 요약해서 담도록 유도하는 역할을 합니다.

        👉 summary_token은 “state 정보를 압축해서 다른 토큰들이 참고하도록 만드는 중심 정보 허브” 역할을 합니다.
        👉 그리고 이 값은 Transformer 안에서 자연스럽게 사용됩니다 (따로 꺼내서 쓰는 게 아님)

    1️⃣ 어디에 쓰이냐? (핵심)

        summary_token은 따로 출력으로 뽑아서 쓰지 않습니다.

        대신 여기에서 쓰입니다:

                [image tokens | text tokens | state tokens(summary + st) | action tokens]

        이 전체가 Transformer에 들어갑니다.

        👉 즉 summary_token도 다른 토큰들과 똑같이 attention에 참여합니다.
    """
    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 예를 들어 state shape이 (2, 15)면 
        B = state.size(0)          # B=2입니다.
        st = self.state_proj(state).unsqueeze(1)         # (2, 15) → (2, 256) → (2, 1, 256)
        summary = self.summary_token.expand(B, -1, -1)   # (1, 1, 256) → (2, 1, 256)
        return torch.cat([summary, st], dim=1)           # (2, 1, 256) + (2, 1, 256) → (2, 2, 256)



# ============================================================
# Flow matching / transformer
# ============================================================

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        device = t.device
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        """
        핵심은:

            작은 주파수도 만들고
            큰 주파수도 만들고
            그 사이 값들도 여러 개 만들어서

        하나의 시간 t 를 여러 스케일에서 동시에 표현하려는 거예요.

        1. 먼저 전체 그림

            ContinuousTimeEmbedding의 sinusoidal()은 대충 이런 흐름입니다.

                    half = dim // 2
                    freqs = ...
                    args = t.unsqueeze(1) * freqs.unsqueeze(0)
                    emb = cat([sin(args), cos(args)])

            즉:

                    시간 t 1개
                    ↓
                    여러 개의 freqs 와 각각 곱함
                    ↓
                    [t×freq1, t×freq2, t×freq3, ...]
                    ↓
                    sin, cos 적용
                    ↓
                    시간 임베딩 벡터 완성

        2. 왜 freqs가 필요한가?

            만약 시간 t=0.3 을 그냥 숫자 하나로만 넣으면:

                    0.3

            이게 전부예요.

            그런데 모델은 시간에 따라 아주 다른 패턴을 배워야 하니까, 이렇게 바꾸는 겁니다:

                    [sin(0.3×큰주파수), sin(0.3×중간주파수), sin(0.3×작은주파수),
                    cos(0.3×큰주파수), cos(0.3×중간주파수), cos(0.3×작은주파수)]

            그러면 모델이:

                    미세한 시간 차이
                    중간 정도 시간 차이
                    전체적인 시간 위치

            를 같이 볼 수 있어요.

        3. 숫자 예시로 보기

            예를 들어 dim = 8 이라고 해볼게요.

            그러면

                    half = dim // 2 = 4

            즉 freqs 는 4개가 만들어집니다.

            코드:

                    torch.arange(0, half) = [0, 1, 2, 3]

            그리고 식은:

                    freqs[i] = exp(-log(10000) * i / (half-1))

            여기서는 half-1 = 3 이니까:

            i = 0
                    exp(-log(10000) * 0/3) = exp(0) = 1
            i = 1
                    exp(-log(10000) * 1/3)
                    ≈ exp(-3.07)
                    ≈ 0.0464
            i = 2
                    exp(-log(10000) * 2/3)
                    ≈ exp(-6.14)
                    ≈ 0.00215
            i = 3
                    exp(-log(10000) * 3/3)
                    = exp(-log(10000))
                    = 1/10000
                    = 0.0001

            그래서 최종적으로:

            freqs ≈ [1, 0.0464, 0.00215, 0.0001]

            입니다.

        4. 이 숫자들이 의미하는 것

            이건 이렇게 보면 쉬워요:

                    1        -> 빠르게 변하는 축
                    0.0464   -> 그보다 천천히 변하는 축
                    0.00215  -> 더 천천히 변하는 축
                    0.0001   -> 아주 천천히 변하는 축

        5. t를 곱하면 어떻게 되나

            예를 들어 t=0.5 라고 해볼게요.

                    args = t * freqs
                        = 0.5 * [1, 0.0464, 0.00215, 0.0001]
                        = [0.5, 0.0232, 0.001075, 0.00005]

            이제 여기에 sin/cos를 적용하면:

                    sin(args) ≈ [sin(0.5), sin(0.0232), sin(0.001075), sin(0.00005)]
                            ≈ [0.479, 0.0232, 0.001075, 0.00005]

                    cos(args) ≈ [cos(0.5), cos(0.0232), cos(0.001075), cos(0.00005)]
                            ≈ [0.878, 0.9997, 0.999999, 1.0]

            그래서 최종 임베딩은 대충:

                    [0.479, 0.0232, 0.001075, 0.00005, 0.878, 0.9997, 0.999999, 1.0]

            이런 벡터가 됩니다.

        6. 그림처럼 표현하면

            시간 t 하나를 여러 주파수 축에 투영하는 느낌입니다.

                    t = 0.5

                    freqs:     [1,      0.0464,   0.00215,   0.0001]
                                │         │          │          │
                                └──곱함────┴────곱함───┴────곱함───┴──곱함
                                            ↓
                    args:      [0.5,    0.0232,   0.001075,  0.00005]
                                            ↓
                    sin(args): [0.479,  0.0232,   0.001075,  0.00005]
                    cos(args): [0.878,  0.9997,   0.999999,  1.0]
                                            ↓
                    concat
                                            ↓
                    embedding vector
        """
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(t, self.dim)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        """
        dim=8일 때 숫자 예시

        예를 들어 dim = 8 이라고 해봅시다.

        그러면:

                torch.arange(0, 8, 2) = [0, 2, 4, 6]

        그리고

                -math.log(10000.0) / dim
                = -log(10000)/8
                ≈ -9.2103 / 8
                ≈ -1.1513

        이제 각각 곱하면:

        0일 때
                exp(0 * -1.1513) = exp(0) = 1
        2일 때
                exp(2 * -1.1513) = exp(-2.3026) ≈ 0.1
        4일 때
                exp(4 * -1.1513) = exp(-4.6052) ≈ 0.01
        6일 때
                exp(6 * -1.1513) = exp(-6.9078) ≈ 0.001

        그래서:

                div_term ≈ [1, 0.1, 0.01, 0.001]

        가 됩니다.

        4. 이 숫자들의 의미

        이 숫자들은 위치(position)에 곱해질 스케일이에요.

        예를 들어 position이 5면:

                position * div_term
                = 5 * [1, 0.1, 0.01, 0.001]
                = [5, 0.5, 0.05, 0.005]

        여기에 sin, cos를 적용하면:

                sin([5, 0.5, 0.05, 0.005])
                cos([5, 0.5, 0.05, 0.005])
        """
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
            # pe라는 positional encoding 텐서를 모델 안에 등록해 두는 것
            # (max_len, dim)
            # →
            # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# 조건(cond)에 따라 LayerNorm 결과를 조절하는데, 처음 학습 시작 때는 거의 0 영향으로 시작하는 구조
class AdaLNZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        """
        보통 LayerNorm은:

                y = gamma * norm(x) + beta

        처럼 내부적으로 학습 가능한 gamma, beta가 있습니다.

        그런데 여기서는:

                elementwise_affine=False

        라서 기본 gamma/beta를 안 둡니다.

        왜냐하면 gamma/beta 역할을 외부 조건 cond가 대신 만들게 하려는 것이기 때문입니다.

        즉:

                일반 LayerNorm: 내부 고정 파라미터로 조절
                AdaLNZero: cond가 scale/shift를 만들어서 조절

        입니다.
        """
        self.to_scale_shift_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 3),
        )
        nn.init.zeros_(self.to_scale_shift_gate[-1].weight)
        nn.init.zeros_(self.to_scale_shift_gate[-1].bias)
        """
        이렇게 마지막 Linear를 0으로 초기화하면, 처음에는:

                scale ≈ 0
                shift ≈ 0
                gate ≈ 0

        가 됩니다.

        그러면 forward에서:

                y = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        는 처음엔 거의

                y ≈ self.norm(x)

        가 되고, gate도 거의 0이라서 attention/MLP residual이 처음엔 거의 꺼진 상태가 됩니다.

        이게 왜 좋냐면:

                학습 초반에 블록이 갑자기 큰 출력을 내지 않아서 더 안정적으로 시작할 수 있기 때문

        입니다.
        """

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.to_scale_shift_gate(cond).chunk(3, dim=-1)
        """
        1. to_scale_shift_gate의 역할?
            즉 cond의 정보로:

                정규화 결과를 얼마나 키울지 (scale)
                얼마나 옮길지 (shift)
                attention/MLP 출력을 얼마나 통과시킬지 (gate)

            를 만듭니다.

            쉬운 숫자 예시

            예를 들어:

            dim = 4
            토큰 하나의 x = [1, 2, 3, 4]
            LayerNorm 결과가 대충 [-1.3, -0.4, 0.4, 1.3] 라고 해봅시다

            그리고 cond에서 나온 값이:

                    scale = [0.5, 0.0, -0.5, 1.0]
                    shift = [0.1, 0.2, 0.3, 0.4]
                    gate  = [0.0, 0.0, 0.0, 0.0]   # 학습 초반 가정

            라고 하면,

                    y = norm(x) * (1 + scale) + shift

            이므로:

                    1 + scale = [1.5, 1.0, 0.5, 2.0]

            따라서 각 차원은:

                    [-1.3, -0.4, 0.4, 1.3]
                    *
                    [ 1.5,  1.0, 0.5, 2.0]
                    +
                    [ 0.1,  0.2, 0.3, 0.4]

            대충

                    [-1.85, -0.2, 0.5, 3.0]

            처럼 바뀔 수 있어요.

            즉 cond가 같은 x라도 다른 방식으로 변형하게 됩니다.

        2. gate는 뭘 하냐?

            TransformerBlock에서 gate는 이렇게 쓰여요.

                    x = x + gate * self.dropout(attn_out)
                    ...
                    x = x + gate * self.dropout(self.mlp(h))

            즉 gate는 attention/MLP 출력에 곱해집니다.

                    gate가 0에 가까우면: 거의 반영 안 함
                    gate가 1이면: 그대로 반영
                    gate가 더 크면: 더 강하게 반영

            즉 gate는 일종의 조건부 볼륨 조절기예요.

        3. 이 코드 전체에서 cond는 무엇인가?

            Pi0LitePolicy에서 cond는 이렇게 만들어집니다.

                    cond = global_cond + self.time_embed(t)
                    for blk in self.blocks:
                        seq = blk(seq, cond, key_padding_mask=key_padding_mask)

            즉 cond 안에는:

                    이미지/텍스트/상태를 요약한 global_cond
                    현재 시간 t를 임베딩한 time_embed(t)

            가 들어 있습니다.

            그래서 AdaLNZero는 사실상:

                    현재 문맥과 현재 diffusion/flow 시간에 따라 Transformer 블록을 다르게 작동시키는 장치

            입니다.

            비유로 이해하기

            일반 Transformer 블록은 매번 같은 기계라고 볼 수 있어요.

            그런데 AdaLNZero가 붙으면, 블록 앞에 상황별 조정 장치가 붙습니다.

            예:

                    지금 초기 시간 t → noise가 많음 → 조심스럽게 처리
                    지금 후반 시간 t → action에 가까움 → 더 강하게 수정
                    현재 이미지/텍스트/상태 문맥이 다름 → attention 방향 달라짐

            즉 AdaLNZero는 블록을 상황 맞춤형으로 튜닝해 주는 역할입니다.
        """
        y = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return y, gate.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ada_attn = AdaLNZero(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)  # (batch, seq, dim) 형태로 입력받도록 batch_first=True
        self.ada_mlp = AdaLNZero(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, gate = self.ada_attn(x, cond)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            # 여기서 h, h, h 는 순서대로 Query, Key, Value 입니다.

        x = x + gate * self.dropout(attn_out)
            # 기존 정보(x)에 새로운 정보(attn_out)를 얼마나 섞을지 gate가 결정한다

        h, gate = self.ada_mlp(x, cond)
        x = x + gate * self.dropout(self.mlp(h))
        """
        1. x는 뭐냐?

            여기서 x는 현재까지 쌓인 토큰 상태입니다.

            TransformerBlock 안에서 흐름은 이렇습니다:

                    처음 x = 입력 토큰들 (image + text + state + action)
                    ↓
                    attention 통과
                    ↓
                    MLP 통과
                    ↓
                    업데이트된 x

            즉 x는:

                    “지금까지 모델이 이해하고 있는 상태”

            라고 보면 됩니다.

        2. attn_out은 뭐냐?

                    attn_out = self.attn(...)

            이건:

                    다른 토큰들을 참고해서 새롭게 계산된 정보

            입니다.

            예를 들어:

                    어떤 action 토큰이
                    이미지 토큰을 참고해서
                    “아 이 방향으로 움직여야겠네” 하고 얻은 정보

        3. 핵심: 왜 x + something 형태인가?

            Transformer는 기본적으로 이런 구조입니다:

                    새 상태 = 기존 상태 + 새 정보

            즉:

                    x_new = x_old + 변화량

            그래서 attn_out은 변화량(delta) 입니다.

        4. gate가 하는 일 (진짜 핵심)

            이제 여기:

                    x = x + gate * attn_out

            이걸 이해하면 끝입니다.

            (1) gate = 0일 때
                    x = x + 0 * attn_out
                      = x

                👉 아무 변화 없음

                즉:

                    “지금은 attention 결과를 믿지 않겠다”

            (2) gate = 1일 때
                    x = x + 1 * attn_out
                      = x + attn_out

                👉 기본 Transformer처럼 정상 반영

            (3) gate = 2일 때
                    x = x + 2 * attn_out

                👉 더 강하게 반영

            (4) gate = -1일 때 (이론적으로 가능)
                    x = x - attn_out

                👉 반대로 밀어버림

        gate는 어떻게 학습되나?
            모든 입력(cond, t 포함)에 대해 gate = 0으로 시작
            학습이 시작되면:
                    loss → backprop → weight 업데이트
            이제:
                    to_scale_shift_gate(cond) ≠ 0
            그래서:
                    gate = 어떤 값 (t에 따라 다르게)
            cond 안에 t가 들어가니까:
                👉 네트워크는 학습하면서 이런 걸 배운다:
                    t 작을 때 → gate 작게
                    t 클 때 → gate 크게
        """
        return x


class FlowMatchingScheduler:
    """
    Rectified-flow style linear path:
        x_t = (1 - t) * x0 + t * x1
        target velocity v* = x1 - x0

    “정답 action까지 가는 길을 만들어 주는 안내자”

    학습할 때는 중간 상태 xt를 만들어주고,
    추론할 때는 noise에서 시작해서 action으로 조금씩 이동시키는 역할을 합니다.

    함수 각 역할은:

        sample_t
            → 시간 t 를 랜덤으로 뽑음
        sample_bridge
            → noise와 정답 action 사이의 중간 상태 xt 를 만듦
        euler_integrate
            → 추론 때 noise에서 시작해서 여러 step에 걸쳐 action으로 이동

    모델이 배워야 하는 건:

        “현재 중간 상태 xt 에서 어느 방향으로 움직이면 정답 action 쪽으로 가는가?”

    입니다.

    그래서 FlowMatchingScheduler가

        학습용 중간 샘플도 만들고
        추론용 이동 과정도 관리합니다.

    [학습]
    정답 action x1 준비
    ↓
    랜덤 시간 t 뽑기
    ↓
    noise와 x1을 섞어서 x_t 만들기
    ↓
    모델이 x_t 보고 "어느 방향으로 가야 하나?" 예측
    ↓
    정답 방향(target_v)과 비교해서 학습

    [추론]
    랜덤 noise에서 시작
    ↓
    t=0 → 1로 조금씩 이동
    ↓
    매 step마다 모델이 방향 v 예측
    ↓
    x = x + dt * v
    ↓
    최종적으로 action 생성
    """
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        이건 배치마다 랜덤 시간 t 를 뽑는 함수예요.

        예를 들어 배치 크기가 3이면:

                t = [0.12, 0.48, 0.83]

        이런 식으로 나올 수 있어요.

        의미:

                첫 번째 샘플은 초반 구간
                두 번째 샘플은 중간 구간
                세 번째 샘플은 후반 구간

        즉 모델이 길의 모든 위치를 골고루 배우게 합니다.
        """
        return torch.rand(batch_size, device=device) * (1.0 - 2 * self.eps) + self.eps

    def sample_bridge(self, x1: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        6. sample_bridge() 역할

            이게 학습의 핵심이에요.

                    x_t = (1.0 - t_view) * noise + t_view * x1
                    target_v = x1 - noise

            여기서:

                    x1 = 정답 action
                    noise = 랜덤 시작점
                    x_t = 둘 사이의 중간 상태
                    target_v = 정답 방향

            입니다.

        6-1. 그림으로 보면

                    noise ------------------------- action
                    x0                              x1
                        \---- 현재 위치 x_t ----/

            t가:

                    0에 가까우면 x_t는 noise 쪽
                    1에 가까우면 x_t는 action 쪽

            입니다.

        6-2. 간단 숫자 예시

            action이 1차원이라고 단순화해서:

                    noise = 2
                    x1    = 10
                    t     = 0.25

            그러면:

                    x_t = (1 - 0.25)*2 + 0.25*10
                        = 0.75*2 + 0.25*10
                        = 1.5 + 2.5
                        = 4

            즉 현재 위치는 4예요.

            그리고 정답 방향은:

                    target_v = x1 - noise = 10 - 2 = 8

            이 뜻은:

                    noise에서 action으로 가려면 +8 방향으로 가야 한다

            입니다.

            중요한 점은 이 코드에서는 직선 경로(linear path) 를 쓰기 때문에 방향이 일정해요. 클래스 설명에도 그렇게 적혀 있어요.

                    x_t = (1 - t) * x0 + t * x1
                    target velocity v* = x1 - x0

        7. 학습에서 실제로 어떻게 쓰이나

            훈련 루프에서 이렇게 연결됩니다.

                    t = flow.sample_t(actions.size(0), device=device)
                    x_t, target_v, _ = flow.sample_bridge(actions, t)
                    pred_v = model(x_t, t, image, input_ids, attention_mask, state)

                    loss = F.mse_loss(pred_v, target_v)

            즉 순서는:

                    1. 정답 action actions 준비
                    2. 랜덤 시간 t 뽑기
                    3. x_t 만들기
                    4. 모델에게 "x_t에서 어느 방향으로 가야 하니?" 물어보기
                    5. 모델 예측 pred_v 와 정답 방향 target_v 비교

            입니다.
        """
        if noise is None:
            noise = torch.randn_like(x1)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * noise + t_view * x1
        target_v = x1 - noise
        return x_t, target_v, noise

    def euler_integrate(
        self,
        model: nn.Module,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        horizon: int,
        action_dim: int,
        num_steps: int = 32,
        device: Optional[torch.device] = None,
        generator=None,
    ) -> torch.Tensor:
        """
        1. euler_integrate() 역할
            이건 추론 때 사용됩니다.
            랜덤 noise에서 출발해서, 모델이 알려주는 방향으로 조금씩 이동하면서 최종 action을 만든다는 함수입니다.

                조금 이동
                또 조금 이동
                또 조금 이동

            이렇게 잘게 쪼개서 갑니다.

            그게 Euler 적분이에요.

            sample_actions()가 결국 euler_integrate()를 부릅니다.
        
        2. 추론 흐름 전체

            이미지 + 텍스트 + 상태 입력
            ↓
            초기 action 토큰은 랜덤 noise
            ↓
            모델이 현재 t에서의 방향 v 예측
            ↓
            x = x + dt*v
            ↓
            반복
            ↓
            최종 action trajectory 출력
        """
        device = device or image.device
        x = torch.randn((image.size(0), horizon, action_dim), device=device, generator=generator)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = torch.full((image.size(0),), float(ts[i].item()), device=device)
            dt = float((ts[i + 1] - ts[i]).item())
            v = model(x, t, image, input_ids, attention_mask, state)
                # 모델은 Pi0LitePolicy의 forward()입니다.
                # 반환값 v는 “현재 x를 어디 방향으로 얼마나 움직여야 하는지”를 나타내는 velocity(속도장) 예측값입니다.

            x = x + dt * v
        return x


# ============================================================
# Pi0-lite policy
# ============================================================

class Pi0LitePolicy(nn.Module):
    """
    이미지 + 텍스트 + 상태를 토큰으로 바꿔서 context를 만들고, 현재 action 상태 x_t도 action 토큰으로 넣은 뒤,
    Transformer가 이 둘을 함께 보고 “다음에 action을 어느 방향으로 수정해야 하는지(velocity)”를 예측하는 모델입니다.

    전체를 단계별로 아주 쉽게 정리하면 이렇습니다.

    1. 입력

        forward()에 들어가는 입력은 6개입니다.

            x_t: 현재 action trajectory의 중간 상태
            t: 현재 flow 시간
            image: 이미지
            input_ids, attention_mask: 텍스트 토큰
            state: 로봇 상태 벡터

        즉 이 모델은 문맥 정보(image/text/state) 와 현재 action 상태(x_t) 를 같이 봅니다.

    2. context 만들기

        build_context()에서 문맥을 만듭니다.

            vision_encoder(image) → 이미지 토큰
            text_encoder(input_ids, attention_mask) → 텍스트 토큰
            state_tokenizer(state) → 상태 토큰

        그리고 각 토큰에

            vision용 modality embedding
            text용 modality embedding
            state용 modality embedding

        을 더해서, 모델이 “이 토큰이 어떤 종류인지” 구분할 수 있게 합니다.
        그 다음 이 셋을 이어붙여서 ctx_tokens를 만듭니다.

        즉:

            context = [image tokens | text tokens | state tokens]

        이 구조입니다.

    3. global condition 만들기

        build_context() 안에서는 토큰 전체를 그냥 쓰는 것뿐 아니라, 요약 벡터도 따로 만듭니다.

            이미지 토큰 평균
            텍스트 토큰 평균(mask 반영)
            상태 토큰 평균

        이 세 요약을 이어 붙인 뒤 global_cond MLP에 넣어 문맥 전체 요약 벡터를 만듭니다.

        이건 뒤에서 Transformer 블록을 조절하는 조건값으로 쓰입니다.

    4. action 토큰 만들기

        현재 action 상태 x_t는 그대로 Transformer에 넣지 않고,

            action_in(x_t)로 d_model 차원으로 변환하고
            action_time_tokens를 더하고
            action용 modality embedding도 더합니다.

        즉:

            action tokens = 현재 action trajectory를 token 형태로 바꾼 것

        입니다.

        여기서 핵심은 이 모델이 action을 숫자 벡터 하나로 바로 보는 게 아니라,
        horizon 길이의 action sequence를 토큰 시퀀스로 본다는 점입니다.
        클래스 설명에도 action trajectory as action tokens라고 적혀 있습니다.

    5. 전체 시퀀스 만들기

        이제 최종 입력 시퀀스는:

            [context tokens | action tokens]

        입니다.
        여기에 positional encoding까지 더해서 Transformer가 순서를 알 수 있게 합니다.

        즉 Transformer 입장에서는

            앞쪽엔 상황 설명 토큰들
            뒤쪽엔 지금 고쳐야 할 action 토큰들

        이 같이 들어오는 구조입니다.

    6. 시간 정보까지 합쳐서 Transformer 블록 통과

        cond = global_cond + self.time_embed(t) 로 조건 벡터를 만듭니다.

        즉 각 블록은

            현재 장면/명령/상태 요약
            현재 flow 시간 t

        을 함께 조건으로 받아 동작합니다.

        그리고 여러 개의 TransformerBlock이 이 시퀀스를 반복적으로 업데이트합니다.
        이 블록 안에서는 AdaLNZero가 cond를 사용해 attention/MLP의 반영 강도를 조절합니다.
        그래서 같은 토큰이라도 현재 시간과 문맥에 따라 다르게 처리됩니다.

    7. action 부분만 잘라서 최종 출력

        Transformer를 다 통과한 뒤에는 전체 시퀀스 중 맨 뒤의 action 토큰 부분만 잘라냅니다.

            action_out = seq[:, -self.horizon:, :]
            self.out(action_out) 으로 action 차원으로 다시 변환

        즉 최종 출력은

            각 time step마다
            각 action 차원에 대해

        예측한 값이고, Flow Matching 기준으로는 이것이 현재 x_t에서 정답 action 쪽으로 가는 velocity v 입니다.

    8. 학습에서의 역할

        훈련 루프에서는

            actions = 정답 trajectory
            flow.sample_bridge() 로 x_t와 target_v 생성
            pred_v = model(x_t, t, image, input_ids, attention_mask, state)
            loss = mse(pred_v, target_v)

        이렇게 학습합니다.

        즉 Pi0LitePolicy는 정답 action 자체를 바로 맞추는 모델이라기보다,

        “현재 noisy한 action 상태 x_t를 보고, 정답 쪽으로 어떻게 움직여야 하는지”를 예측하는 모델입니다.

    9. 추론 때는 어떻게 쓰이나

        추론에서는 랜덤 noise에서 시작해서,

            v = model(x, t, image, input_ids, attention_mask, state)
            x = x + dt * v

        를 반복합니다.

        즉 Pi0LitePolicy는 추론 때도 계속 불리면서, 매 단계마다
        “지금 action을 어느 쪽으로 고쳐야 하는지”를 알려주는 역할을 합니다.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 4,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        vision_encoder_type: str = "pretrained",
        pretrained_backbone: str = "resnet18",
        pretrained_vision: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip_text: bool = False,
            # freeze_clip_text=False
            #     CLIP 텍스트 인코더도 같이 학습됨
            #     사용자의 데이터에 맞게 텍스트 표현이 바뀔 수 있음
            # freeze_clip_text=True
            #     CLIP 텍스트 인코더는 pretrained 상태로 고정
            #     그 뒤의 projection, transformer 쪽만 주로 학습됨
            # 입니다.
            # 쉽게 말하면:
            #     True = 이미 잘 학습된 CLIP 텍스트 이해 능력을 그대로 쓰겠다
            #     False = 내 데이터에 맞게 텍스트 인코더도 조금 더 적응시키겠다
            # 입니다.
            # 보통은 데이터가 적으면 True가 더 안정적일 수 있고, 데이터가 충분하고 특정 작업 문장이 특수하면 False가 도움이 될 수 있습니다.
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model

        if vision_encoder_type == "simple":
            self.vision_encoder = PatchVisionEncoder(out_dim=d_model)
        elif vision_encoder_type == "pretrained":
            self.vision_encoder = ResNetTokenEncoder(
                backbone_name=pretrained_backbone,
                out_dim=d_model,
                pretrained=pretrained_vision,
            )
        else:
            raise ValueError(f"Unsupported vision_encoder_type: {vision_encoder_type}")

        self.text_encoder = CLIPTextTokenEncoder(
            model_name=clip_model_name,
            out_dim=d_model,
            freeze=freeze_clip_text,
        )
        self.state_tokenizer = StateTokenizer(state_dim=state_dim, d_model=d_model)

        # action 토큰을 만드는 부분
        self.action_in = nn.Linear(action_dim, d_model)   # action 벡터를 Transformer용 임베딩 차원으로 바꾸는 층
        self.action_time_tokens = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
            # 이건 각 action step 위치마다 붙는 학습 가능한 시간/위치 토큰입니다.
            # 예를 들어 horizon=4면:
            #     1번째 action step용 벡터
            #     2번째 action step용 벡터
            #     3번째 action step용 벡터
            #     4번째 action step용 벡터
            # 를 따로 들고 있는 셈입니다.
            # 그래서 action token에 더해지면 모델이
            #     “이 토큰은 첫 번째 미래 action인지”
            #     “두 번째인지”
            #     “마지막인지”
            # 를 구분할 수 있습니다.
            # 즉 이건 action sequence 내부의 위치 정보에 가깝습니다.

        # vision/text/state/action이 서로 다른 종류의 토큰임을 표시하는 부분
        self.modality_embeddings = nn.ParameterDict({
            "vision": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "text": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "state": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "action": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
        })
        """
        modality_embeddings는 토큰 종류표 같은 역할입니다.

        코드에서 실제로:

                vis_tokens = vis_tokens + self.modality_embeddings["vision"]
                txt_tokens = txt_tokens + self.modality_embeddings["text"]
                state_tokens = state_tokens + self.modality_embeddings["state"]

        를 하고, action 쪽도

                action_tokens = action_tokens + self.modality_embeddings["action"]

        를 합니다.

        즉 같은 d_model=256 차원 안에 있어도, 모델이

                이건 이미지 토큰
                이건 텍스트 토큰
                이건 상태 토큰
                이건 액션 토큰

        임을 구분하게 해줍니다.
        비유하면 토큰마다 “출신 배지”를 붙이는 것입니다.

        👉 처음엔 랜덤이지만, loss를 통해 “이 토큰은 이런 역할을 해야 한다”는 방향으로 계속 업데이트되면서 의미를 갖게 됩니다.

        1️⃣ 지금 질문의 핵심

                “랜덤값인데 어떻게 ‘출신(vision/text/state/action)’을 알려주는 역할을 하냐?”

            ➡️ 답:
                모델이 직접 의미를 부여한다 (gradient로 학습됨)

        2️⃣ 실제로 무슨 일이 일어나냐 (중요)

            초기 상태
                    self.modality_embeddings["vision"] = 랜덤
                    self.modality_embeddings["text"] = 랜덤
                    ...

                👉 이때는 아무 의미 없음

            forward에서 일어나는 일
                    vis_tokens = vis_tokens + vision_embedding
                    txt_tokens = txt_tokens + text_embedding

                👉 같은 Transformer에 들어감

            Transformer 입장

                Transformer는 이렇게 봅니다:

                    "이 토큰들 패턴이 다르네?"
                    "이건 이미지에서 온 것 같고"
                    "이건 텍스트 같고"

                👉 왜냐하면:

                    이미지 토큰은 CNN에서 나온 특징
                    텍스트 토큰은 CLIP에서 나온 특징
                    state는 숫자 projection

                ➡️ 이미 분포가 다름

        3️⃣ 학습에서 핵심 메커니즘

            loss는 결국 이거죠:

                    pred_v vs target_v

            즉:

                👉 action을 잘 맞추면 loss ↓
                👉 못 맞추면 loss ↑

            여기서 modality embedding 역할

            예를 들어 이런 상황:

                action을 예측하려면 이미지 정보가 중요함
                텍스트는 덜 중요함

            그러면 학습 중에 gradient가 이렇게 흐릅니다:

                "vision 토큰 쪽 attention 더 키워!"
                "state 토큰도 좀 봐!"
                "text는 덜 봐!"

            그러면 embedding이 바뀜

            결과적으로:

                vision_embedding → attention에서 잘 보이게 바뀜
                text_embedding → 덜 영향 주게 바뀜
        """

        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        self.time_embed = ContinuousTimeEmbedding(d_model)

        self.global_cond = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        """
        global_cond는 이미지·텍스트·상태를 한 번 요약한 뒤, Transformer 블록들을 조건부로 조절하는 데 쓰입니다.

        흐름은 이렇습니다.

        먼저 build_context() 안에서

                vis_summary = vis_tokens.mean(dim=1)
                txt_summary = self._masked_mean(txt_tokens, txt_mask)
                state_summary = state_tokens.mean(dim=1)

        이렇게 세 modality의 요약 벡터를 만든 다음, 이 셋을 이어붙여 self.global_cond(...)에 넣습니다. 그래서 global_cond는 vision 요약 + text 요약 + state 요약을 압축한 문맥 벡터입니다.

        그 다음 forward()에서 이 값이 바로 쓰이는 곳은 여기입니다.

                cond = global_cond + self.time_embed(t)
                for blk in self.blocks:
                seq = blk(seq, cond, key_padding_mask=key_padding_mask)

        즉 global_cond는 혼자 직접 action을 출력하는 게 아니라, **현재 시간 정보 time_embed(t)와 합쳐져서 각 TransformerBlock의 조건 입력 cond**가 됩니다.

        그리고 이 cond는 블록 안의 AdaLNZero에서 사용됩니다. 여기서 cond로부터

                scale
                shift
                gate

        를 만들어서, attention/MLP 출력을 얼마나 반영할지 조절합니다.
        """

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, action_dim)

    def freeze_condition_encoders(self) -> None:
        freeze_module(self.vision_encoder)
        freeze_module(self.text_encoder)
        # freeze_module(self.state_tokenizer) <- state_tokenizer는 학습이 필요함 

    def enable_flow_only_training(self) -> None:
        """
        enable_flow_only_training()의 역할은 모델 전체를 다 학습하지 않고,
        flow 예측에 직접 필요한 부분만 선택적으로 학습 가능하게 바꾸는 것입니다.

        코드를 보면 흐름이 이렇게 되어 있습니다.

            1. 먼저 **모든 파라미터를 전부 requires_grad=False**로 꺼 둡니다.
            2. 그다음 freeze_condition_encoders()를 호출해서
                    vision_encoder
                    text_encoder
                    를 고정 상태로 둡니다.
            3. 이후 아래 모듈들만 다시 requires_grad=True로 켭니다.
                    self.action_in
                    self.time_embed
                    self.global_cond
                    self.blocks
                    self.final_norm
                    self.out
            4. 추가로
                    self.action_time_tokens
                    self.modality_embeddings
                    도 학습 가능하게 둡니다.

        즉 의미상으로는:

            이미지/텍스트/상태 인코더는 고정
            현재 action x_t를 받아서 velocity v를 예측하는 핵심 흐름만 학습

        입니다.

        쉽게 말하면 이 함수는
        “조건 인코더는 pretrained/기존 표현을 그대로 쓰고, flow head와 Transformer 중심부만 미세조정하자”
        라는 설정입니다.
        """
        # 1) 전체 파라미터 일단 모두 freeze
        for p in self.parameters():
            p.requires_grad = False
        
        # 2) pretrained condition encoder만 freeze 유지
        self.freeze_condition_encoders()

        # 3) flow 예측에 필요한 모듈 + state_tokenizer는 다시 학습 가능하게 설정
        train_modules = [
            self.state_tokenizer,   # <- state_tokenizer는 학습이 필요함
            self.action_in,
            self.time_embed,
            self.global_cond,
            self.blocks,
            self.final_norm,
            self.out,
        ]
        for module in train_modules:
            set_requires_grad(module, True)

        # 4) 학습 가능한 직접 파라미터들도 다시 켬
        self.action_time_tokens.requires_grad = True
        for param in self.modality_embeddings.parameters():
            param.requires_grad = True

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        “패딩 빼고 평균”

        왜 이 함수가 필요하냐?

            build_context()를 보면:

                vis_summary = vis_tokens.mean(dim=1)
                txt_summary = self._masked_mean(txt_tokens, txt_mask)
                state_summary = state_tokens.mean(dim=1)

            여기서 이미지 토큰과 상태 토큰은 전부 유효하니까 그냥 평균을 내도 됩니다.
            하지만 텍스트는 문장 길이가 매번 다르고 padding이 붙기 때문에,
            그냥 mean(dim=1)을 하면 padding까지 같이 평균돼서 요약이 망가질 수 있습니다.

            그래서 텍스트만 _masked_mean()을 써서 진짜 단어 토큰만 평균내는 겁니다

        아주 쉬운 숫자 예시

            예를 들어 텍스트 토큰 4개가 있고, 마지막 2개는 패딩이라고 해볼게요.

                x =
                [
                [2, 4],
                [6, 8],
                [100, 100],   # padding
                [100, 100]    # padding
                ]

                mask = [1, 1, 0, 0]

            그러면

                x * mask =
                [
                [2, 4],
                [6, 8],
                [0, 0],
                [0, 0]
                ]

            합은:

                [8, 12]

            유효 토큰 수는 2개니까 평균은:

                [4, 6]
        """
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    def build_context(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vis_tokens = self.vision_encoder(image)
        txt_tokens, txt_mask = self.text_encoder(input_ids, attention_mask)
        state_tokens = self.state_tokenizer(state)

        vis_tokens = vis_tokens + self.modality_embeddings["vision"]
        txt_tokens = txt_tokens + self.modality_embeddings["text"]
        state_tokens = state_tokens + self.modality_embeddings["state"]

        B, Nv, _ = vis_tokens.shape
        _, Nt, _ = txt_tokens.shape
        _, Ns, _ = state_tokens.shape

        vis_mask = torch.ones(B, Nv, dtype=torch.bool, device=image.device)
        state_mask = torch.ones(B, Ns, dtype=torch.bool, device=image.device)

        ctx_tokens = torch.cat([vis_tokens, txt_tokens, state_tokens], dim=1)
        ctx_mask = torch.cat([vis_mask, txt_mask, state_mask], dim=1)

        vis_summary = vis_tokens.mean(dim=1)
        txt_summary = self._masked_mean(txt_tokens, txt_mask)
        state_summary = state_tokens.mean(dim=1)
        global_cond = self.global_cond(torch.cat([vis_summary, txt_summary, state_summary], dim=-1))
        return ctx_tokens, ctx_mask, global_cond

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        ctx_tokens, ctx_mask, global_cond = self.build_context(image, input_ids, attention_mask, state)
        B = x_t.size(0)

        action_tokens = self.action_in(x_t)
        action_tokens = action_tokens + self.action_time_tokens[:, :self.horizon]
        action_tokens = action_tokens + self.modality_embeddings["action"]

        seq = torch.cat([ctx_tokens, action_tokens], dim=1)
        seq = self.pos(seq)

        action_mask = torch.ones(B, self.horizon, dtype=torch.bool, device=seq.device)
        full_mask = torch.cat([ctx_mask, action_mask], dim=1)
        key_padding_mask = ~full_mask
        """
        key_padding_mask = ~full_mask의 역할은 Transformer attention에서 
        “봐야 할 토큰”과 “무시해야 할 토큰”을 구분하는 것입니다.
        특히 여기서는 텍스트의 padding 토큰을 attention에서 제외하려고 씁니다.
        """

        cond = global_cond + self.time_embed(t)
        for blk in self.blocks:
            seq = blk(seq, cond, key_padding_mask=key_padding_mask)
        """
        blk 입력값 seq와 cond의 크기는 어떻게 되는가?

            여기서 shape는 이렇게 보면 됩니다.

                seq.shape = (B, N_total, d_model)
                cond.shape = (B, d_model)

            즉 seq는 토큰 시퀀스 전체, cond는 배치마다 1개씩 있는 전역 조건 벡터입니다.

            조금 더 풀면, forward()에서 먼저

                ctx_tokens, ctx_mask, global_cond = self.build_context(...)
                action_tokens = self.action_in(x_t)
                seq = torch.cat([ctx_tokens, action_tokens], dim=1)
                cond = global_cond + self.time_embed(t)

            이렇게 만들어집니다.

            그래서 각 항목 shape는:

                1. cond 크기

                    global_cond는 vision/text/state 요약 3개를 이어붙인 뒤 self.global_cond MLP를 통과해서 만들어지고,
                    self.time_embed(t)도 같은 d_model 차원을 내도록 설계되어 있습니다. 따라서

                        global_cond.shape = (B, d_model)
                        self.time_embed(t).shape = (B, d_model)
                        cond.shape = (B, d_model)

                    입니다. AdaLNZero 안에서도 cond를 받아 chunk(3, dim=-1) 하는 구조라서 배치별 벡터 형태가 맞습니다.

                2. seq 크기

                    seq는 ctx_tokens와 action_tokens를 이어붙인 것이므로

                        ctx_tokens.shape = (B, N_ctx, d_model)
                        action_tokens.shape = (B, horizon, d_model)
                        seq.shape = (B, N_ctx + horizon, d_model)

                    입니다.

                    여기서 N_ctx는 다시

                        Nv: vision token 개수
                        Nt: text token 개수
                        Ns: state token 개수

                    의 합입니다. 즉

                        N_ctx = Nv + Nt + Ns
                        seq.shape = (B, Nv + Nt + Ns + horizon, d_model)

                    입니다.

                3. state token 개수 Ns

                    StateTokenizer는 항상

                        summary token 1개
                        실제 state token 1개

                    를 붙여서 반환하므로

                        Ns = 2

                    입니다. 코드 주석에도 최종 출력이 (2, 2, 256) 예시로 설명되어 있습니다.

                4. text token 개수 Nt

                    텍스트는 CLIP tokenizer에서 max_length=48로 잘라서 쓰고,
                    CLIP text encoder 출력 예시도 (1, 48, 256)으로 설명되어 있으므로 보통

                        Nt = 48

                    로 생각하면 됩니다. 다만 실제로는 뒤쪽 일부가 padding일 수 있고, 그건 key_padding_mask로 가립니다.
                    shape 자체는 48 길이입니다.

                5. vision token 개수 Nv

                    이건 어떤 vision encoder를 쓰느냐에 따라 달라집니다.

                        PatchVisionEncoder를 쓰면 주석 예시상 (B, 256, 14, 14) feature map을 펼쳐서
                            Nv = 14 × 14 = 196
                        ResNetTokenEncoder를 쓰면 마지막 feature map을 펼친 토큰 수가 들어갑니다. 224 입력의 일반적인 ResNet 구조라면 보통 마지막 spatial size가 7×7라
                            Nv ≈ 49

                    입니다.

                6. 실제 예시

                    예를 들어

                        B = 2
                        d_model = 256
                        horizon = 4
                        pretrained ResNet 사용 → Nv ≈ 49
                        Nt = 48
                        Ns = 2

                    라면

                        ctx_tokens.shape = (2, 49 + 48 + 2, 256) = (2, 99, 256)
                        action_tokens.shape = (2, 4, 256)
                        seq.shape = (2, 103, 256)
                        cond.shape = (2, 256)

                    가 됩니다.

                한 줄로 요약하면:

                    seq는 배치 × 전체 토큰 수 × hidden 차원
                    cond는 배치 × hidden 차원

                이고, TransformerBlock은 seq 전체를 업데이트하되 cond로 attention/MLP의 동작 방식을 조절합니다.
        """

        seq = self.final_norm(seq)
        action_out = seq[:, -self.horizon:, :]
        return self.out(action_out)


# ============================================================
# Save / load / sampling
# ============================================================

def save_checkpoint(
    path: str,
    model: Pi0LitePolicy,
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict,
    dataset: OpenVLADeltaDataset,
    step: int,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_cfg": train_cfg,
        "state_stats": None if dataset.state_stats is None else {k: v.cpu() for k, v in dataset.state_stats.items()},
        "action_stats": None if dataset.action_stats is None else {k: v.cpu() for k, v in dataset.action_stats.items()},
        "step": step,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


@torch.no_grad()
def sample_actions(
    model: Pi0LitePolicy,
    flow: FlowMatchingScheduler,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state: torch.Tensor,
    num_inference_steps: int,
    generator=None,
) -> torch.Tensor:
    return flow.euler_integrate(
        model=model,
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state=state,
        horizon=model.horizon,
        action_dim=model.action_dim,
        num_steps=num_inference_steps,
        device=image.device,
        generator=generator,
    )


# ============================================================
# Train / Predict
# ============================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    text_processor = CLIPTextProcessor(
        model_name=args.clip_model_name,
        max_length=args.max_text_len,
    )
    dataset = OpenVLADeltaDataset(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        text_processor=text_processor,
        image_size=args.image_size,
        horizon=args.horizon,
        action_dim=args.action_dim,
        normalize=not args.no_normalize,
            # args.no_normalize = False (기본값)
            # → normalize = not False = True
            # → ✅ 정규화 함
        action_format=args.action_format,
    )
    args.action_dim = dataset.action_dim
    args.action_format = dataset.action_format

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,   # 마지막 배치의 크기가 batch_size보다 작으면 그 배치를 버리겠다는 뜻
    )

    model = Pi0LitePolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        horizon=args.horizon,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        vision_encoder_type=args.vision_encoder_type,
        pretrained_backbone=args.pretrained_backbone,
        pretrained_vision=not args.no_pretrained_weights,
        clip_model_name=args.clip_model_name,
        freeze_clip_text=args.freeze_clip_text,
    ).to(device)

    if args.freeze_pretrained_vision:
        freeze_module(model.vision_encoder)

    if args.train_flow_only:
        model.enable_flow_only_training()

    total, trainable = count_parameters(model)   # 총 파라미터 개수와 학습 가능한 파라미터 개수 반환
    print(f"[INFO] params total={total:,} trainable={trainable:,}")
    print(
        f"[INFO] dataset samples={len(dataset)} state_dim={dataset.state_dim} "
        f"action_dim={dataset.action_dim} action_format={dataset.action_format} horizon={args.horizon}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    """
    3️⃣ weight_decay가 뭐냐

    이건 한마디로:

            👉 가중치를 작게 유지하려는 힘

    입니다.

    수식 느낌으로 보면:

            w = w - lr * grad - lr * λ * w

    즉:

            👉 파라미터를 계속 조금씩 0쪽으로 당김

    6️⃣ 직관적으로 이해하기

        weight_decay 없으면
                모델: "값 크게 만들어서라도 맞추자"
                
        weight_decay 있으면
                모델: "가능하면 작은 값으로 맞춰라"
    """
    flow = FlowMatchingScheduler(eps=args.flow_eps)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        last_loss: Optional[float] = None
        for batch in loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)

            t = flow.sample_t(actions.size(0), device=device)
            x_t, target_v, _ = flow.sample_bridge(actions, t)
                # 입력값 actions는 현재값이 아니라 정답 action trajectory
                # actions는 데이터셋에 들어 있는 정답 미래 action 시퀀스입니다.
                # horizon=4이면 “현재 시점 기준으로 4개의 미래 action 시퀀스”를 의미합니다.
                # 그리고 x_t는 그 정답 action과 랜덤 noise를 섞어서 만든 중간 상태
                # 여기서 actions는
                #         👉 (4, action_dim)짜리 trajectory 전체
                # 이고
                #         👉 x_t도 동일 shape
                # 즉 모델은:
                #         trajectory 전체를 한 번에 이동시키는 방향(v)을 학습

            pred_v = model(x_t, t, image, input_ids, attention_mask, state)
            """
            pred_v 구조는 x_t와 같은 shape의 action velocity 시퀀스입니다.

            즉 보통:

                x_t.shape = (B, horizon, action_dim)
                pred_v.shape = (B, horizon, action_dim)

            입니다.

            왜냐하면 Pi0LitePolicy.forward()의 마지막 부분이 이렇게 되어 있기 때문입니다.

            전체 시퀀스를 Transformer에 통과시킨 뒤
                action_out = seq[:, -self.horizon:, :] 로 마지막 horizon개 action 토큰만 잘라내고
                return self.out(action_out) 으로 d_model → action_dim 선형변환을 합니다.

            그래서 반환 구조를 단계별로 보면:

                1. x_t
                        현재 noisy action trajectory
                        shape: (B, horizon, action_dim)
                2. action_in(x_t)
                        action 토큰으로 변환
                        shape: (B, horizon, d_model)
                3. Transformer 처리 후 action_out
                        action 토큰 부분만 추출
                        shape: (B, horizon, d_model)
                4. self.out(action_out)
                        최종 반환값 pred_v
                        shape: (B, horizon, action_dim)

            의미상으로는 각 원소가:

                배치의 각 샘플마다
                horizon의 각 미래 step마다
                action의 각 차원마다

            “정답 action 쪽으로 얼마나 움직여야 하는지”를 나타내는 velocity 값입니다.
            """

            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()

            global_step += 1
            last_loss = float(loss.item())
            if global_step % args.log_every == 0:
                print(f"[train] epoch={epoch + 1}/{args.epochs} step={global_step} flow_loss={last_loss:.6f}")

        epoch_index = epoch + 1
        if args.save_every_epochs > 0 and epoch_index % args.save_every_epochs == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"pi0_lite_epoch_{epoch_index}.pt")
            train_cfg = vars(args).copy()
            train_cfg["last_epoch_loss"] = last_loss
            save_checkpoint(ckpt_path, model, optimizer, train_cfg, dataset, global_step)
            print(f"[INFO] saved epoch checkpoint: {ckpt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "pi0_lite_final.pt")
    save_checkpoint(final_path, model, optimizer, vars(args).copy(), dataset, global_step)
    print(f"[INFO] training complete. saved: {final_path}")


@torch.no_grad()
def predict_single(
    checkpoint_path: str,
    image_path: str,
    instruction: str,
    state: List[float],
    steps: int = 32,
    image_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    device = device or default_device()
    ckpt = load_checkpoint(checkpoint_path, device=device)
    cfg = ckpt["train_cfg"]

    text_processor = CLIPTextProcessor(
        model_name=cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
        max_length=cfg["max_text_len"],
    )

    model = Pi0LitePolicy(
        state_dim=len(state),
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.1),
        vision_encoder_type=cfg.get("vision_encoder_type", "pretrained"),
        pretrained_backbone=cfg.get("pretrained_backbone", "resnet18"),
        pretrained_vision=not cfg.get("no_pretrained_weights", False),
        clip_model_name=cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
        freeze_clip_text=cfg.get("freeze_clip_text", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- 추론용 이미지를 모델이 받아들일 수 있는 텐서 형태로 바꾸는 전처리 단계 ----
    size = image_size or cfg["image_size"]
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
        # PIL 이미지를 NumPy 배열로 바꾸고, 픽셀값을
        #     원래 0 ~ 255
        #     바꿔서 0.0 ~ 1.0
        # 범위로 스케일링합니다.
    image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        # 모델이 기대하는 입력 형태로 바꾸는 단계
        # 원래 이미지 배열은 보통:
        #         (H, W, C)
        # 형태인데, PyTorch CNN은:
        #         (B, C, H, W)
        # 형태를 기대합니다.
    image = image * 2.0 - 1.0
        # 픽셀 범위를
        #     0 ~ 1
        # 에서 
        #     -1 ~ 1
        # 로 바꿉니다.
        # 학습 데이터 로더에서도 완전히 같은 처리를 하고 있으므로, 추론 입력 분포를 학습 때와 맞추기 위한 정규화

    text_tokens = text_processor.encode(instruction)
    input_ids = text_tokens["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_tokens["attention_mask"].unsqueeze(0).to(device)

    # ---- 추론에 들어온 state 리스트를 모델 입력 텐서로 만들고,
    # 학습 때 사용한 state 정규화(mean/std)를 똑같이 적용하는 단계 ----
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # 즉 원래가 예를 들어 15차원 state라면:
        #     입력 state: (15,)
        #     변환 후 state_t: (1, 15)

    # 여기서는 checkpoint 안에 저장돼 있는 학습 데이터 통계값을 꺼냅니다.
    state_stats = ckpt.get("state_stats")
    action_stats = ckpt.get("action_stats")
    if state_stats is not None:
        state_mean = state_stats["mean"].to(device)
        state_std = state_stats["std"].to(device)
        state_t = (state_t - state_mean) / state_std
            # 즉 추론용 현재 state도 학습 때와 똑같이:
            #         평균을 빼고
            #         표준편차로 나눠서
            # 표준화된 state로 바꿉니다.
            # 그래야 모델이 학습 때 보던 입력 분포와 추론 때 입력 분포가 일치합니다.

    flow = FlowMatchingScheduler(eps=cfg.get("flow_eps", 1e-4))
    actions = sample_actions(
        model,
        flow,
        image,
        input_ids,
        attention_mask,
        state_t,
        num_inference_steps=steps,
    )

    if action_stats is not None:
        action_mean = action_stats["mean"].to(device).view(1, 1, -1)
        action_std = action_stats["std"].to(device).view(1, 1, -1)
        actions = actions * action_std + action_mean

    actions_np = actions[0].cpu().numpy().tolist()
    first = actions_np[0]
    action_format = cfg.get("action_format", "cartesian_delta")
    return {
        "action_format": action_format,
        "predicted_action_sequence": actions_np,
        "first_action": format_output_action(first, action_format),
    }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi0_lite: multimodal token VLA with flow matching")
    sub = parser.add_subparsers(dest="cmd", required=True)
        # 하나의 실행 파일 안에서 여러 하위 명령어(subcommand)를 쓰게 만드는 설정
        # 즉 이런 형태를 가능하게 합니다.
        #         python pi0_lite.py train ...
        #         python pi0_lite.py predict ...

    tr = sub.add_parser("train")
    tr.add_argument("--jsonl_path", type=str, required=True)
        # required=True는 이 인자를 반드시 입력해야 한다는 뜻
    tr.add_argument("--image_root", type=str, required=True)
    tr.add_argument("--output_dir", type=str, required=True)
    tr.add_argument("--device", type=str, default=default_device())
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=20)
    tr.add_argument("--save_every_epochs", type=int, default=100)
    tr.add_argument("--image_size", type=int, default=224)
    tr.add_argument("--max_text_len", type=int, default=48)
    tr.add_argument("--horizon", type=int, default=4)
    tr.add_argument("--action_dim", type=int, default=0)
    tr.add_argument("--action_format", type=str, default="auto", choices=["auto", "cartesian_delta", "joint_delta"])
    tr.add_argument("--d_model", type=int, default=256)
    tr.add_argument("--n_layers", type=int, default=8)
    tr.add_argument("--n_heads", type=int, default=8)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--flow_eps", type=float, default=1e-4)
    tr.add_argument("--no_normalize", action="store_true")
    tr.add_argument("--vision_encoder_type", type=str, default="pretrained", choices=["simple", "pretrained"])
    tr.add_argument("--pretrained_backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    tr.add_argument("--no_pretrained_weights", action="store_true")
    tr.add_argument("--freeze_pretrained_vision", action="store_true")
    tr.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    tr.add_argument("--freeze_clip_text", action="store_true")
    tr.add_argument("--train_flow_only", action="store_true")

    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", type=str, required=True)
    pr.add_argument("--image_path", type=str, required=True)
    pr.add_argument("--instruction", type=str, required=True)
    pr.add_argument("--state_json", type=str, required=True)
    pr.add_argument("--steps", type=int, default=32)
    pr.add_argument("--device", type=str, default=default_device())
    pr.add_argument("--image_size", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        state = json.loads(args.state_json)
        out = predict_single(
            checkpoint_path=args.checkpoint,
            image_path=args.image_path,
            instruction=args.instruction,
            state=state,
            steps=args.steps,
            device=args.device,
            image_size=args.image_size,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))


# ============================================================
# Example
# ============================================================

## Docker 에서 실행 예시
#     경로를 Docker 기준으로 바꿔 !
#     도커 안에서 pip install torchvision 설치 필수

# 기존 openvla-lora 이미지 활용하면서 새 컨테이너를 mount로 띄우기
# docker run --gpus all -it --name openvla_diffusion \
#   -v /home/lst7910/isaac_ros2_ws:/workspace \
#   -w /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/openvla \
#   openvla-lora:latest \
#   /bin/bash

# 기존 컨테이너에 들어가기
# docker start openvla_diffusion
# docker exec -it openvla_diffusion /bin/bash


# python /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite.py train \
#   --jsonl_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_openvla_joint_delta_phase_aware.jsonl \
#   --image_root /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_dir /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_output \
#   --device cuda \
#   --epochs 300 \
#   --batch_size 2 \
#   --lr 5e-5 \
#   --vision_encoder_type pretrained \
#   --pretrained_backbone resnet18 \
#   --freeze_pretrained_vision \
#   --clip_model_name openai/clip-vit-base-patch32 \
#   --freeze_clip_text \
#   --train_flow_only \
#   --action_format joint_delta

# python /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite.py predict \
#   --checkpoint /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_output/pi0_lite_final.pt \
#   --image_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0005/images/sample_000000.jpg \
#   --instruction "의자를 집어" \
#   --state_json "[0.6289, 1.036, 0.3968, -0.6516, -0.34, 1.6468, -1.49, 0.08, 0.5622664110990461, 0.5539575737785446, 0.36995270014208675, 7.761320001389953e-06, 0.9999999993784533, -3.0746761244511756e-05, -1.541077514095348e-05]" \
#   --steps 32 \
#   --device cuda

# 한글 깨짐 방지 버전 
# export LANG=C.UTF-8
# export LC_ALL=C.UTF-8

# python /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite.py predict \
#   --checkpoint /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_output/pi0_lite_final.pt \
#   --image_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0005/images/sample_000000.jpg \
#   --instruction "의자를 집어" \
#   --state_json "[0.6289, 1.036, 0.3968, -0.6516, -0.34, 1.6468, -1.49, 0.08, 0.5622664110990461, 0.5539575737785446, 0.36995270014208675, 7.761320001389953e-06, 0.9999999993784533, -3.0746761244511756e-05, -1.541077514095348e-05]" \
#   --steps 32 \
#   --device cuda

"""
주의할 점
1. 지금 데이터가 너무 “짧은 chunk 정책”일 수 있음

        현재 구조는 보통

            현재 이미지 1장
            현재 상태 1개
            미래 action 몇 step

        이렇게 학습하잖아.

    이건 잘 되지만, Pi0-lite/flow matching 쪽은 원래 더 잘 맞는 데이터가 보통:

        더 다양한 장면
        더 다양한 instruction
        더 긴 horizon
        더 다양한 성공/실패 케이스

    야.

즉 형식은 충분하지만, 데이터 다양성은 부족할 수 있어.
"""