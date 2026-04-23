#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pi0_lite import (
    CLIPTextProcessor,
    CLIPTextTokenEncoder,
    ContinuousTimeEmbedding,
    FlowMatchingScheduler,
    PatchVisionEncoder,
    ResNetTokenEncoder,
    SinusoidalPositionalEncoding,
    StateTokenizer,
    TransformerBlock,
    build_action_vector,
    build_state_vector,
    count_parameters,
    default_device,
    format_output_action,
    freeze_module,
    load_checkpoint,
    set_requires_grad,
    set_seed,
)


DEFAULT_PHASE_ORDER = [
    "pre_grasp",
    "approach_far",
    "approach_mid",
    "approach_near",
    "contact_ready",
    "grasp",
    "close",
    "lift",
    "retreat",
    "policy",
    "policy_apply",
]


@dataclass
class RobotFrameWithPhase:
    image_path: str
    external_image_path: str
    instruction: str
    state: List[float]
    action: List[float]
    episode_key: str
    frame_index: int
    phase_name: str
    phase_id: int
    phase_weight: float


def _load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    return x * 2.0 - 1.0


def _phase_name_from_row(row: Dict) -> str:
    meta = row.get("meta", {})
    phase = row.get("phase") or meta.get("phase") or "unknown"
    return str(phase)


def _external_image_value_from_row(row: Dict) -> str:
    obs = row.get("observation", {})
    meta = row.get("meta", {})
    return str(
        obs.get("external_image")
        or row.get("external_image")
        or meta.get("external_image")
        or ""
    )


def _build_phase_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "pre_grasp": float(args.pre_grasp_loss_weight),
        "approach_far": float(args.approach_far_loss_weight),
        "approach_mid": float(args.approach_mid_loss_weight),
        "approach_near": float(args.approach_near_loss_weight),
        "contact_ready": float(args.contact_ready_loss_weight),
        "grasp": float(args.grasp_loss_weight),
        "close": float(args.close_loss_weight),
        "lift": float(args.lift_loss_weight),
        "retreat": float(args.retreat_loss_weight),
        "policy": float(args.policy_loss_weight),
        "policy_apply": float(args.policy_apply_loss_weight),
    }


class OpenVLADeltaDatasetPhaseAuxExternal(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        text_processor: CLIPTextProcessor,
        image_size: int = 224,
        horizon: int = 4,
        action_dim: int = 0,
        normalize: bool = True,
        action_format: str = "auto",
        external_image_root: Optional[str] = None,
        require_external_image: bool = False,
        phase_weights: Optional[Dict[str, float]] = None,
        phase_order: Optional[Sequence[str]] = None,
    ):
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.external_image_root = external_image_root or image_root
        self.text_processor = text_processor
        self.image_size = image_size
        self.horizon = horizon
        self.action_dim = action_dim
        self.normalize = normalize
        self.action_format = action_format
        self.require_external_image = require_external_image
        self.phase_weights = dict(phase_weights or {})
        self.phase_order = list(phase_order or DEFAULT_PHASE_ORDER)

        self.phase_to_id = self._build_phase_mapping(jsonl_path)
        self.id_to_phase = {idx: name for name, idx in self.phase_to_id.items()}
        self.frames = self._load_frames(jsonl_path)
        self.samples = self._build_sequence_indices(self.frames, horizon)
        self.state_dim = len(self.frames[0].state)
        self.action_stats = self._compute_action_stats() if normalize else None
        self.state_stats = self._compute_state_stats() if normalize else None

    def _resolve_image_path(self, rel_or_abs: str, root: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(root, rel_or_abs)

    def _build_phase_mapping(self, path: str) -> Dict[str, int]:
        found = set()
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                found.add(_phase_name_from_row(json.loads(line)))

        ordered = [name for name in self.phase_order if name in found]
        extras = sorted(name for name in found if name not in self.phase_order)
        ordered.extend(extras)
        if not ordered:
            ordered = ["unknown"]
        return {name: idx for idx, name in enumerate(ordered)}

    def _resolve_external_image_path(self, row: Dict, primary_image_path: str) -> str:
        external_value = _external_image_value_from_row(row)
        if external_value:
            resolved = self._resolve_image_path(external_value, self.external_image_root)
            if os.path.exists(resolved):
                return resolved
            raise FileNotFoundError(f"External image path does not exist: {resolved}")
        if self.require_external_image:
            raise ValueError("Row is missing external_image while require_external_image=True")
        return primary_image_path

    def _phase_weight(self, phase_name: str) -> float:
        return float(self.phase_weights.get(phase_name, 1.0))

    def _load_frames(self, path: str) -> List[RobotFrameWithPhase]:
        frames: List[RobotFrameWithPhase] = []
        inferred_action_format: Optional[str] = None
        with open(path, "r", encoding="utf-8") as handle:
            for line_idx, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue

                row = json.loads(line)
                obs = row.get("observation", {})
                action = row.get("action", {})
                image_path = self._resolve_image_path(obs.get("image", ""), self.image_root)
                external_image_path = self._resolve_external_image_path(row, image_path)
                instruction = obs.get("natural_language_instruction") or obs.get("instruction") or "do the task"
                state_vec = build_state_vector(row)
                action_vec, row_action_format = build_action_vector(action, self.action_format)
                phase_name = _phase_name_from_row(row)
                phase_id = int(self.phase_to_id[phase_name])

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
                frames.append(
                    RobotFrameWithPhase(
                        image_path=image_path,
                        external_image_path=external_image_path,
                        instruction=instruction,
                        state=state_vec,
                        action=action_vec,
                        episode_key=episode_key,
                        frame_index=frame_index,
                        phase_name=phase_name,
                        phase_id=phase_id,
                        phase_weight=self._phase_weight(phase_name),
                    )
                )

        if not frames:
            raise ValueError(f"No frames found in {path}")
        if inferred_action_format is not None:
            self.action_format = inferred_action_format
        return frames

    def _build_sequence_indices(
        self,
        frames: List[RobotFrameWithPhase],
        horizon: int,
    ) -> List[Tuple[int, List[int]]]:
        by_episode: Dict[str, List[int]] = {}
        for idx, frame in enumerate(frames):
            by_episode.setdefault(frame.episode_key, []).append(idx)

        samples: List[Tuple[int, List[int]]] = []
        for inds in by_episode.values():
            inds = sorted(inds, key=lambda i: frames[i].frame_index)
            for start_i in range(len(inds)):
                future = inds[start_i:start_i + horizon]
                if len(future) < horizon:
                    future = future + [future[-1]] * (horizon - len(future))
                samples.append((inds[start_i], future))
        return samples

    def _compute_action_stats(self) -> Dict[str, torch.Tensor]:
        actions = np.stack([self.frames[idx].action for idx, _ in self.samples], axis=0)
        return {
            "mean": torch.tensor(actions.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(actions.std(axis=0) + 1e-6, dtype=torch.float32),
        }

    def _compute_state_stats(self) -> Dict[str, torch.Tensor]:
        states = np.stack([frame.state for frame in self.frames], axis=0)
        return {
            "mean": torch.tensor(states.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(states.std(axis=0) + 1e-6, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_idx, future_indices = self.samples[idx]
        frame = self.frames[cur_idx]
        image = _load_image_tensor(frame.image_path, self.image_size)
        external_image = _load_image_tensor(frame.external_image_path, self.image_size)
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
            "external_image": external_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "phase_id": torch.tensor(frame.phase_id, dtype=torch.long),
            "phase_weight": torch.tensor(frame.phase_weight, dtype=torch.float32),
            "phase_name": frame.phase_name,
            "instruction": frame.instruction,
        }


class Pi0LitePhaseAuxExternalPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_phases: int,
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
        use_external_image: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model
        self.num_phases = num_phases
        self.use_external_image = use_external_image

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
        self.action_in = nn.Linear(action_dim, d_model)
        self.action_time_tokens = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        modality_embeddings = {
            "vision": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "text": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "state": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            "action": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
        }
        if use_external_image:
            modality_embeddings["external_vision"] = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_embeddings = nn.ParameterDict(modality_embeddings)

        self.pos = SinusoidalPositionalEncoding(d_model, max_len=1024)
        self.time_embed = ContinuousTimeEmbedding(d_model)
        context_summary_count = 4 if use_external_image else 3
        self.global_cond = nn.Sequential(
            nn.Linear(d_model * context_summary_count, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.phase_head = nn.Linear(d_model, num_phases)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, action_dim)

    def freeze_condition_encoders(self) -> None:
        freeze_module(self.vision_encoder)
        freeze_module(self.text_encoder)

    def enable_flow_only_training(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        self.freeze_condition_encoders()

        train_modules = [
            self.state_tokenizer,
            self.action_in,
            self.time_embed,
            self.global_cond,
            self.phase_head,
            self.blocks,
            self.final_norm,
            self.out,
        ]
        for module in train_modules:
            set_requires_grad(module, True)

        self.action_time_tokens.requires_grad = True
        for param in self.modality_embeddings.parameters():
            param.requires_grad = True

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    def build_context(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        external_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vis_tokens = self.vision_encoder(image)
        txt_tokens, txt_mask = self.text_encoder(input_ids, attention_mask)
        state_tokens = self.state_tokenizer(state)

        vis_tokens = vis_tokens + self.modality_embeddings["vision"]
        txt_tokens = txt_tokens + self.modality_embeddings["text"]
        state_tokens = state_tokens + self.modality_embeddings["state"]

        token_groups = [vis_tokens, txt_tokens, state_tokens]
        mask_groups = [
            torch.ones(vis_tokens.size(0), vis_tokens.size(1), dtype=torch.bool, device=image.device),
            txt_mask,
            torch.ones(state_tokens.size(0), state_tokens.size(1), dtype=torch.bool, device=image.device),
        ]
        summaries = [
            vis_tokens.mean(dim=1),
            self._masked_mean(txt_tokens, txt_mask),
            state_tokens.mean(dim=1),
        ]

        if self.use_external_image:
            external_source = image if external_image is None else external_image
            ext_tokens = self.vision_encoder(external_source)
            ext_tokens = ext_tokens + self.modality_embeddings["external_vision"]
            token_groups.insert(1, ext_tokens)
            mask_groups.insert(
                1,
                torch.ones(ext_tokens.size(0), ext_tokens.size(1), dtype=torch.bool, device=image.device),
            )
            summaries.insert(1, ext_tokens.mean(dim=1))

        ctx_tokens = torch.cat(token_groups, dim=1)
        ctx_mask = torch.cat(mask_groups, dim=1)
        global_cond = self.global_cond(torch.cat(summaries, dim=-1))
        return ctx_tokens, ctx_mask, global_cond

    def _forward_impl(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        external_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_tokens, ctx_mask, global_cond = self.build_context(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            external_image=external_image,
        )
        batch_size = x_t.size(0)
        action_tokens = self.action_in(x_t)
        action_tokens = action_tokens + self.action_time_tokens[:, :self.horizon]
        action_tokens = action_tokens + self.modality_embeddings["action"]

        seq = torch.cat([ctx_tokens, action_tokens], dim=1)
        seq = self.pos(seq)

        action_mask = torch.ones(batch_size, self.horizon, dtype=torch.bool, device=seq.device)
        full_mask = torch.cat([ctx_mask, action_mask], dim=1)
        key_padding_mask = ~full_mask
        cond = global_cond + self.time_embed(t)

        for block in self.blocks:
            seq = block(seq, cond, key_padding_mask=key_padding_mask)

        seq = self.final_norm(seq)
        action_out = seq[:, -self.horizon:, :]
        pred_v = self.out(action_out)
        phase_logits = self.phase_head(global_cond)
        return pred_v, phase_logits

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        external_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_v, _ = self._forward_impl(
            x_t=x_t,
            t=t,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            external_image=external_image,
        )
        return pred_v

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        external_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(
            x_t=x_t,
            t=t,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            external_image=external_image,
        )


def save_checkpoint(
    path: str,
    model: Pi0LitePhaseAuxExternalPolicy,
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict,
    dataset: OpenVLADeltaDatasetPhaseAuxExternal,
    step: int,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_cfg": train_cfg,
        "state_stats": None if dataset.state_stats is None else {k: v.cpu() for k, v in dataset.state_stats.items()},
        "action_stats": None if dataset.action_stats is None else {k: v.cpu() for k, v in dataset.action_stats.items()},
        "phase_to_id": dict(dataset.phase_to_id),
        "step": step,
    }
    torch.save(ckpt, path)


@torch.no_grad()
def sample_actions(
    model: Pi0LitePhaseAuxExternalPolicy,
    flow: FlowMatchingScheduler,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state: torch.Tensor,
    external_image: Optional[torch.Tensor],
    num_inference_steps: int,
    generator=None,
) -> torch.Tensor:
    x = torch.randn(
        (image.size(0), model.horizon, model.action_dim),
        device=image.device,
        generator=generator,
    )
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=image.device)
    for i in range(num_inference_steps):
        t = torch.full((image.size(0),), float(ts[i].item()), device=image.device)
        dt = float((ts[i + 1] - ts[i]).item())
        v = model(
            x,
            t,
            image,
            input_ids,
            attention_mask,
            state,
            external_image=external_image,
        )
        x = x + dt * v
    return x


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    text_processor = CLIPTextProcessor(
        model_name=args.clip_model_name,
        max_length=args.max_text_len,
    )
    phase_weights = _build_phase_weights_from_args(args)
    use_external_image = not args.disable_external_image

    dataset = OpenVLADeltaDatasetPhaseAuxExternal(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        text_processor=text_processor,
        image_size=args.image_size,
        horizon=args.horizon,
        action_dim=args.action_dim,
        normalize=not args.no_normalize,
        action_format=args.action_format,
        external_image_root=args.external_image_root,
        require_external_image=args.require_external_image,
        phase_weights=phase_weights,
    )
    args.action_dim = dataset.action_dim
    args.action_format = dataset.action_format

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = Pi0LitePhaseAuxExternalPolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        num_phases=len(dataset.phase_to_id),
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
        use_external_image=use_external_image,
    ).to(device)

    if args.freeze_pretrained_vision:
        freeze_module(model.vision_encoder)
    if args.train_flow_only:
        model.enable_flow_only_training()

    total, trainable = count_parameters(model)
    print(f"[INFO] params total={total:,} trainable={trainable:,}")
    print(
        f"[INFO] dataset samples={len(dataset)} state_dim={dataset.state_dim} "
        f"action_dim={dataset.action_dim} action_format={dataset.action_format} "
        f"horizon={args.horizon} phases={dataset.phase_to_id}"
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    flow = FlowMatchingScheduler(eps=args.flow_eps)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        last_total_loss: Optional[float] = None
        last_flow_loss: Optional[float] = None
        last_phase_loss: Optional[float] = None
        for batch in loader:
            image = batch["image"].to(device)
            external_image = batch["external_image"].to(device) if use_external_image else None
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            phase_id = batch["phase_id"].to(device)
            phase_weight = batch["phase_weight"].to(device)

            t = flow.sample_t(actions.size(0), device=device)
            x_t, target_v, _ = flow.sample_bridge(actions, t)
            pred_v, phase_logits = model.forward_with_aux(
                x_t=x_t,
                t=t,
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state,
                external_image=external_image,
            )

            per_sample_flow = F.mse_loss(pred_v, target_v, reduction="none").mean(dim=(1, 2))
            weighted_flow_loss = (per_sample_flow * phase_weight).mean()
            phase_loss = F.cross_entropy(phase_logits, phase_id)
            total_loss = weighted_flow_loss + float(args.phase_aux_loss_weight) * phase_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()

            global_step += 1
            last_total_loss = float(total_loss.item())
            last_flow_loss = float(weighted_flow_loss.item())
            last_phase_loss = float(phase_loss.item())
            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"total_loss={last_total_loss:.6f} flow_loss={last_flow_loss:.6f} "
                    f"phase_loss={last_phase_loss:.6f}"
                )

        epoch_index = epoch + 1
        if args.save_every_epochs > 0 and epoch_index % args.save_every_epochs == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"pi0_lite_phase_aux_external_epoch_{epoch_index}.pt")
            train_cfg = vars(args).copy()
            train_cfg["last_epoch_total_loss"] = last_total_loss
            train_cfg["last_epoch_flow_loss"] = last_flow_loss
            train_cfg["last_epoch_phase_loss"] = last_phase_loss
            train_cfg["use_external_image"] = use_external_image
            save_checkpoint(ckpt_path, model, optimizer, train_cfg, dataset, global_step)
            print(f"[INFO] saved epoch checkpoint: {ckpt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "pi0_lite_phase_aux_external_final.pt")
    final_cfg = vars(args).copy()
    final_cfg["use_external_image"] = use_external_image
    save_checkpoint(final_path, model, optimizer, final_cfg, dataset, global_step)
    print(f"[INFO] training complete. saved: {final_path}")


@torch.no_grad()
def predict_single(
    checkpoint_path: str,
    image_path: str,
    instruction: str,
    state: List[float],
    external_image_path: Optional[str] = None,
    steps: int = 32,
    image_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    device = device or default_device()
    ckpt = load_checkpoint(checkpoint_path, device=device)
    cfg = ckpt["train_cfg"]
    phase_to_id = ckpt.get("phase_to_id", {"unknown": 0})
    id_to_phase = {idx: name for name, idx in phase_to_id.items()}
    use_external_image = bool(cfg.get("use_external_image", True))

    text_processor = CLIPTextProcessor(
        model_name=cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
        max_length=cfg["max_text_len"],
    )
    model = Pi0LitePhaseAuxExternalPolicy(
        state_dim=len(state),
        action_dim=cfg["action_dim"],
        num_phases=len(phase_to_id),
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
        use_external_image=use_external_image,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    size = image_size or cfg["image_size"]
    image = _load_image_tensor(image_path, size).unsqueeze(0).to(device)
    if use_external_image:
        resolved_external_path = external_image_path or image_path
        external_image = _load_image_tensor(resolved_external_path, size).unsqueeze(0).to(device)
    else:
        resolved_external_path = None
        external_image = None

    text_tokens = text_processor.encode(instruction)
    input_ids = text_tokens["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_tokens["attention_mask"].unsqueeze(0).to(device)

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state_stats = ckpt.get("state_stats")
    action_stats = ckpt.get("action_stats")
    if state_stats is not None:
        state_mean = state_stats["mean"].to(device)
        state_std = state_stats["std"].to(device)
        state_t = (state_t - state_mean) / state_std

    flow = FlowMatchingScheduler(eps=cfg.get("flow_eps", 1e-4))
    actions = sample_actions(
        model=model,
        flow=flow,
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state=state_t,
        external_image=external_image,
        num_inference_steps=steps,
    )

    t_eval = torch.ones((1,), dtype=torch.float32, device=device)
    _, phase_logits = model.forward_with_aux(
        x_t=actions,
        t=t_eval,
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state=state_t,
        external_image=external_image,
    )

    if action_stats is not None:
        action_mean = action_stats["mean"].to(device).view(1, 1, -1)
        action_std = action_stats["std"].to(device).view(1, 1, -1)
        actions = actions * action_std + action_mean

    phase_probs = torch.softmax(phase_logits, dim=-1)[0]
    best_phase_id = int(torch.argmax(phase_probs).item())
    actions_np = actions[0].cpu().numpy().tolist()
    first = actions_np[0]
    action_format = cfg.get("action_format", "cartesian_delta")
    return {
        "action_format": action_format,
        "predicted_action_sequence": actions_np,
        "first_action": format_output_action(first, action_format),
        "predicted_phase": id_to_phase.get(best_phase_id, "unknown"),
        "predicted_phase_confidence": float(phase_probs[best_phase_id].item()),
        "phase_probabilities": {
            id_to_phase[idx]: float(phase_probs[idx].item()) for idx in range(len(phase_to_id))
        },
        "used_external_image": bool(use_external_image),
        "external_image_path": resolved_external_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="pi0_lite_phase_aux_external: phase auxiliary + weighted loss + external image VLA"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--jsonl_path", type=str, required=True)
    tr.add_argument("--image_root", type=str, required=True)
    tr.add_argument("--external_image_root", type=str, default=None)
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
    tr.add_argument("--disable_external_image", action="store_true")
    tr.add_argument("--require_external_image", action="store_true")
    tr.add_argument("--phase_aux_loss_weight", type=float, default=0.2)
    tr.add_argument("--pre_grasp_loss_weight", type=float, default=1.0)
    tr.add_argument("--approach_far_loss_weight", type=float, default=1.0)
    tr.add_argument("--approach_mid_loss_weight", type=float, default=1.0)
    tr.add_argument("--approach_near_loss_weight", type=float, default=1.5)
    tr.add_argument("--contact_ready_loss_weight", type=float, default=2.5)
    tr.add_argument("--grasp_loss_weight", type=float, default=1.0)
    tr.add_argument("--close_loss_weight", type=float, default=4.0)
    tr.add_argument("--lift_loss_weight", type=float, default=1.0)
    tr.add_argument("--retreat_loss_weight", type=float, default=1.0)
    tr.add_argument("--policy_loss_weight", type=float, default=1.0)
    tr.add_argument("--policy_apply_loss_weight", type=float, default=1.0)

    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", type=str, required=True)
    pr.add_argument("--image_path", type=str, required=True)
    pr.add_argument("--external_image_path", type=str, default=None)
    pr.add_argument("--instruction", type=str, required=True)
    pr.add_argument("--state_json", type=str, required=True)
    pr.add_argument("--steps", type=int, default=32)
    pr.add_argument("--device", type=str, default=default_device())
    pr.add_argument("--image_size", type=int, default=None)
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


## 실행 예시
# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_phase_aux_external.py train \
#   --jsonl_path /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_pi0_aux_external_delta.jsonl \
#   --image_root /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --external_image_root /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_dir /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/pi0_lite/pi0_lite_output/run_udp_aux_external \
#   --epochs 300
