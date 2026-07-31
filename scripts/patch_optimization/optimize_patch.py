#!/usr/bin/env python3
"""Optimize a shared adversarial patch against DSGN (closest-car LSE evasion).

Always loads clean stereo images and pastes the current shared patch in-graph.
Does not rewrite datasets on disk.

Loss (training):
  L = LogSumExp_tau({ s_q : q in Q(B_clean) })
where Q is the top --max-matches Car proposal scores (after --score-thresh)
at BEV locations within --match-radius of the clean closest-car
(loc_x, loc_z) from the CSV. Defaults: thresh=0.33, max_matches=3.

Example
-------
  external/DSGN_custom/.venv/bin/python scripts/patch_optimization/optimize_patch.py \\
    --images dsgn/datasets/arka/dsgn_awsim/training \\
    --csv dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv \\
    --cfg dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py \\
    --loadmodel dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar \\
    --out dsgn/datasets/adria/2.training_patch_optimization/optimize \\
    --epochs 5
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DSGN_ROOT = ROOT / "external" / "DSGN_custom"
sys.path.insert(0, str(DSGN_ROOT))
sys.path.insert(0, str(DSGN_ROOT / "tools"))

from dsgn.dataloader.kitti_util import Calibration  # noqa: E402
from dsgn.models import StereoNet  # noqa: E402
from dsgn.utils.torch_utils import compute_locations_bev, project_rect_to_image  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FULL_W, FULL_H = 1920, 1080
DOWNSCALE = 0.5
CAR_CLASS_IDX = 1  # labels are 1=Ped, 2=Car, 3=Cyclist → channel index class-1


@dataclass
class FrameSpec:
    frame: str
    center_x: float
    center_y: float
    size: int
    depth_m: float
    loc_x: float
    loc_z: float


@dataclass
class MatchedProposals:
    """Top-k local Car proposals used by the evasion loss."""

    scores: torch.Tensor  # (k,)
    loc_idx: torch.Tensor  # (k,) int64 into flattened BEV
    angle_idx: torch.Tensor  # (k,) int64

    @property
    def n(self) -> int:
        return int(self.scores.numel())


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("dsgn_save_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config: {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg


def load_csv(path: Path) -> list[FrameSpec]:
    rows: list[FrameSpec] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                FrameSpec(
                    frame=f"{int(r['frame']):06d}",
                    center_x=float(r["center_x"]),
                    center_y=float(r["center_y"]),
                    size=int(round(float(r["size"]))),
                    depth_m=float(r["depth_m"]),
                    loc_x=float(r["loc_x"]),
                    loc_z=float(r["loc_z"]),
                )
            )
    rows.sort(key=lambda x: x.frame)
    return rows


def read_image_01(path: Path) -> torch.Tensor:
    """RGB float tensor in [0, 1], shape (3, H, W)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def paste_square(img: torch.Tensor, patch: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
    """Overwrite a square centered at (cx, cy). Gradients flow into `patch`."""
    _, h, w = img.shape
    _, s, _ = patch.shape
    x0 = int(round(cx)) - s // 2
    y0 = int(round(cy)) - s // 2
    x1, y1 = x0 + s, y0 + s
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return img
    px0, py0 = ix0 - x0, iy0 - y0
    px1 = px0 + (ix1 - ix0)
    py1 = py0 + (iy1 - iy0)

    # Soft blend so autograd tracks patch pixels (avoid inplace on a non-grad clone).
    canvas = img.new_zeros(img.shape)
    mask = img.new_zeros(1, h, w)
    canvas[:, iy0:iy1, ix0:ix1] = patch[:, py0:py1, px0:px1]
    mask[:, iy0:iy1, ix0:ix1] = 1.0
    return img * (1.0 - mask) + canvas * mask


def pad_to_multiple(img: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    """img: (1, 3, H, W). Pad bottom/right."""
    _, _, h, w = img.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def prepare_stereo_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    patch: torch.Tensor,
    spec: FrameSpec,
    f_u: float,
    baseline_m: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Paste shared patch, downsample 0.5×, ImageNet-normalize, pad.

    Returns batched (1,3,H',W') tensors and unpadded (H, W).
    """
    left = left.to(device)
    right = right.to(device)
    size = max(1, spec.size)
    p = F.interpolate(
        patch.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)

    left_p = paste_square(left, p, spec.center_x, spec.center_y)
    disp = f_u * baseline_m / spec.depth_m
    right_p = paste_square(right, p, spec.center_x - disp, spec.center_y)

    net_h = int(FULL_H * DOWNSCALE)
    net_w = int(FULL_W * DOWNSCALE)
    left_s = F.interpolate(
        left_p.unsqueeze(0), size=(net_h, net_w), mode="bilinear", align_corners=False
    )
    right_s = F.interpolate(
        right_p.unsqueeze(0), size=(net_h, net_w), mode="bilinear", align_corners=False
    )

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    left_n = (left_s - mean) / std
    right_n = (right_s - mean) / std

    image_size = (net_h, net_w)
    left_n = pad_to_multiple(left_n)
    right_n = pad_to_multiple(right_n)
    return left_n, right_n, image_size


def matched_car_scores(
    bbox_cls: torch.Tensor,
    locations_bev: torch.Tensor,
    loc_x: float,
    loc_z: float,
    radius: float,
    num_classes: int,
    num_angles: int,
    score_thresh: float,
    max_matches: int,
    car_idx: int = CAR_CLASS_IDX,
) -> MatchedProposals:
    """Return up to `max_matches` strong Car proposals near (loc_x, loc_z).

    Collects raw proposal scores at BEV cells within `radius`, keeps those
    >= `score_thresh`, then returns the top-k (scores + BEV/angle indices).
    """
    # bbox_cls: (N, A*C, H, W) with class4angles; channel layout angle-major, class-fast.
    n, c, h, w = bbox_cls.shape
    assert n == 1
    assert c == num_angles * num_classes, f"bbox_cls channels {c} != {num_angles}*{num_classes}"
    scores = (
        bbox_cls.view(n, num_angles, num_classes, h, w)
        .permute(0, 3, 4, 1, 2)
        .reshape(n, -1, num_angles, num_classes)
        .sigmoid()
    )
    dist = torch.hypot(locations_bev[:, 0] - loc_x, locations_bev[:, 1] - loc_z)
    mask = dist <= radius
    if mask.any():
        loc_ids = mask.nonzero(as_tuple=False).squeeze(1)
    else:
        # Target may sit just outside the BEV grid (e.g. z > ~40 m).
        _, nn_idx = torch.topk(dist, k=min(64, dist.numel()), largest=False)
        loc_ids = nn_idx

    # (n_loc, A) → flat; flat i maps to (loc_ids[i // A], i % A)
    local = scores[0, loc_ids, :, car_idx]
    flat = local.reshape(-1)
    empty = MatchedProposals(
        scores=flat.new_zeros((0,)),
        loc_idx=loc_ids.new_zeros((0,)),
        angle_idx=loc_ids.new_zeros((0,)),
    )
    keep = flat >= score_thresh
    if not keep.any() or max_matches <= 0:
        return empty

    flat_kept = flat[keep]
    kept_i = torch.arange(flat.numel(), device=flat.device)[keep]
    k = min(int(max_matches), int(flat_kept.numel()))
    top_vals, top_pos = torch.topk(flat_kept, k=k)
    sel = kept_i[top_pos]
    loc_idx = loc_ids[sel // num_angles]
    angle_idx = sel % num_angles
    return MatchedProposals(scores=top_vals, loc_idx=loc_idx, angle_idx=angle_idx)


def decode_matched_boxes2d(
    bbox_reg: torch.Tensor,
    matched: MatchedProposals,
    locations_bev: torch.Tensor,
    cfg,
    calib_p: np.ndarray,
    car_idx: int = CAR_CLASS_IDX,
) -> list[tuple[float, float, float, float, float]]:
    """Decode top-k matched proposals to full-res 2D AABBs + score.

    Returns list of (x0, y0, x1, y1, score) in full-resolution image pixels.
    """
    if matched.n == 0:
        return []
    if not bool(getattr(cfg, "box_corner_parameters", True)):
        return []

    n, c, h, w = bbox_reg.shape
    num_angles = int(cfg.num_angles)
    num_classes = int(cfg.num_classes)
    pred_dim = 24
    assert c == num_angles * num_classes * pred_dim

    reg = (
        bbox_reg.view(n, num_angles, num_classes, pred_dim, h, w)
        .permute(0, 4, 5, 1, 2, 3)
        .reshape(n, -1, num_angles, num_classes, pred_dim)
    )
    anchors_y = torch.as_tensor(cfg.RPN3D.ANCHORS_Y, device=bbox_reg.device, dtype=torch.float32)
    proj = torch.as_tensor(np.asarray(calib_p, dtype=np.float32), device=bbox_reg.device)

    boxes: list[tuple[float, float, float, float, float]] = []
    with torch.no_grad():
        for i in range(matched.n):
            li = int(matched.loc_idx[i].item())
            ai = int(matched.angle_idx[i].item())
            score = float(matched.scores[i].item())
            corner_off = reg[0, li, ai, car_idx].reshape(8, 3)
            xz = locations_bev[li]
            loc3d = torch.stack([xz[0], anchors_y[car_idx], xz[1]])
            corners = corner_off + loc3d[None, :]
            pts2d = project_rect_to_image(corners, proj)  # half-res (network) pixels
            x0 = float(pts2d[:, 0].min().item()) / DOWNSCALE
            y0 = float(pts2d[:, 1].min().item()) / DOWNSCALE
            x1 = float(pts2d[:, 0].max().item()) / DOWNSCALE
            y1 = float(pts2d[:, 1].max().item()) / DOWNSCALE
            boxes.append((x0, y0, x1, y1, score))
    return boxes


def save_match_visualization(
    out_path: Path,
    left_full: torch.Tensor,
    patch: torch.Tensor,
    spec: FrameSpec,
    boxes: list[tuple[float, float, float, float, float]],
    f_u: float,
    baseline_m: float,
) -> None:
    """Save full-res left image with patch + top-k proposal boxes overlaid."""
    from PIL import ImageDraw, ImageFont

    size = max(1, spec.size)
    p = F.interpolate(
        patch.detach().float().cpu().unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    left = left_full.detach().float().cpu()
    left_p = paste_square(left, p, spec.center_x, spec.center_y)
    arr = (left_p.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Patch square (cyan).
    half = size // 2
    px0 = int(round(spec.center_x)) - half
    py0 = int(round(spec.center_y)) - half
    draw.rectangle([px0, py0, px0 + size, py0 + size], outline=(0, 255, 255), width=3)
    draw.text((px0, max(0, py0 - 14)), "patch", fill=(0, 255, 255), font=font)

    # Top-k loss proposals (red / orange / yellow by rank).
    colors = [(255, 40, 40), (255, 140, 0), (255, 220, 0)]
    for i, (x0, y0, x1, y1, score) in enumerate(boxes):
        color = colors[i % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text(
            (x0, max(0, y0 - 14)),
            f"#{i+1} s={score:.3f}",
            fill=color,
            font=font,
        )

    # Target BEV center projected with depth_m (green cross) — approximate.
    # Uses patch depth; enough to show which car we aimed at.
    _ = f_u, baseline_m  # kept for API symmetry / future stereo overlays
    cx, cy = spec.center_x, spec.center_y
    r = 10
    draw.line([cx - r, cy, cx + r, cy], fill=(0, 255, 0), width=2)
    draw.line([cx, cy - r, cx, cy + r], fill=(0, 255, 0), width=2)
    draw.text((cx + 12, cy - 10), f"target z={spec.loc_z:.1f}", fill=(0, 255, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def lse_loss(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_zeros(())
    t = max(float(temperature), 1e-6)
    return t * torch.logsumexp(scores / t, dim=0)


def load_model(cfg, ckpt_path: Path, device: torch.device) -> nn.Module:
    model = StereoNet(cfg=cfg)
    model.to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model_is_parallel = isinstance(model, nn.DataParallel)
    ckpt_has_module = next(iter(state)).startswith("module.")
    if model_is_parallel and not ckpt_has_module:
        state = {"module." + k: v for k, v in state.items()}
    elif not model_is_parallel and ckpt_has_module:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def calib_for_frame(calib_path: Path) -> tuple[Calibration, Calibration, float, float]:
    calib = Calibration.fromfile(str(calib_path))
    calib_r = Calibration.fromrightfile(str(calib_path))
    f_u = float(calib.f_u)
    baseline = abs(float(calib.P[0, 3] - calib_r.P[0, 3])) / f_u
    calib.scale(DOWNSCALE)
    calib_r.scale(DOWNSCALE)
    return calib, calib_r, f_u, baseline


def count_nms_matched_cars(
    outputs: dict,
    cfg,
    image_size: tuple[int, int],
    calib_p: np.ndarray,
    loc_x: float,
    loc_z: float,
    radius: float,
) -> tuple[int, float]:
    """Post-NMS Car dets near target; for logging only (not used in loss)."""
    from dsgn.models.inference3d import make_fcos3d_postprocessor

    with torch.no_grad():
        box_pred = make_fcos3d_postprocessor(cfg)(
            outputs["bbox_cls"].detach(),
            outputs["bbox_reg"].detach(),
            outputs["bbox_centerness"].detach(),
            image_sizes=[image_size],
            calibs_Proj=torch.as_tensor(np.asarray(calib_p, dtype=np.float32)[None, ...]),
        )
    pred = box_pred[0][0]
    if len(pred) == 0:
        return 0, 0.0
    labels = pred.get_field("labels")
    scores = pred.get_field("scores")
    corners = pred.get_field("box_corner3d")
    n_match = 0
    max_s = 0.0
    for i in range(len(pred)):
        if int(labels[i].item()) != 2:  # Car
            continue
        center = corners[i].mean(dim=0)
        dx = float(center[0].item()) - loc_x
        dz = float(center[2].item()) - loc_z
        if (dx * dx + dz * dz) ** 0.5 <= radius:
            n_match += 1
            max_s = max(max_s, float(scores[i].item()))
    return n_match, max_s


def run_frame(
    model: nn.Module,
    cfg,
    locations_bev: torch.Tensor,
    z: torch.Tensor,
    spec: FrameSpec,
    images_root: Path,
    device: torch.device,
    temperature: float,
    match_radius: float,
    score_thresh: float,
    max_matches: int,
    log_nms: bool = False,
    vis_path: Path | None = None,
) -> tuple[torch.Tensor, float, int, int, float]:
    """Returns (loss, max_score, n_matched, n_nms_match, nms_max_s)."""
    left_path = images_root / "image_2" / f"{spec.frame}.png"
    right_path = images_root / "image_3" / f"{spec.frame}.png"
    calib_path = images_root / "calib" / f"{spec.frame}.txt"
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"missing images for frame {spec.frame}")

    left = read_image_01(left_path)
    right = read_image_01(right_path)
    calib, calib_r, f_u, baseline = calib_for_frame(calib_path)

    p = torch.sigmoid(z)
    img_l, img_r, image_size = prepare_stereo_pair(
        left, right, p, spec, f_u, baseline, device
    )

    calibs_fu = torch.tensor([float(calib.f_u)], device=device, dtype=torch.float32)
    calibs_baseline = torch.tensor([float(baseline)], device=device, dtype=torch.float32)
    calibs_proj = torch.tensor(
        np.asarray(calib.P, dtype=np.float32)[None, ...], device=device
    )
    calibs_proj_r = torch.tensor(
        np.asarray(calib_r.P, dtype=np.float32)[None, ...], device=device
    )

    outputs = model(img_l, img_r, calibs_fu, calibs_baseline, calibs_proj, calibs_Proj_R=calibs_proj_r)
    matched = matched_car_scores(
        outputs["bbox_cls"],
        locations_bev,
        spec.loc_x,
        spec.loc_z,
        match_radius,
        num_classes=int(cfg.num_classes),
        num_angles=int(cfg.num_angles),
        score_thresh=score_thresh,
        max_matches=max_matches,
    )
    loss = lse_loss(matched.scores, temperature)
    max_s = float(matched.scores.max().item()) if matched.n else 0.0
    n_nms, nms_max = 0, 0.0
    if log_nms:
        n_nms, nms_max = count_nms_matched_cars(
            outputs, cfg, image_size, calib.P, spec.loc_x, spec.loc_z, match_radius
        )
    if vis_path is not None:
        boxes = decode_matched_boxes2d(
            outputs["bbox_reg"], matched, locations_bev, cfg, calib.P
        )
        save_match_visualization(
            vis_path, left, p, spec, boxes, f_u, baseline
        )
    return loss, max_s, matched.n, n_nms, nms_max


def save_patch(z: torch.Tensor, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = torch.sigmoid(z.detach().cpu()).clamp(0, 1)
    torch.save({"z": z.detach().cpu(), "patch": p}, out_dir / f"{tag}.pt")
    arr = (p.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(out_dir / f"{tag}.png")


def load_z_init(path: Path, patch_size: int, device: torch.device) -> torch.Tensor:
    """Load learnable logits `z` from a prior run (.pt preferred) or a patch PNG."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--resume not found: {path}")

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "z" in data:
            z = data["z"].float()
        elif isinstance(data, dict) and "patch" in data:
            p = data["patch"].float().clamp(1e-4, 1.0 - 1e-4)
            z = torch.logit(p)
        elif torch.is_tensor(data):
            z = data.float()
            if z.min() >= 0 and z.max() <= 1:
                z = torch.logit(z.clamp(1e-4, 1.0 - 1e-4))
        else:
            raise ValueError(f"{path} has no 'z' or 'patch' tensor")
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        p = read_image_01(path).clamp(1e-4, 1.0 - 1e-4)
        z = torch.logit(p)
    else:
        raise ValueError(f"--resume must be .pt or image, got {path.suffix}")

    if z.ndim != 3 or z.shape[0] != 3:
        raise ValueError(f"expected z shape (3,H,W), got {tuple(z.shape)} from {path}")
    if z.shape[-2:] != (patch_size, patch_size):
        z = F.interpolate(
            z.unsqueeze(0), size=(patch_size, patch_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        print(f"resized resumed patch to {patch_size}x{patch_size}")
    return z.detach().to(device).requires_grad_(True)


def split_frames(frames: list[FrameSpec], val_frac: float) -> tuple[list[FrameSpec], list[FrameSpec]]:
    n = len(frames)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    if n_val == 0:
        return frames, []
    return frames[:-n_val], frames[-n_val:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--images",
        type=Path,
        default=ROOT / "dsgn/datasets/arka/dsgn_awsim/training",
        help="Clean KITTI-layout dataset (image_2, image_3, calib)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv",
        help="Closest-car patch placement CSV",
    )
    p.add_argument(
        "--cfg",
        type=Path,
        default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py",
    )
    p.add_argument(
        "--loadmodel",
        type=Path,
        default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/2.training_patch_optimization/optimize",
    )
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.2, help="LSE temperature")
    p.add_argument("--match-radius", type=float, default=2.0, help="BEV match radius (m)")
    p.add_argument(
        "--score-thresh",
        type=float,
        default=0.33,
        help="Min proposal score to keep (DSGN PRE_NMS_THRESH default)",
    )
    p.add_argument(
        "--max-matches",
        type=int,
        default=3,
        help="Keep at most this many strongest local proposals for the loss",
    )
    p.add_argument(
        "--log-nms",
        action="store_true",
        help="Also log post-NMS matched Car count/score (val + progress prints)",
    )
    p.add_argument(
        "--vis-every",
        type=int,
        default=0,
        help="Save top-k proposal overlays every N train steps (0=disable, default). "
        "Also always saves step 1 and the last step of each epoch when >0.",
    )
    p.add_argument(
        "--vis-val-max",
        type=int,
        default=0,
        help="Max validation frames to visualize per epoch (0=disable, default)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap after CSV load (debug)")
    p.add_argument(
        "--init",
        choices=("color", "gray", "pixel"),
        default="color",
        help="Patch init when not using --resume: low-freq colour (default), uniform gray, "
        "or high-freq pixel noise",
    )
    p.add_argument(
        "--n-inits",
        type=int,
        default=1,
        help="Independent random inits; keep the run with lowest val_max_s. "
        "Ignored when --resume is set. Try 3–5 for a serious search.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue from a previous patch: patch_best.pt / patch_final.pt (preferred) "
        "or a patch PNG. Overrides --init / --n-inits.",
    )
    return p.parse_args()


def make_z_init(
    kind: str,
    patch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create learnable logits z for the shared patch."""
    if kind == "gray":
        return torch.zeros(3, patch_size, patch_size, device=device, requires_grad=True)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    if kind == "pixel":
        z = torch.randn(3, patch_size, patch_size, generator=g, device=device) * 0.1
        return z.requires_grad_(True)

    # Default: low-frequency random colour (8×8 → bilinear upsample).
    small = torch.rand(1, 3, 8, 8, generator=g, device=device)
    p = F.interpolate(
        small, size=(patch_size, patch_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    p = 0.1 + 0.8 * p  # keep away from sigmoid saturation
    z = torch.logit(p.clamp(1e-4, 1.0 - 1e-4))
    return z.detach().requires_grad_(True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    cfg = load_cfg(args.cfg)
    frames = load_csv(args.csv)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("no frames to optimize")

    train_frames, val_frames = split_frames(frames, args.val_frac)
    n_inits = 1 if args.resume is not None else max(1, int(args.n_inits))
    print(
        f"frames: {len(frames)} total | train={len(train_frames)} val={len(val_frames)} | "
        f"score_thresh={args.score_thresh} max_matches={args.max_matches} | "
        f"init={args.init if args.resume is None else 'resume'} n_inits={n_inits}"
    )

    model = load_model(cfg, args.loadmodel, device)
    locations_bev = compute_locations_bev(
        cfg.Z_MIN,
        cfg.Z_MAX,
        cfg.VOXEL_Z_SIZE,
        cfg.X_MIN,
        cfg.X_MAX,
        cfg.VOXEL_X_SIZE,
        device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    global_best_max: float | None = None
    global_best_z: torch.Tensor | None = None

    for trial in range(n_inits):
        trial_seed = args.seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        trial_out = args.out if n_inits == 1 else args.out / f"trial{trial:02d}"
        trial_out.mkdir(parents=True, exist_ok=True)
        vis_root = trial_out / "vis"
        if n_inits > 1:
            print(f"\n=== trial {trial + 1}/{n_inits} (seed={trial_seed}) → {trial_out} ===")

        if args.resume is not None:
            z = load_z_init(args.resume, args.patch_size, device)
            print(f"resumed z from {args.resume}")
        else:
            z = make_z_init(args.init, args.patch_size, device, seed=trial_seed)

        optim = torch.optim.Adam([z], lr=args.lr)
        save_patch(z, trial_out, "patch_init")

        def _run(spec: FrameSpec, *, log_nms: bool, vis_path: Path | None = None):
            return run_frame(
                model,
                cfg,
                locations_bev,
                z,
                spec,
                args.images,
                device,
                args.temperature,
                args.match_radius,
                args.score_thresh,
                args.max_matches,
                log_nms=log_nms,
                vis_path=vis_path,
            )

        best_val_max = float("inf")
        best_val_lse = float("inf")
        for epoch in range(1, args.epochs + 1):
            order = train_frames[:]
            random.shuffle(order)
            train_losses: list[float] = []
            train_max: list[float] = []
            n_skip = 0
            epoch_vis = vis_root / f"epoch{epoch:03d}"

            for i, spec in enumerate(order, 1):
                optim.zero_grad(set_to_none=True)
                do_log = args.log_nms and (i == 1 or i % 20 == 0 or i == len(order))
                do_vis = args.vis_every > 0 and (
                    i == 1 or i == len(order) or (i % args.vis_every == 0)
                )
                vis_path = None
                if do_vis:
                    vis_path = epoch_vis / "train" / f"{spec.frame}_step{i:04d}.png"
                loss, max_s, n_m, n_nms, nms_max = _run(
                    spec, log_nms=do_log, vis_path=vis_path
                )
                if n_m == 0:
                    train_losses.append(0.0)
                    train_max.append(0.0)
                    n_skip += 1
                else:
                    loss.backward()
                    optim.step()
                    train_losses.append(float(loss.item()))
                    train_max.append(max_s)
                if i == 1 or i % 20 == 0 or i == len(order):
                    extra = ""
                    if do_log:
                        extra = f" nms_match={n_nms} nms_max={nms_max:.4f}"
                    if do_vis:
                        extra += f" vis={vis_path.name}"
                    print(
                        f"  epoch {epoch} [{i}/{len(order)}] frame={spec.frame} "
                        f"loss={train_losses[-1]:.4f} max_s={train_max[-1]:.4f} "
                        f"matched={n_m}{extra}"
                    )

            val_losses: list[float] = []
            val_max: list[float] = []
            val_nms_gone = 0
            with torch.no_grad():
                for vi, spec in enumerate(val_frames):
                    vis_path = None
                    if args.vis_val_max > 0 and vi < args.vis_val_max:
                        vis_path = epoch_vis / "val" / f"{spec.frame}.png"
                    loss, max_s, n_m, n_nms, nms_max = _run(
                        spec, log_nms=args.log_nms, vis_path=vis_path
                    )
                    val_losses.append(float(loss.item()) if n_m else 0.0)
                    val_max.append(max_s if n_m else 0.0)
                    if args.log_nms and n_nms == 0:
                        val_nms_gone += 1

            train_l = float(np.mean(train_losses)) if train_losses else 0.0
            val_l = float(np.mean(val_losses)) if val_losses else float("nan")
            val_m = float(np.mean(val_max)) if val_max else 0.0
            msg = (
                f"epoch {epoch}/{args.epochs}  train_LSE={train_l:.4f}  "
                f"val_LSE={val_l:.4f}  val_max_s={val_m:.4f}  "
                f"train_skip0={n_skip}/{len(order)}"
            )
            if args.log_nms and val_frames:
                msg += f"  val_nms_gone={val_nms_gone}/{len(val_frames)}"
            if args.vis_every > 0 or args.vis_val_max > 0:
                msg += f"  vis→{epoch_vis}"
            print(msg)
            save_patch(z, trial_out, f"patch_epoch{epoch:03d}")
            # Prefer lower mean target-detection score on val.
            if val_frames and val_m <= best_val_max:
                best_val_max = val_m
                best_val_lse = val_l
                save_patch(z, trial_out, "patch_best")
                print(f"  saved patch_best (val_max_s={best_val_max:.4f} val_LSE={best_val_lse:.4f})")

        save_patch(z, trial_out, "patch_final")
        trial_score = best_val_max if val_frames else best_val_lse
        print(f"trial {trial + 1}/{n_inits} done  best_val_max_s={trial_score:.4f}")
        if global_best_max is None or trial_score < global_best_max:
            global_best_max = trial_score
            global_best_z = z.detach().cpu().clone()
            save_patch(z, args.out, "patch_best")
            if n_inits > 1:
                print(f"  new global best → {args.out}/patch_best.* (val_max_s={global_best_max:.4f})")

    if global_best_z is not None:
        # Ensure top-level final mirrors the best trial.
        z_final = global_best_z.to(device).requires_grad_(False)
        save_patch(z_final, args.out, "patch_final")
        if n_inits > 1:
            print(f"selected best of {n_inits} inits: val_max_s={global_best_max:.4f}")
    print(f"done. outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())