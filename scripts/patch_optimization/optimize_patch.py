#!/usr/bin/env python3
"""Optimize a shared adversarial patch against DSGN (closest-car LSE evasion).

Always loads clean stereo images and pastes the current shared patch in-graph.
Does not rewrite datasets on disk.

Loss (training), selected by --loss:
  prob (default)  L = LSE_tau({ sigmoid(logit_q) : q in Q(B_clean) })
                  Q = top --max-matches Car proposals above --score-thresh at
                  BEV cells within --match-radius of the clean closest car.
                  Defaults thresh=0.33, max_matches=3. Note the gate means a
                  frame contributes no gradient once it drops below it.
  logit           L = LSE_tau({ logit_q : all Car anchors in the radius }),
                  ungated, optionally floored by --logit-clamp.

Reported max_s / val_max_s are always the ungated max Car probability, so they
stay meaningful after the gate empties.

Geometry, selected by --shape:
  square (default)  the CSV `size` square at the CSV centre.
  face              --area-frac of the rear-face box at its aspect ratio.

Val split, selected by --split:
  contiguous (default)  last val-frac of CSV rows.
  strided               every k-th row (balanced depth mix).

Resume / epochs: --epochs is always "how many more to run". Absolute epoch IDs
are stored in each .pt; on --resume the loop continues from epoch+1 (or from
--start-epoch+1 for old checkpoints without metadata). Snapshots are
patch_epochXXX / patch_best_epochXXX with absolute XXX; patch_best.* is the
latest-best pointer. Stdout is teed to out/run_TIMESTAMP.log.

Example
-------
  external/DSGN_custom/.venv/bin/python scripts/patch_optimization/optimize_patch.py \\
    --images dsgn/datasets/adria/training_kitti_labels \\
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
from datetime import datetime
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
    face_x0: float = 0.0
    face_y0: float = 0.0
    face_x1: float = 0.0
    face_y1: float = 0.0

    @property
    def face_w(self) -> float:
        return self.face_x1 - self.face_x0

    @property
    def face_h(self) -> float:
        return self.face_y1 - self.face_y0


RESAMPLE_MODES = ("bilinear", "bicubic", "area")


@dataclass
class RunOpts:
    """Everything about the objective and the patch's rendering.

    Defaults keep the original objective, geometry and per-frame stepping, so
    changing any of those requires passing a flag. Rendering is the one
    exception: resize_patch now antialiases, which the earlier runs did not,
    because that is what makes training agree with what apply_stereo_patches
    writes. Those runs are therefore not bit-reproducible from here.

    Near-car weighting (near_weight):
      "none"  all frames weighted equally (default, backward-compatible).
      "inv"   w = clamp(near_ref_depth / depth_m, 1.0, near_max_weight);
              frames closer than near_ref_depth get boosted ≤ near_max_weight×,
              frames farther get w=1 (not down-weighted).
      "step"  w = near_boost if depth_m <= near_thresh else 1.0.
    """

    loss: str = "prob"
    temperature: float = 0.2
    logit_temperature: float = 1.0
    logit_clamp: float | None = None
    match_radius: float = 2.0
    score_thresh: float = 0.33
    max_matches: int = 3
    shape: str = "square"
    area_frac: float = 0.23
    resample: str = "bilinear"
    quantize: bool = False
    near_weight: str = "none"   # none / inv / step
    near_thresh: float = 15.0   # step: frames ≤ this (m) are boosted
    near_boost: float = 5.0     # step: boost multiplier for near frames
    near_ref_depth: float = 15.0  # inv: depth (m) at which w=1; closer → >1
    near_max_weight: float = 10.0  # inv: cap on per-frame weight
    near_emphasis: str = "loss"  # loss / sample / both — how near_weight is spent


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
                    face_x0=float(r.get("x0", 0.0) or 0.0),
                    face_y0=float(r.get("y0", 0.0) or 0.0),
                    face_x1=float(r.get("x1", 0.0) or 0.0),
                    face_y1=float(r.get("y1", 0.0) or 0.0),
                )
            )
    rows.sort(key=lambda x: x.frame)
    return rows


def read_image_01(path: Path) -> torch.Tensor:
    """RGB float tensor in [0, 1], shape (3, H, W)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def paste_rect(img: torch.Tensor, patch: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
    """Overwrite a rectangle with its top-left at (x0, y0), clipped to the image.

    Blends through a mask rather than assigning in place so autograd keeps
    tracking the patch pixels.
    """
    _, ih, iw = img.shape
    _, ph, pw = patch.shape
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(iw, x0 + pw), min(ih, y0 + ph)
    if ix0 >= ix1 or iy0 >= iy1:
        return img
    px0, py0 = ix0 - x0, iy0 - y0
    canvas = img.new_zeros(img.shape)
    mask = img.new_zeros(1, ih, iw)
    canvas[:, iy0:iy1, ix0:ix1] = patch[:, py0 : py0 + (iy1 - iy0), px0 : px0 + (ix1 - ix0)]
    mask[:, iy0:iy1, ix0:ix1] = 1.0
    return img * (1.0 - mask) + canvas * mask


def patch_rect(spec: FrameSpec, shape: str, area_frac: float) -> tuple[int, int, int, int]:
    """Target rectangle in full-res pixels as (x0, y0, w, h).

    `square` reproduces the original geometry exactly: the CSV `size` square at
    the CSV centre, which the localizer has already clipped to keep on-image.
    `face` covers `area_frac` of the projected rear-face AABB at the face's own
    aspect ratio, centred on the face box -- the geometry the capacity tests
    used, so it carries their placement too, not just their shape.
    """
    if shape == "square":
        s = max(1, spec.size)
        return int(round(spec.center_x)) - s // 2, int(round(spec.center_y)) - s // 2, s, s

    if spec.face_w <= 0 or spec.face_h <= 0:
        raise ValueError(
            f"frame {spec.frame}: --shape face needs x0/y0/x1/y1 in the CSV"
        )
    k = float(np.sqrt(max(area_frac, 1e-6)))
    w = max(1, int(round(spec.face_w * k)))
    h = max(1, int(round(spec.face_h * k)))
    cx = 0.5 * (spec.face_x0 + spec.face_x1)
    cy = 0.5 * (spec.face_y0 + spec.face_y1)
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h


def right_x0_for(
    spec: FrameSpec, shape: str, rect: tuple[int, int, int, int], disp: float
) -> int:
    """Right-image paste column, keeping each shape's original rounding.

    The two branches differ by up to a pixel. That is deliberate: `square`
    reproduces this script's earlier runs and `face` reproduces ceiling_test's,
    so results stay comparable to the evidence each geometry came from.
    """
    x0, _, w, _ = rect
    if shape == "square":
        return int(round(spec.center_x - disp)) - w // 2
    return int(round(x0 - disp))


def visible_area_px(rect: tuple[int, int, int, int]) -> int:
    """Area of `rect` that actually lands inside the full-res image."""
    x0, y0, w, h = rect
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(FULL_W, x0 + w), min(FULL_H, y0 + h)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def resize_patch(patch: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    """Differentiable resize of a (3, H, W) patch to (h, w).

    antialias=True is required for parity with PIL, which always prefilters on
    downscale: measured against PIL on a saturated patch, bilinear agrees to
    0.9/255 with it and 154/255 without. Only `bilinear` reaches parity --
    bicubic differs by up to 47/255 (different cubic coefficients) and `area`
    by up to 227/255 (adaptive pooling is not PIL's BOX), so those two are for
    robustness augmentation, not for matching deployment.
    """
    kwargs = (
        {"align_corners": False, "antialias": True} if mode in ("bilinear", "bicubic") else {}
    )
    out = F.interpolate(patch.unsqueeze(0), size=(h, w), mode=mode, **kwargs).squeeze(0)
    return out.clamp(0.0, 1.0) if mode == "bicubic" else out


def quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    """Snap to the 8-bit grid with a straight-through gradient.

    The deployed patch is written as a uint8 PNG, so training without this sees
    a precision the attack never actually gets.
    """
    q = torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0
    return x + (q - x).detach()


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
    shape: str = "square",
    area_frac: float = 0.23,
    resample: str = "bilinear",
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Paste shared patch, downsample 0.5×, ImageNet-normalize, pad.

    Returns batched (1,3,H',W') tensors and unpadded (H, W).
    """
    left = left.to(device)
    right = right.to(device)
    rect = patch_rect(spec, shape, area_frac)
    x0, y0, w, h = rect
    p = resize_patch(patch, h, w, resample)
    if quantize:
        p = quantize_uint8(p)

    disp = f_u * baseline_m / spec.depth_m
    left_p = paste_rect(left, p, x0, y0)
    right_p = paste_rect(right, p, right_x0_for(spec, shape, rect, disp), y0)

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
    opts: RunOpts,
    resample: str,
) -> None:
    """Save full-res left image with patch + top-k proposal boxes overlaid."""
    from PIL import ImageDraw, ImageFont

    px0, py0, pw, ph = patch_rect(spec, opts.shape, opts.area_frac)
    p = resize_patch(patch.detach().float().cpu(), ph, pw, resample)
    if opts.quantize:
        p = quantize_uint8(p)
    left = left_full.detach().float().cpu()
    left_p = paste_rect(left, p, px0, py0)
    arr = (left_p.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Pasted patch region (cyan).
    draw.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(0, 255, 255), width=3)
    draw.text((px0, max(0, py0 - 14)), f"patch {pw}x{ph}", fill=(0, 255, 255), font=font)

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


def frame_weight(depth_m: float, opts: RunOpts) -> float:
    """Per-frame loss multiplier based on target depth.

    "none"  always 1.0.
    "inv"   w = clamp(near_ref_depth / depth_m, 1.0, near_max_weight);
            frames closer than near_ref_depth get ≥1× and farther frames
            stay at 1.0 (so far frames are never de-emphasised).
    "step"  w = near_boost if depth_m <= near_thresh else 1.0.
    """
    if opts.near_weight == "none":
        return 1.0
    d = max(depth_m, 0.1)
    if opts.near_weight == "inv":
        raw = opts.near_ref_depth / d
        return float(min(opts.near_max_weight, max(1.0, raw)))
    if opts.near_weight == "step":
        return float(opts.near_boost) if d <= opts.near_thresh else 1.0
    raise ValueError(f"unknown near_weight mode: {opts.near_weight!r}")


def build_epoch_order(frames: list[FrameSpec], opts: RunOpts) -> list[FrameSpec]:
    """Frame visiting order for one epoch.

    Under `--near-emphasis loss` this is the plain shuffle the script has
    always used. Under `sample`/`both` the same number of frames is redrawn
    with probability proportional to frame_weight, so near frames receive more
    optimizer *steps*.

    That distinction matters because of Adam. With --grad-accum 1 each frame
    gets its own step, and Adam's step is ~lr*m/sqrt(v): scaling one frame's
    loss by w scales both m and sqrt(v), so the step size barely changes and a
    per-frame loss weight is largely cancelled. Step count is not something
    Adam normalises away, so resampling actually shifts the objective.

    Frame count per epoch is held constant, so a resampled arm stays
    comparable to an unweighted one in both step count and wall-clock.
    """
    if opts.near_weight == "none" or opts.near_emphasis == "loss":
        order = frames[:]
        random.shuffle(order)
        return order
    w = np.array([frame_weight(f.depth_m, opts) for f in frames], dtype=np.float64)
    p = w / w.sum()
    idx = np.random.choice(len(frames), size=len(frames), replace=True, p=p)
    return [frames[int(i)] for i in idx]


def lse_loss(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_zeros(())
    t = max(float(temperature), 1e-6)
    return t * torch.logsumexp(scores / t, dim=0)


def car_logits_in_radius(
    bbox_cls: torch.Tensor,
    locations_bev: torch.Tensor,
    loc_x: float,
    loc_z: float,
    radius: float,
    num_classes: int,
    num_angles: int,
    car_idx: int = CAR_CLASS_IDX,
) -> torch.Tensor:
    """Every Car class logit at BEV cells within `radius`, ungated."""
    n, c, h, w = bbox_cls.shape
    assert c == num_angles * num_classes, f"bbox_cls channels {c} != {num_angles}*{num_classes}"
    logits = (
        bbox_cls.view(n, num_angles, num_classes, h, w)
        .permute(0, 3, 4, 1, 2)
        .reshape(n, -1, num_angles, num_classes)
    )
    dist = torch.hypot(locations_bev[:, 0] - loc_x, locations_bev[:, 1] - loc_z)
    mask = dist <= radius
    if mask.any():
        loc_ids = mask.nonzero(as_tuple=False).squeeze(1)
    else:
        _, loc_ids = torch.topk(dist, k=min(64, dist.numel()), largest=False)
    return logits[0, loc_ids, :, car_idx].reshape(-1)


def logit_lse_loss(
    logits: torch.Tensor, temperature: float, clamp_min: float | None
) -> torch.Tensor:
    """Ungated LSE in logit space.

    `clamp_min` floors each logit so anchors already driven far negative stop
    absorbing the gradient budget; None reproduces the unclamped form the
    capacity tests actually ran.
    """
    if logits.numel() == 0:
        return logits.new_zeros(())
    t = max(float(temperature), 1e-6)
    x = logits if clamp_min is None else logits.clamp(min=float(clamp_min))
    return t * torch.logsumexp(x / t, dim=0)


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


def frame_visible_fracs(
    spec: FrameSpec, images_root: Path, shape: str, area_frac: float
) -> tuple[float, float]:
    """Fraction of the patch rectangle visible in the (left, right) image."""
    rect = patch_rect(spec, shape, area_frac)
    _, y0, w, h = rect
    area = float(max(1, w * h))
    _, _, f_u, baseline = calib_for_frame(images_root / "calib" / f"{spec.frame}.txt")
    disp = f_u * baseline / spec.depth_m
    right_rect = (right_x0_for(spec, shape, rect, disp), y0, w, h)
    return visible_area_px(rect) / area, visible_area_px(right_rect) / area


def filter_visible_frames(
    frames: list[FrameSpec],
    images_root: Path,
    shape: str,
    area_frac: float,
    min_visible_frac: float,
) -> tuple[list[FrameSpec], list[tuple[str, float, float]]]:
    """Drop frames whose patch is not visible enough to carry a gradient.

    A rear face can project fully outside the image when the target car has
    drawn alongside the ego. `paste_rect` then returns the image untouched, so
    the loss has no path back to the patch and `backward()` raises -- and the
    `n_in_loss == 0` guard misses it, because the BEV anchors are still there.
    Such a frame is also meaningless for the attack: there is nowhere to print.
    """
    kept: list[FrameSpec] = []
    dropped: list[tuple[str, float, float]] = []
    for spec in frames:
        left_frac, right_frac = frame_visible_fracs(spec, images_root, shape, area_frac)
        if min(left_frac, right_frac) <= max(0.0, min_visible_frac):
            dropped.append((spec.frame, left_frac, right_frac))
        else:
            kept.append(spec)
    return kept, dropped


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
    opts: RunOpts,
    log_nms: bool = False,
    vis_path: Path | None = None,
) -> tuple[torch.Tensor, float, int, int, float]:
    """Returns (loss, max_car_prob, n_in_loss, n_nms_match, nms_max_s).

    `max_car_prob` is ungated: the strongest local Car probability whether or
    not the objective is currently looking at it. The gated version reads 0.0
    once everything drops under --score-thresh, which flatters the log exactly
    when the attack starts working.
    """
    left_path = images_root / "image_2" / f"{spec.frame}.png"
    right_path = images_root / "image_3" / f"{spec.frame}.png"
    calib_path = images_root / "calib" / f"{spec.frame}.txt"
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"missing images for frame {spec.frame}")

    left = read_image_01(left_path)
    right = read_image_01(right_path)
    calib, calib_r, f_u, baseline = calib_for_frame(calib_path)

    p = torch.sigmoid(z)
    resample = random.choice(RESAMPLE_MODES) if opts.resample == "random" else opts.resample
    img_l, img_r, image_size = prepare_stereo_pair(
        left,
        right,
        p,
        spec,
        f_u,
        baseline,
        device,
        shape=opts.shape,
        area_frac=opts.area_frac,
        resample=resample,
        quantize=opts.quantize,
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
    logits_all = car_logits_in_radius(
        outputs["bbox_cls"],
        locations_bev,
        spec.loc_x,
        spec.loc_z,
        opts.match_radius,
        num_classes=int(cfg.num_classes),
        num_angles=int(cfg.num_angles),
    )
    with torch.no_grad():
        max_s = float(logits_all.sigmoid().max().item()) if logits_all.numel() else 0.0

    matched: MatchedProposals | None = None
    if opts.loss == "logit":
        loss = logit_lse_loss(logits_all, opts.logit_temperature, opts.logit_clamp)
        n_in_loss = int(logits_all.numel())
    else:
        matched = matched_car_scores(
            outputs["bbox_cls"],
            locations_bev,
            spec.loc_x,
            spec.loc_z,
            opts.match_radius,
            num_classes=int(cfg.num_classes),
            num_angles=int(cfg.num_angles),
            score_thresh=opts.score_thresh,
            max_matches=opts.max_matches,
        )
        loss = lse_loss(matched.scores, opts.temperature)
        n_in_loss = matched.n

    n_nms, nms_max = 0, 0.0
    if log_nms:
        n_nms, nms_max = count_nms_matched_cars(
            outputs, cfg, image_size, calib.P, spec.loc_x, spec.loc_z, opts.match_radius
        )
    if vis_path is not None:
        if matched is None:
            matched = matched_car_scores(
                outputs["bbox_cls"],
                locations_bev,
                spec.loc_x,
                spec.loc_z,
                opts.match_radius,
                num_classes=int(cfg.num_classes),
                num_angles=int(cfg.num_angles),
                score_thresh=opts.score_thresh,
                max_matches=opts.max_matches,
            )
        boxes = decode_matched_boxes2d(
            outputs["bbox_reg"], matched, locations_bev, cfg, calib.P
        )
        save_match_visualization(
            vis_path, left, p, spec, boxes, f_u, baseline, opts, resample
        )
    return loss, max_s, n_in_loss, n_nms, nms_max


class Tee:
    """Duplicate writes to the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def start_run_log(out_dir: Path) -> Path:
    """Tee stdout/stderr into a timestamped log under `out_dir`; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"run_{stamp}.log"
    log_fp = log_path.open("w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fp)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_fp)  # type: ignore[assignment]
    latest = out_dir / "run.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except OSError:
        # Non-fatal on filesystems without symlink support; timestamped file remains.
        pass
    return log_path


def save_patch(
    z: torch.Tensor,
    out_dir: Path,
    tag: str,
    *,
    epoch: int | None = None,
    best_epoch: int | None = None,
    best_val_max_s: float | None = None,
    extra: dict | None = None,
) -> None:
    """Write `{tag}.png` and `{tag}.pt` with absolute-epoch metadata in the .pt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = torch.sigmoid(z.detach().cpu()).clamp(0, 1)
    payload: dict = {"z": z.detach().cpu(), "patch": p}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if best_val_max_s is not None:
        payload["best_val_max_s"] = float(best_val_max_s)
    if extra:
        payload.update(extra)
    torch.save(payload, out_dir / f"{tag}.pt")
    arr = (p.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(out_dir / f"{tag}.png")


@dataclass
class ResumeState:
    """Weights + absolute-epoch bookkeeping restored from --resume."""

    z: torch.Tensor
    epoch: int  # last completed global epoch (0 = fresh / unknown)
    best_epoch: int | None = None
    best_val_max_s: float | None = None


def load_z_init(
    path: Path,
    patch_size: int,
    device: torch.device,
    *,
    start_epoch: int = 0,
) -> ResumeState:
    """Load learnable logits `z` from a prior run (.pt preferred) or a patch PNG.

    Absolute epoch numbering: if the .pt stores `epoch`, the next training loop
    starts at epoch+1. Old checkpoints without metadata use `--start-epoch`
    (passed as `start_epoch`). `--epochs` always means additional epochs to run.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--resume not found: {path}")

    meta_epoch: int | None = None
    meta_best_epoch: int | None = None
    meta_best_val: float | None = None

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "z" in data:
            z = data["z"].float()
            if "epoch" in data and data["epoch"] is not None:
                meta_epoch = int(data["epoch"])
            if "best_epoch" in data and data["best_epoch"] is not None:
                meta_best_epoch = int(data["best_epoch"])
            if "best_val_max_s" in data and data["best_val_max_s"] is not None:
                meta_best_val = float(data["best_val_max_s"])
        elif isinstance(data, dict) and "patch" in data:
            p = data["patch"].float().clamp(1e-4, 1.0 - 1e-4)
            z = torch.logit(p)
            if "epoch" in data and data["epoch"] is not None:
                meta_epoch = int(data["epoch"])
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

    epoch = meta_epoch if meta_epoch is not None else int(start_epoch)
    return ResumeState(
        z=z.detach().to(device).requires_grad_(True),
        epoch=epoch,
        best_epoch=meta_best_epoch,
        best_val_max_s=meta_best_val,
    )


def split_frames(
    frames: list[FrameSpec],
    val_frac: float,
    mode: str = "contiguous",
) -> tuple[list[FrameSpec], list[FrameSpec]]:
    """Hold out `val_frac` of frames.

    contiguous  last N rows (legacy; near-heavy on this CSV's depth order).
    strided     every k-th row (k ≈ 1/val_frac), so train/val share the depth mix.
    """
    n = len(frames)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    if n_val == 0:
        return frames, []
    if mode == "contiguous":
        return frames[:-n_val], frames[-n_val:]
    if mode == "strided":
        # Take every k-th index as val, targeting ~val_frac. Prefer the denser
        # stride that still yields at least n_val when possible.
        k = max(2, int(round(1.0 / max(val_frac, 1e-6))))
        val_idx = set(range(k - 1, n, k))
        # If rounding under-shot, fill extras evenly from remaining indices.
        if len(val_idx) < n_val:
            remaining = [i for i in range(n) if i not in val_idx]
            need = n_val - len(val_idx)
            step = max(1, len(remaining) // need)
            val_idx.update(remaining[::step][:need])
        elif len(val_idx) > n_val:
            val_idx = set(sorted(val_idx)[:n_val])
        train = [f for i, f in enumerate(frames) if i not in val_idx]
        val = [f for i, f in enumerate(frames) if i in val_idx]
        return train, val
    raise ValueError(f"unknown split mode: {mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--images",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/training_kitti_labels",
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
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run in this invocation (additional when "
        "--resume is set). Absolute epoch IDs continue from the resumed "
        "checkpoint's stored epoch (or from --start-epoch).",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save patch_epochXXX.png/.pt every N absolute epochs (also always "
        "on the last epoch of this run). On val improvement also writes "
        "patch_best.* and patch_best_epochXXX.*",
    )
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--split",
        choices=("contiguous", "strided"),
        default="contiguous",
        help="'contiguous' (default) holds out the last val-frac of CSV rows. "
        "'strided' holds out every k-th row so train/val share the depth mix. "
        "Ignored when --val-csv is given.",
    )
    p.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Validate on this CSV instead of splitting --csv. Use it to hold "
        "out a whole driving event: these CSVs come from continuous video, so "
        "adjacent frames are near-duplicates and both --split modes leak them "
        "across the boundary, making val_max_s optimistic.",
    )
    p.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Fallback last-completed global epoch when --resume points at an "
        "old .pt/.png with no stored epoch metadata. Ignored when the "
        "checkpoint already has an 'epoch' field. Next loop starts at start-epoch+1.",
    )
    p.add_argument("--temperature", type=float, default=0.2, help="LSE temperature")
    p.add_argument(
        "--loss",
        choices=("prob", "logit"),
        default="prob",
        help="'prob' (default) is the original gated top-k probability LSE. "
        "'logit' is the ungated LSE over every local Car logit, which the "
        "per-frame capacity tests used.",
    )
    p.add_argument(
        "--logit-temperature",
        type=float,
        default=1.0,
        help="LSE temperature for --loss logit",
    )
    p.add_argument(
        "--logit-clamp",
        type=float,
        default=None,
        help="Floor each logit at this value under --loss logit (e.g. -2.0) so "
        "already-dead anchors stop absorbing gradient. Unset = unclamped, which "
        "is the form the capacity tests validated.",
    )
    p.add_argument(
        "--shape",
        choices=("square", "face"),
        default="square",
        help="'square' keeps the CSV size/centre (original behaviour). 'face' "
        "covers --area-frac of the rear-face box at its own aspect ratio, "
        "centred on that box.",
    )
    p.add_argument(
        "--area-frac",
        type=float,
        default=0.23,
        help="Rear-face area covered under --shape face. 0.23 is the area "
        "equivalent of the original square (0.50 of the face's shorter side).",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Frames to accumulate before an optimizer step. 1 (default) steps "
        "per frame as before; >1 trades update count for a less frame-specific "
        "gradient direction.",
    )
    p.add_argument(
        "--resample",
        choices=(*RESAMPLE_MODES, "random"),
        default="bilinear",
        help="Filter used to render the patch at target size. Only 'bilinear' "
        "has verified parity with the apply step (pair it with "
        "apply_stereo_patches --patch-resample bilinear). 'random' varies the "
        "filter per frame so the patch does not depend on any single one.",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Round the pasted patch to 8-bit in-graph (straight-through), "
        "matching the uint8 PNG that deployment actually writes.",
    )
    p.add_argument(
        "--min-visible-frac",
        type=float,
        default=0.0,
        help="Drop frames where the visible fraction of the patch (in either "
        "image) is <= this. 0.0 drops only fully off-image faces, which would "
        "otherwise crash the backward pass. Raise it to also skip frames where "
        "the target car is mostly out of view.",
    )
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
    p.add_argument(
        "--near-weight",
        choices=("none", "inv", "step"),
        default="none",
        help="Per-frame loss weighting by depth. "
        "'none' (default) weights all frames equally. "
        "'inv' scales each frame's loss by clamp(near-ref-depth/depth, 1, near-max-weight) "
        "so frames closer than near-ref-depth get ≥1× with a cap; farther frames stay at 1×. "
        "'step' applies near-boost× to frames with depth≤near-thresh, 1× otherwise.",
    )
    p.add_argument(
        "--near-thresh",
        type=float,
        default=15.0,
        help="Depth threshold in metres for 'step' weighting and for the near/far "
        "split in epoch logs (default 15 m, aligns with eval_patch.py bins).",
    )
    p.add_argument(
        "--near-boost",
        type=float,
        default=5.0,
        help="Loss multiplier for near frames under --near-weight step (default 5.0).",
    )
    p.add_argument(
        "--near-ref-depth",
        type=float,
        default=15.0,
        help="Reference depth (m) for --near-weight inv: at this depth w=1; "
        "closer frames get w>1 (default 15 m).",
    )
    p.add_argument(
        "--near-max-weight",
        type=float,
        default=10.0,
        help="Cap on per-frame weight under --near-weight inv (default 10.0).",
    )
    p.add_argument(
        "--near-emphasis",
        choices=("loss", "sample", "both"),
        default="loss",
        help="How --near-weight is spent. 'loss' (default) scales each frame's "
        "loss. With --grad-accum 1 that is largely cancelled by Adam, which "
        "normalises the step by the gradient magnitude, so prefer 'sample': it "
        "redraws the same number of frames per epoch with probability "
        "proportional to the weight, giving near frames more optimizer steps "
        "(which Adam cannot normalise away) at constant epoch cost. 'both' "
        "applies the weight to sampling and to the loss.",
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
        "or a patch PNG. Overrides --init / --n-inits. Absolute epoch numbering "
        "continues from the checkpoint's stored epoch (else --start-epoch); "
        "--epochs is how many more epochs to run.",
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

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = start_run_log(args.out)
    print(f"logging to {log_path} (also {args.out / 'run.log'})")
    print(f"argv: {' '.join(sys.argv)}")
    print(f"started: {datetime.now().isoformat(timespec='seconds')}")

    cfg = load_cfg(args.cfg)
    frames = load_csv(args.csv)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("no frames to optimize")

    n_before = len(frames)
    frames, dropped = filter_visible_frames(
        frames, args.images, args.shape, args.area_frac, args.min_visible_frac
    )
    if dropped:
        print(
            f"dropped {len(dropped)}/{n_before} frames whose patch is not visible "
            f"(min_visible_frac={args.min_visible_frac}): the rear face projects "
            f"off-image, so there is nothing to print on and no gradient."
        )
        for frame, lf, rf in dropped:
            print(f"  drop {frame}  visible left={lf:.3f} right={rf:.3f}")
    if not frames:
        raise RuntimeError("every frame was dropped as not visible")

    if args.val_csv is not None:
        val_frames = load_csv(args.val_csv)
        n_val_before = len(val_frames)
        val_frames, val_dropped = filter_visible_frames(
            val_frames, args.images, args.shape, args.area_frac, args.min_visible_frac
        )
        if val_dropped:
            print(
                f"--val-csv: dropped {len(val_dropped)}/{n_val_before} frames as not visible"
            )
        train_frames = frames
        overlap = {f.frame for f in train_frames} & {f.frame for f in val_frames}
        if overlap:
            raise RuntimeError(
                f"--val-csv shares {len(overlap)} frames with --csv "
                f"(e.g. {sorted(overlap)[:5]}); the val split would leak"
            )
        print(f"val split: held-out CSV {args.val_csv} ({len(val_frames)} frames)")
    else:
        train_frames, val_frames = split_frames(frames, args.val_frac, mode=args.split)
    n_inits = 1 if args.resume is not None else max(1, int(args.n_inits))
    opts = RunOpts(
        loss=args.loss,
        temperature=args.temperature,
        logit_temperature=args.logit_temperature,
        logit_clamp=args.logit_clamp,
        match_radius=args.match_radius,
        score_thresh=args.score_thresh,
        max_matches=args.max_matches,
        shape=args.shape,
        area_frac=args.area_frac,
        resample=args.resample,
        quantize=args.quantize,
        near_weight=args.near_weight,
        near_thresh=args.near_thresh,
        near_boost=args.near_boost,
        near_ref_depth=args.near_ref_depth,
        near_max_weight=args.near_max_weight,
        near_emphasis=args.near_emphasis,
    )
    accum = max(1, int(args.grad_accum))
    print(
        f"frames: {len(frames)} total | train={len(train_frames)} val={len(val_frames)} | "
        + (
            f"split=val-csv({args.val_csv.name}) | "
            if args.val_csv is not None
            else f"split={args.split} val_frac={args.val_frac} | "
        )
        + f"score_thresh={args.score_thresh} max_matches={args.max_matches} | "
        + f"init={args.init if args.resume is None else 'resume'} n_inits={n_inits}"
    )
    near_desc = opts.near_weight
    if opts.near_weight == "inv":
        near_desc = (
            f"inv(ref={opts.near_ref_depth}m cap={opts.near_max_weight}x)"
        )
    elif opts.near_weight == "step":
        near_desc = f"step(thresh={opts.near_thresh}m boost={opts.near_boost}x)"
    if opts.near_weight != "none":
        near_desc += f" via={opts.near_emphasis}"
    print(
        f"loss={opts.loss} "
        + (
            f"logit_tau={opts.logit_temperature} clamp={opts.logit_clamp} "
            if opts.loss == "logit"
            else f"tau={opts.temperature} "
        )
        + f"| shape={opts.shape}"
        + (f" area_frac={opts.area_frac}" if opts.shape == "face" else "")
        + f" | near_weight={near_desc}"
        + f" | grad_accum={accum} resample={opts.resample} quantize={opts.quantize}"
    )
    print(
        "NOTE: max_s/val_max_s are ungated max Car probability, so they no "
        "longer read 0 when the gate empties; earlier runs reported the gated "
        "value and are therefore optimistic by comparison."
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

    global_best_max: float | None = None
    global_best_z: torch.Tensor | None = None
    global_best_epoch: int | None = None

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
            resumed = load_z_init(
                args.resume, args.patch_size, device, start_epoch=args.start_epoch
            )
            z = resumed.z
            epoch_offset = resumed.epoch
            print(
                f"resumed z from {args.resume} "
                f"(saved epoch={resumed.epoch}"
                + (
                    f", best_epoch={resumed.best_epoch}, "
                    f"best_val_max_s={resumed.best_val_max_s:.4f}"
                    if resumed.best_epoch is not None and resumed.best_val_max_s is not None
                    else (
                        f", best_epoch={resumed.best_epoch}"
                        if resumed.best_epoch is not None
                        else ""
                    )
                )
                + ")"
            )
            end_epoch = epoch_offset + args.epochs
            print(
                f"continuing from global epoch {epoch_offset + 1} for "
                f"{args.epochs} more epochs → will end at {end_epoch}"
            )
            best_val_max = (
                resumed.best_val_max_s
                if resumed.best_val_max_s is not None
                else float("inf")
            )
            best_val_lse = float("inf")
            best_epoch = resumed.best_epoch
        else:
            z = make_z_init(args.init, args.patch_size, device, seed=trial_seed)
            epoch_offset = 0
            end_epoch = args.epochs
            best_val_max = float("inf")
            best_val_lse = float("inf")
            best_epoch = None

        optim = torch.optim.Adam([z], lr=args.lr)
        save_patch(z, trial_out, "patch_init", epoch=epoch_offset)
        # Seed best pointers from the resumed weights so a new --out still has a
        # usable patch_best even if this run never beats the restored score.
        if args.resume is not None and best_epoch is not None and best_val_max != float("inf"):
            meta = dict(
                epoch=epoch_offset,
                best_epoch=best_epoch,
                best_val_max_s=best_val_max,
            )
            save_patch(z, trial_out, "patch_best", **meta)
            save_patch(z, trial_out, f"patch_best_epoch{best_epoch:03d}", **meta)

        def _run(spec: FrameSpec, *, log_nms: bool, vis_path: Path | None = None):
            return run_frame(
                model,
                cfg,
                locations_bev,
                z,
                spec,
                args.images,
                device,
                opts,
                log_nms=log_nms,
                vis_path=vis_path,
            )

        for local_i, epoch in enumerate(range(epoch_offset + 1, end_epoch + 1), 1):
            order = build_epoch_order(train_frames, opts)
            n_drawn_near = sum(1 for f in order if f.depth_m <= opts.near_thresh)
            train_losses: list[float] = []
            train_max: list[float] = []
            train_near_max: list[float] = []  # max_s for frames <= near_thresh
            train_far_max: list[float] = []   # max_s for frames >  near_thresh
            n_skip = 0
            n_nograd = 0
            epoch_vis = vis_root / f"epoch{epoch:03d}"

            optim.zero_grad(set_to_none=True)
            pending = 0
            for i, spec in enumerate(order, 1):
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
                # Record the score even when the objective saw nothing, so a
                # frame going quiet cannot masquerade as a frame at zero.
                train_max.append(max_s)
                if spec.depth_m <= opts.near_thresh:
                    train_near_max.append(max_s)
                else:
                    train_far_max.append(max_s)
                if n_m == 0 or not loss.requires_grad:
                    # No grad path means the patch never reached the image for
                    # this frame; backward() would raise instead of skipping.
                    if n_m != 0:
                        n_nograd += 1
                    train_losses.append(0.0)
                    n_skip += 1
                else:
                    w = (
                        frame_weight(spec.depth_m, opts)
                        if opts.near_emphasis in ("loss", "both")
                        else 1.0
                    )
                    (loss * w / accum).backward()
                    pending += 1
                    train_losses.append(float(loss.item()))
                if pending >= accum or i == len(order):
                    if pending:
                        optim.step()
                    optim.zero_grad(set_to_none=True)
                    pending = 0
                if i == 1 or i % 20 == 0 or i == len(order):
                    extra = ""
                    if do_log:
                        extra = f" nms_match={n_nms} nms_max={nms_max:.4f}"
                    if do_vis:
                        extra += f" vis={vis_path.name}"
                    w_str = ""
                    if opts.near_weight != "none" and opts.near_emphasis in ("loss", "both"):
                        w_str = f" w={frame_weight(spec.depth_m, opts):.2f}"
                    print(
                        f"  epoch {epoch} [{i}/{len(order)}] frame={spec.frame} "
                        f"depth={spec.depth_m:.1f}m{w_str} "
                        f"loss={train_losses[-1]:.4f} max_s={train_max[-1]:.4f} "
                        f"matched={n_m}{extra}"
                    )

            val_losses: list[float] = []
            val_max: list[float] = []
            val_near_max: list[float] = []
            val_far_max: list[float] = []
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
                    val_max.append(max_s)
                    if spec.depth_m <= opts.near_thresh:
                        val_near_max.append(max_s)
                    else:
                        val_far_max.append(max_s)
                    if args.log_nms and n_nms == 0:
                        val_nms_gone += 1

            train_l = float(np.mean(train_losses)) if train_losses else 0.0
            val_l = float(np.mean(val_losses)) if val_losses else float("nan")
            val_m = float(np.mean(val_max)) if val_max else 0.0
            val_near_m = float(np.mean(val_near_max)) if val_near_max else float("nan")
            val_far_m = float(np.mean(val_far_max)) if val_far_max else float("nan")
            train_near_m = float(np.mean(train_near_max)) if train_near_max else float("nan")
            train_far_m = float(np.mean(train_far_max)) if train_far_max else float("nan")
            msg = (
                f"epoch {epoch}/{end_epoch} (+{local_i}/{args.epochs} this run)  "
                f"train_LSE={train_l:.4f}  "
                f"val_LSE={val_l:.4f}  val_max_s={val_m:.4f}  "
                f"val_near(≤{opts.near_thresh:.0f}m)={val_near_m:.4f}  "
                f"val_far={val_far_m:.4f}  "
                f"train_near={train_near_m:.4f}  train_far={train_far_m:.4f}  "
                f"train_skip0={n_skip}/{len(order)}"
            )
            if opts.near_emphasis in ("sample", "both") and opts.near_weight != "none":
                msg += f"  drawn_near={n_drawn_near}/{len(order)}"
            if n_nograd:
                msg += f"  WARN_nograd={n_nograd}"
            if args.log_nms and val_frames:
                msg += f"  val_nms_gone={val_nms_gone}/{len(val_frames)}"
            if args.vis_every > 0 or args.vis_val_max > 0:
                msg += f"  vis→{epoch_vis}"
            print(msg)
            # Periodic absolute-epoch snapshots; stamp best with its epoch too.
            save_every = max(1, int(args.save_every))
            if epoch % save_every == 0 or epoch == end_epoch:
                save_patch(
                    z,
                    trial_out,
                    f"patch_epoch{epoch:03d}",
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_max_s=None if best_val_max == float("inf") else best_val_max,
                )
                print(f"  saved patch_epoch{epoch:03d}")
            if val_frames and val_m <= best_val_max:
                best_val_max = val_m
                best_val_lse = val_l
                best_epoch = epoch
                meta = dict(
                    epoch=epoch,
                    best_epoch=epoch,
                    best_val_max_s=best_val_max,
                )
                save_patch(z, trial_out, "patch_best", **meta)
                save_patch(z, trial_out, f"patch_best_epoch{epoch:03d}", **meta)
                print(
                    f"  saved patch_best + patch_best_epoch{epoch:03d} "
                    f"(val_max_s={best_val_max:.4f} val_LSE={best_val_lse:.4f})"
                )

        save_patch(
            z,
            trial_out,
            "patch_final",
            epoch=end_epoch,
            best_epoch=best_epoch,
            best_val_max_s=None if best_val_max == float("inf") else best_val_max,
        )
        trial_score = best_val_max if val_frames else best_val_lse
        print(
            f"trial {trial + 1}/{n_inits} done  best_val_max_s={trial_score:.4f}"
            + (f"  best_epoch={best_epoch}" if best_epoch is not None else "")
        )
        if global_best_max is None or trial_score < global_best_max:
            global_best_max = trial_score
            global_best_z = z.detach().cpu().clone()
            global_best_epoch = best_epoch
            meta = dict(
                epoch=end_epoch if best_epoch is None else best_epoch,
                best_epoch=best_epoch,
                best_val_max_s=global_best_max,
            )
            save_patch(z, args.out, "patch_best", **meta)
            if best_epoch is not None:
                save_patch(z, args.out, f"patch_best_epoch{best_epoch:03d}", **meta)
            if n_inits > 1:
                print(
                    f"  new global best → {args.out}/patch_best.* "
                    f"(val_max_s={global_best_max:.4f}"
                    + (f" epoch={best_epoch}" if best_epoch is not None else "")
                    + ")"
                )

    if global_best_z is not None:
        # Ensure top-level final mirrors the best trial.
        z_final = global_best_z.to(device).requires_grad_(False)
        save_patch(
            z_final,
            args.out,
            "patch_final",
            epoch=end_epoch if n_inits == 1 else (global_best_epoch or end_epoch),
            best_epoch=global_best_epoch,
            best_val_max_s=global_best_max,
        )
        if n_inits > 1:
            print(
                f"selected best of {n_inits} inits: val_max_s={global_best_max:.4f}"
                + (f" epoch={global_best_epoch}" if global_best_epoch is not None else "")
            )
    print(f"done. outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())