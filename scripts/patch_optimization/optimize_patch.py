#!/usr/bin/env python3
"""Optimize a shared adversarial patch vs DSGN (closest-car logit evasion). See notes26/PATCH_OPTIMIZATION.md."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsgn.dataloader.kitti_util import Calibration
from dsgn.models import StereoNet
from dsgn.utils.torch_utils import compute_locations_bev

from patch_geometry import (
    FULL_H,
    FULL_W,
    FacePlacement,
    disparity_px,
    filter_visible_frames,
    load_face_csv,
    patch_rect,
    right_x0,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DOWNSCALE = 0.5
CAR_CLASS_IDX = 1  # labels are 1=Ped, 2=Car, 3=Cyclist → channel index class-1
LSE_TEMPERATURE = 1.0
MATCH_RADIUS_M = 2.0
SAVE_EVERY = 10
NEAR_DEPTH_M = 15.0
MIN_VISIBLE_FRAC = 0.0


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("dsgn_save_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config: {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg


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


def resize_patch(patch: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Differentiable bilinear resize of a (3, H, W) patch to (h, w).

    antialias=True is required for parity with PIL, which always prefilters on
    downscale.
    """
    return F.interpolate(
        patch.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


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
    spec: FacePlacement,
    f_u: float,
    baseline_m: float,
    device: torch.device,
    area_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Paste shared patch, downsample 0.5×, ImageNet-normalize, pad."""
    left = left.to(device)
    right = right.to(device)
    rect = patch_rect(spec, area_frac)
    if rect is None:
        raise ValueError(f"frame {spec.frame}: CSV needs a valid x0/y0/x1/y1 face box")
    x0, y0, w, h = rect
    p = resize_patch(patch, h, w)

    disp = disparity_px(f_u, baseline_m, spec.depth_m)
    left_p = paste_rect(left, p, x0, y0)
    right_p = paste_rect(right, p, right_x0(x0, disp), y0)

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


def evasion_loss(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """LSE over Car logits within the match radius."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    t = max(float(temperature), 1e-6)
    return t * torch.logsumexp(logits / t, dim=0)


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
    """Every Car class logit at BEV cells within `radius`."""
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


@dataclass
class FrameResult:
    """One pasted-frame forward through frozen DSGN."""

    loss: torch.Tensor
    max_s: float
    n_anchors: int
    n_nms: int = 0
    nms_max: float = 0.0


def run_frame(
    model: nn.Module,
    cfg,
    locations_bev: torch.Tensor,
    z: torch.Tensor,
    spec: FacePlacement,
    images_root: Path,
    device: torch.device,
    area_frac: float,
    *,
    nms: bool = False,
) -> FrameResult:
    """Paste the patch, run DSGN, return evasion loss and closest-car score.

    `nms=True` also counts post-NMS Car dets near the target (eval only).
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
    img_l, img_r, image_size = prepare_stereo_pair(
        left, right, p, spec, f_u, baseline, device, area_frac
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
        MATCH_RADIUS_M,
        num_classes=int(cfg.num_classes),
        num_angles=int(cfg.num_angles),
    )
    with torch.no_grad():
        max_s = float(logits_all.sigmoid().max().item()) if logits_all.numel() else 0.0

    n_nms, nms_max = 0, 0.0
    if nms:
        n_nms, nms_max = count_nms_matched_cars(
            outputs, cfg, image_size, calib.P, spec.loc_x, spec.loc_z, MATCH_RADIUS_M
        )

    return FrameResult(
        loss=evasion_loss(logits_all, LSE_TEMPERATURE),
        max_s=max_s,
        n_anchors=int(logits_all.numel()),
        n_nms=n_nms,
        nms_max=nms_max,
    )


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
) -> ResumeState:
    """Load learnable logits `z` from a prior run (.pt preferred) or a patch PNG.

    If the .pt stores `epoch`, the next training loop starts at epoch+1.
    Old checkpoints without that field resume from epoch 0.
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

    epoch = meta_epoch if meta_epoch is not None else 0
    return ResumeState(
        z=z.detach().to(device).requires_grad_(True),
        epoch=epoch,
        best_epoch=meta_best_epoch,
        best_val_max_s=meta_best_val,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=ROOT / "dsgn/datasets/adria/patch_train/dataset")
    p.add_argument("--csv", type=Path, default=ROOT / "dsgn/datasets/adria/patch_train/train.csv")
    p.add_argument("--val-csv", type=Path, default=ROOT / "dsgn/datasets/adria/patch_train/val.csv")
    p.add_argument("--cfg", type=Path, default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py")
    p.add_argument("--loadmodel", type=Path, default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar")
    p.add_argument("--out", type=Path, default=ROOT / "dsgn/datasets/adria/patch_train/face050")
    p.add_argument("--area-frac", type=float, default=0.50)
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs to run (additive when --resume is set)")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def init_patch_z(patch_size: int, device: torch.device, seed: int) -> torch.Tensor:
    """Low-frequency random colour patch, parameterised as logits."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    small = torch.rand(1, 3, 8, 8, generator=g, device=device)
    p = F.interpolate(
        small, size=(patch_size, patch_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    p = 0.1 + 0.8 * p
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

    train_frames, dropped = filter_visible_frames(
        load_face_csv(args.csv), args.images, args.area_frac, MIN_VISIBLE_FRAC
    )
    if dropped:
        print(f"dropped {len(dropped)} off-image train frames")
    if not train_frames:
        raise RuntimeError("no visible train frames")

    val_frames, val_dropped = filter_visible_frames(
        load_face_csv(args.val_csv), args.images, args.area_frac, MIN_VISIBLE_FRAC
    )
    if val_dropped:
        print(f"dropped {len(val_dropped)} off-image val frames")
    overlap = {f.frame for f in train_frames} & {f.frame for f in val_frames}
    if overlap:
        raise RuntimeError(
            f"--csv and --val-csv share {len(overlap)} frames "
            f"(e.g. {sorted(overlap)[:5]})"
        )

    print(
        f"train={len(train_frames)} val={len(val_frames)} | "
        f"area_frac={args.area_frac} | "
        f"match_radius={MATCH_RADIUS_M}m tau={LSE_TEMPERATURE}"
    )

    model = load_model(cfg, args.loadmodel, device)
    locations_bev = compute_locations_bev(
        cfg.Z_MIN, cfg.Z_MAX, cfg.VOXEL_Z_SIZE,
        cfg.X_MIN, cfg.X_MAX, cfg.VOXEL_X_SIZE, device,
    )

    if args.resume is not None:
        resumed = load_z_init(args.resume, args.patch_size, device)
        z = resumed.z
        epoch_offset = resumed.epoch
        end_epoch = epoch_offset + args.epochs
        best_val_max = resumed.best_val_max_s if resumed.best_val_max_s is not None else float("inf")
        best_epoch = resumed.best_epoch
        print(f"resumed from {args.resume} (epoch={resumed.epoch}, best_val_max_s={resumed.best_val_max_s})")
        print(f"continuing for {args.epochs} more epochs → end at {end_epoch}")
    else:
        z = init_patch_z(args.patch_size, device, args.seed)
        epoch_offset = 0
        end_epoch = args.epochs
        best_val_max = float("inf")
        best_epoch = None

    optim = torch.optim.Adam([z], lr=args.lr)
    save_patch(z, args.out, "patch_init", epoch=epoch_offset)
    if args.resume is not None and best_epoch is not None and best_val_max != float("inf"):
        meta = dict(epoch=epoch_offset, best_epoch=best_epoch, best_val_max_s=best_val_max)
        save_patch(z, args.out, "patch_best", **meta)
        save_patch(z, args.out, f"patch_best_epoch{best_epoch:03d}", **meta)

    for local_i, epoch in enumerate(range(epoch_offset + 1, end_epoch + 1), 1):
        order = train_frames[:]
        random.shuffle(order)
        train_losses: list[float] = []
        n_skip = 0

        for i, spec in enumerate(order, 1):
            out = run_frame(
                model, cfg, locations_bev, z, spec, args.images, device, args.area_frac
            )
            if out.n_anchors == 0 or not out.loss.requires_grad:
                train_losses.append(0.0)
                n_skip += 1
            else:
                out.loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                train_losses.append(float(out.loss.item()))
            if i == 1 or i % 20 == 0 or i == len(order):
                print(
                    f"  epoch {epoch} [{i}/{len(order)}] frame={spec.frame} "
                    f"depth={spec.depth_m:.1f}m loss={train_losses[-1]:.4f} "
                    f"max_s={out.max_s:.4f} anchors={out.n_anchors}"
                )

        val_losses: list[float] = []
        val_max: list[float] = []
        val_near: list[float] = []
        val_far: list[float] = []
        with torch.no_grad():
            for spec in val_frames:
                out = run_frame(
                    model, cfg, locations_bev, z, spec, args.images, device, args.area_frac
                )
                val_losses.append(float(out.loss.item()) if out.n_anchors else 0.0)
                val_max.append(out.max_s)
                (val_near if spec.depth_m <= NEAR_DEPTH_M else val_far).append(out.max_s)

        val_m = float(np.mean(val_max)) if val_max else 0.0
        print(
            f"epoch {epoch}/{end_epoch} (+{local_i}/{args.epochs})  "
            f"train_LSE={np.mean(train_losses):.4f}  val_LSE={np.mean(val_losses):.4f}  "
            f"val_max_s={val_m:.4f}  "
            f"val_near(≤{NEAR_DEPTH_M:.0f}m)={np.mean(val_near) if val_near else float('nan'):.4f}  "
            f"val_far={np.mean(val_far) if val_far else float('nan'):.4f}  "
            f"train_skip0={n_skip}/{len(order)}"
        )

        if epoch % SAVE_EVERY == 0 or epoch == end_epoch:
            save_patch(
                z, args.out, f"patch_epoch{epoch:03d}", epoch=epoch,
                best_epoch=best_epoch,
                best_val_max_s=None if best_val_max == float("inf") else best_val_max,
            )
            print(f"  saved patch_epoch{epoch:03d}")

        if val_frames and val_m <= best_val_max:
            best_val_max = val_m
            best_epoch = epoch
            meta = dict(epoch=epoch, best_epoch=epoch, best_val_max_s=best_val_max)
            save_patch(z, args.out, "patch_best", **meta)
            save_patch(z, args.out, f"patch_best_epoch{epoch:03d}", **meta)
            print(f"  saved patch_best (val_max_s={best_val_max:.4f})")

    save_patch(
        z, args.out, "patch_final", epoch=end_epoch,
        best_epoch=best_epoch,
        best_val_max_s=None if best_val_max == float("inf") else best_val_max,
    )
    print(f"done. best_val_max_s={best_val_max:.4f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())