#!/usr/bin/env python3
"""Measure the attack ceiling before investing in more patch optimization.

Two experiments, both on clean stereo images with in-graph pasting:

`occlude`  Forward-only sweep: fill the rear-face region with worst-case
           content (black / white / gray / noise / an existing patch) at
           several fractions of the rear-face *area*, and record the
           closest-car score. Answers "at what coverage is suppression even
           possible?" without any gradients.

`overfit`  Per-frame capacity test: optimize a *dedicated* patch for a single
           frame at a fixed area fraction. If a patch that only has to work on
           one image cannot cross the detection threshold, patch area is the
           binding constraint and no universal patch will do better.
           `--loss` selects the training objective, so this doubles as a
           loss ablation at fixed coverage / shape / placement / init:
             logit  ungated LSE over raw Car logits (all anchors in radius)
             prob   optimize_patch.py's objective, imported verbatim: LSE over
                    sigmoid probabilities of the top-k proposals above
                    --score-thresh, with no gradient when nothing matches

Scores reported are the max Car anchor probability within --match-radius of the
CSV target, ungated (cfg.SCORE_MUL_CENTERNESS is False, so this equals the
post-NMS detection score of that car). Suppression means score < cfg PRE_NMS_THRESH.

Example
-------
  external/DSGN_custom/.venv/bin/python scripts/patch_optimization/ceiling_test.py occlude \\
    --n-frames 20 --area-fracs 0.23,0.5,0.75,1.0

  external/DSGN_custom/.venv/bin/python scripts/patch_optimization/ceiling_test.py overfit \\
    --n-frames 5 --area-frac 0.75 --steps 300
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_patch import (  # noqa: E402
    CAR_CLASS_IDX,
    DOWNSCALE,
    FULL_H,
    FULL_W,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROOT,
    calib_for_frame,
    count_nms_matched_cars,
    load_cfg,
    load_model,
    lse_loss,
    matched_car_scores,
    pad_to_multiple,
    read_image_01,
)

from dsgn.utils.torch_utils import compute_locations_bev  # noqa: E402


@dataclass
class Target:
    """One closest-car target with its projected rear-face box."""

    frame: str
    depth_m: float
    loc_x: float
    loc_z: float
    clean_score: float
    face_x0: float
    face_y0: float
    face_x1: float
    face_y1: float

    @property
    def face_w(self) -> float:
        return self.face_x1 - self.face_x0

    @property
    def face_h(self) -> float:
        return self.face_y1 - self.face_y0


def load_targets(csv_path: Path) -> list[Target]:
    out: list[Target] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            out.append(
                Target(
                    frame=f"{int(r['frame']):06d}",
                    depth_m=float(r["depth_m"]),
                    loc_x=float(r["loc_x"]),
                    loc_z=float(r["loc_z"]),
                    clean_score=float(r["score"]),
                    face_x0=float(r["x0"]),
                    face_y0=float(r["y0"]),
                    face_x1=float(r["x1"]),
                    face_y1=float(r["y1"]),
                )
            )
    out.sort(key=lambda t: t.frame)
    return out


def face_visible_frac(t: Target) -> float:
    """Fraction of the projected rear-face box that lies inside the image."""
    area = t.face_w * t.face_h
    if area <= 0:
        return 0.0
    vx = max(0.0, min(FULL_W, t.face_x1) - max(0.0, t.face_x0))
    vy = max(0.0, min(FULL_H, t.face_y1) - max(0.0, t.face_y0))
    return vx * vy / area


def select_targets(
    targets: list[Target],
    n: int,
    min_clean_score: float,
    max_depth: float,
    min_face_visible: float = 0.9,
) -> list[Target]:
    """Evenly spaced sample of hard, safety-relevant, well-posed targets.

    Frames whose rear face falls off-image are excluded: a patch there lands on
    background and can never suppress anything.
    """
    pool = [
        t
        for t in targets
        if t.clean_score >= min_clean_score
        and t.depth_m <= max_depth
        and face_visible_frac(t) >= min_face_visible
    ]
    if not pool:
        raise RuntimeError("no targets pass the clean-score / depth / visibility filters")
    if n >= len(pool):
        return pool
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [pool[i] for i in dict.fromkeys(idx.tolist())]


def write_rows(path: Path, rows: list[dict]) -> None:
    """CSV dump keyed on the first row's fields."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def face_rect(t: Target, area_frac: float) -> tuple[int, int, int, int]:
    """Rear-face box scaled to cover `area_frac` of its area, clipped to image.

    Returns (x0, y0, w, h) in full-resolution pixels.
    """
    s = float(np.sqrt(max(area_frac, 1e-6)))
    w = max(1, int(round(t.face_w * s)))
    h = max(1, int(round(t.face_h * s)))
    cx = 0.5 * (t.face_x0 + t.face_x1)
    cy = 0.5 * (t.face_y0 + t.face_y1)
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    return x0, y0, w, h


def paste_rect(img: torch.Tensor, patch: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
    """Overwrite a rectangle at (x0, y0). Gradients flow into `patch`."""
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


def to_net_input(
    left: torch.Tensor, right: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """0.5x downsample, ImageNet-normalize, pad to /32. Mirrors optimize_patch."""
    net_h, net_w = int(FULL_H * DOWNSCALE), int(FULL_W * DOWNSCALE)
    ls = F.interpolate(
        left.unsqueeze(0), size=(net_h, net_w), mode="bilinear", align_corners=False
    )
    rs = F.interpolate(
        right.unsqueeze(0), size=(net_h, net_w), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return pad_to_multiple((ls - mean) / std), pad_to_multiple((rs - mean) / std), (net_h, net_w)


def car_logits_in_radius(
    bbox_cls: torch.Tensor,
    locations_bev: torch.Tensor,
    loc_x: float,
    loc_z: float,
    radius: float,
    num_classes: int,
    num_angles: int,
) -> torch.Tensor:
    """All Car class logits at BEV cells within `radius` (no score gating)."""
    n, c, h, w = bbox_cls.shape
    assert c == num_angles * num_classes
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
    return logits[0, loc_ids, :, CAR_CLASS_IDX].reshape(-1)


class Scorer:
    """Runs DSGN on a clean pair with optional pasted content."""

    def __init__(self, args: argparse.Namespace):
        self.cfg = load_cfg(args.cfg)
        self.device = torch.device(args.device)
        self.model = load_model(self.cfg, args.loadmodel, self.device)
        self.locations_bev = compute_locations_bev(
            self.cfg.Z_MIN,
            self.cfg.Z_MAX,
            self.cfg.VOXEL_Z_SIZE,
            self.cfg.X_MIN,
            self.cfg.X_MAX,
            self.cfg.VOXEL_X_SIZE,
            self.device,
        )
        self.images = args.images
        self.radius = args.match_radius
        self.thresh = float(self.cfg.RPN3D.PRE_NMS_THRESH)
        self._cache: dict[str, tuple] = {}

    def frame_data(self, frame: str):
        if frame not in self._cache:
            left = read_image_01(self.images / "image_2" / f"{frame}.png").to(self.device)
            right = read_image_01(self.images / "image_3" / f"{frame}.png").to(self.device)
            calib, calib_r, f_u, baseline = calib_for_frame(
                self.images / "calib" / f"{frame}.txt"
            )
            self._cache[frame] = (left, right, calib, calib_r, f_u, baseline)
        return self._cache[frame]

    def forward(
        self,
        t: Target,
        patch_l: torch.Tensor | None = None,
        patch_r: torch.Tensor | None = None,
        rect: tuple[int, int, int, int] | None = None,
        right_dx: int = 0,
    ):
        left, right, calib, calib_r, f_u, baseline = self.frame_data(t.frame)
        if patch_l is not None and rect is not None:
            x0, y0, _, _ = rect
            disp = f_u * baseline / t.depth_m
            left = paste_rect(left, patch_l, x0, y0)
            right = paste_rect(
                right,
                patch_r if patch_r is not None else patch_l,
                int(round(x0 - disp)) + right_dx,
                y0,
            )
        img_l, img_r, image_size = to_net_input(left, right, self.device)
        outputs = self.model(
            img_l,
            img_r,
            torch.tensor([float(calib.f_u)], device=self.device),
            torch.tensor([float(baseline)], device=self.device),
            torch.tensor(np.asarray(calib.P, dtype=np.float32)[None], device=self.device),
            calibs_Proj_R=torch.tensor(
                np.asarray(calib_r.P, dtype=np.float32)[None], device=self.device
            ),
        )
        logits = car_logits_in_radius(
            outputs["bbox_cls"],
            self.locations_bev,
            t.loc_x,
            t.loc_z,
            self.radius,
            int(self.cfg.num_classes),
            int(self.cfg.num_angles),
        )
        return logits, outputs, image_size, calib

    def measure(self, t: Target, *, with_nms: bool = True, **kw) -> dict:
        """Score one configuration. `with_nms=False` skips the FCOS3D
        postprocessor, whose peak allocation is what makes this not fit on a
        busy GPU; `score` is unaffected since it reads bbox_cls directly.
        """
        with torch.no_grad():
            logits, outputs, image_size, calib = self.forward(t, **kw)
            pre_max = float(logits.sigmoid().max().item())
            n_nms, nms_max = (
                count_nms_matched_cars(
                    outputs, self.cfg, image_size, calib.P, t.loc_x, t.loc_z, self.radius
                )
                if with_nms
                else (-1, -1.0)
            )
        return {
            "score": pre_max,
            "nms_n": n_nms,
            "nms_max": nms_max,
            "suppressed": int(pre_max < self.thresh),
        }


def make_content(
    kind: str, h: int, w: int, seed: int, device, static: torch.Tensor | None, shift_px: int
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return (patch_left, patch_right_or_None, extra_right_dx).

    `extra_right_dx` perturbs the right-image paste position, i.e. injects a
    disparity error on top of the geometrically correct one.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "black":
        return torch.zeros(3, h, w, device=device), None, 0
    if kind == "white":
        return torch.ones(3, h, w, device=device), None, 0
    if kind == "gray":
        return torch.full((3, h, w), 0.5, device=device), None, 0
    if kind == "noise":
        return torch.rand(3, h, w, generator=g).to(device), None, 0
    if kind.startswith("stripes"):
        # Physically realizable: a periodic pattern of period P makes stereo
        # matching ambiguous at disparity offsets of +-P (aliasing).
        period = int(kind[len("stripes") :] or 8)
        cols = ((torch.arange(w) // max(1, period // 2)) % 2).float()
        p = cols.view(1, 1, w).repeat(3, h, 1)
        return p.to(device), None, 0
    if kind == "noise_indep":
        # Non-realizable with a flat print (needs view-dependent/lenticular media):
        # independent left/right content destroys stereo matching outright.
        a = torch.rand(3, h, w, generator=g).to(device)
        b = torch.rand(3, h, w, generator=g).to(device)
        return a, b, 0
    if kind == "shift":
        # Diagnostic: correct texture, wrong disparity => surface at a false depth.
        return torch.rand(3, h, w, generator=g).to(device), None, shift_px
    if kind == "patch":
        if static is None:
            raise RuntimeError("--patch-image required for content 'patch'")
        p = F.interpolate(
            static.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return p.to(device), None, 0
    raise ValueError(f"unknown content: {kind}")


def cmd_occlude(args: argparse.Namespace) -> int:
    scorer = Scorer(args)
    targets = select_targets(
        load_targets(args.csv),
        args.n_frames,
        args.min_clean_score,
        args.max_depth,
        args.min_face_visible,
    )
    contents = [c.strip() for c in args.contents.split(",") if c.strip()]
    fracs = [float(x) for x in args.area_fracs.split(",") if x.strip()]
    static = read_image_01(args.patch_image) if args.patch_image else None

    print(
        f"occlusion sweep: {len(targets)} frames x {len(contents)} contents x "
        f"{len(fracs)} area fracs | suppress if score < {scorer.thresh}"
    )
    print(f"selected frames: {', '.join(t.frame for t in targets)}\n")

    rows: list[dict] = []
    for t in targets:
        base = scorer.measure(t)
        rows.append(
            dict(frame=t.frame, content="clean", area_frac=0.0, depth_m=t.depth_m, **base)
        )
        print(
            f"{t.frame} z={t.depth_m:5.1f}m face={t.face_w:.0f}x{t.face_h:.0f}px  "
            f"clean={base['score']:.3f} (csv {t.clean_score:.3f})"
        )
        for content in contents:
            line = f"    {content:<12}"
            for frac in fracs:
                x0, y0, w, h = face_rect(t, frac)
                pl, pr, dx = make_content(
                    content, h, w, int(t.frame), scorer.device, static, args.shift_px
                )
                m = scorer.measure(t, patch_l=pl, patch_r=pr, rect=(x0, y0, w, h), right_dx=dx)
                rows.append(
                    dict(
                        frame=t.frame,
                        content=content,
                        area_frac=frac,
                        depth_m=t.depth_m,
                        px=f"{w}x{h}",
                        **m,
                    )
                )
                flag = "*" if m["suppressed"] else " "
                line += f"  {frac:g}:{m['score']:.3f}{flag}"
            print(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["frame", "content", "area_frac", "depth_m", "px", "score", "nms_n", "nms_max", "suppressed"]
    with args.out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})

    print(f"\n=== summary (mean score / suppression rate over {len(targets)} frames) ===")
    clean = [r["score"] for r in rows if r["content"] == "clean"]
    print(f"{'clean':<12} {np.mean(clean):.3f}   supp 0/{len(clean)}")
    for content in contents:
        parts = []
        for frac in fracs:
            sel = [r for r in rows if r["content"] == content and r["area_frac"] == frac]
            ms = float(np.mean([r["score"] for r in sel]))
            ns = sum(r["suppressed"] for r in sel)
            parts.append(f"{frac:g}: {ms:.3f} ({ns}/{len(sel)})")
        print(f"{content:<12} " + "   ".join(parts))
    print(f"\nwrote {args.out}")
    return 0


def overfit_loss(
    outputs: dict,
    logits: torch.Tensor,
    scorer: Scorer,
    t: Target,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    """One step's training objective. Returns (loss, n_anchors_in_loss).

    `prob` delegates to optimize_patch's own gating and LSE so the ablation
    compares against that pipeline's real objective rather than a restatement
    of it. n_anchors_in_loss == 0 means the step has no gradient.
    """
    if args.loss == "logit":
        tau = max(float(args.temperature), 1e-6)
        return tau * torch.logsumexp(logits / tau, dim=0), int(logits.numel())

    matched = matched_car_scores(
        outputs["bbox_cls"],
        scorer.locations_bev,
        t.loc_x,
        t.loc_z,
        scorer.radius,
        num_classes=int(scorer.cfg.num_classes),
        num_angles=int(scorer.cfg.num_angles),
        score_thresh=args.score_thresh,
        max_matches=args.max_matches,
    )
    return lse_loss(matched.scores, args.prob_temperature), matched.n


PROBE_STEPS = (1, 10, 30, 100, 300)


def cmd_overfit(args: argparse.Namespace) -> int:
    tag = f"{args.loss}_{int(round(args.area_frac * 100)):03d}"
    abl = ROOT / "dsgn/datasets/adria/2.training_patch_optimization/ceiling/ablation"
    if args.out is None:
        args.out = abl / f"summary_{tag}.csv"
    if args.traj is None:
        args.traj = abl / f"traj_{tag}.csv"
    if args.save_patches is None:
        args.save_patches = abl / f"patches_{tag}"

    scorer = Scorer(args)
    targets = select_targets(
        load_targets(args.csv),
        args.n_frames,
        args.min_clean_score,
        args.max_depth,
        args.min_face_visible,
    )
    print(
        f"per-frame capacity test: {len(targets)} frames, area_frac={args.area_frac}, "
        f"loss={args.loss}, {args.steps} steps, lr={args.lr} | "
        f"suppress if score < {scorer.thresh}"
    )
    print(f"selected frames: {', '.join(t.frame for t in targets)}\n")

    results: list[dict] = []
    traj: list[dict] = []
    for t in targets:
        x0, y0, w, h = face_rect(t, args.area_frac)
        clean = scorer.measure(t, with_nms=not args.skip_nms)["score"]
        if args.skip_nms:
            torch.cuda.empty_cache()
        # Parameterize at rendered resolution: maximum possible capacity.
        z = torch.zeros(3, h, w, device=scorer.device, requires_grad=True)
        optim = torch.optim.Adam([z], lr=args.lr)
        best = clean
        best_step = 0
        n_frozen = 0
        step_to_thresh: int | None = None
        at_step: dict[int, float] = {}
        for step in range(1, args.steps + 1):
            optim.zero_grad(set_to_none=True)
            logits, outputs, _, _ = scorer.forward(
                t, patch_l=torch.sigmoid(z), rect=(x0, y0, w, h)
            )
            loss, n_in_loss = overfit_loss(outputs, logits, scorer, t, args)
            # Mirror optimize_patch: an empty match set contributes no gradient.
            if n_in_loss == 0:
                n_frozen += 1
            else:
                loss.backward()
                optim.step()
            cur = float(logits.sigmoid().max().item())
            if cur < best:
                best, best_step = cur, step
            if step_to_thresh is None and cur < scorer.thresh:
                step_to_thresh = step
            if step in PROBE_STEPS:
                at_step[step] = cur
            traj.append(
                dict(
                    loss=args.loss,
                    area_frac=args.area_frac,
                    frame=t.frame,
                    step=step,
                    score=round(cur, 6),
                    obj=round(float(loss.item()), 6),
                    n_in_loss=n_in_loss,
                )
            )
            if step % args.log_every == 0 or step == 1:
                print(
                    f"  {t.frame} step {step:4d}/{args.steps}  score={cur:.4f}  "
                    f"best={best:.4f}  loss={float(loss.item()):.3f}  "
                    f"n_loss={n_in_loss} frozen={n_frozen}"
                )
        final = scorer.measure(
            t,
            with_nms=not args.skip_nms,
            patch_l=torch.sigmoid(z).detach(),
            rect=(x0, y0, w, h),
        )
        ok = best < scorer.thresh
        print(
            f"{t.frame} z={t.depth_m:5.1f}m patch={w}x{h}px  clean={clean:.3f} → "
            f"best={best:.3f} (step {best_step}) final={final['score']:.3f}  "
            f"frozen={n_frozen}/{args.steps}  "
            f"{'SUPPRESSED' if ok else 'survives'}\n"
        )
        results.append(
            dict(
                loss=args.loss,
                area_frac=args.area_frac,
                frame=t.frame,
                depth_m=round(t.depth_m, 2),
                px=f"{w}x{h}",
                clean=round(clean, 4),
                best=round(best, 4),
                best_step=best_step,
                final=round(final["score"], 4),
                step_to_thresh=step_to_thresh if step_to_thresh is not None else "",
                frozen_steps=n_frozen,
                suppressed=int(ok),
                **{f"s{k}": round(at_step.get(k, float("nan")), 4) for k in PROBE_STEPS},
            )
        )
        if args.save_patches:
            args.save_patches.mkdir(parents=True, exist_ok=True)
            arr = (
                torch.sigmoid(z).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
            ).round().astype(np.uint8)
            Image.fromarray(arr).save(args.save_patches / f"{t.frame}_overfit.png")
        del z, optim
        torch.cuda.empty_cache()

    write_rows(args.out, results)
    if args.traj:
        write_rows(args.traj, traj)

    n_ok = sum(r["suppressed"] for r in results)
    print(f"=== per-frame capacity summary (loss={args.loss}, area={args.area_frac}) ===")
    for r in results:
        reached = r["step_to_thresh"]
        print(
            f"{r['frame']}  z={r['depth_m']:5.1f}m  patch={r['px']:>9}  "
            f"clean={r['clean']:.3f}  best={r['best']:.3f}  "
            f"step<thr={reached if reached != '' else '-':>4}  "
            f"frozen={r['frozen_steps']:>3}  "
            f"{'SUPPRESSED' if r['suppressed'] else 'survives'}"
        )
    mean_best = float(np.mean([r["best"] for r in results]))
    reached_all = [int(r["step_to_thresh"]) for r in results if r["step_to_thresh"] != ""]
    print(
        f"\n{n_ok}/{len(results)} frames suppressed with a dedicated patch at "
        f"area_frac={args.area_frac} using loss={args.loss}."
    )
    print(f"mean best score = {mean_best:.4f}   total frozen steps = "
          f"{sum(r['frozen_steps'] for r in results)}/{len(results) * args.steps}")
    if reached_all:
        print(f"median steps to cross {scorer.thresh}: {int(np.median(reached_all))} "
              f"(over the {len(reached_all)} frames that crossed)")
    if n_ok == 0:
        print("→ Patch area is the binding constraint; a universal patch cannot do better.")
    print(f"\nwrote {args.out}")
    return 0


def _fmt_median_steps(rows: list[dict]) -> str:
    crossed = [int(r["step_to_thresh"]) for r in rows if r["step_to_thresh"]]
    if not crossed:
        return "never"
    return f"{int(np.median(crossed))} ({len(crossed)}/{len(rows)})"


def cmd_compare(args: argparse.Namespace) -> int:
    """Tabulate the loss x coverage arms written by the overfit subcommand."""
    arms: dict[tuple[str, float], list[dict]] = {}
    for f in sorted(args.dir.glob("summary_*.csv")):
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            arms[(rows[0]["loss"], float(rows[0]["area_frac"]))] = rows

    if not arms:
        raise RuntimeError(f"no summary_*.csv in {args.dir}")

    # Fixed by the ablation design, so a half-finished grid is reported as such
    # instead of looking complete.
    areas = sorted({k[1] for k in arms})
    expected = [(loss, a) for a in areas for loss in ("logit", "prob")]
    missing = [k for k in expected if k not in arms]

    print(f"=== loss x coverage ablation ({args.dir}) ===\n")
    hdr = (
        f"{'loss':<6} {'area':>5} {'suppr':>7} {'mean_best':>10} {'median_step<thr':>16} "
        f"{'frozen':>8} {'s10':>7} {'s30':>7} {'s100':>7} {'s300':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for loss, area in expected:
        if (loss, area) not in arms:
            print(f"{loss:<6} {area:>5} {'MISSING — not yet run':>7}")
            continue
        rows = arms[(loss, area)]
        n = len(rows)
        n_ok = sum(int(r["suppressed"]) for r in rows)
        mean_best = float(np.mean([float(r["best"]) for r in rows]))
        frozen = sum(int(r["frozen_steps"]) for r in rows)
        means = {
            k: float(np.mean([float(r[f"s{k}"]) for r in rows if r[f"s{k}"] != "nan"]))
            for k in PROBE_STEPS
        }
        print(
            f"{loss:<6} {area:>5.2f} {f'{n_ok}/{n}':>7} {mean_best:>10.4f} "
            f"{_fmt_median_steps(rows):>16} {frozen:>8} "
            f"{means[10]:>7.3f} {means[30]:>7.3f} {means[100]:>7.3f} {means[300]:>7.3f}"
        )

    print("\n=== per-frame best score ===")
    cols = [k for k in expected if k in arms]
    frames = sorted({r["frame"] for rows in arms.values() for r in rows})
    head = f"{'frame':<8}{'z(m)':>7}  " + "  ".join(f"{l[:5]}@{a:.2f}" for l, a in cols)
    print(head)
    print("-" * len(head))
    for fr in frames:
        depth = ""
        cells = []
        for key in cols:
            row = next((r for r in arms[key] if r["frame"] == fr), None)
            depth = depth or (row["depth_m"] if row else "")
            cells.append(f"{float(row['best']):.3f}" if row else "  -  ")
        print(f"{fr:<8}{float(depth):>7.1f}  " + "  ".join(f"{c:>10}" for c in cells))

    if missing:
        print(
            "\nWARNING: incomplete grid, missing "
            + ", ".join(f"{l}@{a:.2f}" for l, a in missing)
            + " — do not draw conclusions about that factor yet."
        )
    return 0


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--images", type=Path, default=ROOT / "dsgn/datasets/arka/dsgn_awsim/training")
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/2.training_patch_optimization/patches_localized.csv",
    )
    p.add_argument("--cfg", type=Path, default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/save_config_awsim.py")
    p.add_argument("--loadmodel", type=Path, default=ROOT / "dsgn/checkpoints/kitti/dsgn_12g_b/finetune_48.tar")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--match-radius", type=float, default=2.0)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--min-clean-score", type=float, default=0.8, help="only attack confident targets")
    p.add_argument("--max-depth", type=float, default=20.0, help="metres; AEB-relevant range")
    p.add_argument(
        "--min-face-visible",
        type=float,
        default=0.9,
        help="drop frames whose projected rear face is not this fraction on-image",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("occlude", help="forward-only coverage sweep")
    add_common(o)
    o.add_argument("--area-fracs", type=str, default="0.23,0.5,0.75,1.0")
    # NB: 'black' is consistently the *weakest* content (a dark region still reads
    # as plausible car body); gray/white/noise are far stronger. Don't use black
    # as an upper bound.
    o.add_argument(
        "--contents", type=str, default="gray,noise,stripes8,stripes16,noise_indep,shift"
    )
    o.add_argument("--patch-image", type=Path, default=None, help="for content 'patch'")
    o.add_argument(
        "--shift-px",
        type=int,
        default=12,
        help="disparity error injected by content 'shift' (full-res px)",
    )
    o.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/2.training_patch_optimization/ceiling/occlusion_sweep.csv",
    )
    o.set_defaults(func=cmd_occlude)

    f = sub.add_parser("overfit", help="per-frame dedicated-patch capacity test")
    add_common(f)
    f.set_defaults(n_frames=5)
    f.add_argument("--area-frac", type=float, default=0.75)
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--lr", type=float, default=5e-2)
    f.add_argument(
        "--loss",
        choices=("logit", "prob"),
        default="logit",
        help="training objective; 'prob' is optimize_patch.py's gated top-k "
        "probability LSE, imported from that module",
    )
    f.add_argument("--temperature", type=float, default=1.0, help="LSE temperature in logit space")
    f.add_argument(
        "--prob-temperature",
        type=float,
        default=0.2,
        help="LSE temperature for --loss prob (optimize_patch default)",
    )
    f.add_argument(
        "--score-thresh",
        type=float,
        default=0.33,
        help="proposal gate for --loss prob (optimize_patch default)",
    )
    f.add_argument(
        "--max-matches",
        type=int,
        default=3,
        help="top-k proposals kept for --loss prob (optimize_patch default)",
    )
    f.add_argument("--log-every", type=int, default=25)
    f.add_argument(
        "--skip-nms",
        action="store_true",
        help="skip post-NMS counting to cut peak GPU memory; does not change "
        "the reported score, which is read from bbox_cls",
    )
    f.add_argument("--out", type=Path, default=None, help="per-frame summary CSV")
    f.add_argument("--traj", type=Path, default=None, help="per-step trajectory CSV")
    f.add_argument("--save-patches", type=Path, default=None)
    f.set_defaults(func=cmd_overfit)

    c = sub.add_parser("compare", help="tabulate the loss x coverage ablation arms")
    c.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "dsgn/datasets/adria/2.training_patch_optimization/ceiling/ablation",
    )
    c.set_defaults(func=cmd_compare)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
