#!/usr/bin/env python3
"""Preview DSGN2 AWSIM crop — run on host."""
from pathlib import Path
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

IMG = Path("~/summer26/dsgn/datasets/arka/dsgn_awsim/testing_offline/image_2/000099.png").expanduser()
MIN_REL_X, MIN_REL_Y = 0, -0.074
MAX_CROP_W, MAX_CROP_H = 1248, 320

img = io.imread(IMG)
H, W = img.shape[:2]
crop_rel_x = MIN_REL_X / 2 + 0.5
crop_rel_y = MIN_REL_Y / 2 + 0.5
crop_w = min(MAX_CROP_W, W)
crop_h = min(MAX_CROP_H, H)
x1 = int((W - crop_w) * crop_rel_x)
y1 = int((H - crop_h) * crop_rel_y)
crop = img[y1:y1+crop_h, x1:x1+crop_w]

out = Path("~/summer26/dsgn/detections/dsgn2/crop_preview").expanduser()
out.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(img)
axes[0].add_patch(Rectangle((x1, y1), crop_w, crop_h, fill=False, edgecolor="lime", lw=2))
axes[0].set_title(f"Full {W}×{H} + crop box")
axes[1].imshow(crop)
axes[1].set_title(f"Network input {crop.shape[1]}×{crop.shape[0]}")
plt.savefig(out / "000099_crop_compare.png", dpi=150, bbox_inches="tight")
io.imsave(str(out / "000099_crop_only.png"), crop)
print(f"Saved to {out}")