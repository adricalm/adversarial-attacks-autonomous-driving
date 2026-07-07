Yes. I compared the **actual repository histories and code**, not only the READMEs.

## Main conclusion

**Arka did not redesign DSGN.** The neural-network architecture, losses, 3D detection head, and general training logic are essentially still the official DSGN.

His fork is better understood as:

> **Official DSGN + an AWSIM compatibility layer + environment fixes + a collection of calibration/debugging/offline-inference scripts.**

The comparison is unusually clean: the custom repository is based directly on the official master tip and adds **9 commits**. I found **29 changed files: 16 added and 13 modified**. The official tip is `ac693e...`; Arka's current custom tip is `971e90...`.

The project documentation confirms the intended role: train/infer DSGN on AWSIM stereo data, save the predicted detections, and later replay those precomputed outputs into Autoware because DSGN was too slow for the original real-time setup. 

---

# The high-level picture

The official repository expects:

```text
KITTI stereo images
+ KITTI calibration
+ KITTI LiDAR-derived depth
+ KITTI object labels
        ↓
      DSGN
        ↓
KITTI-format 3D detections
```

Arka adapted that into:

```text
AWSIM stereo images
+ AWSIM LiDAR
+ AWSIM object labels
+ AWSIM sensor calibration
        ↓
Convert/organize everything to look like KITTI
        ↓
Modified DSGN
        ↓
KITTI-format 3D detection .txt files
        ↓
dsgn_offline ROS 2 node
        ↓
Autoware perception pipeline
```

This is the central design decision you need to understand:

> **He did not teach DSGN a completely new data format. He made AWSIM pretend to be KITTI.**

That explains most of the repository.

---

# 1. Core change: adapt the data pipeline from KITTI to AWSIM

This is the most important category.

## `dsgn/dataloader/kitti_dataset.py`

Official:

```python
data/kitti
```

Arka:

```python
data/awsim
```

He even left the comment:

```python
# Arka change kitti to awsim dataset
```

Everything else continues using the existing KITTI dataset classes and directory conventions.

So AWSIM data was expected to look approximately like:

```text
data/awsim/
├── train.txt
├── val.txt
├── test.txt
├── training/
│   ├── image_2/
│   ├── image_3/
│   ├── calib/
│   ├── label_2/
│   ├── velodyne/
│   └── depth/
└── testing/
```

That is still fundamentally **KITTI's interface**.

---

## `dsgn/dataloader/KITTILoader3D.py`

A tiny but important label change:

```text
Official: DontCare
Custom:   Other
```

The original loader maps KITTI's `DontCare` label to class 4. Arka changed it to recognize AWSIM's `Other` instead.

### My interpretation

The generated AWSIM labels apparently used:

```text
Car
Other
```

rather than KITTI's:

```text
Car
DontCare
```

### Weakness

This unnecessarily breaks backward compatibility with KITTI. A cleaner implementation would accept both:

```python
elif label.type in {"DontCare", "Other"}:
    typ = 4
```

That is the kind of cleanup I would do before building more work on top.

---

# 2. The largest functional change: AWSIM images are downsampled

## `dsgn/dataloader/KITTILoader_dataset3d.py`

The original loader processes KITTI images and pads them to the fixed KITTI dimensions:

```text
384 × 1248
```

Arka added:

```python
downscale_factor = 0.5
```

and resizes both stereo images:

```text
1920 × 1080
        ↓ 0.5
960 × 540
```

He also resizes the depth/disparity map, then pads the tensors to dimensions divisible by 32 instead of forcing KITTI's fixed size.

For AWSIM, this gives approximately:

```text
Raw image:       1080 × 1920
After resize:     540 × 960
After padding:    544 × 960
```

### Why?

Almost certainly GPU memory and computational cost.

DSGN builds expensive 3D volumes. The official README itself warns about significant memory requirements. Arka's trained model is even named:

```text
dsgn_12g_awsim_remote_downsample
```

So downsampling was clearly a central part of getting the AWSIM version to run.

---

# 3. New AWSIM configuration

Arka added:

```text
configs/config_car_12g_awsim.py
```

It is almost identical to the official:

```text
configs/config_car_12g.py
```

The important changes are:

| Parameter     |   Official |        Arka |
| ------------- | ---------: | ----------: |
| `btrain`      |          8 |           1 |
| `input_size`  | 384 × 1248 | 1080 × 1920 |
| `output_size` |   96 × 312 |   270 × 480 |

The rest of the model configuration is essentially unchanged.

This reinforces the main conclusion:

> **Same DSGN model; different input geometry and memory assumptions.**

### An initially confusing detail

You may ask:

> Why does the config say `1080 × 1920` when the actual images are downsampled to `540 × 960`?

I do **not** think this is automatically a bug.

The calibration matrices appear to remain expressed in original-image pixel coordinates. DSGN projects 3D coordinates into image space and then normalizes those pixel coordinates to `[-1, 1]` before `grid_sample`. Keeping the original resolution in the geometry configuration can therefore preserve the correct *relative* position even though the actual image tensor was halved.

However, this needs to be documented and tested. Right now it is implicit.

---

# 4. Training defaults changed from KITTI to AWSIM

In `tools/train_net.py`, Arka changed only the defaults:

```text
Official:
data_path  = ./data/kitti/training/
split_file = ./data/kitti/train.txt

Custom:
data_path  = ./data/awsim/training/
split_file = ./data/awsim/trainval.txt
```

Again: **no new optimizer, loss or training algorithm**.

---

# 5. AWSIM LiDAR → depth-map preprocessing

## `preprocessing/generate_disp.py`

This was modified quite significantly.

The official code filters LiDAR points with:

```python
pc_velo[:, 0] > 2
```

which follows KITTI's convention that the positive `x` direction points forward.

Arka changed it to:

```python
pc_velo[:, 1] < -2
```

and added several projection/debugging utilities.

### What that means

The AWSIM-exported LiDAR coordinate convention was different from KITTI's. Arka therefore had to determine:

```text
AWSIM LiDAR coordinates
        ↓ transformation
camera coordinates
        ↓ projection
image pixels
```

A large percentage of his added tools are experiments for solving precisely this problem.

---

# 6. A lot of the new repository is really calibration debugging

These added scripts are mostly not DSGN itself:

```text
tools/calib_copies.py
tools/convert_bin_to_ply.py
tools/find_transformation_matrix.py
tools/plot_BB3D.py
tools/plot_BB3D_awsim.py
tools/plot_BB3D_awsim_lidar_proj.py
tools/plot_BB3D_awsim_lidar_proj_st_left.py
tools/plot_BB3D_awsim_lidar_proj_st_left_GT.py
tools/plot_BB3D_awsim_lidar_proj_st_right.py
tools/show_P_matrix_Lidar.py
tools/show_bin_open3d.py
tools/show_depth.py
```

Their purpose is largely:

```text
"Where exactly is this LiDAR point?"
"Is my transformation correct?"
"Does P2 project the box correctly?"
"Are left and right cameras calibrated correctly?"
"Does the predicted 3D box line up with the actual car?"
```

For example, `show_P_matrix_Lidar.py` contains hard-coded projection matrices, rotation values and translations for manually checking the LiDAR-camera alignment.

`find_transformation_matrix.py` converts a hard-coded quaternion and translation into a 4×4 transform.

`calib_copies.py` simply copies one calibration file hundreds of times, supporting the assumption that the simulated stereo/LiDAR sensor rig had constant calibration across the recorded frames.

### Important conclusion

These are mostly **research scratch tools**, not production pipeline components.

You should not think:

> "All files in DSGN_custom are necessary to run Arka's final DSGN."

They are not.

---

# 7. CUDA/PyTorch compatibility fixes

Five low-level CUDA files were modified:

```text
BuildCostVolume_cuda.cu
ROIAlign_cuda.cu
ROIPool_cuda.cu
SigmoidFocalLoss_cuda.cu
nms.cu
```

The changes are mostly mechanical modernization:

```text
THCCeilDiv(...)       → manual integer ceiling division
THCudaCheck(...)      → cudaGetLastError()
THCudaMalloc(...)     → cudaMalloc(...)
THCudaFree(...)       → cudaFree(...)
```

For example, this happens directly in the cost-volume kernel.

And `nms.cu` replaces old THC memory APIs with regular CUDA APIs.

### Interpretation

This is not related to AWSIM or adversarial attacks.

The official DSGN code was written for:

```text
PyTorch 1.1–1.3
Ubuntu 16.04
```

Arka needed it to compile in a newer environment.

### My criticism

The fix is functional but not clean modern PyTorch/CUDA code. In several places an error is merely printed rather than raised.

I would **not extend this style**. When we modernize the project, these should eventually use current PyTorch/CUDA error macros rather than accumulating more raw compatibility patches.

---

# 8. One model-code modification — but not an architecture change

The only changed core model file is:

```text
dsgn/models/stereonet.py
```

The modification moves tensors onto the same device before doing coordinate normalization:

```python
self.coord_rect = self.coord_rect.to(coord_img.device)
```

and constructs the normalization tensors on that device.

This fixes a CPU/GPU device mismatch.

Again:

> **It does not change how DSGN fundamentally works.**

No new layer. No changed backbone. No new detection objective. No attack logic.

---

# 9. Offline inference support

Arka added:

```text
tools/test_no_eval.py
dsgn/dataloader/KITTILoader_dataset3d_inference.py
```

The intended idea is sensible:

```text
stereo images
     ↓
DSGN inference
     ↓
do not require GT labels / depth evaluation
     ↓
write KITTI-style detection .txt files
```

`test_no_eval.py` directly writes the predicted class, 2D box, 3D dimensions, center, orientation and score to text files.

Those files fit the documented offline architecture: precompute DSGN outputs and later feed them to Autoware through `dsgn_offline`. The plotting script even hard-codes the offline ROS package's output directory.

---

# The important part: I found probable broken or abandoned code

This repo is a successful research prototype, but you should **not assume every added file is part of the final working path**.

## Confirmed problem 1: the inference loader contains an undefined variable

It defines:

```python
disp_L_path = ...
```

but later executes:

```python
dataL = self.dploader(disp_L)
```

`disp_L` was commented out and does not exist.

So, when the disparity file exists, that branch should fail.

---

## Confirmed problem 2: `test_no_eval.py` does not use the new inference loader

It imports:

```python
from dsgn.dataloader import KITTILoader_dataset3d as DA
```

not:

```text
KITTILoader_dataset3d_inference
```

Yet its comment claims it is skipping disparity/depth ground truth.

That means these two additions look like **an unfinished attempt to create clean inference without GT depth**.

This is very important for us: we should identify the exact command that produced the final 212 outputs rather than assume `test_no_eval.py` is the canonical path.

---

## Confirmed problem 3: `generate_disp.py` says “Skip” but does not skip

Arka changed:

```python
if predix not in file_names:
    print('Skip {}'.format(predix))
    # continue
```

The `continue` is commented out, so it processes the file anyway.

That looks like leftover debugging.

---

# The biggest things I would verify before modifying the model

## 1. Are the 2D labels already downsampled?

The loader resizes the images by 0.5, but I do not see explicit:

```python
boxes *= 0.5
```

before the 2D boxes are used.

There are two possibilities:

1. Arka generated AWSIM labels already at half resolution → fine.
2. Labels use 1920×1080 coordinates → there is a geometry inconsistency.

We need to check one actual image + one label file to know.

---

## 2. What exactly is the final working inference command?

The README says to use normal:

```text
tools/test_net.py
```

while the repo also contains:

```text
tools/test_no_eval.py
```

and a seemingly unfinished:

```text
KITTILoader_dataset3d_inference.py
```

The final pipeline clearly worked, because the project has the trained model and precomputed outputs, but the repository contains several experimental routes. The documentation itself warns that paths were sometimes hard-coded. 

We should reconstruct the **actual successful route**, not just choose the newest-looking file.

---

## 3. The train/validation split may be weak

`generate_train_txt.py` does:

```python
train_ids = train_ids_all[:272]
val_ids = train_ids_all[272:]
```

If these are consecutive video frames, this is not a strong experimental split. Adjacent frames are extremely correlated.

For your future attack experiments, we should split by:

```text
scenario / trajectory / recording sequence
```

rather than simply by frame number.

---

# My final classification of all changes

| Category                              | Importance | What Arka changed                        |
| ------------------------------------- | ---------- | ---------------------------------------- |
| **DSGN architecture**                 | Very high  | Essentially nothing                      |
| **AWSIM dataset compatibility**       | Very high  | Main custom work                         |
| **Image resolution/memory**           | Very high  | 0.5 downsampling + dynamic padding       |
| **Calibration and coordinate frames** | Very high  | Extensive debugging/adaptation           |
| **Training algorithm**                | Low        | Only paths/defaults changed              |
| **Offline inference**                 | High       | Added output-generation experiments      |
| **Autoware integration**              | High       | Outside this repo, mainly `dsgn_offline` |
| **CUDA compatibility**                | Medium     | Modernized obsolete APIs                 |
| **Debugging utilities**               | Medium     | Many hard-coded exploratory scripts      |
| **Adversarial attacks**               | None       | Not implemented                          |

## The single most important takeaway

**DSGN_custom is not a new DSGN. It is an old DSGN wrapped in enough AWSIM-specific assumptions to train on Arka's simulated stereo dataset and emit detections that Autoware can consume.**

That is good news for your next work. For adding patches, we almost certainly **do not need to touch the CUDA kernels or the DSGN architecture**. The clean insertion point is around:

```text
load stereo pair
        ↓
apply physically consistent patch to left/right images
        ↓
resize / normalize
        ↓
DSGN
```

Before modifying anything, though, I would reconstruct the exact current clean inference path and validate the resolution/calibration conventions on one sample. That will stop us from building an attack pipeline on top of accidental prototype behavior.
