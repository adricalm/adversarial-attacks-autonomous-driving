# Adding a stereo camera to AWSIM (binary-only) — journey log

Goal: record new KITTI-style stereo datasets like Arka did, but **without AWSIM Unity
sources** — we only have the prebuilt `awsim_labs_v1.6.1` binary.

Status: **the stereo camera works.** The modded build publishes a validated, rectified
0.54 m stereo pair (median |dy| = 0.00 px, positive disparity, sane depths) at ~6 Hz,
with the left camera's intrinsics unchanged so Arka's calib files still apply. No AWSIM
file is rewritten and `extracted/` is untouched, so the existing pipeline is unaffected.
Not yet done: recording a dataset, and a joint run with Autoware under GPU contention.

Related: [`DSGN_AWSIM_FINDINGS.md`](DSGN_AWSIM_FINDINGS.md), [`autoware-awsim-startup.md`](autoware-awsim-startup.md).

---

## The core insight

Arka's recorder ([AWSIM_to_KITTI](https://github.com/DF-Autoware-AWSIM/AWSIM_to_KITTI/tree/only_kitti_style))
needs four topics. Three of them the **stock binary already publishes**:

| Recorder needs | Stock binary | Gap |
|---|---|---|
| `/sensing/camera_left/traffic_light/image_raw` | `/sensing/camera/traffic_light/image_raw` | name only |
| `/sensing/camera_right/traffic_light/image_raw` | — | **missing** |
| `/sensing/lidar/top/pointcloud_raw` | same | none |
| `/perception/.../centerpoint/objects` | from Autoware | none |

So the entire gap is **one extra camera**. This is much smaller than "rebuild AWSIM in
Unity", which is what a first-pass plan assumed.

---

## Measured facts (not assumptions)

### Arka's calibration is fully decoded

All 213 calib files in `dsgn/datasets/arka/dsgn_awsim/testing_offline/calib/` are
byte-identical (1 unique md5). Decoded:

| Quantity | Value |
|---|---|
| Resolution | 1920 x 1080 |
| fx, fy | 960.0, 959.3908081054688 |
| cx, cy | 960.5, 540.5 |
| `P0` Tx | 0 (reference camera) |
| `P2` Tx | +259.2000122 |
| `P3` Tx | −259.2000122 |

KITTI camera centre is `-Tx/fx`, so cam2 sits at **x = −0.27 m**, cam3 at **+0.27 m** →
**baseline 0.54 m**, centred on the reference camera.

### The stock camera is that reference camera

Live from the running stock binary:

```
/sensing/camera/traffic_light/camera_info
  height 1080, width 1920
  k = [960.0, 0, 960.5,  0, 959.3908081054688, 540.5,  0, 0, 1]
  d = [0,0,0,0,0]        # no distortion
  r = identity
  frame_id = traffic_light_left_camera/camera_optical_link
```

This matches Arka's `P0` **to the last digit of fy**. Conclusion: Arka never touched the
camera intrinsics or pose. He centred a stereo rig on the stock camera. Therefore:

- Clone the stock camera, offset the copies by ∓0.27 m in local X. Nothing else changes.
- Arka's calib files can be reused verbatim.
- `d = 0` and `r = identity` mean a clone yields a **perfectly rectified, parallel** pair
  by construction — this is the alignment work that would otherwise be hand-tuned in the
  Unity Editor.

Visual cross-check: a frame grabbed from the stock camera and Arka's
`testing_offline/image_2/000000.png` show the same intersection with identical hood
position, horizon and FOV. Saved under `multimedia/awsim_step1/`.

### The LiDAR was never modified

Earlier notes/advice claimed Arka switched to a 128-channel LiDAR. **This is wrong for
this dataset.** Measured from `testing_offline/velodyne/000000.bin`: 26,996 points with
16 distinct elevation rings at exactly −15° … +15° in 2° steps — textbook VLP-16, i.e.
the stock `SensorVLP16`. The live binary reports width 27,509, height 1. No LiDAR work
needed.

### The binary is moddable without Unity

| Property | Value | Why it matters |
|---|---|---|
| Unity | 2022.3.62f1, URP, x64 | — |
| Scripting backend | **Mono** (`MonoBleedingEdge/`, `Assembly-CSharp.dll` + `.pdb`) | C# is readable/patchable; IL2CPP would have blocked this |
| Scene container | `awsim_labs_Data/data.unity3d` | readable with UnityPy (41,827 objects) |
| Source available | public at tag `v1.6.1` | exact-version source to read alongside the DLL |

`AWSIM.CameraSensorHolder` already holds a `List<CameraSensor>` plus `publishHz` and
`renderInQueue`. **Multi-camera is a native feature** — nothing needs inventing. Setting
`renderInQueue = false` renders all cameras on the *same* frame, which is required for a
valid stereo pair (with `true`, cameras render on consecutive frames and ego motion
corrupts disparity).

The scene contains exactly one sensor camera per ego variant, on a GameObject named
`CameraObject` under `traffic_light_left_camera/camera_link`.

---

## Step 1 findings (things that cost time)

### `--config` auto-load races ros2-for-unity init (the one real gotcha)

Without a config file the binary shows a launcher window (Map / Ego / Position / `Load`
button) and publishes nothing until `Load` is clicked. `Loader.cs` accepts
`--config <path>` and then auto-loads with no clicking, which is what we want for
scripted/reproducible runs.

But auto-loading exposes a **startup race**. If the scene loads before
`ros2-for-unity` has finished initialising, every C# publisher fails at creation:

```
RuntimeError: topic name is invalid, at ./src/rcl/expand_topic_name.c:73
  at AWSIM.ClockPublisher.Awake ()
NullReferenceException  (in every sensor's FixedUpdate, because the publisher is null)
```

Symptom: only 17 topics — no `/clock`, no camera, no vehicle status. Misleading detail:
the **LiDAR topics still appear**, because RGL publishes through its own native library
rather than `ros2cs`. So it looks half-working rather than broken.

Measured matrix (all with the map + ego identical):

| `--config` | `ROS_DISTRO` present | Result |
|---|---|---|
| yes | yes | **fails** — 17 topics, no `/clock` |
| yes | scrubbed | works — 31 topics |
| no (GUI `Load`) | yes | works — 32 topics |

Interpretation: AWSIM's `ros2-for-unity` is a *standalone* build bundling its own ROS 2
libraries. When it detects a sourced ROS 2 it logs `You should not source ROS2 in
'ros2-for-unity' standalone build` and takes a slower init path — slow enough that
auto-load overtakes it. The human delay of clicking `Load` always wins the race, which
is why **the pre-existing manual recipe was never broken.** The `ROS_DISTRO` detection
trigger is baked into the Autoware image as a Docker ENV, so it is present even in a
non-login shell and even if you never source anything.

Verified-good combination for scripted launches (used by `scripts/awsim/awsim_launch.sh`):

```bash
env -u ROS_DISTRO -u AMENT_PREFIX_PATH -u CYCLONEDDS_URI -u COLCON_PREFIX_PATH \
    ROS_DOMAIN_ID=26 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    awsim_labs.x86_64 --config <path> ...
```

Because this is a race and not a hard guarantee, **always run `scripts/awsim/awsim_verify.sh`
after launching** rather than assuming a successful start.

Sourcing ROS 2 to *inspect* the graph from another process in the same container is
always fine — the restriction applies only to the AWSIM process itself.

### AWSIM cannot run on the host

A host-native run (June 2026, `Player.log`) died with `UnsatisfiedLinkError: librcl.so`.
It must run inside the Autoware container. Rendering is genuinely GPU-accelerated over
xrdp: Vulkan enumerates both devices and selects `"NVIDIA L40S"` over `llvmpipe`.

### Rates are scaled by timeScale

`config.json` sets `timeScale: 0.6`, and `CameraSensorHolder.publishHz` defaults to 10,
so camera and LiDAR publish at **~6 Hz**, not 10. `/clock` runs ~99 Hz.

---

## Safety / rollback

The working setup is never modified. Layout under `data/awsim/`:

| Path | Role |
|---|---|
| `awsim_labs_v1.6.1.zip` | pristine release, `unzip -t` verified clean |
| `extracted/awsim_labs_v1.6.1/` | **known-good build — do not modify** |
| `modded/awsim_labs_v1.6.1/` | working copy for the stereo mod |
| `PRISTINE_CHECKSUMS.md5` | md5 of 955 binaries/DLLs/scenes in `extracted/` |

Re-verify the good build at any time:

```bash
cd ~/summer26/data/awsim && md5sum -c --quiet PRISTINE_CHECKSUMS.md5
```

**Footgun:** the old launch command located the binary with
`find extracted -type f -name "*.x86_64" | head -1`. A second copy placed *inside*
`extracted/` would be silently picked up. That is why `modded/` is a sibling, and why
`scripts/awsim/awsim_launch.sh` uses explicit paths instead of `find`.

The stereo mod as shipped is **additive and confined to `modded/`**. It never rewrites
`awsim_labs.x86_64`, `UnityPlayer.so` or `Assembly-CSharp.dll`. It only:

- adds `awsim_labs_Data/Managed/StereoMod.dll` (a new file), and
- appends one entry each to `ScriptingAssemblies.json` and
  `RuntimeInitializeOnLoads.json`, both backed up to `*.json.orig` on first install.

So there are three independent levels of undo:

```bash
scripts/awsim/awsim_stereo_install.py --uninstall   # modded/ back to stock behaviour
rm -rf data/awsim/modded && cp -a data/awsim/extracted data/awsim/modded   # fresh copy
unzip data/awsim/awsim_labs_v1.6.1.zip        # ultimate: from the release archive
```

`extracted/` is untouched throughout, so **the pipeline you run today is unaffected
either way** — it launches from `extracted/`, which still passes `md5sum -c`.

**Do not rename the left camera topic in the simulator.** Autoware's traffic-light
recognition subscribes to `/sensing/camera/traffic_light/image_raw`; renaming it (as
Arka apparently did) would quietly break traffic-light detection in the existing
pipeline. Point the *recorder* at the stock name instead and add only `camera_right`.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/awsim/awsim_launch.sh [pristine\|modded]` | Launch AWSIM in Docker with the correct scrubbed env |
| `scripts/awsim/awsim_verify.sh [container]` | Check required topics, rates, and live camera geometry |
| `scripts/awsim/awsim_stereo_build.sh` | Compile `StereoMod.dll` in a throwaway Mono container |
| `scripts/awsim/awsim_stereo_install.py [--uninstall]` | Register/unregister the mod with Unity's loader |
| `scripts/awsim/awsim_stereo_check.py` | Quantitative stereo validation (run inside the container) |

---

## Step 3: the mod — DONE, and it works

### BepInEx does not work on this build (dead end, ~1 h)

Doorstop injected fine (`libdoorstop.so` mapped, `BepInEx.Preloader` reached), but the
preloader died immediately:

```
System.MissingMethodException: Method not found: !0 System.Linq.IGrouping`2.get_Key()
  at BepInEx.Configuration.ConfigFile..ctor
```

AWSIM was built with Unity **managed-code stripping** enabled, so the shipped
`System.Core.dll` has had unused LINQ members removed. BepInEx's config system needs
them. The documented fix is to supply unstripped Unity corlibs, which we do not have
without a Unity install. Abandoned — see below for the better route.

Two smaller traps found on the way, worth knowing if BepInEx is ever revisited:
`run_bepinex.sh` loses its executable bit when unzipped, and it shells out to `file`,
which is **not installed in the Autoware image**.

### What actually worked: Unity's own startup hook

Unity ships two plain-JSON registries next to the game data, and honours them at boot:

| File | Meaning |
|---|---|
| `awsim_labs_Data/ScriptingAssemblies.json` | assemblies to load (`16` = user assembly) |
| `awsim_labs_Data/RuntimeInitializeOnLoads.json` | static methods to call at startup |

The second is how AWSIM itself bootstraps: it already contains an entry calling
`AWSIM.SimulatorROS2Node.Initialize`. So we add our own assembly and our own entry, and
Unity calls us on the main thread with no injection at all. `loadTypes: 0` is
`RuntimeInitializeLoadType.AfterSceneLoad`.

This is strictly better than BepInEx or Cecil IL patching: no `LD_PRELOAD`, no
decompiling, and **no existing game file is rewritten** — we only append to two text
files and drop one new DLL in `Managed/`.

### The mod itself (`src/awsim_stereo_mod/StereoMod.cs`)

`StereoRig` polls once a second (the ego is spawned well after the scene loads, and F12
reloads it) and on finding a `CameraSensorHolder` with exactly one sensor:

- sets `renderInQueue = false` so **both cameras render on the same frame** — with the
  default queue they render on consecutive frames and ego motion corrupts disparity;
- **clones** the existing `CameraSensor` GameObject, which is what makes the pair
  rectified by construction: rotation, FOV, resolution and clipping planes are inherited
  exactly rather than re-entered by hand;
- clones it under a temporary **inactive** parent, so `Awake()`/`Start()` are deferred
  until after the clone's ROS topics have been rewritten (`CameraRos2Publisher.Awake()`
  creates the publishers from those fields, so ordering matters). This also avoids
  toggling the original — `UICameraBridge` has `OnEnable`/`OnDisable` side effects;
- offsets original and clone by ∓0.27 m along the **camera's own right axis**, converted
  into the parent frame, so it is correct whatever axis convention the parent uses;
- appends the clone to the private `cameraSensors` list, which the holder re-reads every
  cycle, so it takes effect live.

Field names were confirmed against the compiled `Assembly-CSharp.dll` metadata (via
`dnfile`) before writing any code — they match the public v1.6.1 source exactly.

The mod avoids `System.Linq` entirely, for the same stripping reason that killed BepInEx.

### Verified result

`scripts/awsim/awsim_stereo_check.py` on a live pair:

```
pair timestamp delta : 0.00 ms          <- same simulation frame
mean abs pixel diff  : 32.33            <- genuinely different viewpoints
median |dy|          : 0.00 px          <- RECTIFIED
matches |dy|<1px     : 83%
median disparity     : +27 px           <- positive => left/right not swapped
implied depth        : median 19 m (near 7 m, far 61 m)
dense SGBM depth     : median 15 m, 74% of pixels matched
```

Rates over 25 s, with the GPU at 95% from other users: left 6.000 Hz, right 5.889 Hz,
LiDAR 5.334 Hz, `/clock` 99.3 Hz. The ~2% shortfall on the right camera is dropped
`AsyncGPUReadback` under contention; harmless, since pairs are matched on timestamp.

Left camera intrinsics are **unchanged** (`fx=960.0 cx=960.5 fy=959.3908081054688`), so
Arka's calib files still apply.

Images: `multimedia/awsim_stereo/{left,right,anaglyph,disparity}.png`.

### Caveat worth remembering

Centring the rig on the stock camera means the traffic-light camera now sits 27 cm left
of where Autoware's `awsim_labs_sensor_kit` URDF thinks it is. This does not affect
recording (the recorder uses a hardcoded `base_link`→camera transform, not TF), and the
traffic lights are forced green anyway, but do not rely on that camera for precise
map projection.

The alternative — leave the left camera at the stock pose and put the right at +0.54 m —
avoids the discrepancy but no longer matches Arka's `P2`/`P3`, which place the two
cameras symmetrically at ∓0.27 m about the reference frame.

### Open question for later

Arka's `R0_rect` is not identity — it is real-KITTI's `R0_rect` (~0.5° rotation) copied
verbatim, as is his `Tr_imu_to_velo`. The true AWSIM value is identity (live `camera_info`
`r` = identity). His `Tr_velo_to_cam` is also hand-rounded and implies a LiDAR frame
rotated 90° from the KITTI convention. Decide whether to reproduce his calib exactly (for
comparability with his checkpoints) or to write a correct one. Do not change it silently.
