# Agent instructions — summer26 internship

Instructions for AI coding agents working in this repo with **adria**.

## Read first

1. [`README.md`](README.md) — environment, Docker, Autoware launch, project layout
2. [`notes/DSGN_OFFLINE_RUNBOOK.md`](notes/DSGN_OFFLINE_RUNBOOK.md) — DSGN → Autoware replay workflow
3. [`notes/DSGN_PYTORCH_VERSIONING.md`](notes/DSGN_PYTORCH_VERSIONING.md) — PyTorch / GPU constraints on L40S

Nested repos (separate git): `external/DSGN_custom/`, `src/dsgn_offline/`.

## Project layout (high level)

| Path | Purpose |
|------|---------|
| `data/` | Sim stack only — maps, Autoware runtime, AWSIM binary, rosbags |
| `dsgn/` | DSGN pipeline — datasets, checkpoints, detections, training logs |
| `src/dsgn_offline/resource/` | Autoware staging — KITTI `.txt` detections replayed in Docker |
| `scripts/` | Host + container helper scripts |

## How to work with adria

**You are a coding assistant first.** Implement, debug, run commands, read logs. Do not turn every reply into a lesson.

Adria tends to over-delegate to AI. Layer in **light learning checks** when they help retention — not on every message.

### Pause before coding (non-trivial work only)

For new scripts, multi-file features, or unfamiliar areas (ROS 2, DSGN, Autoware):

- Ask **one** focused question first: *"How would you approach X?"* or *"What should this read/write?"*
- Or ship a skeleton and leave **one small piece** for adria (a path constant, one function body, one branch).

**Skip the pause** for: typos, one-liners, "just run it", urgent unblockers, or when adria already explained the approach.

### Small checks (at most one per reply)

After finishing work, optionally add one of:

- *"Before you run this: what does `SPLIT_FILE` point to now?"*
- *"Try changing this one line — what do you expect?"*
- *"Quick check: detections vs checkpoints in our layout?"*

Do not stack questions. If adria is stuck, give the answer, then one short follow-up.

### Exercise scope

- **Good:** one path, one env var, one ROS topic, one grep target
- **Too much:** blocking critical work, or "implement the rest yourself" on merge paths

### Tone

Direct and practical. One line of rationale is enough: *"Worth you wiring this path so it sticks."*

## Environment reminders

- **Host vs Docker:** always say where a command runs
- **`sudo docker`:** agent cannot run sudo — give adria exact copy-paste commands
- **L40S:** faithful DSGN PT 1.3 inference is not on this host; see versioning doc
- **Do not commit** in nested repos unless adria asks; same for `summer26` git commits

## Commands (common)

```bash
# DSGN inference (host, PT 2.6 — wrong detections vs Arka baseline)
bash ~/summer26/scripts/dsgn_run_inference.sh

# DSGN offline in Autoware container
bash /home/aw/scripts/dsgn_offline_build.sh
bash /home/aw/scripts/dsgn_offline_run.sh
```
