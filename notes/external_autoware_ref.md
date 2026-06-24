# external/autoware — reference checkout

This folder is a **read-only upstream reference**, not part of the project repo.
It is excluded from version control (see root `.gitignore`).

If source-level Autoware modifications become necessary, convert it to a proper
git submodule pointing to a fork/branch rather than committing it directly.

## Checkout details

| Field | Value |
|-------|-------|
| Remote | `https://github.com/autowarefoundation/autoware.git` |
| Branch | `main` |
| Commit | `59468eedea0e2f40c5ad7265082dacfc81631392` |
| Commit message | `chore(ansible): update ptv3 artifacts (#7160)` |
| Local diffs | none (clean upstream checkout) |

## Runtime image

The Autoware Docker image used by this project is:

```
ghcr.io/autowarefoundation/autoware:universe-cuda-humble
```

This is separate from the source checkout above. The source checkout is only
used for reading config files, understanding launch arguments, and reference.
The container runs the pre-built image, not a locally compiled source tree.

## To update the reference

```bash
cd external/autoware
git fetch origin
git checkout main
git pull
```

Then update the commit hash in this file.
