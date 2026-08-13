#!/usr/bin/env bash
# Compile StereoMod.dll against the AWSIM binary's own managed assemblies.
#
# Compiled in a throwaway Mono container so nothing is installed on this shared host.
# -nostdlib/-noconfig + referencing the game's own mscorlib guarantees the output binds
# to exactly the runtime AWSIM ships, rather than whatever the compiler defaults to.
#
# Output: data/awsim/modded/awsim_labs_v1.6.1/awsim_labs_Data/Managed/StereoMod.dll
set -euo pipefail

SRC_DIR="$HOME/summer26/src/awsim_stereo_mod"
GAME="$HOME/summer26/data/awsim/modded/awsim_labs_v1.6.1"
MANAGED="$GAME/awsim_labs_Data/Managed"

[[ -f "$SRC_DIR/StereoMod.cs" ]] || { echo "ERROR: missing $SRC_DIR/StereoMod.cs" >&2; exit 1; }
[[ -d "$MANAGED" ]] || { echo "ERROR: missing $MANAGED (is modded/ present?)" >&2; exit 1; }

docker run --rm \
  -v "$SRC_DIR:/src:ro" \
  -v "$MANAGED:/managed" \
  -w /work \
  mono:6.12 \
  bash -c '
    set -e
    mkdir -p /work && cp /src/StereoMod.cs /work/
    mcs -target:library -out:/work/StereoMod.dll \
        -nostdlib -noconfig -warn:2 \
        -lib:/managed \
        -r:mscorlib.dll \
        -r:System.dll \
        -r:System.Core.dll \
        -r:UnityEngine.dll \
        -r:UnityEngine.CoreModule.dll \
        -r:Assembly-CSharp.dll \
        /work/StereoMod.cs
    cp /work/StereoMod.dll /managed/StereoMod.dll
    echo "built: $(stat -c%s /managed/StereoMod.dll) bytes"
  '

echo "installed -> $MANAGED/StereoMod.dll"
