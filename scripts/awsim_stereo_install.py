#!/usr/bin/env python3
"""Register (or unregister) StereoMod.dll with the AWSIM binary's Unity loader.

We cannot rebuild AWSIM from source, and BepInEx cannot load into this build (Unity
managed-code stripping removed LINQ members that BepInEx's config system needs). Unity
however keeps its own plain-JSON registry of managed assemblies and startup callbacks,
so we can hook in through the engine's documented mechanism instead of injecting:

  awsim_labs_Data/ScriptingAssemblies.json     - assemblies Unity loads (type 16 = user)
  awsim_labs_Data/RuntimeInitializeOnLoads.json - static methods Unity calls at startup

Only these two text files are touched. `--uninstall` restores them byte-for-byte from
the .orig backups taken on first install, so the modded tree returns to stock.

Usage:
  scripts/awsim_stereo_install.py [--uninstall] [--game DIR]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_GAME = Path.home() / "summer26/data/awsim/modded/awsim_labs_v1.6.1"

ASSEMBLY_DLL = "StereoMod.dll"
USER_ASSEMBLY_TYPE = 16  # same value AWSIM's own Assembly-CSharp.dll uses

# loadTypes 0 == RuntimeInitializeLoadType.AfterSceneLoad: the engine and first scene
# are up, so we may create GameObjects. The ego (and its camera) is spawned later, which
# is why StereoRig polls instead of acting immediately.
INIT_ENTRY = {
    "assemblyName": "StereoMod",
    "nameSpace": "StereoMod",
    "className": "Bootstrap",
    "methodName": "Init",
    "loadTypes": 0,
    "isUnityClass": False,
}


def backup_once(path: Path) -> None:
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)


def restore(path: Path) -> bool:
    orig = path.with_suffix(path.suffix + ".orig")
    if orig.exists():
        shutil.copy2(orig, path)
        return True
    return False


def install(data_dir: Path) -> None:
    managed = data_dir / "Managed" / ASSEMBLY_DLL
    if not managed.exists():
        sys.exit(f"ERROR: {managed} missing. Run scripts/awsim_stereo_build.sh first.")

    asm_path = data_dir / "ScriptingAssemblies.json"
    backup_once(asm_path)
    asm = json.loads(asm_path.read_text())
    if ASSEMBLY_DLL in asm["names"]:
        print(f"  ScriptingAssemblies.json     already registered")
    else:
        asm["names"].append(ASSEMBLY_DLL)
        asm["types"].append(USER_ASSEMBLY_TYPE)
        asm_path.write_text(json.dumps(asm, separators=(",", ":")))
        print(f"  ScriptingAssemblies.json     + {ASSEMBLY_DLL}")

    rio_path = data_dir / "RuntimeInitializeOnLoads.json"
    backup_once(rio_path)
    rio = json.loads(rio_path.read_text())
    already = any(e.get("assemblyName") == "StereoMod" for e in rio["root"])
    if already:
        print("  RuntimeInitializeOnLoads.json already registered")
    else:
        rio["root"].append(INIT_ENTRY)
        rio_path.write_text(json.dumps(rio, separators=(",", ":")))
        print("  RuntimeInitializeOnLoads.json + StereoMod.Bootstrap.Init")


def uninstall(data_dir: Path) -> None:
    for name in ("ScriptingAssemblies.json", "RuntimeInitializeOnLoads.json"):
        path = data_dir / name
        print(f"  {name:29s} {'restored' if restore(path) else 'no backup, unchanged'}")
    dll = data_dir / "Managed" / ASSEMBLY_DLL
    if dll.exists():
        dll.unlink()
        print(f"  Managed/{ASSEMBLY_DLL:21s} removed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--game", type=Path, default=DEFAULT_GAME)
    args = ap.parse_args()

    data_dir = args.game / "awsim_labs_Data"
    if not data_dir.is_dir():
        sys.exit(f"ERROR: {data_dir} is not a directory")

    print(f"{'Uninstalling' if args.uninstall else 'Installing'} StereoMod in {args.game}")
    (uninstall if args.uninstall else install)(data_dir)
    print("done")


if __name__ == "__main__":
    main()
