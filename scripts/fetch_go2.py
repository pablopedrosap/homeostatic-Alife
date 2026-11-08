#!/usr/bin/env python3
"""Fetch the Unitree Go2 model from DeepMind's MuJoCo Menagerie (BSD-3, © Unitree Robotics).

Pinned commit for reproducibility; sparse checkout of unitree_go2/ only (~29 MB of meshes,
kept out of this repository). Destination: third_party/mujoco_menagerie/.

    python scripts/fetch_go2.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "mujoco_menagerie"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    if (DEST / "unitree_go2" / "scene_mjx.xml").exists():
        print(f"already present: {DEST / 'unitree_go2'}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--filter=blob:none", "--no-checkout", URL, str(DEST))
    run("git", "sparse-checkout", "set", "unitree_go2", cwd=DEST)
    run("git", "checkout", COMMIT, cwd=DEST)
    print(f"fetched Go2 @ {COMMIT[:10]} → {DEST / 'unitree_go2'}")


if __name__ == "__main__":
    main()
