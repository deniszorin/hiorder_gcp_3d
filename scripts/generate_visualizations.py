"""Generate validation visualizations for smoothed offset potential."""

from __future__ import annotations

import argparse
from pathlib import Path

import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, os.fspath(PROJECT_ROOT))

from viz import run_validation_visualizations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory to write output images (or HTML fallback).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = Path(__file__).resolve().parent.parent / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    run_validation_visualizations(output_dir=str(output_dir))

    try:
        from viz import build_validation_scene_specs
    except ImportError:
        return
    scenes = build_validation_scene_specs()
    for scene in scenes:
        if scene.name != "cube_face_centers":
            continue
        obj_path = output_dir / "cube_face_centers.obj"
        with obj_path.open("w", encoding="utf-8") as handle:
            for v in scene.mesh.V:
                handle.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in scene.mesh.faces:
                handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


if __name__ == "__main__":
    main()
