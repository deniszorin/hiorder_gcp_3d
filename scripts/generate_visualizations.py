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
    parser.add_argument(
        "--use-cpp",
        action="store_true",
        help="Use C++ potential evaluation in isosurface plots.",
    )
    parser.add_argument(
        "--use-accelerated",
        action="store_true",
        help="Use accelerated evaluation in isosurface plots.",
    )
    parser.add_argument(
        "--use-simplified",
        action="store_true",
        help="Use simplified potential in isosurface plots.",
    )
    parser.add_argument(
        "--localized",
        action="store_true",
        help="Enable localization in isosurface plots.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Localization epsilon for isosurface plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Heaviside smoothing alpha for isosurface plots.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="Potential exponent p for isosurface plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = Path(__file__).resolve().parent.parent / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    run_validation_visualizations(
        output_dir=str(output_dir),
        use_cpp=args.use_cpp,
        use_accelerated=args.use_accelerated,
        use_simplified=args.use_simplified,
        localized=args.localized,
        epsilon=args.epsilon,
        alpha=args.alpha,
        p=args.p,
    )

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
