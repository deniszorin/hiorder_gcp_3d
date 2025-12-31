"""Python wrapper for the C++ cone convex hull module."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pybind11
from pybind11.setup_helpers import Pybind11Extension
from setuptools import Distribution
from setuptools.command.build_ext import build_ext


def _extension_path(build_dir: Path, name: str) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        raise RuntimeError("Missing Python extension suffix.")
    return build_dir / f"{name}{suffix}"


def _lib_name() -> str:
    if sys.platform == "darwin":
        return "libpredicates.dylib"
    return "libpredicates.so"


def _build_predicates_library(build_dir: Path, src: Path) -> Path:
    output_path = build_dir / _lib_name()
    if output_path.exists():
        return output_path
    build_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        cmd = [
            "cc",
            "-O3",
            "-fPIC",
            "-dynamiclib",
            str(src),
            "-o",
            str(output_path),
        ]
    else:
        cmd = [
            "cc",
            "-O3",
            "-fPIC",
            "-shared",
            str(src),
            "-o",
            str(output_path),
            "-lm",
        ]
    subprocess.check_call(cmd)
    return output_path


def _extension_sources(cpp_dir: Path) -> list[Path]:
    return [
        cpp_dir / "cone_convex_hull.cpp",
        cpp_dir / "cone_convex_hull_cpp.cpp",
    ]


def _build_extension() -> Path:
    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    build_dir = cpp_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    name = "_cone_convex_hull_cpp"
    ext_path = _extension_path(build_dir, name)

    predicate_src = cpp_dir / "predicates.c"
    libpred = _build_predicates_library(build_dir, predicate_src)

    sources = [str(path) for path in _extension_sources(cpp_dir)]

    include_dirs = [
        str(cpp_dir),
        pybind11.get_include(),
        np.get_include(),
    ]
    eigen_candidates = [
        Path("/usr/local/include/eigen3"),
        Path("/opt/homebrew/include/eigen3"),
    ]
    eigen_dir = next((p for p in eigen_candidates if (p / "Eigen" / "Core").exists()), None)
    if eigen_dir is None:
        raise RuntimeError("Eigen headers not found; install eigen3 and retry.")
    include_dirs.append(str(eigen_dir))

    extra_link_args = []
    if sys.platform == "darwin":
        extra_link_args.append(f"-Wl,-rpath,{build_dir}")

    ext = Pybind11Extension(
        name,
        sources=sources,
        include_dirs=include_dirs,
        extra_objects=[str(libpred)],
        extra_link_args=extra_link_args,
        cxx_std=17,
    )

    dist = Distribution({"name": name, "ext_modules": [ext]})
    cmd = build_ext(dist)
    cmd.build_lib = str(build_dir)
    cmd.build_temp = str(build_dir / "temp_cone_hull")
    cmd.ensure_finalized()
    cmd.run()

    if not ext_path.exists():
        raise RuntimeError(f"Failed to build extension: {ext_path}")
    return ext_path


def _load_extension():
    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    build_dir = cpp_dir / "build"
    ext_path = _extension_path(build_dir, "_cone_convex_hull_cpp")
    predicate_src = cpp_dir / "predicates.c"
    predicate_lib = build_dir / _lib_name()

    if ext_path.exists():
        sources = _extension_sources(cpp_dir)
        ext_mtime = ext_path.stat().st_mtime
        dep_times = [path.stat().st_mtime for path in sources]
        if predicate_src.exists():
            dep_times.append(predicate_src.stat().st_mtime)
        if predicate_lib.exists():
            dep_times.append(predicate_lib.stat().st_mtime)
        if dep_times and max(dep_times) > ext_mtime:
            ext_path = _build_extension()
    else:
        ext_path = _build_extension()

    module_name = "_cone_convex_hull_cpp"
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load extension: {ext_path}")
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cone_convex_hull_cpp(
    e: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, bool, bool]:
    module = _load_extension()
    return module.cone_convex_hull_cpp(e, eps)


__all__ = ["cone_convex_hull_cpp"]
