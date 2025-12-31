"""Python wrapper for the C++ potential module."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np


def _extension_path(build_dir: Path, name: str) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        raise RuntimeError("Missing Python extension suffix.")
    return build_dir / f"{name}{suffix}"


def _configure_cmake(
    cpp_dir: Path,
    build_dir: Path,
) -> None:
    cmake_args = [
        "cmake",
        "-S",
        str(cpp_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.check_call(cmake_args)


def _is_multi_config(build_dir: Path) -> bool:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return False
    contents = cache_path.read_text(encoding="utf-8", errors="ignore")
    return "CMAKE_CONFIGURATION_TYPES" in contents


def _build_extension() -> Path:
    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    build_dir = cpp_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    name = "_potential_cpp"
    ext_path = _extension_path(build_dir, name)
    _configure_cmake(cpp_dir, build_dir)

    build_cmd = ["cmake", "--build", str(build_dir), "--target", name]
    if _is_multi_config(build_dir):
        build_cmd.extend(["--config", "Release"])
    subprocess.check_call(build_cmd)

    if not ext_path.exists():
        raise RuntimeError(f"Failed to build extension: {ext_path}")
    return ext_path


def _load_extension():
    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    build_dir = cpp_dir / "build"
    ext_path = _extension_path(build_dir, "_potential_cpp")
    if not ext_path.exists():
        ext_path = _build_extension()
    spec = importlib.util.spec_from_file_location("_potential_cpp", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load extension: {ext_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def smoothed_offset_potential_cpp(
    q: np.ndarray, V: np.ndarray, F: np.ndarray,
    *,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True,
    localized: bool = False, one_sided: bool = False,
) -> np.ndarray:
    module = _load_extension()
    return module.smoothed_offset_potential_cpp(
        q, V, F,
        alpha, p, epsilon,
        include_faces, include_edges, include_vertices,
        localized, one_sided,
    )


def pointed_vertices_cpp(
    V: np.ndarray, F: np.ndarray,
) -> np.ndarray:
    module = _load_extension()
    return module.pointed_vertices_cpp(
        V, F,
    )
