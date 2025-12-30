"""Python wrapper for Shewchuk predicates."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent
_CPP_DIR = _ROOT / "cpp"
_BUILD_DIR = _CPP_DIR / "build"
_SRC = _CPP_DIR / "predicates.c"


def _lib_name() -> str:
    if sys.platform == "darwin":
        return "libpredicates.dylib"
    return "libpredicates.so"


def _build_library(output_path: Path) -> None:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        cmd = [
            "cc",
            "-O3",
            "-fPIC",
            "-dynamiclib",
            str(_SRC),
            "-o",
            str(output_path),
        ]
    else:
        cmd = [
            "cc",
            "-O3",
            "-fPIC",
            "-shared",
            str(_SRC),
            "-o",
            str(output_path),
            "-lm",
        ]
    subprocess.check_call(cmd)


def _load_library() -> ctypes.CDLL:
    output_path = _BUILD_DIR / _lib_name()
    if not output_path.exists():
        _build_library(output_path)
    return ctypes.CDLL(str(output_path))


_LIB = _load_library()
_LIB.exactinit.argtypes = []
_LIB.exactinit.restype = None
_LIB.exactinit()

_ORIENT3D = _LIB.orient3d
_ORIENT3D.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_ORIENT3D.restype = ctypes.c_double


def orient3d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, pd: np.ndarray) -> float:
    pa = np.asarray(pa, dtype=np.float64).reshape(3)
    pb = np.asarray(pb, dtype=np.float64).reshape(3)
    pc = np.asarray(pc, dtype=np.float64).reshape(3)
    pd = np.asarray(pd, dtype=np.float64).reshape(3)
    return float(
        _ORIENT3D(
            pa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )


__all__ = ["orient3d"]
