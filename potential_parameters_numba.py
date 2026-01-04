"""Numba-friendly potential parameter helpers."""

from __future__ import annotations

from typing import NamedTuple

from numba import njit


class PotentialParameters(NamedTuple):
    alpha: float
    p: float
    epsilon: float
    localized: bool
    one_sided: bool


# ****************************************************************************
# Potential blending/localization functions

@njit(cache=True)
def H_scalar(z: float) -> float:
    if z < -1.0:
        return 0.0
    if z > 1.0:
        return 1.0
    return ((2.0 - z) * (z + 1.0) ** 2) / 4.0


@njit(cache=True)
def H_alpha_scalar(t: float, alpha: float) -> float:
    return H_scalar(t / alpha)


@njit(cache=True)
def h_local_scalar(z: float) -> float:
    if z > 1.0:
        return 0.0
    return (2.0 * z + 1.0) * (z - 1.0) ** 2


@njit(cache=True)
def h_epsilon_scalar(z: float, epsilon: float) -> float:
    return h_local_scalar(z / epsilon)
