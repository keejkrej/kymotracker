"""Preprocessing utilities for real kymograph data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class PreprocessingResult:
    """Outputs from real-data kymograph preprocessing."""

    normalized: np.ndarray
    background: np.ndarray


def estimate_background(
    intensity: np.ndarray,
    *,
    median_kernel: Sequence[int] | int = (1, 31),
) -> np.ndarray:
    """Estimate background intensity with a median filter."""

    intensity_array = np.asarray(intensity, dtype=np.float32)
    if intensity_array.ndim != 2:
        raise ValueError(
            f"intensity must be 2D (time, space); got shape {intensity_array.shape}"
        )

    return median_filter(intensity_array, size=median_kernel).astype(np.float32, copy=False)


def normalize_intensity(
    intensity: np.ndarray,
    background: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize interference contrast using (background - intensity) / sqrt(background).

    This follows the scattering interference form
    I = I_background - 2 * sqrt(I_background * I_particle) + I_particle.
    """

    intensity_array = np.asarray(intensity, dtype=np.float32)
    background_array = np.asarray(background, dtype=np.float32)

    if intensity_array.shape != background_array.shape:
        raise ValueError(
            "intensity and background must have the same shape; "
            f"got {intensity_array.shape} and {background_array.shape}"
        )

    safe_background = np.maximum(background_array, eps)
    normalized = (safe_background - intensity_array) / np.sqrt(safe_background)
    return normalized.astype(np.float32, copy=False)


def preprocess_kymograph(
    intensity: np.ndarray,
    *,
    median_kernel: Sequence[int] | int = (1, 31),
    eps: float = 1e-6,
) -> PreprocessingResult:
    """Estimate background and normalize a real kymograph."""

    background = estimate_background(intensity, median_kernel=median_kernel)
    normalized = normalize_intensity(intensity, background, eps=eps)
    return PreprocessingResult(normalized=normalized, background=background)


__all__ = [
    "PreprocessingResult",
    "estimate_background",
    "normalize_intensity",
    "preprocess_kymograph",
]
