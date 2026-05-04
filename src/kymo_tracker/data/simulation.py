"""Multi-particle simulation entry points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kymo_tracker.utils.helpers import generate_kymograph


def generate_multiparticle_kymograph(
    *,
    length: int = 16,
    width: int = 512,
    contrast: Sequence[float],
    diffusion: Sequence[float],
    noise_level: float = 0.1,
    peak_width: float = 1.0,
    dt: float = 1.0,
    dx: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a kymograph with two or more simulated particles."""

    contrasts = np.asarray(contrast, dtype=float)
    diffusions = np.asarray(diffusion, dtype=float)
    if contrasts.ndim != 1 or diffusions.ndim != 1:
        raise ValueError("contrast and diffusion must be 1D sequences")
    if len(contrasts) < 2 or len(diffusions) < 2:
        raise ValueError("at least two particles are required")
    if len(contrasts) != len(diffusions):
        raise ValueError("contrast and diffusion must describe the same number of particles")

    return generate_kymograph(
        length=length,
        width=width,
        contrast=contrasts,
        diffusion=diffusions,
        noise_level=noise_level,
        peak_width=peak_width,
        dt=dt,
        dx=dx,
        seed=seed,
    )


__all__ = ["generate_multiparticle_kymograph"]
