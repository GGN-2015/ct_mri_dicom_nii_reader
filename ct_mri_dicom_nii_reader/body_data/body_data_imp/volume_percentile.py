"""Compute a percentile from all voxels in a three-dimensional NumPy array."""

from __future__ import annotations

from operator import index

import numpy as np


def get_volume_percentile(volume: np.ndarray, idx: int) -> float:
    """Return the requested percentile across all values in a 3-D array.

    Args:
        volume: Non-empty, real-valued, three-dimensional NumPy array.
        idx: Integer percentile in ``[0, 100]``. Zero returns the minimum,
            100 returns the maximum, and values from 1 through 99 use NumPy's
            percentile calculation over all voxels.

    Returns:
        The minimum, maximum or requested percentile as a Python ``float``.

    Notes:
        The input array is not modified. NaN values are not ignored; consistent
        with ``numpy.percentile``, their presence can make the result NaN.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy array")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if volume.size == 0:
        raise ValueError("volume must not be empty")
    if not np.issubdtype(volume.dtype, np.number) or np.issubdtype(
        volume.dtype, np.complexfloating
    ):
        raise TypeError("volume must contain real numeric values")
    if isinstance(idx, (bool, np.bool_)):
        raise TypeError("idx must be an integer from 0 to 100")
    try:
        idx = index(idx)
    except TypeError as exc:
        raise TypeError("idx must be an integer from 0 to 100") from exc
    if not 0 <= idx <= 100:
        raise ValueError("idx must be in the closed interval [0, 100]")

    if idx == 0:
        return float(np.min(volume))
    if idx == 100:
        return float(np.max(volume))
    return float(np.percentile(volume, idx))


__all__ = ["get_volume_percentile"]
