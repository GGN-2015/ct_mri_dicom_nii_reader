"""Keep an x-z rectangle in a 3-D NumPy volume and fill everything else."""

from __future__ import annotations

from operator import index

import numpy as np


def fill_outside_xz_rectangle(
    volume: np.ndarray,
    l: int,
    r: int,
    t: int,
    b: int,
    air_value: object,
) -> np.ndarray:
    """Return a copy with values outside ``[l, r) x [t, b)`` replaced.

    The input must use axis order ``(x, y, z)``. The x-z rectangle applies to
    every y coordinate. Rectangle bounds are clipped to the volume, so a
    partially or completely out-of-range rectangle does not cause an indexing
    error. An empty intersection produces an array filled with ``air_value``.

    Args:
        volume: Three-dimensional NumPy array in ``(x, y, z)`` axis order.
        l: Inclusive lower x index.
        r: Exclusive upper x index.
        t: Inclusive lower z index.
        b: Exclusive upper z index.
        air_value: Value assigned outside the retained rectangle. It must be
            representable by ``volume.dtype`` according to NumPy's assignment
            rules.

    Returns:
        A new NumPy array with the same shape and dtype as ``volume``.
        ``volume`` itself is never modified.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy array")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")

    try:
        l, r, t, b = index(l), index(r), index(t), index(b)
    except TypeError as exc:
        raise TypeError("l, r, t and b must be integer indices") from exc

    x_size, _, z_size = volume.shape
    x_begin = min(max(l, 0), x_size)
    x_end = min(max(r, 0), x_size)
    z_begin = min(max(t, 0), z_size)
    z_end = min(max(b, 0), z_size)

    result = volume.copy()
    result[...] = air_value

    if x_begin < x_end and z_begin < z_end:
        result[x_begin:x_end, :, z_begin:z_end] = volume[
            x_begin:x_end, :, z_begin:z_end
        ]
    return result


__all__ = ["fill_outside_xz_rectangle"]
