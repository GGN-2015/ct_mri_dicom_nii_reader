"""Compute a scalar 3-D aggregation of the six-neighbourhood MIND descriptor."""

from __future__ import annotations

from operator import index

import numpy as np


_NEIGHBOURS = (
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 1),
    (2, -1),
    (2, 1),
)


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum(dtype=np.float64)
    return kernel.astype(np.float32, copy=False)


def _gaussian_filter_3d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return image.copy()

    result = image
    for axis in range(3):
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        pad_mode = "reflect" if result.shape[axis] > 1 else "edge"
        padded = np.pad(result, pad_width, mode=pad_mode)
        filtered = np.zeros_like(result, dtype=np.float32)
        for kernel_index, weight in enumerate(kernel):
            source_slice = [slice(None), slice(None), slice(None)]
            source_slice[axis] = slice(
                kernel_index, kernel_index + result.shape[axis]
            )
            filtered += weight * padded[tuple(source_slice)]
        result = filtered
    return result


def _reflected_shift(image: np.ndarray, axis: int, displacement: int) -> np.ndarray:
    axis_length = image.shape[axis]
    if axis_length == 1:
        indices = np.zeros(1, dtype=np.intp)
    else:
        period = 2 * axis_length - 2
        indices = np.mod(
            np.arange(axis_length, dtype=np.int64) + displacement,
            period,
        )
        indices = np.where(indices < axis_length, indices, period - indices)
    return np.take(image, indices.astype(np.intp, copy=False), axis=axis)


def _patch_distance(
    image: np.ndarray,
    axis: int,
    displacement: int,
    gaussian_kernel: np.ndarray,
) -> np.ndarray:
    shifted = _reflected_shift(image, axis, displacement)
    difference = image - shifted
    np.square(difference, out=difference)
    distance = _gaussian_filter_3d(difference, gaussian_kernel)
    # Small negative values can only arise from floating-point accumulation.
    np.maximum(distance, 0.0, out=distance)
    return distance


def compute_mind_image(
    volume: np.ndarray,
    *,
    patch_radius: int = 1,
    patch_sigma: float = 0.8,
    neighbour_radius: int = 1,
) -> np.ndarray:
    """Return a same-size scalar 3-D MIND representation of ``volume``.

    Standard 3-D MIND is a six-component descriptor at every voxel. To satisfy
    a strictly 3-D output contract, this function computes those six components,
    normalizes them so their per-voxel maximum is one, then returns their mean.
    The result therefore has exactly ``volume.shape`` rather than an additional
    descriptor-channel axis.

    Patch SSD values are Gaussian weighted. Local variance is estimated from
    the mean patch distance to the six face-connected neighbours. All spatial
    parameters are measured in voxels because a NumPy array alone contains no
    physical voxel spacing.

    Args:
        volume: Finite numeric three-dimensional NumPy array. Its modality and
            intensity units do not need to be provided. The input is not changed.
        patch_radius: Gaussian patch radius in voxels. The patch width is
            ``2 * patch_radius + 1``.
        patch_sigma: Standard deviation of the patch Gaussian, in voxels.
        neighbour_radius: Distance from the centre to each of the six compared
            neighbour patches, in voxels.

    Returns:
        A new contiguous ``float32`` array with the same shape as ``volume``.
        Values are in ``[0, 1]``; larger values indicate greater average local
        self-similarity across the six MIND components.

    Notes:
        A scalar reduction loses directional descriptor information. For image
        registration losses, retaining the standard six channels is usually
        more discriminative than this requested same-size representation.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy array")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if not np.issubdtype(volume.dtype, np.number) or np.issubdtype(
        volume.dtype, np.complexfloating
    ):
        raise TypeError("volume must contain real numeric values")

    try:
        patch_radius = index(patch_radius)
        neighbour_radius = index(neighbour_radius)
    except TypeError as exc:
        raise TypeError("patch_radius and neighbour_radius must be integers") from exc
    if patch_radius < 0:
        raise ValueError("patch_radius must be greater than or equal to zero")
    if neighbour_radius <= 0:
        raise ValueError("neighbour_radius must be greater than zero")
    if isinstance(patch_sigma, bool):
        raise ValueError("patch_sigma must be a finite number greater than zero")
    try:
        patch_sigma = float(patch_sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError("patch_sigma must be a finite number greater than zero") from exc
    if not np.isfinite(patch_sigma) or patch_sigma <= 0:
        raise ValueError("patch_sigma must be a finite number greater than zero")

    image = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("volume must not contain NaN or infinite values")
    gaussian_kernel = _gaussian_kernel(patch_radius, patch_sigma)

    local_variance = np.zeros(image.shape, dtype=np.float32)
    for axis, direction in _NEIGHBOURS:
        distance = _patch_distance(
            image,
            axis=axis,
            displacement=direction * neighbour_radius,
            gaussian_kernel=gaussian_kernel,
        )
        local_variance += distance / len(_NEIGHBOURS)

    mean_variance = float(np.mean(local_variance, dtype=np.float64))
    variance_floor = max(
        mean_variance * 1e-6,
        float(np.finfo(np.float32).tiny),
    )
    np.maximum(local_variance, variance_floor, out=local_variance)

    descriptor_sum = np.zeros(image.shape, dtype=np.float32)
    descriptor_max = np.zeros(image.shape, dtype=np.float32)
    for axis, direction in _NEIGHBOURS:
        descriptor = _patch_distance(
            image,
            axis=axis,
            displacement=direction * neighbour_radius,
            gaussian_kernel=gaussian_kernel,
        )
        np.divide(descriptor, local_variance, out=descriptor)
        np.clip(descriptor, 0.0, 80.0, out=descriptor)
        np.negative(descriptor, out=descriptor)
        np.exp(descriptor, out=descriptor)
        descriptor_sum += descriptor
        np.maximum(descriptor_max, descriptor, out=descriptor_max)

    scalar_mind = descriptor_sum / (len(_NEIGHBOURS) * descriptor_max)
    np.clip(scalar_mind, 0.0, 1.0, out=scalar_mind)
    return np.ascontiguousarray(scalar_mind, dtype=np.float32)


__all__ = ["compute_mind_image"]
