"""Compatibility helpers backed by the public bone segmentation API."""

from typing import Optional, TypeAlias

import numpy as np

from ...bone_segmentation import (
    apply_bone_mask,
    estimate_bone_thresholds,
    extract_bone_mask,
)


PackedBoneMasks: TypeAlias = tuple[np.ndarray, tuple[int, int]]


def _estimate_cbct_bone_threshold(array: np.ndarray) -> float:
    """Compatibility helper returning the adaptive CBCT seed threshold."""
    return estimate_bone_thresholds(array, "cbct")[1]


def _bone_threshold(
    array: np.ndarray, image_type: Optional[str]
) -> Optional[float]:
    if image_type not in {"ct", "cbct"}:
        return None
    return estimate_bone_thresholds(array, image_type)[0]


def _blacken_non_bone(
    pixels: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> None:
    bone_mask = np.isfinite(values) & (values >= threshold)
    pixels[:] = apply_bone_mask(pixels, bone_mask)


def _precompute_packed_bone_masks(
    array: np.ndarray,
    image_type: Optional[str],
    mmpd: float = 1.0,
) -> Optional[PackedBoneMasks]:
    """Precompute every S-plane mask and store it at one bit per voxel."""
    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError("bone mask input must be a three-dimensional array")
    if image_type not in {"ct", "cbct"}:
        return None

    mask = extract_bone_mask(values, image_type, mmpd)
    size_l, size_p, size_s = values.shape
    packed_width = (size_l * size_p + 7) // 8
    packed = np.empty((size_s, packed_width), dtype=np.uint8)
    for index in range(size_s):
        packed[index] = np.packbits(mask[:, :, index], bitorder="little")
    return packed, (size_l, size_p)


def _unpack_bone_mask(
    packed_masks: PackedBoneMasks,
    index: int,
) -> np.ndarray:
    packed, shape = packed_masks
    pixel_count = shape[0] * shape[1]
    return np.unpackbits(
        packed[index], count=pixel_count, bitorder="little"
    ).reshape(shape).astype(bool, copy=False)
