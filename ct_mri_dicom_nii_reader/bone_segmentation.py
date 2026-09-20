"""Reusable threshold-based CT and CBCT bone segmentation."""

from __future__ import annotations

from typing import Literal, Optional, cast

import numpy as np


BoneImageType = Literal["ct", "cbct"]

_CT_LOW_THRESHOLD_HU = 100.0
_CT_HIGH_THRESHOLD_HU = 300.0
_CBCT_HISTOGRAM_BINS = 256
_THRESHOLD_SAMPLE_SIZE = 500_000


def _validated_volume(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume)
    if values.ndim != 3:
        raise ValueError("volume must be a three-dimensional array")
    if any(size == 0 for size in values.shape):
        raise ValueError("volume dimensions must not be empty")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("volume must contain numeric values")
    return values


def _validated_image_type(image_type: str) -> BoneImageType:
    normalized = str(image_type).strip().lower()
    if normalized not in {"ct", "cbct"}:
        raise ValueError("image_type must be 'ct' or 'cbct'")
    return cast(BoneImageType, normalized)


def _finite_sample(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    stride = max(1, flat.size // _THRESHOLD_SAMPLE_SIZE)
    sample = np.asarray(flat[::stride], dtype=np.float32)
    return sample[np.isfinite(sample)]


def _cbct_three_class_statistics(
    array: np.ndarray,
) -> Optional[tuple[float, float, float]]:
    """Return middle mean, upper separator, and high-density mean."""
    sample = _finite_sample(array)
    if sample.size == 0:
        return None

    low, high = np.percentile(sample, (0.5, 99.5))
    low = float(low)
    high = float(high)
    if high <= low:
        return None

    histogram, bin_edges = np.histogram(
        np.clip(sample, low, high),
        bins=_CBCT_HISTOGRAM_BINS,
        range=(low, high),
    )
    probabilities = histogram.astype(np.float64)
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0.0:
        return None
    probabilities /= probability_sum
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    cumulative_weight = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    total_mean = cumulative_mean[-1]

    best_score = -np.inf
    best: Optional[tuple[int, float, float]] = None
    minimum_class_weight = 0.005
    for lower_separator in range(_CBCT_HISTOGRAM_BINS - 2):
        weight_low = cumulative_weight[lower_separator]
        if weight_low < minimum_class_weight:
            continue

        candidates = np.arange(
            lower_separator + 1, _CBCT_HISTOGRAM_BINS - 1
        )
        weight_middle = cumulative_weight[candidates] - weight_low
        weight_high = 1.0 - cumulative_weight[candidates]
        valid = (
            (weight_middle >= minimum_class_weight)
            & (weight_high >= minimum_class_weight)
        )
        if not np.any(valid):
            continue

        candidates = candidates[valid]
        weight_middle = weight_middle[valid]
        weight_high = weight_high[valid]
        mean_low = cumulative_mean[lower_separator] / weight_low
        mean_middle = (
            cumulative_mean[candidates]
            - cumulative_mean[lower_separator]
        ) / weight_middle
        mean_high = (
            total_mean - cumulative_mean[candidates]
        ) / weight_high
        scores = (
            weight_low * (mean_low - total_mean) ** 2
            + weight_middle * (mean_middle - total_mean) ** 2
            + weight_high * (mean_high - total_mean) ** 2
        )
        local_index = int(np.argmax(scores))
        if scores[local_index] > best_score:
            best_score = float(scores[local_index])
            best = (
                int(candidates[local_index]),
                float(mean_middle[local_index]),
                float(mean_high[local_index]),
            )

    if best is None:
        return None
    upper_separator, mean_middle, mean_high = best
    boundary = float(bin_edges[upper_separator + 1])
    return mean_middle, boundary, mean_high


def estimate_bone_thresholds(
    volume: np.ndarray,
    image_type: BoneImageType | str,
) -> tuple[float, float]:
    """Estimate lower candidate and upper seed thresholds for bone.

    CT uses stable HU defaults of 100 and 300. CBCT thresholds are estimated
    from a robust three-class histogram because its values are scanner- and
    acquisition-dependent. The return order is ``(low, high)``.
    """
    values = _validated_volume(volume)
    normalized_type = _validated_image_type(image_type)
    if normalized_type == "ct":
        return _CT_LOW_THRESHOLD_HU, _CT_HIGH_THRESHOLD_HU

    statistics = _cbct_three_class_statistics(values)
    if statistics is None:
        return float("inf"), float("inf")
    mean_middle, boundary, mean_high = statistics
    low_threshold = mean_middle + 0.5 * (boundary - mean_middle)
    high_threshold = boundary + 0.15 * (mean_high - boundary)
    if high_threshold <= low_threshold:
        high_threshold = np.nextafter(low_threshold, float("inf"))
    return float(low_threshold), float(high_threshold)


def extract_bone_mask(
    volume: np.ndarray,
    image_type: BoneImageType | str,
    mmpd: float = 1.0,
    *,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    denoise_sigma_mm: float = 0.4,
    closing_radius_mm: float = 1.0,
    minimum_component_volume_mm3: float = 8.0,
    fully_connected: bool = True,
) -> np.ndarray:
    """Extract a three-dimensional CT or CBCT bone mask.

    High-threshold voxels form confident bone seeds. Binary reconstruction
    retains lower-threshold partial-volume voxels only when connected to a
    seed. A small physical-radius closing repairs narrow cortical breaks, and
    connected components below the requested physical volume are removed.

    All spatial parameters use millimetres. Threshold overrides use the
    source intensity units: HU for CT and scanner-specific values for CBCT.
    The returned Boolean array has the same ``(L, P, S)`` shape as ``volume``.
    """
    values = _validated_volume(volume)
    normalized_type = _validated_image_type(image_type)
    mmpd = float(mmpd)
    if not np.isfinite(mmpd) or mmpd <= 0.0:
        raise ValueError("mmpd must be finite and greater than zero")
    for name, value in (
        ("denoise_sigma_mm", denoise_sigma_mm),
        ("closing_radius_mm", closing_radius_mm),
        ("minimum_component_volume_mm3", minimum_component_volume_mm3),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    estimated_low, estimated_high = estimate_bone_thresholds(
        values, normalized_type
    )
    low = estimated_low if low_threshold is None else float(low_threshold)
    high = (
        estimated_high if high_threshold is None else float(high_threshold)
    )
    if not np.isfinite(low) or not np.isfinite(high):
        return np.zeros(values.shape, dtype=bool)
    if high < low:
        raise ValueError(
            "high_threshold must be greater than or equal to low_threshold"
        )

    import SimpleITK as sitk

    finite = np.isfinite(values)
    fill_value = low - max(abs(high - low), 1.0)
    working = np.asarray(
        np.where(finite, values, fill_value), dtype=np.float32
    )
    image = sitk.GetImageFromArray(
        np.ascontiguousarray(np.transpose(working, (2, 1, 0)))
    )
    image.SetSpacing([mmpd, mmpd, mmpd])
    if denoise_sigma_mm > 0.0:
        variance = float(denoise_sigma_mm) ** 2
        image = sitk.DiscreteGaussian(
            image,
            variance=[variance, variance, variance],
            useImageSpacing=True,
        )

    seeds = sitk.Cast(image >= high, sitk.sitkUInt8)
    candidates = sitk.Cast(image >= low, sitk.sitkUInt8)
    mask = sitk.BinaryReconstructionByDilation(
        seeds,
        candidates,
        backgroundValue=0.0,
        foregroundValue=1.0,
        fullyConnected=bool(fully_connected),
    )

    closing_radius = int(round(float(closing_radius_mm) / mmpd))
    if closing_radius > 0:
        mask = sitk.BinaryMorphologicalClosing(
            mask,
            [closing_radius] * 3,
            sitk.sitkBall,
            1.0,
            True,
        )

    minimum_voxels = int(
        np.ceil(float(minimum_component_volume_mm3) / (mmpd ** 3))
    )
    if minimum_voxels > 1:
        components = sitk.ConnectedComponent(mask, bool(fully_connected))
        components = sitk.RelabelComponent(
            components,
            minimum_voxels,
            False,
        )
        mask = sitk.Cast(components > 0, sitk.sitkUInt8)

    mask_zyx = sitk.GetArrayFromImage(mask).astype(bool, copy=False)
    result = np.ascontiguousarray(np.transpose(mask_zyx, (2, 1, 0)))
    result[~finite] = False
    return result


def extract_ct_bone_mask(
    volume: np.ndarray,
    mmpd: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper for :func:`extract_bone_mask` on CT HU data."""
    return extract_bone_mask(volume, "ct", mmpd, **kwargs)


def extract_cbct_bone_mask(
    volume: np.ndarray,
    mmpd: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper for adaptive CBCT bone segmentation."""
    return extract_bone_mask(volume, "cbct", mmpd, **kwargs)


def apply_bone_mask(
    volume: np.ndarray,
    bone_mask: np.ndarray,
    outside_value: float | int = 0,
) -> np.ndarray:
    """Return a copy of ``volume`` with non-bone voxels replaced."""
    values = np.asarray(volume)
    mask = np.asarray(bone_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("volume and bone_mask must have the same shape")
    result = np.full_like(values, outside_value)
    np.copyto(result, values, where=mask)
    return result


__all__ = [
    "BoneImageType",
    "estimate_bone_thresholds",
    "extract_bone_mask",
    "extract_ct_bone_mask",
    "extract_cbct_bone_mask",
    "apply_bone_mask",
]
