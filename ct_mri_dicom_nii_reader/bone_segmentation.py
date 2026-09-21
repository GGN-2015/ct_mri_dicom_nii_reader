"""Reusable threshold-based CT and CBCT bone segmentation."""

from __future__ import annotations

from typing import Literal, Optional, cast

import numpy as np


BoneImageType = Literal["ct", "cbct"]

_CT_LOW_THRESHOLD_HU = 100.0
_CT_HIGH_THRESHOLD_HU = 300.0
_CBCT_HISTOGRAM_BINS = 256
_CBCT_DENSITY_CLASSES = 5
_CBCT_MIN_CLASS_WEIGHT = 0.005
_CBCT_MIN_HIGH_DENSITY_FRACTION = 0.02
_CBCT_MAX_HIGH_DENSITY_FRACTION = 0.35
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


def _optimal_histogram_partition(
    histogram: np.ndarray,
    centers: np.ndarray,
    class_count: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Partition a weighted 1-D histogram with exact dynamic programming."""
    weights = np.asarray(histogram, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    bin_count = int(weights.size)
    total_weight = float(weights.sum())
    if class_count < 2 or total_weight <= 0.0:
        return None

    cumulative_weight = np.concatenate(([0.0], np.cumsum(weights)))
    cumulative_sum = np.concatenate(([0.0], np.cumsum(weights * centers)))
    cumulative_square_sum = np.concatenate(
        ([0.0], np.cumsum(weights * centers * centers))
    )
    minimum_weight = total_weight * _CBCT_MIN_CLASS_WEIGHT

    costs = np.full((class_count + 1, bin_count + 1), np.inf)
    previous = np.full(
        (class_count + 1, bin_count + 1), -1, dtype=np.int32
    )
    costs[0, 0] = 0.0

    for classes in range(1, class_count + 1):
        for end in range(classes, bin_count + 1):
            starts = np.arange(classes - 1, end)
            interval_weight = cumulative_weight[end] - cumulative_weight[starts]
            valid = (
                (interval_weight >= minimum_weight)
                & np.isfinite(costs[classes - 1, starts])
            )
            if not np.any(valid):
                continue
            starts = starts[valid]
            interval_weight = interval_weight[valid]
            interval_sum = cumulative_sum[end] - cumulative_sum[starts]
            interval_square_sum = (
                cumulative_square_sum[end] - cumulative_square_sum[starts]
            )
            within_class_error = interval_square_sum - (
                interval_sum * interval_sum / interval_weight
            )
            scores = costs[classes - 1, starts] + np.maximum(
                within_class_error, 0.0
            )
            best = int(np.argmin(scores))
            costs[classes, end] = scores[best]
            previous[classes, end] = int(starts[best])

    if not np.isfinite(costs[class_count, bin_count]):
        return None

    intervals = []
    end = bin_count
    for classes in range(class_count, 0, -1):
        start = int(previous[classes, end])
        if start < 0:
            return None
        intervals.append((start, end))
        end = start
    intervals.reverse()

    class_means = []
    class_weights = []
    separators = []
    for index, (start, end) in enumerate(intervals):
        weight = cumulative_weight[end] - cumulative_weight[start]
        value_sum = cumulative_sum[end] - cumulative_sum[start]
        class_means.append(value_sum / weight)
        class_weights.append(weight / total_weight)
        if index < len(intervals) - 1:
            separators.append(end)
    return (
        np.asarray(class_means),
        np.asarray(class_weights),
        np.asarray(separators, dtype=np.int32),
    )


def _cbct_density_statistics(
    array: np.ndarray,
) -> Optional[tuple[float, float, float]]:
    """Return lower-class mean, bone separator and high-density mean."""
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
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    maximum_classes = min(
        _CBCT_DENSITY_CLASSES, int(np.count_nonzero(histogram))
    )
    partition = None
    for class_count in range(maximum_classes, 1, -1):
        partition = _optimal_histogram_partition(
            histogram, centers, class_count
        )
        if partition is not None:
            break
    if partition is None:
        return None

    class_means, class_weights, separator_bins = partition
    class_count = int(class_means.size)
    first_upper_split = (class_count - 1) // 2
    possible_splits = list(range(first_upper_split, class_count - 1))
    preferred_splits = [
        split
        for split in possible_splits
        if _CBCT_MIN_HIGH_DENSITY_FRACTION
        <= float(class_weights[split + 1 :].sum())
        <= _CBCT_MAX_HIGH_DENSITY_FRACTION
    ]
    if not preferred_splits:
        preferred_splits = possible_splits
    split = max(
        preferred_splits,
        key=lambda index: float(class_means[index + 1] - class_means[index]),
    )

    upper_weights = class_weights[split + 1 :]
    upper_mean = float(
        np.average(class_means[split + 1 :], weights=upper_weights)
    )
    boundary = float(bin_edges[int(separator_bins[split])])
    return float(class_means[split]), boundary, upper_mean


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

    statistics = _cbct_density_statistics(values)
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
    denoise_sigma_mm: Optional[float] = None,
    closing_radius_mm: float = 1.0,
    minimum_component_volume_mm3: Optional[float] = None,
    minimum_seed_volume_mm3: Optional[float] = None,
    fully_connected: bool = True,
) -> np.ndarray:
    """Extract a three-dimensional CT or CBCT bone mask.

    High-threshold voxels form confident bone seeds. For CBCT, tiny seed
    components are removed before binary reconstruction so isolated bright
    noise cannot grow through the lower threshold. Reconstruction retains
    lower-threshold partial-volume voxels only when connected to a seed. A
    small physical-radius closing repairs narrow cortical breaks, and small
    final connected components are removed.

    All spatial parameters use millimetres. Threshold overrides use the
    source intensity units: HU for CT and scanner-specific values for CBCT.
    The returned Boolean array has the same ``(L, P, S)`` shape as ``volume``.
    """
    values = _validated_volume(volume)
    normalized_type = _validated_image_type(image_type)
    mmpd = float(mmpd)
    if not np.isfinite(mmpd) or mmpd <= 0.0:
        raise ValueError("mmpd must be finite and greater than zero")
    physical_volume_mm3 = float(values.size) * (mmpd ** 3)
    if denoise_sigma_mm is None:
        denoise_sigma_mm = 0.6 if normalized_type == "cbct" else 0.4
    if minimum_component_volume_mm3 is None:
        minimum_component_volume_mm3 = (
            max(64.0, physical_volume_mm3 * 1e-4)
            if normalized_type == "cbct"
            else 8.0
        )
    if minimum_seed_volume_mm3 is None:
        minimum_seed_volume_mm3 = (
            max(2.0, physical_volume_mm3 * 1e-6)
            if normalized_type == "cbct"
            else 0.0
        )
    denoise_sigma_mm = float(denoise_sigma_mm)
    closing_radius_mm = float(closing_radius_mm)
    minimum_component_volume_mm3 = float(minimum_component_volume_mm3)
    minimum_seed_volume_mm3 = float(minimum_seed_volume_mm3)
    for name, value in (
        ("denoise_sigma_mm", denoise_sigma_mm),
        ("closing_radius_mm", closing_radius_mm),
        ("minimum_component_volume_mm3", minimum_component_volume_mm3),
        ("minimum_seed_volume_mm3", minimum_seed_volume_mm3),
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
    minimum_seed_voxels = int(
        np.ceil(float(minimum_seed_volume_mm3) / (mmpd ** 3))
    )
    if minimum_seed_voxels > 1:
        seed_components = sitk.ConnectedComponent(
            seeds, bool(fully_connected)
        )
        seed_components = sitk.RelabelComponent(
            seed_components,
            minimum_seed_voxels,
            False,
        )
        seeds = sitk.Cast(seed_components > 0, sitk.sitkUInt8)
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
