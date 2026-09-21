import numpy as np
import pytest

from ct_mri_dicom_nii_reader import (
    apply_bone_mask,
    estimate_bone_thresholds,
    extract_bone_mask,
    extract_cbct_bone_mask,
    extract_ct_bone_mask,
)


def cortical_phantom():
    volume = np.full((17, 17, 17), -100.0, dtype=np.float32)
    volume[3:12, 3:12, 3:12] = 150.0
    volume[5:10, 5:10, 5:10] = 500.0
    volume[13:16, 13:16, 13:16] = 150.0
    return volume


def test_ct_hysteresis_recovers_connected_low_density_cortex_only():
    volume = cortical_phantom()

    mask = extract_ct_bone_mask(
        volume,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=0.0,
    )

    assert np.all(mask[3:12, 3:12, 3:12])
    assert not np.any(mask[13:16, 13:16, 13:16])


def test_small_cortical_surface_gap_is_closed_in_physical_space():
    volume = cortical_phantom()
    volume[3, 7, 7] = -100.0

    open_mask = extract_ct_bone_mask(
        volume,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=0.0,
    )
    closed_mask = extract_ct_bone_mask(
        volume,
        denoise_sigma_mm=0.0,
        closing_radius_mm=1.0,
        minimum_component_volume_mm3=0.0,
    )

    assert not open_mask[3, 7, 7]
    assert closed_mask[3, 7, 7]


def test_small_isolated_high_density_component_is_removed():
    volume = np.full((15, 15, 15), -100.0, dtype=np.float32)
    volume[4:9, 4:9, 4:9] = 500.0
    volume[12, 12, 12] = 500.0

    mask = extract_ct_bone_mask(
        volume,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=8.0,
    )

    assert np.all(mask[4:9, 4:9, 4:9])
    assert not mask[12, 12, 12]


def test_cbct_thresholds_and_mask_are_affine_intensity_invariant():
    rng = np.random.default_rng(20260920)
    volume = np.full((30, 30, 12), -250.0, dtype=np.float32)
    volume += rng.normal(0.0, 8.0, volume.shape).astype(np.float32)
    volume[4:26, 4:26, 2:10] = rng.normal(
        420.0, 25.0, (22, 22, 8)
    )
    volume[8:22, 8:22, 3:9] = rng.normal(
        1_350.0, 45.0, (14, 14, 6)
    )
    transformed = volume * 3.25 + 470.0

    low, high = estimate_bone_thresholds(volume, "cbct")
    transformed_low, transformed_high = estimate_bone_thresholds(
        transformed, "cbct"
    )
    np.testing.assert_allclose(
        (transformed_low, transformed_high),
        (low * 3.25 + 470.0, high * 3.25 + 470.0),
        rtol=1e-5,
    )

    mask = extract_cbct_bone_mask(
        volume,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=0.0,
    )
    transformed_mask = extract_cbct_bone_mask(
        transformed,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=0.0,
    )
    np.testing.assert_array_equal(mask, transformed_mask)


def test_cbct_uses_the_high_density_tail_in_a_multimodal_scan():
    rng = np.random.default_rng(41)
    values = np.concatenate(
        (
            rng.normal(-900.0, 30.0, 20_000),
            rng.normal(-350.0, 45.0, 25_000),
            rng.normal(-50.0, 40.0, 35_000),
            rng.normal(220.0, 45.0, 17_000),
            rng.normal(760.0, 70.0, 3_000),
        )
    ).reshape(100, 100, 10)

    low, high = estimate_bone_thresholds(values, "cbct")

    assert np.percentile(values[values < 400.0], 99.5) < low
    assert low < high < np.percentile(values[values > 500.0], 50.0)


def test_cbct_removes_bright_noise_without_losing_bone():
    rng = np.random.default_rng(73)
    volume = rng.normal(50.0, 25.0, (64, 64, 32)).astype(np.float32)
    bone = np.zeros(volume.shape, dtype=bool)
    bone[20:44, 20:44, 8:24] = True
    volume[bone] = rng.normal(800.0, 50.0, int(bone.sum()))

    noise = np.zeros(volume.shape, dtype=bool)
    noise_starts = ((3, 3, 3), (52, 5, 8), (5, 52, 18), (52, 52, 25))
    for start in noise_starts:
        slices = tuple(slice(value, value + 2) for value in start)
        noise[slices] = True
    volume[noise] = 1_100.0

    mask = extract_cbct_bone_mask(volume, mmpd=1.0)

    assert np.mean(mask[bone]) > 0.95
    assert not np.any(mask[noise])
    bone_neighborhood = np.zeros_like(bone)
    bone_neighborhood[19:45, 19:45, 7:25] = True
    assert np.count_nonzero(mask & ~bone_neighborhood) < 50


def test_constant_cbct_returns_an_empty_mask():
    volume = np.full((5, 5, 5), 42.0, dtype=np.float32)

    mask = extract_cbct_bone_mask(volume)

    assert mask.dtype == bool
    assert not np.any(mask)


def test_threshold_overrides_and_apply_mask_are_public_operations():
    volume = np.asarray(
        [[[0.0, 10.0], [20.0, 30.0]], [[40.0, 50.0], [60.0, 70.0]]],
        dtype=np.float32,
    )
    mask = extract_bone_mask(
        volume,
        "ct",
        low_threshold=20.0,
        high_threshold=40.0,
        denoise_sigma_mm=0.0,
        closing_radius_mm=0.0,
        minimum_component_volume_mm3=0.0,
    )

    masked = apply_bone_mask(volume, mask, outside_value=-1.0)

    np.testing.assert_array_equal(masked[mask], volume[mask])
    assert np.all(masked[~mask] == -1.0)


def test_invalid_public_bone_segmentation_arguments_are_rejected():
    volume = np.zeros((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="image_type"):
        extract_bone_mask(volume, "mri")
    with pytest.raises(ValueError, match="high_threshold"):
        extract_bone_mask(
            volume, "ct", low_threshold=300.0, high_threshold=100.0
        )
    with pytest.raises(ValueError, match="minimum_seed_volume_mm3"):
        extract_bone_mask(volume, "cbct", minimum_seed_volume_mm3=-1.0)
    with pytest.raises(ValueError, match="same shape"):
        apply_bone_mask(volume, np.zeros((2, 2, 2), dtype=bool))
