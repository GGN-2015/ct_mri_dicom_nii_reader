import numpy as np

from ct_mri_dicom_nii_reader import extract_bone_mask
from ct_mri_dicom_nii_reader.body_data import main as body_data_main
from ct_mri_dicom_nii_reader.body_data.main import BodyData
from ct_mri_dicom_nii_reader.body_data.body_data_imp.numpy_3d_viewer import (
    _blacken_non_bone,
    _bone_threshold,
    _estimate_cbct_bone_threshold,
    _precompute_single_frames,
)
from ct_mri_dicom_nii_reader.body_data.body_data_imp._bone_mask import (
    _precompute_packed_bone_masks,
    _unpack_bone_mask,
)


def test_bone_threshold_is_only_available_for_ct_and_cbct():
    image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    assert _bone_threshold(image, "ct") == 100.0
    assert np.isfinite(_bone_threshold(image, "cbct"))
    assert _bone_threshold(image, "mri") is None
    assert _bone_threshold(image, "mask") is None
    assert _bone_threshold(image, None) is None


def test_cbct_threshold_separates_the_high_density_class():
    rng = np.random.default_rng(20260920)
    background = rng.normal(-250.0, 15.0, 60_000)
    soft_tissue = rng.normal(420.0, 35.0, 30_000)
    bone = rng.normal(1_350.0, 70.0, 10_000)
    volume = np.concatenate((background, soft_tissue, bone)).reshape(100, 100, 10)

    threshold = _estimate_cbct_bone_threshold(volume)

    assert np.percentile(soft_tissue, 99.9) < threshold
    assert threshold < np.percentile(bone, 0.1)


def test_cbct_threshold_adapts_to_linear_pseudo_hu_changes():
    rng = np.random.default_rng(7)
    values = np.concatenate(
        (
            rng.normal(-100.0, 10.0, 50_000),
            rng.normal(250.0, 25.0, 35_000),
            rng.normal(900.0, 50.0, 15_000),
        )
    ).reshape(100, 100, 10)
    transformed = values * 3.25 + 470.0

    threshold = _estimate_cbct_bone_threshold(values)
    transformed_threshold = _estimate_cbct_bone_threshold(transformed)

    original_mask = values >= threshold
    transformed_mask = transformed >= transformed_threshold
    np.testing.assert_array_equal(original_mask, transformed_mask)


def test_non_bone_and_non_finite_pixels_are_blackened():
    values = np.asarray(
        [[-100.0, 199.9], [200.0, np.nan]], dtype=np.float32
    )
    pixels = np.asarray([[10, 20], [30, 40]], dtype=np.uint8)

    _blacken_non_bone(pixels, values, threshold=200.0)

    np.testing.assert_array_equal(
        pixels,
        np.asarray([[0, 0], [30, 0]], dtype=np.uint8),
    )


def test_constant_or_non_finite_cbct_has_no_bone_class():
    constant = np.full((3, 3, 3), 42.0, dtype=np.float32)
    non_finite = np.full((3, 3, 3), np.nan, dtype=np.float32)

    assert np.isinf(_estimate_cbct_bone_threshold(constant))
    assert np.isinf(_estimate_cbct_bone_threshold(non_finite))


def test_body_data_preview_forwards_image_type(monkeypatch):
    received = {}

    def fake_show_numpy_3d(
        array, value_min, value_max, image_type=None, mmpd=1.0
    ):
        received["array"] = array
        received["image_type"] = image_type
        received["mmpd"] = mmpd

    monkeypatch.setattr(body_data_main, "show_numpy_3d", fake_show_numpy_3d)
    body_data = BodyData()
    array = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    body_data.from_array(array, "cbct", 1.5)

    body_data.gui_preview()

    assert received["array"] is array
    assert received["image_type"] == "cbct"
    assert received["mmpd"] == 1.5


def test_ct_bone_masks_are_precomputed_for_every_slice_and_packed():
    volume = np.full((7, 7, 3), -100.0, dtype=np.float32)
    volume[1:6, 1:6, :] = 150.0
    volume[2:5, 2:5, :] = 500.0
    expected = extract_bone_mask(volume, "ct", 1.0)

    packed_masks = _precompute_packed_bone_masks(volume, "ct")

    assert packed_masks is not None
    packed, shape = packed_masks
    assert shape == volume.shape[:2]
    assert packed.shape == (volume.shape[2], 7)
    for index in range(volume.shape[2]):
        np.testing.assert_array_equal(
            _unpack_bone_mask(packed_masks, index), expected[:, :, index]
        )


def test_mri_and_mask_do_not_allocate_bone_masks():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    assert _precompute_packed_bone_masks(volume, "mri") is None
    assert _precompute_packed_bone_masks(volume, "mask") is None


def test_single_view_normal_and_bone_frames_are_fully_precomputed():
    volume = np.asarray(
        [
            [[0.0, 50.0], [200.0, 250.0]],
            [[100.0, 150.0], [300.0, np.nan]],
        ],
        dtype=np.float32,
    )

    bone_mask = np.isfinite(volume) & (volume >= 200.0)
    normal, bone = _precompute_single_frames(
        volume, 0.0, 300.0, bone_mask=bone_mask
    )

    assert normal.shape == bone.shape == (2, 2, 2)
    for index in range(volume.shape[2]):
        values = volume[:, :, index].astype(np.float64)
        expected = np.rint(
            np.nan_to_num(np.clip(values / 300.0, 0.0, 1.0), nan=0.0)
            * 255.0
        ).astype(np.uint8)
        expected_bone = expected.copy()
        expected_bone[~(np.isfinite(values) & (values >= 200.0))] = 0
        np.testing.assert_array_equal(normal[index], expected.T)
        np.testing.assert_array_equal(bone[index], expected_bone.T)
