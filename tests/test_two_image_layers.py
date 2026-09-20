import itertools
from pathlib import Path

import numpy as np

import ct_mri_dicom_nii_reader.visualization as visualization_module
from ct_mri_dicom_nii_reader import BodyData
from ct_mri_dicom_nii_reader.body_data.body_data_imp._bone_mask import (
    _precompute_packed_bone_masks,
)
from ct_mri_dicom_nii_reader.visualization import (
    _EDGE_FUSION_OPACITY,
    _FIXED_EDGE_COLOR,
    _MOVING_EDGE_COLOR,
    MultimodalFusionComposer,
    _SliceRenderWorker,
    _compose_edge_fusion_u8,
    _compose_multimodal_fusion,
    _compose_two_image_layers,
    _compose_two_image_layers_u8,
    _edge_strength,
    _precompute_three_image_frames,
    _precompute_two_image_frames,
    _release_frame_cache,
    normalize_to_u8,
)


FIXED = np.asarray(
    [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]], dtype=np.float32
)
MOVING = np.asarray(
    [[100.0, 125.0, 150.0], [175.0, 200.0, 225.0]],
    dtype=np.float32,
)
FIXED_WINDOW = (0.0, 10.0)
MOVING_WINDOW = (100.0, 225.0)


def compose(show_fixed, show_moving, show_boundary):
    return _compose_two_image_layers(
        FIXED,
        MOVING,
        FIXED_WINDOW,
        MOVING_WINDOW,
        show_fixed,
        show_moving,
        show_boundary,
    )


def body_data(array, image_type):
    result = BodyData()
    result.from_array(np.asarray(array, dtype=np.float32), image_type, 1.0)
    return result


def test_all_layer_combinations_are_rgb_u8():
    for layers in itertools.product((False, True), repeat=3):
        result = compose(*layers)
        assert result.shape == (*FIXED.shape, 3)
        assert result.dtype == np.uint8


def test_no_selected_image_is_black_regardless_of_boundary():
    assert not np.any(compose(False, False, False))
    assert not np.any(compose(False, False, True))


def test_single_images_are_true_rgb_grayscale():
    for show_fixed, show_moving in ((True, False), (False, True)):
        result = compose(show_fixed, show_moving, False)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])


def test_single_image_boundary_is_grayscale_brightening():
    for show_fixed, show_moving in ((True, False), (False, True)):
        plain = compose(show_fixed, show_moving, False)
        boundary = compose(show_fixed, show_moving, True)
        assert np.all(boundary >= plain)
        assert np.any(boundary > plain)
        np.testing.assert_array_equal(boundary[:, :, 0], boundary[:, :, 1])
        np.testing.assert_array_equal(boundary[:, :, 1], boundary[:, :, 2])


def test_default_selection_preserves_original_fusion_exactly():
    expected = _compose_multimodal_fusion(
        FIXED, MOVING, FIXED_WINDOW, MOVING_WINDOW
    )
    np.testing.assert_array_equal(compose(True, True, True), expected)


def test_two_image_fusion_normalizes_each_modality_independently():
    result = compose(True, True, False)
    fixed_u8 = normalize_to_u8(FIXED, FIXED_WINDOW)
    moving_u8 = normalize_to_u8(MOVING, MOVING_WINDOW)

    assert fixed_u8[0, 0] == moving_u8[0, 0] == 0
    assert fixed_u8[-1, -1] == moving_u8[-1, -1] == 255
    assert np.any(result[:, :, 0] != result[:, :, 2])

    fixed_rescaled = _compose_two_image_layers(
        FIXED * 100.0,
        MOVING,
        (FIXED_WINDOW[0] * 100.0, FIXED_WINDOW[1] * 100.0),
        MOVING_WINDOW,
        True,
        True,
        False,
    )
    np.testing.assert_array_equal(result, fixed_rescaled)


def test_bone_masks_are_applied_to_each_fusion_component_separately():
    fixed_u8 = normalize_to_u8(FIXED, FIXED_WINDOW)
    moving_u8 = normalize_to_u8(MOVING, MOVING_WINDOW)
    fixed_edge = _edge_strength(fixed_u8)
    moving_edge = _edge_strength(moving_u8)
    fixed_mask = np.asarray(
        [[True, False, True], [False, True, False]], dtype=bool
    )
    moving_mask = np.asarray(
        [[False, True, False], [True, False, True]], dtype=bool
    )

    actual = _compose_edge_fusion_u8(
        fixed_u8,
        moving_u8,
        fixed_edge,
        moving_edge,
        bone_only=True,
        fixed_bone_mask=fixed_mask,
        moving_bone_mask=moving_mask,
    )

    fixed_component = (
        np.repeat(fixed_u8[:, :, None], 3, axis=2).astype(np.float32)
        * 0.32
        + fixed_edge[:, :, None] * _FIXED_EDGE_COLOR
    )
    moving_component = (
        moving_edge[:, :, None]
        * _EDGE_FUSION_OPACITY
        * _MOVING_EDGE_COLOR
    )
    expected = np.where(
        fixed_mask[:, :, None], fixed_component, 0.0
    ) + np.where(moving_mask[:, :, None], moving_component, 0.0)
    expected = np.ascontiguousarray(
        np.clip(expected, 0.0, 255.0), dtype=np.uint8
    )
    np.testing.assert_array_equal(actual, expected)


def test_boundary_is_brightened_before_single_image_bone_mask():
    fixed_u8 = normalize_to_u8(FIXED, FIXED_WINDOW)
    moving_u8 = normalize_to_u8(MOVING, MOVING_WINDOW)
    fixed_edge = _edge_strength(fixed_u8)
    fixed_mask = np.asarray(
        [[True, False, True], [False, True, False]], dtype=bool
    )

    actual = _compose_two_image_layers_u8(
        fixed_u8,
        moving_u8,
        True,
        False,
        True,
        fixed_edge=fixed_edge,
        bone_only=True,
        fixed_bone_mask=fixed_mask,
    )

    brightened = (
        np.repeat(fixed_u8[:, :, None], 3, axis=2).astype(np.float32)
        + fixed_edge[:, :, None] * (255.0 * _EDGE_FUSION_OPACITY)
    )
    expected = np.where(fixed_mask[:, :, None], brightened, 0.0)
    expected = np.ascontiguousarray(
        np.clip(expected, 0.0, 255.0), dtype=np.uint8
    )
    np.testing.assert_array_equal(actual, expected)


def test_missing_bone_mask_leaves_that_modality_unchanged():
    fixed_u8 = normalize_to_u8(FIXED, FIXED_WINDOW)
    moving_u8 = normalize_to_u8(MOVING, MOVING_WINDOW)
    moving_mask = np.asarray(
        [[False, True, False], [True, False, True]], dtype=bool
    )

    without_bone = _compose_two_image_layers_u8(
        fixed_u8, moving_u8, True, True, False
    )
    mri_only_bone = _compose_two_image_layers_u8(
        fixed_u8,
        moving_u8,
        True,
        False,
        False,
        bone_only=True,
        fixed_bone_mask=None,
    )
    mri_plain = _compose_two_image_layers_u8(
        fixed_u8, moving_u8, True, False, False
    )
    mixed = _compose_two_image_layers_u8(
        fixed_u8,
        moving_u8,
        True,
        True,
        False,
        bone_only=True,
        fixed_bone_mask=None,
        moving_bone_mask=moving_mask,
    )

    np.testing.assert_array_equal(mri_only_bone, mri_plain)
    assert np.any(mixed != without_bone)


def test_every_precomputed_two_image_state_matches_original_renderer():
    fixed = body_data(
        np.asarray(
            [
                [[-100.0, 220.0], [300.0, 50.0], [450.0, 180.0]],
                [[700.0, 100.0], [250.0, 800.0], [-50.0, 500.0]],
            ]
        ),
        "ct",
    )
    moving = body_data(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2), "mri"
    )
    composer = MultimodalFusionComposer(fixed, moving)
    bone_masks = (
        _precompute_packed_bone_masks(
            np.asarray(composer.get_fixed()._body_data), "ct"
        ),
        None,
    )
    worker = _SliceRenderWorker(composer, bone_masks=bone_masks)
    cache, cache_directory = _precompute_two_image_frames(composer)
    try:
        for layers in itertools.product((False, True), repeat=4):
            for index in range(composer.get_size()[2]):
                expected = worker._make_two_image_slice(
                    "s", index, layers
                )
                actual = cache[layers][index]
                if actual.ndim == 2:
                    actual = np.repeat(actual[:, :, None], 3, axis=2)
                np.testing.assert_array_equal(
                    actual, expected
                )
    finally:
        worker.stop()
        _release_frame_cache(list(cache.values()), cache_directory)


def test_precomputed_three_image_frames_match_original_renderer():
    fixed = body_data(
        np.linspace(-500.0, 900.0, 24).reshape(2, 3, 4), "ct"
    )
    moving = body_data(
        np.linspace(0.0, 1.0, 24).reshape(2, 3, 4), "mri"
    )
    mask = body_data(
        np.asarray(
            [
                [[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0]],
                [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
            ]
        ),
        "mask",
    )
    composer = MultimodalFusionComposer(fixed, moving, mask)

    frames, cache_directory = _precompute_three_image_frames(composer)
    try:
        for index in range(composer.get_size()[2]):
            np.testing.assert_array_equal(
                frames[index], composer.make_slice("s", index)
            )
    finally:
        _release_frame_cache([frames], cache_directory)


def test_large_frame_cache_uses_and_releases_temporary_memory_maps(
    monkeypatch,
):
    monkeypatch.setattr(
        visualization_module, "_FRAME_CACHE_RAM_LIMIT_BYTES", 0
    )
    fixed = body_data(
        np.linspace(-500.0, 900.0, 24).reshape(2, 3, 4), "ct"
    )
    moving = body_data(
        np.linspace(0.0, 1.0, 24).reshape(2, 3, 4), "mri"
    )
    composer = MultimodalFusionComposer(fixed, moving)

    cache, cache_directory = _precompute_two_image_frames(composer)
    assert cache_directory is not None
    cache_path = Path(cache_directory.name)
    assert cache_path.is_dir()
    assert any(isinstance(array, np.memmap) for array in cache.values())

    _release_frame_cache(list(cache.values()), cache_directory)

    assert not cache_path.exists()
