import itertools

import numpy as np

from ct_mri_dicom_nii_reader.visualization import (
    _compose_multimodal_fusion,
    _compose_two_image_layers,
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
