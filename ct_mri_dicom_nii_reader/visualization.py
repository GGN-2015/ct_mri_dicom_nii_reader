"""Visualization classes for :class:`BodyData` medical image volumes.

The rendering approach mirrors the visualization of the regknee project:

* :class:`BodyDataCommonGrid` aligns any number of volumes at the LPS origin,
  resamples them onto the finest unified voxel spacing and places them into
  one common NumPy grid. Blank regions are filled with the modality-specific
  air value (-1024 for CT/CBCT, 0 for MRI and masks).
* :class:`MultimodalFusionComposer` renders two aligned volumes per slice
  using the reference amber/cyan edge fusion (where both edges overlap the
  colors add up to white), with an optional semi-transparent red mask overlay
  for a third volume.
* :class:`TwoImageFusionViewer` and :class:`ThreeImageOverlayViewer` open the
  corresponding preview windows with a bottom slice slider. Tkinter is
  imported lazily and remains an optional dependency.

Single-volume grayscale preview with a bottom slice slider is already
available as :meth:`BodyData.gui_preview`.
"""

from __future__ import annotations

import queue
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .body_data import BodyData, BodyDataNotInitialized
from .body_data.body_data_imp._bone_mask import (
    PackedBoneMasks,
    _precompute_packed_bone_masks,
    _unpack_bone_mask,
)


_FIXED_EDGE_COLOR = np.asarray((220.0, 145.0, 28.0), dtype=np.float32)
_MOVING_EDGE_COLOR = np.asarray((20.0, 225.0, 255.0), dtype=np.float32)
_EDGE_FUSION_OPACITY = 0.48
_MASK_OVERLAY_COLOR = np.asarray((255.0, 0.0, 0.0), dtype=np.float32)
_MASK_OVERLAY_MAX_ALPHA = 0.5
_FRAME_CACHE_RAM_LIMIT_BYTES = 512 * 1024 * 1024

LayerSelection = tuple[bool, bool, bool, bool]
FrameCache = dict[LayerSelection, np.ndarray]


def _air_value(image_type: Optional[str]) -> float:
    return -1024.0 if image_type in {"ct", "cbct"} else 0.0


def display_window(body_data: BodyData) -> tuple[float, float]:
    """Estimate the bounded display window used by multimodal fusion."""
    flat = np.asarray(body_data._body_data).reshape(-1)
    stride = max(1, flat.size // 500_000)
    sample = np.asarray(flat[::stride], dtype=np.float32)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    sample_min = float(np.min(sample))
    sample_max = float(np.max(sample))
    is_unit_scaled = sample_min >= -1e-6 and sample_max <= 1.0 + 1e-6
    if body_data.get_type() == "mask":
        low = 0.0
        high = max(1.0, sample_max)
    elif is_unit_scaled:
        # MIND values are normalized to [0, 1] but retain the source modality.
        low = max(0.0, float(np.percentile(sample, 1.0)))
        high = min(1.0, float(np.percentile(sample, 99.5)))
        if high <= low + 1e-6:
            return 0.0, 1.0
    elif body_data.get_type() in {"ct", "cbct"}:
        low = max(-250.0, float(np.percentile(sample, 1.0)))
        high = max(350.0, float(np.percentile(sample, 99.5)))
    else:
        non_background = sample[sample > float(np.percentile(sample, 2.0))]
        if non_background.size >= 64:
            sample = non_background
        low = float(np.percentile(sample, 1.0))
        high = float(np.percentile(sample, 99.5))
    if high <= low:
        high = low + 1.0
    return low, high


def _display_window(body_data: BodyData) -> tuple[float, float]:
    """Compatibility forwarder for :func:`display_window`."""
    return display_window(body_data)


def normalize_to_u8(
    values: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    low, high = window
    normalized = (
        np.asarray(values, dtype=np.float32) - low
    ) / max(high - low, 1e-6)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(
        np.clip(normalized * 255.0, 0.0, 255.0), dtype=np.uint8
    )


def _normalize_to_u8(
    values: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    """Compatibility forwarder for :func:`normalize_to_u8`."""
    return normalize_to_u8(values, window)


def _edge_strength(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    gradient_y = (
        np.gradient(array, axis=0)
        if array.shape[0] > 1 else np.zeros_like(array)
    )
    gradient_x = (
        np.gradient(array, axis=1)
        if array.shape[1] > 1 else np.zeros_like(array)
    )
    magnitude = np.hypot(gradient_x, gradient_y)
    nonzero = magnitude[magnitude > 0.0]
    if nonzero.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    scale = max(float(np.percentile(nonzero, 90.0)), 1.0)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def _compose_multimodal_fusion(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_window: tuple[float, float],
    moving_window: tuple[float, float],
) -> np.ndarray:
    """Compose the reference amber/cyan edge fusion as an RGB image."""
    fixed_u8 = normalize_to_u8(fixed, fixed_window)
    moving_u8 = normalize_to_u8(moving, moving_window)
    return _compose_edge_fusion_u8(fixed_u8, moving_u8)


def _compose_edge_fusion_u8(
    fixed_u8: np.ndarray,
    moving_u8: np.ndarray,
    fixed_edge: Optional[np.ndarray] = None,
    moving_edge: Optional[np.ndarray] = None,
    bone_only: bool = False,
    fixed_bone_mask: Optional[np.ndarray] = None,
    moving_bone_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose the original fusion from independently normalized inputs."""
    fixed_rgb = np.repeat(fixed_u8[:, :, None], 3, axis=2).astype(np.float32)
    if fixed_edge is None:
        fixed_edge = _edge_strength(fixed_u8)
    if moving_edge is None:
        moving_edge = _edge_strength(moving_u8)

    fixed_component = fixed_rgb * 0.32
    fixed_component += (
        fixed_edge[:, :, None]
        * _FIXED_EDGE_COLOR
    )
    moving_component = (
        moving_edge[:, :, None]
        * _EDGE_FUSION_OPACITY
        * _MOVING_EDGE_COLOR
    )
    if bone_only:
        if fixed_bone_mask is not None:
            np.multiply(
                fixed_component,
                fixed_bone_mask[:, :, None],
                out=fixed_component,
            )
        if moving_bone_mask is not None:
            np.multiply(
                moving_component,
                moving_bone_mask[:, :, None],
                out=moving_component,
            )
    fusion = fixed_component
    fusion += moving_component
    return np.ascontiguousarray(
        np.clip(fusion, 0.0, 255.0), dtype=np.uint8
    )


def _compose_two_image_layers_u8(
    fixed_u8: np.ndarray,
    moving_u8: np.ndarray,
    show_fixed: bool,
    show_moving: bool,
    show_boundary: bool,
    fixed_edge: Optional[np.ndarray] = None,
    moving_edge: Optional[np.ndarray] = None,
    bone_only: bool = False,
    fixed_bone_mask: Optional[np.ndarray] = None,
    moving_bone_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose one of the layer selections in the two-image viewer."""
    if not show_fixed and not show_moving:
        return np.zeros((*fixed_u8.shape, 3), dtype=np.uint8)

    if show_fixed and show_moving:
        if show_boundary:
            return _compose_edge_fusion_u8(
                fixed_u8,
                moving_u8,
                fixed_edge,
                moving_edge,
                bone_only,
                fixed_bone_mask,
                moving_bone_mask,
            )

        fixed_color = (
            fixed_u8[:, :, None].astype(np.float32)
            * (_FIXED_EDGE_COLOR / 255.0)
        )
        moving_color = (
            moving_u8[:, :, None].astype(np.float32)
            * (_MOVING_EDGE_COLOR / 255.0)
        )
        if bone_only:
            if fixed_bone_mask is not None:
                np.multiply(
                    fixed_color,
                    fixed_bone_mask[:, :, None],
                    out=fixed_color,
                )
            if moving_bone_mask is not None:
                np.multiply(
                    moving_color,
                    moving_bone_mask[:, :, None],
                    out=moving_color,
                )
        return np.ascontiguousarray(
            np.clip(fixed_color + moving_color, 0.0, 255.0),
            dtype=np.uint8,
        )

    grayscale = fixed_u8 if show_fixed else moving_u8
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    if show_boundary:
        edge = fixed_edge if show_fixed else moving_edge
        if edge is None:
            edge = _edge_strength(grayscale)
        rgb = (
            rgb.astype(np.float32)
            + edge[:, :, None] * (255.0 * _EDGE_FUSION_OPACITY)
        )

    selected_bone_mask = (
        fixed_bone_mask if show_fixed else moving_bone_mask
    )
    if bone_only and selected_bone_mask is not None:
        np.multiply(rgb, selected_bone_mask[:, :, None], out=rgb)
    return np.ascontiguousarray(
        np.clip(rgb, 0.0, 255.0), dtype=np.uint8
    )


def _compose_two_image_layers(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_window: tuple[float, float],
    moving_window: tuple[float, float],
    show_fixed: bool,
    show_moving: bool,
    show_boundary: bool,
) -> np.ndarray:
    """Normalize both modalities independently and compose selected layers."""
    fixed_u8 = normalize_to_u8(fixed, fixed_window)
    moving_u8 = normalize_to_u8(moving, moving_window)
    return _compose_two_image_layers_u8(
        fixed_u8,
        moving_u8,
        show_fixed,
        show_moving,
        show_boundary,
    )


def _overlay_red_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    mask_min: float,
    mask_max: float,
) -> np.ndarray:
    """Blend a semi-transparent red mask over an RGB fusion image.

    The smallest mask value maps to a fully transparent overlay and the
    largest maps to 50% opacity; the blending happens in image (RGB) space.
    """
    if mask_max <= mask_min:
        return rgb
    normalized = (
        np.asarray(mask, dtype=np.float32) - mask_min
    ) / (mask_max - mask_min)
    alpha = np.clip(
        normalized * _MASK_OVERLAY_MAX_ALPHA,
        0.0,
        _MASK_OVERLAY_MAX_ALPHA,
    )
    blended = (
        rgb.astype(np.float32) * (1.0 - alpha[:, :, None])
        + _MASK_OVERLAY_COLOR * alpha[:, :, None]
    )
    return np.ascontiguousarray(
        np.clip(blended, 0.0, 255.0), dtype=np.uint8
    )


def _take_slice(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "l":
        return arr[index, :, :]
    if axis == "p":
        return arr[:, index, :]
    if axis == "s":
        return arr[:, :, index]
    raise ValueError("axis must be one of 'l', 'p', 's'")


def _frame_cache_directory(
    total_bytes: int,
) -> Optional[tempfile.TemporaryDirectory]:
    if total_bytes <= _FRAME_CACHE_RAM_LIMIT_BYTES:
        return None
    return tempfile.TemporaryDirectory(prefix="ct_mri_reader_frames_")


def _allocate_frame_volume(
    shape: tuple[int, ...],
    cache_directory: Optional[tempfile.TemporaryDirectory],
    name: str,
) -> np.ndarray:
    if cache_directory is None:
        return np.empty(shape, dtype=np.uint8)
    path = Path(cache_directory.name) / f"{name}.dat"
    return np.memmap(path, mode="w+", dtype=np.uint8, shape=shape)


def _release_frame_cache(
    arrays: list[np.ndarray],
    cache_directory: Optional[tempfile.TemporaryDirectory],
) -> None:
    seen = set()
    for array in arrays:
        owner = array
        while isinstance(getattr(owner, "base", None), np.ndarray):
            owner = owner.base
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        if isinstance(owner, np.memmap):
            owner.flush()
            mmap = getattr(owner, "_mmap", None)
            if mmap is not None:
                mmap.close()
    if cache_directory is not None:
        cache_directory.cleanup()


def _single_grayscale_frame(
    image: np.ndarray,
    edge: np.ndarray,
    show_boundary: bool,
    bone_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if show_boundary:
        frame = image.astype(np.float32)
        frame += edge * (255.0 * _EDGE_FUSION_OPACITY)
        frame = np.ascontiguousarray(
            np.clip(frame, 0.0, 255.0), dtype=np.uint8
        )
    else:
        frame = image.copy()
    if bone_mask is not None:
        np.multiply(frame, bone_mask, out=frame)
    return frame.T


def _precompute_two_image_frames(
    composer: "MultimodalFusionComposer",
) -> tuple[FrameCache, Optional[tempfile.TemporaryDirectory]]:
    """Render every dual-view layer state before creating the Tk window."""
    fixed = composer.get_fixed()
    moving = composer.get_moving()
    fixed_array = np.asarray(fixed._body_data)
    moving_array = np.asarray(moving._body_data)
    size_l, size_p, size_s = composer.get_size()

    bone_masks = (
        _precompute_packed_bone_masks(
            fixed_array, fixed.get_type(), fixed.get_mmpd()
        ),
        _precompute_packed_bone_masks(
            moving_array, moving.get_type(), moving.get_mmpd()
        ),
    )
    has_fixed_bone = bone_masks[0] is not None
    has_moving_bone = bone_masks[1] is not None
    voxel_count = size_l * size_p * size_s
    scalar_volume_count = (
        4 + 2 * int(has_fixed_bone) + 2 * int(has_moving_bone)
    )
    rgb_volume_count = 2 + 2 * int(
        has_fixed_bone or has_moving_bone
    )
    total_bytes = voxel_count * (
        scalar_volume_count + 3 * rgb_volume_count
    )
    cache_directory = _frame_cache_directory(total_bytes)

    scalar_shape = (size_s, size_p, size_l)
    rgb_shape = (*scalar_shape, 3)
    fixed_plain = _allocate_frame_volume(
        scalar_shape, cache_directory, "fixed_plain"
    )
    fixed_boundary = _allocate_frame_volume(
        scalar_shape, cache_directory, "fixed_boundary"
    )
    moving_plain = _allocate_frame_volume(
        scalar_shape, cache_directory, "moving_plain"
    )
    moving_boundary = _allocate_frame_volume(
        scalar_shape, cache_directory, "moving_boundary"
    )
    both_plain = _allocate_frame_volume(
        rgb_shape, cache_directory, "both_plain"
    )
    both_boundary = _allocate_frame_volume(
        rgb_shape, cache_directory, "both_boundary"
    )

    fixed_bone_plain = fixed_plain
    fixed_bone_boundary = fixed_boundary
    if has_fixed_bone:
        fixed_bone_plain = _allocate_frame_volume(
            scalar_shape, cache_directory, "fixed_bone_plain"
        )
        fixed_bone_boundary = _allocate_frame_volume(
            scalar_shape, cache_directory, "fixed_bone_boundary"
        )

    moving_bone_plain = moving_plain
    moving_bone_boundary = moving_boundary
    if has_moving_bone:
        moving_bone_plain = _allocate_frame_volume(
            scalar_shape, cache_directory, "moving_bone_plain"
        )
        moving_bone_boundary = _allocate_frame_volume(
            scalar_shape, cache_directory, "moving_bone_boundary"
        )

    both_bone_plain = both_plain
    both_bone_boundary = both_boundary
    if has_fixed_bone or has_moving_bone:
        both_bone_plain = _allocate_frame_volume(
            rgb_shape, cache_directory, "both_bone_plain"
        )
        both_bone_boundary = _allocate_frame_volume(
            rgb_shape, cache_directory, "both_bone_boundary"
        )

    for index in range(size_s):
        fixed_u8 = normalize_to_u8(
            fixed_array[:, :, index], composer._fixed_window
        )
        moving_u8 = normalize_to_u8(
            moving_array[:, :, index], composer._moving_window
        )
        fixed_edge = _edge_strength(fixed_u8)
        moving_edge = _edge_strength(moving_u8)
        fixed_plain[index] = _single_grayscale_frame(
            fixed_u8, fixed_edge, False
        )
        fixed_boundary[index] = _single_grayscale_frame(
            fixed_u8, fixed_edge, True
        )
        moving_plain[index] = _single_grayscale_frame(
            moving_u8, moving_edge, False
        )
        moving_boundary[index] = _single_grayscale_frame(
            moving_u8, moving_edge, True
        )
        both_plain[index] = np.transpose(
            _compose_two_image_layers_u8(
                fixed_u8,
                moving_u8,
                True,
                True,
                False,
                fixed_edge,
                moving_edge,
            ),
            (1, 0, 2),
        )
        both_boundary[index] = np.transpose(
            _compose_two_image_layers_u8(
                fixed_u8,
                moving_u8,
                True,
                True,
                True,
                fixed_edge,
                moving_edge,
            ),
            (1, 0, 2),
        )

        fixed_bone_mask = (
            _unpack_bone_mask(bone_masks[0], index)
            if bone_masks[0] is not None
            else None
        )
        moving_bone_mask = (
            _unpack_bone_mask(bone_masks[1], index)
            if bone_masks[1] is not None
            else None
        )
        if has_fixed_bone:
            fixed_bone_plain[index] = _single_grayscale_frame(
                fixed_u8, fixed_edge, False, fixed_bone_mask
            )
            fixed_bone_boundary[index] = _single_grayscale_frame(
                fixed_u8, fixed_edge, True, fixed_bone_mask
            )
        if has_moving_bone:
            moving_bone_plain[index] = _single_grayscale_frame(
                moving_u8, moving_edge, False, moving_bone_mask
            )
            moving_bone_boundary[index] = _single_grayscale_frame(
                moving_u8, moving_edge, True, moving_bone_mask
            )
        if has_fixed_bone or has_moving_bone:
            both_bone_plain[index] = np.transpose(
                _compose_two_image_layers_u8(
                    fixed_u8,
                    moving_u8,
                    True,
                    True,
                    False,
                    fixed_edge,
                    moving_edge,
                    True,
                    fixed_bone_mask,
                    moving_bone_mask,
                ),
                (1, 0, 2),
            )
            both_bone_boundary[index] = np.transpose(
                _compose_two_image_layers_u8(
                    fixed_u8,
                    moving_u8,
                    True,
                    True,
                    True,
                    fixed_edge,
                    moving_edge,
                    True,
                    fixed_bone_mask,
                    moving_bone_mask,
                ),
                (1, 0, 2),
            )

    black = np.broadcast_to(
        np.zeros((1, size_p, size_l, 3), dtype=np.uint8),
        rgb_shape,
    )
    cache: FrameCache = {}
    for show_fixed in (False, True):
        for show_moving in (False, True):
            for show_boundary in (False, True):
                for show_bone in (False, True):
                    key = (
                        show_fixed,
                        show_moving,
                        show_boundary,
                        show_bone,
                    )
                    if not show_fixed and not show_moving:
                        cache[key] = black
                    elif show_fixed and not show_moving:
                        if show_bone:
                            cache[key] = (
                                fixed_bone_boundary
                                if show_boundary
                                else fixed_bone_plain
                            )
                        else:
                            cache[key] = (
                                fixed_boundary
                                if show_boundary
                                else fixed_plain
                            )
                    elif show_moving and not show_fixed:
                        if show_bone:
                            cache[key] = (
                                moving_bone_boundary
                                if show_boundary
                                else moving_bone_plain
                            )
                        else:
                            cache[key] = (
                                moving_boundary
                                if show_boundary
                                else moving_plain
                            )
                    elif show_bone:
                        cache[key] = (
                            both_bone_boundary
                            if show_boundary
                            else both_bone_plain
                        )
                    else:
                        cache[key] = (
                            both_boundary if show_boundary else both_plain
                        )
    return cache, cache_directory


def _precompute_three_image_frames(
    composer: "MultimodalFusionComposer",
) -> tuple[np.ndarray, Optional[tempfile.TemporaryDirectory]]:
    """Render every final three-image slice before creating the window."""
    size_l, size_p, size_s = composer.get_size()
    shape = (size_s, size_p, size_l, 3)
    total_bytes = size_l * size_p * size_s * 3
    cache_directory = _frame_cache_directory(total_bytes)
    frames = _allocate_frame_volume(shape, cache_directory, "three_image")
    for index in range(size_s):
        frames[index] = composer.make_slice("s", index)
    return frames, cache_directory


def _resampled_size(
    size: Tuple[int, int, int], old_mmpd: float, new_mmpd: float
) -> Tuple[int, int, int]:
    """Voxel count of a volume on the unified grid, preserving its extent."""
    return tuple(
        max(1, int(round((n - 1) * old_mmpd / new_mmpd)) + 1)
        for n in size
    )


def resample_to_mmpd(body_data: BodyData, mmpd: float) -> BodyData:
    """Resample one volume onto the unified grid spacing, keeping its extent."""
    import SimpleITK as sitk

    if np.isclose(body_data.get_mmpd(), mmpd):
        return body_data

    source_array = np.asarray(body_data._body_data, dtype=np.float32)
    source = sitk.GetImageFromArray(
        np.ascontiguousarray(np.transpose(source_array, (2, 1, 0)))
    )
    source.SetSpacing([float(body_data.get_mmpd())] * 3)
    source.SetOrigin([0.0, 0.0, 0.0])
    source.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    target_size = list(
        _resampled_size(body_data.get_size(), body_data.get_mmpd(), mmpd)
    )
    reference = sitk.Image(target_size, sitk.sitkFloat32)
    reference.SetSpacing([float(mmpd)] * 3)
    reference.SetOrigin([0.0, 0.0, 0.0])
    reference.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
        if body_data.get_type() == "mask"
        else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(_air_value(body_data.get_type()))
    output = resampler.Execute(source)

    array = np.ascontiguousarray(
        np.transpose(sitk.GetArrayFromImage(output), (2, 1, 0)),
        dtype=np.float32,
    )
    image_type = body_data.get_type()
    assert image_type is not None
    result = BodyData()
    result.from_array(array, image_type, float(mmpd))
    return result


def _resample_to_mmpd(body_data: BodyData, mmpd: float) -> BodyData:
    """Compatibility forwarder for :func:`resample_to_mmpd`."""
    return resample_to_mmpd(body_data, mmpd)


class BodyDataCommonGrid:
    """Align several volumes at the LPS origin on one common voxel grid.

    Every volume is first resampled onto the finest unified voxel spacing
    (the smallest ``mmpd`` among the inputs) and then copied into a common
    array whose axis lengths are the per-axis maxima of the resampled
    volumes (the unified grid therefore ends up slightly larger than any
    individual volume). The volumes keep their [0, 0, 0] voxel at the
    common origin; regions covered by no volume hold the modality-specific
    air value (-1024 for CT, 0 for MRI and masks).
    """

    def __init__(self, body_data_list: List[BodyData]) -> None:
        if not body_data_list:
            raise ValueError("body_data_list must not be empty")
        for body_data in body_data_list:
            if not body_data.get_initialized():
                raise BodyDataNotInitialized()
        self._body_data_list = list(body_data_list)

    def get_body_data_list(self) -> List[BodyData]:
        return list(self._body_data_list)

    def get_unified_mmpd(self) -> float:
        """The finest voxel spacing among the inputs."""
        return min(
            body_data.get_mmpd() for body_data in self._body_data_list
        )

    def get_common_size(self) -> Tuple[int, int, int]:
        """The (L, P, S) size of the common grid."""
        unified_mmpd = self.get_unified_mmpd()
        size_l, size_p, size_s = 0, 0, 0
        for body_data in self._body_data_list:
            resampled = _resampled_size(
                body_data.get_size(), body_data.get_mmpd(), unified_mmpd
            )
            size_l = max(size_l, resampled[0])
            size_p = max(size_p, resampled[1])
            size_s = max(size_s, resampled[2])
        return (size_l, size_p, size_s)

    def get_aligned(self) -> List[BodyData]:
        """Return origin-aligned copies sharing the common grid.

        All returned volumes have :meth:`BodyData.get_size` equal to
        :meth:`get_common_size` and ``mmpd`` equal to
        :meth:`get_unified_mmpd`.
        """
        unified_mmpd = self.get_unified_mmpd()
        common_size = self.get_common_size()
        aligned: List[BodyData] = []
        for body_data in self._body_data_list:
            resampled = resample_to_mmpd(body_data, unified_mmpd)
            array = np.full(
                common_size,
                _air_value(resampled.get_type()),
                dtype=np.float32,
            )
            l, p, s = resampled.get_size()
            array[:l, :p, :s] = np.asarray(resampled._body_data)
            image_type = resampled.get_type()
            assert image_type is not None
            new_body_data = BodyData()
            new_body_data.from_array(array, image_type, unified_mmpd)
            aligned.append(new_body_data)
        return aligned


class MultimodalFusionComposer:
    """Compose per-slice visualizations of two or three volumes.

    The first two volumes are rendered with the reference amber/cyan edge
    fusion: the fixed volume contributes a dimmed grayscale background with
    amber edges, the moving volume contributes cyan edges, and where both
    edges overlap the colors add up to white. When ``mask`` is given, its
    values are blended as a red overlay on top of the fusion: the smallest
    mask value maps to a fully transparent overlay and the largest to 50%
    opacity. All volumes are aligned by an internal
    :class:`BodyDataCommonGrid` first.
    """

    def __init__(
        self,
        bd1: BodyData,
        bd2: BodyData,
        mask: Optional[BodyData] = None,
    ) -> None:
        volumes = [bd1, bd2] + ([] if mask is None else [mask])
        grid = BodyDataCommonGrid(volumes)
        aligned = grid.get_aligned()
        self._grid = grid
        self._fixed = aligned[0]
        self._moving = aligned[1]
        self._mask = aligned[2] if mask is not None else None
        self._fixed_window = display_window(self._fixed)
        self._moving_window = display_window(self._moving)
        self._mask_min = 0.0
        self._mask_max = 1.0
        if self._mask is not None:
            mask_array = np.asarray(self._mask._body_data, dtype=np.float32)
            self._mask_min = float(np.min(mask_array))
            self._mask_max = float(np.max(mask_array))

    def get_common_grid(self) -> BodyDataCommonGrid:
        return self._grid

    def get_fixed(self) -> BodyData:
        return self._fixed

    def get_moving(self) -> BodyData:
        return self._moving

    def get_mask(self) -> Optional[BodyData]:
        return self._mask

    def get_size(self) -> Tuple[int, int, int]:
        """The (L, P, S) size shared by all aligned volumes."""
        return self._fixed.get_size()

    def make_slice(self, axis: str, index: int) -> np.ndarray:
        """Return a PIL-ready ``(height, width, 3)`` uint8 RGB slice."""
        fixed_slice = _take_slice(
            np.asarray(self._fixed._body_data), axis, index
        )
        moving_slice = _take_slice(
            np.asarray(self._moving._body_data), axis, index
        )
        rgb = _compose_multimodal_fusion(
            fixed_slice,
            moving_slice,
            self._fixed_window,
            self._moving_window,
        )
        if self._mask is not None:
            mask_slice = _take_slice(
                np.asarray(self._mask._body_data), axis, index
            )
            rgb = _overlay_red_mask(
                rgb, mask_slice, self._mask_min, self._mask_max
            )
        return np.transpose(rgb, (1, 0, 2))


class _SliceRenderWorker:
    """Background worker rendering fused RGB slices for the preview GUI.

    The GUI thread only records the latest requested ``(axis, index)`` pair
    together with a monotonic generation counter; the worker picks up the
    newest request whenever it becomes free, so intermediate indices are
    dropped automatically. Finished results are tagged with their generation
    and index and handed back through a queue; the GUI rejects stale results
    by comparing them against the newest request. Bounded LRU caches retain
    normalized components, edges, and completed layer combinations. Bone
    masks are computed for the whole volume before this worker is started.
    """

    _CACHE_CAPACITY = 24
    _COMPONENT_CACHE_CAPACITY = 8

    def __init__(
        self,
        composer: MultimodalFusionComposer,
        cache_seed: Optional[
            tuple[
                tuple[str, int, Optional[tuple[bool, bool, bool, bool]]],
                np.ndarray,
            ]
        ] = None,
        bone_masks: tuple[
            Optional[PackedBoneMasks], Optional[PackedBoneMasks]
        ] = (None, None),
    ) -> None:
        self._composer = composer
        self._bone_masks = bone_masks
        self._condition = threading.Condition()
        self._request: Optional[
            tuple[
                str,
                int,
                int,
                Optional[tuple[bool, bool, bool, bool]],
            ]
        ] = None
        self._stopped = False
        self._results: "queue.Queue[tuple[int, str, int, np.ndarray]]" = (
            queue.Queue()
        )
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._normalized_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._edge_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        if cache_seed is not None:
            key, rgb = cache_seed
            self._cache[key] = rgb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(
        self,
        axis: str,
        index: int,
        generation: int,
        layers: Optional[tuple[bool, bool, bool, bool]] = None,
    ) -> None:
        """Record the latest requested slice; older requests are dropped."""
        with self._condition:
            self._request = (axis, index, generation, layers)
            self._condition.notify()

    def poll_result(self) -> Optional[tuple[int, str, int, np.ndarray]]:
        """Return the next finished result or None when the queue is empty."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop the worker thread and wait briefly for it to finish."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _trim_cache(cache: OrderedDict, capacity: int) -> None:
        while len(cache) > capacity:
            cache.popitem(last=False)

    def _normalized_slices(
        self, axis: str, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (axis, index)
        if key in self._normalized_cache:
            self._normalized_cache.move_to_end(key)
            return self._normalized_cache[key]

        composer = self._composer
        fixed = _take_slice(
            np.asarray(composer._fixed._body_data), axis, index
        )
        moving = _take_slice(
            np.asarray(composer._moving._body_data), axis, index
        )
        normalized = (
            normalize_to_u8(fixed, composer._fixed_window),
            normalize_to_u8(moving, composer._moving_window),
        )
        self._normalized_cache[key] = normalized
        self._trim_cache(
            self._normalized_cache, self._COMPONENT_CACHE_CAPACITY
        )
        return normalized

    def _edge(
        self, axis: str, index: int, image_number: int, image: np.ndarray
    ) -> np.ndarray:
        key = (axis, index, image_number)
        if key in self._edge_cache:
            self._edge_cache.move_to_end(key)
            return self._edge_cache[key]
        edge = _edge_strength(image)
        self._edge_cache[key] = edge
        self._trim_cache(
            self._edge_cache, self._COMPONENT_CACHE_CAPACITY * 2
        )
        return edge

    def _make_two_image_slice(
        self,
        axis: str,
        index: int,
        layers: tuple[bool, bool, bool, bool],
    ) -> np.ndarray:
        show_fixed, show_moving, show_boundary, show_bone = layers
        if not show_fixed and not show_moving:
            fixed = _take_slice(
                np.asarray(self._composer._fixed._body_data), axis, index
            )
            black = np.zeros((*fixed.shape, 3), dtype=np.uint8)
            return np.transpose(black, (1, 0, 2))

        fixed_u8, moving_u8 = self._normalized_slices(axis, index)
        fixed_edge = None
        moving_edge = None
        if show_boundary and show_fixed:
            fixed_edge = self._edge(axis, index, 1, fixed_u8)
        if show_boundary and show_moving:
            moving_edge = self._edge(axis, index, 2, moving_u8)

        fixed_bone_mask = None
        moving_bone_mask = None
        if show_bone and axis == "s":
            if show_fixed and self._bone_masks[0] is not None:
                fixed_bone_mask = _unpack_bone_mask(
                    self._bone_masks[0], index
                )
            if show_moving and self._bone_masks[1] is not None:
                moving_bone_mask = _unpack_bone_mask(
                    self._bone_masks[1], index
                )
        rgb = _compose_two_image_layers_u8(
            fixed_u8,
            moving_u8,
            show_fixed,
            show_moving,
            show_boundary,
            fixed_edge,
            moving_edge,
            show_bone,
            fixed_bone_mask,
            moving_bone_mask,
        )
        return np.transpose(rgb, (1, 0, 2))

    def _make_slice_cached(
        self,
        axis: str,
        index: int,
        layers: Optional[tuple[bool, bool, bool, bool]],
    ) -> np.ndarray:
        key = (axis, index, layers)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if layers is None:
            rgb = self._composer.make_slice(axis, index)
        else:
            rgb = self._make_two_image_slice(axis, index, layers)
        self._cache[key] = rgb
        self._trim_cache(self._cache, self._CACHE_CAPACITY)
        return rgb

    def _run(self) -> None:
        last_generation = -1
        while True:
            with self._condition:
                while (
                    not self._stopped
                    and (
                        self._request is None
                        or self._request[2] == last_generation
                    )
                ):
                    self._condition.wait()
                if self._stopped:
                    return
                axis, index, generation, layers = self._request
            last_generation = generation
            rgb = self._make_slice_cached(axis, index, layers)
            self._results.put((generation, axis, index, rgb))


class _ComposerGuiViewer:
    """Shared resizable preview window driven by a fusion composer.

    The window opens at the default upscaled size and rescales the image
    with it while preserving the aspect ratio. The bottom slider selects the
    third (S) coordinate.
    """

    def __init__(self, composer: MultimodalFusionComposer, title: str) -> None:
        self._composer = composer
        self._title = title

    def get_composer(self) -> MultimodalFusionComposer:
        return self._composer

    def gui_preview(self) -> None:
        """Open the blocking preview window.

        Every S-plane and control combination is rendered before the Tk
        window is created. The slider callback only records the latest index;
        a fixed-rate Tk update selects the corresponding cached frame and
        redraws it without normalization, boundary detection, mask blending,
        or multimodal composition during interaction.

        Raises:
            ImportError: When tkinter is not installed. The error message
                includes the pip command required to install it.
        """
        from .body_data.body_data_imp._display_scale import (
            _scale_image_to_fit,
            upscale_for_display,
        )
        from .body_data.body_data_imp._tk_gui import require_tkinter

        tk = require_tkinter()
        from PIL import Image, ImageTk

        composer = self._composer
        _, _, s_size = composer.get_size()

        is_two_image_viewer = composer.get_mask() is None
        initial_layers: Optional[LayerSelection] = (
            (True, True, True, False) if is_two_image_viewer else None
        )
        two_image_frames: Optional[FrameCache] = None
        three_image_frames: Optional[np.ndarray] = None
        if is_two_image_viewer:
            two_image_frames, cache_directory = (
                _precompute_two_image_frames(composer)
            )
            assert initial_layers is not None
            initial_rgb = two_image_frames[initial_layers][0]
            cached_arrays = list(two_image_frames.values())
        else:
            three_image_frames, cache_directory = (
                _precompute_three_image_frames(composer)
            )
            initial_rgb = three_image_frames[0]
            cached_arrays = [three_image_frames]
        initial_base = upscale_for_display(
            Image.fromarray(np.asarray(initial_rgb)).copy()
        )[0]

        root = tk.Tk()
        root.title(self._title)

        canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        layer_vars = None
        layer_buttons = []
        bone_var = None
        bone_button = None
        if is_two_image_viewer:
            controls = tk.Frame(root)
            controls.pack(fill=tk.X, padx=8, pady=(6, 0))
            layer_vars = (
                tk.BooleanVar(master=root, value=True),
                tk.BooleanVar(master=root, value=True),
                tk.BooleanVar(master=root, value=True),
            )
            for label, variable in zip(
                ("Image 1", "Image 2", "Boundary"), layer_vars
            ):
                button = tk.Checkbutton(
                    controls, text=label, variable=variable
                )
                button.pack(side=tk.LEFT, padx=(0, 12))
                layer_buttons.append(button)
            bone_var = tk.BooleanVar(master=root, value=False)
            bone_button = tk.Checkbutton(
                controls, text="Bone", variable=bone_var
            )
            bone_button.pack(side=tk.LEFT)

        z_label = tk.Label(root)
        z_label.pack(pady=(6, 0))

        z_slider = tk.Scale(
            root,
            from_=0,
            to=s_size - 1,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=False,
        )
        z_slider.pack(fill=tk.X, padx=8, pady=(0, 8))

        state = {
            "photo": None,
            "z": 0,  # index currently displayed
            "target": 0,  # latest slider index
            "layers": initial_layers,
            "base": initial_base,
            "rendered": (0, initial_layers),
            "needs_redraw": True,
        }

        def make_base_image(index, layers):
            if two_image_frames is not None:
                assert layers is not None
                frame = two_image_frames[layers][index]
            else:
                assert three_image_frames is not None
                frame = three_image_frames[index]
            return upscale_for_display(
                Image.fromarray(np.asarray(frame)).copy()
            )[0]

        def redraw():
            box_width = max(1, canvas.winfo_width())
            box_height = max(1, canvas.winfo_height())
            image, _, _ = _scale_image_to_fit(
                state["base"], box_width, box_height
            )
            photo = ImageTk.PhotoImage(image)
            canvas.delete("all")
            canvas.create_image(
                box_width // 2, box_height // 2, image=photo, anchor=tk.CENTER
            )
            state["photo"] = photo # type:ignore
            z_label.configure(text=f"z = {state['z']}")
            state["needs_redraw"] = False

        def update_slice(z_value):
            z = int(float(z_value))
            if z == state["target"]:
                return
            state["target"] = z

        def update_layers():
            assert layer_vars is not None
            assert bone_var is not None
            layers = (
                bool(layer_vars[0].get()),
                bool(layer_vars[1].get()),
                bool(layer_vars[2].get()),
                bool(bone_var.get()),
            )
            if layers == state["layers"]:
                return
            state["layers"] = layers

        def on_configure(_event):
            state["needs_redraw"] = True

        def tick():
            requested = (state["target"], state["layers"])
            if requested != state["rendered"]:
                state["base"] = make_base_image(*requested)
                state["z"] = state["target"]
                state["rendered"] = requested
                state["needs_redraw"] = True
            if state["needs_redraw"]:
                redraw()
            root.after(16, tick)

        canvas.bind("<Configure>", on_configure)
        z_slider.configure(command=update_slice)
        for button in layer_buttons:
            button.configure(command=update_layers)
        if bone_button is not None:
            bone_button.configure(command=update_layers)

        base_width, base_height = state["base"].size
        controls_height = 132 if is_two_image_viewer else 100
        root.geometry(f"{base_width}x{base_height + controls_height}")
        root.after(16, tick)
        try:
            root.mainloop()
        finally:
            state["photo"] = None
            state["base"] = None
            _release_frame_cache(cached_arrays, cache_directory)


class TwoImageFusionViewer(_ComposerGuiViewer):
    """Preview two volumes with the amber/cyan edge fusion.

    The volumes may differ in size and voxel spacing; they are origin
    aligned and placed on a common grid before rendering.
    """

    def __init__(self, bd1: BodyData, bd2: BodyData) -> None:
        composer = MultimodalFusionComposer(bd1, bd2)
        _ComposerGuiViewer.__init__(
            self, composer, "Two-image amber/cyan fusion"
        )


class ThreeImageOverlayViewer(_ComposerGuiViewer):
    """Preview two fused volumes plus a semi-transparent red mask overlay.

    The third volume is aligned on the same common grid as the first two.
    Its smallest value maps to a fully transparent overlay and its largest
    value to 50% opacity.
    """

    def __init__(self, bd1: BodyData, bd2: BodyData, bd3: BodyData) -> None:
        composer = MultimodalFusionComposer(bd1, bd2, mask=bd3)
        _ComposerGuiViewer.__init__(
            self, composer, "Two-image fusion with red mask overlay"
        )


__all__ = [
    "BodyDataCommonGrid",
    "MultimodalFusionComposer",
    "TwoImageFusionViewer",
    "ThreeImageOverlayViewer",
    "display_window",
    "normalize_to_u8",
    "resample_to_mmpd",
]
