"""Visualization classes for :class:`BodyData` medical image volumes.

The rendering approach mirrors the visualization of the regknee project:

* :class:`BodyDataCommonGrid` aligns any number of volumes at the LPS origin,
  resamples them onto the finest unified voxel spacing and places them into
  one common NumPy grid. Blank regions are filled with the modality-specific
  air value (-1024 for CT, 0 for MRI and masks).
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
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from .body_data import BodyData, BodyDataNotInitialized


_FIXED_EDGE_COLOR = np.asarray((220.0, 145.0, 28.0), dtype=np.float32)
_MOVING_EDGE_COLOR = np.asarray((20.0, 225.0, 255.0), dtype=np.float32)
_EDGE_FUSION_OPACITY = 0.48
_MASK_OVERLAY_COLOR = np.asarray((255.0, 0.0, 0.0), dtype=np.float32)
_MASK_OVERLAY_MAX_ALPHA = 0.5


def _air_value(image_type: Optional[str]) -> float:
    return -1024.0 if image_type == "ct" else 0.0


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
    elif body_data.get_type() == "ct":
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
    fixed_rgb = np.repeat(fixed_u8[:, :, None], 3, axis=2).astype(np.float32)

    fusion = fixed_rgb * 0.32
    fusion += (
        _edge_strength(fixed_u8)[:, :, None]
        * _FIXED_EDGE_COLOR
    )
    fusion += (
        _edge_strength(moving_u8)[:, :, None]
        * _EDGE_FUSION_OPACITY
        * _MOVING_EDGE_COLOR
    )
    return np.ascontiguousarray(
        np.clip(fusion, 0.0, 255.0), dtype=np.uint8
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
    by comparing them against the newest request. A small LRU cache keyed by
    ``(axis, index)`` avoids recomputing slices that were already rendered.
    """

    _CACHE_CAPACITY = 8

    def __init__(
        self,
        composer: MultimodalFusionComposer,
        cache_seed: Optional[tuple[tuple[str, int], np.ndarray]] = None,
    ) -> None:
        self._composer = composer
        self._condition = threading.Condition()
        self._request: Optional[tuple[str, int, int]] = None
        self._stopped = False
        self._results: "queue.Queue[tuple[int, str, int, np.ndarray]]" = (
            queue.Queue()
        )
        self._cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()
        if cache_seed is not None:
            key, rgb = cache_seed
            self._cache[key] = rgb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, axis: str, index: int, generation: int) -> None:
        """Record the latest requested slice; older requests are dropped."""
        with self._condition:
            self._request = (axis, index, generation)
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

    def _make_slice_cached(self, axis: str, index: int) -> np.ndarray:
        key = (axis, index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        rgb = self._composer.make_slice(axis, index)
        self._cache[key] = rgb
        while len(self._cache) > self._CACHE_CAPACITY:
            self._cache.popitem(last=False)
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
                axis, index, generation = self._request
            last_generation = generation
            rgb = self._make_slice_cached(axis, index)
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

        Slices are rendered at a fixed frame rate instead of debouncing the
        slider: the slider callback only records the latest index, a single
        background worker thread computes the NumPy RGB slices (with a small
        LRU cache), and the Tk main thread only converts the newest result
        into a PhotoImage and redraws the canvas. Stale results are rejected
        through a generation/index check, and nothing is recomputed when the
        displayed slice has not changed.

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

        initial_rgb = composer.make_slice("s", 0)
        initial_base = upscale_for_display(
            Image.fromarray(initial_rgb, mode="RGB")
        )[0]

        root = tk.Tk()
        root.title(self._title)

        canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
        canvas.pack(fill=tk.BOTH, expand=True)

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

        worker = _SliceRenderWorker(
            composer, cache_seed=(("s", 0), initial_rgb)
        )

        state = {
            "photo": None,
            "z": 0,  # index currently displayed
            "target": 0,  # latest slider index
            "generation": 0,  # bumped on every target change
            "base": initial_base,
            "needs_redraw": True,
        }

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

        def apply_result(generation, index, rgb):
            if generation != state["generation"] or index != state["target"]:
                return
            if index == state["z"]:
                return
            state["base"] = upscale_for_display(
                Image.fromarray(rgb, mode="RGB")
            )[0]
            state["z"] = index
            state["needs_redraw"] = True

        def update_slice(z_value):
            z = int(float(z_value))
            if z == state["target"]:
                return
            state["target"] = z
            state["generation"] += 1
            if z != state["z"]:
                worker.request("s", z, state["generation"])

        def on_configure(_event):
            state["needs_redraw"] = True

        def tick():
            while True:
                result = worker.poll_result()
                if result is None:
                    break
                generation, _, index, rgb = result
                apply_result(generation, index, rgb)
            if state["needs_redraw"]:
                redraw()
            root.after(16, tick)

        canvas.bind("<Configure>", on_configure)
        z_slider.configure(command=update_slice)

        base_width, base_height = state["base"].size
        root.geometry(f"{base_width}x{base_height + 100}")
        root.after(16, tick)
        root.mainloop()
        worker.stop()


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
