"""Blocking Tkinter comparison window for two LPS projection triplets."""

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ._tk_gui import require_tkinter


ProjectionTriplet = tuple[np.ndarray, np.ndarray, np.ndarray]


def _validate_projection_row(
    projections: Sequence[np.ndarray], name: str
) -> ProjectionTriplet:
    if len(projections) != 3:
        raise ValueError(f"{name} must contain PS, SL and LP projections")
    arrays = tuple(np.asarray(projection) for projection in projections)
    for array in arrays:
        if array.ndim != 2:
            raise ValueError(f"every {name} projection must be two-dimensional")
        if array.size == 0:
            raise ValueError(f"every {name} projection must not be empty")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise TypeError(f"every {name} projection must contain real numbers")
    return arrays[0], arrays[1], arrays[2]


def _resolve_window(
    projections: ProjectionTriplet, window: tuple[float, float]
) -> tuple[float, float]:
    value_min, value_max = map(float, window)
    if np.isfinite(value_min) and np.isfinite(value_max) and value_max > value_min:
        return value_min, value_max

    finite_min = np.inf
    finite_max = -np.inf
    for projection in projections:
        finite = projection[np.isfinite(projection)]
        if finite.size:
            finite_min = min(finite_min, float(np.min(finite)))
            finite_max = max(finite_max, float(np.max(finite)))
    if not np.isfinite(finite_min):
        return 0.0, 1.0
    if finite_max <= finite_min:
        return finite_min, finite_min + 1.0
    return finite_min, finite_max


def _projection_to_image(
    projection: np.ndarray, window: tuple[float, float]
) -> Image.Image:
    value_min, value_max = window
    normalized = np.clip(
        (projection.astype(np.float64) - value_min) / (value_max - value_min),
        0.0,
        1.0,
    )
    pixels = np.rint(
        np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0) * 255.0
    ).astype(np.uint8)
    # The package maps array[x, y] to image pixel (x, y).
    return Image.fromarray(pixels.T)


def _prepare_projection_images(
    lhs_projections: Sequence[np.ndarray],
    rhs_projections: Sequence[np.ndarray],
    lhs_window: tuple[float, float],
    rhs_window: tuple[float, float],
) -> tuple[tuple[Image.Image, ...], tuple[Image.Image, ...]]:
    """Normalize and transpose all six images before the Tk window opens."""
    lhs = _validate_projection_row(lhs_projections, "lhs_projections")
    rhs = _validate_projection_row(rhs_projections, "rhs_projections")
    lhs_window = _resolve_window(lhs, lhs_window)
    rhs_window = _resolve_window(rhs, rhs_window)
    return (
        tuple(_projection_to_image(item, lhs_window) for item in lhs),
        tuple(_projection_to_image(item, rhs_window) for item in rhs),
    )


def _scale_projection_images(
    images: Sequence[Image.Image],
    box_sizes: Sequence[tuple[int, int]],
) -> tuple[tuple[Image.Image, ...], float]:
    """Fit every image with one shared scale while preserving aspect ratios."""
    if not images or len(images) != len(box_sizes):
        raise ValueError("images and box_sizes must have the same non-zero length")

    common_scale = min(
        min(max(1, box_width) / image.width, max(1, box_height) / image.height)
        for image, (box_width, box_height) in zip(images, box_sizes)
    )
    scaled = []
    for image in images:
        target_size = (
            max(1, int(round(image.width * common_scale))),
            max(1, int(round(image.height * common_scale))),
        )
        if target_size == image.size:
            scaled.append(image)
        else:
            scaled.append(
                image.resize(target_size, Image.Resampling.NEAREST)
            )
    return tuple(scaled), common_scale


def show_projection_comparison(
    lhs_projections: Sequence[np.ndarray],
    rhs_projections: Sequence[np.ndarray],
    lhs_window: tuple[float, float],
    rhs_window: tuple[float, float],
) -> None:
    """Show two rows of PS, SL and LP projections and block until closing."""
    base_images = _prepare_projection_images(
        lhs_projections, rhs_projections, lhs_window, rhs_window
    )

    tk = require_tkinter()
    from PIL import ImageTk

    root = tk.Tk()
    root.title("BodyData Projection Comparison")

    content = tk.Frame(root)
    content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    content.grid_columnconfigure(0, weight=0)
    for column in range(1, 4):
        content.grid_columnconfigure(column, weight=1, uniform="projection")
    for row in range(1, 3):
        content.grid_rowconfigure(row, weight=1, uniform="image")

    for column, text in enumerate(("PS", "SL", "LP"), start=1):
        tk.Label(content, text=text, font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=column, sticky="ew", pady=(0, 6)
        )

    canvases = []
    for row, (row_name, images) in enumerate(
        (("Image1", base_images[0]), ("Image2", base_images[1])), start=1
    ):
        tk.Label(content, text=row_name, font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="ns", padx=(0, 10)
        )
        for column, _image in enumerate(images, start=1):
            canvas = tk.Canvas(
                content,
                width=300,
                height=300,
                background="black",
                highlightthickness=1,
                highlightbackground="#707070",
            )
            canvas.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            canvases.append(canvas)

    flat_images = base_images[0] + base_images[1]
    state = {"photos": [None] * 6, "redraw_job": None}

    def redraw() -> None:
        state["redraw_job"] = None
        box_sizes = tuple(
            (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
            for canvas in canvases
        )
        scaled_images, _scale = _scale_projection_images(
            flat_images, box_sizes
        )
        for index, (canvas, image, (box_width, box_height)) in enumerate(
            zip(canvases, scaled_images, box_sizes)
        ):
            photo = ImageTk.PhotoImage(image)
            canvas.delete("all")
            canvas.create_image(
                box_width // 2,
                box_height // 2,
                image=photo,
                anchor=tk.CENTER,
            )
            state["photos"][index] = photo

    def schedule_redraw(_event=None) -> None:
        redraw_job = state["redraw_job"]
        if redraw_job is not None:
            try:
                root.after_cancel(redraw_job)
            except tk.TclError:
                pass
        state["redraw_job"] = root.after(30, redraw)

    for canvas in canvases:
        canvas.bind("<Configure>", schedule_redraw)

    root.geometry("1100x760")
    root.minsize(680, 480)
    schedule_redraw()
    try:
        root.mainloop()
    finally:
        state["photos"] = [None] * 6


__all__ = ["show_projection_comparison"]
