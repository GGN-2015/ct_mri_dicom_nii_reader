"""Display a 2-D NumPy array as a grayscale image (optional Tkinter GUI)."""

from typing import Optional

import numpy as np
from PIL import Image

from ._display_scale import _scale_image_to_fit, _upscale_for_display
from ._tk_gui import require_tkinter


def show_numpy_gray(
        array_2d,
        value_min:Optional[float],
        value_max:Optional[float]):
    """
    Display a 2D NumPy array as a grayscale image.

    Coordinate mapping:
        array_2d[x, y] -> image pixel (x, y)

    The window opens at the default upscaled size (smallest image side at
    least 512 px). Resizing the window rescales the image with it while
    keeping its original aspect ratio. The function returns after the
    window is closed.

    Raises:
        ImportError: When tkinter is not installed. The error message
            includes the pip command required to install it.
    """
    tk = require_tkinter()
    from PIL import ImageTk

    array = np.asarray(array_2d)
    if value_min is None:
        value_min = float(array.min())
    if value_max is None:
        value_max = float(array.max())

    if array.ndim != 2:
        raise ValueError("array_2d must be a two-dimensional array")
    if array.size == 0:
        raise ValueError("array_2d must not be empty")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")

    normalized = (
        array.astype(np.float64) - value_min
    ) / (value_max - value_min)

    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)

    pixels = np.rint(normalized * 255).astype(np.uint8)

    # Pillow indexes pixels as [y, x], so transpose the array.
    base_image = _upscale_for_display(Image.fromarray(pixels.T, mode="L"))[0]

    root = tk.Tk()
    root.title("NumPy Grayscale Image")
    base_width, base_height = base_image.size
    root.geometry(f"{base_width}x{base_height}")

    canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    state = {"photo": None}

    def redraw():
        box_width = max(1, canvas.winfo_width())
        box_height = max(1, canvas.winfo_height())
        image, _, _ = _scale_image_to_fit(base_image, box_width, box_height)
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(
            box_width // 2, box_height // 2, image=photo, anchor=tk.CENTER
        )
        state["photo"] = photo # type:ignore

    pending = {"job": None}

    def schedule_redraw(_event=None):
        if pending["job"] is not None:
            try:
                root.after_cancel(pending["job"])
            except tk.TclError:
                pass
        pending["job"] = root.after(30, redraw)

    canvas.bind("<Configure>", schedule_redraw)
    schedule_redraw()
    root.mainloop()
