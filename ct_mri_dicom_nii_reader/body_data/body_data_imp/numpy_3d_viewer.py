"""Blocking 3-D grayscale slice browser (optional Tkinter GUI)."""

import numpy as np
from PIL import Image

from ._display_scale import _scale_image_to_fit, _upscale_for_display
from ._tk_gui import require_tkinter


def show_numpy_3d(array_3d, value_min, value_max):
    """
    Display slices of a 3D NumPy array in a blocking Tkinter window.

    The slider selects the third coordinate, z. Array values are mapped to
    grayscale linearly between value_min and value_max. The coordinate mapping
    is array_3d[x, y, z] -> image pixel (x, y).

    The window opens at the default upscaled size (smallest image side at
    least 512 px). Resizing the window rescales the image with it while
    keeping its original aspect ratio.

    Raises:
        ImportError: When tkinter is not installed. The error message
            includes the pip command required to install it.
    """
    tk = require_tkinter()
    from PIL import ImageTk

    array = np.asarray(array_3d)

    if array.ndim != 3:
        raise ValueError("array_3d must be a three-dimensional array")
    if any(size == 0 for size in array.shape):
        raise ValueError("array_3d dimensions must not be empty")

    value_min = float(value_min)
    value_max = float(value_max)
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        raise ValueError("value_min and value_max must be finite")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")

    def make_base_image(z_index):
        slice_2d = array[:, :, z_index].astype(np.float64)
        normalized = (slice_2d - value_min) / (value_max - value_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0)
        pixels = np.rint(normalized * 255).astype(np.uint8)
        return _upscale_for_display(Image.fromarray(pixels.T))[0]

    root = tk.Tk()
    root.title("NumPy 3D Grayscale Viewer")

    canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    z_label = tk.Label(root)
    z_label.pack(pady=(6, 0))

    z_slider = tk.Scale(
        root,
        from_=0,
        to=array.shape[2] - 1,
        orient=tk.HORIZONTAL,
        resolution=1,
        showvalue=False,
    )
    z_slider.pack(fill=tk.X, padx=8, pady=(0, 8))

    state = {"photo": None, "z": 0, "base": make_base_image(0)}

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

    pending = {"job": None}

    def schedule_redraw(_event=None):
        if pending["job"] is not None:
            try:
                root.after_cancel(pending["job"])
            except tk.TclError:
                pass
        pending["job"] = root.after(30, redraw)

    def update_slice(z_value):
        state["z"] = int(float(z_value))
        state["base"] = make_base_image(state["z"])
        schedule_redraw()

    canvas.bind("<Configure>", schedule_redraw)
    z_slider.configure(command=update_slice)

    base_width, base_height = state["base"].size
    root.geometry(f"{base_width}x{base_height + 100}")
    schedule_redraw()
    root.mainloop()


if __name__ == "__main__":
    sample = np.random.default_rng().normal(size=(320, 240, 20))
    show_numpy_3d(sample, value_min=-2.0, value_max=2.0)
