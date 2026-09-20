"""Blocking 3-D grayscale slice browser (optional Tkinter GUI)."""

from typing import Optional

import numpy as np
from PIL import Image

from ...bone_segmentation import extract_bone_mask
from ._bone_mask import (
    _blacken_non_bone,
    _bone_threshold,
    _estimate_cbct_bone_threshold,
)
from ._display_scale import _scale_image_to_fit, _upscale_for_display
from ._tk_gui import require_tkinter


def _precompute_single_frames(
    array: np.ndarray,
    value_min: float,
    value_max: float,
    bone_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Render every normal and Bone slice before the Tk window opens."""
    size_l, size_p, size_s = array.shape
    normal_frames = np.empty((size_s, size_p, size_l), dtype=np.uint8)
    bone_frames = (
        np.empty_like(normal_frames)
        if bone_mask is not None
        else normal_frames
    )
    if bone_mask is not None and bone_mask.shape != array.shape:
        raise ValueError("bone_mask must have the same shape as array")

    scale = value_max - value_min
    for index in range(size_s):
        values = array[:, :, index].astype(np.float64)
        normalized = np.clip((values - value_min) / scale, 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0)
        pixels = np.rint(normalized * 255).astype(np.uint8)
        normal_frames[index] = pixels.T
        if bone_mask is not None:
            bone_pixels = pixels.copy()
            bone_pixels[~bone_mask[:, :, index]] = 0
            bone_frames[index] = bone_pixels.T
    return normal_frames, bone_frames


def show_numpy_3d(
    array_3d,
    value_min,
    value_max,
    image_type: Optional[str] = None,
    mmpd: float = 1.0,
):
    """
    Display slices of a 3D NumPy array in a blocking Tkinter window.

    The slider selects the third coordinate, z. Array values are mapped to
    grayscale linearly between value_min and value_max. When image_type is
    ``"ct"`` or ``"cbct"``, the Bone checkbox can hide non-bone voxels. The
    ``mmpd`` value supplies the isotropic voxel width used by physical-space
    bone-mask cleanup. The coordinate mapping is array_3d[x, y, z] -> image
    pixel (x, y).

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

    bone_mask = (
        extract_bone_mask(array, image_type, mmpd)
        if image_type in {"ct", "cbct"}
        else None
    )
    normal_frames, bone_frames = _precompute_single_frames(
        array, value_min, value_max, bone_mask
    )

    def make_base_image(z_index, bone_only):
        frames = bone_frames if bone_only else normal_frames
        return _upscale_for_display(Image.fromarray(frames[z_index]))[0]

    root = tk.Tk()
    root.title("NumPy 3D Grayscale Viewer")

    canvas = tk.Canvas(root, highlightthickness=0, borderwidth=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    controls = tk.Frame(root)
    controls.pack(fill=tk.X, padx=8, pady=(6, 0))
    bone_var = tk.BooleanVar(master=root, value=False)
    bone_button = tk.Checkbutton(
        controls,
        text="Bone",
        variable=bone_var,
        state=tk.NORMAL if bone_mask is not None else tk.DISABLED,
    )
    bone_button.pack(side=tk.LEFT)

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

    state = {
        "photo": None,
        "z": 0,
        "target": 0,
        "bone": False,
        "base": make_base_image(0, False),
        "rendered": (0, False),
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

    def update_slice(z_value):
        state["target"] = int(float(z_value))

    def update_bone():
        state["bone"] = bool(bone_var.get())

    def on_configure(_event=None):
        state["needs_redraw"] = True

    def tick():
        requested = (state["target"], state["bone"])
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
    bone_button.configure(command=update_bone)

    base_width, base_height = state["base"].size
    root.geometry(f"{base_width}x{base_height + 132}")
    root.after(16, tick)
    root.mainloop()


if __name__ == "__main__":
    sample = np.random.default_rng().normal(size=(320, 240, 20))
    show_numpy_3d(sample, value_min=-2.0, value_max=2.0)
