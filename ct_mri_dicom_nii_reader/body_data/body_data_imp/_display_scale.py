"""Private image-display helpers shared by the GUI and PNG preview paths."""

from math import ceil

from PIL import Image


_MIN_DISPLAY_SIDE = 512


def upscale_for_display(image: Image.Image) -> tuple[Image.Image, float, float]:
    """Return a nearest-neighbor image whose dimensions are at least 512 px.

    This is the default size used when a preview window opens. Interactive
    resizing afterwards is handled by :func:`_scale_image_to_fit`.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("display image dimensions must be positive")

    scale = max(1.0, _MIN_DISPLAY_SIDE / min(width, height))
    display_width = ceil(width * scale)
    display_height = ceil(height * scale)
    if (display_width, display_height) == (width, height):
        return image, 1.0, 1.0

    displayed = image.resize(
        (display_width, display_height), Image.Resampling.NEAREST
    )
    return displayed, display_width / width, display_height / height


def _upscale_for_display(
    image: Image.Image,
) -> tuple[Image.Image, float, float]:
    """Compatibility forwarder for :func:`upscale_for_display`."""
    return upscale_for_display(image)


def _scale_image_to_fit(
    image: Image.Image, box_width: int, box_height: int
) -> tuple[Image.Image, float, float]:
    """Return a nearest-neighbor scaled copy that fits inside the given box.

    The original aspect ratio is preserved and the scaled image may become
    larger or smaller than the source. The returned scale factors relate the
    displayed size to the source size, so display coordinates can be mapped
    back to source coordinates.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("display image dimensions must be positive")

    box_width = max(1, int(box_width))
    box_height = max(1, int(box_height))
    scale = min(box_width / width, box_height / height)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if (target_width, target_height) == (width, height):
        return image, 1.0, 1.0

    resized = image.resize(
        (target_width, target_height), Image.Resampling.NEAREST
    )
    return resized, target_width / width, target_height / height
