"""Optional Tkinter support helpers for the preview GUIs.

Tkinter is not a mandatory dependency of this package. The preview windows
import it lazily and raise a descriptive error, including the pip command
needed to install it, when it is missing.
"""

from __future__ import annotations

import importlib.util


_TKINTER_PIP_COMMAND = "pip install tk"

_MISSING_TKINTER_MESSAGE = (
    "Tkinter is not installed, so the GUI preview cannot be started.\n"
    "Install tkinter with pip:\n"
    f"    {_TKINTER_PIP_COMMAND}\n"
    "On Debian/Ubuntu you can alternatively run:\n"
    "    sudo apt-get install python3-tk\n"
    "The official Windows and macOS Python installers already include tkinter."
)


def tkinter_available() -> bool:
    """Return True when the tkinter module can be found."""
    return importlib.util.find_spec("tkinter") is not None


def require_tkinter():
    """Import tkinter, or raise an ImportError with install instructions.

    The error message includes the pip command required to install tkinter.
    """
    try:
        import tkinter
    except ImportError as exc:
        raise ImportError(_MISSING_TKINTER_MESSAGE) from exc
    return tkinter
