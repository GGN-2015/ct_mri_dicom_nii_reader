"""Reusable maximum-intensity projections for LPS-ordered volumes."""

from typing import Literal, TypeAlias, cast

import numpy as np


ProjectionPlane: TypeAlias = Literal["ps", "sl", "lp"]


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume must be a three-dimensional LPS array")
    if array.size == 0:
        raise ValueError("volume must not be empty")
    return array


def _project_validated_volume(
    volume: np.ndarray, plane: ProjectionPlane
) -> np.ndarray:
    if plane == "ps":
        return np.max(volume, axis=0)
    if plane == "sl":
        # Reducing P leaves (L, S); transpose to the requested (S, L) order.
        return np.ascontiguousarray(np.max(volume, axis=1).T)
    return np.max(volume, axis=2)


def max_intensity_projection(
    volume: np.ndarray, plane: ProjectionPlane
) -> np.ndarray:
    """Project an ``(L, P, S)`` volume onto one named anatomical plane.

    The reduction takes the maximum along the omitted LPS dimension. Plane
    names also define the returned 2-D axis order: ``ps`` returns ``(P, S)``,
    ``sl`` returns ``(S, L)``, and ``lp`` returns ``(L, P)``.
    """
    if not isinstance(plane, str):
        raise TypeError("plane must be one of: 'ps', 'sl', 'lp'")
    normalized_plane = plane.lower()
    if normalized_plane not in {"ps", "sl", "lp"}:
        raise ValueError("plane must be one of: 'ps', 'sl', 'lp'")
    return _project_validated_volume(
        _validate_volume(volume), cast(ProjectionPlane, normalized_plane)
    )


def get_lps_max_projections(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(PS, SL, LP)`` maximum projections of an LPS volume."""
    array = _validate_volume(volume)
    return (
        _project_validated_volume(array, "ps"),
        _project_validated_volume(array, "sl"),
        _project_validated_volume(array, "lp"),
    )


__all__ = [
    "ProjectionPlane",
    "max_intensity_projection",
    "get_lps_max_projections",
]
