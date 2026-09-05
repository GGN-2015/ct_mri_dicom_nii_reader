"""Load a CT DICOM series as an isotropic HU volume in L/P/S axis order."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import SimpleITK as sitk


Metadata: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class _SeriesInfo:
    uid: str
    file_names: list[str]
    depth: int
    voxel_count: int
    single_file_dimension: int


def _inspect_series(dicom_dir: Path, uid: str) -> _SeriesInfo:
    file_names = list(
        sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
    )
    if not file_names:
        raise ValueError(f"DICOM series contains no readable files: {uid}")

    # Series members are expected to have the same matrix size. Reading only
    # the first header avoids loading every candidate series into memory.
    header_reader = sitk.ImageFileReader()
    header_reader.SetFileName(file_names[0])
    header_reader.ReadImageInformation()
    single_file_size = tuple(int(value) for value in header_reader.GetSize())
    single_file_dimension = header_reader.GetDimension()
    frames_per_file = int(np.prod(single_file_size[2:], dtype=np.int64))
    depth = max(1, frames_per_file) * len(file_names)
    voxel_count = int(np.prod(single_file_size, dtype=np.int64)) * len(file_names)
    return _SeriesInfo(
        uid=uid,
        file_names=file_names,
        depth=depth,
        voxel_count=voxel_count,
        single_file_dimension=single_file_dimension,
    )


def _choose_series(dicom_dir: Path, series_uid: str | None) -> _SeriesInfo:
    series_uids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or [])
    if not series_uids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    if series_uid is not None:
        if series_uid not in series_uids:
            raise ValueError(
                f"Series UID {series_uid!r} was not found. "
                f"Available UIDs: {', '.join(series_uids)}"
            )
        return _inspect_series(dicom_dir, series_uid)

    candidates: list[_SeriesInfo] = []
    inspection_errors: list[str] = []
    for uid in sorted(series_uids):
        try:
            info = _inspect_series(dicom_dir, uid)
        except (RuntimeError, ValueError) as exc:
            inspection_errors.append(f"{uid}: {exc}")
            continue
        if info.depth > 1:
            candidates.append(info)

    if not candidates:
        details = (
            f" Header errors: {'; '.join(inspection_errors)}" if inspection_errors else ""
        )
        raise ValueError(
            "No multi-slice DICOM series was found after excluding single-slice "
            f"series.{details}"
        )

    largest_voxel_count = max(info.voxel_count for info in candidates)
    tied = [info for info in candidates if info.voxel_count == largest_voxel_count]
    # Candidates were inspected in sorted UID order, making ties deterministic.
    return min(tied, key=lambda info: info.uid)


def _read_ct_series(info: _SeriesInfo, require_ct: bool) -> tuple[sitk.Image, str]:
    # Enhanced multi-frame DICOM is already 3-D and should not gain a fourth
    # dimension by being passed through ImageSeriesReader.
    if len(info.file_names) == 1 and info.single_file_dimension == 3:
        reader = sitk.ImageFileReader()
        reader.SetFileName(info.file_names[0])
        reader.SetOutputPixelType(sitk.sitkFloat32)
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        metadata_has = reader.HasMetaDataKey
        metadata_get = reader.GetMetaData
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(info.file_names)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        reader.SetForceOrthogonalDirection(False)  # Preserve CT gantry tilt.
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()
        metadata_has = lambda tag: reader.HasMetaDataKey(0, tag)
        metadata_get = lambda tag: reader.GetMetaData(0, tag)

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3-D DICOM series, got {image.GetDimension()} dimensions")

    modality = ""
    modality_tag = "0008|0060"
    if metadata_has(modality_tag):
        modality = metadata_get(modality_tag).strip().upper()
    if require_ct and modality != "CT":
        shown = modality or "missing"
        raise ValueError(
            f"The selected series has Modality={shown!r}, not 'CT'; HU is a CT concept. "
            "Pass require_ct=False only if the stored/rescaled values are meaningful "
            "for your modality."
        )

    return image, modality


def _image_support_bounds_lps(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return the LPS physical bounds of the complete voxel support, in mm."""
    size = image.GetSize()
    continuous_corners = product(
        *[(-0.5, float(axis_size) - 0.5) for axis_size in size]
    )
    physical_corners = np.asarray(
        [
            image.TransformContinuousIndexToPhysicalPoint(list(corner))
            for corner in continuous_corners
        ],
        dtype=np.float64,
    )
    return physical_corners.min(axis=0), physical_corners.max(axis=0)


def _resample_to_isotropic_lps(
    image: sitk.Image,
    mmpd: float,
    interpolation: Literal["linear", "nearest"],
    outside_hu: float,
) -> sitk.Image:
    lower_edge, upper_edge = _image_support_bounds_lps(image)
    size_ratio = (upper_edge - lower_edge) / mmpd

    # Stabilize ratios that should be integers but differ by floating-point noise.
    rounded_ratio = np.rint(size_ratio)
    size_ratio = np.where(
        np.isclose(size_ratio, rounded_ratio, rtol=1e-9, atol=1e-9),
        rounded_ratio,
        size_ratio,
    )
    output_size = np.maximum(1, np.ceil(size_ratio).astype(np.int64))
    output_origin = lower_edge + 0.5 * mmpd

    interpolators = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
    }
    try:
        sitk_interpolator = interpolators[interpolation]
    except KeyError as exc:
        raise ValueError("interpolation must be 'linear' or 'nearest'") from exc

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk_interpolator)
    resampler.SetDefaultPixelValue(float(outside_hu))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetOutputSpacing([mmpd, mmpd, mmpd])
    resampler.SetOutputOrigin(output_origin.tolist())
    resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    resampler.SetSize([int(value) for value in output_size])
    return resampler.Execute(image)


def load_dicom_hu_lps(
    dicom_dir: str | Path,
    mmpd: float,
    *,
    series_uid: str | None = None,
    interpolation: Literal["linear", "nearest"] = "linear",
    outside_hu: float = -1024.0,
    require_ct: bool = True,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, Metadata]:
    """Load one DICOM series into an isotropic NumPy HU volume.

    The returned array has shape ``(n_L, n_P, n_S)``. Increasing indices on
    axes 0, 1 and 2 move in the patient's Left, Posterior and Superior
    directions, respectively. Every voxel is ``mmpd`` mm on each side.

    Args:
        dicom_dir: Directory containing DICOM files. Automatic selection first
            excludes single-slice series, then chooses the multi-slice series
            with the largest voxel count. A tie is resolved by choosing the
            lexicographically smallest Series Instance UID.
        mmpd: Isotropic output voxel width in millimetres; must be finite and
            greater than zero.
        series_uid: Optional DICOM Series Instance UID.
        interpolation: ``"linear"`` is appropriate for HU images. ``"nearest"``
            is available when exact stored values are preferred.
        outside_hu: Fill value for the axis-aligned bounding box outside the
            scanned field of view, commonly -1024 HU for CT air.
        require_ct: Reject a non-CT series by default because HU is CT-specific.
        return_metadata: If true, return ``(volume, metadata)``. The metadata
            origin is the LPS patient coordinate of ``volume[0, 0, 0]``.

    Returns:
        A contiguous ``float32`` NumPy array, or ``(array, metadata)`` when
        ``return_metadata=True``.

    Notes:
        SimpleITK/GDCM applies DICOM Rescale Slope and Rescale Intercept while
        reading CT pixels. The canonical resampling grid uses the DICOM patient
        LPS physical coordinate system and identity direction.
    """
    directory = Path(dicom_dir).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {directory}")
    if isinstance(mmpd, bool) or not np.isfinite(mmpd) or mmpd <= 0:
        raise ValueError("mmpd must be a finite number greater than zero")
    mmpd = float(mmpd)

    selected_series = _choose_series(directory, series_uid)
    source_image, modality = _read_ct_series(
        selected_series, require_ct=require_ct
    )
    output_image = _resample_to_isotropic_lps(
        source_image,
        mmpd=mmpd,
        interpolation=interpolation,
        outside_hu=outside_hu,
    )

    # SimpleITK exposes [S, P, L] (z, y, x); transpose to requested [L, P, S].
    volume_lps = np.ascontiguousarray(
        np.transpose(sitk.GetArrayFromImage(output_image), (2, 1, 0)),
        dtype=np.float32,
    )
    if not return_metadata:
        return volume_lps

    metadata: Metadata = {
        "axis_order": ("L", "P", "S"),
        "shape_lps": tuple(int(value) for value in volume_lps.shape),
        "spacing_lps_mm": (mmpd, mmpd, mmpd),
        "origin_lps_mm": tuple(float(value) for value in output_image.GetOrigin()),
        "series_uid": selected_series.uid,
        "series_depth": selected_series.depth,
        "series_voxel_count": selected_series.voxel_count,
        "modality": modality,
        "number_of_files": len(selected_series.file_names),
        "source_size_xyz": tuple(int(value) for value in source_image.GetSize()),
        "source_spacing_xyz_mm": tuple(
            float(value) for value in source_image.GetSpacing()
        ),
        "source_origin_lps_mm": tuple(
            float(value) for value in source_image.GetOrigin()
        ),
        "source_direction_xyz_to_lps": np.asarray(
            source_image.GetDirection(), dtype=np.float64
        ).reshape(3, 3),
    }
    return volume_lps, metadata


__all__ = ["load_dicom_hu_lps"]
