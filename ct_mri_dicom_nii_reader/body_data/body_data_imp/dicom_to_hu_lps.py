"""Load a CT DICOM series as an isotropic HU volume in L/P/S axis order."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError


Metadata: TypeAlias = dict[str, object]

_CLASSIFICATION_TAGS = {
    "image_type": "0008|0008",
    "manufacturer": "0008|0070",
    "station_name": "0008|1010",
    "series_description": "0008|103e",
    "manufacturer_model_name": "0008|1090",
    "scan_options": "0018|0022",
    "protocol_name": "0018|1030",
    "acquisition_type": "0018|9302",
    "table_feed_per_rotation": "0018|9310",
    "spiral_pitch_factor": "0018|9311",
}

_INDEX_TAGS = [
    "SeriesInstanceUID",
    "SeriesDescription",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "InstanceNumber",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "SpacingBetweenSlices",
    "SliceThickness",
]
_DIRECTORY_INDEX_CACHE_CAPACITY = 8
_DIRECTORY_INDEX_CACHE: "OrderedDict[tuple, tuple[_SeriesInfo, ...]]" = (
    OrderedDict()
)
_DIRECTORY_INDEX_CACHE_LOCK = threading.Lock()


def _normalized_identifier(value: object) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _infer_dicom_modality(
    dicom_modality: str,
    attributes: dict[str, str],
) -> tuple[str, str, str]:
    """Classify CT-derived CBCT only when the headers contain strong evidence."""
    modality = dicom_modality.strip().upper() or "UNKNOWN"
    if modality == "CBCT":
        return "CBCT", "dicom:modality", "explicit"
    if modality != "CT":
        confidence = "explicit" if modality != "UNKNOWN" else "unknown"
        return modality, "dicom:modality", confidence

    explicit_markers = (
        "CBCT",
        "CONEBEAM",
        "3DCARM",
        "OARM",
        "CIOSSPIN",
        "DYNACT",
        "XPERCT",
        "\u9525\u5f62\u675f",
    )
    descriptive_fields = (
        "manufacturer_model_name",
        "station_name",
        "series_description",
        "protocol_name",
        "scan_options",
        "image_type",
    )
    normalized_fields = {
        field: _normalized_identifier(attributes.get(field, ""))
        for field in descriptive_fields
    }
    for field, value in normalized_fields.items():
        if any(marker in value for marker in explicit_markers):
            return "CBCT", f"dicom:{field}", "explicit"

    device_text = "".join(
        normalized_fields[field]
        for field in ("manufacturer_model_name", "station_name")
    )
    image_type = normalized_fields["image_type"]
    if "CARM" in device_text and (
        "3D" in device_text or "3DSLICE" in image_type
    ):
        return "CBCT", "dicom:3d_carm_reconstruction", "inferred"

    acquisition_type = _normalized_identifier(
        attributes.get("acquisition_type", "")
    )
    if (
        acquisition_type == "SPIRAL"
        or attributes.get("spiral_pitch_factor", "").strip()
        or attributes.get("table_feed_per_rotation", "").strip()
    ):
        return "CT", "dicom:spiral_acquisition", "explicit"

    return "CT", "dicom:modality_default", "inferred"


@dataclass(frozen=True)
class _SeriesInfo:
    uid: str
    file_names: list[str]
    depth: int
    voxel_count: int
    single_file_dimension: int
    description: str = ""
    image_orientation_patient: tuple[float, ...] = ()
    first_image_position_patient: tuple[float, ...] = ()
    pixel_spacing: tuple[float, ...] = ()
    spacing_between_slices: float | None = None


def _float_tuple(value: object, expected_length: int) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ()
    return result if len(result) == expected_length else ()


def _positive_int(value: object, default: int = 1) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return converted if converted > 0 else default


def _directory_snapshot(
    dicom_dir: Path,
) -> tuple[tuple[str, int, int, int, str], list[Path]]:
    resolved = dicom_dir.resolve()
    directory_mtime_ns = resolved.stat().st_mtime_ns
    files: list[Path] = []
    total_size = 0
    digest = hashlib.blake2b(digest_size=16)
    for entry in sorted(resolved.iterdir(), key=lambda path: path.name.casefold()):
        try:
            if not entry.is_file():
                continue
            stat = entry.stat()
        except OSError:
            continue
        files.append(entry)
        total_size += stat.st_size
        digest.update(entry.name.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b":")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    fingerprint = (
        str(resolved),
        len(files),
        directory_mtime_ns,
        total_size,
        digest.hexdigest(),
    )
    return fingerprint, files


def _scan_dicom_headers(files: list[Path]) -> tuple[_SeriesInfo, ...]:
    groups: dict[str, list[dict[str, object]]] = {}
    for path in files:
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=_INDEX_TAGS,
                force=True,
            )
        except (InvalidDicomError, OSError, EOFError, ValueError):
            continue

        uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not uid:
            continue
        frames = _positive_int(getattr(dataset, "NumberOfFrames", 1))
        rows = _positive_int(getattr(dataset, "Rows", 0), default=0)
        columns = _positive_int(getattr(dataset, "Columns", 0), default=0)
        try:
            instance_number = float(getattr(dataset, "InstanceNumber", "inf"))
        except (TypeError, ValueError):
            instance_number = float("inf")
        groups.setdefault(uid, []).append(
            {
                "path": str(path),
                "frames": frames,
                "rows": rows,
                "columns": columns,
                "instance_number": instance_number,
                "description": str(
                    getattr(dataset, "SeriesDescription", "")
                ).strip(),
                "orientation": _float_tuple(
                    getattr(dataset, "ImageOrientationPatient", ()), 6
                ),
                "position": _float_tuple(
                    getattr(dataset, "ImagePositionPatient", ()), 3
                ),
                "pixel_spacing": _float_tuple(
                    getattr(dataset, "PixelSpacing", ()), 2
                ),
                "spacing_between_slices": getattr(
                    dataset,
                    "SpacingBetweenSlices",
                    getattr(dataset, "SliceThickness", None),
                ),
            }
        )

    series: list[_SeriesInfo] = []
    for uid, records in groups.items():
        reference_orientation = next(
            (
                record["orientation"]
                for record in records
                if record["orientation"]
            ),
            (),
        )
        normal = None
        if reference_orientation:
            orientation = np.asarray(reference_orientation, dtype=np.float64)
            candidate = np.cross(orientation[:3], orientation[3:])
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-8:
                normal = candidate / norm

        def slice_sort_key(record: dict[str, object]) -> tuple:
            position = record["position"]
            if normal is not None and position:
                coordinate = float(
                    np.dot(np.asarray(position, dtype=np.float64), normal)
                )
                return (
                    0,
                    coordinate,
                    record["instance_number"],
                    str(record["path"]).casefold(),
                )
            return (
                1,
                record["instance_number"],
                str(record["path"]).casefold(),
            )

        records.sort(key=slice_sort_key)
        first = records[0]
        file_names = [str(record["path"]) for record in records]
        depth = sum(int(record["frames"]) for record in records)
        voxel_count = sum(
            int(record["rows"])
            * int(record["columns"])
            * int(record["frames"])
            for record in records
        )
        try:
            slice_spacing = float(first["spacing_between_slices"])
        except (TypeError, ValueError):
            slice_spacing = None
        series.append(
            _SeriesInfo(
                uid=uid,
                file_names=file_names,
                depth=depth,
                voxel_count=voxel_count,
                single_file_dimension=(
                    3 if len(records) == 1 and depth > 1 else 2
                ),
                description=str(first["description"]),
                image_orientation_patient=tuple(first["orientation"]),
                first_image_position_patient=tuple(first["position"]),
                pixel_spacing=tuple(first["pixel_spacing"]),
                spacing_between_slices=slice_spacing,
            )
        )
    return tuple(sorted(series, key=lambda info: info.uid))


def _directory_index(dicom_dir: Path) -> tuple[_SeriesInfo, ...]:
    fingerprint, files = _directory_snapshot(dicom_dir)
    with _DIRECTORY_INDEX_CACHE_LOCK:
        cached = _DIRECTORY_INDEX_CACHE.get(fingerprint)
        if cached is not None:
            _DIRECTORY_INDEX_CACHE.move_to_end(fingerprint)
            return cached

    index = _scan_dicom_headers(files)
    with _DIRECTORY_INDEX_CACHE_LOCK:
        _DIRECTORY_INDEX_CACHE[fingerprint] = index
        _DIRECTORY_INDEX_CACHE.move_to_end(fingerprint)
        while len(_DIRECTORY_INDEX_CACHE) > _DIRECTORY_INDEX_CACHE_CAPACITY:
            _DIRECTORY_INDEX_CACHE.popitem(last=False)
    return index


def clear_dicom_directory_cache() -> None:
    """Clear cached DICOM directory header indexes."""
    with _DIRECTORY_INDEX_CACHE_LOCK:
        _DIRECTORY_INDEX_CACHE.clear()


def list_dicom_series(dicom_dir: str | Path) -> list[Metadata]:
    """Return cached, pixel-free metadata for each series in a directory."""
    directory = Path(dicom_dir).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {directory}")
    return [
        {
            "series_uid": info.uid,
            "series_description": info.description,
            "file_names": tuple(info.file_names),
            "number_of_files": len(info.file_names),
            "series_depth": info.depth,
            "series_voxel_count": info.voxel_count,
            "image_orientation_patient": info.image_orientation_patient,
            "first_image_position_patient": info.first_image_position_patient,
            "pixel_spacing": info.pixel_spacing,
            "spacing_between_slices": info.spacing_between_slices,
        }
        for info in _directory_index(directory)
    ]


def _inspect_series(dicom_dir: Path, uid: str) -> _SeriesInfo:
    matches = {
        info.uid: info
        for info in _directory_index(dicom_dir)
    }
    try:
        return matches[uid]
    except KeyError as exc:
        raise ValueError(f"DICOM series contains no readable files: {uid}") from exc


def _choose_series(dicom_dir: Path, series_uid: str | None) -> _SeriesInfo:
    series = _directory_index(dicom_dir)
    if not series:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    if series_uid is not None:
        matches = {info.uid: info for info in series}
        if series_uid not in matches:
            raise ValueError(
                f"Series UID {series_uid!r} was not found. "
                f"Available UIDs: {', '.join(matches)}"
            )
        return matches[series_uid]

    candidates = [info for info in series if info.depth > 1]

    if not candidates:
        raise ValueError(
            "No multi-slice DICOM series was found after excluding single-slice "
            "series."
        )

    largest_voxel_count = max(info.voxel_count for info in candidates)
    tied = [info for info in candidates if info.voxel_count == largest_voxel_count]
    # The index is sorted by UID, making ties deterministic.
    return min(tied, key=lambda info: info.uid)


def _read_ct_series(
    info: _SeriesInfo,
    require_ct: bool,
) -> tuple[sitk.Image, str, dict[str, str]]:
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

    attributes = {
        name: metadata_get(tag).strip() if metadata_has(tag) else ""
        for name, tag in _CLASSIFICATION_TAGS.items()
    }
    modality = ""
    modality_tag = "0008|0060"
    if metadata_has(modality_tag):
        modality = metadata_get(modality_tag).strip().upper()
    if require_ct and modality not in {"CT", "CBCT"}:
        shown = modality or "missing"
        raise ValueError(
            f"The selected series has Modality={shown!r}, not CT/CBCT; "
            "HU is a CT concept. "
            "Pass require_ct=False only if the stored/rescaled values are meaningful "
            "for your modality."
        )

    return image, modality, attributes


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
    interpolators = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
    }
    try:
        sitk_interpolator = interpolators[interpolation]
    except KeyError as exc:
        raise ValueError("interpolation must be 'linear' or 'nearest'") from exc

    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    if np.allclose(spacing, mmpd, rtol=1e-7, atol=1e-7) and np.allclose(
        direction, np.eye(3), rtol=0.0, atol=1e-7
    ):
        return image

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
    source_image, dicom_modality, dicom_attributes = _read_ct_series(
        selected_series, require_ct=require_ct
    )
    modality, modality_source, modality_confidence = _infer_dicom_modality(
        dicom_modality, dicom_attributes
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
        "series_description": selected_series.description,
        "series_depth": selected_series.depth,
        "series_voxel_count": selected_series.voxel_count,
        "image_orientation_patient": (
            selected_series.image_orientation_patient
        ),
        "first_image_position_patient": (
            selected_series.first_image_position_patient
        ),
        "pixel_spacing": selected_series.pixel_spacing,
        "spacing_between_slices": selected_series.spacing_between_slices,
        "modality": modality,
        "dicom_modality": dicom_modality or "UNKNOWN",
        "modality_source": modality_source,
        "modality_confidence": modality_confidence,
        "dicom_attributes": dicom_attributes,
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


__all__ = [
    "clear_dicom_directory_cache",
    "list_dicom_series",
    "load_dicom_hu_lps",
]
