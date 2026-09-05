"""Load a 3-D NIfTI image onto an isotropic L/P/S NumPy grid."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import tempfile
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import SimpleITK as sitk


Metadata: TypeAlias = dict[str, object]


def _explicit_modality(value: object) -> Literal["CT", "MR"] | None:
    if isinstance(value, dict) and "Value" in value:
        value = value["Value"]
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    normalized = re.sub(r"[^A-Z]+", " ", str(value).upper()).strip()
    if normalized in {"CT", "COMPUTED TOMOGRAPHY"}:
        return "CT"
    if normalized in {
        "MR",
        "MRI",
        "MAGNETIC RESONANCE",
        "MAGNETIC RESONANCE IMAGING",
    }:
        return "MR"
    return None


def _modality_from_mapping(mapping: object) -> Literal["CT", "MR"] | None:
    if not isinstance(mapping, dict):
        return None
    for key, value in mapping.items():
        normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if normalized_key in {"modality", "00080060"}:
            modality = _explicit_modality(value)
            if modality is not None:
                return modality
    return None


def _modality_from_text(text: str) -> Literal["CT", "MR"] | None:
    searchable = re.sub(r"[_-]+", " ", text)
    ct_hit = bool(
        re.search(r"\bct\b|computed\s*tomograph", searchable, re.IGNORECASE)
    )
    mr_hit = bool(
        re.search(
            r"\bmri?\b|magnetic\s*resonance|"
            r"\b(?:t1w?|t2w?|flair|dwi|adc|bold|asl|swi|mprage)\b",
            searchable,
            re.IGNORECASE,
        )
    )
    if ct_hit != mr_hit:
        return "CT" if ct_hit else "MR"
    return None


def _sidecar_path(nifti_path: Path) -> Path:
    lower_name = nifti_path.name.lower()
    suffix_length = 7 if lower_name.endswith(".nii.gz") else 4
    return nifti_path.with_name(nifti_path.name[:-suffix_length] + ".json")


def _nifti_header_text(image: sitk.Image) -> str:
    wanted_keys = {
        "auxfile",
        "dbname",
        "descrip",
        "intentname",
        "itkfilenotes",
    }
    values: list[str] = []
    for key in image.GetMetaDataKeys():
        normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized_key in wanted_keys:
            values.append(image.GetMetaData(key).strip("\x00 "))
    return " ".join(values)


def _infer_modality(
    nifti_path: Path,
    image: sitk.Image,
    data_zyx: np.ndarray,
) -> tuple[Literal["CT", "MR", "UNKNOWN"], str | None, str]:
    sidecar = _sidecar_path(nifti_path)
    if sidecar.is_file():
        try:
            sidecar_data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            sidecar_data = None
        modality = _modality_from_mapping(sidecar_data)
        if modality is not None:
            return modality, "json_sidecar", "explicit"

    modality = _modality_from_text(_nifti_header_text(image))
    if modality is not None:
        return modality, "nifti_header", "inferred"

    modality = _modality_from_text(nifti_path.name)
    if modality is not None:
        return modality, "filename", "inferred"

    finite_values = data_zyx[np.isfinite(data_zyx)]
    if finite_values.size:
        sample_step = max(1, finite_values.size // 1_000_000)
        sample = finite_values[::sample_step]
        percentile_1, percentile_99 = np.percentile(sample, (1.0, 99.0))
        if percentile_1 <= -500.0 and percentile_99 >= 200.0:
            return "CT", "intensity_heuristic", "inferred"

    return "UNKNOWN", None, "unknown"


def _image_support_bounds_lps(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
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
    outside_value: float,
) -> sitk.Image:
    lower_edge, upper_edge = _image_support_bounds_lps(image)
    size_ratio = (upper_edge - lower_edge) / mmpd
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
    resampler.SetDefaultPixelValue(outside_value)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetOutputSpacing([mmpd, mmpd, mmpd])
    resampler.SetOutputOrigin(output_origin.tolist())
    resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    resampler.SetSize([int(value) for value in output_size])
    return resampler.Execute(image)


def _image_affine_lps(image: sitk.Image) -> np.ndarray:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = image.GetOrigin()
    return affine_lps


def _is_gzip_compressed(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _read_nifti_image(path: Path) -> sitk.Image:
    """Read a NIfTI image regardless of its compression or path encoding.

    SimpleITK selects its ImageIO from the filename extension and, on Windows,
    cannot reliably open paths containing non-ASCII characters (the narrow
    filename is decoded using the ANSI code page). When the on-disk compression
    disagrees with the extension, or the path is not pure ASCII, stage the image
    into a plain ``.nii`` file under an ASCII-only temporary path before handing
    it to SimpleITK.
    """
    is_gzipped = _is_gzip_compressed(path)
    lower_name = path.name.lower()

    extension_matches_content = (
        lower_name.endswith(".nii.gz")
        if is_gzipped
        else lower_name.endswith(".nii")
    )
    path_is_ascii = str(path).isascii()

    if path_is_ascii and extension_matches_content:
        reader = sitk.ImageFileReader()
        reader.SetOutputPixelType(sitk.sitkFloat32)
        reader.SetFileName(str(path))
        return reader.Execute()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "image.nii"
        if is_gzipped:
            with gzip.open(path, "rb") as source, temp_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        else:
            shutil.copyfile(path, temp_path)
        reader = sitk.ImageFileReader()
        reader.SetOutputPixelType(sitk.sitkFloat32)
        reader.SetFileName(str(temp_path))
        return reader.Execute()


def load_nifti_lps(
    nifti_file: str | Path,
    mmpd: float,
    *,
    interpolation: Literal["linear", "nearest"] = "linear",
    outside_value: float | None = None,
) -> tuple[np.ndarray, Metadata]:
    """Load a 3-D ``.nii`` or ``.nii.gz`` file in isotropic L/P/S order.

    The returned array has shape ``(n_L, n_P, n_S)``. Increasing indices on
    axes 0, 1 and 2 move in the patient's Left, Posterior and Superior
    directions. Each output voxel is ``mmpd`` mm on all three sides.

    Modality detection is conservative. It checks a same-name JSON sidecar,
    NIfTI header text, the filename, then a strong CT intensity pattern. The
    metadata modality is ``"CT"``, ``"MR"`` or ``"UNKNOWN"``.

    Args:
        nifti_file: Path to a three-dimensional ``.nii`` or ``.nii.gz`` file.
        mmpd: Isotropic output voxel width in millimetres.
        interpolation: ``"linear"`` for intensity images or ``"nearest"`` for
            discrete values such as segmentation labels.
        outside_value: Fill value outside the source image support. If omitted,
            it is ``-1024`` for detected CT and ``0`` otherwise.

    Returns:
        ``(volume_lps, metadata)``. ``volume_lps`` is a contiguous ``float32``
        array and metadata describes its geometry, source geometry and modality.
    """
    path = Path(nifti_file).expanduser()
    lower_name = path.name.lower()
    if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
        raise ValueError("nifti_file must end with .nii or .nii.gz")
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")

    if isinstance(mmpd, bool):
        raise ValueError("mmpd must be a finite number greater than zero")
    try:
        mmpd = float(mmpd)
    except (TypeError, ValueError) as exc:
        raise ValueError("mmpd must be a finite number greater than zero") from exc
    if not np.isfinite(mmpd) or mmpd <= 0:
        raise ValueError("mmpd must be a finite number greater than zero")

    image = _read_nifti_image(path)
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3-D NIfTI image, got {image.GetDimension()} dimensions")

    source_data_zyx = sitk.GetArrayViewFromImage(image)
    modality, modality_source, modality_confidence = _infer_modality(
        path, image, source_data_zyx
    )
    if outside_value is None:
        outside_value = -1024.0 if modality == "CT" else 0.0
    outside_value = float(outside_value)

    output_image = _resample_to_isotropic_lps(
        image,
        mmpd=mmpd,
        interpolation=interpolation,
        outside_value=outside_value,
    )
    # SimpleITK exposes [S, P, L] (z, y, x); transpose to [L, P, S].
    volume_lps = np.ascontiguousarray(
        np.transpose(sitk.GetArrayFromImage(output_image), (2, 1, 0)),
        dtype=np.float32,
    )

    source_affine_lps = _image_affine_lps(image)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    output_affine_lps = _image_affine_lps(output_image)
    sidecar = _sidecar_path(path)
    metadata: Metadata = {
        "axis_order": ("L", "P", "S"),
        "shape_lps": tuple(int(value) for value in volume_lps.shape),
        "spacing_lps_mm": (mmpd, mmpd, mmpd),
        "origin_lps_mm": tuple(float(value) for value in output_image.GetOrigin()),
        "output_affine_lps_mm": output_affine_lps,
        "outside_value": outside_value,
        "modality": modality,
        "modality_source": modality_source,
        "modality_confidence": modality_confidence,
        "source_path": str(path.resolve()),
        "source_size_xyz": tuple(int(value) for value in image.GetSize()),
        "source_spacing_xyz_mm": tuple(float(value) for value in image.GetSpacing()),
        "source_origin_lps_mm": tuple(float(value) for value in image.GetOrigin()),
        "source_direction_xyz_to_lps": np.asarray(
            image.GetDirection(), dtype=np.float64
        ).reshape(3, 3),
        "source_affine_lps_mm": source_affine_lps,
        "source_affine_ras_mm": lps_to_ras @ source_affine_lps,
        "json_sidecar": str(sidecar.resolve()) if sidecar.is_file() else None,
    }
    return volume_lps, metadata


__all__ = ["load_nifti_lps"]
