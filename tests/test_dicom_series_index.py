from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from ct_mri_dicom_nii_reader.body_data.body_data_imp.dicom_to_hu_lps import (
    _choose_series,
    _resample_to_isotropic_lps,
    clear_dicom_directory_cache,
    list_dicom_series,
    load_dicom_hu_lps,
)
from ct_mri_dicom_nii_reader.body_data.main import DicomBodyDataLoader


@pytest.fixture(autouse=True)
def clear_series_cache():
    clear_dicom_directory_cache()
    yield
    clear_dicom_directory_cache()


def write_dicom_header(
    path: Path,
    series_uid: str,
    instance_number: int,
    z_position: float,
    *,
    rows: int = 16,
    columns: int = 16,
    description: str = "",
    study_uid: str | None = None,
    pixel_value: int | None = None,
    pixel_spacing: float = 0.5,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=bytes(128),
    )
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid or generate_uid()
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.SeriesDescription = description
    dataset.Modality = "CT"
    dataset.InstanceNumber = instance_number
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z_position]
    dataset.PixelSpacing = [pixel_spacing, pixel_spacing]
    dataset.SpacingBetweenSlices = 1.0
    dataset.SliceThickness = 1.0
    if pixel_value is not None:
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = 0
        dataset.PixelData = np.full(
            (rows, columns), pixel_value, dtype=np.int16
        ).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def make_multi_series_directory(tmp_path: Path) -> tuple[str, str]:
    smaller_uid = "1.2.826.0.1.3680043.10.1000.1"
    larger_uid = "1.2.826.0.1.3680043.10.1000.2"
    write_dicom_header(
        tmp_path / "small-2.dcm", smaller_uid, 2, 2.0,
        description="smaller",
    )
    write_dicom_header(
        tmp_path / "small-1.dcm", smaller_uid, 1, 1.0,
        description="smaller",
    )
    write_dicom_header(
        tmp_path / "large-high.dcm", larger_uid, 20, 20.0,
        rows=32, columns=32, description="larger",
    )
    write_dicom_header(
        tmp_path / "large-low.dcm", larger_uid, 10, 10.0,
        rows=32, columns=32, description="larger",
    )
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")
    return smaller_uid, larger_uid


def test_one_header_scan_replaces_all_gdcm_directory_discovery(tmp_path):
    _, larger_uid = make_multi_series_directory(tmp_path)

    with (
        patch.object(
            sitk.ImageSeriesReader,
            "GetGDCMSeriesIDs",
            side_effect=AssertionError("GDCM directory scan was called"),
        ),
        patch.object(
            sitk.ImageSeriesReader,
            "GetGDCMSeriesFileNames",
            side_effect=AssertionError("GDCM directory scan was called"),
        ),
    ):
        selected = _choose_series(tmp_path, series_uid=None)

    assert selected.uid == larger_uid
    assert [Path(path).name for path in selected.file_names] == [
        "large-low.dcm",
        "large-high.dcm",
    ]
    assert selected.depth == 2
    assert selected.voxel_count == 2 * 32 * 32


def test_directory_index_is_cached_and_invalidated_by_file_changes(tmp_path):
    smaller_uid, _ = make_multi_series_directory(tmp_path)
    file_count = len(list(tmp_path.iterdir()))

    with patch(
        "ct_mri_dicom_nii_reader.body_data.body_data_imp."
        "dicom_to_hu_lps.pydicom.dcmread",
        wraps=pydicom.dcmread,
    ) as dcmread:
        first = list_dicom_series(tmp_path)
        second = list_dicom_series(tmp_path)
        assert first == second
        assert dcmread.call_count == file_count

        write_dicom_header(
            tmp_path / "small-3.dcm", smaller_uid, 3, 3.0,
            description="smaller",
        )
        refreshed = list_dicom_series(tmp_path)

    assert dcmread.call_count == file_count + file_count + 1
    smaller = next(
        item for item in refreshed if item["series_uid"] == smaller_uid
    )
    assert smaller["number_of_files"] == 3


def test_known_series_uid_uses_index_lookup(tmp_path):
    smaller_uid, larger_uid = make_multi_series_directory(tmp_path)

    selected = _choose_series(tmp_path, series_uid=smaller_uid)

    assert selected.uid == smaller_uid
    with pytest.raises(ValueError, match="Available UIDs"):
        _choose_series(tmp_path, series_uid=larger_uid + ".404")


def test_series_listing_exposes_header_index_metadata(tmp_path):
    _, larger_uid = make_multi_series_directory(tmp_path)

    listed = list_dicom_series(tmp_path)
    larger = next(item for item in listed if item["series_uid"] == larger_uid)

    assert larger["series_description"] == "larger"
    assert larger["image_orientation_patient"] == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    assert larger["first_image_position_patient"] == (0.0, 0.0, 10.0)
    assert larger["pixel_spacing"] == (0.5, 0.5)
    assert larger["spacing_between_slices"] == 1.0
    assert tuple(Path(path).name for path in larger["file_names"]) == (
        "large-low.dcm",
        "large-high.dcm",
    )


def test_canonical_target_grid_skips_resampling():
    image = sitk.Image([4, 5, 6], sitk.sitkFloat32)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    with patch(
        "ct_mri_dicom_nii_reader.body_data.body_data_imp."
        "dicom_to_hu_lps._image_support_bounds_lps",
        side_effect=AssertionError("resampling path was used"),
    ):
        result = _resample_to_isotropic_lps(
            image,
            mmpd=1.0,
            interpolation="linear",
            outside_hu=-1024.0,
        )

    assert result is image


def test_canonical_fast_path_still_validates_interpolation():
    image = sitk.Image([2, 2, 2], sitk.sitkFloat32)

    with pytest.raises(ValueError, match="interpolation"):
        _resample_to_isotropic_lps(
            image,
            mmpd=1.0,
            interpolation="invalid",  # type: ignore[arg-type]
            outside_hu=-1024.0,
        )


def test_selected_series_file_order_is_valid_for_simpleitk_pixel_read(tmp_path):
    series_uid = "1.2.826.0.1.3680043.10.1000.20"
    study_uid = generate_uid()
    write_dicom_header(
        tmp_path / "second.dcm",
        series_uid,
        2,
        1.0,
        rows=3,
        columns=4,
        study_uid=study_uid,
        pixel_value=300,
        pixel_spacing=1.0,
    )
    write_dicom_header(
        tmp_path / "first.dcm",
        series_uid,
        1,
        0.0,
        rows=3,
        columns=4,
        study_uid=study_uid,
        pixel_value=100,
        pixel_spacing=1.0,
    )

    volume, metadata = load_dicom_hu_lps(
        tmp_path,
        mmpd=1.0,
        series_uid=series_uid,
        return_metadata=True,
    )

    assert volume.shape == (4, 3, 2)
    assert np.all(volume[:, :, 0] == 100.0)
    assert np.all(volume[:, :, 1] == 300.0)
    assert metadata["series_uid"] == series_uid


def test_dicom_loader_passes_configured_series_uid():
    loader = DicomBodyDataLoader()
    loader.set_series_uid("1.2.826.0.1.3680043.10.1000.9")

    with patch(
        "ct_mri_dicom_nii_reader.body_data.main.load_dicom_hu_lps",
        return_value=(
            np.zeros((2, 2, 2), dtype=np.float32),
            {"modality": "CT"},
        ),
    ) as load:
        loader.load_data("example/series.dcm")

    assert load.call_args.kwargs["series_uid"] == loader.get_series_uid()
