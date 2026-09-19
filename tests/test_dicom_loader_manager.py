from unittest.mock import patch

import pytest

from ct_mri_dicom_nii_reader.body_data.main import (
    BodyDataLoaderManager,
    DicomBodyDataLoader,
    NoAvailableDataLoader,
)


def test_manager_uses_dicom_loader_for_ten_extensionless_files(tmp_path):
    files = [tmp_path / str(index) for index in range(10)]
    for file in files:
        file.touch()
    (tmp_path / "notes.txt").touch()

    sentinel = object()
    with patch.object(DicomBodyDataLoader, "load_file", return_value=sentinel) as load:
        result = BodyDataLoaderManager().load_file(str(files[0]))

    assert result is sentinel
    load.assert_called_once_with(str(files[0]))


def test_manager_rejects_only_nine_extensionless_files(tmp_path):
    files = [tmp_path / str(index) for index in range(9)]
    for file in files:
        file.touch()
    for index in range(5):
        (tmp_path / f"note-{index}.txt").touch()

    with pytest.raises(NoAvailableDataLoader):
        BodyDataLoaderManager().load_file(str(files[0]))


def test_manager_uses_dicom_loader_for_dicom_with_numeric_suffix(tmp_path):
    file = tmp_path / "CT.X.1.2.3.861"
    file.write_bytes(bytes(128) + b"DICM")

    sentinel = object()
    with patch.object(DicomBodyDataLoader, "load_file", return_value=sentinel) as load:
        result = BodyDataLoaderManager().load_file(str(file))

    assert result is sentinel
    load.assert_called_once_with(str(file))
