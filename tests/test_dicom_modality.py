from unittest.mock import patch

import numpy as np

from ct_mri_dicom_nii_reader.body_data.body_data_imp.dicom_to_hu_lps import (
    _infer_dicom_modality,
)
from ct_mri_dicom_nii_reader.body_data.main import DicomBodyDataLoader
from ct_mri_dicom_nii_reader.visualization import _air_value


def test_3d_carm_station_is_classified_as_cbct():
    modality, source, confidence = _infer_dicom_modality(
        "CT",
        {
            "station_name": "YiYing_3DCarm",
            "image_type": "DERIVED\\PRIMARY\\AXIAL\\3DSLICE",
        },
    )

    assert modality == "CBCT"
    assert source == "dicom:station_name"
    assert confidence == "explicit"


def test_spiral_ct_is_not_misclassified_as_cbct():
    modality, source, confidence = _infer_dicom_modality(
        "CT",
        {
            "manufacturer_model_name": "SOMATOM Force",
            "station_name": "CTAWP76709",
            "image_type": "ORIGINAL\\PRIMARY\\AXIAL\\CT_SOM5 SPI",
            "spiral_pitch_factor": "0.8",
            "table_feed_per_rotation": "46",
        },
    )

    assert modality == "CT"
    assert source == "dicom:spiral_acquisition"
    assert confidence == "explicit"


def test_ambiguous_derived_ct_remains_ct():
    modality, source, confidence = _infer_dicom_modality(
        "CT", {"image_type": "DERIVED\\PRIMARY\\AXIAL\\3DSLICE"}
    )

    assert modality == "CT"
    assert source == "dicom:modality_default"
    assert confidence == "inferred"


def test_dicom_loader_returns_cbct_body_data():
    metadata = {"modality": "CBCT"}
    with patch(
        "ct_mri_dicom_nii_reader.body_data.main.load_dicom_hu_lps",
        return_value=(np.zeros((2, 2, 2), dtype=np.float32), metadata),
    ):
        body_data = DicomBodyDataLoader().load_data("example/series.dcm")

    assert body_data.get_type() == "cbct"
    assert body_data._get_air_value() == -1024
    assert _air_value(body_data.get_type()) == -1024.0
