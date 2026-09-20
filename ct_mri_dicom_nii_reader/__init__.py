"""Medical image (CT / CBCT / MRI / mask) reading and writing utilities.

This package loads CT DICOM series and NIfTI volumes into isotropic NumPy
volumes with ``(L, P, S)`` axis order, wraps them in the :class:`BodyData`
container, and offers slice access, preview GUIs and a compact ``.ubd.npz``
storage format.

It is a pure Python programming library; no command-line interface is
provided.
"""

from .body_data import (
    DataLoaderNotMatch,
    BodyDataNotInitialized,
    BodyDataTypeError,
    NoAvailableDataLoader,
    RoiRect,
    BodyDataLoaderManager,
    BodyDataLoader,
    DicomBodyDataLoader,
    NiiBodyDataLoader,
    UnifiedBodyDataLoader,
    BodyDataSlice,
    BodyData,
)
from .visualization import (
    BodyDataCommonGrid,
    MultimodalFusionComposer,
    TwoImageFusionViewer,
    ThreeImageOverlayViewer,
    display_window,
    normalize_to_u8,
    resample_to_mmpd,
)
from .bone_segmentation import (
    BoneImageType,
    apply_bone_mask,
    estimate_bone_thresholds,
    extract_bone_mask,
    extract_cbct_bone_mask,
    extract_ct_bone_mask,
)
from .projection import (
    ProjectionPlane,
    get_lps_max_projections,
    max_intensity_projection,
)

__all__ = [
    "DataLoaderNotMatch",
    "BodyDataNotInitialized",
    "BodyDataTypeError",
    "NoAvailableDataLoader",
    "RoiRect",
    "BodyDataLoaderManager",
    "BodyDataLoader",
    "DicomBodyDataLoader",
    "NiiBodyDataLoader",
    "UnifiedBodyDataLoader",
    "BodyDataSlice",
    "BodyData",
    "BodyDataCommonGrid",
    "MultimodalFusionComposer",
    "TwoImageFusionViewer",
    "ThreeImageOverlayViewer",
    "display_window",
    "normalize_to_u8",
    "resample_to_mmpd",
    "BoneImageType",
    "apply_bone_mask",
    "estimate_bone_thresholds",
    "extract_bone_mask",
    "extract_cbct_bone_mask",
    "extract_ct_bone_mask",
    "ProjectionPlane",
    "get_lps_max_projections",
    "max_intensity_projection",
]
