"""Medical image (CT / MRI / mask) reading and writing utilities.

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
]
