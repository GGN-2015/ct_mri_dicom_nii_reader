# ct_mri_dicom_nii_reader

A small, dependency-light Python library for reading medical image volumes
(CT / MRI / masks) from **DICOM series** and **NIfTI** files into isotropic,
LPS-ordered NumPy arrays, plus a compact container class with slice access,
processing helpers and optional Tkinter/Pillow preview GUIs.

The classes are extracted verbatim from the `regknee` knee-registration
project so they can be reused by future projects as a self-contained data
layer. No command-line interface is provided; this is a pure Python
programming interface.

## Features

- **DICOM series -> isotropic HU volume**: `load_dicom_hu_lps` reads a DICOM
  directory (auto-selecting the largest multi-slice series), applies Rescale
  Slope / Intercept, and resamples onto an isotropic LPS grid
  (`float32` array with shape `(n_L, n_P, n_S)`).
- **NIfTI -> isotropic LPS volume**: `load_nifti_lps` reads `.nii` /
  `.nii.gz` files (including non-ASCII paths and mismatched compression on
  Windows) with conservative CT/MR modality detection.
- **`BodyData` container**: keeps a volume together with its voxel size
  (`mmpd`) and image type (`ct` / `mri` / `mask`), provides slice access,
  cloning, percentile queries, ROI extraction, MIND descriptor computation
  and GUI preview.
- **Loader abstraction**: `BodyDataLoaderManager` dispatches by file
  extension so one call loads DICOM, NIfTI or the compact unified
  `.ubd.npz` format.
- **Compact `.ubd.npz` storage**: `BodyData.save` / `UnifiedBodyDataLoader`
  round-trip a volume, its type and voxel size through a single compressed
  NumPy archive.
- **Optional GUIs**: 2-D and 3-D grayscale preview windows built on tkinter
  and Pillow (kept for interactive exploration; headless workflows never
  need them). Windows open at a default upscaled size (smallest image side
  512 px); resizing the window rescales the image with it while preserving
  the original aspect ratio.
- **Visualization classes**: `BodyDataCommonGrid` (origin alignment +
  unified sampling + common grid), `MultimodalFusionComposer` (per-slice
  RGB rendering), `TwoImageFusionViewer` (amber/cyan edge fusion of two
  volumes) and `ThreeImageOverlayViewer` (fusion + semi-transparent red
  mask overlay).

## Installation

```bash
# install from pypi
pip install ct_mri_dicom_nii_reader

# install from git repo
pip install -e .
```

Python >= 3.10 is required (the code uses modern type syntax and `match`).

Runtime dependencies (declared in `pyproject.toml`):

| Dependency | Purpose |
| --- | --- |
| `numpy` | array storage and processing |
| `simpleitk` | DICOM and NIfTI decoding, resampling |
| `pillow` | GUI preview and display helpers |
| `tkinter` (optional) | preview windows; imported only when a preview is started |

## Quick start

```python
from ct_mri_dicom_nii_reader import BodyDataLoaderManager

manager = BodyDataLoaderManager()

# 1. Load a CT volume from a DICOM series (any .dcm file inside the folder).
#    Voxels are resampled to 1 mm isotropic on an LPS grid.
ct = manager.load_file("path/to/dicom/IM0001.dcm")

# 2. Load an MRI volume from a NIfTI file.
mri = manager.load_file("path/to/image.nii.gz")

# 3. Save as the compact unified format, then reload it.
ct.save("out/ct_volume")               # writes out/ct_volume.ubd.npz
ct_again = manager.load_file("out/ct_volume.ubd.npz")
```

### Working with `BodyData`

```python
from ct_mri_dicom_nii_reader import BodyData, RoiRect

print(ct.get_type())        # 'ct' | 'mri' | 'mask'  (or None if uninitialized)
print(ct.get_size())        # (n_L, n_P, n_S)
print(ct.get_mmpd())        # mm per dot (voxel width)
print(ct.get_pos(0, 0, 0))  # value at voxel [l, p, s]

slice_l = ct.get_slice_l(100)      # BodyDataSlice view of a plane
slice_s = ct.get_slice_s(50)
slice_s.gui_preview()              # blocking 2-D preview window
ct.gui_preview()                   # blocking 3-D preview window with z slider

print(ct.percentile(1), ct.percentile(99))   # intensity percentiles
clone = ct.clone()                  # deep copy of the volume
mind = ct.get_mind()                # scalar MIND descriptor as BodyData

# Keep an x-z rectangle and blank everything else (air value is modality-aware).
roi = RoiRect(xmin=80, xmax=180, ymin=40, ymax=120)
part = ct.get_part(roi)
```

## Core concepts

### LPS axis order

All loaded volumes use the patient **LPS** convention:

- axis 0 -> **L**eft, axis 1 -> **P**osterior, axis 2 -> **S**uperior;
- `volume[l, p, s]` with increasing indices moving Left, Posterior, Superior;
- every output voxel is exactly `mmpd` mm on each side (isotropic).

DICOM rescaling (Rescale Slope / Intercept) is applied by
SimpleITK/GDCM while reading, so CT values are Hounsfield units.

### `mmpd` (mm per dot)

`mmpd` is the isotropic output voxel width in millimetres. All loaders
resample the source image onto an `mmpd`-spaced LPS grid:

```python
ct = manager.load_file("path/to/series.dcm", mmpd=1.5)  # 1.5 mm voxels
```

If omitted it defaults to `1.0` mm.

### Image types

`BodyData.get_type()` returns one of:

| Type | Meaning |
| --- | --- |
| `"ct"` | CT / DX / CR / CBCT volumes (HU) |
| `"mri"` | MR volumes |
| `"mask"` | segmentation labels or unknown modality |

Masks whose values already lie in `[0, 1]` are automatically binarized to
`int8` (`0` / `1`) on load.

## API reference

### `BodyDataLoaderManager`

```python
manager = BodyDataLoaderManager()
body_data = manager.load_file(filepath, mmpd=None)
```

Tries each registered loader (DICOM, NIfTI, unified) in turn and returns the
first match. Raises `NoAvailableDataLoader(filepath)` when no loader can
handle the path. `mmpd` defaults to `1.0`.

### Loader classes

`BodyDataLoader` is the abstract base class: subclasses implement
`check_match(filepath)` and `load_data(filepath)`. Users should call
`load_file(filepath)` instead of `load_data`, which validates the match and
post-processes masks.

| Class | Handles |
| --- | --- |
| `DicomBodyDataLoader` | `*.dcm` (loads the folder containing the file) |
| `NiiBodyDataLoader` | `*.nii`, `*.nii.gz` |
| `UnifiedBodyDataLoader` | `*.ubd.npz` |

All loaders accept `get_mmpd()` / `set_mmpd()` for the output voxel width.

### `BodyData`

| Method | Description |
| --- | --- |
| `from_array(data, image_type, mmpd)` | Wrap a NumPy array in LPS order |
| `to_numpy(copy=False)` | Return the internal 3-D NumPy array (copy only when `copy=True`) |
| `get_initialized()` | Whether `image_type` has been set |
| `get_type()` | `"ct"`, `"mri"`, `"mask"` or `None` |
| `unify_to_mask()` | Binarize values to `int8` at threshold 0.5 |
| `get_size()` | `(n_L, n_P, n_S)` |
| `get_pos(l, p, s)` | Value at a voxel |
| `get_slice_l(l)` / `get_slice_p(p)` / `get_slice_s(s)` | Plane views (`BodyDataSlice`) |
| `get_mmpd()` / `set_mmpd(mmpd)` | Voxel width accessors |
| `percentile(idx)` | `idx` in `[0, 100]` across all voxels |
| `clone()` | Deep copy |
| `get_part(roi_rect)` | Keep an x-z rectangle, fill elsewhere with the modality-specific air value |
| `get_mind()` | Scalar 3-D MIND descriptor of the volume |
| `save(filepath)` | Write `.ubd.npz` (extension added if missing, parent dirs created) |
| `gui_preview()` | Blocking 3-D slice browser window |

### `BodyDataSlice`

A 2-D plane view returned by `BodyData.get_slice_*`:

- `get_pos(x, y)`, `get_size()` -> `(x, y)`,
- `to_numpy(copy=False)` -> internal 2-D NumPy array,
- `get_mmpd()` / `set_mmpd()`, `get_body_data()` -> owning `BodyData` or `None`,
- `clone()` -> deep copy,
- `gui_preview()` -> blocking grayscale window.

### `RoiRect`

```python
roi = RoiRect(xmin, xmax, ymin, ymax)   # x-z rectangle in voxel indices
```

Accessors: `get_xmin()`, `get_xmax()`, `get_ymin()`, `get_ymax()`.

### Exceptions

| Exception | Raised when |
| --- | --- |
| `NoAvailableDataLoader(filepath)` | No loader matches the file extension |
| `DataLoaderNotMatch` | A loader's `load_file` is called on a non-matching path |
| `BodyDataNotInitialized` | An operation requires an initialized `BodyData` |
| `BodyDataTypeError(image_type)` | The image type is not `ct` / `mri` / `mask` |

## Low-level loading functions

For full control, the implementation functions can be used directly
(they are also what the loader classes call internally):

```python
from ct_mri_dicom_nii_reader.body_data.body_data_imp.dicom_to_hu_lps import load_dicom_hu_lps
from ct_mri_dicom_nii_reader.body_data.body_data_imp.nifti_to_lps import load_nifti_lps

volume, metadata = load_dicom_hu_lps(
    "path/to/dicom_dir",
    mmpd=1.0,
    series_uid=None,            # optional DICOM Series Instance UID
    interpolation="linear",     # or "nearest"
    outside_hu=-1024.0,         # fill value outside the scanned field of view
    require_ct=True,            # reject non-CT series
    return_metadata=True,
)

volume, metadata = load_nifti_lps(
    "path/to/image.nii.gz",
    mmpd=1.0,
    interpolation="linear",     # use "nearest" for segmentation labels
    outside_value=None,         # defaults to -1024 (CT) or 0
)
```

DICOM metadata keys include `series_uid`, `series_depth`,
`series_voxel_count`, `modality`, `number_of_files`, and the source
geometry (`source_size_xyz`, `source_spacing_xyz_mm`,
`source_origin_lps_mm`, `source_direction_xyz_to_lps`).

NIfTI metadata keys include `modality` (`"CT"`, `"MR"` or `"UNKNOWN"`),
`modality_source` and `modality_confidence`, the output geometry
(`origin_lps_mm`, `output_affine_lps_mm`, `outside_value`) and the source
geometry (`source_size_xyz`, `source_spacing_xyz_mm`,
`source_origin_lps_mm`, `source_direction_xyz_to_lps`,
`source_affine_lps_mm`, `source_affine_ras_mm`, `json_sidecar`).

### DICOM series selection rules

When `series_uid` is not given, `load_dicom_hu_lps`:

1. lists every series in the directory,
2. excludes single-slice series,
3. picks the multi-slice series with the largest voxel count,
4. breaks ties deterministically by lexicographically smallest UID.

The loader class `DicomBodyDataLoader` relaxes `require_ct` to `False` and
maps DICOM Modality to image type: CT/DX/CR/CBCT -> `"ct"`, MR -> `"mri"`,
everything else -> `"mask"`.

### NIfTI modality detection order

1. same-name JSON sidecar (`Modality` or `00080060` key),
2. NIfTI header text (auxfile / descrip / intent ...),
3. filename (e.g. `t1w`, `flair`, `ct`),
4. intensity heuristic (P1 <= -500 and P99 >= 200 -> CT).

If nothing matches, the modality is `"UNKNOWN"` and the loader treats the
image as a `"mask"`.

## Processing utilities

| Function | Description |
| --- | --- |
| `get_volume_percentile(volume, idx)` | Percentile over all voxels of a 3-D array |
| `compute_mind_image(volume, *, patch_radius, patch_sigma, neighbour_radius)` | Scalar MIND descriptor in `[0, 1]`, same shape as input |
| `fill_outside_xz_rectangle(volume, l, r, t, b, air_value)` | Keep `[l, r) x [t, b)` in the x-z plane, fill the rest |

Import paths:

```python
from ct_mri_dicom_nii_reader.body_data.body_data_imp.volume_percentile import get_volume_percentile
from ct_mri_dicom_nii_reader.body_data.body_data_imp.mind_3d import compute_mind_image
from ct_mri_dicom_nii_reader.body_data.body_data_imp.mask_xz_rectangle import fill_outside_xz_rectangle
```

## GUI previews

```python
from ct_mri_dicom_nii_reader.body_data.body_data_imp.slice_display import show_numpy_gray
from ct_mri_dicom_nii_reader.body_data.body_data_imp.numpy_3d_viewer import show_numpy_3d

show_numpy_gray(slice_2d, vmin, vmax)   # 2-D window, array[x, y] -> pixel (x, y)
show_numpy_3d(array_3d, vmin, vmax)     # 3-D window with z slider
```

Both are blocking (they open a Tk main loop) and use
`BodyData.gui_preview()` / `BodyDataSlice.gui_preview()` as convenience
wrappers.

The windows open at the default upscaled size produced by
`upscale_for_display` (smallest image side at least 512 px). Resizing the
window rescales the image together with it while keeping its original
aspect ratio.

Tkinter is optional and is imported lazily only when a preview starts; all
data handling works headlessly without it. When tkinter is missing the
preview functions raise an `ImportError` whose message includes the
required install command:

    pip install tk

(On Debian/Ubuntu, `sudo apt-get install python3-tk` works as an
alternative.) To check availability from code without opening a window:

```python
from ct_mri_dicom_nii_reader.body_data.body_data_imp._tk_gui import tkinter_available

print(tkinter_available())
```

## Visualization

Three visualization modes are provided for `BodyData` volumes.

### 1. Single volume (grayscale)

`BodyData.gui_preview()` opens a grayscale window with a bottom slice
slider — see `show_numpy_3d` above.

### 2. Two volumes (amber/cyan fusion)

`TwoImageFusionViewer` renders two volumes with the regknee-style
amber/cyan edge fusion: the fixed volume contributes a dimmed grayscale
background with amber edges, the moving volume contributes cyan edges, and
where both edges overlap the colors add up to white.

Its preview window has independent `Image 1`, `Image 2`, and `Boundary`
checkboxes, all enabled by default. A single selected image is displayed as
true RGB grayscale, optionally brightened at its boundaries. Two selected
images are independently normalized to 0-255 and shown as an amber/cyan
fusion. With neither image selected the canvas is black, including when only
`Boundary` is enabled. These controls are exclusive to the two-image viewer.

```python
from ct_mri_dicom_nii_reader import TwoImageFusionViewer

viewer = TwoImageFusionViewer(ct_body_data, mri_body_data)
viewer.gui_preview()   # blocking window with a bottom slice slider
```

If the two volumes differ in size or voxel spacing, they are first aligned
at the LPS origin, resampled onto the finest unified spacing (the smallest
`mmpd`), and placed into one common NumPy array whose per-axis lengths are
the maxima of the resampled volumes. Regions covered by no volume hold the
modality-specific air value (-1024 for CT, 0 for MRI and masks). The same
combination is available programmatically:

```python
from ct_mri_dicom_nii_reader import BodyDataCommonGrid, MultimodalFusionComposer

grid = BodyDataCommonGrid([ct_body_data, mri_body_data])
aligned = grid.get_aligned()          # list of BodyData on the common grid
grid.get_unified_mmpd()               # finest spacing
grid.get_common_size()                # (n_L, n_P, n_S) of the common array

composer = MultimodalFusionComposer(ct_body_data, mri_body_data)
rgb = composer.make_slice("s", 100)   # (height, width, 3) uint8 RGB slice
```

### 3. Three volumes (fusion + semi-transparent red mask)

`ThreeImageOverlayViewer` visualizes the first two volumes exactly as
above; the third volume is aligned onto the same common grid and drawn as a
red overlay on top of the fusion. The smallest mask value maps to a fully
transparent overlay and the largest to 50% opacity; the blending happens in
image (RGB) space.

```python
from ct_mri_dicom_nii_reader import ThreeImageOverlayViewer

viewer = ThreeImageOverlayViewer(ct_body_data, mri_body_data, mask_body_data)
viewer.gui_preview()
```

Both viewer windows behave like the grayscale previews: they open at the
default upscaled size, rescale with the window while preserving the aspect
ratio, and raise an `ImportError` with the tkinter installation command
when tkinter is missing.

While the slider is dragged, the viewer renders at a fixed frame rate
instead of debouncing: the slider callback only records the latest slice
index, a single background worker thread computes the NumPy RGB slices
(dropping intermediate indices automatically), and the Tk main thread only
converts the newest result into a `PhotoImage` and redraws the canvas.
Stale worker results are rejected through a generation/index check, and
normalized slices, boundaries, and completed layer combinations are reused
from bounded LRU caches.

### Public display helpers

The formerly private display helpers are now available under public names;
the old names remain as compatibility forwarders for the regknee project:

| Public function | Compatibility forwarder | Description |
| --- | --- | --- |
| `display_window(body_data)` | `_display_window` | Bounded display window for fusion rendering |
| `normalize_to_u8(values, window)` | `_normalize_to_u8` | Window/normalize values to `uint8` |
| `resample_to_mmpd(body_data, mmpd)` | `_resample_to_mmpd` | Resample a volume to a unified voxel spacing |
| `upscale_for_display(image)` | `_upscale_for_display` | Nearest-neighbor upscale (smallest side 512 px) |

```python
from ct_mri_dicom_nii_reader import display_window, normalize_to_u8, resample_to_mmpd
from ct_mri_dicom_nii_reader.body_data.body_data_imp._display_scale import upscale_for_display
```

## Requirements

- Python >= 3.10
- numpy, SimpleITK, Pillow (see `pyproject.toml`)
- tkinter is optional and only needed for the preview GUIs; when it is
  missing the GUI functions report an error with the pip command to
  install it (`pip install tk`)

## License

MIT. The classes were extracted from the `regknee` project
(MIT, Copyright (c) 2026 GGN_2015) and keep their original design; see
`LICENSE`.
