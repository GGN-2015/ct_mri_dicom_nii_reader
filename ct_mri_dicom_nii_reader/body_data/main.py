# 数据类是可复用的类
# 在未来的项目，我们可能也有机会直接服用这些数据类
# 因此数据类本身一定不要依赖非数据类
# 数据类的依赖也要尽可能保证跨平台可用

from abc import ABC, abstractmethod
import os
import numpy
from typing import Optional, Tuple, List
from pathlib import Path

from .body_data_imp.dicom_to_hu_lps import load_dicom_hu_lps
from .body_data_imp.mask_xz_rectangle import fill_outside_xz_rectangle
from .body_data_imp.nifti_to_lps import load_nifti_lps
from .body_data_imp.mind_3d import compute_mind_image
from .body_data_imp.volume_percentile import get_volume_percentile
from .body_data_imp.slice_display import show_numpy_gray
from .body_data_imp.numpy_3d_viewer import show_numpy_3d

class DataLoaderNotMatch(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class BodyDataNotInitialized(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class BodyDataTypeError(Exception):
    def __init__(self, image_type, *args: object) -> None:
        super().__init__(*args)
        self._image_type = image_type
    def get_image_type(self):
        return self._image_type

class NoAvailableDataLoader(Exception):
    def __init__(self, filepath:str, *args: object) -> None:
        super().__init__(*args)
        self._filepath = filepath
    def get_filepath(self) -> str:
        return self._filepath

class RoiRect:
    def __init__(self, xmin:int, xmax:int, ymin:int, ymax:int) -> None:
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
    def get_xmin(self) -> int:
        return self._xmin
    def get_ymin(self) -> int:
        return self._ymin
    def get_xmax(self) -> int:
        return self._xmax
    def get_ymax(self) -> int:
        return self._ymax

# 抽象的数据加载器类
# 需要子类实现 check_match 与 load_data
class BodyDataLoader(ABC):
    def __init__(self) -> None:
        ABC.__init__(self)
        self._mmpd = 1 # 默认 mmpd 为 1mm

    # mmpd: mm per dot 意思是每个体素的毫米宽度
    # 我们要求数据加载时，直接采样到各向同性的 LPS 坐标系
    def get_mmpd(self) -> float:
        return self._mmpd

    def set_mmpd(self, mmpd:float) -> None:
        self._mmpd = mmpd

    # 在使用加载器类时，使用者应该使用 load_file
    # 而不是使用 load_data，从而增加安全性
    def load_file(self, filepath:str) -> 'BodyData':
        if not self.check_match(filepath):
            raise DataLoaderNotMatch()
        body_data = self.load_data(filepath)
        EPS = 1e-6
        if (-EPS <= body_data.percentile(0) and 
            body_data.percentile(100) <= 1.0 + EPS and
            body_data.get_type() == "mask"):
            body_data.unify_to_mask()
        return body_data

    # 根据文件类型判断是否当前加载器有能力加载
    @abstractmethod
    def check_match(self, filepath:str) -> bool:
        ...

    # load_file 调用 load_data 中的业务代码
    # load_data 负责加载文件的功能实现
    @abstractmethod
    def load_data(self, filepath:str) -> 'BodyData':
        ...

class DicomBodyDataLoader(BodyDataLoader):
    def __init__(self) -> None:
        BodyDataLoader.__init__(self)

    def check_match(self, filepath:str) -> bool:
        return filepath.lower().endswith(".dcm")

    # 这里我们一般假设 filepath 是一个 .dcm 文件
    # 而且这个 .dcm 同文件夹中的所有 .dcm 文件只有一个 DICOM 序列
    # 如果有多个 DICOM 序列我们就有限取体数据元素个数最多的
    def load_data(self, filepath:str) -> 'BodyData':
        arr3d, metadata = load_dicom_hu_lps(
            os.path.dirname(filepath),
            self.get_mmpd(),
            require_ct=False,
            return_metadata=True)
        modality = str(metadata["modality"]).lower()
        if modality in ["ct", "dx", "cr", "cbct"]:
            image_type = "ct"
        elif modality in ["mr"]:
            image_type = "mri"
        else:
            image_type = "mask"
        body_data = BodyData()
        body_data.from_array(arr3d, image_type, self.get_mmpd())
        return body_data

class NiiBodyDataLoader(BodyDataLoader):
    def __init__(self) -> None:
        BodyDataLoader.__init__(self)

    def check_match(self, filepath:str) -> bool:
        return (
            filepath.lower().endswith(".nii") or
            filepath.lower().endswith(".nii.gz"))
    
    def load_data(self, filepath:str) -> 'BodyData':
        arr3d, metadata = load_nifti_lps(
            filepath,
            self.get_mmpd()
        )
        image_type = str(metadata["modality"]).lower()
        if image_type == "mr":
            image_type = "mri"
        elif image_type == "unknown":
            image_type = "mask"
        new_body_data = BodyData()
        new_body_data.from_array(arr3d, image_type, self.get_mmpd())
        return new_body_data

class UnifiedBodyDataLoader(BodyDataLoader):
    def __init__(self) -> None:
        BodyDataLoader.__init__(self)

    def check_match(self, filepath:str) -> bool:
        return filepath.lower().endswith(".ubd.npz")

    def load_data(self, filepath: str) -> "BodyData":
        with numpy.load(filepath, allow_pickle=False) as archive:
            required_keys = {
                "body_data",
                "image_type",
                "mmpd",
            }
            missing_keys = required_keys - set(archive.files)
            if missing_keys:
                raise ValueError(
                    f"Invalid body data file; missing fields: "
                    f"{sorted(missing_keys)}"
                )
            body_array = archive["body_data"]
            image_type = str(archive["image_type"].item())
            mmpd = float(archive["mmpd"].item())
        new_body_data = BodyData()
        new_body_data.from_array(body_array, image_type, mmpd)
        return new_body_data

class BodyDataSlice:
    def __init__(self) -> None:
        self._slice_data = numpy.zeros((1, 1))
        self._body_data = None
        self._image_type = None
        self._mmpd = 1.0

    def set_body_data(self, bd:Optional['BodyData']) -> None:
        self._body_data = bd

    def get_body_data(self) -> Optional['BodyData']:
        return self._body_data

    def set_mmpd(self, mmpd:float) -> None:
        self._mmpd = mmpd

    def get_mmpd(self) -> float:
        return self._mmpd

    def from_array(self, data:numpy.ndarray, image_type:str) -> None:
        self._image_type = image_type
        self._slice_data = data

    def to_numpy(self, copy:bool=False) -> numpy.ndarray:
        """Return the underlying NumPy array of this slice.

        Args:
            copy: When true, return a copy instead of the internal array.

        Returns:
            The internal 2-D array, or a copy of it when ``copy=True``.
        """
        if copy:
            return self._slice_data.copy()
        return self._slice_data

    def get_pos(self, x:int, y:int) -> float:
        return float(self._slice_data[x, y])

    def get_size(self) -> Tuple[int, int]:
        return (
            int(self._slice_data.shape[0]),
            int(self._slice_data.shape[1])
        )

    def clone(self) -> 'BodyDataSlice':
        new_slice = BodyDataSlice()
        if self._image_type is None:
            return new_slice
        new_slice._slice_data = self._slice_data.copy()
        new_slice._image_type = self._image_type
        new_slice._mmpd = self._mmpd
        new_slice._body_data = self._body_data
        return new_slice

    def gui_preview(self) -> None:
        vmin = None
        vmax = None
        if self._body_data is not None:
            vmin = self._body_data.percentile(1)
            vmax = self._body_data.percentile(99)
        show_numpy_gray(
            self._slice_data,
            vmin, vmax)

class BodyData:
    def __init__(self) -> None:
        self._body_data = numpy.zeros((1, 1, 1))
        self._mmpd = 1
        self._image_type = None # 未初始化

    def from_array(
        self, 
        data:numpy.ndarray, 
        image_type:str, 
        mmpd:float
    ) -> None:
        self._body_data = data
        self._image_type = image_type
        self._mmpd = mmpd

    def to_numpy(self, copy:bool=False) -> numpy.ndarray:
        """Return the underlying NumPy array of this volume.

        Args:
            copy: When true, return a copy instead of the internal array.

        Returns:
            The internal 3-D array, or a copy of it when ``copy=True``.
        """
        if copy:
            return self._body_data.copy()
        return self._body_data

    def unify_to_mask(self) -> None:
        s1 = (self._body_data >= 0.5)
        s0 = (self._body_data <  0.5)
        new_arr3d = numpy.zeros(
            self._body_data.shape, dtype=numpy.int8)
        new_arr3d[s1] = 1
        new_arr3d[s0] = 0
        self._body_data = new_arr3d

    def get_initialized(self) -> bool:
        return self._image_type is not None

    def get_type(self) -> Optional[str]:
        return self._image_type

    def save(self, filepath: str) -> None:
        if not self.get_initialized():
            raise BodyDataNotInitialized()
        path = Path(filepath)
        if not filepath.lower().endswith(".ubd.npz"):
            path = Path(filepath + ".ubd.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(
            path,
            format_version=numpy.array(1, dtype=numpy.int32),
            body_data=self._body_data,
            image_type=numpy.array(self._image_type, dtype=numpy.str_),
            mmpd=numpy.array(self._mmpd, dtype=numpy.float64),
        )

    def get_mmpd(self) -> float:
        return self._mmpd

    def set_mmpd(self, mmpd:float) -> None:
        self._mmpd = mmpd

    def get_size(self) -> Tuple[int, int, int]:
        return (
            int(self._body_data.shape[0]),
            int(self._body_data.shape[1]),
            int(self._body_data.shape[2]))

    def get_pos(self, l:int, p:int, s:int) -> float:
        return float(self._body_data[l, p, s])

    def _make_body_data_slice_from_slice(self, slice_value) -> 'BodyDataSlice':
        if not self.get_initialized():
            raise BodyDataNotInitialized()
        assert self._image_type is not None
        body_slice = BodyDataSlice()
        body_slice.from_array(
            self._body_data[slice_value], self._image_type
        )
        body_slice.set_mmpd(self.get_mmpd())
        body_slice.set_body_data(self)
        return body_slice

    def get_slice_l(self, l:int) -> BodyDataSlice:
        return self._make_body_data_slice_from_slice(
            numpy.s_[l, :, :]
        )

    def get_slice_p(self, p:int) -> BodyDataSlice:
        return self._make_body_data_slice_from_slice(
            numpy.s_[:, p, :]
        )

    def get_slice_s(self, s:int) -> BodyDataSlice:
        return self._make_body_data_slice_from_slice(
            numpy.s_[:, :, s]
        )

    def clone(self) -> 'BodyData':
        new_body_data = BodyData()
        if not self.get_initialized():
            return new_body_data
        assert self._image_type is not None
        new_body_data.from_array(
            self._body_data.copy(),
            self._image_type,
            self._mmpd
        )
        return new_body_data

    def percentile(self, idx:int) -> float:
        if not self.get_initialized():
            raise BodyDataNotInitialized()
        return get_volume_percentile(self._body_data, idx)

    def _get_air_value(self) -> float:
        if self._image_type is None:
            raise BodyDataNotInitialized()
        match self._image_type:
            case "ct":
                return -1024
            case "mri":
                return 0
            case "mask":
                return 0
            case _:
                raise BodyDataTypeError(self._image_type)

    def get_part(self, roi_rect:'RoiRect') -> 'BodyData':
        new_body_data = BodyData()
        if not self.get_initialized():
            return new_body_data
        arr3d = fill_outside_xz_rectangle(
            self._body_data,
            roi_rect.get_xmin(),
            roi_rect.get_xmax(),
            roi_rect.get_ymin(),
            roi_rect.get_ymax(),
            air_value=self._get_air_value()
        )
        assert self._image_type is not None
        new_body_data.from_array(
            arr3d, self._image_type, self._mmpd)
        return new_body_data

    def get_mind(self) -> 'BodyData':
        if not self.get_initialized():
            raise BodyDataNotInitialized()
        mind_data = compute_mind_image(self._body_data)
        new_body_data = BodyData()
        assert self._image_type is not None
        new_body_data.from_array(
            mind_data, self._image_type, self.get_mmpd()
        )
        return new_body_data

    def gui_preview(self) -> None:
        show_numpy_3d(
            self._body_data, self.percentile(1), self.percentile(99))

class BodyDataLoaderManager:
    def __init__(self) -> None:
        self._loader_list:List[BodyDataLoader] = [
            DicomBodyDataLoader(),
            NiiBodyDataLoader(),
            UnifiedBodyDataLoader()
        ]

    def load_file(self, filepath:str, mmpd:Optional[float]=None) -> BodyData:
        if mmpd is None:
            mmpd = 1.0
        for loader in self._loader_list:
            if loader.check_match(filepath):
                loader.set_mmpd(mmpd)
                return loader.load_file(filepath)
        raise NoAvailableDataLoader(filepath)
