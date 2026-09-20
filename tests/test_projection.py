import numpy as np
import pytest

from ct_mri_dicom_nii_reader.body_data import main as body_data_main
from ct_mri_dicom_nii_reader.body_data.body_data_imp import (
    projection_comparison,
)
from ct_mri_dicom_nii_reader import (
    BodyData,
    BodyDataNotInitialized,
    BodyDataSlice,
    get_lps_max_projections,
    max_intensity_projection,
)


def _volume() -> np.ndarray:
    l, p, s = np.indices((2, 3, 4))
    return (100 * l + 10 * p + s).astype(np.int16)


def test_get_lps_max_projections_returns_ps_sl_lp_axis_order():
    volume = _volume()

    ps, sl, lp = get_lps_max_projections(volume)

    assert ps.shape == (3, 4)
    assert sl.shape == (4, 2)
    assert lp.shape == (2, 3)
    np.testing.assert_array_equal(ps, np.max(volume, axis=0))
    np.testing.assert_array_equal(sl, np.max(volume, axis=1).T)
    np.testing.assert_array_equal(lp, np.max(volume, axis=2))


@pytest.mark.parametrize("plane", ["ps", "PS", "sl", "SL", "lp", "LP"])
def test_max_intensity_projection_supports_each_plane(plane):
    expected = {
        "ps": np.max(_volume(), axis=0),
        "sl": np.max(_volume(), axis=1).T,
        "lp": np.max(_volume(), axis=2),
    }

    projection = max_intensity_projection(_volume(), plane)

    np.testing.assert_array_equal(projection, expected[plane.lower()])


def test_projection_functions_reject_invalid_inputs():
    with pytest.raises(ValueError, match="three-dimensional"):
        get_lps_max_projections(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="must not be empty"):
        get_lps_max_projections(np.zeros((2, 0, 3)))
    with pytest.raises(ValueError, match="plane must be"):
        max_intensity_projection(_volume(), "ls")
    with pytest.raises(TypeError, match="plane must be"):
        max_intensity_projection(_volume(), 0)


def test_body_data_get_project_slices_wraps_projection_metadata():
    body_data = BodyData()
    volume = _volume()
    body_data.from_array(volume, "ct", 0.75)

    projections = body_data.getProjectSlices()

    assert isinstance(projections, tuple)
    assert len(projections) == 3
    assert all(isinstance(item, BodyDataSlice) for item in projections)
    assert tuple(item.get_size() for item in projections) == (
        (3, 4),
        (4, 2),
        (2, 3),
    )
    assert all(item.get_mmpd() == 0.75 for item in projections)
    assert all(item.get_body_data() is body_data for item in projections)

    for actual, expected in zip(
        projections, get_lps_max_projections(volume)
    ):
        np.testing.assert_array_equal(actual.to_numpy(), expected)


def test_body_data_get_project_slices_requires_initialization():
    with pytest.raises(BodyDataNotInitialized):
        BodyData().getProjectSlices()


def test_projection_comparison_prepares_six_images_with_xy_mapping():
    lhs = get_lps_max_projections(_volume())
    rhs = tuple(item.astype(np.float32) * 10.0 for item in lhs)

    lhs_images, rhs_images = projection_comparison._prepare_projection_images(
        lhs, rhs, (0.0, 123.0), (0.0, 1230.0)
    )

    expected_sizes = ((3, 4), (4, 2), (2, 3))
    assert tuple(image.size for image in lhs_images) == expected_sizes
    assert tuple(image.size for image in rhs_images) == expected_sizes
    for lhs_image, rhs_image in zip(lhs_images, rhs_images):
        np.testing.assert_array_equal(np.asarray(lhs_image), np.asarray(rhs_image))


def test_projection_comparison_uses_one_scale_for_all_images():
    images = (
        projection_comparison.Image.new("L", (100, 50)),
        projection_comparison.Image.new("L", (50, 100)),
        projection_comparison.Image.new("L", (80, 40)),
    )

    scaled, scale = projection_comparison._scale_projection_images(
        images, ((200, 200), (200, 200), (200, 200))
    )

    assert scale == pytest.approx(2.0)
    assert tuple(image.size for image in scaled) == (
        (200, 100),
        (100, 200),
        (160, 80),
    )
    for source, displayed in zip(images, scaled):
        assert displayed.width / source.width == pytest.approx(scale)
        assert displayed.height / source.height == pytest.approx(scale)


def test_projection_comparison_builds_labeled_two_by_three_grid(monkeypatch):
    state = {"labels": [], "canvas_count": 0, "mainloop_calls": 0}

    class FakeWidget:
        def pack(self, **_kwargs):
            return None

        def grid(self, **_kwargs):
            return None

        def grid_columnconfigure(self, *_args, **_kwargs):
            return None

        def grid_rowconfigure(self, *_args, **_kwargs):
            return None

        def bind(self, *_args, **_kwargs):
            return None

    class FakeRoot(FakeWidget):
        def title(self, value):
            state["title"] = value

        def geometry(self, value):
            state["geometry"] = value

        def minsize(self, width, height):
            state["minsize"] = (width, height)

        def after(self, _delay, _callback):
            return "redraw-job"

        def after_cancel(self, _job):
            return None

        def mainloop(self):
            state["mainloop_calls"] += 1

    class FakeTk:
        BOTH = "both"
        CENTER = "center"
        TclError = RuntimeError

        def Tk(self):
            return FakeRoot()

        def Frame(self, _parent):
            return FakeWidget()

        def Label(self, _parent, **kwargs):
            state["labels"].append(kwargs["text"])
            return FakeWidget()

        def Canvas(self, _parent, **_kwargs):
            state["canvas_count"] += 1
            return FakeWidget()

    monkeypatch.setattr(
        projection_comparison, "require_tkinter", lambda: FakeTk()
    )
    projections = get_lps_max_projections(_volume())

    projection_comparison.show_projection_comparison(
        projections, projections, (0.0, 123.0), (0.0, 123.0)
    )

    assert state["title"] == "BodyData Projection Comparison"
    assert state["labels"] == ["PS", "SL", "LP", "Image1", "Image2"]
    assert state["canvas_count"] == 6
    assert state["mainloop_calls"] == 1


def test_body_data_gui_compare_forwards_both_projection_rows(monkeypatch):
    received = {}

    def fake_show(lhs, rhs, lhs_window, rhs_window):
        received["lhs"] = lhs
        received["rhs"] = rhs
        received["lhs_window"] = lhs_window
        received["rhs_window"] = rhs_window

    monkeypatch.setattr(
        body_data_main, "show_projection_comparison", fake_show
    )
    lhs = BodyData()
    rhs = BodyData()
    lhs_volume = _volume().astype(np.float32)
    rhs_volume = lhs_volume * 10.0
    lhs.from_array(lhs_volume, "ct", 1.0)
    rhs.from_array(rhs_volume, "mri", 1.0)

    result = lhs.guiCompare(rhs)

    assert result is None
    assert tuple(item.shape for item in received["lhs"]) == (
        (3, 4),
        (4, 2),
        (2, 3),
    )
    assert tuple(item.shape for item in received["rhs"]) == (
        (3, 4),
        (4, 2),
        (2, 3),
    )
    assert received["lhs_window"] == (
        lhs.percentile(1),
        lhs.percentile(99),
    )
    assert received["rhs_window"] == (
        rhs.percentile(1),
        rhs.percentile(99),
    )


def test_body_data_gui_compare_resamples_both_to_larger_mmpd(monkeypatch):
    from ct_mri_dicom_nii_reader import visualization

    resample_calls = []
    received = {}
    real_resample = visualization.resample_to_mmpd

    def recording_resample(body_data, mmpd):
        resample_calls.append((body_data, mmpd))
        return real_resample(body_data, mmpd)

    def fake_show(lhs, rhs, lhs_window, rhs_window):
        received["lhs"] = lhs
        received["rhs"] = rhs
        received["lhs_window"] = lhs_window
        received["rhs_window"] = rhs_window

    monkeypatch.setattr(
        visualization, "resample_to_mmpd", recording_resample
    )
    monkeypatch.setattr(
        body_data_main, "show_projection_comparison", fake_show
    )
    lhs = BodyData()
    rhs = BodyData()
    lhs_volume = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    rhs_volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    lhs.from_array(lhs_volume, "ct", 0.5)
    rhs.from_array(rhs_volume, "ct", 1.0)

    lhs.guiCompare(rhs)

    assert resample_calls == [(lhs, 1.0), (rhs, 1.0)]
    assert lhs.get_mmpd() == 0.5
    assert rhs.get_mmpd() == 1.0
    assert tuple(item.shape for item in received["lhs"]) == (
        (4, 5),
        (5, 3),
        (3, 4),
    )
    assert tuple(item.shape for item in received["rhs"]) == (
        (4, 5),
        (5, 3),
        (3, 4),
    )


def test_body_data_gui_compare_validates_rhs():
    body_data = BodyData()
    body_data.from_array(_volume(), "ct", 1.0)

    with pytest.raises(TypeError, match="rhs must be a BodyData"):
        body_data.guiCompare(object())
    with pytest.raises(BodyDataNotInitialized):
        body_data.guiCompare(BodyData())
