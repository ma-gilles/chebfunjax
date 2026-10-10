"""Source-derived inset policy: current renderer text, no screenshot fitting."""
# uses-numpy: fixed graphics fixtures, unchanged geometry and bbox assertions.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from chebfunjax.plotting import (
    chebfun_style,
    matlab_axes_layout,
    matlab_view,
    plot_sphere,
    save_chebfun_figure,
)


def save_and_capture(fig, path):
    records = []
    original = fig.draw

    def observe(renderer):
        original(renderer)
        ax = fig.axes[0]
        records.append([artist.get_window_extent(renderer).frozen()
                        for artist in (ax.title, ax.xaxis.label, ax.yaxis.label)
                        if artist.get_text() and artist.get_visible()])
    fig.draw = observe
    try:
        save_chebfun_figure(fig, path, size=(600, 270), layout="matlab")
    finally:
        fig.draw = original
    assert Image.open(path).size == (600, 270)
    assert records
    # Pixel-coordinate cancellation at zero: the helper predeclares 64eps
    # in normalized geometry. Apply the same derived scale symmetrically.
    width, height = 600.000001, 270.000001
    allowance = 64*np.finfo(float).eps*max(width, height, 1.)
    for boxes in records:
        for box in boxes:
            assert box.x0 >= -allowance and box.y0 >= -allowance
            assert box.x1 <= width+allowance and box.y1 <= height+allowance


@pytest.mark.parametrize("label", [r"Longitude, $\lambda$", r"Co-latitude, $\theta$"])
def test_plain_labels_fit_without_data_or_style_change(tmp_path, label):
    chebfun_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    try:
        x = np.linspace(-3, 3, 31)
        (line,) = ax.plot(x, 20+10*np.cos(x))
        ax.set_xlabel(label)
        ax.set_ylabel("Temperature (Celsius)")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([10, 20, 30])
        before = (ax.get_xlim(), ax.get_ylim(), ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator(), ax.xaxis.label.get_fontsize(), fig.dpi)
        data = np.array(line.get_ydata(), copy=True)
        save_and_capture(fig, tmp_path/"line.png")
        assert before == (ax.get_xlim(), ax.get_ylim(), ax.xaxis.get_major_locator(), ax.yaxis.get_major_locator(), ax.xaxis.label.get_fontsize(), fig.dpi)
        np.testing.assert_array_equal(line.get_ydata(), data)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("title", ["Original dataset", r"Smoothed Temp., $\sigma$=20 degrees"])
def test_sphere_title_fits_without_camera_or_geometry_change(tmp_path, title):
    chebfun_style()
    fig, ax = plot_sphere(lambda l, t: np.cos(np.asarray(t)), n_pts=17, cmap="jet")
    try:
        ax.set_axis_off()
        matlab_view(ax, 50., 0.)
        ax.set_title(title)
        projection = ax.get_proj().copy()
        box_aspect = ax.get_box_aspect().copy()
        geometry = _surface_geometry(ax.collections[0]).copy()
        before = (ax.elev, ax.azim, ax.get_xlim(), ax.get_ylim(), ax.get_zlim(), ax.title.get_fontsize(), fig.dpi)
        save_and_capture(fig, tmp_path/"sphere.png")
        np.testing.assert_array_equal(ax.get_proj(), projection)
        np.testing.assert_array_equal(ax.get_box_aspect(), box_aspect)
        np.testing.assert_array_equal(_surface_geometry(ax.collections[0]), geometry)
        assert before == (ax.elev, ax.azim, ax.get_xlim(), ax.get_ylim(), ax.get_zlim(), ax.title.get_fontsize(), fig.dpi)
    finally:
        plt.close(fig)


def test_no_decorations_matches_native_loose_inset_position():
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    try:
        ax.set_axis_off()
        result = matlab_axes_layout(ax)
        np.testing.assert_allclose(result["position"], [.13, .11, .775, .815], rtol=0, atol=8*np.finfo(float).eps)
        assert result["effective_inset"] == (.13, .11, .095, .075)
        assert result["units"] == "normalized figure"
    finally:
        plt.close(fig)


def test_large_label_expands_only_required_inset():
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    try:
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("Two lines\nTemperature", fontsize=22)
        result = matlab_axes_layout(ax)
        assert result["effective_inset"][1] > .11
        assert ax.xaxis.label.get_fontsize() == 22
        assert result["outer_position"] == (0., 0., 1., 1.)
    finally:
        plt.close(fig)


def test_late_annotation_recomputes_on_second_save(tmp_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    try:
        ax.plot([0, 1], [0, 1])
        save_and_capture(fig, tmp_path/"first.png")
        first = ax.get_position(original=True).bounds
        ax.set_xlabel("Late label\nSecond line", fontsize=18)
        save_and_capture(fig, tmp_path/"second.png")
        assert ax.get_position(original=True).y0 > first[1]
    finally:
        plt.close(fig)


def test_nondefault_outer_rectangle_keeps_figure_normalized_insets():
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    try:
        ax.set_axis_off()
        result = matlab_axes_layout(ax, outer_position=(.1, .1, .8, .8))
        # Insets are parentfigure units: never multiplied by .8.
        np.testing.assert_allclose(result["position"], [.23, .21, .575, .615], rtol=0, atol=8*np.finfo(float).eps)
    finally:
        plt.close(fig)


def test_infeasible_and_unsupported_layouts_raise_without_fitting(tmp_path):
    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    try:
        ax.set_xlabel("huge\nlabel", fontsize=300)
        initial = ax.get_position(original=True).bounds
        with pytest.raises(ValueError, match="infeasible|did not converge"):
            matlab_axes_layout(ax)
        assert ax.xaxis.label.get_fontsize() == 300
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, initial)
        image = ax.imshow([[0, 1], [1, 0]])
        colorbar = fig.colorbar(image, ax=ax)
        with pytest.raises(ValueError, match="one axes"):
            save_chebfun_figure(fig, tmp_path/"unsupported.png", layout="matlab")
        assert not (tmp_path/"unsupported.png").exists()
        ax.remove()  # A sole colorbar axes is also outside the contract.
        assert fig.axes == [colorbar.ax]
        with pytest.raises(ValueError, match="one axes"):
            matlab_axes_layout(colorbar.ax)
    finally:
        plt.close(fig)


def _surface_geometry(artist):
    # Gouraud artist retains source vertices; legacy polygons retain _vec.
    geometry = artist._source_vertices if hasattr(artist, "_source_vertices") else artist._vec
    assert geometry.size > 0
    return geometry
