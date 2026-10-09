"""Independent matrix-artist controls for historical MATLAB spy semantics."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.plotting import spy

matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_native_coordinates_nonfinite_and_axes():
    data = np.array([[0., 3., 0.], [2., 0., np.nan]])
    fig, ax = spy(data)
    line, = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(line.get_ydata(), [2, 1, 2])
    assert ax.get_xlim() == (0., 4.)
    assert ax.get_ylim() == (3., 0.)
    assert ax.get_xlabel() == 'nz = 3'
    assert ax.get_title() == ''
    assert ax.xaxis.get_ticks_position() == 'bottom'
    assert line.get_marker() == '.' and line.get_linestyle() == 'None'
    fig.canvas.draw()
    box = ax.get_window_extent()
    assert box.width / box.height == pytest.approx(4/3)


@pytest.mark.parametrize('dpi', [72, 144])
def test_dot_size_is_physical_points(dpi):
    fig = plt.figure(figsize=(4, 2), dpi=dpi)
    ax = fig.add_axes([.1, .1, .8, .8])
    spy(np.eye(100), ax=ax)
    # Axes height115.2pt, denominator101 ->6.8435 rounds to7.
    assert ax.lines[0].get_markersize() == 7
    np.testing.assert_array_equal(fig.get_size_inches(), [4., 2.])


@pytest.mark.parametrize('order,expected', [(1, 14), (1000, 4)])
def test_dot_size_clamps(order, expected):
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_axes([.1, .1, .8, .8])
    spy(np.zeros((order, order)), ax=ax)
    assert ax.lines[0].get_markersize() == expected
    assert np.isnan(ax.lines[0].get_xdata()).all()


def test_explicit_marker_size_and_style():
    _, ax = spy(np.eye(2), markersize=9, marker='s', color='red', title='A')
    line, = ax.lines
    assert line.get_markersize() == 9 and line.get_marker() == 's'
    assert line.get_color() == 'red' and ax.get_title() == 'A'
    with plt.rc_context({'lines.markersize': 6}):
        _, other = spy(np.eye(2), marker='o')
        assert other.lines[0].get_markersize() == 6


@pytest.mark.parametrize('shape', [(0, 3), (2, 0)])
def test_empty_matrix(shape):
    _, ax = spy(np.empty(shape))
    assert ax.lines[0].get_marker() == 'None'
    assert ax.get_xlabel() == 'nz = 0'
    assert ax.get_xlim() == (0., shape[1]+1.)
    assert ax.get_ylim() == (shape[0]+1., 0.)


def test_explicit_python_display_options():
    _, ax = spy(np.array([[.1, 2.], [0., 0.]]), precision=.5,
                origin='lower', aspect='auto')
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1])
    assert ax.get_ylim() == (0., 3.) and ax.get_aspect() == 'auto'


def test_native_point_glyph_keeps_nominal_size():
    _, ax = spy(np.eye(2), markersize=9)
    line, = ax.lines
    glyph = line._marker.get_path().transformed(line._marker.get_transform())
    assert line.get_markersize() == 9 and line.get_marker() == '.'
    assert glyph.get_extents().width * line.get_markersize() == pytest.approx(3)
    assert line.get_markeredgewidth() == 0
    np.testing.assert_array_equal(ax.get_position(original=True).bounds,
                                  [.13, .11, .775, .815])


def test_axes_first_color_is_independent_of_cycle_position():
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=['magenta', 'cyan'])
    ax.plot([0, 1], [0, 1])
    spy(np.eye(2), ax=ax)
    assert ax.lines[-1].get_color() == 'magenta'
    _, other = plt.subplots()
    other.set_prop_cycle(linestyle=['-', '--'])
    spy(np.eye(2), ax=other)
    assert other.lines[-1].get_color() == 'k'
