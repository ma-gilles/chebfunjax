"""Source-backed realness and whole-object complex plot controls.

The original MATLAB tree has no tests/chebfun/test_isreal.m.  The expected
realness rules here come directly from @chebfun/isreal.m,
@chebtech/isreal.m, and @deltafun/isreal.m.  Plot routing comes from
@chebfun/plot.m and @chebfun/plotData.m at Chebfun commit 7574c77.

Plot value comparisons below use exact low-degree Chebyshev data and NumPy's
independent polynomial evaluator. Their 128-epsilon bounds are diagnostic
roundoff checks, not a MATLAB test tolerance.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.plotting import matlab_plot
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _cf(coeffs, *, kind=2, deltas=(), transposed=False):
    """Build a low-degree Chebfun directly from source-format coefficients."""
    tech_type = Chebtech1 if kind == 1 else Chebtech2
    tech = tech_type.from_coeffs(jnp.asarray(coeffs))
    domain = Domain((-1.0, 1.0))
    f = Chebfun(
        funs=[_Piece(tech=tech, interval=(-1.0, 1.0))],
        domain=domain,
        deltas=deltas,
    )
    return f.T if transposed else f


def _close_data(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0,
                               atol=128 * np.finfo(float).eps * scale)


def _eval_cheb(coeffs, x):
    """Independent NumPy oracle for a low-degree Chebyshev coefficient row."""
    return np.polynomial.chebyshev.chebval(x, np.asarray(coeffs))


def _line_artists(ax):
    return [artist for artist in ax.lines
            if artist.get_linestyle() not in ("None", "none", "")]


def _assert_source_separator(artist):
    """plotData appends separate NaN rows after converting piece coordinates."""
    x, y = map(np.asarray, artist.get_data())
    assert np.isnan(x[0]) and np.isnan(y[0])


def _marker_artists(ax):
    return [artist for artist in ax.lines
            if artist.get_marker() not in ("None", "none", "")]


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize(
    "coeffs, expected",
    [([0.25, -0.5, 0.125], True),
     ([0.25 + 0.5j, -0.5 + 0.125j, 0.125 - 0.25j], False)],
)
def test_isreal_scalar_chebtech_coefficients(kind, coeffs, expected):
    """MATLAB @chebtech/isreal.m tests coefficient realness for both kinds."""
    assert _cf(coeffs, kind=kind).isreal() is expected


def test_isreal_empty_chebfun_is_true():
    """MATLAB @chebfun/isreal.m reduces the empty column vector with all."""
    assert Chebfun.empty().isreal() is True


@pytest.mark.parametrize(
    "point_values, expected",
    [([2.0, 3.0], True), ([2.0 + 1.0j, 3.0 - 0.5j], False)],
)
def test_isreal_point_value_storage_is_jit_static(point_values, expected):
    """Point-value dtype is static PyTree metadata for source isreal logic."""
    f = _cf([0.5, -0.25]).set_point_values(jnp.asarray(point_values))
    compiled = jax.jit(lambda obj: obj.isreal())
    assert bool(compiled(f)) is expected


@pytest.mark.parametrize(
    "coeffs, expected",
    [([[1.0, 2.0], [0.25, -0.5]], True),
     ([[1.0, 2.0 + 0.5j], [0.25, -0.5 + 0.25j]], False)],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_isreal_array_chebfun_is_column_and_orientation_independent(
        coeffs, expected, transposed):
    """MATLAB @chebfun/isreal.m reduces every column, independent of row flag."""
    f = _cf(coeffs, transposed=transposed)
    assert f.n_columns == 2
    assert f.isreal() is expected


@pytest.mark.parametrize(
    "coeffs, point_values, expected",
    [([1.0, 0.25], [2.0, 3.0], True),
     ([1.0, 0.25], [2.0 + 3.0j, 3.0 - 1.0j], False),
     # MATLAB isreal(complex(A)) is false even when stored imaginary parts
     # are all zero; this assignment is direct pointValues storage.
     ([1.0, 0.25], [2.0 + 0.0j, 3.0 + 0.0j], False),
     ([1.0 + 0.25j, 0.5], [2.0, 3.0], False)],
)
def test_isreal_includes_explicit_breakpoint_values(coeffs, point_values,
                                                     expected):
    """@chebfun/isreal.m checks pointValues before the smooth pieces."""
    f = _cf(coeffs).set_point_values(jnp.asarray(point_values))
    assert f.isreal() is expected


@pytest.mark.parametrize(
    "magnitude, expected",
    [(2.0, True), (2.0 + 0.75j, False)],
)
def test_isreal_includes_distribution_magnitudes(magnitude, expected):
    """Python stores MATLAB Deltafun magnitudes on the enclosing Chebfun."""
    f = _cf([0.5, -0.25], deltas=((0.0, magnitude),))
    assert f.isreal() is expected


@pytest.mark.parametrize("container", ["column_list", "quasimatrix"])
@pytest.mark.parametrize("fmt", ["-", "-o"])
def test_plot_mixed_real_and_complex_columns_use_objectwide_complex_plane(
        container, fmt):
    """@chebfun/plot.m computes ~isreal(f) once before its column loop."""
    real_f = _cf([1.0, 1.0])
    complex_f = _cf([2.0 + 1.0j, 0.5 + 1.0j])
    if container == "column_list":
        grouped = [real_f, complex_f]
    else:
        grouped = Quasimatrix([real_f, complex_f], Domain((-1.0, 1.0)))

    fig, ax = matlab_plot(grouped, fmt, numpts=41)
    try:
        lines = _line_artists(ax)
        for artist in lines:
            _assert_source_separator(artist)
        assert len(lines) == 2
        # The real column is still plotted in the complex plane as (f(x), 0)
        # because the array as a whole is complex. The second column is
        # plotted as (real(g(x)), imag(g(x))).
        x0, y0 = np.asarray(lines[0].get_xdata()), np.asarray(lines[0].get_ydata())
        x1, y1 = np.asarray(lines[1].get_xdata()), np.asarray(lines[1].get_ydata())
        assert np.nanmin(x0) >= -128 * np.finfo(float).eps
        assert np.nanmax(x0) <= 2.0 + 128 * np.finfo(float).eps
        _close_data(y0[np.isfinite(y0)], np.zeros(np.isfinite(y0).sum()))
        assert 1.5 - 128 * np.finfo(float).eps <= np.nanmin(x1)
        assert np.nanmax(x1) <= 2.5 + 128 * np.finfo(float).eps
        assert np.nanmin(y1) >= -128 * np.finfo(float).eps
        assert np.nanmax(y1) <= 2.0 + 128 * np.finfo(float).eps
        finite = np.isfinite(x1) & np.isfinite(y1)
        _close_data(y1[finite], 2.0 * x1[finite] - 3.0)
        if fmt == "-o":
            markers = _marker_artists(ax)
            for artist in markers:
                _assert_source_separator(artist)
            assert len(markers) == 2
            assert markers[0].get_color() == lines[0].get_color()
            assert markers[1].get_color() == lines[1].get_color()
            mx = np.asarray(markers[0].get_xdata())
            my = np.asarray(markers[0].get_ydata())
            assert np.isnan(mx[0]) and np.isnan(my[0])
            _close_data(my[np.isfinite(my)], np.zeros(np.isfinite(my).sum()))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["-", "-o"])
def test_plot_mixed_columns_inside_array_chebfun_are_all_complex_plane(fmt):
    """Source plotData splits array columns only after global isreal routing."""
    # A single matrix-valued Chebfun: first column real, second genuinely
    # complex. This avoids relying on storage dtype alone for the test oracle.
    f = _cf([[1.0, 2.0 + 1.0j], [1.0, 0.5 + 1.0j]])
    fig, ax = matlab_plot(f, fmt, numpts=41)
    try:
        lines = _line_artists(ax)
        for artist in lines:
            _assert_source_separator(artist)
        assert len(lines) == 2
        first_x, first_y = map(np.asarray,
                               (lines[0].get_xdata(), lines[0].get_ydata()))
        second_x, second_y = map(np.asarray,
                                 (lines[1].get_xdata(), lines[1].get_ydata()))
        assert np.nanmin(first_x) >= -128 * np.finfo(float).eps
        assert np.nanmax(first_x) <= 2.0 + 128 * np.finfo(float).eps
        _close_data(first_y[np.isfinite(first_y)],
                    np.zeros(np.isfinite(first_y).sum()))
        assert np.nanmin(second_x) >= 1.5 - 128 * np.finfo(float).eps
        assert np.nanmax(second_x) <= 2.5 + 128 * np.finfo(float).eps
        assert np.nanmin(second_y) >= -128 * np.finfo(float).eps
        assert np.nanmax(second_y) <= 2.0 + 128 * np.finfo(float).eps
        finite = np.isfinite(second_x) & np.isfinite(second_y)
        _close_data(second_y[finite], 2.0 * second_x[finite] - 3.0)
        if fmt == "-o":
            markers = _marker_artists(ax)
            assert len(markers) == 2
            for artist in markers:
                _assert_source_separator(artist)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["-", "-o"])
def test_complex_point_values_route_plot_but_do_not_replace_curve_data(fmt):
    """isreal sees pointValues; plotData continues sampling smooth tech data."""
    f = _cf([1.0, 1.0]).set_point_values(
        jnp.asarray([2.0 + 3.0j, 3.0 - 1.0j]))
    assert f.isreal() is False
    fig, ax = matlab_plot(f, fmt, numpts=41)
    try:
        lines = _line_artists(ax)
        for artist in lines:
            _assert_source_separator(artist)
        assert len(lines) == 1
        xline = np.asarray(lines[0].get_xdata())
        yline = np.asarray(lines[0].get_ydata())
        assert np.nanmin(xline) >= -128 * np.finfo(float).eps
        assert np.nanmax(xline) <= 2.0 + 128 * np.finfo(float).eps
        _close_data(yline[np.isfinite(yline)],
                    np.zeros(np.isfinite(yline).sum()))
        if fmt == "-o":
            markers = _marker_artists(ax)
            for artist in markers:
                _assert_source_separator(artist)
            assert len(markers) == 1
            # For a two-coefficient Chebtech2 the representation points are
            # the endpoints, whose underlying smooth values are 0 and 2.
            mx = np.asarray(markers[0].get_xdata())
            my = np.asarray(markers[0].get_ydata())
            assert np.isnan(mx[0]) and np.isnan(my[0])
            finite = np.isfinite(mx) & np.isfinite(my)
            _close_data(mx[finite], _eval_cheb([1.0, 1.0], [-1.0, 1.0]))
            _close_data(my[finite], np.zeros(finite.sum()))
    finally:
        plt.close(fig)
