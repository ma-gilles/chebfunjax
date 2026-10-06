"""Focused source-grid controls for the bounded real Singfun plot path.

These tests target public ``matlab_plot`` plus its sampling/axis helpers;
they are source-contract controls, not image-pixel tests.

Chebfun source commit: 7574c77680d7e82b79626300bf255498271a72df.
MATLAB sources: @singfun/plotData.m, @bndfun/plotData.m,
@chebtech/plotData.m, @chebfun/plotData.m, @chebfun/plot.m.
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.plotting import _function_lims, _sample_pieces, matlab_plot
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts

COEFFS = np.array([0.4, 0.2, -0.1, 0.05])


def _singfun_chebfun(exponents, interval=(-2.0, 3.0)):
    smooth = Chebtech2.from_coeffs(jnp.asarray(COEFFS))
    piece = _Piece(tech=Singfun(smooth, exponents), interval=interval)
    return Chebfun(funs=[piece], domain=Domain(interval))


def _source_reference(piece, n_line):
    """Independent coefficient-series oracle for source plotData formulas."""
    smooth = piece.tech.smoothPart
    coefficients = np.asarray(smooth.coeffs)
    kind = 2
    line_nodes = np.asarray(chebpts(n_line, kind=kind))
    point_nodes = np.asarray(chebpts(smooth.n, kind=kind))
    alpha, beta = piece.tech.exponents

    def source_values(nodes):
        values = np.polynomial.chebyshev.chebval(nodes, coefficients)
        if alpha != 0.0:
            values = values * (1.0 + nodes) ** alpha
        if beta != 0.0:
            values = values * (1.0 - nodes) ** beta
        return values

    a, b = piece.interval
    def physical(nodes):
        return b * (nodes + 1.0) / 2.0 + a * (1.0 - nodes) / 2.0

    return (physical(line_nodes), source_values(line_nodes),
            physical(point_nodes), source_values(point_nodes))


def test_public_source_sampling_uses_smooth_grid_factors_and_physical_map():
    f = _singfun_chebfun((0.25, 0.5))
    piece = f.funs[0]
    n_line = max(501, int(np.floor(4.0 * np.pi * piece.tech.smoothPart.n + 0.5)))
    actual = _sample_pieces(f, numpts=16, _source_grid=True)
    assert len(actual) == 1
    x, y = actual[0]
    expected_x, expected_y, _xp, _yp = _source_reference(piece, n_line)
    assert x.shape == (n_line,)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_allclose(y, expected_y, rtol=0.0,
                               atol=100 * np.finfo(float).eps
                               * max(1.0, np.max(np.abs(expected_y))))
    assert y[0] == 0.0 and y[-1] == 0.0


def test_public_marked_plot_uses_scaled_representation_points():
    f = _singfun_chebfun((0.25, 0.5))
    piece = f.funs[0]
    n_line = max(501, int(np.floor(4.0 * np.pi * piece.tech.smoothPart.n + 0.5)))
    expected_x, expected_y, expected_px, expected_py = _source_reference(piece, n_line)
    fig, ax = plt.subplots()
    try:
        matlab_plot(f, "-o", ax=ax)
        assert len(ax.lines) == 2  # source line style and representation marks
        line, points = ax.lines
        np.testing.assert_array_equal(np.asarray(line.get_xdata())[1:], expected_x)
        np.testing.assert_allclose(np.asarray(line.get_ydata())[1:], expected_y,
                                   rtol=0.0,
                                   atol=100 * np.finfo(float).eps
                                   * max(1.0, np.max(np.abs(expected_y))))
        np.testing.assert_array_equal(np.asarray(points.get_xdata())[1:], expected_px)
        np.testing.assert_allclose(np.asarray(points.get_ydata())[1:], expected_py,
                                   rtol=0.0,
                                   atol=100 * np.finfo(float).eps
                                   * max(1.0, np.max(np.abs(expected_py))))
    finally:
        plt.close(fig)


def _source_ylim(y_line, exponents):
    values = np.asarray(y_line)
    values = values[np.isfinite(values)]
    n = values.size
    keep = np.ones(n, dtype=bool)
    for side, exponent in enumerate(exponents):
        if exponent < 0.0:
            count = max(int(np.ceil(min(-0.2 * exponent, 0.5) * n)), 5)
            if side == 0:
                keep[:count] = False
            else:
                keep[n - count:] = False
    selected = values[keep]
    sd = np.std(selected, ddof=1) if selected.size > 1 else 0.0
    return (max(values.min(), selected.min() - sd),
            min(values.max(), selected.max() + sd))


def test_function_limits_use_scaled_source_grid_for_negative_exponents():
    interval = (-2.0, 3.0)
    f = _singfun_chebfun((-0.25, -0.4), interval)
    piece = f.funs[0]
    n_line = max(501, int(np.floor(4.0 * np.pi * piece.tech.smoothPart.n + 0.5)))
    source_x, source_y, _xp, _yp = _source_reference(piece, n_line)
    del source_x
    expected_y_lim = _source_ylim(source_y, piece.tech.exponents)

    x_lim, y_lim, default_y = _function_lims(f)
    np.testing.assert_array_equal(x_lim, interval)  # exact source endpoints
    assert default_y is False
    np.testing.assert_allclose(y_lim, expected_y_lim, rtol=0.0,
                               atol=100 * np.finfo(float).eps
                               * max(1.0, np.max(np.abs(expected_y_lim))))

    fig, ax = plt.subplots()
    try:
        matlab_plot(f, ax=ax)
        np.testing.assert_allclose(ax.get_xlim(), interval, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ax.get_ylim(), expected_y_lim, rtol=0.0,
                                   atol=100 * np.finfo(float).eps
                                   * max(1.0, np.max(np.abs(expected_y_lim))))
    finally:
        plt.close(fig)


_COEFFS = np.asarray([0.4, 0.2, -0.1, 0.05], dtype=np.float64)
_INTERVAL = (-2.0, 3.0)
_EXPS = (-0.25, -0.4)
_ATOL = 100.0 * np.finfo(np.float64).eps


def _singfun():
    smooth = Chebtech2.from_coeffs(jnp.asarray(_COEFFS))
    piece = _Piece(
        tech=Singfun(smooth, _EXPS),
        interval=_INTERVAL,
    )
    return Chebfun(funs=[piece], domain=Domain(_INTERVAL))


def _source_y_lim():
    """Independent Chebyshev-series oracle for source plotData yLim."""
    smooth = Chebtech2.from_coeffs(jnp.asarray(_COEFFS))
    n = int(smooth.coeffs.shape[0])
    n_line = min(max(501, int(np.floor(4.0 * np.pi * n + 0.5))), 65537)
    nodes = np.asarray(chebpts(n_line, kind=2))
    values = np.polynomial.chebyshev.chebval(nodes, np.asarray(smooth.coeffs))
    values = values * (1.0 + nodes) ** _EXPS[0]
    values = values * (1.0 - nodes) ** _EXPS[1]

    finite = values[np.isfinite(values)]
    count = finite.size
    keep = np.ones(count, dtype=bool)
    for side, exponent in enumerate(_EXPS):
        if exponent < 0.0:
            trim = max(int(np.ceil(min(-0.2 * exponent, 0.5) * count)), 5)
            if side == 0:
                keep[:trim] = False
            else:
                keep[count - trim:] = False
    selected = finite[keep]
    sd = np.std(selected, ddof=1)
    return (max(float(finite.min()), float(selected.min() - sd)),
            min(float(finite.max()), float(selected.max() + sd)))


def _expected_union(old_limits, new_limits):
    return (min(old_limits[0], new_limits[0]),
            max(old_limits[1], new_limits[1]))


def test_supplied_empty_axes_with_manual_limits_retains_entry_union():
    fig, ax = plt.subplots()
    try:
        ax.set_xlim((-4.0, -1.0))
        ax.set_ylim((-3.0, 0.0))
        assert not ax.has_data()
        assert not ax.get_autoscalex_on()
        assert not ax.get_autoscaley_on()
        entry_x = tuple(ax.get_xlim())
        entry_y = tuple(ax.get_ylim())

        matlab_plot(_singfun(), ax=ax)

        expected_x = _expected_union(entry_x, _INTERVAL)
        expected_y = _expected_union(entry_y, _source_y_lim())
        np.testing.assert_allclose(ax.get_xlim(), expected_x, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ax.get_ylim(), expected_y, rtol=0.0, atol=_ATOL)
    finally:
        plt.close(fig)


def test_supplied_populated_auto_axes_retains_entry_union():
    fig, ax = plt.subplots()
    try:
        ax.plot(np.asarray([-4.0, 0.0]), np.asarray([-1.0, 0.0]))
        assert ax.has_data()
        assert ax.get_autoscalex_on()
        assert ax.get_autoscaley_on()
        entry_x = tuple(ax.get_xlim())
        entry_y = tuple(ax.get_ylim())

        matlab_plot(_singfun(), ax=ax)

        expected_x = _expected_union(entry_x, _INTERVAL)
        expected_y = _expected_union(entry_y, _source_y_lim())
        np.testing.assert_allclose(ax.get_xlim(), expected_x, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ax.get_ylim(), expected_y, rtol=0.0, atol=_ATOL)
    finally:
        plt.close(fig)
