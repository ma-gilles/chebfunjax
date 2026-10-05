"""Independent controls for source plotting grids and function line grouping.

Provenance
----------
MATLAB source: @chebtech/plotData.m, @bndfun/plotData.m,
    @chebfun/plotData.m, @chebfun/plot.m
Chebfun commit: 7574c77
The source test_plot assertions check callability. These additional controls
check the stated source data contracts; roundoff bounds are diagnostic IEEE
bounds, not tolerances copied from a MATLAB test.
"""
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import matlab_plot
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps


def _piece(coefficients, interval, kind=2):
    tech = (Chebtech1 if kind == 1 else Chebtech2).from_coeffs(jnp.asarray(coefficients))
    return _Piece(tech=tech, interval=interval)


def _function(pieces):
    breaks = [pieces[0].interval[0], *[p.interval[1] for p in pieces]]
    return Chebfun(funs=pieces, domain=Domain(tuple(breaks)))


def _coefficients(length):
    coefficients = np.zeros(length)
    coefficients[0], coefficients[1], coefficients[-1] = 0.3, 0.2, 0.4
    return coefficients


@pytest.mark.parametrize('kind', [1, 2])
@pytest.mark.parametrize(('length', 'count'), [(3, 501), (41, 515)])
def test_source_grid_and_polynomial_values_on_physical_interval(kind, length, count):
    coefficients = _coefficients(length)
    f = _function([_piece(coefficients, (-2.0, 3.0), kind)])
    fig, ax = matlab_plot(f)
    try:
        assert len(ax.lines) == 1
        x, y = map(np.asarray, ax.lines[0].get_data())
        finite = np.isfinite(x)
        assert finite.sum() == count
        # Independent cosine grid. Evaluating a degree-40 polynomial at
        # rounded cosine nodes amplifies endpoint node error, so evaluate
        # the reference at the exact grid angles in higher precision.
        angles = (np.arange(count) + 0.5) * np.pi / count if kind == 1 else np.arange(count) * np.pi / (count - 1)
        nodes = -np.cos(angles)
        expected_x = 0.5 + 2.5 * nodes
        with mp.workdps(70):
            expected_y = np.array([
                float(mp.fsum(
                    mp.mpf(float(c)) * (-1) ** degree * mp.cos(degree * angle)
                    for degree, c in enumerate(coefficients) if c
                ))
                for angle in (
                    (mp.mpf(j) + mp.mpf('0.5')) * mp.pi / count
                    if kind == 1 else mp.mpf(j) * mp.pi / (count - 1)
                    for j in range(count)
                )
            ])
        np.testing.assert_allclose(x[finite], expected_x, rtol=0, atol=16 * EPS * 5)
        np.testing.assert_allclose(y[finite], expected_y, rtol=0,
                                   atol=64 * EPS * np.sum(np.abs(coefficients)))
    finally:
        plt.close(fig)


@pytest.mark.parametrize('numpts', [17, 100, 2001])
def test_source_deprecated_numpts_does_not_change_bounded_plot_grid(numpts):
    f = _function([_piece(_coefficients(41), (-1.0, 1.0))])
    fig, ax = matlab_plot(f, numpts=numpts)
    try:
        assert np.isfinite(ax.lines[0].get_xdata()).sum() == 515
    finally:
        plt.close(fig)


@pytest.mark.parametrize('value_type', ['real', 'complex', 'array'])
def test_one_line_per_function_column_across_piece_boundaries(value_type):
    left, right = np.array([1.0, 0.5]), np.array([2.0, -0.25])
    if value_type == 'complex':
        left = left * (1 + 2j)
        right = right * (1 - 3j)
    elif value_type == 'array':
        left, right = np.column_stack((left, 2 * left)), np.column_stack((right, 2 * right))
    f = _function([_piece(left, (-2.0, 0.0)), _piece(right, (0.0, 3.0))])
    fig, ax = matlab_plot(f)
    try:
        assert len(ax.lines) == (2 if value_type == 'array' else 1)
        for line in ax.lines:
            x, y = map(np.asarray, line.get_data())
            # Source plotData prefixes each smooth piece with a NaN row.
            assert len(x) == len(y) == 1004
            assert np.isnan(x[0]) and np.isnan(x[502])
            assert np.isfinite(x).sum() == 1002
    finally:
        plt.close(fig)


def test_source_each_piece_has_its_own_degree_based_sampling_count():
    f = _function([_piece(_coefficients(3), (-2.0, 0.0)),
                   _piece(_coefficients(41), (0.0, 3.0))])
    fig, ax = matlab_plot(f)
    try:
        assert len(ax.lines) == 1
        x = np.asarray(ax.lines[0].get_xdata())
        assert len(x) == 1018
        np.testing.assert_array_equal(np.flatnonzero(np.isnan(x)), [0, 502])
        assert x[1] == -2.0 and x[501] == 0.0
        assert x[503] == 0.0 and x[-1] == 3.0
    finally:
        plt.close(fig)


def test_color_cycle_advances_once_per_function_instead_of_once_per_piece():
    first = _function([_piece([1.0, 0.5], (-1.0, 0.0)),
                       _piece([2.0, -0.25], (0.0, 1.0))])
    second = _function([_piece([3.0, 0.2], (-1.0, 1.0))])
    fig, ax = matlab_plot([first, second])
    try:
        assert len(ax.lines) == 2
        assert ax.lines[0].get_color() != ax.lines[1].get_color()
        assert np.isnan(ax.lines[0].get_xdata()).sum() == 2
        assert np.isnan(ax.lines[1].get_xdata()).sum() == 1
    finally:
        plt.close(fig)


def test_source_plot_grid_respects_factory_maxlength_cap():
    # 4*pi*5216 exceeds the source factory maxLength of 2**16+1.
    coefficients = np.zeros(5216)
    coefficients[-1] = 1.0
    f = _function([_piece(coefficients, (-1.0, 1.0))])
    fig, ax = matlab_plot(f)
    try:
        x = np.asarray(ax.lines[0].get_xdata())
        assert np.isfinite(x).sum() == 65537
        assert x[1] == -1.0 and x[-1] == 1.0
    finally:
        plt.close(fig)
