"""Source-contract controls for MATLAB plotregion radius behavior.

Provenance: MATLAB Chebfun commit 7574c77,
@chebtech/plotregionData.m and @chebfun/plotregion.m.
These controls validate ellipse data and limits for finite scalar Chebtech1/2
pieces; they do not claim Trigtech, Singfun, array-valued, or style parity.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import plotregion
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _source_boundary(n, eps, n_pts):
    """Independent host oracle for MATLAB's displayed ellipse formula."""
    unit_parameter = np.linspace(0.0, 1.0, n_pts)
    c = np.cos(2.0 * np.pi * unit_parameter) + 1j * np.sin(2.0 * np.pi * unit_parameter)
    rho = np.exp(abs(np.log(eps)) / n)
    boundary = 0.5 * (rho * c + 1.0 / (rho * c))
    return boundary, rho


def _assert_piece_ellipse(ax, line_index, n, eps, n_pts, a, b):
    boundary, rho = _source_boundary(n, eps, n_pts)
    midpoint = 0.5 * (a + b)
    halfwidth = 0.5 * (b - a)
    expected_x = midpoint + halfwidth * boundary.real
    expected_y = halfwidth * boundary.imag
    actual_x, actual_y = ax.lines[line_index].get_data()
    scale = max(1.0, abs(a), abs(b), float(np.max(np.abs(expected_x))), float(np.max(np.abs(expected_y))))
    atol = 100.0 * np.finfo(np.float64).eps * scale
    np.testing.assert_allclose(actual_x, expected_x, rtol=0.0, atol=atol)
    np.testing.assert_allclose(actual_y, expected_y, rtol=0.0, atol=atol)
    major = 1.1 * (rho + 1.0 / rho)
    minor = 1.1 * (rho - 1.0 / rho)
    return (midpoint - halfwidth * major, midpoint + halfwidth * major), (-halfwidth * minor, halfwidth * minor)


@pytest.mark.parametrize("tech_class", [Chebtech1, Chebtech2])
def test_plotregion_explicit_user_eps_and_translated_interval(tech_class):
    """User epsilon and physical map match source for either Chebtech kind."""
    tech = tech_class.from_coeffs(jnp.asarray([1.0, 0.25, -0.125, 0.0, 0.0]))
    f = Chebfun(
        funs=[_Piece(tech=tech, interval=(-2.0, 4.0))],
        domain=Domain((-2.0, 4.0)),
    )
    eps = 1.0e-8
    n_pts = 37
    fig, ax = plotregion(f, eps=eps, n_pts=n_pts)
    n = int(f.simplify().funs[0].tech.n)
    assert n == 3  # independent degree-two coefficient fixture
    _assert_piece_ellipse(ax, 0, n, eps, n_pts, -2.0, 4.0)
    plt.close(fig)


def test_plotregion_default_uses_machine_epsilon_simplified_length_and_101_points():
    """Default epsilon radius uses simplified length and public source count."""
    coeffs = jnp.zeros((96,), dtype=jnp.float64).at[:5].set(
        jnp.asarray([1.0, 0.2, 0.05, 0.0125, 0.003125])
    )
    f = Chebfun.from_coeffs(coeffs, domain=(0.25, 1.75))
    simple_n = int(f.simplify().funs[0].tech.n)
    assert simple_n == 5  # the known nonzero coefficients have degree four
    assert simple_n < int(f.funs[0].tech.n)
    fig, ax = plotregion(f)
    assert len(ax.lines[0].get_xdata()) == 101
    _assert_piece_ellipse(ax, 0, simple_n, np.finfo(np.float64).eps, 101, 0.25, 1.75)
    plt.close(fig)


def test_plotregion_piecewise_uses_each_simplified_length_and_unions_limits():
    """Each smooth piece gets its own rho; physical limits are unioned."""
    intervals = ((-3.0, -0.25), (-0.25, 2.0))
    coeffs_by_piece = (
        jnp.asarray([1.0, 0.2, 0.0, 0.0]),
        jnp.asarray([2.0, -0.3, 0.12, 0.04, 0.0, 0.0]),
    )
    pieces = [
        _Piece.from_coeffs(coeffs, a, b)
        for coeffs, (a, b) in zip(coeffs_by_piece, intervals)
    ]
    f = Chebfun(funs=pieces, domain=Domain((-3.0, -0.25, 2.0)))
    eps = 1.0e-6
    n_pts = 23
    simplified = f.simplify()
    lengths = [int(piece.tech.n) for piece in simplified.funs]
    assert lengths == [2, 4]  # independent polynomial degree+1 counts
    fig, ax = plotregion(f, eps=eps, n_pts=n_pts)
    limits = [
        _assert_piece_ellipse(ax, i, n, eps, n_pts, *interval)
        for i, (n, interval) in enumerate(zip(lengths, intervals))
    ]
    expected_xlim = (min(pair[0][0] for pair in limits), max(pair[0][1] for pair in limits))
    expected_ylim = (min(pair[1][0] for pair in limits), max(pair[1][1] for pair in limits))
    scale = max(1.0, *(abs(x) for x in expected_xlim + expected_ylim))
    atol = 100.0 * np.finfo(np.float64).eps * scale
    np.testing.assert_allclose(ax.get_xlim(), expected_xlim, rtol=0.0, atol=atol)
    np.testing.assert_allclose(ax.get_ylim(), expected_ylim, rtol=0.0, atol=atol)
    plt.close(fig)


def test_plotregion_explicit_eps_rejects_nonfinite_or_nonscalar_values():
    f = Chebfun.from_coeffs(jnp.asarray([1.0, 0.25]), domain=(-1.0, 2.0))
    for eps in (0.0, -1.0, np.inf, np.nan, np.asarray([1.0e-8, 1.0e-7])):
        try:
            plotregion(f, eps=eps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for eps={eps!r}")
