"""Direct coefficient probes for narrow-panel piecewise linearization.

Provenance
----------
MATLAB source : @chebop/linearize.m (seeded coefficient differentiation)
Chebfun commit: 7574c77

MATLAB obtains differential coefficients through ADChebfun coefficient AD.
The Python piecewise collocation adapter applies the operator to the same
``x**k/k!`` basis polynomials; these tests check the direct Chebyshev
representation independently of the production recurrence.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, jump
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop, _physical_monomial_cheb_coeffs

TINY_INTERVAL = (0.2, 0.2000004)
EPS = np.finfo(np.float64).eps


def _extract_coefficients(op, orders, intervals, nn=8):
    """Exercise the production extraction path on prescribed physical panels."""
    from chebfunjax.chebfun1d.chebfun import _Piece

    m = len(orders)
    bps = [intervals[0][0], *(b for _a, b in intervals)]
    dom = Domain(tuple(bps))
    P = len(intervals)
    Pn = P * nn
    tref = np.cos(np.pi * np.arange(nn) / (nn - 1))[::-1]
    xps = [a + (b - a) * (tref + 1.0) / 2.0 for a, b in intervals]
    cop = Chebop(op, tuple(bps))
    x_fun = Chebfun.identity(dom)

    def to_funs(values):
        us = []
        for var in range(m):
            pieces = [
                _Piece.from_values(
                    jnp.asarray(values[var * Pn + p * nn:
                                       var * Pn + (p + 1) * nn]), a, b)
                for p, (a, b) in enumerate(intervals)
            ]
            us.append(Chebfun(funs=pieces, domain=dom))
        return us

    def apply_op(us):
        result = cop.op(x_fun, *us)
        return result if isinstance(result, (list, tuple)) else [result]

    cop._piecewise_linear_matrix(
        m, P, nn, Pn, bps, intervals, xps, orders, to_funs, apply_op,
        lambda _bc, _us, _x: [], lambda _u: None, [], [], [],
        np.zeros(m * Pn),
    )
    return cop._pw_lin_cache["c_funs"]


def test_physical_monomial_coefficients_match_independent_cheb_oracle():
    a, b = TINY_INTERVAL
    center = 0.5 * (a + b)
    halfwidth = 0.5 * (b - a)
    t = np.linspace(-1.0, 1.0, 17)

    for degree in range(5):
        got = np.asarray(_physical_monomial_cheb_coeffs(degree, a, b))
        # Independent NumPy polynomial multiplication in Chebyshev basis.
        expected = np.polynomial.chebyshev.chebpow(
            np.array([center, halfwidth]), degree
        ) / math.factorial(degree)
        npt.assert_allclose(got, expected, rtol=20 * EPS, atol=0)
        x = center + halfwidth * t
        npt.assert_allclose(
            np.polynomial.chebyshev.chebval(t, got),
            x**degree / math.factorial(degree),
            rtol=20 * EPS,
            atol=0,
        )


def test_direct_probe_piece_derivatives_on_tiny_translated_interval():
    a, b = TINY_INTERVAL
    x = np.linspace(a, b, 9)
    for degree in range(4):
        coeffs = _physical_monomial_cheb_coeffs(degree, a, b)
        piece = _Piece.from_coeffs(coeffs, a, b)
        for derivative_order in range(degree + 1):
            derivative = (piece if derivative_order == 0
                          else piece.diff(derivative_order))
            expected = x ** (degree - derivative_order) / math.factorial(
                degree - derivative_order
            )
            got = np.asarray(derivative(jnp.asarray(x)))
            bound = 100 * EPS * np.maximum(1.0, np.abs(expected))
            assert np.all(np.abs(got - expected) <= bound)


def test_piecewise_extraction_recovers_boundary_layer_operator_coefficients():
    """The extracted c0/c1/c2 remain correct on a 4e-7 physical panel."""
    a, b = TINY_INTERVAL
    c_funs = _extract_coefficients(
        lambda x, u: 1e-4 * u.diff(2) + u.diff() + 0.5 * u,
        [2], [TINY_INTERVAL],
    )
    c0, c1, c2 = c_funs[0][0]
    probes = np.linspace(a, b, 7)
    for got, expected in ((c0(probes), 0.5), (c1(probes), 1.0),
                          (c2(probes), 1e-4)):
        bound = 100 * EPS * max(1.0, abs(expected))
        assert np.all(np.abs(np.asarray(got) - expected) <= bound)


def test_piecewise_extraction_keeps_mixed_system_probe_fields_separate():
    """Mixed-order probes separate variables over both narrow panels."""
    a, b = TINY_INTERVAL
    mid = 0.5 * (a + b)

    def operator(x, u, v):
        return [u.diff(2) + 2.0 * u + 4.0 * v,
                v.diff() + 3.0 * v]

    intervals = [(a, mid), (mid, b)]
    coeffs = _extract_coefficients(operator, [2, 1], intervals)
    expected = {
        (0, 0): (2.0, 0.0, 1.0),
        (0, 1): (4.0, 0.0),
        (1, 0): (0.0, 0.0, 0.0),
        (1, 1): (3.0, 1.0),
    }
    for (eq, var), values in expected.items():
        assert len(coeffs[eq][var]) == len(values)
        for got, value in zip(coeffs[eq][var], values):
            bound = 100 * EPS * max(1.0, abs(value))
            for a_panel, b_panel in intervals:
                x = np.linspace(a_panel, b_panel, 5)[1:-1]
                assert np.all(np.abs(np.asarray(got(x)) - value) <= bound)


def test_piecewise_cosine_bvp_source_tolerance_and_continuity():
    """Primary piecewise chebcolloc2 control from MATLAB pass(5)/pass(8).

    MATLAB source : tests/chebop/test_linearScalarODEs.m, pass(5) checks
        residual/endpoints and pass(8) checks the solution value jump.
    The source's factory bvpTol is 5e-13 and both assertion groups use
    ``1000*bvpTol``; this test keeps that unchanged source bound.

    The derivative jump assertion is an additional consistency check.
    """
    bvp_tol = 5e-13  # cheboppref bvpTol used by MATLAB source fixture
    source_bound = 1000 * bvp_tol
    domain = (-1.0, 0.0, float(np.pi))
    x = Chebfun.identity(Domain(domain))
    problem = Chebop(lambda xx, u: u.diff(2) + xx.cos() * u, domain)
    problem.lbc = 2.0
    problem.rbc = -1.0
    rhs = x.sin()
    u = problem.solve(rhs, discretization="chebcolloc2")

    residual = u.diff(2) + x.cos() * u - rhs
    assert float(residual.norm(2)) < source_bound
    assert abs(float(u(jnp.asarray(domain[0]))) - 2.0) < source_bound
    assert abs(float(u(jnp.asarray(domain[-1]))) + 1.0) < source_bound
    assert abs(float(jump(u, 0.0))) < source_bound
    assert abs(float(jump(u.diff(), 0.0))) < source_bound
