"""Independent finite controls for complex powers of endpoint Singfuns.

These are focused API regressions, not a replacement for source power.m
pass 33. No MATLAB RNG stream is claimed.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m pass 33, @chebfun/power.m,
    @singfun/power.m, @singfun/restrict.m, @chebfun/imag.m,
    @singfun/imag.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(np.float64).eps
DOMAIN = (-2.0, 7.0)
EXPONENTS = (-0.3, -0.2)


def _case(*, point_values=None, transposed=False):
    # Exact smooth factor s(t)=-1+i*t, represented by Chebyshev T0,T1.
    smooth = Chebtech2.from_coeffs(jnp.asarray([-1.0 + 0.0j, 1.0j]))
    singular = Singfun(smooth, EXPONENTS)
    f = Chebfun(
        funs=[_Piece(tech=singular, interval=DOMAIN)],
        domain=Domain(DOMAIN),
    )
    if transposed:
        f = f.T
    if point_values is not None:
        f = f.set_point_values(jnp.asarray(point_values, dtype=jnp.complex128))
    return f


def _reference(x):
    x = np.asarray(x, dtype=np.float64)
    t = (x - 2.5) / 4.5
    return (-1.0 + 1.0j * t) * (1.0 + t) ** EXPONENTS[0] \
        * (1.0 - t) ** EXPONENTS[1]


@pytest.mark.parametrize("transposed", [False, True])
def test_finite_complex_singfun_power_splits_and_scales_endpoint_exponents(transposed):
    f = _case(transposed=transposed)
    b = 0.5
    g = f ** b
    assert g.is_transposed == transposed

    # Restriction at the interior imaginary root leaves one singular endpoint
    # on each outside panel; @singfun/power multiplies those exponents by b.
    assert len(g.funs) == 2
    assert isinstance(g.funs[0].tech, Singfun)
    assert isinstance(g.funs[1].tech, Singfun)
    np.testing.assert_allclose(g.funs[0].tech.exponents, (-0.15, 0.0),
                               rtol=0, atol=20 * EPS)
    np.testing.assert_allclose(g.funs[1].tech.exponents, (0.0, -0.1),
                               rtol=0, atol=20 * EPS)

    # Compare the full principal value away from endpoints, including both
    # sides of the negative-real-axis crossing. This formula does not assume
    # that a numerical root is exactly the algebraic point x=2.5.
    x = jnp.asarray([-1.75, -0.1, 0.1, 2.5 - 1e-4, 2.5 + 1e-4,
                     4.0, 6.8])
    expected = np.sqrt(_reference(x))
    np.testing.assert_allclose(np.asarray(g(x)), np.asarray(expected),
                               rtol=100 * EPS, atol=100 * EPS)

    # Probe on both sides of the actual returned breakpoint. The one-sided
    # limits are checked analytically; no assertion is made about the stored
    # principal value exactly at the branch point after restriction.
    root = min((float(q) for q in g.domain.breakpoints),
               key=lambda q: abs(q - 2.5))
    np.testing.assert_allclose(g(root, side="left"), -1j,
                               rtol=0, atol=100 * EPS)
    np.testing.assert_allclose(g(root, side="right"), 1j,
                               rtol=0, atol=100 * EPS)
    delta = 1e-4
    probes = jnp.asarray([root - delta, root + delta])
    np.testing.assert_allclose(
        np.asarray(g(probes)),
        np.asarray(np.sqrt(_reference(probes))),
        rtol=100 * EPS, atol=100 * EPS,
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_imag_preserves_singfun_representation_values_and_row_orientation(transposed):
    # MATLAB @chebfun/imag.m maps both pointValues and every FUN; the
    # Singfun smooth part is mapped by @singfun/imag.m while exponents remain.
    f = _case(point_values=[2.0 + 3.0j, 4.0 - 5.0j], transposed=transposed)
    imag_f = f.imag()
    assert imag_f.is_transposed == transposed
    assert len(imag_f.funs) == 1
    assert isinstance(imag_f.funs[0].tech, Singfun)
    assert imag_f.funs[0].tech.exponents == EXPONENTS
    np.testing.assert_array_equal(np.asarray(imag_f.point_values), [3.0, -5.0])

    x = jnp.asarray([-1.5, 0.0, 2.5, 5.0, 6.5])
    t = (np.asarray(x) - 2.5) / 4.5
    expected = t * (1.0 + t) ** EXPONENTS[0] * (1.0 - t) ** EXPONENTS[1]
    np.testing.assert_allclose(np.asarray(imag_f(x)), np.asarray(expected),
                               rtol=100 * EPS, atol=100 * EPS)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_complex_power_casts_smooth_endpoint_zero_to_fractional_singfun(Tech):
    # Exact polynomial (1+t)*(-1+i*t), in the Chebyshev basis.
    coefficients = jnp.asarray([-1.0 + 0.5j, -1.0 + 1j, 0.5j])
    f = Chebfun(funs=[_Piece(Tech.from_coeffs(coefficients), DOMAIN)],
                domain=Domain(DOMAIN))
    g = f**0.5
    assert g.ishappy
    assert isinstance(g.funs[0].tech, Singfun)
    np.testing.assert_allclose(g.funs[0].tech.exponents, (0.5, 0.0),
                               rtol=0.0, atol=20 * EPS)
    t = np.asarray([-0.99, -0.7, -0.1, 0.1, 0.6, 0.99])
    x = jnp.asarray(2.5 + 4.5 * t)
    expected = np.sqrt((1.0 + t) * (-1.0 + 1j * t))
    np.testing.assert_allclose(np.asarray(g(x)), expected,
                               rtol=100 * EPS, atol=100 * EPS)
