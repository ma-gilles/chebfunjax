"""Analytic controls for exact-zero root suppression and bounded absolute value.

These controls are independent of source40's loose inherited absolute bound.
The nonzero functions have amplitude eps/8 and must retain their own relative
accuracy, roots, orientation, and continuous extrema.

Provenance
----------
MATLAB source : @chebfun/roots.m, @chebtech/roots.m, @chebfun/abs.m,
    @chebfun/getRootsForBreaks.m, @chebfun/addBreaks.m, @chebtech/abs.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(np.float64).eps
ALPHA = EPS / 8


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
def test_nozerofun_retains_tiny_nonzero_linear_root(Tech):
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(
        ALPHA * jnp.asarray([-0.25, 1.0])), interval=(-1.0, 1.0))],
        domain=Domain((-1.0, 1.0)))
    np.testing.assert_allclose(f.roots(nozerofun=True), [0.25],
                               rtol=0, atol=100 * EPS)
    magnitude = f.abs()
    assert any(abs(float(v) - 0.25) < 100 * EPS
               for v in magnitude.domain.breakpoints)
    assert all(isinstance(p.tech, Tech) for p in magnitude.funs)
    np.testing.assert_allclose(float(magnitude.norm(jnp.inf)), 1.25 * ALPHA,
                               rtol=100 * EPS, atol=0)
    root = min(magnitude.domain.breakpoints, key=lambda v: abs(float(v) - 0.25))
    assert float(magnitude(jnp.asarray(root))) == 0.0


def test_nozerofun_suppresses_only_exact_zero_array_column():
    f = Chebfun.from_coeffs(jnp.asarray([[0.0, -ALPHA / 4],
                                           [0.0, ALPHA]]))
    roots = np.asarray(f.roots(nozerofun=True))
    assert roots.shape == (1, 2)
    assert np.isnan(roots[0, 0])
    np.testing.assert_allclose(roots[0, 1], 0.25, rtol=0, atol=100 * EPS)


def test_exact_zero_absolute_value_does_not_add_breakpoint():
    f = Chebfun.from_coeffs(jnp.zeros(2))
    assert f.roots(nozerofun=True).size == 0
    np.testing.assert_array_equal(f.roots(), [0.0])
    assert f.abs().domain == f.domain
    assert float(f.abs().norm(jnp.inf)) == 0.0


@pytest.mark.parametrize('transposed', [False, True])
def test_tiny_array_norm_retains_off_grid_continuous_maximum(transposed):
    # q(x)=1-(x-1/4)^2, max |q|=1 at x=1/4, a root at -3/4.
    # [q,-q,0] has continuous matrix infinity norm 2, regardless of alpha.
    q = ALPHA * jnp.asarray([0.4375, 0.5, -0.5])
    f = Chebfun.from_coeffs(jnp.stack([q, -q, jnp.zeros_like(q)], axis=1))
    if transposed:
        f = f.T
    np.testing.assert_allclose(float(f.norm(jnp.inf)), 2 * ALPHA,
                               rtol=100 * EPS, atol=0)


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
def test_bounded_absolute_value_preserves_physical_map_and_orientation(Tech):
    # Canonical t in [-1,1] maps to physical x=4.5*t+2.5 on [-2,7].
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(jnp.asarray([-0.25, 1.0])),
                            interval=(-2.0, 7.0))],
                domain=Domain((-2.0, 7.0))).T
    magnitude = f.abs()
    assert magnitude.is_transposed
    points = jnp.asarray([-2.0, -0.125, 3.625, 5.0, 7.0])
    expected = jnp.abs((points - 2.5) / 4.5 - 0.25)
    np.testing.assert_allclose(magnitude(points), expected,
                               rtol=100 * EPS, atol=100 * EPS)
