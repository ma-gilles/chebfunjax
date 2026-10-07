"""Source CDR zero-pivot semantics through actual Spherefun evaluation routes.

Provenance
----------
MATLAB source : @separableApprox/cdr.m, @separableApprox/feval.m,
    @spherefun/sample.m, tests/spherefun/test_plus.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
New analytic controls supplement unchanged literal plus clauses and bounds.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._cdr import inverse_pivots
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _field(pivots):
    one = Trigtech.from_coeffs(jnp.asarray([1.0]), is_real=True)
    return Spherefun(cols=[one] * len(pivots), rows=[one] * len(pivots),
                     pivots=jnp.asarray(pivots), idx_plus=tuple(range(len(pivots))),
                     idx_minus=())


@pytest.mark.parametrize('disable', [False, True])
def test_inverse_inf_only_preserves_nan(disable):
    p = jnp.asarray([0.0, -0.0, jnp.inf, -jnp.inf, 2.0, -4.0, jnp.nan])
    with jax.disable_jit(disable):
        d = jax.jit(inverse_pivots)(p)
    np.testing.assert_array_equal(d[:6], [0.0, 0.0, 0.0, -0.0, .5, -.25])
    assert bool(jnp.isnan(d[-1]))


@pytest.mark.parametrize('disable', [False, True])
def test_zero_pivot_all_evaluation_shapes(disable):
    # One discarded zero-pivot term plus a nonzero independent constant term.
    f = _field([0.0, 2.0])
    original_pivots = f.pivots
    x = jnp.asarray([-.7, .2, .8])
    t = jnp.asarray([.3, .8, 1.2])
    with jax.disable_jit(disable):
        results = [f(.2, .8), f(x, t), f(x, .8), f(.2, t),
                   f.fevalm(x, t), f(*jnp.meshgrid(x, t, indexing='ij')),
                   f._eval_impl(x, t), jax.jit(lambda a, b: f(a, b))(x, t),
                   jax.vmap(lambda a, b: f(a, b))(x, t)]
        for result in results:
            np.testing.assert_array_equal(result, jnp.full(jnp.shape(result), .5))
        assert float(jax.grad(lambda a: f(a, .8))(.2)) == 0.0
        _, d, _ = f.cdr()
        np.testing.assert_array_equal(jnp.diag(d), [0.0, .5])
        np.testing.assert_array_equal(f.coeffs2(), [[.5]])
        u, d, v = f.sample_cdr(3, 3)
        np.testing.assert_array_equal(u @ d @ v.T, jnp.full((3, 3), .5))
    np.testing.assert_array_equal(f.pivots, original_pivots)
    assert len(f.cols) == 2  # No metadata/rank pruning to hide invalid division.


@pytest.mark.parametrize('disable', [False, True])
def test_public_exact_cancellation_evaluates_zero(disable):
    f = _field([1.0])
    with jax.disable_jit(disable):
        h = f - f
        np.testing.assert_array_equal(h(jnp.asarray([-.2, .7]), .4), [0.0, 0.0])
        assert float(h.norm(jnp.inf)) == 0.0
    # Original literal clause (g-11*f).norm(inf) bound remains unchanged.


def test_complex_finite_pivots_not_narrowed():
    p = jnp.asarray([1 + 2j, -2 + 1j])
    np.testing.assert_allclose(inverse_pivots(p), np.asarray([1 / (1 + 2j), 1 / (-2 + 1j)]),
                               rtol=0, atol=2e-16)
