"""Source matrix simplification controls; no fitted oracle inputs.

Provenance
----------
MATLAB source : @spherefun/constructor.m, @separableApprox/simplify.m,
    @chebfun/simplify.m, @trigtech/simplify.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun import _bmci
from chebfunjax.tech.trigtech import Trigtech


def test_common_cutoff_retains_small_interior_coefficient():
    short = jnp.zeros(65).at[32].set(1.).at[24].set(1e-18).at[40].set(1e-18)
    long = jnp.zeros(65).at[16].set(.5).at[48].set(.5)
    out = _bmci.simplify_factors([Trigtech(short, is_real=True), Trigtech(long, is_real=True)])
    assert out[0].n == out[1].n >= 33
    mid = out[0].n // 2
    assert float(out[0].coeffs[mid - 8].real) == 1e-18
    assert float(out[0].coeffs[mid + 8].real) == 1e-18


def test_global_tolerance_row_has_no_half_clip(monkeypatch):
    seen = []
    def capture(coefficients, tolerances):
        seen.append(np.asarray(tolerances))
        return coefficients
    monkeypatch.setattr(_bmci, '_simplify_coefficients', capture)
    _bmci.simplify_factors([Trigtech(jnp.asarray([1.]), is_real=True),
                            Trigtech(jnp.asarray([1e-18]), is_real=True)])
    eps = np.finfo(float).eps
    np.testing.assert_allclose(seen[0], [eps, eps / 1e-18], rtol=4*eps, atol=0)
    assert seen[0][1] > 1.


def test_zero_column_shares_retained_nonzero_column_length():
    c = jnp.zeros(33).at[6].set(.5).at[26].set(.5)
    out = _bmci.simplify_factors([Trigtech(c, is_real=True),
                                  Trigtech(jnp.zeros(33), is_real=True)])
    assert out[0].n == out[1].n >= 21
    np.testing.assert_array_equal(out[1].coeffs, jnp.zeros(out[1].n))


def test_scalar_factor_retains_scalar_simplify_contract():
    t = Trigtech(jnp.asarray([.25, 0., 1., 0., .25]), is_real=True)
    expected = t.simplify()
    actual = _bmci.simplify_factors([t])[0]
    np.testing.assert_array_equal(actual.coeffs, expected.coeffs)
    assert actual.is_real == expected.is_real


def test_unhappy_array_does_not_simplify():
    t = Trigtech(jnp.ones(19), is_real=False, ishappy=False)
    assert _bmci.simplify_factors([t])[0] is t


def test_phase_two_full_matrix_is_not_prematurely_chopped():
    n = 64
    x = -1 + 2*jnp.arange(n)/n
    values = jnp.stack([jnp.ones(n), jnp.cos(12*jnp.pi*x)], axis=1)
    factors = _bmci.factors_from_values(values)
    assert [t.n for t in factors] == [n, n]
    assert all(t.is_real and t.ishappy for t in factors)
