"""Source-oriented nonempty Spherefun squeeze controls.

Provenance: MATLAB @separableApprox/squeeze.m, @separableApprox/cdr.m,
Chebfun commit 7574c77. Low-degree factors are constructed directly to keep
these contract controls bounded and independent of adaptive sphere fitting.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _constant(value):
    return Trigtech.from_coeffs(jnp.asarray([value]), is_real=True)


def _cosine():
    # Trigtech's even-length centered coefficient layout: cos(pi*x).
    return Trigtech.from_coeffs(jnp.asarray([0.5, 0.0, 0.5]), is_real=True)


def _sine():
    # sin(pi*x), in the same centered coefficient layout.
    return Trigtech.from_coeffs(jnp.asarray([0.5j, 0.0, -0.5j]), is_real=True)


def _rank_one(col, row):
    return Spherefun(cols=[col], rows=[row], pivots=jnp.asarray([1.0]),
                     idx_plus=(0,), idx_minus=())


def _assert_chebfun(g, interval, *, transposed, expected):
    assert isinstance(g, Chebfun)
    a, b = map(float, interval)
    assert float(g.domain.a) == a
    assert float(g.domain.b) == b
    assert g.is_transposed is transposed
    assert len(g.funs) == 1
    assert isinstance(g.funs[0].tech, Trigtech)
    x = jnp.linspace(a, b, 19)
    actual = np.asarray(g(x))
    reference = np.asarray(expected(x)).reshape(actual.shape)
    np.testing.assert_allclose(actual, reference,
                               rtol=2e-11, atol=2e-11)


def test_constant_squeeze_obeys_native_branch_order_and_longitude_domain():
    f = _rank_one(_constant(2.5), _constant(1.0))
    # Native tests length(cols)==1 first, so a constant follows the
    # longitude branch and is returned as a row-oriented Chebfun.
    _assert_chebfun(f.squeeze(), (-np.pi, np.pi), transposed=True,
                    expected=lambda x: jnp.full_like(x, 2.5))
    np.testing.assert_allclose(
        np.asarray(f.squeeze().funs[0].tech.coeffs).reshape(-1), [2.5])


def test_longitude_only_squeeze_returns_row_chebfun_on_native_domain():
    f = _rank_one(_constant(1.0), _cosine())
    g = f.squeeze()
    _assert_chebfun(g, (-np.pi, np.pi), transposed=True, expected=jnp.cos)
    np.testing.assert_allclose(np.asarray(g.funs[0].tech.coeffs).reshape(-1),
                               [0.5, 0.0, 0.5], atol=2e-15, rtol=0)


def test_colatitude_only_squeeze_returns_column_chebfun_on_native_domain():
    f = _rank_one(_cosine(), _constant(1.0))
    # Native source keeps the C factor's doubled representation domain;
    # newdomain is applied only if Chebfun algebra returns a numeric scalar.
    g = f.squeeze()
    _assert_chebfun(g, (-np.pi, np.pi), transposed=False, expected=jnp.cos)
    np.testing.assert_allclose(np.asarray(g.funs[0].tech.coeffs).reshape(-1),
                               [0.5, 0.0, 0.5], atol=2e-15, rtol=0)


def test_two_coordinate_rank_one_and_rank_greater_than_one_are_unchanged():
    rank_one = _rank_one(_sine(), _cosine())
    assert int(rank_one.numerical_rank()) == 1
    assert rank_one.squeeze() is rank_one

    # Independent C and R factors make a two-term diagonal CDR rank two.
    rank_two = Spherefun(
        cols=[_constant(1.0), _cosine()],
        rows=[_constant(1.0), _sine()],
        pivots=jnp.ones(2), idx_plus=(0, 1), idx_minus=())
    assert int(rank_two.numerical_rank()) > 1
    assert rank_two.squeeze() is rank_two


def test_rank_one_with_multiple_constant_cdr_columns_still_squeezes():
    # Stored CDR column count is not MATLAB length(cols): that is the
    # maximum technology resolution. The duplicate row dependence leaves
    # numerical rank one while both CDR columns are constant technologies.
    f = Spherefun(
        cols=[_constant(2.0), _constant(3.0)],
        rows=[_cosine(), Trigtech.from_coeffs(
            jnp.asarray([1.0, 0.0, 1.0]), is_real=True)],
        pivots=jnp.ones(2), idx_plus=(0, 1), idx_minus=())
    assert int(f.numerical_rank()) == 1
    g = f.squeeze()
    _assert_chebfun(g, (-np.pi, np.pi), transposed=True,
                    expected=lambda x: 8.0 * jnp.cos(x))


def test_squeeze_uses_current_global_tolerance_preference():
    # The C factor has a small cosine term. At factory tolerance it remains
    # two-coordinate; a caller-selected tolerance makes it constant before
    # the native branch test, without changing any fitted tolerance.
    c = Trigtech.from_coeffs(jnp.asarray([5e-4, 1.0, 5e-4]), is_real=True)
    f = _rank_one(c, _cosine())
    assert f.squeeze() is f

    old_tol = float(ChebfunPref().techPrefs.chebfuneps)
    try:
        ChebfunPref.setDefaults("chebfuneps", 1e-2)
        g = f.squeeze()
        assert isinstance(g, Chebfun)
        assert g.is_transposed
        assert float(ChebfunPref().techPrefs.chebfuneps) == 1e-2
        x = jnp.linspace(-np.pi, np.pi, 17)
        np.testing.assert_allclose(np.asarray(g(x)).reshape(-1),
                                   np.asarray(jnp.cos(x)),
                                   rtol=0, atol=1.1e-3)
    finally:
        ChebfunPref.setDefaults("chebfuneps", old_tol)
