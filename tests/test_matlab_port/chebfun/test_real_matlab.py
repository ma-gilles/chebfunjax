"""All6native real predicates, exactpaths/norms; MATLAB RNG identity unverified.
Source tests/chebfun/test_real.m, commit7574c77. NumPyRandomState7681 stream
adapter preserves count/ranges/order, not a native stream equivalence claim.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps
_rng = np.random.RandomState(7681)
XR = jnp.asarray(2 * _rng.rand(1000) - 1)
XU = jnp.asarray((-3 * np.pi + 1e6) * _rng.rand(100) - 1e6)


def pref():
    p = ChebfunPref()
    p.splitting = True
    return p


class TestNativeChebfunReal:
    def test_empty(self):
        assert cj.chebfun().real().isempty()

    @pytest.mark.parametrize("array", [False, True])
    def test_native_finite(self, array):
        scalar = lambda x: jnp.exp(1j * x) * jnp.abs(x - 0.1)
        op = lambda x: jnp.stack([scalar(x), 1j * jnp.exp(-1j * x) * jnp.cos(x)], axis=-1)
        f = cj.chebfun(op if array else scalar, domain=(-1.0, 1.0), pref=pref())
        g = f.real()
        expected = (
            jnp.stack([jnp.cos(XR) * jnp.abs(XR - 0.1), jnp.sin(XR) * jnp.cos(XR)], axis=-1)
            if array
            else jnp.cos(XR) * jnp.abs(XR - 0.1)
        )
        err = float(jnp.max(jnp.abs(g(XR) - expected)))
        bound = (1000 if array else 100) * g.vscale * EPS
        print({"native_clause": 3 if array else 2, "error": err, "source_bound": bound}, flush=True)
        assert g.isreal() and err < bound

    def test_native_pure_imaginary(self):
        f = cj.chebfun(
            lambda x: 1j * jnp.cos(x) * jnp.abs(x - 0.1), domain=(-1.0, 1.0), pref=pref()
        )
        assert f.real().iszero()

    def test_native_singular_identity(self):
        dom = (-2.0, 7.0)
        power = -1.64
        f = cj.chebfun(lambda x: (x - dom[0]) ** power, domain=dom, exps=(power, 0), splitting=True)
        assert f.real().isequal(f)

    def test_native_unbounded(self):
        op = lambda x: x * jnp.exp(x) + 1j * (1 - jnp.exp(x)) / x
        f = cj.chebfun(op, domain=(-jnp.inf, -3 * jnp.pi))
        g = f.real()
        err = float(jnp.max(jnp.abs(g(XU) - XU * jnp.exp(XU))))
        bound = EPS * f.vscale
        print({"native_clause": 6, "error": err, "source_bound": bound}, flush=True)
        assert err < bound


@pytest.mark.parametrize("row", [False, True])
def test_real_native_point_values_and_orientation(row):
    op = lambda x: jnp.stack([jnp.sign(x) + 1j, 2 * jnp.sign(x) + 3j], axis=-1)
    f = cj.chebfun(op, domain=(-1.0, 0.0, 1.0), pref=pref())
    if row:
        f = f.transpose()
    g = f.real()
    x = jnp.asarray([-0.5, 0.0, 0.5])
    expected = jnp.stack([jnp.sign(x), 2 * jnp.sign(x)], axis=-1)
    if row:
        expected = expected.T
    assert g.is_transposed == row
    np.testing.assert_array_equal(np.asarray(g(x)), np.asarray(expected))


XS = jnp.asarray(np.linspace(-0.95, 0.95, 60))


class TestPreviousRealControls:
    def test_real_of_complex_exponential(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * np.pi * x))
        g = f.real()
        err = jnp.abs(g(XS) - jnp.cos(np.pi * XS))
        assert float(jnp.max(err)) < 100 * EPS

    def test_real_of_real_is_identity(self):
        f = cj.chebfun(jnp.sin)
        g = f.real()
        err = jnp.abs(g(XS) - f(XS))
        assert float(jnp.max(err)) < 10 * EPS


@pytest.mark.parametrize("row", [False, True])
def test_real_explicit_native_point_values(row):
    # Native REAL explicitly maps F.pointValues, which can differ from FUN limits.
    op = lambda x: jnp.stack([jnp.sign(x) + 1j, 2 * jnp.sign(x) + 3j], axis=-1)
    f = cj.chebfun(op, domain=(-1.0, 0.0, 1.0), pref=pref())
    f = f.set_point_values(jnp.asarray([[-1 + 1j, -2 + 3j], [7 + 2j, 11 + 5j], [1 + 1j, 2 + 3j]]))
    if row:
        f = f.transpose()
    g = f.real()
    expected = np.asarray([[-1.0, -2.0], [7.0, 11.0], [1.0, 2.0]])
    if row:
        expected = expected.T
    assert g.is_transposed == row
    np.testing.assert_array_equal(np.asarray(g(jnp.asarray([-1.0, 0.0, 1.0]))), expected)
