"""Allsix native IMAG predicates; MATLAB RNG stream identity unverified.
NumPyRandomState7681 preserves source counts/ranges/order as Python adapter.
Native4 literally tests real(f), which is retained unchanged.
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


class TestNativeImag:
    def test_empty(self):
        assert cj.chebfun().imag().isempty()

    @pytest.mark.parametrize("array", [False, True])
    def test_finite(self, array):
        scalar = lambda x: jnp.exp(1j * x) * jnp.abs(x - 0.1)
        op = lambda x: jnp.stack([scalar(x), 1j * jnp.exp(-1j * x) * jnp.cos(x)], axis=-1)
        f = cj.chebfun(op if array else scalar, domain=(-1.0, 1.0), pref=pref())
        g = f.imag()
        exact = (
            jnp.stack([jnp.sin(XR) * jnp.abs(XR - 0.1), jnp.cos(XR) ** 2], axis=-1)
            if array
            else jnp.sin(XR) * jnp.abs(XR - 0.1)
        )
        error = float(jnp.max(jnp.abs(g(XR) - exact)))
        bound = (100 if array else 10) * g.vscale * EPS
        print({"native_clause": 3 if array else 2, "error": error, "bound": bound}, flush=True)
        assert g.isreal() and error < bound

    def test_literal_real_zero(self):
        f = cj.chebfun(
            lambda x: 1j * jnp.cos(x) * jnp.abs(x - 0.1), domain=(-1.0, 1.0), pref=pref()
        )
        assert f.real().iszero()

    def test_singular(self):
        dom = (-2.0, 7.0)
        power = -1.64
        f = cj.chebfun(
            lambda x: 1j * (x - dom[0]) ** power, domain=dom, exps=(power, 0), splitting="on"
        )
        assert f.imag().isequal(-1j * f)

    def test_unbounded(self):
        op = lambda x: x * jnp.exp(x) + 1j * (1 - jnp.exp(x)) / x
        f = cj.chebfun(op, domain=(-jnp.inf, -3 * jnp.pi))
        g = f.imag()
        error = float(jnp.max(jnp.abs(g(XU) - (1 - jnp.exp(XU)) / XU)))
        bound = 30 * EPS * f.vscale
        print({"native_clause": 6, "error": error, "bound": bound}, flush=True)
        assert error < bound


class TestSupplementalImag:
    @pytest.mark.parametrize("row", [False, True])
    def test_explicit_point_values_and_orientation(self, row):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sign(x) + 1j, 2 * jnp.sign(x) + 3j], axis=-1),
            domain=(-1.0, 0.0, 1.0),
            pref=pref(),
        )
        f = f.set_point_values(
            jnp.asarray([[-1 + 1j, -2 + 3j], [7 + 2j, 11 + 5j], [1 + 1j, 2 + 3j]])
        )
        if row:
            f = f.transpose()
        g = f.imag()
        expected = np.asarray([[1.0, 3.0], [2.0, 5.0], [1.0, 3.0]])
        if row:
            expected = expected.T
        assert g.is_transposed == row
        np.testing.assert_array_equal(np.asarray(g(jnp.asarray([-1.0, 0.0, 1.0]))), expected)
