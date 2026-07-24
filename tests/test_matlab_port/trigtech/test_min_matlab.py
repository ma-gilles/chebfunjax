"""Port of MATLAB Chebfun tests/trigtech/test_min.m (Opus 4.8[1m]).

min(f) returns the minimum value and its location on [-1, 1].

Provenance
----------
MATLAB source : tests/trigtech/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _spotcheck(fn, exact_min):
    f = _tt(fn)
    y, x = f.min()
    fx = jnp.asarray(fn(jnp.atleast_1d(x))).ravel()[0]
    vs = f.vscale
    return (abs(float(np.real(y)) - exact_min) < 100 * vs * EPS
            and abs(complex(np.asarray(fx)) - exact_min) < 10 * vs * EPS)


class TestTrigtechMin:
    def test_exp_neg_cos(self):
        assert _spotcheck(lambda x: -jnp.exp(-jnp.cos(2 * jnp.pi * x)),
                          -np.exp(1))

    def test_sin10(self):
        assert _spotcheck(lambda x: -jnp.sin(10 * jnp.pi * x), -1)

    def test_exp_sin100(self):
        assert _spotcheck(lambda x: -jnp.exp(jnp.sin(jnp.pi * x) ** 100),
                          -np.exp(1))

    def test_exp_neg_sin100(self):
        assert _spotcheck(lambda x: -jnp.exp(-jnp.sin(jnp.pi * x) ** 100), -1)

    def test_sign_approx(self):
        def fn(x):
            return -4 / jnp.pi * (
                jnp.sin(jnp.pi * x) + 1 / 3 * jnp.sin(3 * jnp.pi * x)
                + 1 / 5 * jnp.sin(5 * jnp.pi * x)
                + 1 / 7 * jnp.sin(7 * jnp.pi * x)
                + 1 / 9 * jnp.sin(9 * jnp.pi * x))
        assert _spotcheck(fn, -1.182328208857607)

    def test_array_valued(self):
        def fun(x):
            return -jnp.stack([
                jnp.exp(-jnp.cos(2 * jnp.pi * x)),
                jnp.sin(10 * jnp.pi * x),
                jnp.exp(-jnp.sin(jnp.pi * (x - 0.32)) ** 100)], axis=-1)
        f = _tt(fun)
        y, x = f.min()
        exact = -jnp.array([np.exp(1), 1.0, 1.0])
        fx = jnp.array([
            jnp.asarray(fun(jnp.atleast_1d(x[k]))).ravel()[k]
            for k in range(3)])
        assert float(jnp.max(jnp.abs(y - exact))) < 100 * EPS
        assert float(jnp.max(jnp.abs(fx - exact))) < 10 * EPS

    def test_complex_valued(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x) + jnp.exp(1j * jnp.pi * x))
        y, x = f.min()
        yv = complex(np.asarray(y))
        vs = f.vscale
        assert (abs(yv - 1j) < 1e2 * vs * EPS) or (abs(yv + 1j) < 1e2 * vs * EPS)
