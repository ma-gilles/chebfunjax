"""Port of MATLAB Chebfun tests/chebfun/test_constructor_basic_periodic.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_basic_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps
FF = [
    lambda x: jnp.exp(jnp.sin(np.pi * x)),
    lambda x: jnp.exp(jnp.stack([jnp.sin(np.pi * x), jnp.cos(np.pi * x)], axis=-1)),
    lambda x: jnp.exp(jnp.stack([jnp.sin(np.pi * x), jnp.cos(np.pi * x),
                                 -jnp.cos(np.pi * x) ** 2], axis=-1)),
]


def _err(f, F, xx):
    xx = jnp.asarray(xx)
    return float(np.max(np.abs(np.asarray(f(xx)) - np.asarray(F(xx)))))


def _n(f):
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


def _hscale(f):
    bp = list(f.domain.breakpoints)
    return max(abs(bp[0]), abs(bp[-1]))


class TestChebfunConstructorBasicPeriodic:
    def test_all_matlab_assertions(self):
        cheps = ChebfunPref().chebfuneps
        for F in FF:
            f = chebfun(F, domain=(-1.0, 1.0), periodic=True)
            g = chebfun(F, domain=(-1.0, 1.0), trig=True)
            err = _err(f, F, np.linspace(-1, 1, 100))
            assert err < 10 * EPS * f.vscale and _n(f - g) < cheps
            assert err < 50 * cheps

            f = chebfun(F, periodic=True)
            err = _err(f, F, np.linspace(-1, 1, 100))
            assert err < 10 * EPS * f.vscale
            assert err < 500 * cheps

            f = chebfun(F, domain=(-100.0, 100.0), periodic=True)
            g = chebfun(F, domain=(-100.0, 100.0), trig=True)
            err = _err(f, F, np.linspace(-100, 100, 100))
            assert err < 1e3 * EPS * f.vscale
            assert err < 100 * _hscale(f) * cheps and _n(f - g) < 100 * cheps
