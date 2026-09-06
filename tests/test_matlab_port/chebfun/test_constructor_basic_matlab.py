"""Port of MATLAB Chebfun tests/chebfun/test_constructor_basic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_basic.m
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
    jnp.sin,
    lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1),
    lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(-x)], axis=-1),
]


def _err(f, F, xx):
    xx = jnp.asarray(xx)
    return float(np.max(np.abs(np.asarray(f(xx)) - np.asarray(F(xx)))))


def _hscale(f):
    bp = list(f.domain.breakpoints)
    return max(abs(bp[0]), abs(bp[-1]))


class TestChebfunConstructorBasic:
    def test_all_matlab_assertions(self):
        cheps = ChebfunPref().chebfuneps
        for F in FF:
            f = chebfun(F, domain=(-1.0, 1.0))
            err = _err(f, F, np.linspace(-1, 1, 100))
            assert err < 10 * EPS * f.vscale
            assert err < 50 * cheps

            f = chebfun(F)
            err = _err(f, F, np.linspace(-1, 1, 100))
            assert err < 10 * EPS * f.vscale
            assert err < 500 * cheps

            f = chebfun(F, domain=(0.0, 10000.0))
            err = _err(f, F, np.linspace(0, 10000, 100))
            assert err < 1e4 * EPS * f.vscale
            assert err < 1e2 * _hscale(f) * cheps

            f = chebfun(F, domain=(-1.0, 0.0, 0.5, np.sqrt(np.pi / 4), 1.0))
            err = _err(f, F, np.linspace(-1, 1, 100))
            assert err < 10 * EPS * f.vscale
            assert err < 100 * cheps
