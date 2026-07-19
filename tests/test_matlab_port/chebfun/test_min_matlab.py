"""Port of MATLAB Chebfun tests/chebfun/test_min.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
Y_EXACT = 0.710869767377087


def _f(x):
    return ((x - 0.2) ** 3 - (x - 0.2) + 1) / jnp.cos(x - 0.2)


class TestChebfunMin:
    def test_empty(self):
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        y, x = chebfun().min()
        assert _np.asarray(y).size == 0 and _np.asarray(x).size == 0

    def test_global_min_reference(self):
        f = cj.chebfun(_f)
        xmin, fmin = f.min()
        assert abs(float(fmin) - Y_EXACT) <= 100 * f.vscale * EPS
        assert abs(float(f(jnp.asarray(float(xmin)))) - Y_EXACT) \
            <= 100 * f.vscale * EPS

    def test_two_arg_min(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = f.minimum(g)
        xs = jnp.asarray(np.linspace(-0.98, 0.98, 100))
        exact = jnp.minimum(jnp.sin(xs), jnp.cos(xs))
        err = jnp.abs(h(xs) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS
