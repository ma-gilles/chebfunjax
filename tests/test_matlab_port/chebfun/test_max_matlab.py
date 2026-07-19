"""Port of MATLAB Chebfun tests/chebfun/test_max.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_max.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
Y_EXACT = 1.884217141925336


def _f(x):
    return ((x - 0.2) ** 3 - (x - 0.2) + 1) / jnp.cos(x - 0.2)


class TestChebfunMax:
    def test_empty(self):
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        y, x = chebfun().max()
        assert _np.asarray(y).size == 0 and _np.asarray(x).size == 0

    def test_global_max_reference(self):
        f = cj.chebfun(_f)
        xmax, fmax = f.max()
        assert abs(float(fmax) - Y_EXACT) <= 100 * f.vscale * EPS
        assert abs(float(f(jnp.asarray(float(xmax)))) - Y_EXACT) \
            <= 100 * f.vscale * EPS

    def test_piecewise_same(self):
        f = cj.chebfun(_f, domain=list(np.linspace(-1, 1, 10)))
        xmax, fmax = f.max()
        assert abs(float(fmax) - Y_EXACT) <= 1e3 * f.vscale * EPS

    def test_two_arg_max(self):
        # max(f, g) pointwise = chebfunjax maximum
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = f.maximum(g)
        xs = jnp.asarray(np.linspace(-0.98, 0.98, 100))
        exact = jnp.maximum(jnp.sin(xs), jnp.cos(xs))
        mask = jnp.abs(xs - (-3 * np.pi / 4)) > 1e-6
        err = jnp.abs(h(xs) - exact)[mask]
        assert float(jnp.max(err)) < 1e3 * EPS
