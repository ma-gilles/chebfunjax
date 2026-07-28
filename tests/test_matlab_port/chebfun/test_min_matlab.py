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

    def test_local_minima(self):
        # MATLAB test_min.m pass(7): local minima incl. endpoints.
        f = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x ** 2),
                       domain=[0, 4])
        x, y = f.min("local")
        x = np.asarray(x)
        y = np.asarray(y)
        y_exact = np.array([0.0, -0.342247088203205,
                            -0.971179645473729, 0.284846700239241])
        x_exact = np.array([0.0, 2.220599667639221,
                            3.308480466603983, 4.0])
        assert len(y) == 4
        assert np.max(np.abs(y - y_exact)) < 10 * f.vscale * EPS
        assert np.max(np.abs(x - x_exact)) < 1e-10

    def test_local_minima_array_nan_padded(self):
        # MATLAB test_min.m pass(9)/(10): array-valued local minima padded.
        op = lambda t: jnp.sin(t) ** 2 + jnp.sin(t ** 2)  # noqa: E731
        f = cj.chebfun(lambda x: jnp.stack([op(x), op(x / 2)], axis=-1),
                       domain=[0, 4])
        x, y = f.min("local")
        y = np.asarray(y)
        assert y.shape == (4, 2)
        assert np.max(np.abs(y[:, 0] - np.array([0.0, -0.342247088203205,
                                                 -0.971179645473729,
                                                 0.284846700239241]))) < 1e-10
        assert np.max(np.abs(y[:2, 1] - np.array([0.0,
                                                  0.070019315123878]))) < 1e-10
        assert np.all(np.isnan(y[2:, 1]))
