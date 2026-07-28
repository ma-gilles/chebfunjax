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

    def test_local_maxima(self):
        # MATLAB test_max.m pass(7): local maxima (interior only here).
        f = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x ** 2),
                       domain=[0, 4])
        x, y = f.max("local")
        x = np.asarray(x)
        y = np.asarray(y)
        y_exact = np.array([1.923771282655145, 1.117294907913736,
                            1.343997479566445])
        x_exact = np.array([1.323339426259694, 2.781195946808315,
                            3.776766383330969])
        assert len(y) == 3
        assert np.max(np.abs(y - y_exact)) < 10 * f.vscale * EPS
        assert np.max(np.abs(x - x_exact)) < 1e-10

    def test_local_maxima_array_nan_padded(self):
        # MATLAB test_max.m array-valued local maxima, NaN-padded.
        op = lambda t: jnp.sin(t) ** 2 + jnp.sin(t ** 2)  # noqa: E731
        f = cj.chebfun(lambda x: jnp.stack([op(x), op(x / 2)], axis=-1),
                       domain=[0, 4])
        x, y = f.max("local")
        y = np.asarray(y)
        assert y.shape[1] == 2
        assert abs(y[0, 0] - 1.923771282655145) < 1e-10
        assert abs(y[0, 1] - 1.923771282655145) < 1e-10
        assert np.all(np.isnan(y[1:, 1]))
