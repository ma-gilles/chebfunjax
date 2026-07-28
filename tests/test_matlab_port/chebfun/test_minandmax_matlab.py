"""Port of MATLAB Chebfun tests/chebfun/test_minandmax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
Y_EXACT = np.array([0.710869767377087, 1.884217141925336])


def _f(x):
    return ((x - 0.2) ** 3 - (x - 0.2) + 1) / jnp.cos(x - 0.2)


class TestChebfunMinandmax:
    def test_empty(self):
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        (ymn, xmn), (ymx, xmx) = chebfun().minandmax()
        assert all(_np.asarray(z).size == 0 for z in (ymn, xmn, ymx, xmx))

    def test_smooth_reference_values(self):
        f = cj.chebfun(_f)
        (xmin, fmin), (xmax, fmax) = f.minandmax()
        y = np.array([float(fmin), float(fmax)])
        assert float(np.max(np.abs(y - Y_EXACT))) <= 100 * f.vscale * EPS
        fx = np.asarray(f(jnp.asarray(np.array([float(xmin),
                                                float(xmax)]))))
        assert float(np.max(np.abs(fx - Y_EXACT))) \
            <= 100 * f.vscale * EPS

    def test_piecewise_same_function(self):
        f = cj.chebfun(_f, domain=list(np.linspace(-1, 1, 10)))
        (xmin, fmin), (xmax, fmax) = f.minandmax()
        y = np.array([float(fmin), float(fmax)])
        assert float(np.max(np.abs(y - Y_EXACT))) <= 1e3 * f.vscale * EPS

    # FIXED (Fable 5): minandmax on complex chebfuns now follows
    # MATLAB's |f| ordering (tech-level |f|^2 path + magnitude
    # aggregation across pieces).
    def test_complex_piecewise(self):
        # pass(4): f = {exp((1+1i)x) on [-1,0], 1 - x/10 on [0,1]}.
        f = cj.chebfun(
            lambda x: jnp.where(x < 0, jnp.exp((1 + 1j) * x),
                                1 - x / 10 + 0j),
            domain=(-1.0, 0.0, 1.0))
        (xmin, fmin), (xmax, fmax) = f.minandmax()
        y = np.array([complex(fmin), complex(fmax)])
        y_exact = np.array([np.exp(-1 - 1j), 1.0])
        fx = np.array([complex(f(jnp.asarray(float(xmin)))),
                       complex(f(jnp.asarray(float(xmax))))])
        assert np.max(np.abs(y - y_exact)) <= 10 * f.vscale * EPS
        assert np.max(np.abs(fx - y_exact)) <= 10 * f.vscale * EPS


class TestChebfunMinandmaxLocal:
    """MATLAB test_minandmax.m pass(7): local extrema (incl. endpoints)."""

    def test_local_extrema_scalar(self):
        f = cj.chebfun(lambda x: jnp.sin(x) ** 2 + jnp.sin(x ** 2),
                       domain=[0, 4])
        x, y = f.minandmax("local")
        x = np.asarray(x)
        y = np.asarray(y)
        y_exact = np.array([
            0.0, 1.923771282655145, -0.342247088203205, 1.117294907913736,
            -0.971179645473729, 1.343997479566445, 0.284846700239241])
        x_exact = np.array([
            0.0, 1.323339426259694, 2.220599667639221, 2.781195946808315,
            3.308480466603983, 3.776766383330969, 4.0])
        assert len(y) == 7
        assert np.max(np.abs(y - y_exact)) < 10 * f.vscale * EPS
        assert np.max(np.abs(x - x_exact)) < 1e-10

    def test_local_extrema_array_valued_nan_padded(self):
        # pass(7'): columns of unequal extrema counts are NaN-padded.
        op = lambda t: jnp.sin(t) ** 2 + jnp.sin(t ** 2)  # noqa: E731
        f = cj.chebfun(lambda x: jnp.stack([op(x), op(x / 2)], axis=-1),
                       domain=[0, 4])
        x, y = f.minandmax("local")
        y = np.asarray(y)
        assert y.shape == (7, 2)
        col0 = np.array([
            0.0, 1.923771282655145, -0.342247088203205, 1.117294907913736,
            -0.971179645473729, 1.343997479566445, 0.284846700239241])
        assert np.max(np.abs(y[:, 0] - col0)) < 1e-10
        assert np.max(np.abs(y[:3, 1]
                             - np.array([0.0, 1.923771282655145,
                                         0.070019315123878]))) < 1e-10
        assert np.all(np.isnan(y[3:, 1]))
