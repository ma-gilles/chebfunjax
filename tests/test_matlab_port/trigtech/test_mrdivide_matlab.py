"""Port of MATLAB Chebfun tests/trigtech/test_mrdivide.m (Opus 4.8[1m]).

mrdivide (A/B) divides a trigtech by a scalar/matrix (least squares), or a
numeric by a trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False) + 0.0031415926)
ALPHA = -0.194758928283640 + 0.075474485412665j


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _f2(x):
    return jnp.stack(
        [jnp.exp(jnp.sin(jnp.pi * x)), 3.0 / (4 - jnp.cos(jnp.pi * x))],
        axis=-1)


class TestTrigtechMrdivide:
    def test_div_by_zero_nan(self):
        f = _tt(_f2)
        assert Trigtech.mrdivide(f, 0).isnan()

    def test_div_by_scalar(self):
        f = _tt(_f2)
        g = Trigtech.mrdivide(f, ALPHA)
        gx = jnp.asarray(g(X))
        exact = jnp.asarray(_f2(X)) / ALPHA
        assert _ninf(gx - exact) < 50 * g.vscale * EPS

    def test_least_squares_identity(self):
        f = _tt(_f2)
        g = Trigtech.mrdivide(f, jnp.eye(2))
        err = (g @ jnp.eye(2)) - f
        assert _ninf(err(X)) < 10 * g.vscale * EPS

    def test_least_squares_row(self):
        f = _tt(_f2)
        g = Trigtech.mrdivide(f, jnp.array([[1.0, 1.0]]))
        gx = jnp.squeeze(jnp.asarray(g(X)))
        exact = (jnp.exp(jnp.sin(jnp.pi * X))
                 + 3.0 / (4 - jnp.cos(jnp.pi * X))) / 2
        assert _ninf(gx - exact) < 1e2 * g.vscale * EPS

    def test_scalar_over_function(self):
        f = _tt(lambda x: jnp.cos(jnp.sin(jnp.pi * x)))
        g = Trigtech.mrdivide(ALPHA, f)
        ip = jnp.asarray(f.innerProduct(g)).ravel()[0]
        assert abs(complex(ip) - ALPHA) < 10 * g.vscale * EPS

    def test_row_over_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)], axis=-1))
        g = Trigtech.mrdivide(jnp.array([[1.0, 1.0]]), f)
        gx = jnp.squeeze(jnp.asarray(g(X)))
        exact = jnp.sin(2 * jnp.pi * X) + jnp.cos(2 * jnp.pi * X)
        assert _ninf(gx - exact) < 10 * g.vscale * EPS

    def test_error_dim(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)], axis=-1))
        with pytest.raises(ValueError) as exc:
            Trigtech.mrdivide(f, jnp.array([[1.0, 2.0, 3.0]]))
        assert "CHEBFUN:TRIGTECH:mrdivide:size" in str(exc.value)

    def test_error_trigtech_div_trigtech(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        g = _tt(lambda x: jnp.cos(jnp.pi * x))
        with pytest.raises(ValueError) as exc:
            Trigtech.mrdivide(f, g)
        assert "CHEBFUN:TRIGTECH:mrdivide:trigtechDivTrigtech" \
            in str(exc.value)

    def test_error_bad_arg(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        with pytest.raises(ValueError) as exc:
            Trigtech.mrdivide(f, True)
        assert "CHEBFUN:TRIGTECH:mrdivide:badArg" in str(exc.value)
