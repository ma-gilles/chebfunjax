"""Chebfun3 arithmetic (added by Claude Fable 5 — previously absent).

Mirrors MATLAB @chebfun3 plus/minus/times/rdivide/power: +/- by exact
block-diagonal Tucker-core embedding, scalar ops on the core, f.*g,
f./g and f.^p by constructor re-approximation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

RNG = np.random.default_rng(0)
X = jnp.asarray(RNG.uniform(-1, 1, 25))
Y = jnp.asarray(RNG.uniform(-1, 1, 25))
Z = jnp.asarray(RNG.uniform(-1, 1, 25))


@pytest.fixture(scope="module")
def fg():
    f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
    g = Chebfun3.from_function(lambda x, y, z: x + y * z)
    return f, g


def _err(h, ref):
    return float(jnp.max(jnp.abs(h(X, Y, Z) - ref)))


class TestChebfun3Arithmetic:
    def test_plus_minus_neg(self, fg):
        f, g = fg
        fx, gx = jnp.cos(X * Y * Z), X + Y * Z
        assert _err(f + g, fx + gx) < 1e-13
        assert _err(f - g, fx - gx) < 1e-13
        assert _err(-f, -fx) < 1e-13

    def test_scalar_ops(self, fg):
        f, _ = fg
        fx = jnp.cos(X * Y * Z)
        assert _err(2 * f, 2 * fx) < 1e-13
        assert _err(f + 1, fx + 1) < 1e-13
        assert _err(1 - f, 1 - fx) < 1e-13
        assert _err(f / 2, fx / 2) < 1e-13

    def test_product_quotient_power(self, fg):
        f, g = fg
        fx, gx = jnp.cos(X * Y * Z), X + Y * Z
        assert _err(f * g, fx * gx) < 1e-12
        assert _err(g / (f + 2), gx / (fx + 2)) < 1e-12
        assert _err(f ** 2, fx ** 2) < 1e-12

    def test_sum3_linear(self, fg):
        f, g = fg
        npt.assert_allclose(float((f + g).sum3()),
                            float(f.sum3()) + float(g.sum3()), atol=1e-12)

    def test_domain_mismatch_raises(self, fg):
        f, _ = fg
        h = Chebfun3.from_function(lambda x, y, z: x * y * z,
                                   domain=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0))
        with pytest.raises(ValueError, match="matching domains"):
            _ = f + h
