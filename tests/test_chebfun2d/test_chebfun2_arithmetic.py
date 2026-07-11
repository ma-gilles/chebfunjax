"""Chebfun2 arithmetic (added by Claude Fable 5 — previously absent).

Mirrors MATLAB @separableApprox plus/minus/times/rdivide/power semantics:
+/- by exact low-rank concatenation, scalar ops on pivots, f.*g and f./g
and f.^p by constructor re-approximation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

RNG = np.random.default_rng(0)
X = jnp.asarray(RNG.uniform(-1, 1, 50))
Y = jnp.asarray(RNG.uniform(-1, 1, 50))


@pytest.fixture(scope="module")
def fg():
    f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
    g = Chebfun2.from_function(lambda x, y: x + y ** 2)
    return f, g


def _err(h, ref):
    return float(jnp.max(jnp.abs(h(X, Y) - ref)))


class TestChebfun2Arithmetic:
    def test_plus_minus_neg(self, fg):
        f, g = fg
        fx, gx = jnp.cos(X * Y), X + Y ** 2
        assert _err(f + g, fx + gx) < 1e-13
        assert _err(f - g, fx - gx) < 1e-13
        assert _err(-f, -fx) < 1e-13
        # compression keeps rank at most the concatenation
        assert 1 <= (f + g).rank <= f.rank + g.rank

    def test_scalar_ops(self, fg):
        f, _ = fg
        fx = jnp.cos(X * Y)
        assert _err(2 * f, 2 * fx) < 1e-13
        assert _err(f * 2, 2 * fx) < 1e-13
        assert _err(f + 1, fx + 1) < 1e-13
        assert _err(1 - f, 1 - fx) < 1e-13
        assert _err(f / 2, fx / 2) < 1e-13

    def test_product_quotient_power(self, fg):
        f, g = fg
        fx, gx = jnp.cos(X * Y), X + Y ** 2
        assert _err(f * g, fx * gx) < 1e-12
        assert _err(g / (f + 2), gx / (fx + 2)) < 1e-12
        assert _err(f ** 2, fx ** 2) < 1e-12

    def test_sum2_linear(self, fg):
        f, g = fg
        npt.assert_allclose(float((f + g).sum2()),
                            float(f.sum2()) + float(g.sum2()), atol=1e-12)

    def test_domain_mismatch_raises(self, fg):
        f, _ = fg
        h = Chebfun2.from_function(lambda x, y: x * y,
                                   domain=(0.0, 2.0, 0.0, 2.0))
        with pytest.raises(ValueError, match="matching domains"):
            _ = f + h
