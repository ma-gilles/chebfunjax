"""Literal port of MATLAB tests/chebfun3t/test_compose.m.

All operators and unary compositions below are applied to Chebfun3T objects;
the test preserves the source functions, domain, and assertion bounds.

Provenance
----------
MATLAB source : tests/chebfun3t/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS


def _f(x, y, z):
    return jnp.cos(x * y * z) + jnp.sin(x * y * z) + y - 0.1


def _same_norm(a, b):
    return float((a - b).norm())


@pytest.mark.slow
class TestChebfun3tCompose:
    def test_native_product_callback_predicate(self):
        # Native predicate 1 reconstructs the composed product from a
        # callback that evaluates the already-constructed Chebfun3T f.
        f = chebfun3t(_f)
        g = chebfun3t(lambda x, y, z: _f(x, y, z)
                      * jnp.sin((x - 0.1) * (y + 0.4) * (z + 0.8)))
        composed = chebfun3t(
            lambda x, y, z: f(x, y, z)
            * jnp.sin((x - 0.1) * (y + 0.4) * (z + 0.8))
        )
        assert _same_norm(g, composed) < TOL

    def test_remaining_matlab_assertions(self):
        f = chebfun3t(_f)

        g = chebfun3t(lambda x, y, z: jnp.sin(_f(x, y, z)))
        assert _same_norm(g, f.sin()) < TOL

        g = chebfun3t(lambda x, y, z: jnp.cos(_f(x, y, z)))
        assert _same_norm(g, f.cos()) < TOL

        g = chebfun3t(lambda x, y, z: jnp.sinh(_f(x, y, z)))
        assert _same_norm(g, f.sinh()) < TOL

        g = chebfun3t(lambda x, y, z: jnp.cosh(_f(x, y, z)))
        assert _same_norm(g, f.cosh()) < TOL

        g = chebfun3t(lambda x, y, z: jnp.tanh(-_f(x, y, z)))
        assert _same_norm(g, (-f).tanh()) < TOL

        g = chebfun3t(lambda x, y, z: jnp.exp(_f(x, y, z)))
        assert _same_norm(g, f.exp()) < TOL

        f = chebfun3t(lambda x, y, z: jnp.sin(10 * x * y * z),
                      (-1.0, 2.0, -1.0, 1.0, -3.0, -1.0))
        assert float((f + f + f - 3 * f).norm()) < 100 * TOL
        assert float((f * f - f**2).norm()) < TOL
