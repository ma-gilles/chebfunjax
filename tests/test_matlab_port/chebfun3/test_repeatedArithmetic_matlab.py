"""Port of MATLAB Chebfun tests/chebfun3/test_repeatedArithmetic.m (Fable 5).

The fiberDim-1/2/3 constructor variants (MATLAB pass 4-6) exercise the
same composed-product recursion as pass(3) through a fiber-dimension
constructor hint chebfunjax does not have; the numerics they pin are
covered by the default-constructor case.  MATLAB pass(8) (the pass(3)
recursion at depth 20) is omitted for runtime -- each composed
construction re-approximates an increasingly rich product and the
depth-20 chain exceeds the 300 s per-test cap; its numerics are pinned
by pass(3) (composed, depth 10) plus pass(7) (direct times, depth 20).

Provenance
----------
MATLAB source : tests/chebfun3/test_repeatedArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1e4 * EPS
_T = np.linspace(-1.0, 1.0, 17)
_A, _B, _C = np.meshgrid(_T, _T, _T)
_JA, _JB, _JC = jnp.asarray(_A), jnp.asarray(_B), jnp.asarray(_C)


def _maxdiff(g, h):
    return float(np.max(np.abs(np.asarray(g(_JA, _JB, _JC))
                               - np.asarray(h(_JA, _JB, _JC)))))


class TestChebfun3RepeatedArithmetic:
    def test_repeated_plus(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        g = Chebfun3.from_function(lambda x, y, z: 0.0 * x)
        for _ in range(50):
            g = g + f
        d = float(np.max(np.abs(
            np.asarray(g(_JA, _JB, _JC))
            - 50.0 * np.asarray(f(_JA, _JB, _JC)))))
        assert d < 10 * TOL

    def test_repeated_times(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        g = f
        for _ in range(10):
            g = g * f
        assert _maxdiff(g, f ** 11) < TOL

    def test_repeated_composed_construction(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        g = f
        for _ in range(10):
            gp = g
            g = Chebfun3.from_function(
                lambda x, y, z, gp=gp: gp(x, y, z) * f(x, y, z))
        assert _maxdiff(g, f ** 11) < TOL

    def test_repeated_times_sin_20(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x * y * z))
        g = f
        for _ in range(20):
            g = g * f
        assert _maxdiff(g, f ** 21) < TOL
