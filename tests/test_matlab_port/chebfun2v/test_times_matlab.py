"""Port of MATLAB Chebfun tests/chebfun2v/test_times.m (Fable 5).

FIXED (Fable 5): Chebfun2v.component-wise .* and scalar times implemented; the norm/arithmetic
cancellation that previously lost half the digits is gone
(_add_separable now recompresses like MATLAB plus.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)


def _v(fn):
    f = Chebfun2.from_function(fn)
    return Chebfun2v([f.approx, f.approx])


class TestChebfun2vTimes:
    def test_scalar_battery(self):
        # pass(1-8): 2*F == F*2 == G, F/2 == H (tol 1e3*cheb2eps).
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        F = Chebfun2v([f.approx, f.approx])
        G = Chebfun2v([(2 * f).approx, (2 * f).approx])
        H = Chebfun2v([(f * 0.5).approx, (f * 0.5).approx])
        tol = 1e3 * EPS
        assert float(((2 * F) - G).norm()) < tol
        assert float(((F * 2) - G).norm()) < tol
        assert float(((F / 2) - H).norm()) < tol
        assert float(((0.5 * F) - H).norm()) < tol

    def test_componentwise_power(self):
        # Native test_times pass(9): norm(F.^2 - K) < tol.
        # Source input is cos(x*y) on chebfun2's default domain.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        F = Chebfun2v([f.approx, f.approx])
        f_squared = f ** 2
        K = Chebfun2v([f_squared.approx, f_squared.approx])
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        assert float(((F ** 2) - K).norm()) < tol

    def test_componentwise_product(self):
        # Native test_times pass(10): norm(F.*F - K) < tol.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        F = Chebfun2v([f.approx, f.approx])
        K = Chebfun2v([(f * f).approx, (f * f).approx])
        assert float(((F * F) - K).norm()) < 1e3 * EPS
