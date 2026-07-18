"""Port of MATLAB Chebfun tests/chebfun2v/test_real.m (Fable 5).

FIXED (Fable 5): Chebfun2v.real() implemented; the norm/arithmetic
cancellation that previously lost half the digits is gone
(_add_separable now recompresses like MATLAB plus.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)


def _v(fn):
    f = Chebfun2.from_function(fn)
    return Chebfun2v([f.approx, f.approx])


class TestChebfun2vReal:
    def test_real_identity(self):
        # pass(1): real(F) == F for real F, tol 100*cheb2eps.
        F = _v(lambda x, y: jnp.cos(x * y))
        assert float((F - F.real()).norm()) < 100 * EPS

    def test_real_of_imaginary(self):
        # pass(2): real(1i*F) == 0.
        F = _v(lambda x, y: jnp.cos(x * y))
        assert float((1j * F).real().norm()) < 100 * EPS

    def test_real_of_complex_sum(self):
        # pass(3): real(F1 + 1i*F2) == F1.
        F1 = _v(lambda x, y: jnp.cos(x * y))
        F2 = _v(lambda x, y: jnp.sin(x + y ** 2))
        assert float((F1 - (F1 + 1j * F2).real()).norm()) < 100 * EPS
