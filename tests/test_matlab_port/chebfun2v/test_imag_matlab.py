"""Port of MATLAB Chebfun tests/chebfun2v/test_imag.m (Fable 5).

FIXED (Fable 5): Chebfun2v.imag() implemented; the norm/arithmetic
cancellation that previously lost half the digits is gone
(_add_separable now recompresses like MATLAB plus.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_imag.m
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


class TestChebfun2vImag:
    def test_imag_of_real_is_zero(self):
        # pass(1): imag(F) == 0 for real F, tol 100*cheb2eps.
        F = _v(lambda x, y: jnp.cos(x * y))
        assert float(F.imag().norm()) < 100 * EPS

    def test_imag_of_complex_sum(self):
        # pass(2): imag(F1 + 1i*F2) == F2.
        F1 = _v(lambda x, y: jnp.cos(x * y))
        F2 = _v(lambda x, y: jnp.sin(x + y ** 2))
        assert float((F2 - (F1 + 1j * F2).imag()).norm()) < 100 * EPS
