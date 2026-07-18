"""Port of MATLAB Chebfun tests/chebfun2v/test_conj.m (Fable 5).

FIXED (Fable 5): Chebfun2v.conj() implemented; the norm/arithmetic
cancellation that previously lost half the digits is gone
(_add_separable now recompresses like MATLAB plus.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_conj.m
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


class TestChebfun2vConj:
    def test_conj_of_real_identity(self):
        # pass(1): conj(F) == F for real F, tol 100*cheb2eps.
        F = _v(lambda x, y: jnp.cos(x * y))
        assert float((F - F.conj()).norm()) < 100 * EPS

    def test_conj_of_imaginary(self):
        # pass(2): conj(1i*F) == -1i*F.
        F = _v(lambda x, y: jnp.cos(x * y))
        assert float(((1j * F).conj() - (-1j) * F).norm()) < 100 * EPS

    def test_conj_of_complex_sum(self):
        # pass(3): conj(F1 + 1i*F2) == F1 - 1i*F2.
        F1 = _v(lambda x, y: jnp.cos(x * y))
        F2 = _v(lambda x, y: jnp.sin(x + y ** 2))
        assert float(((F1 + 1j * F2).conj()
                      - (F1 - 1j * F2)).norm()) < 100 * EPS
