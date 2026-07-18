"""Port of MATLAB Chebfun tests/chebfun2/test_conj.m (Fable 5).

FIXED (Fable 5 audit): ``Chebfun2.conj()`` now exists.

Provenance
----------
MATLAB source : tests/chebfun2/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Conj:
    def test_conj_real_identity(self):
        # pass(1): conj of a real chebfun2 is itself.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert float((f - f.conj()).norm()) < TOL

    def test_conj_imaginary(self):
        # pass(2): conj(1i f) = -1i f, so 1i f + conj(1i f) == 0.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert float(((1j * f) + (1j * f).conj()).norm()) < 100 * TOL

    def test_conj_complex_sum(self):
        # pass(3): conj(f1 + 1i f2) == f1 - 1i f2.
        f1 = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        f2 = Chebfun2.from_function(lambda x, y: jnp.sin(x + y ** 2))
        assert float(((f1 - 1j * f2) - (f1 + 1j * f2).conj()).norm()) < TOL
