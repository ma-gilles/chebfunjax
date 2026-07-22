"""Port of MATLAB Chebfun tests/chebfun3v/test_conj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 100 * EPS


def _ff():
    return lambda x, y, z: jnp.cos(x * y * z)


class TestChebfun3vConj:
    def test_conj_real_is_identity(self):
        f = Chebfun3v.from_functions(_ff(), _ff())
        g = f.conj()
        assert float((f - g).norm()) < TOL

    def test_conj_imaginary(self):
        f = Chebfun3v.from_functions(_ff(), _ff())
        g = (1j * f).conj()
        assert float((1j * f + g).norm()) < TOL

    def test_conj_complex_sum(self):
        f1 = Chebfun3v.from_functions(_ff(), _ff())
        f2 = Chebfun3v.from_functions(lambda x, y, z: jnp.sin(x + y ** 2),
                                      lambda x, y, z: jnp.sin(x + y ** 2))
        g = (f1 + 1j * f2).conj()
        assert float((f1 - 1j * f2 - g).norm()) < TOL
