"""Port of MATLAB Chebfun tests/chebfun3v/test_real.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 100 * EPS


def _ff():
    return lambda x, y, z: jnp.cos(x * y * z)


class TestChebfun3vReal:
    def test_real_of_real(self):
        f = Chebfun3v.from_functions(_ff(), _ff())
        g = f.real()
        assert float((g - f).norm()) < TOL

    def test_real_of_imaginary(self):
        f = Chebfun3v.from_functions(_ff(), _ff())
        g = (1j * f).real()
        assert float(g.norm()) < TOL

    def test_real_of_complex_sum(self):
        f1 = Chebfun3v.from_functions(_ff(), _ff())
        f2 = Chebfun3v.from_functions(
            lambda x, y, z: jnp.sin(x + y ** 2 + z ** 3),
            lambda x, y, z: jnp.sin(x + y ** 2 + z ** 3))
        g = (f1 + 1j * f2).real()
        assert float((g - f1).norm()) < TOL
