"""Port of MATLAB Chebfun tests/chebfun3v/test_imag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 10 * EPS


def _triple():
    return (lambda x, y, z: jnp.cos(x * y * z),
            lambda x, y, z: jnp.sin(x * y * z),
            lambda x, y, z: jnp.exp(x * y * z))


class TestChebfun3vImag:
    def test_imag_of_real(self):
        f = Chebfun3v.from_functions(*_triple())
        g = f.imag()
        assert float(g.norm()) < TOL

    def test_imag_of_imaginary(self):
        f = Chebfun3v.from_functions(*_triple())
        g = (1j * f).imag()
        assert float((f - g).norm()) < 100 * TOL

    def test_imag_of_complex_sum(self):
        f1 = Chebfun3v.from_functions(*_triple())
        f2 = Chebfun3v.from_functions(
            lambda x, y, z: jnp.sin(x + y ** 2 + z),
            lambda x, y, z: jnp.sin(x + y ** 2 + z),
            lambda x, y, z: jnp.exp(x * y * z))
        g = (f1 + 1j * f2).imag()
        assert float((f2 - g).norm()) < 100 * TOL
