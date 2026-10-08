"""Port of MATLAB Chebfun tests/singfun/test_imag.m (Opus 4.8).

Original source predicates, including zero differences and norm estimates.

Provenance
----------
MATLAB source : tests/singfun/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun


def _sf(f):
    return Singfun.from_function(f)


class TestSingfunImag:
    def test_empty(self):
        f = Singfun.empty()
        assert f.imag().isempty()

    def test_real_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x))
        assert not isinstance(f.imag(), Singfun)

    def test_imaginary_smooth(self):
        f = _sf(lambda x: 1j * jnp.exp(x))
        assert (1j*f.imag()-f.smoothPart).normest() < 10*jnp.finfo(jnp.float64).eps

    def test_imaginary_with_exponents(self):
        f = _sf(lambda x: 1j / ((1 + x) * (1 - x)))
        assert (1j*f.imag()-f).normest() < 10*jnp.finfo(jnp.float64).eps

    def test_purely_real(self):
        f = 1j * _sf(lambda x: 1j / ((1 + x) * (1 - x)))
        assert f.imag().iszero()

    def test_complex_smooth_part(self):
        f = _sf(lambda x: (jnp.sin(x) + 1j * jnp.cos(x)) / ((1 + x) * (1 - x)))
        g = _sf(lambda x: jnp.cos(x) / ((1 + x) * (1 - x)))
        assert (f.imag() - g).iszero()
