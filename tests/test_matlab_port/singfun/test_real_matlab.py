"""Port of MATLAB Chebfun tests/singfun/test_real.m (Opus 4.8).

Original source predicates, including zero differences and norm estimates.

Provenance
----------
MATLAB source : tests/singfun/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun


def _sf(f):
    return Singfun.from_function(f)


class TestSingfunReal:
    def test_empty(self):
        f = Singfun.empty()
        assert f.real().isempty()

    def test_imaginary_smooth_not_singfun(self):
        f = _sf(lambda x: 1j * jnp.sin(x))
        assert not isinstance(f.real(), Singfun)

    def test_real_smooth(self):
        f = _sf(lambda x: jnp.exp(x))
        assert f.real().isequal(f.smoothPart)

    def test_real_with_exponents(self):
        f = _sf(lambda x: 1.0 / ((1 + x) * (1 - x)))
        assert f.real() == f

    def test_purely_imaginary(self):
        f = 1j * _sf(lambda x: 1.0 / ((1 + x) * (1 - x)))
        assert f.real().iszero()

    def test_complex_smooth_part(self):
        f = _sf(lambda x: (jnp.sin(x) + 1j * jnp.cos(x)) / ((1 + x) * (1 - x)))
        g = _sf(lambda x: jnp.sin(x) / ((1 + x) * (1 - x)))
        assert (f.real() - g).iszero()
