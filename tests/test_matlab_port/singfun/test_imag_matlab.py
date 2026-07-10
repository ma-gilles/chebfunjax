"""Port of MATLAB Chebfun tests/singfun/test_imag.m (Opus 4.8).

chebfunjax Singfun implements no ``imag`` method and does not support complex
smooth parts, so the non-trivial assertions are xfailed (the ``imag`` call
raises ``AttributeError``).  The empty case is skipped.

Provenance
----------
MATLAB source : tests/singfun/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun

_REASON = "chebfunjax Singfun has no imag() method / no complex support"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunImag:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty Singfun representation")

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_real_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        assert not isinstance(f.imag(), Singfun)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_imaginary_smooth(self):
        f = _sf(lambda x: 1j * jnp.exp(x), (0.0, 0.0))
        assert f.imag() is not None

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_imaginary_with_exponents(self):
        f = _sf(lambda x: 1j / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.imag() is not None

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_purely_real(self):
        f = 1j * _sf(lambda x: 1j / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.imag() is not None

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_complex_smooth_part(self):
        f = _sf(lambda x: (jnp.sin(x) + 1j * jnp.cos(x)) / ((1 + x) * (1 - x)), (-1.0, -1.0))
        g = _sf(lambda x: jnp.cos(x) / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.imag() == g
