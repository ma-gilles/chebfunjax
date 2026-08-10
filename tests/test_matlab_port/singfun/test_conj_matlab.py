"""Port of MATLAB Chebfun tests/singfun/test_conj.m (Opus 4.8).

chebfunjax Singfun implements no ``conj`` method and does not support complex
smooth parts, so the non-trivial assertions are xfailed (the ``conj`` call
raises ``AttributeError``).  The empty case is skipped.

Provenance
----------
MATLAB source : tests/singfun/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun

_REASON = "chebfunjax Singfun has no conj() method / no complex support"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunConj:
    def test_empty(self):
        f = Singfun.empty()
        assert f.conj().isempty()

    def test_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        assert not isinstance(f.conj(), Singfun)

    def test_real_smooth(self):
        f = _sf(lambda x: jnp.exp(x), (0.0, 0.0))
        assert f.conj() is f.smoothPart

    def test_real_with_exponents(self):
        f = _sf(lambda x: 1.0 / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.conj() == f

    def test_purely_imaginary(self):
        f = 1j * _sf(lambda x: 1.0 / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.conj() == -f

    def test_complex_smooth_part(self):
        f = _sf(lambda x: (jnp.sin(x) + 1j * jnp.cos(x)) / ((1 + x) * (1 - x)), (-1.0, -1.0))
        g = _sf(lambda x: (jnp.sin(x) - 1j * jnp.cos(x)) / ((1 + x) * (1 - x)), (-1.0, -1.0))
        assert f.conj() == g
