"""Port of MATLAB Chebfun tests/singfun/test_isequal.m (Opus 4.8).

chebfunjax Singfun has no ``isequal`` method; MATLAB isequal compares the
exponents and the smooth-part representation.  We reproduce that predicate
directly (exponents equal AND smooth-part coefficients bit-identical).
Construction is deterministic, so equal inputs give bit-identical coeffs.

The empty-Singfun and zeroSingFun cases (pass 1-2) are skipped: chebfunjax
has neither an empty representation nor a ``zeroSingFun`` factory.

Provenance
----------
MATLAB source : tests/singfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _isequal(f, g):
    if tuple(f.exponents) != tuple(g.exponents):
        return False
    cf, cg = f.coeffs, g.coeffs
    return cf.shape == cg.shape and bool(jnp.array_equal(cf, cg))


class TestSingfunIsequal:
    def test_empty_equal(self):
        assert Singfun.empty() == Singfun.empty()

    def test_zerosingfun_equal(self):
        assert Singfun.zeroSingFun() == Singfun.zeroSingFun()

    def test_identical_nonzero_equal(self):
        f = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        assert _isequal(f, g)

    def test_different_exponents_not_equal(self):
        f = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: 1.0 / (1 + x), (-1.8, 0.0))
        assert not _isequal(f, g)

    def test_different_smoothpart_not_equal(self):
        f = _sf(lambda x: jnp.cos(x) / (1 + x), (-1.0, 0.0))
        g = _sf(lambda x: jnp.sin(x) / (1 + x), (-1.0, 0.0))
        assert not _isequal(f, g)
