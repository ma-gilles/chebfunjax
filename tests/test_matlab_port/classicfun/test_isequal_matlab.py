"""Port of MATLAB Chebfun tests/classicfun/test_isequal.m (Fable 5).

``Classicfun.isequal`` (inherited by ``Bndfun``) and ``Unbndfun.isequal`` now
compare the domain and the underlying onefun, so the non-singular MATLAB
assertions are ported at MATLAB's tolerances (``isequal`` is exact, so no
tolerance is involved).

Gaps vs MATLAB (honest skip):
* Pass 6 builds two BNDFUNs with ``exponents`` (SingFun endpoint blow-up) and
  pass 8-9 build an UNBNDFUN with ``exponents`` under ``blowup = true``.  Those
  belong to the SingFun subsystem, not to this file's ``isequal`` logic.

Provenance
----------
MATLAB source : tests/classicfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

# MATLAB: data.domain = [-2 7].
DOM = Domain((-2.0, 7.0))


class TestClassicfunIsequalBndfun:
    def test_identical_is_symmetric(self):
        # pass(1): isequal(f, g) && isequal(g, f) for g = f.
        f = Bndfun.from_function(jnp.sin, DOM)
        g = f
        assert f.isequal(g) and g.isequal(f)

    def test_different_function(self):
        # pass(2): sin vs cos.
        f = Bndfun.from_function(jnp.sin, DOM)
        g = Bndfun.from_function(jnp.cos, DOM)
        assert not f.isequal(g)

    def test_scalar_vs_array_valued(self):
        # pass(3): sin(x) vs [sin(x) cos(x)].
        f = Bndfun.from_function(jnp.sin, DOM)
        g = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        assert not f.isequal(g)

    def test_array_valued_identical(self):
        # pass(4): f = g, both [sin(x) cos(x)].
        g = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        f = g
        assert f.isequal(g)

    def test_array_valued_different_columns(self):
        # pass(5): [sin cos] vs [sin exp].
        f = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM)
        g = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1), DOM)
        assert not f.isequal(g)

    def test_different_domain(self):
        # Implied by MATLAB's `all(f.domain == g.domain)` conjunct.
        f = Bndfun.from_function(jnp.sin, DOM)
        g = Bndfun.from_function(jnp.sin, Domain((-2.0, 8.0)))
        assert not f.isequal(g)

    def test_singular_bndfuns(self):
        # pass(6)
        pytest.skip(
            "MATLAB pass 6 compares two BNDFUNs built with 'exponents' "
            "(SingFun endpoint blow-up), which belongs to the SingFun "
            "subsystem rather than to classicfun/isequal")


class TestClassicfunIsequalUnbndfun:
    def test_doubly_infinite_self_equal(self):
        # pass(7): f = unbndfun((1-exp(-x^2))/x on [-inf inf]); isequal(f, f).
        dom = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(
            lambda x: (1 - jnp.exp(-(x**2))) / x, dom)
        assert f.isequal(f)

    def test_left_infinite_array_valued_self_equal(self):
        # pass(10): array-valued unbndfun on [-inf, -3*pi].
        dom = Domain((-jnp.inf, -3.0 * np.pi))
        f = Unbndfun.from_function(
            lambda x: jnp.stack(
                [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1),
            dom)
        assert f.isequal(f)

    def test_blowup_unbndfun(self):
        # pass(8:9)
        pytest.skip(
            "MATLAB passes 8-9 compare against an UNBNDFUN built with "
            "'exponents' under blowup = true (SingFun endpoint blow-up), "
            "which belongs to the SingFun subsystem")
