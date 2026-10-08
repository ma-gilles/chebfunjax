"""Port of MATLAB Chebfun tests/classicfun/test_isequal.m (Fable 5).

All ten original predicates, including singular bounded and unbounded inputs.

Provenance
----------
MATLAB source : tests/classicfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
        # Source uses fractional powers of negative values: MATLAB promotes
        # to complex; JAX requires the explicit complex dtype adapter.
        f = Bndfun.from_function(
            lambda x: (x.astype(jnp.complex128)-7)**-.5*jnp.sin(x),
            DOM, exponents=(0, -.5))
        g = Bndfun.from_function(
            lambda x: (x.astype(jnp.complex128)-7)**-.6*(jnp.cos(x)**2+1),
            DOM, exponents=(0, -.6))
        assert not f.isequal(g)



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
        dom = Domain((-jnp.inf, jnp.inf))
        f = Unbndfun.from_function(lambda x: (1-jnp.exp(-x**2))/x, dom)
        g = Unbndfun.from_function(lambda x: x**2*(1-jnp.exp(-x**2)), dom, exps=(2, 2))
        assert not f.isequal(g)
        assert not g.isequal(f)
