"""Native division clauses16–17; raw uniform stream in source order.

Provenance
----------
MATLAB source: tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77
Raw rand uses the separately native-qualified explicit-state MT19937 helper.
Source clause16 reseeds6178; clause17 uses the next draw. The involved
constructor/sampleTest paths do not call rand; no constructor RNG draw is
inserted. Affine maps execute as separate eager multiply and add operations.
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform

EPS = jnp.finfo(jnp.float64).eps


def _draws():
    u16, state = matlab_rand_uniform(init_matlab_twister(6178), (100,))
    u17, state = matlab_rand_uniform(state, (100,))
    return 9*u16-2, 98*u17+2


def test_source_16_singular_complex_fractional():
    x, _ = _draws()
    pow1, pow2 = -.5, -.3
    op1 = lambda x: (x.astype(jnp.complex128)-7)**pow1*jnp.sin(100*x)
    op2 = lambda x: (x.astype(jnp.complex128)-7)**pow2*(jnp.cos(300*x)**2+1)
    f = cj.chebfun(op1, domain=(-2.,7.), exps=(0,pow1), splitting=True)
    g = cj.chebfun(op2, domain=(-2.,7.), exps=(0,pow2), splitting=True)
    h = f/g
    exact = (x.astype(jnp.complex128)-7)**(pow1-pow2)*(jnp.sin(100*x)/(jnp.cos(300*x)**2+1))
    assert jnp.max(jnp.abs(h(x)-exact)) < 1e5*EPS*jnp.max(jnp.abs(exact))


def test_source_17_unbounded():
    _, x = _draws()
    f = cj.chebfun(lambda x: jnp.exp(-x**2), domain=(2.,jnp.inf))
    g = cj.chebfun(lambda x: x**2, domain=(2.,jnp.inf), exps=(0,2))
    h = f/g
    exact = jnp.exp(-x**2)*x**-2
    assert jnp.max(jnp.abs(h(x)-exact)) < 10*EPS*f.vscale
