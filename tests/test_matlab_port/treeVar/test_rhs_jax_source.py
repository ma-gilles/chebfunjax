"""Runtime scalar leaves and assembly for native first-order RHS contracts.

Provenance
----------
MATLAB source: @treeVar/toRHS.m, @treeVar/tree2infix.m.
Chebfun commit: 7574c77.
JIT/AD are supplemental Python controls on dtype-stable polynomial expressions;
no unary real-to-complex fallback is exercised or claimed traceable here.
"""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.treevar import to_first_order

EPS = jnp.finfo(jnp.float64).eps
DOM = (-1.,1.)


@pytest.mark.parametrize('coefficient', [2., 2, 2.+1j, jnp.asarray(2.), jnp.asarray(2.+1j)])
def test_numeric_scalar_leading_coefficient(coefficient):
    fun, *_ = to_first_order(lambda u: u.diff(2)*coefficient+u, 1., DOM)
    got = fun(.25,jnp.asarray([.5,.2]))
    expected = jnp.asarray([.2, .5/coefficient])[:,None]
    assert got.shape == (2,1)
    assert jnp.max(jnp.abs(got-expected)) < 10*EPS
    assert jnp.iscomplexobj(got) == jnp.iscomplexobj(coefficient)


def test_complex_chebfun_rhs_leaf():
    rhs = cj.chebfun(lambda t: 1+2j*t, domain=DOM)
    fun, *_ = to_first_order(lambda u: u.diff(2)+u, rhs, DOM)
    got = fun(.25,jnp.asarray([.5,.2]))
    assert jnp.max(jnp.abs(got-jnp.asarray([[.2],[.5+.5j]]))) < 10*EPS
    assert jnp.iscomplexobj(got)


def test_polynomial_jit_and_state_derivatives():
    fun, *_ = to_first_order(lambda u: u.diff(2)+u**2, 1., DOM)
    u = jnp.asarray([.5,.2])
    expected = jnp.asarray([[.2],[.75]])
    assert jnp.array_equal(jax.jit(fun)(.25,u), expected)
    jac = jax.jacfwd(lambda v: fun(.25,v).ravel())(u)
    assert jnp.array_equal(jac,jnp.asarray([[0.,1.],[-1.,0.]]))
    reverse = jax.grad(lambda v: jnp.sum(fun(.25,v)))(u)
    assert jnp.array_equal(reverse,jnp.asarray([-1.,1.]))


def test_time_dependent_leaves_jit_and_time_derivative():
    coefficient = cj.chebfun(lambda t: 2+t,domain=DOM)
    lower = cj.chebfun(lambda t: 3*t,domain=DOM)
    rhs = cj.chebfun(lambda t: 1+t**2,domain=DOM)
    fun, *_ = to_first_order(lambda u: u.diff(2)*coefficient+u*lower,rhs,DOM)
    t = jnp.asarray(.25)
    u = jnp.asarray([.5,.2])
    def reference(time):
        return jnp.asarray([u[1],(1+time**2-3*time*u[0])/(2+time)])[:,None]
    assert jnp.max(jnp.abs(jax.jit(fun)(t,u)-reference(t))) < 10*EPS
    derivative = jax.jacfwd(lambda time: fun(time,u))(t)
    analytic = ((2*t-3*u[0])*(2+t)-(1+t**2-3*t*u[0]))/(2+t)**2
    assert jnp.max(jnp.abs(derivative-jnp.asarray([[0.],[analytic]]))) < 10*EPS


def test_complex_polynomial_state_jvp():
    fun, *_ = to_first_order(lambda u: u.diff(2)+u**2, 1., DOM)
    u = jnp.asarray([.5+.25j,.2-.1j])
    tangent=jnp.asarray([1.+2j,3.-1j])
    got = jax.jit(fun)(.25,u)
    assert got.dtype == jnp.complex128
    expected = jnp.asarray([u[1],1-u[0]**2])[:,None]
    assert jnp.array_equal(got,expected)
    _, result = jax.jvp(lambda v: fun(.25,v),(u,),(tangent,))
    expected_tangent = jnp.asarray([tangent[1],-2*u[0]*tangent[0]])[:,None]
    assert jnp.array_equal(result,expected_tangent)


@pytest.mark.parametrize('direction', ['forward','reverse'])
def test_state_ad_independent_of_jit(direction):
    fun, *_ = to_first_order(lambda u: u.diff(2)+u**2,1.,DOM)
    u=jnp.asarray([.5,.2])
    if direction == 'forward':
        got=jax.jacfwd(lambda v: fun(.25,v).ravel())(u)
        expected=jnp.asarray([[0.,1.],[-1.,0.]])
    else:
        got=jax.grad(lambda v:jnp.sum(fun(.25,v)))(u)
        expected=jnp.asarray([-1.,1.])
    assert jnp.array_equal(got,expected)


@pytest.mark.parametrize('dtype', [jnp.float32,jnp.float64,jnp.complex64,jnp.complex128])
def test_eager_output_dtype_with_double_coefficient(dtype):
    fun, *_ = to_first_order(lambda u:u.diff(2)+u**2,1.,DOM)
    got=fun(.25,jnp.asarray([.5,.2],dtype=dtype))
    expected_dtype=jnp.complex128 if jnp.issubdtype(dtype,jnp.complexfloating) else jnp.float64
    assert got.dtype == expected_dtype
    assert got.shape == (2,1)
