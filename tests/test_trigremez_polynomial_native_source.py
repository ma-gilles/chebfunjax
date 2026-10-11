"""Original polynomial terminal predicates; Chebfun7574c77 test_trigremez.m."""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


def test_native_polynomial_predicates_1_2_3():
    tol = 1e-12
    f = cj.chebfun(lambda x: jnp.cos(4 * jnp.pi * x) + 1, domain=(0, 2), trig=True)
    n = 1
    p, errMax, status = cj.trigremez(f, n)
    err = f - p
    assert p.isPeriodicTech() and len(p) == 2 * n + 1
    assert jnp.max(jnp.abs(jnp.abs(err(status["xk"])) - errMax)) < 100 * tol
    assert (p - 1).norm(np.inf) < 100 * tol


def test_native_polynomial_predicates_4_5_6():
    tol = 1e-12
    f = cj.chebfun(lambda x: jnp.cos(4 * jnp.pi * x) + 1, domain=(0, 2), trig=True)
    n = 4
    f = cj.chebfun(f)
    p, errMax, status = cj.trigremez(f, n)
    err = f - p
    assert p.isPeriodicTech() and len(p) == 2 * n + 1
    assert jnp.max(jnp.abs(jnp.abs(err(status["xk"])) - errMax)) < 100 * tol
    assert (p - f).norm(np.inf) < 100 * tol


def test_native_empty_operation_and_predicate_7():
    assert cj.trigremez(cj.chebfun()).isempty()
    tol = 1e-12
    a, b = 1, 5
    s = a + 0.7 * (b - a)
    f = cj.chebfun(
        lambda x: (s - x) / (s - a) * (x < s) + (x - s) / (b - s) * (x >= s),
        domain=(a, b),
        splitting=True,
    )
    n = 2
    p, errMax, status = cj.trigremez(f, n)
    err = p - f
    assert jnp.max(jnp.abs(jnp.abs(err(status["xk"])) - errMax)) < 100 * tol


def test_native_rational_predicates_8_9_10_11():
    tol=1e-12
    m=n=2
    x=cj.chebfun("x")
    f=abs(x)
    p,q,r,err,status=cj.trigremez(f,m,n)
    assert len(p)==2*m+1 and len(q)==2*n+1
    xk=status['xk']
    equi_err=f(xk)-r(xk)
    assert jnp.std(jnp.abs(equi_err),ddof=1) < tol
    if equi_err[0] < 0:
        equi_err=-equi_err
    assert jnp.all(jnp.sign(equi_err[::2])==1)
    assert jnp.all(jnp.sign(equi_err[1::2])==-1)


def test_native_rational_predicates_12_13():
    tol=1e-12
    m=n=6
    x=cj.chebfun("x")
    f=(jnp.pi*x).sin().exp()
    p,q,r,err,status=cj.trigremez(f,m,n)
    assert (f-p/q).norm(np.inf) < tol
    assert len(p)==2*m+1 and len(q)==2*n+1
