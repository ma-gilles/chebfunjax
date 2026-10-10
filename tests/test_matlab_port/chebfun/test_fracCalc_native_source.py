"""All seven native test_fracCalc.m predicates, Chebfun commit7574c77.

Python entry adapters: fracDiff(kind=...) for diff(f,q,kind), and explicit
Quasimatrix columns for cheb2quasi. Bounds and continuous norms are native.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf
from scipy.special import gamma

from chebfunjax import chebfun
from chebfunjax.chebfun1d.chebfun import _hscale
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = np.finfo(float).eps


def check_values(truth, actual, xx, factor):
    error = float(jnp.max(jnp.abs(truth(xx)-actual(xx))))
    bound = factor * EPS * actual.vscale * _hscale(actual)
    assert error < bound, (error, bound)


@pytest.mark.parametrize("n", [1, 4])
def test_native_polynomial_fractional_derivative(n):
    x = chebfun(lambda t: t, domain=(0., 1.))
    q = np.sqrt(2)/2
    actual = (x**n).diff(q)
    truth = (gamma(n+1)/gamma(n+1-q) *
             chebfun(lambda t: t**(n-q), domain=(0., 1.), exps=[n-q, 0.]))
    check_values(truth, actual, jnp.linspace(.1, .9, 10), 10)


@pytest.mark.parametrize("kind,factor", [("Caputo", 10), ("RL", 100)])
def test_native_exponential_fractional_derivative(kind, factor):
    u = chebfun(jnp.exp, domain=(0., 1.))
    if kind == "Caputo":
        truth = chebfun(lambda x: erf(jnp.sqrt(x))*jnp.exp(x),
                        domain=(0., 1.), exps=[.5, 0.])
    else:
        truth = chebfun(lambda x: erf(jnp.sqrt(x))*jnp.exp(x)+1/jnp.sqrt(jnp.pi*x),
                        domain=(0., 1.), exps=[-.5, 0.])
    check_values(truth, u.fracDiff(.5, kind), jnp.linspace(.1, .9, 10), factor)


def test_native_integrate_twice():
    domain = (-np.sqrt(2)*np.pi, np.pi)
    f = chebfun(jnp.sin, domain=domain)
    actual = f.fracInt(.3).fracInt(.7)
    check_values(f.cumsum(), actual, jnp.linspace(domain[0]+.1, domain[1]-.1, 10), 10)


def test_native_differentiate_twice():
    domain = (-np.sqrt(2)*np.pi, np.pi)
    f = chebfun(jnp.sin, domain=domain)
    actual = f.fracDiff(.3).fracDiff(.7)
    check_values(f.diff(), actual, jnp.linspace(domain[0]+.1, domain[1]-.1, 10), 100)


def test_native_quasimatrix_fractional_derivative():
    x = chebfun(lambda t: t, domain=(0., 1.))
    vandermonde = x.vander(5)
    V = Quasimatrix([vandermonde[:, k] for k in range(5)], x.domain)
    F = V.diff(.5)
    error = sum(float((V[k].diff(.5)-F[k]).norm()) for k in range(len(V)))
    assert error < 10*EPS
