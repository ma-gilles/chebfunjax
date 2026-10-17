"""Original aggregate predicates from MATLAB tests/chebop/test_paramODE.m.

Provenance
----------
MATLAB source: tests/chebop/test_paramODE.m
Chebfun commit: 7574c77
Natural and forced setups use default adaptive solves and original norm bounds.
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


def _norm(value):
    return value.norm(2) if isinstance(value, Chebfun) else jnp.linalg.norm(jnp.asarray(value).reshape(-1))


def test_original_natural_and_forced_parameter_aggregate():
    x = cj.chebfun(lambda t: t)
    natural = Chebop(lambda x, u, a: x*u + .001*u.diff(2) + a)
    natural.lbc = lambda u, a: [u+a+1, u.diff()]
    natural.rbc = lambda u, a: u-1
    u = natural.solve(cj.chebfun(0.))
    res1 = natural(x, u)
    forced = Chebop(lambda x, u, a: [x*u + .001*u.diff(2) + a, a.diff()])
    forced.lbc = lambda u, a: [u+a+1, u.diff()]
    forced.rbc = lambda u, a: u-1
    v = forced.solve([cj.chebfun(0.), cj.chebfun(0.)])
    res2 = forced(x, v)
    error = _norm(res1) + _norm(res2[0]) + _norm(u[0]-v[0]) + _norm(u[1]-v[1])
    assert float(error) < 100*ChebopPref().bvpTol


def test_original_two_functions_two_parameters_aggregate():
    x = cj.chebfun(lambda t: t)
    operator = Chebop(lambda x, u, v, a, b:
                     [x*v + .001*u.diff(2) + a + 2*b, a+v.diff()-u], (-1., 1.))
    operator.lbc = lambda u, v, a, b: [u+a, u.diff()]
    operator.rbc = lambda u, v, a, b: [u, u.diff()-a, v-b]
    rhs = [x.sin(), x]
    solution = operator.solve(rhs)
    residual = operator(x, solution)
    error = (residual[0]-rhs[0]).norm(jnp.inf) + (residual[1]-rhs[1]).norm(jnp.inf)
    assert float(error) < 100*ChebopPref().bvpTol
