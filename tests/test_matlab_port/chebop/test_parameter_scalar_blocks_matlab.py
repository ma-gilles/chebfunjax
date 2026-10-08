"""Literal natural setup from test_paramODE_nonlin_C2.m, including scalar output.

Provenance
----------
MATLAB source: tests/chebop/test_paramODE_nonlin_C2.m, @linop/linsolve.m.
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def test_natural_nonlinear_parameter_keeps_scalar_block():
    x = cj.chebfun(lambda t: t)
    operator = Chebop(lambda x, u, a: (1 - x**2)*u + .1*u.diff(2) + a*u.exp())
    operator.lbc = lambda u, a: [u + a + 1, u.diff()]
    operator.rbc = lambda u, a: u - 1
    operator.init = [x, -1.]
    # Bounded fixed-grid check retains the source continuous residual and bound.
    u, parameter = operator.solve(0., n=48)
    residual = operator.op(x, u, parameter)
    left = operator.lbc(u, parameter)
    right = operator.rbc(u, parameter)
    error = residual.norm() + left[0](-1.) + left[1](-1.) + right(1.)
    # MATLAB tol = 1e3*pref.bvpTol, factory bvpTol=5e-13.
    assert float(error) < 5e-10
    assert float(residual.norm()) < 5e-10
    # linsolve only converts isFun blocks; parameter blocks remain numeric.
    assert isinstance(parameter, float)
    assert jnp.asarray(parameter).shape == ()
