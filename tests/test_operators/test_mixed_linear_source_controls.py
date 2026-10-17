"""Natural scalar parameters: original aggregate tests and analytic controls.

Provenance
----------
MATLAB source: tests/chebop/test_paramODE.m, @chebop/linearize.m,
@linop/linsolve.m. Chebfun commit:7574c77.
Independent manufactured controls of numeric columns and their ordering.
Original aggregate predicates are in the mapped test_paramODE_matlab.py.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('parameter_first', [False, True])
@pytest.mark.parametrize('complex_rhs', [False, True])
def test_analytic_numeric_column_order_and_complex_values(parameter_first, complex_rhs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Natural parameter solve used augmented square field adapter')
    monkeypatch.setattr(Chebop, '_solve_linear_system', forbidden)
    slope = 1+2j if complex_rhs else 1.
    if parameter_first:
        operator = Chebop(lambda x, p, u: u.diff()-slope)
        operator.lbc = lambda p, u: u
        operator.rbc = lambda p, u: u-p
        p, u = operator.solve(0.)
    else:
        operator = Chebop(lambda x, u, p: u.diff()-slope)
        operator.lbc = lambda u, p: u
        operator.rbc = lambda u, p: u-p
        u, p = operator.solve(0.)
    assert isinstance(u, Chebfun) and isinstance(p, (float, complex))
    assert abs(p-2*slope) < 1e-12
    assert (u-slope*(cj.chebfun(lambda t:t)+1)).norm(jnp.inf) < 1e-12
