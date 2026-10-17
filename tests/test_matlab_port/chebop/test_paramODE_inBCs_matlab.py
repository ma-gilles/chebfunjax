"""Four literal predicates from native test_paramODE_inBCs.m.

Provenance
----------
MATLAB source: tests/chebop/test_paramODE_inBCs.m, Chebfun7574c77.
Original operators, boundary data, default adaptive solves and10*bvpTol bound.
The source two-point column vector [0;2] is a scalar sampled-value Chebfun;
it is retained as two data values, without a callable line replacement.
Inherited fixed-grid controls remain in the separate legacy companion.
"""
import numbers

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


def _norm(value):
    """Native continuous2-norm for a Chebfun, numeric2-norm otherwise."""
    return value.norm(2) if isinstance(value, Chebfun) else jnp.linalg.norm(jnp.asarray(value).reshape(-1))


@pytest.mark.parametrize('case', range(1, 5), ids=lambda n: f'source_clause_{n:02d}')
def test_literal_parameter_in_boundary_conditions(case, record_property):
    tol = 10*ChebopPref().bvpTol
    if case == 1:
        domain = (0., 1.)
        x = cj.chebfun(lambda t: t, domain=domain)
        operator = Chebop(lambda x, u, p: u.diff(2)+u, domain)
        bc = lambda x, u, p: [u(0.), u(1.)-2, u.diff()(0.)-p]
        operator.bc = bc
        u, parameter = operator.solve(0.)
        residual = operator(x, u, parameter)
        error = _norm(residual)+_norm(bc(x, u, parameter))
    else:
        if case in (2, 3):
            operator = Chebop(lambda t, u, p: u.diff()-1, (-1., 1.))
        else:
            operator = Chebop(lambda t, u, p: u.diff()-p, (-1., 1.))
        if case == 2:
            operator.lbc = lambda u, p: u
            operator.rbc = lambda u, p: u-p
        elif case == 3:
            operator.bc = lambda t, u, p: [u(-1.), u(1.)-p]
        else:
            operator.bc = lambda t, u, p: [u(-1.), u(1.)-2]
        u, parameter = operator.solve(0.)
        line = cj.chebfun(jnp.asarray([0., 2.]))
        error = _norm(u-line)+_norm(parameter-(1 if case == 4 else 2))
    record_property('source_clause', case)
    record_property('source_error', float(error))
    record_property('source_tolerance', float(tol))
    record_property('actual_parameter_type', type(parameter).__name__)
    record_property('actual_parameter_is_numeric', isinstance(parameter, numbers.Number))
    assert float(error) < tol
