"""Literal16 source predicates; legacy fixed-grid/field adapters are separate.

MATLAB source: tests/chebop/test_paramODE_linearization.m, lines1–101.
Chebfun commit:7574c77680d7e82b79626300bf255498271a72df.

Each case retains its natural operator, boundary conditions, source initial
state (only case5), default adaptive solve and1e-10bound. Python lists adapt
native block columns; block norm uses the native square root of the sum of
squared block2-norms. Endpoint feval maps over those blocks. Numeric parameters
remain numeric; no function evaluation converts a field into a parameter.
The eight numeric type predicates are independent from the residual predicates.
"""
import numbers

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


def _norm(value):
    """Native numeric2-norm or continuous Chebfun/block2-norm."""
    if isinstance(value, (list, tuple)):
        return jnp.sqrt(sum(_norm(block)**2 for block in value))
    if isinstance(value, Chebfun):
        return value.norm(2)
    return jnp.linalg.norm(jnp.asarray(value).reshape(-1))


def _feval(value, point):
    """Native block feval at the literal physical endpoint."""
    if isinstance(value, (list, tuple)):
        return [_feval(block, point) for block in value]
    return cj.feval(value, point) if isinstance(value, Chebfun) else value


@pytest.fixture(scope="module", params=range(1, 9), ids=lambda case: f"case{case}")
def source_case(request):
    case = request.param
    domain = (0., 1.) if case in (3, 4, 5) else (-1., 1.)
    x = cj.chebfun(lambda t: t, domain=domain)
    if case in (1, 2):
        operator = Chebop(lambda x, u, c: u.diff(), domain)
        if case == 1:
            operator.bc = lambda x, u, c: [u(-1.), u(0.) - c]
        else:
            operator.bc = lambda x, u, c: u(0.) - c
            operator.lbc = lambda u, c: u
        rhs = cj.chebfun(jnp.exp)
    elif case in (3, 4, 5):
        operator = Chebop(lambda x, u, p: u.diff(2) + u + p, domain)
        if case == 4:
            operator.lbc = lambda u, p: [u, u.diff() - p]
            operator.rbc = lambda u, p: u - 2
        else:
            if case == 5:
                operator.op = lambda x, u, p: u.diff(2) + u
                operator.init = [0*x, 0.]
            operator.bc = lambda x, u, p: [u(0.), u(1.) - 2, u.diff()(0.) - p]
        rhs = 0.
    else:
        if case in (6, 7):
            operator = Chebop(lambda x, u, p: u.diff() - 1, domain)
        else:
            operator = Chebop(lambda x, u, p: u.diff() - p, domain)
        if case == 6:
            operator.lbc = lambda u, p: u
            operator.rbc = lambda u, p: u - p
        elif case == 7:
            operator.bc = lambda x, u, p: [u(-1.), u(1.) - p]
        else:
            operator.bc = lambda x, u, p: [u(-1.), u(1.) - 2]
        rhs = 0.
    # Literal default adaptive public solve: no n, tolerance or discretization override.
    solution = operator.solve(rhs)
    return case, operator, x, rhs, solution[0], solution[1]


def test_native_residual_aggregate(source_case):
    case, operator, x, rhs, u, parameter = source_case
    error = _norm(operator(u, parameter) - rhs)
    if case in (1, 2, 3, 5, 7, 8):
        error += _norm(operator.bc(x, u, parameter))
    if case in (2, 4, 6):
        error += _norm(_feval(operator.lbc(u, parameter), x.domain.a))
    if case in (4, 6):
        error += _norm(_feval(operator.rbc(u, parameter), x.domain.b))
    assert float(error) < TOL, f"native pass(1,{case}): err={float(error):.17g}"


def test_native_numeric_parameter_type(source_case):
    case, _, _, _, u, parameter = source_case
    numeric = (isinstance(parameter, numbers.Number)
               and not isinstance(parameter, (bool, np.bool_))) or (
        isinstance(parameter, (jax.Array, np.ndarray))
        and jnp.issubdtype(parameter.dtype, jnp.number))
    assert isinstance(u, Chebfun) and numeric, (
        f"native pass(2,{case}): u={type(u).__name__}, parameter={type(parameter).__name__}")
