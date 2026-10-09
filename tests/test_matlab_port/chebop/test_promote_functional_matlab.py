"""Strict source promote_functional slots1–6, with explicit C1 qualification.

Provenance
----------
MATLAB source: tests/chebop/test_promote_functional.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.

All ten original slots remain mapped in the package source_mapping.json.
Slots1/2 below retain continuous infinity norms and the original1e-10 bound.
C1 is an additional explicitly requested backend, not an extra original slot.
Slots3–6 below preserve the two separate coupled source objects and explicit
init only on the first object. Slots7–10 remain unresolved periodic residual
and trigtech predicates. Native ultraS rejects finite-interval nonlocal
equations; periodic coeffs means trigspec instead.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('backend', ['chebcolloc2', 'chebcolloc1'])
@pytest.mark.parametrize('clause', [1, 2], ids=['source_clause_01', 'source_clause_02'])
def test_literal_scalar_functional_promotion(clause, backend):
    if clause == 1:
        N = Chebop(lambda u: u.diff(2)+u.sum())
    else:
        N = Chebop(lambda u: u.diff(2)+u*u.sum())
    N.lbc, N.rbc = 0., 0.
    u = N.solve(1., discretization=backend)
    assert (N(u)-1).norm(jnp.inf) < 1e-10


def _coupled_source(problem):
    if problem == 1:
        N = Chebop(lambda x, u, v: [u.diff(), v.diff()])
        N.bc = lambda x, u, v: [u(0)-1, u(0)*v(.5)]
        N.init = [1, 0]
    else:
        N = Chebop(lambda x, u, v: [u.diff(), v.diff()+v*u.sum()+u(.3)])
        N.bc = lambda x, u, v: [u(0)-1, u(0)*v(.5)]
    return N


@pytest.mark.parametrize('problem', [1, 2], ids=['source_clause_03', 'source_clause_05'])
def test_literal_coupled_matrix(problem):
    N = _coupled_source(problem)
    L, _, _ = N.linearize()
    assert L.matrix(3).shape[0] == 8


@pytest.mark.parametrize('backend', ['chebcolloc2', 'chebcolloc1'])
@pytest.mark.parametrize('problem', [1, 2], ids=['source_clause_04', 'source_clause_06'])
def test_literal_coupled_solve(problem, backend):
    N = _coupled_source(problem)
    u = N.solve(1, discretization=backend)
    assert max((r-1).norm(jnp.inf) for r in N(*u)) < 1e-10
    # Independent controls, additional to the literal source residual.
    from chebfunjax.chebfun1d.chebfun import chebfun
    x = chebfun(lambda t: t)
    assert max(abs(value) for value in N.bc(x, *u)) < 1e-10
    exact = [x+1, x-.5 if problem == 1 else
             chebfun(lambda t: .15*(jnp.exp(1-2*t)-1))]
    assert max((a-b).norm(jnp.inf) for a, b in zip(u, exact)) < 1e-10
