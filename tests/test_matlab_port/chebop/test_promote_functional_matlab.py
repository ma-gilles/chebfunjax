"""Strict source promote_functional slots1/2, with explicit C1 qualification.

Provenance
----------
MATLAB source: tests/chebop/test_promote_functional.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.

All ten original slots remain mapped in the package source_mapping.json.
Slots1/2 below retain continuous infinity norms and the original1e-10 bound.
C1 is an additional explicitly requested backend, not an extra original slot.
Unresolved slots3/5 check coupled linearize matrix row count8; slots4/6 check
coupled nonlinear continuous residuals; slots7/9 check periodic continuous
residuals and slots8/10 require returned trigtech. None is replaced by a proxy
or counted as qualified by this two-slot package. Native ultraS rejects the
finite-interval nonlocal equation; periodic coeffs means trigspec instead.
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
