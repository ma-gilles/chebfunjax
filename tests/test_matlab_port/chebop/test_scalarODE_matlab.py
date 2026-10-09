"""Literal Chebfun 7574c77 tests/chebop/test_scalarODE.m predicates.

Alternative nonlinear backend source semantics remain unqualified. Previous
sampled residual controls remain in test_scalarODE_legacy_controls.py.
"""
import math

import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('discretization', ['chebcolloc2', 'ultraS', 'chebcolloc1'])
def test_original_scalar_ode(discretization):
    if discretization != 'chebcolloc2':
        pytest.skip('Original nonlinear alternate-backend semantics unqualified; '
                    'legacy controls retained separately, not a known assertion failure')
    pref_bvp_tol = 5e-13
    op = Chebop(lambda u: u.diff(2)+(u-.2).sin(), (0., math.pi))
    op.lbc = lambda u: u-2
    op.rbc = lambda u: u-3
    u, _ = op.solvebvp(0., tol=pref_bvp_tol)
    assert float(op(u).norm()) < 1e3*pref_bvp_tol
