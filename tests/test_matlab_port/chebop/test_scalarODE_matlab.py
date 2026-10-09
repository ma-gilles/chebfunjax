"""Literal Chebfun 7574c77 tests/chebop/test_scalarODE.m predicates.

Previous sampled residual controls remain separately in
test_scalarODE_legacy_controls.py. Every source clause requests its backend.
"""
import math

import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('discretization', ['chebcolloc2', 'ultraS', 'chebcolloc1'])
def test_original_scalar_ode(discretization):
    pref_bvp_tol = 5e-13
    op = Chebop(lambda u: u.diff(2)+(u-.2).sin(), (0., math.pi))
    op.lbc = lambda u: u-2
    op.rbc = lambda u: u-3
    u, _ = op.solvebvp(0., tol=pref_bvp_tol, discretization=discretization)
    assert float(op(u).norm()) < 1e3*pref_bvp_tol
