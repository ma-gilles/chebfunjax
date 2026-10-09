"""Literal Chebfun 7574c77 tests/chebop/test_scalarODE_damping.m.

Source explicitly sets bvpTol=1e-13 for Chebcolloc2/Chebcolloc1 and1e-12
for ultraS. Previous controls are retained separately with their old prefs.
"""
import math

import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('discretization', ['chebcolloc2', 'chebcolloc1', 'ultraS'])
def test_original_scalar_ode_damping(discretization):
    pref_bvp_tol = 1e-12 if discretization == 'ultraS' else 1e-13
    op = Chebop(lambda x, u: .05*u.diff(2)+(5*x).cos()*u.sin(), (0., math.pi))
    op.lbc = lambda u: u-2
    op.rbc = lambda u: u-3
    u, _ = op.solvebvp(0., tol=pref_bvp_tol, discretization=discretization)
    assert float(op(u).norm()) < 1e-9
