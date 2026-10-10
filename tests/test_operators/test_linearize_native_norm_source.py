"""Literal mathematical clauses 1--5 of tests/chebop/test_linearize.m.

Native continuous norms and tolerance 1e-14 are retained. The existing Python
mirror (unchanged) instead uses a 41-point maximum and 1e-8. Python's no-state
information triple and one-column ChebMatrix are explicit output adapters;
the source paramReshapeFlag=false argument in clause3 has no public keyword.
"""
import math

import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop


@pytest.fixture(scope='module')
def native_data():
    dom = (0., 1., math.pi)
    x = chebfun(lambda t: t, domain=dom)
    return dom, x, x.sin(), x.cos(), (-x).exp()


@pytest.mark.parametrize('clause', [1, 2, 3, 4, 5])
def test_native_continuous_norm_clause(native_data, clause):
    dom, x, u, v, w = native_data
    if clause == 1:
        op = Chebop(lambda u: u.diff(), domain=dom)
        operator, _, _ = op.linearize()
        residual = (operator*u)[0]-u.diff()
    elif clause == 2:
        op = Chebop(lambda u: u.diff(2)+u.diff()+x.sin()*u, domain=dom)
        operator, _, _ = op.linearize()
        residual = (operator*u)[0]-(u.diff(2)+u.diff()+x.sin()*u)
    elif clause == 3:
        op = Chebop(lambda u: u**2, domain=dom)
        residual = op.linearize(u)*v-2*u*v
    elif clause == 4:
        op = Chebop(lambda u: u.diff(2)+u**2, domain=dom)
        residual = op.linearize(u)*v-(v.diff(2)+2*u*v)
    else:
        op = Chebop(lambda x, u, v: [u.diff(2)+v**2, v.diff()-u.cos()], domain=dom)
        actual = op.linearize([u, v])*[w, v]
        residual = actual-ChebMatrix([[w.diff(2)+2*v**2], [v.diff()+u.sin()*w]])
    error = float(residual.norm())
    assert error < 1e-14, (clause, error)
