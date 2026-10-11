"""Native polynomial predicates and source controls.

Provenance
----------
MATLAB sources: tests/misc/test_jacpoly.m, tests/misc/test_ultrapoly.m.
Chebfun commit: 7574c77.
All original numerical bounds retained; fresh MATLAB execution not asserted.
"""
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("name", ["jacpoly", "ultrapoly"])
def test_selected_orientation_and_piecewise_domain(name):
    args=(.1,-.3) if name=="jacpoly" else (1.9,)
    constructor=getattr(cj,name)
    degrees=np.array([[3],[0],[3]])
    row=constructor(degrees,*args,[0,1,3])
    col=constructor(degrees.ravel(),*args,[0,3])
    assert row.is_transposed and row.n_columns==3
    assert np.array_equal(row.domain.breakpoints,[0,1,3])
    assert np.max(np.abs(np.asarray(row(.4)).ravel()-np.asarray(col(.4)).ravel()))<1e-14
    with pytest.raises(ValueError,match="infdomain"):
        constructor([0,1],*args,[-np.inf,np.inf])


@pytest.mark.parametrize("lam", [.5,1.])
def test_ultraspherical_special_delegates(lam):
    x=np.linspace(-1,1,11)
    actual=np.asarray(cj.ultrapoly(2,lam)(x)).ravel()
    expected=(3*x*x-1)/2 if lam==.5 else 4*x*x-1
    assert np.max(np.abs(actual-expected))<1e-14


def test_ultraspherical_parameter_error():
    with pytest.raises(ValueError,match="ultrapoly:infdomain"):
        cj.ultrapoly(2,0)

