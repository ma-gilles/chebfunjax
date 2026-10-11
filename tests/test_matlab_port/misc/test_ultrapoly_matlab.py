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
from chebfunjax.utils.quadrature import jacpts


@pytest.mark.parametrize("n,lam",[(10,2.1),(11,1.9)])
def test_native_ultraspherical_roots(n,lam):
    actual=np.asarray(cj.ultrapoly(n,lam).roots()).ravel()
    exact=np.asarray(jacpts(n,lam-.5,lam-.5)[0])
    assert np.linalg.norm(np.sort(actual)-np.sort(exact))<1e-14


@pytest.mark.parametrize("lam,bound",[(1.9,1e-10),(4.,1e-9)])
def test_native_ultraspherical_jacobi_ratio(lam,bound):
    x=np.linspace(-1,1,100)
    u=cj.ultrapoly(np.arange(16),lam)
    j=cj.jacpoly(np.arange(16),lam-.5,lam-.5)
    ratio=np.asarray(u(x))/np.asarray(j(x))
    assert np.linalg.norm(ratio-np.ones((100,1))*ratio[0,:],2)<bound


def test_native_ultraspherical_selected_coefficients():
    actual=np.asarray(cj.ultrapoly([0,2,3],.25).chebcoeffs())
    true=np.array([[1,.0625,0],[0,0,.078125],[0,.3125,0],[0,0,.234375]])
    assert actual.shape==true.shape
    assert np.linalg.norm((actual-true).ravel())<1e-14
