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


@pytest.mark.parametrize("domain,reference", [([-1,1], [1,.29,-.4137,-.254186]), ([0,3], [1,-.64,.442244444444444,-.271506962962963])])
def test_native_jacobi_values(domain,reference):
    actual=np.asarray(cj.jacpoly(np.arange(4),.1,-.3,domain)(.1)).ravel()
    assert np.linalg.norm(actual-np.asarray(reference),np.inf)<1e-14


def test_native_jacobi_selected_coefficients():
    actual=np.asarray(cj.jacpoly([0,2,3],-.25,-.25).chebcoeffs())
    true=np.array([[1,.109375,0],[0,0,.150390625],[0,.546875,0],[0,0,.451171875]])
    assert actual.shape==true.shape
    assert np.linalg.norm((actual-true).ravel())<1e-14
