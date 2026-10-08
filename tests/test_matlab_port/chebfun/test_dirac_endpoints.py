"""Endpoint half-strength from pinned @chebfun/dirac.m 7574c77.

No standalone MATLAB dirac test exists in the pinned tree; these additional
source-derived controls use the existing delta integration tolerance 1e-13.
"""
import numpy as np
import pytest

from chebfunjax import chebfun


@pytest.mark.parametrize('domain,root,slope,mass', [
    ((0,1),0,1,.5), ((0,1),1,1,.5), ((0,1),.5,1,1.),
    ((-3,2),-3,-2,.25), ((-3,2),2,4,.125), ((-3,2),0,4,.25)])
def test_source_half_strength(domain,root,slope,mass):
    f=chebfun(lambda x:slope*(x-root),domain=domain)
    d=f.dirac()
    assert float(d.sum())==pytest.approx(mass,rel=0,abs=1e-13)
    assert d._delta_weights==pytest.approx([mass],rel=0,abs=1e-13)
    x=chebfun('x',domain=domain)
    assert float((x*d).sum())==pytest.approx(root*mass,rel=0,abs=1e-13)

def test_both_endpoints_quadratic():
    f=chebfun(lambda x:x*x-1,domain=(-1,1))
    d=f.dirac()
    np.testing.assert_allclose([v[1] for v in d.deltas],[.25,.25],rtol=0,atol=1e-13)
    assert float(d.sum())==pytest.approx(.5,rel=0,abs=1e-13)

def test_digital_payoff_distribution():
    x=chebfun('x',domain=(0,1))
    prob=.37
    pdf=2*(1-prob)*x.dirac()+2*prob*(x-1).dirac()
    assert float(pdf.sum())==pytest.approx(1,rel=0,abs=1e-13)
    assert float((x*pdf).sum())==pytest.approx(prob,rel=0,abs=1e-13)

def test_dirac_endpoint_derivative():
    x=chebfun('x',domain=(0,1))
    assert x.dirac(1).deltas==((0.,.5,1),)
    assert float(x.dirac(1).sum())==0.
