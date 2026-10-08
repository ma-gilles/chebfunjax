"""Source default lagpts method threshold; lagpts.m110-126, Chebfun7574c77."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import quadrature


@pytest.mark.parametrize("n,alpha,method", [(300,12.,"gw"),(1000,12.,"gw"),(2999,12.,"gw"),(3000,12.,"rh"),(3000,50.,"rh"),(10000,500.,"rh"),(3000,0.,"rh")])
def test_literal_source_default_threshold(monkeypatch,n,alpha,method):
    seen=[]
    def core(n,a,interval,m):
        seen.append(m)
        return jnp.asarray([1.]),jnp.asarray([1.])
    monkeypatch.setattr(quadrature,"_lagpts_core",core)
    quadrature.lagpts(n,alpha)
    assert seen==[method]

def test_alpha12_default_matches_explicit_source_rh():
    x,w,v=map(np.asarray,quadrature.lagpts(3000,12.,bary=True))
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(np.diff(x)>0) and np.all(w>=0)
    np.testing.assert_allclose([w@x**k for k in range(5)], [math.gamma(13+k) for k in range(5)],rtol=2e-9,atol=0)
    for got,want in zip(quadrature.lagpts(3000,12.,bary=True,method="RH"),(x,w,v),strict=True):
        np.testing.assert_array_equal(got,want)

def test_default_preserves_source_newton_failure():
    with pytest.raises(jax.errors.JaxRuntimeError,match="MATLAB lagpts RH Newton convergence guard"):
        quadrature.lagpts(3000,20.3)[0].block_until_ready()
