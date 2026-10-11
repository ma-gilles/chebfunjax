"""Source linearize/solvebvpLinear coupled nonlocal routing controls.

Chebfun commit: 7574c77. Analytic fixtures are added controls, not native tests.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators import chebop_altdisc
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('delayed', [False, True])
def test_local_and_delayed_routes(monkeypatch, delayed):
    calls=[]
    original=chebop_altdisc._solve_linear_system_ad
    def observe(*args, **kwargs):
        calls.append((args[4:9], kwargs.get('data') is not None))
        return original(*args, **kwargs)
    monkeypatch.setattr(chebop_altdisc, '_solve_linear_system_ad', observe)
    N=Chebop(lambda t,x,y:[x.diff(), y.diff()-(x((t-.2).maximum(0.)) if delayed else x)],domain=(0.,.5,1.))
    N.lbc=jnp.array([1.,2.])
    x,y=N.solve(n=8,tol=1e-11,n_min=8,n_max=32)
    points=jnp.linspace(0.,1.,31)
    assert float(jnp.max(jnp.abs(x(points)-1.)))<100*jnp.finfo(float).eps
    assert float(jnp.max(jnp.abs(y(points)-(2.+points))))<100*jnp.finfo(float).eps
    assert bool(calls)==delayed
    if delayed:
        assert calls==[(('chebcolloc2',8,8,32,1e-11),True)]
