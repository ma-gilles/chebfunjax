"""Native linearize parameter/linearity exclusions and numeric BC adapters."""
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_numeric_bc_alternative_system(backend):
    N=Chebop(lambda t,u,v:[u.diff(),v.diff()],domain=(0.,1.))
    N.lbc=jnp.array([1.,2.])
    u,v=N.solve(n=9,discretization=backend)
    x=jnp.linspace(0.,1.,11)
    assert float(jnp.max(jnp.abs(u(x)-1)))<100*jnp.finfo(float).eps
    assert float(jnp.max(jnp.abs(v(x)-2)))<100*jnp.finfo(float).eps

@pytest.mark.parametrize('kind',['nonlinear','parameter','fewer_equations'])
def test_piecewise_exclusion_retains_existing_route(monkeypatch,kind):
    ops={'nonlinear':lambda t,u,v:[u.diff()-u(v),v.diff()-v*v],
         'parameter':lambda t,u,p:[u.diff()-u(t/2)-p,u-p],
         'fewer_equations':lambda t,u,p:u.diff()-p}
    N=Chebop(ops[kind],domain=(0.,.5,1.))
    N.lbc=[1.,1.]
    token=object()
    monkeypatch.setattr(Chebop,'_solve_impl',lambda self,*a,**k:token)
    monkeypatch.setattr(Chebop,'_simplify_solution',lambda self,value:value)
    assert N.solve() is token
