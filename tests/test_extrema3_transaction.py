"""Independent dyadic controls of native seed and paired Newton transaction."""
from types import SimpleNamespace

import jax.numpy as jnp

from chebfunjax.chebfun3d import _extrema3_source as source
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def polynomial(coefficients):
    basis=[Chebtech2.from_coeffs(jnp.eye(4)[:,i]) for i in range(4)]
    core=jnp.zeros((4,4,4))
    for index,value in coefficients.items():
        core=core.at[index].set(value)
    return Chebfun3(basis,basis,basis,core,(-1.,1.,-1.,1.,-1.,1.))


def inject(monkeypatch,points):
    count=iter(range(2))
    def solver(fun,initial,lower,upper,**kwargs):
        point=jnp.asarray(points[next(count)])
        return SimpleNamespace(x=point,value=fun(point))
    monkeypatch.setattr(source,'active_set_box',solver)


def test_f_order_seed_first_ties():
    f=polynomial({(1,0,0):1.})
    locations=source.source_seed(f)
    assert jnp.array_equal(locations,jnp.asarray([[-1.,-1.,-1.],[1.,-1.,-1.]]))


def test_boundary_skips_both_newton_steps(monkeypatch):
    f=polynomial({(1,0,0):1.})
    points=[[-1.,0.,0.],[.5,0.,0.]]
    inject(monkeypatch,points)
    def forbidden(*args,**kwargs):
        raise AssertionError('Native boundary transaction must skip both Hessians')
    monkeypatch.setattr(Chebfun3,'diff',forbidden)
    _,locations=f.minandmax3()
    assert jnp.array_equal(locations,jnp.asarray(points))


def test_singular_newton_restores_both(monkeypatch):
    f=polynomial({(0,0,0):.5,(2,0,0):.5})
    points=[[-.5,.125,0.],[.5,-.125,0.]]
    inject(monkeypatch,points)
    values,locations=f.minandmax3()
    assert jnp.array_equal(locations,jnp.asarray(points))
    assert jnp.array_equal(values,jnp.asarray([.25,.25]))


def test_outside_newton_restores_both(monkeypatch):
    # x^3-3*x+y^2+z^2; at x=.5 Newton moves to1.25, outsidebox.
    f=polynomial({(3,0,0):.25,(1,0,0):-2.25,(0,0,0):1.,
                  (0,2,0):.5,(0,0,2):.5})
    points=[[.5,.125,-.25],[-.5,-.125,.25]]
    inject(monkeypatch,points)
    values,locations=f.minandmax3()
    assert jnp.array_equal(locations,jnp.asarray(points))
    assert jnp.array_equal(values,jnp.asarray([f(*p) for p in points]))


def test_successful_paired_interior_newton(monkeypatch):
    # Sum of x_i^3-x_i. At +/-.5, grad=-.25 and Hessian=+/-3I.
    # A single native Newton update is therefore exactly +/-7/12.
    coefficients={}
    for axis in range(3):
        index=[0,0,0]
        index[axis]=3
        coefficients[tuple(index)]=.25
        index[axis]=1
        coefficients[tuple(index)]=-.25
    f=polynomial(coefficients)
    inject(monkeypatch,[[.5,.5,.5],[-.5,-.5,-.5]])
    values,locations=f.minandmax3()
    expected=jnp.asarray([[7/12]*3,[-7/12]*3])
    assert jnp.array_equal(locations,expected)
    # Independent rational cubic values;8eps covers the handful of rounded
    # polynomial operations, not a fitted native-output envelope.
    expected_values=jnp.asarray([-665/576,665/576])
    assert jnp.all(jnp.abs(values-expected_values)
                   <=8*jnp.finfo(jnp.float64).eps*jnp.abs(expected_values))


def test_source_grid_cap_threshold():
    for ndf,first_length in [(50000,130),(50001,129)]:
        observed=[]
        def factor(axis):
            def evaluate(x):
                observed.append((axis,x.size))
                return jnp.ones_like(x)
            return evaluate
        f=SimpleNamespace(length=lambda:(130,3,2),ndf=lambda:ndf,
                          domain=(-1.,1.,-1.,1.,-1.,1.),core=jnp.ones((1,1,1)),
                          cols=[factor(0)],rows=[factor(1)],tubes=[factor(2)])
        source.source_seed(f)
        assert observed==[(0,first_length),(1,3),(2,2)]
