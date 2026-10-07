"""Independent scalar/array evaluation and complex linear boundary controls.

Provenance
----------
MATLAB source : @functionalBlock/functionalBlock.m (feval and mtimes),
    @operatorBlock/operatorBlock.m (diff), @linBlock/linBlock.m (toFunction),
    @linop/linsolve.m, @valsDiscretization/mldivide.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import D, I, eval_at
from chebfunjax.operators.chebop import _derivative_eval_at
from chebfunjax.operators.linop import Linop


def test_scalar_functional_return_types_and_derivative_action():
    dom=(-1.,1.)
    real=cj.chebfun(lambda x: 1+2*x, domain=dom)
    complex_fun=cj.chebfun(lambda x: (1+2j)*(1+x*x), domain=dom)
    value=eval_at(.25,dom).apply(real)
    assert isinstance(value,float) and abs(value-1.5)<1e-12
    value=eval_at(.25,dom).apply(complex_fun)
    assert isinstance(value,complex)
    assert abs(value-(1+2j)*1.0625)<1e-12
    assert abs(_derivative_eval_at(.25,dom,1).apply(complex_fun)-(1+2j)*.5)<1e-12


def test_array_functional_keeps_shape_and_complex_components():
    dom=(-1.,1.)
    u=cj.chebfun(lambda x:jnp.stack([1+x,2j*(1-x)],axis=-1),domain=dom)
    actual=eval_at(.25,dom).apply(u)
    assert actual.shape==(2,)
    assert jnp.iscomplexobj(actual)
    assert bool(jnp.all(jnp.abs(actual-jnp.asarray([1.25,1.5j]))<1e-12))


def test_complex_one_sided_functional():
    dom=Domain((-1.,0.,1.))
    u=Chebfun(funs=[_Piece.from_coeffs(jnp.asarray([1+2j]),-1.,0.),
                    _Piece.from_coeffs(jnp.asarray([3-4j]),0.,1.)],domain=dom)
    assert eval_at(0.,dom.breakpoints,direction='left').apply(u)==1+2j
    assert eval_at(0.,dom.breakpoints,direction='right').apply(u)==3-4j


@pytest.mark.parametrize('n',[None,24])
def test_complex_boundary_with_real_operator(n):
    dom=(-.5,.75)
    a,b=dom
    problem=Linop(D(dom,1),[eval_at(a,dom)],dom,[1+2j])
    assert problem.bc_values==[1+2j]
    u=problem.solve(2.,n=n)
    query=jnp.asarray([-.413,.071,.611])
    assert bool(jnp.all(jnp.abs(u(query)-(1+2j+2*(query-a)))<1e-10))
    assert (u.diff()-2).norm()<1e-10
    assert abs(u(a)-(1+2j))<1e-10


@pytest.mark.parametrize('n',[None,24])
def test_real_operator_complex_forcing_zero_boundaries(n):
    dom=(-.5,.75)
    a,b=dom
    u=Linop(D(dom,2),[eval_at(a,dom),eval_at(b,dom)],dom,[0.,0.]).solve(-2*(1+2j),n=n)
    query=jnp.asarray([-.413,.071,.611])
    expected=(1+2j)*(query-a)*(b-query)
    assert bool(jnp.all(jnp.abs(u(query)-expected)<1e-10))
    assert (u.diff(2)+2*(1+2j)).norm()<1e-10
    assert abs(u(a))<1e-10 and abs(u(b))<1e-10


@pytest.mark.parametrize('n',[None,24])
def test_complex_scalar_rhs_without_boundaries(n):
    u=Linop(I()).solve(2+3j,n=n)
    assert abs(u(.137)-(2+3j))<1e-12
    assert (u-(2+3j)).norm()<1e-12


@pytest.mark.parametrize('n',[None,24])
def test_complex_boundary_row_with_real_prescribed_value(n):
    # i*u(a)=1 implies u(a)=-i; u'=2 gives u(x)=2*(x-a)-i.
    dom=(-.5,.75)
    a,b=dom
    problem=Linop(D(dom,1),[1j*eval_at(a,dom)],dom,[1.])
    u=problem.solve(2.,n=n)
    query=jnp.asarray([-.413,.071,.611])
    assert bool(jnp.all(jnp.abs(u(query)-(2*(query-a)-1j))<1e-10))
    assert (u.diff()-2).norm()<1e-10
    assert abs(1j*u(a)-1)<1e-10


@pytest.mark.parametrize('order',[0,1,2,3])
@pytest.mark.parametrize('point',[-1.,.25,1.])
def test_derivative_functional_polynomial_actions(order,point):
    # Derivatives are independently expanded from x^3+2*x^2-x+4.
    amplitude=1+2j
    u=cj.chebfun(lambda x:amplitude*(x**3+2*x*x-x+4),domain=(-1.,1.))
    expected=[point**3+2*point*point-point+4,
              3*point*point+4*point-1,6*point+4,6.][order]*amplitude
    actual=_derivative_eval_at(point,(-1.,1.),order).apply(u)
    assert abs(actual-expected)<1e-10
