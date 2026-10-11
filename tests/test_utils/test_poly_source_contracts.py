"""Source contracts for public polynomial coefficient conversion.

Provenance
----------
MATLAB source: @chebfun/poly.m, @bndfun/poly.m, @chebtech/poly.m,
@trigtech/poly.m; Chebfun commit 7574c77.
"""
import jax.numpy as jnp
import pytest

from chebfunjax import poly
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


def make(coeffs, interval=(-1.,1.)):
    return Chebfun(funs=[_Piece(tech=Chebtech2(coeffs=jnp.asarray(coeffs),ishappy=True),interval=interval)],domain=Domain(interval))

def test_small_stored_coefficient():
    out=poly(make([1.,0.,1e-15]))
    assert out.shape==(3,)
    assert jnp.array_equal(out,jnp.asarray([2e-15,0.,1.-1e-15]))

def test_array_coefficients():
    out=poly(make([[1.,2.],[0.,3.],[2.,4.]]))
    assert jnp.array_equal(out,jnp.asarray([[4.,0.,-1.],[8.,3.,-2.]]))

def test_piece_selection_and_warning():
    a=make([1.,2.],(-1.,0.))
    b=make([3.,4.],(0.,1.))
    f=Chebfun(funs=a.funs+b.funs,domain=Domain((-1.,0.,1.)))
    assert jnp.array_equal(poly(f,2),jnp.asarray([8.,-1.]))
    with pytest.warns(UserWarning,match='more than one FUN'):
        assert jnp.array_equal(poly(f),jnp.asarray([4.,3.]))

def test_complex_mapped_coefficients():
    out=poly(make([1.+2j,3.-1j,2j],(0.,2.)))
    assert jnp.array_equal(out,jnp.asarray([4j,3.-9j,-2.+5j]))

@pytest.mark.parametrize('n',[0,-1,1.5,[1,2],float('nan'),float('inf')])
def test_invalid_piece(n):
    with pytest.raises(ValueError,match='positive scalar integer'):
        poly(make([1.,2.]),n)

def test_empty_and_out_of_range():
    with pytest.raises(ValueError,match='only has 0 FUNS'):
        poly(Chebfun.empty())
    with pytest.raises(ValueError,match='only has 1 FUNS'):
        poly(make([1.,2.]),2)

def test_trig_coefficients_follow_native_poly():
    from chebfunjax.tech.trigtech import Trigtech
    c=jnp.asarray([1.+2j,3.-1j,2j])
    tech=Trigtech(coeffs=c,ishappy=True)
    f=Chebfun(funs=[_Piece(tech=tech,interval=(-1.,1.))],domain=Domain((-1.,1.)))
    assert jnp.array_equal(poly(f),c)

def test_constant_array_native_row():
    assert jnp.array_equal(poly(make([[2.,3.]])),jnp.asarray([[2.,3.]]))
