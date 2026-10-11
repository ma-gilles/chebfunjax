"""Complex Horner contracts.

Provenance
----------
MATLAB source: @chebfun/polyval.m; Chebfun commit 7574c77.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun, polyval


@pytest.mark.parametrize('transpose',[False,True])
def test_complex_coefficients(transpose):
    x=chebfun('x',domain=(-2.,3.))
    if transpose:
        x=x.T
    y=polyval(jnp.asarray([1.+2j,3.-1j,-2j]),x)
    expected=(1+2j)*x*x+(3-1j)*x-2j
    assert y.is_transposed==transpose
    assert (y-expected).norm(jnp.inf)<100*jnp.finfo(jnp.float64).eps*expected.vscale

def test_complex_matrix():
    x=chebfun('x')
    y=polyval(jnp.asarray([[1.+2j,2.-1j],[3j,-2.]]),x)
    assert len(y)==2
    assert (y[0]-((1+2j)*x+3j)).norm(jnp.inf)<1e-14
    assert (y[1]-((2-1j)*x-2)).norm(jnp.inf)<1e-14

def test_scalar_constant():
    x=chebfun('x')
    assert (polyval(2j,x)-2j).norm(jnp.inf)==0

def test_array_argument_with_matrix_rejected():
    x=chebfun(lambda t:jnp.column_stack((t,t*t)))
    with pytest.raises(ValueError,match='dimMismatch'):
        polyval(jnp.ones((2,2)),x)
