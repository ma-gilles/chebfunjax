"""Source domain/polyfit contracts.

Provenance
----------
MATLAB source: @domain/polyfit.m; Chebfun commit 7574c77.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import polyfit


@pytest.mark.parametrize('transpose',[False,True])
def test_complex_data(transpose):
    x=jnp.linspace(-2.,3.,17)
    y=jnp.column_stack(((1+2j)*x*x+3j,(2-1j)*x+4))
    f=polyfit(x,y.T if transpose else y,2,(-2.,3.))
    assert jnp.max(jnp.abs(f(x)-y))<1e-13

def test_matrix_abscissae_rejected():
    with pytest.raises(ValueError,match='column vector'):
        polyfit(jnp.ones((2,2)),jnp.ones(4),2)

def test_zero_degree_source_second_column():
    x=jnp.asarray([-1.,0.,1.])
    y=2*x+3
    f=polyfit(x,y,0)
    assert jnp.max(jnp.abs(f(x)-y))<1e-14


def test_complex_underdetermined_basic_solution():
    x=jnp.asarray([-.7,.5])
    y=jnp.asarray([1.+2j,3.-1j])
    f=polyfit(x,y,3)
    assert jnp.max(jnp.abs(f(x)-y))<100*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(y))
    assert jnp.count_nonzero(f.funs[0].coeffs)<=2
