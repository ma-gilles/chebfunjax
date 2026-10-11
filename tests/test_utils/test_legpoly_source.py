"""Literal legpoly.m contracts, Chebfun 7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import legpoly


def test_empty():
    assert legpoly([],[-1,.66,1],'normalize').isempty()

@pytest.mark.parametrize('method',[1,2,3])
def test_explicit_methods(method):
    f=legpoly([0,1,2],[-1,1],0,method)
    x=jnp.linspace(-1,1,11)
    want=jnp.stack([jnp.ones_like(x),x,(3*x*x-1)/2],axis=1)
    assert jnp.max(jnp.abs(f(x)-want))<50*jnp.finfo(float).eps

def test_column_orientation():
    f=legpoly(jnp.asarray([[1],[2]]))
    assert f.is_transposed and f.size(1)==2

def test_method_in_normalized_short_form():
    f=legpoly(2,'normalize',1)
    assert abs(float(f(1.))-jnp.sqrt(2.5))<50*jnp.finfo(float).eps

def test_method3_native_normalization_on_mapped_domain():
    f=legpoly(1,[0,4],'normalize',3)
    assert abs(float(f(4.))-jnp.sqrt(1.5))<50*jnp.finfo(float).eps
