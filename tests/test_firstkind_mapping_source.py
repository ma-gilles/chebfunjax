"""First-kind public sampling: Chebfun 7574c77 mapping.m and bndfun.m."""
import struct

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import chebpts


def bits(values):
    return b''.join(struct.pack('=d',x) for x in jax.device_get(values).tolist())


@pytest.mark.parametrize('domain',[(-1.,1.),(-2.5,3.),(-7.,2.),(1.,4.)])
def test_firstkind_public_sample_mapping(domain):
    sampled=[]
    def function(x):
        if x.ndim==1 and x.size==17:
            sampled.append(x)
        return jnp.sin(x)
    chebfun(function,domain=domain,chebkind=1)
    reference=jax.device_get(chebpts(17,1)).tolist()
    a,b=domain
    expected=jnp.asarray(reference if domain==(-1.,1.) else [b*(y+1)/2+a*(1-y)/2 for y in reference])
    assert sampled
    assert bits(sampled[0])==bits(expected)


def test_firstkind_pde_initial_representation():
    for kind in [1,2]:
        f=chebfun(lambda x:jnp.sin(jnp.pi*x),domain=(-2.5,3.),chebkind=kind)
        assert len(f)==33
        assert len(f.simplify())==33
