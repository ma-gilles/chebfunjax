"""Literal scalar-x callback route for direct integral actions.

Provenance
----------
MATLAB source: @chebfun/fred.m and @chebfun/volt.m.
Chebfun commit: 7574c77.
Manufactured constants use 10eps; native Gaussian bounds are separate.
"""
import jax.numpy as jnp

import chebfunjax as cj


def scalar_x_kernel(x,y):
 assert jnp.ndim(x)==0
 return jnp.ones_like(y)

def test_fred_scalar_x_source_callback():
 f=cj.chebfun(lambda x:jnp.ones_like(x))
 result=cj.fred(scalar_x_kernel,f)
 assert jnp.max(jnp.abs(result(jnp.array([-.5,0.,.5]))-2))<10*jnp.finfo(jnp.float64).eps

def test_volt_scalar_x_source_callback_and_endpoints():
 f=cj.chebfun(lambda x:jnp.ones_like(x))
 result=cj.volt(scalar_x_kernel,f)
 x=jnp.array([-1.,-.5,0.,.5,1.])
 assert result(-1.)==0
 assert jnp.max(jnp.abs(result(x)-(x+1)))<10*jnp.finfo(jnp.float64).eps
