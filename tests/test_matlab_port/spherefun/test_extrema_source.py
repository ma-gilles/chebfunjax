"""Source zero and CDR seed controls for real Spherefun extrema.

Provenance: @spherefun/minandmax2.m and @chebfun2/outerProduct.m,
Chebfun7574c77. These controls supplement the ten original predicates.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d._extrema_fallback import original_objective
from chebfunjax.spherefun._extrema_source import _reconstruct, _seed
from chebfunjax.spherefun.spherefun import Spherefun


def test_zero_uses_source_domain_center():
    f = Spherefun.from_function(lambda l,t: jnp.zeros_like(l+t))
    y,x=f.minandmax2()
    np.testing.assert_array_equal(y,[0,0])
    np.testing.assert_array_equal(x,[[0,0],[0,0]])


def test_outer_product_retains_source_cdr_and_unit_pivots():
    f=Spherefun.from_function(lambda l,t: jnp.cos(l)*jnp.sin(t)+jnp.cos(t))
    g,rows,cols=_reconstruct(f)
    np.testing.assert_array_equal(g.approx.pivots,np.ones(len(f.cols)))
    x=jnp.asarray([-.71,.15,1.3])
    y=jnp.asarray([.37,-1.1,2.4])
    np.testing.assert_allclose(g(x,y),f(x,y),atol=20*np.finfo(float).eps,rtol=0)
    values,locs=_seed(rows,cols,original_objective(g.approx))
    np.testing.assert_allclose(values,jnp.asarray([g(*locs[0]),g(*locs[1])]),atol=20*np.finfo(float).eps,rtol=0)
