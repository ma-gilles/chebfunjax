"""Native scalar-index metadata used by adaptive trig construction.

MATLAB source: @trigtech/populate.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech, _trig_probe_mask


def test_indexed_complex_zero_and_tiny_nonzero():
    assert _trig_probe_mask(jnp.asarray([[1+0j, 2j, 1+1e-200j]])) == (True, False, False)


def test_original_native_diff_input_metadata():
    f=Trigtech.from_function(lambda x:jnp.stack([jnp.exp(-50*x**2),jnp.sin(4*jnp.pi*(x-.2)),1j*jnp.exp(jnp.cos(jnp.pi*x))],axis=-1))
    assert f.real_columns == (True,True,False)
    assert f.diff(1,dim=2).real_columns == (True,False)
    assert f.diff(2,dim=2).real_columns == (False,)
