"""Actual source denominator rounding and finite transform controls.

Provenance
----------
MATLAB source : ratinterp.m/ratbary, Chebfun commit7574c77
Source stage archive: e8ae12fe7fada1d1a5e49c4be6baed3f5a373fa233f9b3c61ea2cc36594fb3a1
"""
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils._ratbary import _source_denominator_sum


def test_source_cancellation_positive_zero_eager_jit():
    # Independently written literal inputs to source qxw*dxqinv. MATLAB
    # recorded products[-.5,.5], denominator+0; not a fitted answer input.
    qx=jnp.array([1.,3.])
    wq=jnp.array([.5,-.5])
    inverse=jnp.array([[-1.,-1/3]])
    for evaluate in (_source_denominator_sum,jax.jit(_source_denominator_sum)):
        value=np.asarray(evaluate(qx,wq,inverse))
        np.testing.assert_array_equal(value,np.zeros(1))
        assert not np.signbit(value[0])


def test_denominator_helper_batched_finite_reverse_derivative():
    qx=jnp.array([1.,3.])
    wq=jnp.array([.5,-.5])
    inverse=jnp.array([[-1.,-.25],[.5,.25]])
    batch=jax.jit(jax.vmap(lambda row:_source_denominator_sum(qx,wq,row)))
    np.testing.assert_allclose(batch(inverse),inverse@(qx*wq),rtol=0,atol=100*np.finfo(float).eps)
    gradient=jax.jit(jax.grad(lambda row:_source_denominator_sum(qx,wq,row)))(inverse[0])
    np.testing.assert_array_equal(gradient,qx*wq)
