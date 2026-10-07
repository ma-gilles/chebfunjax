# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Mapped division supplemental controls; source hermpts.m, commit7574c77."""
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._gradual import gradual_positive_divide


def _exact(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.asarray([float(Fraction.from_float(float(x))/Fraction.from_float(float(y)))
                       for x,y in zip(a.flat,b.flat)]).reshape(a.shape)


@pytest.mark.parametrize("disabled", [True, False])
def test_mapped_mixed_ordinary_ranks(disabled):
    a = np.asarray([.75,1.25,2.5,1.5,2.5,5.]).reshape(2,3,1)
    b = np.asarray([[1.75,3.5,7.,14.], [2.25,4.5,9.,18.]])
    expected = _exact(a,b[:,None,:])
    with jax.disable_jit(disabled):
        got=jax.jit(jax.vmap(gradual_positive_divide,in_axes=(0,0)))(jnp.asarray(a),jnp.asarray(b))
    np.testing.assert_array_equal(np.asarray(got).view(np.uint64),expected.view(np.uint64))


@pytest.mark.parametrize("disabled", [True, False])
def test_mapped_subnormal_exact_fraction(disabled):
    # Actual failed numerator words plus independent least-subnormal controls.
    a=np.asarray([1,3,0x01bbf63226957,0x0420340c89c07,0x09cf50f5a8574,0x375d1a4d604a6,0x000fffffffffffff],dtype=np.uint64).view(np.float64)
    b=np.asarray(float.fromhex("0x1.cf5e8ca3a13d0p-9"))
    expected=_exact(a,b)
    with jax.disable_jit(disabled):
        got=jax.jit(jax.vmap(gradual_positive_divide,in_axes=(0,None)))(jnp.asarray(a),jnp.asarray(b))
    np.testing.assert_array_equal(np.asarray(got).view(np.uint64),expected.view(np.uint64))
