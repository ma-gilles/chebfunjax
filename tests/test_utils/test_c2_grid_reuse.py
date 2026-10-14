"""Exact active grids and traced fallback for the CPU compile adapter."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.quadrature import _chebpts_generic, chebpts


@pytest.mark.parametrize("n", [0,1,2,3,14,17,33,64,65,129,257,513,1025,2049,4000,4096,4097,8193,16385,32768,32769])
def test_exact_active_grid(n):
    actual=np.asarray(chebpts(n,2))
    expected=np.asarray(_chebpts_generic(n,2))
    assert actual.shape==(n,)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))

@pytest.mark.parametrize("n", [2,17,129,4097])
def test_traced_fallback(n):
    actual=jax.jit(lambda scale: chebpts(n,2)*scale)(jnp.asarray(1.))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),np.asarray(_chebpts_generic(n,2)).view(np.uint64))


@pytest.mark.parametrize("n", [np.int32(17), np.int64(129)])
def test_integer_scalar_compatibility(n):
    np.testing.assert_array_equal(np.asarray(chebpts(n, 2)).view(np.uint64),
                                  np.asarray(_chebpts_generic(n, 2)).view(np.uint64))


def test_noninteger_count_still_rejected():
    with pytest.raises(TypeError):
        chebpts(17.5, 2)
