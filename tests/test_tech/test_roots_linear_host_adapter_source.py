"""Dtype/shape/ownership contract at the inherited roots recursion boundary.

The n=2 source arithmetic is JAX. A writable host array remains the adapter
expected by the existing recursive caller and all-root entry point.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("all_roots", [False, True])
def test_linear_host_adapter_dtype_shape_and_ownership(dtype, all_roots):
    for root in [.25, 2.]:
        coeffs = jax.device_get(jnp.asarray([-root, 1.], dtype=dtype))
        result = module._roots_main(coeffs, 100*float(jnp.finfo(jnp.float64).eps),
                                    all_roots=all_roots)
        expected_dtype = jnp.dtype(dtype if all_roots else jnp.real(jnp.zeros((), dtype=dtype)).dtype)
        expected = jnp.asarray([root] if all_roots or root == .25 else [], dtype=expected_dtype)
        assert type(result).__module__ == "numpy"
        assert type(result).__name__ == "ndarray"
        assert result.shape == expected.shape
        assert result.dtype == expected_dtype
        assert result.flags.writeable and result.flags.owndata
        assert bool(jnp.array_equal(jnp.asarray(result), expected))
        if result.size:
            # A byte/data adapter must preserve the previous writable return.
            result[0] = root
