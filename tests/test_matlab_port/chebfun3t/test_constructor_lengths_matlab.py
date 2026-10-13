"""Port MATLAB chebfun3t constructor length assertions 8–10.

Source: ``tests/chebfun3t/test_constructor.m`` at Chebfun commit 7574c77.
The public ``Chebfun3T.coeffs`` property reconstructs the dense coefficient
array, matching the MATLAB dimensions tested here.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3t import chebfun3t


def test_native_length_bounds_use_full_coefficient_dimensions():
    # MATLAB size(f.coeffs, 2:3) map to Python shape indices 1:2.
    f = chebfun3t(lambda x, y, z: jnp.sin(80 * x + y + z))
    shape = tuple(int(n) for n in f.coeffs.shape)
    print("CONSTRUCTOR_LENGTHS_PASS8_9", shape, flush=True)
    assert shape[1] < 50  # native pass(8)
    assert shape[2] < 50  # native pass(9)

    f1 = chebfun(lambda x: 1.0 / (1 + 25 * x**2))
    f2 = chebfun3t(lambda x, y, z: 1.0 / (1 + 25 * x**2))
    f2_shape = tuple(int(n) for n in f2.coeffs.shape)
    one_dimensional_length = len(f1)
    print("CONSTRUCTOR_LENGTH_PASS10", f2_shape, one_dimensional_length, flush=True)
    assert f2_shape[1] < one_dimensional_length + 20  # native pass(10)
