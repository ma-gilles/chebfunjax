"""Native ACA2 rounding, Chebfun pin7574c77, MATLAB R2025b.

105 completeACA2D steps captured on classic grids17/23 and BTD compression.
Inputs and expected residuals are exact native arrays, not fitted or generated
from the Python implementation. Native-stage provenance accompanies fixture.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d._chebfun3double import _subtract2


@pytest.mark.parametrize('outer_jit', [False, True])
def test_native_105_residual_steps(outer_jit):
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        pytest.skip('Native rounding qualification is CPU binary64')
    update = jax.jit(_subtract2) if outer_jit else _subtract2
    with np.load(Path(__file__).parent/'fixtures/aca2_native_steps.npz') as data:
        for k in range(105):
            matrix = jnp.asarray(data[f'input_{k}'])
            row, col = data[f'pivot_{k}']
            actual = np.asarray(update(matrix, row, col))
            np.testing.assert_array_equal(actual.view(np.uint64), data[f'expected_{k}'].view(np.uint64))


def test_shared_binary_compatibility_identity():
    from chebfunjax.chebfun1d._pde_ndf.source_arithmetic import binary as pde
    from chebfunjax.diskfun._rank_one import binary as disk
    from chebfunjax.utils._ordered_binary import binary
    assert pde is disk is binary


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.complex128])
def test_existing_generic_dtype_arithmetic(dtype):
    matrix = jnp.asarray([[1., 2., 3.], [2., 4., 6.]], dtype=dtype)
    actual = _subtract2(matrix, 0, 0)
    expected = matrix-matrix[:, 0, None]*(matrix[0, :]/matrix[0, 0])[None, :]
    np.testing.assert_array_equal(actual, expected)


def test_source_update_derivative():
    matrix = jnp.asarray([[2., 3.], [5., 7.]])
    tangent = jnp.asarray([[1., 0.], [0., 0.]])
    _, derivative = jax.jvp(lambda a: _subtract2(a, 0, 0), (matrix,), (tangent,))
    np.testing.assert_array_equal(derivative, [[0., 0.], [0., 15./4]])
