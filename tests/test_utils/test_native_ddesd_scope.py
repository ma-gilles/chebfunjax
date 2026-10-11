"""Explicit subset boundaries, no silent matrix ordering or precision casts."""

import jax.numpy as jnp
import pytest

from chebfunjax.utils.native_ddesd import ddesd


@pytest.mark.parametrize("target", ["delays", "history", "initial"])
def test_matrix_vectors_rejected(target):
    matrix = jnp.ones((2, 2))
    with pytest.raises(NotImplementedError):
        ddesd(
            lambda t, y, z: y,
            matrix if target == "delays" else [1.0],
            matrix if target == "history" else [1.0],
            [0.0, 1.0],
            {"InitialY": matrix} if target == "initial" else {},
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_lower_precision_rejected(dtype):
    with pytest.raises(NotImplementedError):
        ddesd(lambda t, y, z: y, [1.0], jnp.ones(1, dtype=dtype), [0.0, 1.0])


def test_matrix_absolute_tolerance_rejected():
    with pytest.raises(NotImplementedError):
        ddesd(lambda t, y, z: y, [1.], [1., 1., 1., 1.], [0., 1.],
              {'AbsTol': jnp.ones((2, 2))})
