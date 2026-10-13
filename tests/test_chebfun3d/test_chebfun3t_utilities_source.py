"""Focused source controls for dense Chebfun3T utility methods."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import Chebfun3T, chebfun3t


def test_norm_matches_literal_native_power_sum_and_empty_contract():
    real = Chebfun3T.from_coeffs(jnp.array([[[2.0]]]))
    np.testing.assert_allclose(real.norm(), np.sqrt(32.0))

    imaginary = Chebfun3T.from_coeffs(jnp.array([[[2.0j]]]))
    result = imaginary.norm()
    assert jnp.iscomplexobj(result)
    np.testing.assert_allclose(result, 4.0j * np.sqrt(2.0))

    assert Chebfun3T.empty().norm().shape == (0,)
    with pytest.raises(ValueError) as error:
        real.norm("inf")
    assert error.value.identifier == "CHEBFUN:CHEBFUN3T:norm:norm"


def test_length_returns_native_maximum_dimensions_and_three_output_form():
    f = Chebfun3T.from_coeffs(jnp.ones((2, 4, 3)))
    assert f.length() == 4
    assert f.length(outputs=3) == (2, 4, 3)
    assert Chebfun3T.empty().length().shape == (0,)
    assert all(v.shape == (0,) for v in Chebfun3T.empty().length(outputs=3))


def test_public_transform_wrappers_roundtrip_and_mode_contraction():
    values = jnp.arange(24.0).reshape((2, 3, 4))
    coeffs = Chebfun3T.vals2coeffs(values)
    restored = Chebfun3T.coeffs2vals(coeffs)
    np.testing.assert_allclose(restored, values, atol=2e-14, rtol=2e-14)

    unfolded_native_scalar = Chebfun3T.unfold(values, 1)
    folded_native_scalar = Chebfun3T.fold(
        unfolded_native_scalar, values.shape, 1, (2, 3)
    )
    np.testing.assert_array_equal(folded_native_scalar, values)

    unfolded = Chebfun3T.unfold(values, (1, 3))
    folded = Chebfun3T.fold(unfolded, values.shape, (1, 3))
    np.testing.assert_array_equal(folded, values)

    matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    contracted = Chebfun3T.txm(values, matrix, 2)
    assert contracted.shape == (2, 2, 4)
    np.testing.assert_array_equal(contracted[:, 1, :], values[:, 1, :] + values[:, 2, :])

    matrix2d = jnp.arange(6.0).reshape((2, 3))
    values2d = Chebfun3T.coeffs2vals(Chebfun3T.vals2coeffs(matrix2d))
    assert values2d.shape == (2, 3, 1)
    np.testing.assert_allclose(values2d[..., 0], matrix2d, atol=2e-14, rtol=2e-14)

    mode3 = Chebfun3T.txm(matrix2d, jnp.array([[1.0], [2.0]]), 3)
    assert mode3.shape == (2, 3, 2)
    np.testing.assert_array_equal(mode3[..., 0], matrix2d)
    np.testing.assert_array_equal(mode3[..., 1], 2 * matrix2d)

    with pytest.raises(ValueError) as error:
        Chebfun3T.txm(matrix2d, jnp.ones((2, 2)), 3)
    assert error.value.identifier == "CHEBFUN:CHEBFUN3:txm"


def test_native_four_dimensional_unfold_fold_and_txm_examples():
    tensor = jnp.arange(2 * 3 * 4 * 5.0).reshape((2, 3, 4, 5))
    unfolded = Chebfun3T.unfold(tensor, (2, 4))
    assert unfolded.shape == (15, 8)
    restored = Chebfun3T.fold(unfolded, tensor.shape, (2, 4))
    np.testing.assert_array_equal(restored, tensor)

    tensor = jnp.arange(3 * 5 * 2 * 4.0).reshape((3, 5, 2, 4))
    matrix = jnp.arange(18.0).reshape((6, 3))
    result = Chebfun3T.txm(tensor, matrix, 1)
    assert result.shape == (6, 5, 2, 4)
    expected = matrix @ tensor.reshape((3, -1), order="F")
    np.testing.assert_array_equal(
        result.reshape((6, -1), order="F"), expected
    )


def test_norm_constant_callable_remains_available_for_compose_predicate():
    f = chebfun3t(lambda x, y, z: jnp.ones_like(x))
    assert float(f.norm()) == pytest.approx(np.sqrt(8.0))
