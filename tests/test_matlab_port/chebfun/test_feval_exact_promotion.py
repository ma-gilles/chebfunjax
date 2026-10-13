"""IEEE single-to-double evaluation promotion and its existing AD contract.

MATLAB source: @chebfun/feval.m and @classicfun/feval.m, commit7574c77.
CPU compilation must not flush binary32 inputs before double evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain


def linear(width, coefficient=1.0):
    return Chebfun(
        funs=[_Piece.from_coeffs(jnp.asarray([0.0, coefficient]), -width, width)],
        domain=Domain((-width, width)),
    )


def queries(complex_input):
    values = np.array([1, 0x80000001, 0x7FFFFF, 0x807FFFFF], np.uint32).view(np.float32)
    if complex_input:
        z = np.zeros(values.shape, np.complex64)
        z.real = values
        z.imag = values[::-1]
        return z
    return values


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("width", [1e-37, 1e-40])
def test_jit_value_promotion_preserves_subnormal_inputs(complex_input, width):
    f = linear(width)
    x = queries(complex_input) if width == 1e-37 else queries(complex_input)[:2]
    expected = np.asarray(f(x.astype(np.complex128 if complex_input else np.float64)))
    np.testing.assert_allclose(
        jax.jit(lambda z: f(z))(jnp.asarray(x)), expected, rtol=0, atol=4 * np.finfo(float).eps
    )


@pytest.mark.parametrize("width", [1.0, 1e-40])
def test_float32_identity_autodiff(width):
    f = linear(width, width)
    x = jnp.asarray(np.array(1, np.uint32).view(np.float32))
    assert jax.jit(jax.grad(f))(x) == 1
    value, tangent = jax.jit(lambda z: jax.jvp(f, (z,), (jnp.ones_like(z),)))(x)
    assert tangent == 1


def test_complex64_identity_autodiff():
    f = linear(1e-40, 1e-40)
    x = jnp.asarray(queries(True)[0])
    t = jnp.asarray(1 + 2j, jnp.complex64)
    _, tangent = jax.jit(lambda z: jax.jvp(f, (z,), (t,)))(x)
    np.testing.assert_allclose(tangent, 1 + 2j, rtol=0, atol=4 * np.finfo(float).eps)
    assert jax.jit(jax.grad(f, holomorphic=True))(x) == 1 + 0j


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("pieces", [1, 2])
def test_fused_evaluation_promotion(complex_input, pieces):
    width = 1e-37
    if pieces == 1:
        f = linear(width)
    else:
        f = Chebfun(
            funs=[
                _Piece.from_coeffs(jnp.asarray([-0.5, 0.5]), -width, 0.0),
                _Piece.from_coeffs(jnp.asarray([0.5, 0.5]), 0.0, width),
            ],
            domain=Domain((-width, 0.0, width)),
        )
    x = queries(complex_input)
    promoted = x.astype(np.complex128 if complex_input else np.float64)
    expected = np.asarray(f(promoted)) + 1
    result = jax.jit(lambda z: f(z) + (z + 1))(jnp.asarray(x))
    np.testing.assert_allclose(result, expected, rtol=0, atol=4 * np.finfo(float).eps)


def test_widening_preserves_ieee_words():
    from chebfunjax.chebfun1d.chebfun import _widen_float32

    words = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x807FFFFF,
            0x800000,
            0x80800000,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
        ],
        np.uint32,
    )
    x = words.view(np.float32)
    expected = x.astype(np.float64).view(np.uint64)
    for result in [_widen_float32(jnp.asarray(x)), jax.jit(_widen_float32)(jnp.asarray(x))]:
        assert result.dtype == jnp.float64 and result.shape == x.shape
        np.testing.assert_array_equal(np.asarray(result).view(np.uint64), expected)


def test_existing_normal_point_and_second_derivative_contract():
    f = linear(1.0)
    x = jnp.asarray(0.25, jnp.float32)
    assert jax.jit(jax.grad(f))(x) == 1
    assert jax.jit(jax.grad(jax.grad(f)))(x) == 0
