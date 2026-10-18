"""Independent mathematical and IEEE controls for the JAX dense provider."""
import math
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators import _qz_ieee as binary
from chebfunjax.operators._eig_tridiagonal import eigh_tridiagonal
from chebfunjax.operators.blocklinop import _geig


@pytest.mark.parametrize('complex_storage', [False, True])
def test_finite_tiny_mass_and_higher_index_infinity(complex_storage):
    dtype = np.complex128 if complex_storage else np.float64
    values, _ = _geig(jnp.asarray(np.diag([1., 2.]).astype(dtype)),
                      jnp.asarray(np.diag([1., 1e-18]).astype(dtype)))
    np.testing.assert_allclose(np.sort(np.asarray(values).real), [1., 2e18], rtol=2e-14)
    # det(I-lambda*N)=1: a regular index-three infinite block, not a singular pencil.
    A = np.eye(3, dtype=dtype)
    B = np.diag(np.ones(2), 1).astype(dtype)
    values, vectors = _geig(jnp.asarray(A), jnp.asarray(B))
    assert np.all(np.isinf(np.asarray(values).real))
    assert np.all(np.asarray(values).imag == 0)
    assert np.linalg.norm(B @ np.asarray(vectors)) < 2e-14


@pytest.mark.parametrize('complex_storage', [False, True])
def test_hermitian_residual_and_b_normalization(complex_storage):
    rng = np.random.default_rng(728)
    X = rng.normal(size=(4, 4))
    if complex_storage:
        X = X + 1j * rng.normal(size=(4, 4))
    # Mirror explicitly: general GEMM can introduce tiny asymmetry and change dispatch.
    A = np.tril(X, -1) + np.tril(X, -1).conj().T + np.diag(X.diagonal().real)
    B = np.diag([1., 2., 3., 4.]).astype(A.dtype)
    values, vectors = _geig(jnp.asarray(A), jnp.asarray(B))
    values, vectors = np.asarray(values), np.asarray(vectors)
    assert np.linalg.norm(A @ vectors - (B @ vectors) * values) < 2e-14
    assert np.linalg.norm(vectors.conj().T @ B @ vectors - np.eye(4)) < 2e-14


def test_empty_invalid_and_explicit_nonconvergence():
    values, vectors = _geig(jnp.empty((0, 0)), jnp.empty((0, 0)))
    assert values.shape == (0,) and vectors.shape == (0, 0)
    for value in [np.inf, np.nan]:
        with pytest.raises(ValueError, match='infs or NaNs'):
            _geig(jnp.array([[value]]), jnp.ones((1, 1)))
    with pytest.raises(ValueError, match='equal square'):
        _geig(jnp.ones((2, 3)), jnp.eye(2))
    with pytest.raises(RuntimeError):
        eigh_tridiagonal(jnp.array([1., 2., 3.]), jnp.array([.7, .8]), maxiter=0)


def test_jitted_gradual_arithmetic_and_ieee_specials():
    # The public dispatcher is host-controlled; fixed-shape arithmetic kernels are JIT-able.
    x = np.array([0x0000000000000001, 0x000012688b70e62b,
                  0x0010000000000000, 0x3ff0000000000000], np.uint64).view(np.float64)
    y = np.array([2., 2., .5, 2.])
    for op, reference in [(binary.multiply, np.multiply), (binary.divide, np.divide),
                          (binary.add, np.add)]:
        actual = np.asarray(jax.jit(op)(jnp.asarray(x), jnp.asarray(y)))
        expected = reference(x, y)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    actual = np.asarray(jax.jit(binary.multiply)(jnp.array([np.inf, -np.inf, np.nan]),
                                               jnp.array([0., 2., 1.])))
    assert np.isnan(actual[0]) and np.isneginf(actual[1]) and np.isnan(actual[2])


def _scalar_word(word):
    return struct.unpack(">d", struct.pack(">Q", word))[0]


@pytest.mark.parametrize("complex_storage", [False, True])
def test_scalar_positive_mass_extremes(complex_storage):
    dtype = np.complex128 if complex_storage else np.float64
    minimum = _scalar_word(1)
    maximum = _scalar_word(0x7FEFFFFFFFFFFFFF)
    for a, b, expected in [
        (1.0, minimum, math.inf),
        (0.0, minimum, 0.0),
        (minimum, minimum, 1.0),
        (-3.0, 1.0, -3.0),
        (1.0, maximum, 1.0 / maximum),
        (maximum, maximum, 1.0),
    ]:
        values, vectors = _geig(jnp.asarray([[a]], dtype=dtype), jnp.asarray([[b]], dtype=dtype))
        values, vectors = np.asarray(values), np.asarray(vectors)
        assert not np.iscomplexobj(values) and not np.iscomplexobj(vectors)
        v = float(vectors[0, 0])
        assert math.isfinite(v) and v > 0
        assert abs((v * b) * v - 1) < 2e-14
        assert abs(v / (1 / math.sqrt(b)) - 1) < 2e-14
        if math.isinf(expected):
            assert values[0] == expected
        elif expected == 0:
            assert values[0] == 0
        else:
            assert abs(float(values[0]) / expected - 1) < 2e-14
