"""Independent parity-split and tracing controls for the source solver.

Provenance
----------
MATLAB source: @chebop2/bartelsStewart.m
Chebfun commit: 7574c77
These well-conditioned controls supplement the unchanged original assertions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop2 import bartels_stewart


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (4, 3)])
@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("complex_data", [False, True])
def test_parity_splits_and_jit(shape, flags, complex_data):
    m, n = shape
    xsplit, ysplit = flags

    def pencil(n, split, phase):
        i = np.arange(1, n + 1)
        off = 0.05 * np.sin(i[:, None] + 2 * i[None, :] + phase)
        if complex_data:
            off = off + 0.03j * np.cos(i[:, None] - i[None, :] + phase)
        if split:
            off = np.where((i[:, None] - i[None, :]) % 2 == 0, off, 0)
        return np.diag(2 + i * 0.2) + off

    A, C = pencil(m, ysplit, 0), pencil(m, ysplit, 1)
    B, D = pencil(n, xsplit, 2), pencil(n, xsplit, 3)
    X = np.sin(np.arange(m * n).reshape(m, n) + 0.4)
    if complex_data:
        X = X + 0.2j * np.cos(np.arange(m * n).reshape(m, n) + 0.7)
    E = A @ X @ B.T + C @ X @ D.T
    solve = jax.jit(lambda a, b, c, d, e: bartels_stewart(a, b, c, d, e, xsplit, ysplit))
    actual = np.asarray(solve(*map(jnp.asarray, (A, B, C, D, E))))
    assert np.linalg.norm(actual - X, 2) < 128 * np.finfo(float).eps * np.linalg.norm(X, 2)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_rhs(shape):
    m, n = shape
    out = bartels_stewart(jnp.eye(m), jnp.eye(n), jnp.eye(m), jnp.eye(n), jnp.zeros(shape))
    assert out.shape == shape


@pytest.mark.parametrize(
    "scales", [(1e-200, 1e200), (1e200, 1e-200), (1e-300, 1e300), (1e300, 1e-300)]
)
def test_qz_extreme_independent_scales(scales):
    from chebfunjax.utils._generalized_schur import qz

    a = np.array([[2 + 0.3j, 0.2 - 0.1j, -0.1], [0.4j, 3, 0.1], [0.2, -0.3j, 4]])
    b = np.array([[3, -0.2j, 0.1], [0.3, 2 + 0.2j, -0.2], [0.1j, 0.1, 2]])
    sa, sb = scales
    h, t, q, z, info, _ = qz(jnp.asarray(a * sa), jnp.asarray(b * sb))
    h, t, q, z = map(np.asarray, (h, t, q, z))
    assert int(info) == 0
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(t))
    bound = 128 * len(a) * np.finfo(float).eps
    # Normalize factors before products/norms to avoid overflow in the oracle.
    assert np.linalg.norm(q @ (h / sa) @ z.conj().T - a, 2) / np.linalg.norm(a, 2) < bound
    assert np.linalg.norm(q @ (t / sb) @ z.conj().T - b, 2) / np.linalg.norm(b, 2) < bound
    assert np.linalg.norm(q.conj().T @ q - np.eye(len(a)), 2) < bound
    assert np.linalg.norm(z.conj().T @ z - np.eye(len(a)), 2) < bound


@pytest.mark.parametrize("complex_data", [False, True])
def test_implicit_jvp_vjp_and_tiny_tangent(complex_data):
    A = np.array([[3.0, 0.1], [-0.2, 2.0]])
    C = np.array([[1.0, -0.1], [0.2, 1.5]])
    B = np.array([[2.0, 0.2, -0.1], [0.1, 3.0, 0.2], [0.1, -0.2, 2.5]])
    D = np.array([[1.0, 0.1, 0.2], [-0.1, 1.5, 0.1], [0.2, 0.1, 1.2]])
    E = np.arange(6).reshape(2, 3) * 0.1 + 0.3
    matrices = [A, B, C, D, E]
    if complex_data:
        matrices = [
            x + 0.05j * np.cos(np.arange(x.size).reshape(x.shape) + i)
            for i, x in enumerate(matrices)
        ]
    A, B, C, D, E = matrices
    tangents = [
        0.02 * np.sin(np.arange(x.size).reshape(x.shape) + 0.4 + i) for i, x in enumerate(matrices)
    ]
    if complex_data:
        tangents = [x + 0.01j * np.cos(x) for x in tangents]
    dA, dB, dC, dD, dE = tangents
    K = np.kron(B, A) + np.kron(D, C)

    def dense(rhs, transpose=False):
        return np.linalg.solve(K.T if transpose else K, rhs.ravel(order="F")).reshape(
            E.shape, order="F"
        )

    X = dense(E)
    expected = dense(dE - dA @ X @ B.T - A @ X @ dB.T - dC @ X @ D.T - C @ X @ dD.T)
    values = tuple(map(jnp.asarray, matrices))
    directions = tuple(map(jnp.asarray, tangents))
    primal, tangent = jax.jvp(bartels_stewart, values, directions)
    np.testing.assert_allclose(primal, X, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(tangent, expected, rtol=1e-12, atol=1e-14)
    _, tiny = jax.jvp(bartels_stewart, values, tuple(x * 1e-20 for x in directions))
    np.testing.assert_allclose(tiny, expected * 1e-20, rtol=1e-12, atol=1e-34)
    G = np.cos(np.arange(6).reshape(2, 3)) + 0.1j
    _, pullback = jax.vjp(bartels_stewart, *values)
    gradients = pullback(jnp.asarray(G))
    L = dense(G, True)
    expected_gradients = [-L @ B @ X.T, -L.T @ A @ X, -L @ D @ X.T, -L.T @ C @ X, L]
    for actual, oracle in zip(gradients, expected_gradients):
        np.testing.assert_allclose(
            actual, oracle if complex_data else oracle.real, rtol=1e-12, atol=1e-14
        )


def test_implicit_second_derivative():
    A = np.array([[3.0, 0.2], [0.1, 2.0]])
    H = np.array([[0.1, -0.2], [0.05, 0.1]])
    B = np.array([[2.0, 0.1], [-0.2, 3.0]])
    C = np.eye(2)
    D = np.array([[1.0, 0.1], [0.2, 1.0]])
    E = np.array([[1.0, 2.0], [-0.2, 0.7]])
    K = np.kron(B, A) + np.kron(D, C)
    dK = np.kron(B, H)
    x = np.linalg.solve(K, E.ravel(order="F"))
    expected = np.sum(2 * np.linalg.solve(K, dK @ np.linalg.solve(K, dK @ x)))

    def loss(t):
        return jnp.real(
            jnp.sum(
                bartels_stewart(
                    jnp.asarray(A) + t * jnp.asarray(H),
                    jnp.asarray(B),
                    jnp.asarray(C),
                    jnp.asarray(D),
                    jnp.asarray(E),
                )
            )
        )

    np.testing.assert_allclose(jax.grad(jax.grad(loss))(0.0), expected, rtol=1e-12, atol=1e-14)
