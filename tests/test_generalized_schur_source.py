"""Independent reconstruction controls for the source generalized Schur kernel.

Provenance
----------
MATLAB source: @chebop2/bartelsStewart.m (QZ consumer)
Chebfun commit: 7574c77
Algorithm source: LAPACK ZGGHRD/ZHGEQZ; controls include singular pencils.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._generalized_schur import _qz_eigenvalue_factors, qz


@pytest.mark.parametrize("n", [0, 1, 2, 3, 9, 10, 11])
def test_dense_and_singular_pencil_reconstruction(n):
    rng = np.random.default_rng(918273 + n)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    if n == 0:
        h, t, q, z, info, _ = qz(jnp.asarray(a), jnp.asarray(b))
        assert int(info) == 0
        assert h.shape == t.shape == q.shape == z.shape == (0, 0)
        return
    cases = [(a, b), (a, np.zeros_like(b)), (np.zeros_like(a), np.zeros_like(b))]
    for j in range(n):
        bb = np.triu(b).copy()
        bb[j, j] = 0
        cases.append((np.triu(a, -1), bb))
    bound = 128 * n * np.finfo(float).eps
    for aa, bb in cases:
        h, t, q, z, info, _ = qz(jnp.asarray(aa), jnp.asarray(bb))
        h, t, q, z = map(np.asarray, (h, t, q, z))
        assert int(info) == 0
        assert np.linalg.norm(q @ h @ z.conj().T - aa, 2) / max(np.linalg.norm(aa, 2), 1.0) < bound
        assert np.linalg.norm(q @ t @ z.conj().T - bb, 2) / max(np.linalg.norm(bb, 2), 1.0) < bound
        assert np.linalg.norm(q.conj().T @ q - np.eye(n), 2) < bound
        assert np.linalg.norm(z.conj().T @ z - np.eye(n), 2) < bound
        assert np.count_nonzero(np.tril(h, -1)) == np.count_nonzero(np.tril(t, -1)) == 0


def test_zero_second_rotation_argument_preserves_first_exactly():
    from chebfunjax.utils._generalized_schur import rotation

    values = np.array(
        [0.0, 0.3 + 0.7j, -0.23 + 0.91j, 1e-200 + 2e-200j, 1e200 - 2e200j], dtype=np.complex128
    )
    c, s, r = rotation(jnp.asarray(values), jnp.zeros(values.shape, dtype=jnp.complex128))
    np.testing.assert_array_equal(c, np.ones(values.shape))
    np.testing.assert_array_equal(s, np.zeros(values.shape))
    np.testing.assert_array_equal(np.asarray(r).view(np.uint64), values.view(np.uint64))


def test_nearly_repeated_resultant_pencil_converges():
    """Retain the original roots04 pencil that exhausted the old QZ shifts."""
    from pathlib import Path

    with np.load(Path(__file__).parent / "references/qz_resultant_sine_pencil.npz") as data:
        a, b = data["a"], data["b"]
    h, t, q, z, info, _ = qz(jnp.asarray(a), jnp.asarray(b))
    h, t, q, z = map(np.asarray, (h, t, q, z))
    assert int(info) == 0
    # Same reconstruction bound as the existing dense-pencil controls.
    bound = 128 * a.shape[0] * np.finfo(float).eps
    assert np.linalg.norm(q @ h @ z.conj().T - a, 2) / np.linalg.norm(a, 2) < bound
    assert np.linalg.norm(q @ t @ z.conj().T - b, 2) / np.linalg.norm(b, 2) < bound
    assert np.linalg.norm(q.conj().T @ q - np.eye(a.shape[0]), 2) < bound
    assert np.linalg.norm(z.conj().T @ z - np.eye(a.shape[0]), 2) < bound
    assert np.count_nonzero(np.tril(h, -1)) == np.count_nonzero(np.tril(t, -1)) == 0


@pytest.mark.parametrize(
    "a,b",
    [
        (np.empty((0, 0), complex), np.empty((0, 0), complex)),
        (np.array([[2. + 3.j]]), np.array([[4. - 1.j]])),
        (np.array([[2.]]), np.zeros((1, 1))),  # infinite generalized eigenvalue
        (np.zeros((1, 1)), np.zeros((1, 1))),  # indeterminate 0/0 pair
        (
            np.array([[2., 1., 0.], [0., 3., 2.], [1., 0., 4.]], complex),
            np.array([[1., 0., .2], [.1, 2., 0.], [0., .3, 1.]], complex),
        ),
        (
            np.array([[2., 1., 0.], [0., 3., 2.], [1., 0., 4.]], complex),
            np.diag(np.array([1., 0., 0.])),  # singular pencil
        ),
    ],
    ids=["empty", "scalar-finite", "scalar-infinite", "scalar-indeterminate", "dense", "singular-dense"],
)
def test_eigenvalue_only_qz_matches_full_factor_path(a, b):
    """The value-only path retains H/T bits and status without Q/Z work."""
    full_h, full_t, full_q, full_z, full_info, full_sweeps = qz(jnp.asarray(a), jnp.asarray(b))
    h, t, info, sweeps = _qz_eigenvalue_factors(jnp.asarray(a), jnp.asarray(b))

    # Exact equality makes this a same-arithmetic control, not a relaxed
    # eigenvalue comparison; existing full-factor reconstruction tests below
    # continue to verify Q/Z and are left unchanged.
    for actual, expected in ((h, full_h), (t, full_t)):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape == a.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    assert int(info) == int(full_info) == 0
    assert int(sweeps) == int(full_sweeps)
    assert np.asarray(full_q).shape == np.asarray(full_z).shape == a.shape
