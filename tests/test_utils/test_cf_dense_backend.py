"""Manufactured exact signed-eigenpair controls for the large CF backend.

Provenance
----------
MATLAB source: @chebfun/cf.m polynomialCF/getBlock largest-magnitude ordering.
Chebfun commit: 7574c77.
These diagonal controls supplement eight literal native public CF clauses and
saved 4000-square Pushnitski evidence; they do not assert native eigs bits.
"""

import jax.numpy as jnp

from chebfunjax.utils._cf_kernel import _largest_abs_eigs_dense_candidate


def check_signed_diagonal(size):
    diagonal = jnp.zeros(size, dtype=jnp.float64).at[:3].set(jnp.array([-7.0, 5.0, -3.0]))
    matrix = jnp.diag(diagonal)
    values, vectors = _largest_abs_eigs_dense_candidate(matrix, 3)
    assert values.dtype == vectors.dtype == jnp.float64
    assert values.shape == (3,) and vectors.shape == (size, 3)
    assert jnp.array_equal(values, jnp.array([-7.0, 5.0, -3.0]))
    assert jnp.array_equal(matrix @ vectors, vectors * values)
    assert jnp.array_equal(vectors.T @ vectors, jnp.eye(3))


def test_signed_eigenpair_selection_small():
    check_signed_diagonal(4)


def test_signed_eigenpair_selection_large():
    check_signed_diagonal(1025)
