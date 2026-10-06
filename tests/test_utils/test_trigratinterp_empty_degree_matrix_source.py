"""Source matrix-shape control for robustified empty Fourier coefficients.

Provenance
----------
MATLAB source: trigratinterp.m, construct_matrices and trig_rat_interp.
Chebfun commit: 7574c77
The MATLAB assignment P(:,1)=1 (and Q(:,1)=1) expands an N-by-0
allocation to N-by-1 when a chopped coefficient vector has degree -0.5.
This controls matrix construction only; it does not qualify the full retry
path for empty coefficients.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import ratapprox as candidate


@pytest.mark.parametrize(
    ("m", "n"),
    [(-0.5, 0), (0, -0.5), (-0.5, -0.5), (1, -0.5), (-0.5, 2)],
)
def test_empty_chopped_degree_keeps_source_constant_matrix_column(m, n):
    th = jnp.asarray([-1.0, -0.25, 0.4], dtype=jnp.float64)
    P, Q = candidate._trigrat_construct_matrices(th, m, n)

    p_cols = max(1, int(2 * m + 1))
    q_cols = max(1, int(2 * n + 1))
    expected_p = np.zeros((th.size, p_cols))
    expected_q = np.zeros((th.size, q_cols))
    expected_p[:, 0] = 1.0
    expected_q[:, 0] = 1.0
    for j in range(1, int(m) + 1):
        expected_p[:, 2 * j - 1] = np.sin(np.pi * j * np.asarray(th))
        expected_p[:, 2 * j] = np.cos(np.pi * j * np.asarray(th))
    for j in range(1, int(n) + 1):
        expected_q[:, 2 * j - 1] = np.sin(np.pi * j * np.asarray(th))
        expected_q[:, 2 * j] = np.cos(np.pi * j * np.asarray(th))

    assert P.shape == expected_p.shape
    assert Q.shape == expected_q.shape
    np.testing.assert_array_equal(np.asarray(P)[:, 0], np.ones(th.size))
    np.testing.assert_array_equal(np.asarray(Q)[:, 0], np.ones(th.size))
    eps = np.finfo(float).eps
    np.testing.assert_allclose(np.asarray(P), expected_p, rtol=4*eps, atol=4*eps)
    np.testing.assert_allclose(np.asarray(Q), expected_q, rtol=4*eps, atol=4*eps)
