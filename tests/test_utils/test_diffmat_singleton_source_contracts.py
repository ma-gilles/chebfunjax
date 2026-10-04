"""MATLAB singleton differentiation-matrix source contract.

Provenance
----------
MATLAB sources: ``diffmat.m``, ``@chebcolloc/baryDiffMat.m``,
``@chebcolloc1/chebcolloc1.m``, and ``@chebcolloc2/chebcolloc2.m``.
Chebfun commit: 7574c77.

The collocation source handles ``N == 1`` before its ``k == 0`` identity
case. Consequently, a singleton matrix is zero even for differentiation
order zero. For multiple nodes, order zero remains the identity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.diffmat import diffmat


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
@pytest.mark.parametrize("domain", [(-1.0, 1.0), (0.25, 3.75)])
def test_singleton_matrix_is_zero_and_annihilates_data(kind, order, domain):
    matrix = diffmat(1, order, domain=domain, kind=kind)

    assert matrix.shape == (1, 1)
    np.testing.assert_array_equal(np.asarray(matrix), np.zeros((1, 1)))
    np.testing.assert_array_equal(np.asarray(matrix @ jnp.array([7.0])), [0.0])


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_singleton_zero_is_jittable(kind, order):
    compiled = jax.jit(
        lambda endpoints: diffmat(1, order, domain=endpoints, kind=kind)
    )
    got = compiled(jnp.array([0.25, 3.75]))
    np.testing.assert_array_equal(np.asarray(got), np.zeros((1, 1)))


def test_singleton_zero_has_zero_domain_gradient():
    def output_width(endpoints):
        matrix = diffmat(1, 2, domain=endpoints, kind=2)
        return (matrix @ jnp.array([7.0]))[0]

    gradient = jax.grad(output_width)(jnp.array([0.25, 3.75]))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((2,)))


@pytest.mark.parametrize("kind", [1, 2])
def test_order_zero_remains_identity_for_multiple_nodes(kind):
    matrix = diffmat(3, 0, kind=kind)
    np.testing.assert_array_equal(np.asarray(matrix), np.eye(3))
