"""Source contracts for per-column Chebtech boundary-root extraction.

Provenance
----------
MATLAB source : @chebtech/extractBoundaryRoots.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

The independent dense solve below is test-only. Production extraction uses
the source upper-band recurrence in JAX and does not allocate a dense matrix.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    _extract_boundary_roots,
)

EPS = np.finfo(np.float64).eps


def _left_factor_times_q(q):
    """Chebyshev coefficients of ``(1+x)*q`` for a quadratic q."""
    a0, a1, a2 = q
    return jnp.asarray(
        [a0 + 0.5 * a1, a0 + a1 + 0.5 * a2,
         0.5 * a1 + a2, 0.5 * a2]
    )


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_boundary_tolerance_uses_each_column_scale(Tech):
    # First column is 1e12*(1+x), second is a nonzero 1e-6 constant.
    # A single array-global tolerance would classify the small constant's
    # endpoints as roots; MATLAB vscale(arrayTech) supplies one scale per col.
    coeffs = jnp.asarray([[1e12, 1e-6], [1e12, 0.0]])
    f = Tech(coeffs=coeffs, ishappy=True)
    large_scale = (
        2e12 if Tech is Chebtech2
        else 1e12 * (1.0 + 1.0 / np.sqrt(2.0))
    )
    np.testing.assert_allclose(
        np.asarray(f.vscale_columns), [large_scale, 1e-6], rtol=2e-15
    )

    g, left, right = f.extractBoundaryRoots()
    np.testing.assert_array_equal(np.asarray(left), [1, 0])
    np.testing.assert_array_equal(np.asarray(right), [0, 0])
    x = jnp.asarray([-0.75, -0.1, 0.8])
    expected = jnp.stack([jnp.full_like(x, 1e12), jnp.full_like(x, 1e-6)], axis=-1)
    np.testing.assert_allclose(np.asarray(g(x)), np.asarray(expected), rtol=2e-14, atol=0.0)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_boundary_tolerance_opposite_endpoint_uses_each_column_scale(Tech):
    # Same scale contrast, with the large column's root at +1 instead.
    coeffs = jnp.asarray([[1e12, 1e-6], [-1e12, 0.0]])
    f = Tech(coeffs=coeffs, ishappy=True)
    g, left, right = f.extractBoundaryRoots()
    np.testing.assert_array_equal(np.asarray(left), [0, 0])
    np.testing.assert_array_equal(np.asarray(right), [1, 0])
    x = jnp.asarray([-0.75, -0.1, 0.8])
    expected = jnp.stack([jnp.full_like(x, 1e12), jnp.full_like(x, 1e-6)], axis=-1)
    np.testing.assert_allclose(np.asarray(g(x)), np.asarray(expected), rtol=2e-14, atol=0.0)


def test_backward_band_recurrence_matches_independent_dense_source_matrix():
    # Source D has diagonal (1,.5,.5), first superdiagonal +1, second
    # superdiagonal .5. Build an exact left factor so the explicit source path
    # extracts one boundary root, then compare its quotient with dense D\rhs.
    q = np.asarray([2.0 + 1.0j, 0.5 - 0.2j, -0.25 + 0.1j])
    c = _left_factor_times_q(q)
    c = c.astype(jnp.complex128)
    out, left, right = _extract_boundary_roots(
        c, jnp.asarray([10.0]), num_roots=[[1], [0]]
    )

    dense = np.diag([1.0, 0.5, 0.5])
    dense += np.diag([1.0, 1.0], k=1)
    dense += np.diag([0.5], k=2)
    expected = np.concatenate(
        [np.linalg.solve(dense, np.asarray(c)[1:, None])[:, 0], [0.0 + 0.0j]]
    )
    # MATLAB applies sgn=+1 to D\rhs for a left root.
    np.testing.assert_allclose(np.asarray(out), expected, rtol=20 * EPS, atol=20 * EPS)
    assert int(left) == 1
    assert int(right) == 0
    assert jnp.iscomplexobj(out)


def test_source_no_root_early_return_keeps_complex_coefficients_and_counts():
    c = jnp.asarray([1.25 + 0.5j, 0.125 - 0.25j, -0.03125 + 0.0625j])
    out, left, right = _extract_boundary_roots(
        c, jnp.asarray([2.0]), num_roots=[[1], [1]]
    )
    # MATLAB performs the endpoint/tolerance early return before considering
    # explicit numRoots, so a rootless function is returned untouched.
    np.testing.assert_array_equal(np.asarray(out), np.asarray(c))
    assert int(left) == int(right) == 0
    assert jnp.iscomplexobj(out)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_no_root_early_return_preserves_padded_coefficients(Tech):
    # Source returns before numRoots handling and simplify. Retaining trailing
    # zero rows verifies the public method returns the original tech object.
    coeffs = jnp.asarray(
        [[1.25, 2.0], [0.125, -0.25], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    f = Tech(coeffs=coeffs, ishappy=True)
    g, left, right = f.extractBoundaryRoots(num_roots=[[1, 1], [1, 1]])
    assert g is f
    np.testing.assert_array_equal(np.asarray(g.coeffs), np.asarray(coeffs))
    np.testing.assert_array_equal(np.asarray(left), [0, 0])
    np.testing.assert_array_equal(np.asarray(right), [0, 0])


def test_empty_shape_adapter_and_nonzero_singleton_source_no_root_return():
    # Empty-helper counts are shape-adapted; MATLAB vscale(empty) itself is
    # scalar zero. The nonzero singleton follows the source early return.
    empty = jnp.empty((0, 2), dtype=jnp.float64)
    out, left, right = _extract_boundary_roots(empty, jnp.zeros((2,)))
    assert out.shape == (0, 2)
    np.testing.assert_array_equal(np.asarray(left), [0, 0])
    np.testing.assert_array_equal(np.asarray(right), [0, 0])

    singleton = jnp.asarray([2.5])
    out, left, right = _extract_boundary_roots(singleton, jnp.asarray([2.5]))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(singleton))
    assert int(left) == int(right) == 0


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_empty_sentinel_returns_empty_count_vectors(Tech):
    f = Tech.empty()
    assert not hasattr(f, "coeffs")
    assert f.vscale_columns.shape == (0,)
    g, left, right = f.extractBoundaryRoots()
    assert g is f
    assert left.shape == right.shape == (0,)
