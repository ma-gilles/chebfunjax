"""Source-shaped angular projection for Chebyshev barycentric matrices.

Provenance
----------
MATLAB source : barymat.m, @chebcolloc/reduce.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.interpolation import barymat, cheb_bary_weights


def _source_angle_matrix(y, x, weights, s, r, do_flip):
    """Small NumPy transcription of MATLAB's angle formula and flip mask."""
    y = np.asarray(y)
    x = np.asarray(x)
    weights = np.asarray(weights)
    s = np.asarray(s)
    r = np.asarray(r)
    if y.shape == x.shape and np.all(y == x):
        return np.eye(x.size)
    delta = (2.0 * np.sin((s[:, None] + r[None, :]) / 2.0)
             * np.sin((r[None, :] - s[:, None]) / 2.0))
    matrix = weights[None, :] / delta
    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
    matrix[np.isnan(matrix)] = 1.0
    if do_flip:
        mask = np.fliplr(np.rot90(np.tril(np.ones(matrix.shape)), 2)).astype(bool)
        rotated = np.rot90(matrix, 2)
        matrix[mask] = rotated[mask]
        matrix[np.isnan(matrix)] = 1.0
    return matrix


@pytest.mark.parametrize("flip", [False, True])
def test_angular_projection_recovers_chebyshev_polynomials_under_affine_map(
    flip,
):
    r = jnp.arange(18, -1, -1) * jnp.pi / 18
    s = jnp.arange(10.5, 0, -1) * jnp.pi / 11
    # The source collocation helper supplies reference angles together with
    # physical nodes mapped to this translated, narrow interval.
    x = 100.0 + 1.0e-5 * jnp.cos(r)
    y = 100.0 + 1.0e-5 * jnp.cos(s)
    weights = cheb_bary_weights(19)
    matrix = jax.jit(
        lambda: barymat(y, x, weights, s, r, flip)
    )()
    for degree in (0, 1, 2, 7, 18):
        npt.assert_allclose(
            np.asarray(matrix @ jnp.cos(degree * r)),
            np.asarray(jnp.cos(degree * s)),
            atol=100 * np.finfo(float).eps,
            rtol=0,
        )


@pytest.mark.parametrize("flip", [False, True])
def test_angle_branch_matches_numpy_transcription_of_matlab_formula(flip):
    r = np.arange(18, -1, -1, dtype=np.float64) * np.pi / 18
    s = np.asarray([np.pi - 0.19, np.pi - 0.63, np.pi / 2, 0.63, 0.19])
    x = 4.0 + 0.125 * np.cos(r)
    y = 4.0 + 0.125 * np.cos(s)
    weights = np.asarray(cheb_bary_weights(r.size))
    got = barymat(y, x, jnp.asarray(weights), s, r, flip)
    expected = _source_angle_matrix(y, x, weights, s, r, flip)
    npt.assert_allclose(
        np.asarray(got), expected,
        rtol=0,
        atol=100 * np.finfo(float).eps,
    )


def test_angle_projection_preserves_partial_coincident_rows():
    r = jnp.arange(12, -1, -1) * jnp.pi / 12
    s = jnp.asarray([r[2], 0.371, r[9]])
    x = 0.75 + 0.2 * jnp.cos(r)
    y = 0.75 + 0.2 * jnp.cos(s)
    matrix = barymat(y, x, cheb_bary_weights(13), s, r, False)
    expected = np.zeros((3, 13))
    expected[0, 2] = 1.0
    expected[2, 9] = 1.0
    npt.assert_array_equal(
        np.asarray(matrix[jnp.asarray([0, 2]), :]), expected[[0, 2], :])
    npt.assert_allclose(
        np.asarray(matrix[1] @ jnp.cos(5 * r)),
        np.asarray(jnp.cos(5 * s[1])),
        atol=100 * np.finfo(float).eps,
        rtol=0,
    )


def test_exact_physical_grid_identity_precedes_stale_angle_metadata():
    r = jnp.arange(8, -1, -1) * jnp.pi / 8
    x = jnp.cos(r)
    # MATLAB returns eye(N) as soon as physical x and y are exactly equal.
    # Even valid-shape but stale rounded angles must not perturb that result.
    s = r + 0.037
    got = jax.jit(
        lambda yy, ss: barymat(yy, x, cheb_bary_weights(9), ss, r, True)
    )(x, s)
    npt.assert_array_equal(np.asarray(got), np.eye(9))


def test_dynamic_jit_flip_argument():
    r = jnp.arange(18, -1, -1) * jnp.pi / 18
    s = jnp.asarray([jnp.pi - 0.21, jnp.pi - 0.76, 0.76, 0.21])
    x = -3.0 + 0.4 * jnp.cos(r)
    weights = cheb_bary_weights(19)
    degree = 7
    values = jnp.cos(degree * r)

    def project(target_angles, flip):
        y = -3.0 + 0.4 * jnp.cos(target_angles)
        return barymat(y, x, weights, target_angles, r, flip) @ values

    compiled = jax.jit(project)
    for flip in (jnp.asarray(False), jnp.asarray(True),
                 jnp.asarray(0.0), jnp.asarray(1.0)):
        npt.assert_allclose(
            np.asarray(compiled(s, flip)),
            np.asarray(jnp.cos(degree * s)),
            atol=100 * np.finfo(float).eps,
            rtol=0,
        )


def test_numeric_matlab_flip_flags_apply_literal_rectangular_mask():
    r = jnp.arange(18, -1, -1) * jnp.pi / 18
    s = jnp.arange(10.5, 0, -1) * jnp.pi / 11
    x = 100.0 + 1.0e-5 * jnp.cos(r)
    y = 100.0 + 1.0e-5 * jnp.cos(s)
    weights = cheb_bary_weights(19)
    base = np.asarray(barymat(y, x, weights, s, r, False))
    rows, cols = np.indices(base.shape)
    flip_mask = rows + cols <= base.shape[0] - 1
    rotated = np.rot90(base, 2)
    expected = base.copy()
    expected[flip_mask] = rotated[flip_mask]

    for flag in (1.0, 2.0):
        got = barymat(y, x, weights, s, r, flag)
        npt.assert_array_equal(np.asarray(got), expected)


def test_angle_gradient_away_from_nodes_matches_chebyshev_derivative():
    r = jnp.arange(18, -1, -1) * jnp.pi / 18
    s = jnp.asarray([2.73, 1.61, 0.76, 0.21])
    x = -3.0 + 0.4 * jnp.cos(r)
    weights = cheb_bary_weights(19)
    degree = 7
    values = jnp.cos(degree * r)

    def project(target_angles):
        y = -3.0 + 0.4 * jnp.cos(target_angles)
        return barymat(y, x, weights, target_angles, r, False) @ values

    gradient = jax.jit(jax.jacrev(project))(s)
    expected = -degree * jnp.sin(degree * s)
    npt.assert_allclose(
        np.asarray(gradient), np.diag(np.asarray(expected)),
        atol=300 * np.finfo(float).eps,
        rtol=0,
    )
