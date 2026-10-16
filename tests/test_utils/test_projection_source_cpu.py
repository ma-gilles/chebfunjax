"""Source comparisons and numerical controls for projected collocation.

Provenance
----------
MATLAB sources: barymat.m, chebcolloc/reduce.m, chebcolloc/baryDiffMat.m.
Chebfun commit: 7574c77.
The native matrix fixture is captured from the pinned MATLAB source.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.operators.piecewise_linear import _panel_projection
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.interpolation import barymat, cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts

_NATIVE = Path(__file__).parents[1] / "references" / "projection_source_cpu_native.npz"
_RAW_SIZES = (34, 66, 130, 258, 514)


@pytest.mark.parametrize("raw_n", _RAW_SIZES)
def test_projection_and_diffmat_match_native_matrices(raw_n):
    """Compare source projection and canonical raw derivatives by matrix."""
    base_n = raw_n - 2
    with np.load(_NATIVE) as reference:
        expected_projection = reference[f"B_{raw_n}"]
        expected_d1 = reference[f"D1_{raw_n}"]
        expected_d2 = reference[f"D2_{raw_n}"]

    projection, _, _ = _panel_projection(-1.0, 1.0, base_n, raw_n)
    npt.assert_allclose(
        np.asarray(projection), expected_projection, rtol=0.0, atol=5e-17,
    )
    npt.assert_array_equal(np.asarray(diffmat(raw_n, 1)), expected_d1)
    npt.assert_array_equal(np.asarray(diffmat(raw_n, 2)), expected_d2)


def test_barymat_matches_native_projection_from_captured_angles():
    """Exercise the public angle-aware matrix API with pinned source angles."""
    raw_n = 34
    base_n = raw_n - 2
    with np.load(_NATIVE) as reference:
        expected = reference[f"B_{raw_n}"]
        target_angles = jnp.asarray(reference[f"s_{raw_n}"]).reshape(-1)
        source_angles = jnp.asarray(reference[f"r_{raw_n}"]).reshape(-1)

    targets = chebpts(base_n, kind=1)
    nodes = chebpts(raw_n, kind=2)
    actual = barymat(
        targets, nodes, cheb_bary_weights(raw_n),
        target_angles, source_angles, do_flip=True,
    )
    npt.assert_allclose(
        np.asarray(actual), expected, rtol=0.0, atol=5e-17,
    )


def test_barycentric_polynomial_interpolation_is_jittable_and_differentiable():
    """Track target and source-node gradients while preserving a cubic exactly."""
    reference_nodes = chebpts(9, kind=2)
    weights = cheb_bary_weights(9)

    def polynomial(x):
        return x**3 - 2.0 * x + 0.5

    def interpolate(targets, node_shift):
        nodes = reference_nodes + node_shift
        values = polynomial(nodes)
        return barymat(targets, nodes, weights) @ values

    targets = jnp.asarray([-0.71, 0.17, 0.62])
    compiled = jax.jit(interpolate)
    got = compiled(targets, jnp.asarray(0.03))
    npt.assert_allclose(
        np.asarray(got), np.asarray(polynomial(targets)), rtol=0.0, atol=2e-14,
    )

    def scalar_value(target, node_shift):
        return interpolate(target[None], node_shift)[0]

    gradients = jax.jit(jax.grad(scalar_value, argnums=(0, 1)))(
        jnp.asarray(0.17), jnp.asarray(0.03),
    )
    expected_target_gradient = 3.0 * 0.17**2 - 2.0
    npt.assert_allclose(
        np.asarray(gradients[0]), expected_target_gradient,
        rtol=0.0, atol=2e-13,
    )
    npt.assert_allclose(
        np.asarray(gradients[1]), 0.0, rtol=0.0, atol=2e-13,
    )
