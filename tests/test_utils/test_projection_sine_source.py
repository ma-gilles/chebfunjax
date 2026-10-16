"""Source sine and projection controls with independent native references.

Provenance
----------
Chebfun barymat.m, commit 7574c77; Netlib fdlibm e_rem_pio2.c/k_sin.c/s_sin.c.
Native fixture capture source and hashes: references/projection_sine_native.json.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.piecewise_linear import _panel_projection
from chebfunjax.utils._fdlibm_projection_sine import _sin_projection_angles

_REF = Path(__file__).parents[1] / "references"


def test_dense_and_boundary_angles_match_native_sine():
    with np.load(_REF / "projection_sine_native.npz") as reference:
        angles = jnp.asarray(reference["angles"])
        expected = reference["sines"]
    actual = np.asarray(jax.jit(_sin_projection_angles)(angles))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("raw_n", [34, 66, 130, 258, 514])
def test_complete_projection_matches_native_bits(raw_n):
    with np.load(_REF / "projection_source_cpu_native.npz") as reference:
        expected = reference[f"B_{raw_n}"]
    actual, _, _ = _panel_projection(-1.0, 1.0, raw_n - 2, raw_n)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint64),
        expected.view(np.uint64),
    )


def test_projection_sine_jvp_follows_cosine():
    angles = jnp.asarray(
        [
            -3.14,
            -3.0,
            -2.6,
            -2.36,
            -2.0,
            -1.6,
            -0.7,
            0.0,
            0.7,
            1.6,
            2.0,
            2.36,
            2.6,
            3.0,
            3.14,
        ]
    )
    _, tangent = jax.jit(
        lambda x: jax.jvp(
            _sin_projection_angles,
            (x,),
            (jnp.ones_like(x),),
        )
    )(angles)
    np.testing.assert_allclose(
        np.asarray(tangent),
        np.asarray(jnp.cos(angles)),
        atol=8 * np.finfo(float).eps,
        rtol=0.0,
    )


def test_projection_sine_rejects_values_outside_bounded_domain():
    outside = np.nextafter(np.pi, np.inf)
    actual = _sin_projection_angles(
        jnp.asarray(
            [
                outside,
                -outside,
                np.inf,
                -np.inf,
                np.nan,
            ]
        )
    )
    assert bool(jnp.all(jnp.isnan(actual)))
