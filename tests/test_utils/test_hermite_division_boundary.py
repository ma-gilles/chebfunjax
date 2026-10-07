# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Division boundary controls; expected outputs are source-only diagnostic fixtures.

Provenance
----------
MATLAB source : hermpts.m, hermpts_asy barycentric division/normalization
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import json
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import _gradual as gradual
from chebfunjax.utils._signed_gradual import (
    flip_sign_bits,
    gradual_signed_divide,
    source_barycentric_normalize,
)


@pytest.mark.parametrize("disabled", [True, False])
def test_unchanged_full_source_normalization(disabled, tmp_path):
    with np.load((Path(__file__).parent / "fixtures/hermite_division_boundary.npz")) as f:
        half, divisor, expected = f["raw_v"], f["normalization_divisor"], f["final_v"]

    def action(h, d):
        return gradual_signed_divide(jnp.concatenate((h[::-1], flip_sign_bits(h))), d)

    with jax.disable_jit(disabled):
        got = jax.jit(action)(jnp.asarray(half), jnp.asarray(divisor))
        full = jnp.concatenate((jnp.asarray(half)[::-1], flip_sign_bits(jnp.asarray(half))))
        normalized = jax.jit(source_barycentric_normalize)(full)
    if not disabled:
        compiled = jax.jit(action).lower(jnp.asarray(half), jnp.asarray(divisor)).compile()
        hlo = compiled.as_text()
        output = tmp_path
        (output / "whole_normalization_optimized_hlo.txt").write_text(hlo)
        (output / "division_boundary_hlo_observation.json").write_text(
            json.dumps(
                {
                    "scope": "Same whole-output graph; source words asserted below; HLO needs review",
                    "optimization_barrier_mentions": hlo.count("opt-barrier"),
                    "divide_lines": [
                        line.strip() for line in hlo.splitlines() if " divide(" in line
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    for result in [got, normalized]:
        np.testing.assert_array_equal(np.asarray(result).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("disabled", [True, False])
def test_eight_original_failures_exact_quotient(disabled):
    with np.load((Path(__file__).parent / "fixtures/hermite_division_boundary.npz")) as f:
        half, divisor = f["raw_v"], f["normalization_divisor"]
    full = np.concatenate((half[::-1], -half))
    indices = [3326, 3327, 3328, 3330, 6669, 6671, 6672, 6673]
    x = full[indices]
    expected = np.array(
        [float(Fraction.from_float(float(v)) / Fraction.from_float(float(divisor))) for v in x]
    )
    with jax.disable_jit(disabled):
        actual = jax.jit(gradual_signed_divide)(jnp.asarray(x), jnp.asarray(divisor))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("disabled", [True, False])
def test_finite_division_forward_reverse_and_batch(disabled):
    # Analytic ordinary finite derivatives; no subnormal AD claim.
    a = jnp.asarray([0.75, 1.25, 2.5])
    b = jnp.asarray(1.75)
    f = gradual.gradual_positive_divide
    with jax.disable_jit(disabled):
        primal, tangent = jax.jit(
            lambda x, y: jax.jvp(f, (x, y), (jnp.ones_like(x), jnp.asarray(0.25)))
        )(a, b)
        da, db = jax.jit(jax.grad(lambda x, y: jnp.sum(f(x, y)), argnums=(0, 1)))(a, b)
        batched = jax.jit(jax.vmap(f, in_axes=(0, None)))(a, b)
    np.testing.assert_allclose(primal, np.asarray(a) / 1.75, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        tangent, (1 - np.asarray(a) * 0.25 / 1.75) / 1.75, rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(da, np.full(3, 1 / 1.75), rtol=2e-15, atol=0)
    np.testing.assert_allclose(db, -np.sum(np.asarray(a)) / 1.75**2, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(batched, primal)


def test_literal_source_n251_moments():
    # Exact original test_hermpts.m clause8, not the n201 extra mass control.
    from chebfunjax.utils.quadrature import hermpts

    x, w, _ = hermpts(251, "phys", method="ASY", bary=True)
    x, w = np.asarray(x), np.asarray(w)
    tol = 10 * np.finfo(float).eps
    assert abs(w @ x) < tol
    assert abs(w @ (x**2) - np.sqrt(np.pi) / 2) < 300 * tol
