"""Source controls for Chebtech composition used by right division."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_compose_samples_at_least_operand_length_without_sample_test(Tech):
    # @chebtech/compose.m raises minSamples to length(f) and disables its
    # separate 2-point off-grid sampleTest.
    f = Tech.from_coeffs(jnp.zeros(33).at[16].set(1.0))
    observed_lengths = []

    def identity_with_sample_trace(values):
        observed_lengths.append(values.shape[0])
        return values

    result = f.compose(identity_with_sample_trace)
    assert observed_lengths
    assert observed_lengths[0] >= max(17, f.n)
    assert 2 not in observed_lengths
    points = jnp.array([-0.83, -0.2, 0.37, 0.91])
    np.testing.assert_allclose(
        np.asarray(result(points)), np.asarray(f(points)),
        rtol=100 * EPS, atol=100 * EPS,
    )


def test_chebtech1_compose_uses_source_resampling_sequence():
    # Starting from n=33, source composeResample1's next full grid has n=65.
    # Chebtech1's generic nested refinement would instead use n=99.
    f = Chebtech1.from_coeffs(jnp.pad(jnp.array([0.0, 1.0]), (0, 31)))
    observed_lengths = []

    def demanding_operator(values):
        observed_lengths.append(values.shape[0])
        return jnp.exp(40.0 * values)

    f.compose(demanding_operator)
    assert observed_lengths[:2] == [33, 65]


# Provenance: MATLAB @chebtech/compose.m, @chebtech1/compose.m,
# @chebtech2/compose.m and tests/chebtech/test_rdivide.m, commit 7574c77.
