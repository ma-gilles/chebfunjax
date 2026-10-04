"""Independent component-scale checks for source tech convergence.

Provenance
----------
MATLAB source : @chebtech/happinessCheck.m, standardCheck.m, sampleTest.m,
    @chebtech/populate.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_large_neighbor_cannot_hide_small_component_sample_error(tech):
    coeffs = jnp.zeros((33, 2)).at[0].set(jnp.array([1e12, 1.0]))
    values = tech.coeffs2vals(coeffs)

    def incorrect_operator(x):
        return jnp.broadcast_to(jnp.array([1e12, 1.1]), (x.size, 2))

    happy, cutoff = tech.happiness_check(coeffs, values, op=incorrect_operator)
    assert not happy
    assert cutoff == 33


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_zero_component_sample_error_is_rejected(tech):
    coeffs = jnp.zeros((33, 2)).at[0, 0].set(1e12)
    values = tech.coeffs2vals(coeffs)

    def incorrect_operator(x):
        return jnp.broadcast_to(jnp.array([1e12, 1e-12]), (x.size, 2))

    assert tech.happiness_check(coeffs, values, op=incorrect_operator)[0] is False


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_array_cutoff_matches_separate_component_checks(tech):
    coeffs = jnp.zeros((65, 3)).at[0].set(jnp.array([1e12, 1.0, 0.0]))
    coeffs = coeffs.at[20, 1].set(1e-5)
    values = tech.coeffs2vals(coeffs)
    scales = jnp.array([1e12, 100.0, 0.0])
    tolerances = jnp.array([1e-8, 1e-10, 1e-8])
    separate = [tech.happiness_check(
        coeffs[:, k], values[:, k], vscale=float(scales[k]),
        tol=float(tolerances[k]), sample_test=False,
    ) for k in range(3)]
    happy, cutoff = tech.happiness_check(
        coeffs, values, vscale=scales, tol=tolerances, sample_test=False,
    )
    assert happy == all(item[0] for item in separate)
    assert cutoff == max(item[1] for item in separate)
    assert cutoff > 1  # The small component's degree20 coefficient is retained.


def test_adaptive_global_component_scales_preserve_small_signal():
    def operator(x):
        return jnp.stack((jnp.full_like(x, 1e12), jnp.cos(20*x)), axis=-1)

    f = Chebtech2.from_function(
        operator, maxpow2=8, tol=1e-12,
        vscale=jnp.array([1e12, 1.0]), sample_test=False,
    )
    x = jnp.linspace(-1, 1, 97)
    assert f.ishappy
    npt.assert_allclose(np.asarray(f(x))[:, 1], np.cos(20*np.asarray(x)),
                        atol=2e-12, rtol=0)
