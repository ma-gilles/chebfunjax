"""Native trigtech vscale uses coefficients, independently of cached values.

MATLAB provenance: @trigtech/vscale.m and @trigtech/happinessCheck.m.
Chebfun commit: 7574c77. Chebfun3 happinessCheck3 itself mutates coefficients
without replacing the original stored values.
"""

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from chebfunjax.tech._trig_constructor import happiness, resolve_pref
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("length", [1, 3])
def test_coefficient_scales_ignore_independent_cached_samples(length):
    coeffs = jnp.zeros((length, 2), dtype=jnp.complex128)
    coeffs = coeffs.at[length // 2].set(jnp.asarray([2., 3.+4j]))
    cached = jnp.full((length, 2), 1000.+0j)
    f = Trigtech(coeffs=coeffs, real_columns=(True, False), _values=cached)
    assert f.vscale == 5.
    assert jnp.array_equal(f.vscale_columns(), jnp.asarray([2., 5.]))
    assert jnp.array_equal(f.values, cached)


@pytest.mark.parametrize("global_scale,expected", [(0., [2., 5.]), (10., [10., 10.])])
def test_happiness_callback_receives_coefficient_and_global_scale(global_scale, expected):
    f = Trigtech(coeffs=jnp.asarray([[2., 3.+4j]]),
                real_columns=(True, False), _values=jnp.full((1, 2), 1000.))
    seen = []

    def callback(tech, values, data, pref):
        seen.append(data["vscale"])
        assert tech is f
        return True, 1

    pref = resolve_pref({"happinessCheck": callback})
    assert happiness(f, values=f.values, data={"vscale": global_scale}, pref=pref) == (True, 1)
    assert jnp.array_equal(seen[0], jnp.asarray(expected))


def test_empty_coefficient_scale_is_source_scalar_zero():
    f = Trigtech()
    assert f.vscale == 0.
    scale = f.vscale_columns()
    assert jnp.shape(scale) == ()
    assert scale == 0.


@pytest.mark.parametrize("case_index", range(4))
def test_native_mutated_multicolumn_fiber_scale(case_index):
    fixture = Path(__file__).parents[1] / "fixtures/trig_vscale_mutated_coefficients_native_20261015.json"
    case = json.loads(fixture.read_text())["cases"][case_index]
    values = jnp.asarray(case["postMutationValues"])
    f = Trigtech.from_values(values)
    f = eqx.tree_at(lambda tech: tech.coeffs, f, jnp.asarray(case["envelope"]))
    assert jnp.array_equal(f.values, values)
    assert jnp.array_equal(f.isReal, jnp.asarray(case["postMutationIsReal"]))
    # The held native coefficients isolate vscale from the forward transform.
    # Allow n*eps for the inverse FFT across native/JAX providers; the stale
    # cache discrepancy is orders of magnitude larger than this error bound.
    expected = case["postMutationVscale"]
    bound = case["n"] * 2.**-52 * abs(expected)
    assert abs(f.vscale - expected) <= bound
    assert jnp.all(jnp.abs(f.vscale_columns() - expected) <= bound)
