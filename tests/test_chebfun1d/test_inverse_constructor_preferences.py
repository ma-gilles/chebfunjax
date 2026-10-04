"""Source inverse construction preferences observed at the tech boundary.

Provenance
----------
MATLAB source : @chebfun/inv.m, local parseInputs; @chebfun/constructor.m
Chebfun commit: 7574c77

These independent boundary controls supplement the original test_inv.m battery.
They also exercise recursive forwarding used by bounded splitting construction.
"""

import copy

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2


def _observe_tech_constructor(monkeypatch):
    original = Chebtech2.from_function.__func__
    observed = []

    def record(cls, *args, **kwargs):
        observed.append(dict(kwargs))
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(Chebtech2, "from_function", classmethod(record))
    return observed


@pytest.mark.parametrize("algorithm,caller_refinement,expected_refinement", [
    ("brent", "nested", "nested"),
    ("brent", "resampling", "resampling"),
    ("newton", "nested", "resampling"),
])
def test_inverse_overrides_only_local_source_preferences(
        monkeypatch, algorithm, caller_refinement, expected_refinement):
    forward = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 2.0]))
    caller = ChebfunPref(refinementFunction=caller_refinement,
                        sampleTest=True, minSamples=35)
    saved_caller = copy.deepcopy(caller.__dict__)
    global_defaults = ChebfunPref(sampleTest=True)
    monkeypatch.setattr(ChebfunPref, "_defaults", global_defaults)
    saved_defaults = copy.deepcopy(global_defaults.__dict__)
    observed = _observe_tech_constructor(monkeypatch)
    inverse = forward.inv(caller, algorithm=algorithm)
    assert observed
    assert all(call["sample_test"] is False for call in observed)
    assert all(call["min_samples"] == len(forward) for call in observed)
    assert all(call["refinement_function"] == expected_refinement
               for call in observed)
    targets = jnp.linspace(-0.9, 0.9, 17)
    bound = 100 * float(jnp.finfo(jnp.float64).eps) * inverse.vscale
    assert float(jnp.max(jnp.abs(inverse(targets) - (targets - 1.0) / 2))) < bound
    assert caller.__dict__ == saved_caller
    assert global_defaults.__dict__ == saved_defaults


@pytest.mark.parametrize("global_sample_test", [False, True])
def test_ordinary_constructor_retains_global_sample_test_default(
        monkeypatch, global_sample_test):
    monkeypatch.setattr(ChebfunPref, "_defaults",
                        ChebfunPref(sampleTest=global_sample_test))
    observed = _observe_tech_constructor(monkeypatch)
    result = cj.chebfun(lambda x: 1.0 + x)
    assert observed
    assert all(call["sample_test"] is global_sample_test for call in observed)
    assert float(jnp.max(jnp.abs(result(jnp.asarray([-0.5, 0.5]))
                                - jnp.asarray([0.5, 1.5])))) < 100 * jnp.finfo(jnp.float64).eps


def test_split_locator_children_and_pieces_receive_local_preferences(monkeypatch):
    monkeypatch.setattr(ChebfunPref, "_defaults", ChebfunPref(sampleTest=True))
    observed = _observe_tech_constructor(monkeypatch)
    result = cj.chebfun(jnp.sign, splitting=True, sample_test=False,
                        refinement_function="resampling", min_samples=7,
                        max_length=129, split_length=32, split_max_length=256)
    assert len(observed) > 2  # Locator, recursive fits and final pieces.
    assert all(call["sample_test"] is False for call in observed)
    assert all(call["refinement_function"] == "resampling" for call in observed)
    assert all(call["min_samples"] == 7 for call in observed)
    points = jnp.asarray([-0.8, -0.2, 0.2, 0.8])
    assert float(jnp.max(jnp.abs(result(points) - jnp.sign(points)))) < 100 * jnp.finfo(jnp.float64).eps
    assert ChebfunPref().sampleTest is True


def test_conflicting_refinement_selectors_raise():
    with pytest.raises(ValueError, match="conflict"):
        cj.chebfun(jnp.sin, resampling=True, refinement_function="nested")


def test_default_recursive_coefficient_builder_retains_ordinary_behavior():
    # A resolved global default must not become an explicit adaptive override
    # when doubleLength recursively rebuilds a coefficient input.
    result = cj.chebfun(jnp.asarray([1.0, 2.0]), coeffs=True,
                        doubleLength=True)
    points = jnp.asarray([-1.0, -0.25, 0.5, 1.0])
    assert float(jnp.max(jnp.abs(result(points) - (1.0 + 2.0 * points)))) < 100 * jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("options", [
    {"trig": True},
    {"domain": (-jnp.inf, jnp.inf)},
    {"exps": (0.5, 0.0)},
])
def test_unpropagated_representation_overrides_raise_explicitly(options):
    with pytest.raises(ValueError, match="overrides are not yet supported"):
        cj.chebfun(jnp.sin, sample_test=False, **options)
