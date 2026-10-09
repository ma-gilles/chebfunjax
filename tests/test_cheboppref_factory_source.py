"""Factory field/storage controls, native @cheboppref/cheboppref.m459-476.

Pinned source 7574c77. Selector strings are Python representations; arbitrary
function handles, complete parser parity and solver use of scale/lambdaMin
are outside this package. No numerical solver is invoked by these controls.
"""

import math

import pytest

from chebfunjax.chebpref import ChebfunPref, ChebopPref
from tests.test_matlab_port.chebpref.test_cheboppref_matlab import _isequal_nan


@pytest.fixture(autouse=True)
def restore_factory_preferences():
    saved_fun, saved_op = ChebfunPref._defaults, ChebopPref._defaults
    ChebfunPref.setDefaults("factory")
    ChebopPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebfunPref._defaults, ChebopPref._defaults = saved_fun, saved_op


def test_factory_values_and_top_storage():
    p = ChebopPref.getFactoryDefaults()
    assert math.isnan(p.scale)
    assert p.lambdaMin == 1e-6
    assert p.happinessCheck == "standard"
    for key in ("scale", "lambdaMin", "happinessCheck"):
        assert key in p._top
        assert key not in p._tech_overrides


def test_technology_check_does_not_replace_operator_factory():
    ChebfunPref.setDefaults("happinessCheck", "strict")
    assert ChebfunPref().happinessCheck == "strict"
    p = ChebopPref({"techPrefs": {"happinessCheck": "classic"}})
    assert p.techPrefs.happinessCheck == "classic"
    assert p.happinessCheck == "standard"
    assert ChebopPref().happinessCheck == "standard"


def test_constructor_overrides_copy_and_input_independence():
    source = {"scale": 3.0, "lambdaMin": 2e-6, "happinessCheck": "strict"}
    p = ChebopPref(source)
    q = ChebopPref(p)
    assert _isequal_nan(p, q)
    assert (q.scale, q.lambdaMin, q.happinessCheck) == (3.0, 2e-6, "strict")
    q.scale = 4.0
    assert p.scale == source["scale"] == 3.0
    assert source == {"scale": 3.0, "lambdaMin": 2e-6, "happinessCheck": "strict"}


def test_property_updates_are_top_level():
    p = ChebopPref()
    p.scale, p.lambdaMin, p.happinessCheck = 7.0, 3e-6, "strict"
    assert (p.scale, p.lambdaMin, p.happinessCheck) == (7.0, 3e-6, "strict")
    for key in ("scale", "lambdaMin", "happinessCheck"):
        assert key in p._top
        assert key not in p._tech_overrides


def test_object_and_dictionary_session_defaults_reset():
    overrides = {"scale": 2.0, "lambdaMin": 4e-6, "happinessCheck": "classic"}
    for supplied in (ChebopPref(overrides), overrides):
        ChebopPref.setDefaults(supplied)
        p = ChebopPref()
        assert (p.scale, p.lambdaMin, p.happinessCheck) == (2.0, 4e-6, "classic")
        ChebopPref.setDefaults("factory")
        p = ChebopPref()
        assert math.isnan(p.scale)
        assert (p.lambdaMin, p.happinessCheck) == (1e-6, "standard")


def test_pair_updates_preserve_top_storage():
    ChebopPref.setDefaults("scale", 8.0, "lambdaMin", 5e-6, "happinessCheck", "strict")
    p = ChebopPref()
    assert (p.scale, p.lambdaMin, p.happinessCheck) == (8.0, 5e-6, "strict")
    assert not {"scale", "lambdaMin", "happinessCheck"}.intersection(p._tech_overrides)


def test_nan_adapter_full_state_and_non_scalar_rejection():
    p, q = ChebopPref(), ChebopPref()
    p.scale, q.scale = float("nan"), float("nan")
    assert p.scale is not q.scale
    assert _isequal_nan(p, q)
    q.lambdaMin = 2e-6
    assert not _isequal_nan(p, q)
    q = ChebopPref()
    q._tech_overrides["extra"] = 1
    assert not _isequal_nan(p, q)
    assert _isequal_nan({"x": [float("nan"), 1]}, {"x": [float("nan"), 1]})
    assert not _isequal_nan({"x": 1}, {"y": 1})

    class NonScalarEquality:
        def __eq__(self, other):
            return [True, True]

    with pytest.raises(TypeError, match="scalar Boolean"):
        _isequal_nan(NonScalarEquality(), NonScalarEquality())


def test_default_consumer_read_values_unchanged():
    p = ChebopPref()
    eps = 2.220446049250313e-16
    assert p.happinessCheck == "standard"
    assert p.ivpAbsTol == 1e5 * eps
    assert p.ivpRelTol == 100 * eps
    assert p.ivpRestartSolver is True
    assert (p.minDimension, p.maxDimension, p.bvpTol, p.ivpSolver) == (32, 4096, 5e-13, "ode113")
