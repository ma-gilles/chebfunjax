"""Source dimension schedules and factory BVP preference contracts.

Provenance
----------
MATLAB source : @valsDiscretization/valsDiscretization.m (dimensionValues),
    @cheboppref/cheboppref.m (factoryDefaultPrefs)
Chebfun commit: 7574c77
"""

import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import _piecewise_dimension_values


@pytest.mark.parametrize("minimum,maximum,expected", [
    (32, 4096, (32, 64, 128, 256, 512, 724, 1024, 1448, 2048, 2896, 4096)),
    (64, 256, (64, 128, 256)),
    (512, 1024, (512, 724, 1024)),
    (724, 1448, (724, 1024, 1448)),
    (5, 80, (5, 10, 20, 40, 80)),
])
def test_source_dimension_schedule(minimum, maximum, expected):
    assert _piecewise_dimension_values(minimum, maximum) == expected


@pytest.mark.parametrize("minimum,maximum", [(0, 32), (64, 32)])
def test_source_dimension_schedule_rejects_invalid_limits(minimum, maximum):
    with pytest.raises(ValueError):
        _piecewise_dimension_values(minimum, maximum)


def test_source_factory_bvp_tolerance_and_dimensions():
    saved = ChebopPref._defaults
    try:
        ChebopPref.setDefaults("factory")
        factory = ChebopPref.getFactoryDefaults()
        assert factory.bvpTol == 5e-13
        assert factory.minDimension == 32
        assert factory.maxDimension == 4096
        assert factory.happinessCheck == "standard"
        overridden = ChebopPref({"bvpTol": 1e-12, "minDimension": 64, "maxDimension": 256})
        assert overridden.bvpTol == 1e-12
        assert overridden.minDimension == 64
        assert overridden.maxDimension == 256
        assert not {"bvpTol", "minDimension", "maxDimension"}.intersection(overridden.techPrefs)
        keyword = ChebopPref(bvpTol=2e-12, minDimension=128, maxDimension=512)
        assert (keyword.bvpTol, keyword.minDimension, keyword.maxDimension) == (2e-12, 128, 512)
        combined = ChebopPref({"minDimension": 64}, minDimension=128)
        assert combined.minDimension == 128
        ChebopPref.setDefaults({"bvpTol": 3e-12, "minDimension": 64, "maxDimension": 256})
        inherited = ChebopPref()
        assert (inherited.bvpTol, inherited.minDimension, inherited.maxDimension) == (3e-12, 64, 256)
        partial = ChebopPref({"maxDimension": 512})
        assert (partial.bvpTol, partial.minDimension, partial.maxDimension) == (3e-12, 64, 512)
        assert ChebopPref(inherited) == inherited
        assert ChebopPref.getFactoryDefaults().bvpTol == 5e-13
        assert ChebopPref().bvpTol == 3e-12
        assert _piecewise_dimension_values(512, 724) == (512,)
        assert _piecewise_dimension_values(512, 725) == (512, 724)
    finally:
        ChebopPref._defaults = saved
