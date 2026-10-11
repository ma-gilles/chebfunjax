"""Safety adapter controls; native nonadvancing loop is not executed."""

import math

import pytest

from chebfunjax.domain import merge


@pytest.mark.parametrize("value", [1.0, -1.0, float(2**53)])
def test_positive_subulp_tolerance_rejected(value):
    tol = math.ulp(value) / 4
    assert tol > 0 and value + tol == value
    with pytest.raises(ValueError, match="tolerance must advance each finite breakpoint"):
        merge([value, value + 4], tol)


def test_infinite_endpoints_keep_native_removal_semantics():
    tiny = math.ulp(1.0) / 4
    assert merge([-math.inf, math.inf], tiny) == [-math.inf, math.inf]
    assert merge([math.inf], tiny) == [math.inf]
    assert merge([-math.inf], tiny) == [-math.inf]
    assert merge([-math.inf, 1.0, math.inf], math.inf) == [-math.inf, 1.0, math.inf]
