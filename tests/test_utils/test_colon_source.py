"""Source endpoint/count controls; no solver tolerance is changed."""

import numpy as np
import pytest

from chebfunjax.utils.colon import matlab_colon


@pytest.mark.parametrize(
    "a,d,b,expected",
    [
        (0.0, 0.5, 2.0, [0.0, 0.5, 1.0, 1.5, 2.0]),
        (2.0, -0.5, 0.0, [2.0, 1.5, 1.0, 0.5, 0.0]),
        (0.0, 0.5, 1.4, [0.0, 0.5, 1.0]),
        (0.0, 0.5, 1.1, [0.0, 0.5, 1.0]),
        (1.0, 0.5, 1.0, [1.0]),
        (0.0, 0.5, 0.0, [0.0]),
        (0.0, 0.0, 1.0, []),
        (1.0, 0.5, 0.0, []),
        (0.0, -0.5, 1.0, []),
        (0.0, 0.5, 1.5, [0.0, 0.5, 1.0, 1.5]),
    ],
)
def test_exact_count_and_values(a, d, b, expected):
    actual = np.asarray(matlab_colon(a, d, b))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_source_dde_native_colon_words():
    # ddI1 source MATLAB 0:(1/3):2 observation, captured before any constructor.
    expected = [
        float.fromhex(x)
        for x in (
            "0x0p+0",
            "0x1.5555555555555p-2",
            "0x1.5555555555555p-1",
            "0x1p+0",
            "0x1.5555555555556p+0",
            "0x1.aaaaaaaaaaaabp+0",
            "0x1p+1",
        )
    ]
    np.testing.assert_array_equal(matlab_colon(0.0, 1 / 3, 2.0), expected)
    np.testing.assert_array_equal(matlab_colon(2.0, -1 / 3, 0.0), expected[::-1])


def test_source_close_endpoint_retained():
    endpoint = np.nextafter(1.0, 2.0)
    np.testing.assert_array_equal(matlab_colon(0.0, 0.5, endpoint), [0.0, endpoint / 2, endpoint])


@pytest.mark.parametrize("a,d,b", [(np.inf, 1.0, 2.0), (0.0, np.nan, 2.0), (0.0, 1.0, np.inf)])
def test_explicit_finite_scope(a, d, b):
    with pytest.raises(ValueError, match="finite"):
        matlab_colon(a, d, b)
