"""Literal source interpolation quotient: one binary64 division.

@chebtech/extrapolate.m 7574c77, actual native stage operands plus exact
rational arithmetic; no fitted correction or new tolerance oracle.
"""
import json
import math
import struct
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech2, _extrapolate_values
from chebfunjax.utils.quadrature import chebpts

RAW = json.loads((Path(__file__).parents[1] / "fixtures/disk_remaining_source_native_20261018.json").read_text())


def array(record):
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                       for w in record["re_hex"]]).reshape(record["shape"], order="F")


def exact_words(actual, expected):
    actual = np.ascontiguousarray(np.asarray(actual, dtype=np.float64))
    expected = np.ascontiguousarray(np.asarray(expected, dtype=np.float64))
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("case", RAW["transforms"], ids=lambda c: c["name"])
def test_native_extrapolated_values_strict(case):
    values = jnp.asarray(array(case["input"]))
    actual, mask_nan, mask_inf = _extrapolate_values(
        values, chebpts(values.shape[0]), Chebtech2.barywts(values.shape[0]))
    exact_words(actual, array(case["filled"]))
    np.testing.assert_array_equal(mask_nan, array(case["mask_nan"]).reshape(-1).astype(bool))
    np.testing.assert_array_equal(mask_inf, array(case["mask_inf"]).reshape(-1).astype(bool))


@pytest.mark.parametrize("case", [c for c in RAW["transforms"] if "interpolation" in c],
                         ids=lambda c: c["name"])
def test_actual_interpolation_quotients_independent_exact_rational(case):
    values = jnp.asarray(array(case["input"]))
    actual, mask, _ = _extrapolate_values(
        values, chebpts(values.shape[0]), Chebtech2.barywts(values.shape[0]))
    rows = np.flatnonzero(np.asarray(mask))
    for row, stage in zip(rows, case["interpolation"]):
        denominator = float(array(stage["denominator"]).reshape(()))
        numerator = array(stage["numerator"]).reshape(-1)
        assert np.all(np.isfinite(numerator)) and math.isfinite(denominator) and denominator != 0
        expected = np.asarray([float(Fraction(float(v))/Fraction(denominator))
                               for v in numerator])
        exact_words(actual[row], expected)


def test_extrapolation_polynomial_values_and_jvp():
    # Exact dyadic grid/weights: all operations and these degree-two values
    # are representable. This independent interpolant checks the shared
    # consumer and its ordinary JVP, rather than just its divide helper.
    x = jnp.asarray([-1., -.5, 0., .5, 1.])
    weights = jnp.asarray([1., -4., 6., -4., 1.])
    direction = 3*x+2
    base = 5*x*x-3*x+2
    mask = jnp.asarray([False, True, False, True, False])

    def interpolate(t):
        values = jnp.where(mask, jnp.nan, base+t*direction)
        return _extrapolate_values(values, x, weights)[0]

    primal, tangent = jax.jvp(interpolate, (jnp.float64(0),), (jnp.float64(1),))
    exact_words(primal, base)
    exact_words(tangent, direction)
