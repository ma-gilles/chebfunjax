"""Source GE order, native stages, and independently exact tiny arithmetic.

Native acceptance computes the production tolerance. NaN payload bits are
reported separately in the packet; these tests require finite words and
nonfinite classes/signs, not an invented NaN payload policy.
"""
import json
import math
import struct
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._numeric_phase_one import (
    _exact_tiny_sum,
    _origin_mean,
    _update,
    phase_one_numeric,
)
from chebfunjax.diskfun.diskfun import _get_tol
from chebfunjax.operators import _qz_ieee as ieee

RAW = json.loads((Path(__file__).parents[1] / "fixtures" /
                  "disk_phaseone_jax_native_20261018.json").read_text())


def _array(record):
    shape = tuple(record["shape"])
    if math.prod(shape) == 0:
        return np.empty(shape)
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                       for w in record["re_hex"]]).reshape(shape, order="F")


def _words(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert actual.size == expected.size
    actual = actual.reshape(expected.shape)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual[finite]).view(np.uint64),
        np.ascontiguousarray(expected[finite]).view(np.uint64))


@pytest.mark.parametrize("case", RAW["cases"], ids=lambda c: c["name"])
def test_native_phase_one_at_production_tolerance(case):
    values = _array(case["input"])
    m, n = values.shape
    tol = _get_tol(values, 2 * np.pi / n, np.pi / max(m - 1, 1),
                   np.finfo(float).eps, source_numeric=True)[0]
    result = phase_one_numeric(jnp.asarray(values), tol, 100.)
    native = case["source"]
    for name in ("cols", "rows", "pivots", "pivot_array"):
        _words(getattr(result, name), _array(native[name]))
    _words(result.indices + 1, _array(native["pivot_indices"]))
    assert result.remove_pole == native["remove_poles"]
    assert result.happy == bool(native["is_happy"])
    assert result.plus == (_array(native["idx_plus"]).reshape(-1) - 1).tolist()
    assert result.minus == (_array(native["idx_minus"]).reshape(-1) - 1).tolist()


@pytest.mark.parametrize("case", RAW["means"], ids=lambda c: c["name"])
def test_native_origin_sum_then_divide(case):
    _words(_origin_mean(jnp.asarray(_array(case["input"]))), _array(case["mean"]))


def _update_cases():
    cases = []
    for case in RAW["cases"]:
        if "first_stages" not in case:
            continue
        for parity in ("p", "m"):
            for stage in ("inv", "scaled", "outer", "updated"):
                marks = ()
                if case["name"] == "one_inf3x4" and parity == "m" and stage == "outer":
                    marks = pytest.mark.xfail(
                        strict=True,
                        reason="Native inactive2x1*1x2 GEMM gives+0 for one negative-zero product; scalar IEEE outer gives-0. No guessed normalization.")
                cases.append(pytest.param(case, parity, stage, marks=marks,
                                         id=case["name"] + "-" + parity + "-" + stage))
    return cases


@pytest.mark.parametrize("case,parity,stage_name", _update_cases())
def test_native_update_uses_actual_captured_operands(case, parity, stage_name):
    stage = case["first_stages"]
    before = jnp.asarray(_array(stage["F" + parity + "_afterpole"]))
    col = jnp.asarray(_array(stage["col_" + parity])).reshape(-1)
    row = jnp.asarray(_array(stage["row_" + parity])).reshape(-1)
    pivot = jnp.asarray(_array(stage["ev" + parity])).reshape(())
    reciprocal = ieee.divide(jnp.float64(1), pivot)
    scaled = ieee.multiply(row, reciprocal)
    actual = {
        "inv": reciprocal,
        "scaled": scaled,
        "outer": ieee.multiply(col[:, None], scaled[None, :]),
        "updated": _update(before, col, row, pivot),
    }[stage_name]
    _words(actual, _array(stage[stage_name + "_" + parity]))


def _from_units(units):
    bits = abs(units) | ((1 << 63) if units < 0 else 0)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


@pytest.mark.parametrize("units", [
    [1, 0], [2, 1], [2, 3], [-1, 0], [1, -1], [0] * 32,
    [1 << 53], [(1 << 53) - 1, 1], [-(1 << 53)],
    list(range(-16, 17)), [1] * 4097,
])
def test_tiny_sum_and_mean_independent_exact_rational(units):
    values = np.asarray([_from_units(u) for u in units])
    eligible, total = _exact_tiny_sum(jnp.asarray(values))
    assert bool(eligible)
    exact = sum((Fraction(float(x)) for x in values), Fraction(0))
    mean = float(exact / len(units))
    if mean == 0:
        mean = math.copysign(0., -1 if exact < 0 else 1)
    _words(total, np.asarray(float(exact)))
    _words(_origin_mean(jnp.asarray(values)), np.asarray(mean))


@pytest.mark.parametrize("values", [
    [_from_units(1 << 53), _from_units(1)],
    [_from_units(1 << 53)] * 4097,
    [float("nan")], [float("inf")], [1.], [1., _from_units(1)],
])
def test_tiny_sum_bound_rejects_unsafe_or_nonfinite_domain(values):
    eligible, _ = _exact_tiny_sum(jnp.asarray(values))
    assert not bool(eligible)


def test_reciprocal_then_product_has_two_independent_roundings():
    reciprocal = float(Fraction(1, 6))
    expected = float(Fraction(5) * Fraction(reciprocal))
    direct = float(Fraction(5, 6))
    assert expected.hex() != direct.hex()
    _words(ieee.multiply(jnp.float64(5), ieee.divide(jnp.float64(1), jnp.float64(6))),
           np.asarray(expected))


def test_inactive_native_outer_gap_keeps_component_classes_strict():
    # Keep the known signedzero word diagnostic separate from class evidence;
    # its strictxfail must not hide NaN/Inf/zero-mask regressions.
    case = next(c for c in RAW["cases"] if c["name"] == "one_inf3x4")
    stage = case["first_stages"]
    col = jnp.asarray(_array(stage["col_m"])).reshape(-1)
    row = jnp.asarray(_array(stage["row_m"])).reshape(-1)
    pivot = jnp.asarray(_array(stage["evm"])).reshape(())
    actual = np.asarray(ieee.multiply(
        col[:, None], ieee.multiply(row, ieee.divide(jnp.float64(1), pivot))[None, :]))
    expected = _array(stage["outer_m"])
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_array_equal(actual == 0, expected == 0)


def test_public_tiny_origin_keeps_nonzero_pivot_words():
    from chebfunjax.diskfun.diskfun import Diskfun
    case = next(c for c in RAW["cases"] if c["name"] == "tiny_constant_2x6")
    result = Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)
    _words(result.pivots, _array(case["source"]["pivots"]))
    assert len(result) == 2
    assert result.nonzero_poles


def test_public_tiny_coupled_pivots_repeat_the_source_location():
    from chebfunjax.diskfun.diskfun import Diskfun
    # Exact source arithmetic: Fp=[2,4]units, Fm=[1,2]units. The common
    # pivot is [4,2]units, so alpha100 couples them and the rankcap ends GE.
    tiny = _from_units(1)
    values = jnp.asarray([[0., 0., 0., 0.], [tiny, 2*tiny, 3*tiny, 6*tiny]])
    result = Diskfun.from_values(values, preserve_lengths=True)
    _words(result.pivots, np.asarray([4*tiny, 2*tiny]))
    assert result.idx_plus == (0,) and result.idx_minus == (1,)
    assert len(result.pivot_locations) == 2
    assert result.pivot_locations[0] == result.pivot_locations[1]


@pytest.mark.parametrize("case", RAW["public_tiny_cases"], ids=lambda c: c["name"])
def test_native_public_tiny_metadata_and_component_classes(case):
    from chebfunjax.diskfun.diskfun import Diskfun
    native = case["public"]
    result = Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)
    assert len(result) == native["rank"]
    _words(result.pivots, _array(native["pivots"]))
    _words(np.asarray(result.pivot_locations), _array(native["locations"]))
    assert list(result.idx_plus) == (_array(native["idx_plus"]).reshape(-1) - 1).tolist()
    assert list(result.idx_minus) == (_array(native["idx_minus"]).reshape(-1) - 1).tolist()
    assert result.nonzero_poles == native["nonzero_poles"]
    for kind, techs in [("cols", result.cols), ("rows", result.rows)]:
        actual = np.column_stack([np.asarray(t.coeffs) for t in techs])
        expected = _array(native[kind])
        assert actual.shape == expected.shape
        expected_imag = np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                                    for w in native[kind]["im_hex"]]).reshape(expected.shape, order="F")
        for part, reference in [(actual.real, expected), (actual.imag, expected_imag)]:
            np.testing.assert_array_equal(np.isnan(part), np.isnan(reference))
            np.testing.assert_array_equal(np.isposinf(part), np.isposinf(reference))
            np.testing.assert_array_equal(np.isneginf(part), np.isneginf(reference))
