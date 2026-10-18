"""Literal native numeric rank and missing-data branch controls.

Original tests/bounds remain unchanged. No tolerance oracle is injected into
production: captured tolerance is only an immutable diagnostic record.
"""

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._numeric_extrema import source_abs_extrema
from chebfunjax.diskfun.diskfun import Diskfun, _get_tol, _phase_one_matrix_disk

RAW = json.loads(
    (
        Path(__file__).parents[1] / "fixtures" / "disk_numeric_branch_native_20261018.json"
    ).read_text()
)
CASES = {c["name"]: c for c in RAW["cases"]}


def _array(record):
    if np.prod(record["shape"]) == 0:
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0] for w in record["re_hex"]]).reshape(
        record["shape"], order="F"
    )


def _imaginary(record):
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0] for w in record["im_hex"]]).reshape(
        record["shape"], order="F"
    )


def _words(actual, expected):
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint64), np.ascontiguousarray(expected).view(np.uint64)
    )


@pytest.mark.parametrize("case", RAW["extrema_cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("kind", ["p", "m"])
def test_native_absolute_max_and_first_fortran_index(case, kind):
    stage = case["first_stages"]
    _, maximum, index = source_abs_extrema(jnp.asarray(_array(stage["F" + kind + "_afterpole"])))
    _words(np.asarray(maximum).reshape(1, 1), _array(stage["max" + kind]))
    assert int(index) == int(_array(stage["idx" + kind]).item()) - 1


@pytest.mark.parametrize(
    "values,minimum,maximum,index",
    [
        ([[float("nan"), -0.0]], 0.0, 0.0, 1),
        ([[1e-323, 5e-324]], 5e-324, 1e-323, 0),
        ([[1.0, 3.0], [3.0, 2.0]], 1.0, 3.0, 1),
        ([[float("nan"), -0.0, 1e-323], [-2.0, float("inf"), 0.0]], 0.0, float("inf"), 3),
    ],
)
def test_independent_magnitude_order_zero_nan_and_subnormal(values, minimum, maximum, index):
    low, high, idx = source_abs_extrema(jnp.asarray(values))
    _words(low, np.asarray(minimum))
    _words(high, np.asarray(maximum))
    assert int(idx) == index


@pytest.mark.parametrize("name", ["rank_cap3x8", "one_nan3x4"])
def test_public_numeric_native_rank_pivots_and_component_classes(name):
    case = CASES[name]
    native = case["public"]
    g = Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)
    assert len(g) == native["rank"]
    _words(np.asarray(g.pivots).reshape(-1, 1), _array(native["pivots"]))
    _words(np.asarray(g.pivot_locations), _array(native["locations"]))
    assert list(g.idx_plus) == (_array(native["idx_plus"]).reshape(-1) - 1).tolist()
    assert list(g.idx_minus) == (_array(native["idx_minus"]).reshape(-1) - 1).tolist()
    assert g.nonzero_poles == native["nonzero_poles"]
    for kind, techs in [("cols", g.cols), ("rows", g.rows)]:
        actual = np.column_stack([np.asarray(t.coeffs) for t in techs])
        for part, expected in [
            (actual.real, _array(native[kind])),
            (actual.imag.copy(), _imaginary(native[kind])),
        ]:
            assert part.shape == expected.shape
            np.testing.assert_array_equal(np.isnan(part), np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(part), np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(part), np.isneginf(expected))


@pytest.mark.parametrize("name", ["one_inf3x4", "all_nan2x4"])
def test_public_numeric_preserves_native_exhausted_factor_error(name):
    case = CASES[name]
    assert case["public_error"]["identifier"] == "MATLAB:badsubscript"
    with pytest.raises(
        IndexError, match="MATLAB:badsubscript: Index in position 2 exceeds array bounds"
    ):
        Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)


def test_nan_tolerance_reduction_keeps_native_valid_nonempty_route():
    case = CASES["one_nan3x4"]
    values = _array(case["input"])
    old = _get_tol(
        values, 2 * np.pi / values.shape[1], np.pi / (values.shape[0] - 1), np.finfo(float).eps
    )[0]
    source = _get_tol(
        values,
        2 * np.pi / values.shape[1],
        np.pi / (values.shape[0] - 1),
        np.finfo(float).eps,
        source_numeric=True,
    )[0]
    assert np.isnan(old) and np.isfinite(source) and source > 0


def test_native_subnormal_reciprocal_rank():
    case = CASES["subnormal2x4"]
    g = Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)
    _words(np.asarray(g.pivots).reshape(-1, 1), _array(case["public"]["pivots"]))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "rank_cap3x8",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Inherited post-GE Cheb transform1ulp word gap; native raw GE words remain strict",
            ),
        ),
        pytest.param(
            "one_nan3x4",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Remaining finite populate/interpolation/transform word gap; source structure and NaN classes remain strict",
            ),
        ),
    ],
)
def test_remaining_native_public_finite_factor_words(name):
    case = CASES[name]
    native = case["public"]
    g = Diskfun.from_values(jnp.asarray(_array(case["input"])), preserve_lengths=True)
    for kind, techs in [("cols", g.cols), ("rows", g.rows)]:
        actual = np.column_stack([np.asarray(t.coeffs) for t in techs])
        for part, expected in [
            (actual.real, _array(native[kind])),
            (actual.imag.copy(), _imaginary(native[kind])),
        ]:
            finite = np.isfinite(expected)
            _words(part[finite], expected[finite])


@pytest.mark.parametrize("name", ["rank_cap3x8", "one_nan3x4"])
def test_source_raw_ge_words_with_actual_production_tolerance(name):
    # Compute production tolerance; no captured source tolerance injection.
    case = next(c for c in RAW["extrema_cases"] if c["name"] == name)
    F = _array(case["input"])
    m, n = F.shape
    tol = _get_tol(F, 2 * np.pi / n, np.pi / (m - 1), np.finfo(float).eps, source_numeric=True)[0]
    result = _phase_one_matrix_disk(F, tol, 100.0, source_numeric=True)
    native = case["source"]
    for value, key in [
        (result[3], "cols"),
        (result[5], "rows"),
        (result[4].reshape(-1, 1), "pivots"),
        (result[1], "pivot_array"),
    ]:
        _words(value, _array(native[key]))
    _words(result[0].astype(float), _array(native["pivot_indices"]) - 1)


def test_source_raw_zero_keeps_native_nonnegative_grid_length():
    case = next(c for c in RAW["extrema_cases"] if c["name"] == "zero5x4")
    F = _array(case["input"])
    result = _phase_one_matrix_disk(F, 0.0, 100.0, source_numeric=True)
    _words(result[3], _array(case["source"]["cols"]))
    _words(result[5], _array(case["source"]["rows"]))
    _words(result[4].reshape(-1, 1), _array(case["source"]["pivots"]))


def test_inf_only_at_removed_origin_follows_source_zero_not_blanket_error():
    # Literal source: tolInf, poleInf is not >tolInf; discard origin;
    # both residuals0 then sourcezero branch retains original radial length.
    values = jnp.asarray([[jnp.inf] * 4, [0.0] * 4, [0.0] * 4])
    with np.errstate(invalid="ignore"):
        g = Diskfun.from_values(values, preserve_lengths=True)
    assert len(g) == 1 and g.cols[0].n == 3 and g.rows[0].n == 4
    _words(g.cols[0].coeffs, np.zeros(3))
    _words(np.asarray(g.rows[0].coeffs).real, np.zeros(4))
    _words(np.asarray(g.rows[0].coeffs).imag.copy(), np.zeros(4))
    _words(g.pivots, np.asarray([np.inf]))
    assert not g.nonzero_poles and g.idx_plus == (0,) and g.idx_minus == ()
