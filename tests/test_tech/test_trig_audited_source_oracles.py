"""Expected-only MATLAB oracles; analytic inputs are written independently.

Provenance
----------
MATLAB source : @trigtech/refine.m, @trigtech/standardCheck.m,
    @trigtech/sampleTest.m, trigpts.m, @trigtech/trigpts.m
Chebfun commit: 7574c77
Fixtures record actual R2025b execution with source/input hash guards.
These new output comparisons use 64eps times each column's input amplitude;
no original MATLAB test tolerance is changed. Grid/callback words are exact.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_abs_coeffs_for_chop,
    _trig_standard_check,
)
from chebfunjax.tech.trigtech import (
    trigpts as static_trigpts,
)
from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import trigpts as global_trigpts

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_CONSTRUCTOR = json.loads((_FIXTURES / "trig_constructor_source_2025b.json").read_text())
_GRIDS = json.loads((_FIXTURES / "trig_grid_source_2025b.json").read_text())
# Inputs are literal and do not come from expected outputs or fitted coefficients.
_CASES = ((1, 1.), (1, 1e-12), (2, 1.), (2, 1e-12), (3, 1.), (3, 1e-12))
_GRID_SIZES = (0, 1, 2, 3, 7, 8, 16, 17, 32, 33, 64, 100, 129, 1024)


def _decode(record):
    def scalar(value):
        if isinstance(value, dict):
            return complex(float.fromhex(value["real_hex"]), float.fromhex(value["imag_hex"]))
        return float.fromhex(value)

    values = np.asarray(record["values"], dtype=object)
    return np.asarray([scalar(v) for v in values.ravel()]).reshape(record["shape"])


def _assert_words(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _literal_input(x, kind, scale):
    if kind == 1:
        return scale * jnp.sin(17 * jnp.pi * x)
    if kind == 2:
        return scale * jnp.exp(17j * jnp.pi * x)
    return jnp.stack((jnp.sin(jnp.pi * x), scale * jnp.sin(17 * jnp.pi * x)), axis=-1)


@pytest.mark.parametrize("case_index", range(6))
def test_constructor_matches_actual_source_sites_and_outputs(case_index):
    kind, scale = _CASES[case_index]
    seen = []

    def op(x):
        seen.append(np.asarray(x).copy())
        return _literal_input(x, kind, scale)

    tech = Trigtech.from_function(op)
    expected = _CONSTRUCTOR["source_constructor_cases"][case_index]
    assert expected["case_index"] == case_index
    assert tech.ishappy == expected["ishappy"]
    assert tech.coeffs.shape[0] == expected["result_piece_length"]
    calls = expected["callback_points_in_call_order"]
    assert len(seen) == len(calls)
    for actual, wanted in zip(seen, calls, strict=True):
        # The source scalar probe is the Python API's length-one array.
        # Normalize that rank adapter only; every point and call is checked.
        assert actual.size == np.prod(wanted["shape"], dtype=int)
        _assert_words(actual, _decode(wanted))
    query = jnp.asarray([-.91, -.31, .07, .61, .93])
    source_values = _decode(expected["source_chebfun_values"])
    actual = np.asarray(tech(query))
    assert actual.shape == source_values.shape
    amplitudes = np.asarray([1., scale]) if kind == 3 else scale
    assert np.all(np.abs(actual-source_values) <= 64*np.finfo(float).eps*amplitudes)
    # In kind3/scale1e-12 the source itself accepts the aliased tiny column.
    # Thus the oracle is its interpolant, not the ideal analytic tiny column.


def _standard_input(x, opid):
    if opid == 1:
        return jnp.sin(8*jnp.pi*x)
    if opid == 2:
        return jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(3*jnp.pi*x),
                          jnp.sin(7*jnp.pi*x)+jnp.cos(7*jnp.pi*x)), axis=1)
    if opid == 3:
        return jnp.exp(3j*jnp.pi*x)+.25*jnp.cos(5*jnp.pi*x)
    return 1/(2-jnp.cos(jnp.pi*x))


@pytest.mark.parametrize("n", (16, 32, 33))
@pytest.mark.parametrize("opid", (1, 2, 3, 4))
@pytest.mark.parametrize("scale", (1, 8))
def test_existing_source_standardcheck_decisions(n, opid, scale):
    # These 24 common decisions already matched before the patch.
    expected = next(item for item in _CONSTRUCTOR["literal_standardcheck_decisions"]
                    if (item["n"], item["opid"], item["scale"]) == (n, opid, scale))
    values = _standard_input(static_trigpts(n), opid)
    tech = Trigtech.from_values(values)
    local = jnp.max(jnp.abs(values), axis=0)
    happy, cutoff = _trig_standard_check(tech.coeffs, values, np.finfo(float).eps, scale*local)
    assert happy == expected["happy"]
    assert cutoff == expected["cutoff"]
    coefficients = tech.coeffs[:, None] if tech.coeffs.ndim == 1 else tech.coeffs
    tol = np.broadcast_to(np.asarray(np.finfo(float).eps*(scale*local)/local),
                          (coefficients.shape[1],))
    raw = [standard_chop(_trig_abs_coeffs_for_chop(coefficients[:, j]), float(tol[j]))
           for j in range(coefficients.shape[1])]
    np.testing.assert_array_equal(raw, np.atleast_1d(expected["raw_cutoffs"]))


@pytest.mark.parametrize("factor,expected", ((.5, True), (2., False)))
def test_actual_source_sampletest_threshold(factor, expected):
    values = jnp.ones(33)
    tech = Trigtech.from_values(values)
    seen = []

    def op(x):
        seen.append(np.asarray(x).copy())
        return jnp.ones_like(x)+factor*np.sqrt(np.finfo(float).eps)

    happy, _ = Trigtech.happiness_check(tech.coeffs, values, op=op)
    assert happy is expected
    assert len(seen) == 1
    _assert_words(seen[0], [-.357998918959666, .036785641195074])


@pytest.mark.parametrize("n", _GRID_SIZES)
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_public_global_grid_actual_source_words(n, compiled):
    expected = next(item["expected_source_outputs_hex"] for item in _GRIDS["cases"]
                    if item["n"] == n)
    for interval, prefix in ((None, "global"), ((-2., 3.), "mapped")):
        def compute():
            return global_trigpts(n, interval=interval)

        points, weights = jax.jit(compute)() if compiled else compute()
        for actual, suffix in ((points, "x"), (weights, "w")):
            wanted = np.asarray([int(word, 16) for word in expected[f"{prefix}_{suffix}"]],
                                dtype=np.uint64)
            assert np.asarray(actual).shape == (n,)
            np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), wanted)


@pytest.mark.parametrize("n", _GRID_SIZES)
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_static_technology_grid_actual_source_words(n, compiled):
    expected = next(item["expected_source_outputs_hex"] for item in _GRIDS["cases"]
                    if item["n"] == n)
    def compute():
        return static_trigpts(n)
    points = jax.jit(compute)() if compiled else compute()
    wanted = np.asarray([int(word, 16) for word in expected["static_x"]], dtype=np.uint64)
    assert np.asarray(points).shape == (n,)
    np.testing.assert_array_equal(np.asarray(points).view(np.uint64), wanted)


@pytest.mark.parametrize("n", (1, 7, 17, 33))
def test_fixed_constructor_uses_static_source_points(n):
    seen = []
    def op(x):
        seen.append(np.asarray(x).copy())
        return jnp.cos(3*jnp.pi*x)
    tech = Trigtech.from_function(op, n=n)
    expected = next(item["expected_source_outputs_hex"] for item in _GRIDS["cases"]
                    if item["n"] == n)
    words = np.asarray([int(word, 16) for word in expected["static_x"]], dtype=np.uint64)
    # Pinned trigtech.m fixed-size branch calls trigtech.trigpts(n), whereas
    # refine.m calls global trigpts. Captured source nodes are expectations.
    # The existing Python validation probe is an adapter; this control
    # qualifies the final sampling grid, not its complete fixed-n trace.
    _assert_words(seen[-1], np.r_[words.view(np.float64), 1.])
    assert tech.ishappy and tech.coeffs.shape[0] == n
