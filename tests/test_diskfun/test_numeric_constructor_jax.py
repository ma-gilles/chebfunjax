"""Native projection words and actual no-host numeric matrix construction."""

import ast
import importlib
import inspect
import json
import math
import struct
import textwrap
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._numeric_constructor import even_columns
from chebfunjax.diskfun.diskfun import Diskfun

ROOT = Path(__file__).parents[1] / "fixtures"
RAW = json.loads((ROOT / "disk_numeric_projection_native_20261018.json").read_text())


def _array(record):
    if not math.prod(record["shape"]):
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                       for w in record["re_hex"]]).reshape(record["shape"], order="F")


def _words(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.ascontiguousarray(actual[finite]).view(np.uint64),
                                  np.ascontiguousarray(expected[finite]).view(np.uint64))


@pytest.mark.parametrize("case", RAW["cases"], ids=lambda c: c["name"])
def test_native_final_even_projection_words(case):
    _words(even_columns(jnp.asarray(_array(case["input"]))),
           _array(case["output"]))


def test_independent_reciprocal_then_product_rounding_witness():
    delta = math.nextafter(1., math.inf)
    reciprocal = float(Fraction(1, 3))
    source_factor = float(Fraction.from_float(reciprocal) * Fraction.from_float(delta))
    direct_division = float(Fraction.from_float(delta) / 3)
    assert source_factor != direct_division
    coeffs = jnp.asarray([[0., delta], [0., 0.], [0., 0.],
                          [0., 0.], [0., 0.]])
    projected = even_columns(coeffs)
    expected = np.asarray([[0., float(Fraction.from_float(delta) + Fraction.from_float(source_factor))],
                           [0., 0.], [0., -source_factor], [0., 0.],
                           [0., source_factor]])
    _words(projected, expected)


PUBLIC = json.loads((ROOT / "disk_phaseone_jax_native_20261018.json").read_text())
NOHOST_CASES = PUBLIC["cases"] + PUBLIC["public_tiny_cases"]
EXPECTED_ERRORS = {'one_inf3x4': {'error_type': 'IndexError', 'error_message': 'MATLAB:badsubscript: Index in position 2 exceeds array bounds.'}, 'all_nan2x4': {'error_type': 'IndexError', 'error_message': 'MATLAB:badsubscript: Index in position 2 exceeds array bounds.'}, 'extreme2x4': {'error_type': 'IndexError', 'error_message': 'MATLAB:badsubscript: Index in position 2 exceeds array bounds.'}, 'mean_cancel2x8': {'error_type': 'IndexError', 'error_message': 'MATLAB:badsubscript: Index in position 2 exceeds array bounds.'}}


@pytest.mark.parametrize("case", NOHOST_CASES, ids=lambda c: c["name"])
def test_actual_numeric_matrix_constructor_without_host_numpy(case, monkeypatch):
    values = jnp.asarray(_array(case["input"]))

    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError("host NumPy accessed: " + name)

    for module_name in ("chebfunjax.diskfun.diskfun", "chebfunjax.tech.chebtech",
                        "chebfunjax.tech.trigtech"):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "np", ForbidNumpy())
    # Source error semantics stay qualified by separate unchanged native tests.
    try:
        function = Diskfun.from_values(values, preserve_lengths=True)
    except (IndexError, ValueError) as error:
        assert case["name"] in EXPECTED_ERRORS
        expected = EXPECTED_ERRORS[case["name"]]
        assert type(error).__name__ == expected["error_type"]
        assert str(error) == expected["error_message"]
        return
    assert case["name"] not in EXPECTED_ERRORS
    assert all(isinstance(t.coeffs, jax.Array) for t in function.cols + function.rows)
    assert isinstance(function.pivots, jax.Array)


def test_no_host_gate_negative_control(monkeypatch):
    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    values = jnp.asarray([[1., 1.], [2., 2.]])

    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError("host NumPy accessed: " + name)

    monkeypatch.setattr(module, "np", ForbidNumpy())
    with pytest.raises(AssertionError, match="host NumPy accessed: array"):
        Diskfun.from_values(values)


def test_scalar_route_remains_explicit_adaptive_adapter():
    source = ast.parse(textwrap.dedent(inspect.getsource(Diskfun.from_values.__func__)))
    # The new dispatch covers nonscalar matrices only; no constant replacement.
    dispatch = source.body[0].body[2]
    assert isinstance(dispatch, ast.If)
    assert ast.unparse(dispatch.test) == "preserve_lengths and values.size > 1"


@pytest.mark.parametrize("kind", ["origin_only", "zero_matrix", "complex_matrix", "empty_adapter"])
def test_special_constructor_routes_without_host_numpy(kind, monkeypatch):
    values = {
        "origin_only": jnp.asarray([[.7, -.2, 1., 0., 2., -.3]]),
        "zero_matrix": jnp.zeros((3, 4), jnp.float64),
        "complex_matrix": jnp.asarray([[1.+2j]*4, [2.-3j]*4, [3.+1j]*4]),
        "empty_adapter": jnp.empty((0, 0), jnp.float64),
    }[kind]

    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError("host NumPy accessed: " + name)

    for module_name in ("chebfunjax.diskfun.diskfun", "chebfunjax.tech.chebtech",
                        "chebfunjax.tech.trigtech"):
        monkeypatch.setattr(importlib.import_module(module_name), "np", ForbidNumpy())
    function = Diskfun.from_values(values, preserve_lengths=True)
    if kind == "empty_adapter":
        assert function.isempty()
    else:
        assert all(isinstance(t.coeffs, jax.Array) for t in function.cols + function.rows)
        assert isinstance(function.pivots, jax.Array)
