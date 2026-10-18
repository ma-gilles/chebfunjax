"""Explicit source Chebfun2 preferences with fixed sizes and numeric values.

Provenance
----------
MATLAB source: @chebfun2/constructor.m parseInputs/constructFromDouble.
Chebfun commit:7574c77680d7e82b79626300bf255498271a72df.
Native capture18cases: chebfun2_fixed_pref_native_run001_20261018.
The100eps component bounds are authored diagnostics, not replacements for
any original MATLAB constructor predicate or tolerance.
"""

import copy
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

DATA = json.loads(
    (Path(__file__).with_name("data") / "fixed_preferences_native_20261018.json").read_text()
)


def _decode(packed):
    if 0 in packed["shape"]:
        assert all(word == "" for word in packed["real_hex"] + packed["imag_hex"])
        return np.empty(packed["shape"], dtype=np.float64)
    real = np.array([struct.unpack(">d", bytes.fromhex(x))[0] for x in packed["real_hex"]])
    imag = np.array([struct.unpack(">d", bytes.fromhex(x))[0] for x in packed["imag_hex"]])
    value = np.empty(real.shape, dtype=np.complex128)
    value.real = real
    value.imag = imag
    if np.all(imag == 0):
        value = real
    return value.reshape(packed["shape"], order="F")


def _bounded(actual, packed):
    expected = _decode(packed)
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    bound = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(expected))))
    assert np.max(np.abs(actual - expected)) < bound


@pytest.mark.parametrize("row", DATA["records"], ids=lambda r: f"{r['tech']}-case{r['case']}")
def test_explicit_preferences_native_fixed_and_numeric(row):
    pref = ChebfunPref(tech=row["tech"])
    before = copy.deepcopy(pref.__dict__)
    domain = tuple(row["domain"])
    case = row["case"]
    if row["tech"] == "trigtech":

        def op(x, y):
            return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) + 0.25 * jnp.cos(2 * jnp.pi * x)
    else:

        def op(x, y):
            return 1 + x + 2 * y + x * y + x**2 * y

    if case == 6:

        def op(x, y):
            return 1 + x / 8 - y / 4 + x * y / 16

    kwargs = {"domain": domain, "pref": pref}
    if case in [3, 4]:
        matrix = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0], [3.0, 1.0, -2.0], [0.0, 2.0, 3.0]])
        if case == 4:
            matrix = matrix + 1j * jnp.flipud(matrix) / 8
        result = chebfun2(matrix, **kwargs)
    else:
        if case == 5:
            kwargs["rank"] = 1
        result = chebfun2(op, n=tuple(row["input_sizes"]), **kwargs)
    assert pref.__dict__ == before
    assert result.rank == row["rank"]
    assert tuple(map(float, result.domain)) == domain
    expected_tech = {"chebtech1": "cheb1", "chebtech2": "cheb", "trigtech": "trig"}[row["tech"]]
    assert result.approx.techs == (expected_tech, expected_tech)
    assert all(
        type(t).__name__.lower() == row["tech"] for t in result.approx.cols + result.approx.rows
    )
    _bounded(jnp.stack([t.coeffs for t in result.approx.cols], axis=1), row["cols"])
    _bounded(jnp.stack([t.coeffs for t in result.approx.rows], axis=1), row["rows"])
    _bounded(jnp.asarray(result.approx.pivot_values).reshape(1, -1), row["pivots"])
    xa, xb, ya, yb = domain
    values = result(jnp.asarray([xa, (xa + xb) / 2, xb]), jnp.asarray([ya, (ya + yb) / 2, yb]))
    _bounded(jnp.asarray(values).reshape(1, -1), row["samples"])


EXTRA = json.loads(
    (Path(__file__).with_name("data") / "preferences_extra_native_20261018.json").read_text()
)


@pytest.mark.parametrize(
    "row", [r for r in EXTRA["records"] if r["ok"]], ids=lambda r: f"{r['tech']}-extra{r['case']}"
)
def test_explicit_preferences_native_rank_scalar_flags_and_tolerance(row):
    pref = ChebfunPref(tech=row["tech"])
    domain = tuple(row["domain"])
    case = row["case"]
    if row["tech"] == "trigtech":

        def op(x, y):
            return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) + 0.25 * jnp.cos(2 * jnp.pi * x)
    else:

        def op(x, y):
            return 1 + x + 2 * y + x * y + x**2 * y

    kwargs = {"domain": domain, "pref": pref}
    if case in [1, 2, 5, 7]:
        kwargs["n"] = (5, 6)
    if case == 1:
        kwargs["rank"] = 4
    if case == 2:
        kwargs["rank"] = 0
    if case == 3:
        op = 2.0
    if case == 4:
        op = jnp.zeros((4, 3))
    if case == 5:
        kwargs["trigx"] = True
    if case == 6:
        pref.cheb2Prefs.chebfun2eps = 1 / 64
        matrix = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0], [3.0, 1.0, -2.0], [0.0, 2.0, 3.0]])
        op = jnp.ones((4, 1)) * jnp.asarray([[1.0, 2.0, 3.0]]) + 2**-12 * matrix
    if case == 7:
        op = "1+x/8-y/4+x.*y/16"
    if case == 8:
        kwargs["rank"] = 1
    saved = copy.deepcopy(pref.__dict__)
    result = chebfun2(op, **kwargs)
    assert pref.__dict__ == saved
    assert result.rank == row["rank"]
    assert tuple(map(float, result.domain)) == domain
    assert all(type(t).__name__.lower() == row["column_tech"] for t in result.approx.cols)
    assert all(type(t).__name__.lower() == row["row_tech"] for t in result.approx.rows)
    _bounded(jnp.stack([t.coeffs for t in result.approx.cols], axis=1), row["cols"])
    _bounded(jnp.stack([t.coeffs for t in result.approx.rows], axis=1), row["rows"])
    _bounded(jnp.asarray(result.approx.pivot_values).reshape(1, -1), row["pivots"])
    values = result(jnp.asarray([-1.0, 0.0, 1.0]), jnp.asarray([-1.0, 0.0, 1.0]))
    _bounded(jnp.asarray(values).reshape(1, -1), row["samples"])


RANK_NATIVE = json.loads(
    (
        Path(__file__).with_name("data") / "preferences_rank_quasimatrix_native_20261018.json"
    ).read_text()
)


@pytest.mark.parametrize("row", RANK_NATIVE["records"], ids=lambda r: f"mixed-padding{r['case']}")
def test_native_periodic_rank_padding_retains_nonperiodic_zero_factors(row):
    pref = ChebfunPref(tech="trigtech" if row["case"] == 1 else "chebtech1")

    def op(x, y):
        return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) + 0.25 * jnp.cos(2 * jnp.pi * x)

    result = chebfun2(op, n=(5, 6), rank=4, pref=pref, trigx=row["case"] == 2)
    assert result.rank == row["rank"] == 4
    for actual, expected in [(result.approx.cols, row["cols"]), (result.approx.rows, row["rows"])]:
        assert len(actual) == len(expected)
        for factor, record in zip(actual, expected):
            assert type(factor).__name__.lower() == record["tech"]
            _bounded(factor.coeffs.reshape(-1, 1), record["coeffs"])
    _bounded(jnp.asarray(result.approx.pivot_values).reshape(1, -1), row["pivots"])
    _bounded(
        result(jnp.asarray([-1.0, 0.0, 1.0]), jnp.asarray([-1.0, 0.0, 1.0])).reshape(1, -1),
        row["samples"],
    )


METADATA_NATIVE = json.loads(
    (Path(__file__).with_name("data") / "preferences_metadata_native_20261018.json").read_text()
)


@pytest.mark.parametrize(
    "row", METADATA_NATIVE["records"], ids=lambda r: f"{r['tech']}-metadata{r['case']}"
)
def test_source_pivot_location_lifecycle_and_scalar_reset(row):
    pref = ChebfunPref(tech=row["tech"])
    if row["tech"] == "trigtech":

        def op(x, y):
            return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) + 0.25 * jnp.cos(2 * jnp.pi * x)
    else:

        def op(x, y):
            return 1 + x + 2 * y + x * y + x**2 * y

    case = row["case"]
    kwargs = {"domain": tuple(row["domain"]), "pref": pref}
    if case == 1:
        op = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0], [3.0, 1.0, -2.0], [0.0, 2.0, 3.0]])
    if case in [2, 9]:
        kwargs["n"] = (5, 6)
    if case == 4:
        kwargs["rank"] = 1
    if case in [5, 6, 7, 8]:
        op = {5: 2.0, 6: 0.0, 7: -0.0, 8: 2 + 3j}[case]
    if case == 9:
        kwargs["rank"] = 4
    if case == 10:
        op = jnp.zeros((4, 3))
    saved = copy.deepcopy(pref.__dict__)
    result = chebfun2(op, **kwargs)
    assert pref.__dict__ == saved
    assert result.rank == row["rank"]
    expected = _decode(row["locations"])
    if expected.size:
        actual = np.asarray(result.pivot_locations)
        assert actual.shape == expected.shape
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    else:
        assert result.pivot_locations == ()
    actual = np.asarray(result.approx.pivot_values).reshape(1, -1)
    expected = _decode(row["pivots"])
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isposinf(actual.real), np.isposinf(expected.real))
    assert np.array_equal(np.isneginf(actual.real), np.isneginf(expected.real))
    finite = np.isfinite(expected)
    if np.any(finite):
        bound = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(expected[finite]))))
        assert np.max(np.abs(actual[finite] - expected[finite])) < bound
    xa, xb, ya, yb = row["domain"]
    values = result(jnp.asarray([xa, (xa + xb) / 2, xb]), jnp.asarray([ya, (ya + yb) / 2, yb]))
    _bounded(values.reshape(1, -1), row["samples"])
