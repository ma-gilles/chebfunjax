"""Numeric matrix construction against untouched public source and analytic data.

Provenance
----------
MATLAB source : @chebfun2/{chebfun2,constructor}.m, @chebtech/extrapolate.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Capture: chebfun2_numeric_source_v1_matlab_cpu_20261007, all32 public cases.
The64eps coefficient comparison is a new rounding diagnostic, not a replacement
for any original constructor assertion or tolerance.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d import Chebfun2
from chebfunjax.chebfun2d import _numeric_constructor as numeric
from chebfunjax.chebfun2d import separable_approx as legacy
from chebfunjax.utils import transforms

_DATA = Path(__file__).parent / "data" / "numeric_source"


def _decode(packed):
    if 0 in packed["shape"]:
        assert all(h == "" for h in packed["real_hex"] + packed["imag_hex"])
        return np.empty(packed["shape"], dtype=np.float64)
    r = [struct.unpack(">d", bytes.fromhex(h))[0] for h in packed["real_hex"]]
    i = [struct.unpack(">d", bytes.fromhex(h))[0] for h in packed["imag_hex"]]
    # Preserve actual real dtype unless source contains an imaginary component.
    a = np.asarray(r)
    if any(v != 0 for v in i):
        a = a + 1j*np.asarray(i)
    return a.reshape(packed["shape"], order="F")


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("case", range(1, 17))
@pytest.mark.parametrize("kind", [1, 2])
def test_public_numeric_source_capture(disabled, case, kind, monkeypatch):
    record = json.loads((_DATA/f"case{case:02d}_{kind}.json").read_text())
    assert record["public_ok"]

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric constructor reached an inherited NumPy helper")

    monkeypatch.setattr(legacy, "_get_tol", forbidden)
    monkeypatch.setattr(legacy, "_complete_aca", forbidden)
    monkeypatch.setattr(transforms, "vals2coeffs", forbidden)
    with jax.disable_jit(disabled):
        f = Chebfun2.from_values(jnp.asarray(_decode(record["input"])),
                                domain=tuple(record["domain"]), trig=kind == 2)
    assert f.isempty() == record["isempty"]
    if f.isempty():
        return
    assert type(f.approx.cols[0]).__name__.lower() == record["column_tech"]
    assert type(f.approx.rows[0]).__name__.lower() == record["row_tech"]
    for factors, key in [(f.approx.cols, "columns"), (f.approx.rows, "rows")]:
        actual = np.stack([np.asarray(t.coeffs) for t in factors], axis=1)
        expected = _decode(record[key])
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=64*np.finfo(float).eps,
                                   atol=64*np.finfo(float).eps, equal_nan=True)
    pivots = _decode(record["pivots"]).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1/pivots
    inverse[np.isinf(np.abs(inverse))] = 0
    np.testing.assert_allclose(np.asarray(f.approx.pivots), inverse,
                               rtol=64*np.finfo(float).eps, atol=0, equal_nan=True)
    np.testing.assert_allclose(np.asarray(f(jnp.asarray([-1.7, .4, 2.1]),
                                          jnp.asarray([-.6, 1.2, 3.7]))),
                               _decode(record["query_values"]).ravel(),
                               rtol=64*np.finfo(float).eps,
                               atol=64*np.finfo(float).eps, equal_nan=True)


@pytest.mark.parametrize("disabled", [False, True])
def test_rectangular_tie_uses_column_major_pivots(disabled):
    a = jnp.asarray([[0., 1., 0.], [1., 0., 0.]])
    with jax.disable_jit(disabled):
        p, positions, rows, cols, failed = numeric.numeric_aca(a, 0.)
    assert positions == ((1, 0), (0, 1))
    assert not failed
    np.testing.assert_array_equal(cols @ jnp.diag(1/p) @ rows, a)


@pytest.mark.parametrize("disabled", [False, True])
def test_missing_row_is_extrapolated_jointly(disabled):
    # Any missing column masks the row for BOTH factor columns.
    values = jnp.asarray([[jnp.nan, 100.], [3., 7.]])
    with jax.disable_jit(disabled):
        fixed = numeric.extrapolate_cheb_values(values)
    np.testing.assert_array_equal(fixed, [[3., 7.], [3., 7.]])


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("trig", [False, True])
def test_complex_zero_dimensions_and_type(disabled, trig):
    with jax.disable_jit(disabled):
        f = Chebfun2.from_values(jnp.zeros((5, 4), dtype=jnp.complex128), trig=trig)
    assert f.approx.cols[0].coeffs.shape == (5,)
    assert f.approx.rows[0].coeffs.shape == (4,)
    np.testing.assert_array_equal(f.approx.pivots, [0.])
    assert type(f.approx.cols[0]).__name__ == ("Trigtech" if trig else "Chebtech2")


@pytest.mark.parametrize("disabled", [False, True])
def test_scalar_source_rejects_nonfinite_before_ge(disabled):
    with jax.disable_jit(disabled):
        for value, name in [(float("inf"), "inf"), (float("nan"), "nan")]:
            with pytest.raises(ValueError, match=f"CHEBFUN:CHEBFUN2:constructor:{name}"):
                Chebfun2.from_values(jnp.asarray([[value]]), trig=True)


@pytest.mark.parametrize("disabled", [False, True])
def test_complex_zero_aca_uses_source_real_storage(disabled):
    # Source completeACA uses zeros(...), not zeros(..., 'like', A).
    with jax.disable_jit(disabled):
        pivots, positions, rows, cols, failed = numeric.numeric_aca(
            jnp.zeros((3, 4), dtype=jnp.complex128), 0.)
    assert positions == ((0, 0),)
    assert not failed
    for array, shape in [(pivots, (1,)), (rows, (1, 4)), (cols, (3, 1))]:
        assert array.shape == shape
        assert array.dtype == jnp.float64
        np.testing.assert_array_equal(array, jnp.zeros(shape))
