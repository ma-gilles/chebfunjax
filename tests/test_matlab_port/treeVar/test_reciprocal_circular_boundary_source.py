"""TreeVar reciprocal circular kernels against existing R2025b raw36 capture.

These supplemental captured cases complement the unchanged
49 original TreeVar predicates. All four methods retain the native packet component/storage bounds.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

DATA = json.loads((Path(__file__).parent / "data" /
                   "reciprocal_circular_native.json").read_text())
NAMES = tuple(DATA["outputs_by_method"])
RECORDS = [
    {"method": method, "input": DATA["inputs"][i], "expected": expected}
    for method in NAMES
    for i, expected in enumerate(DATA["outputs_by_method"][method])
]


def _decode(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _input(row):
    raw = row["input"]
    real, imag = _decode(raw["real_hex"]), _decode(raw["imag_hex"])
    return jnp.asarray(real if raw["isreal"] else complex(real, imag))


def _check_component(value, target):
    assert np.isnan(value) == np.isnan(target)
    assert np.isposinf(value) == np.isposinf(target)
    assert np.isneginf(value) == np.isneginf(target)
    if np.isfinite(target):
        assert abs(value-target) <= 8*np.finfo(float).eps*max(1., abs(target))
    if target == 0:
        assert value == 0
        assert np.signbit(value) == np.signbit(target)


@pytest.mark.parametrize("row", RECORDS,
                         ids=[r["method"]+"-"+r["input"]["label"]
                              for r in RECORDS])
def test_treevar_reciprocal_circular_native36(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    _check_component(float(np.real(got)), _decode(expected["real_hex"]))
    _check_component(float(np.imag(got)), _decode(expected["imag_hex"]))



@pytest.mark.parametrize("name", ["acsc", "acscd", "asec", "asecd"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
def test_reciprocal_circular_dtype_shape_identity(name, dtype):
    """Supplemental precision/shape control, independent inverse identity."""
    complex_input = jnp.issubdtype(dtype, jnp.complexfloating)
    x = jnp.asarray([[2.+.3j, -.7+.2j], [3j, -.25j]] if complex_input
                    else [[2., -2.], [.5, -.5]], dtype=dtype)
    result = _UNARY_FNS[name](x)
    expected_dtype = jnp.complex64 if dtype in (jnp.float32, jnp.complex64) else jnp.complex128
    assert result.shape == x.shape
    assert result.dtype == expected_dtype
    angle = result * (jnp.pi/180) if name.endswith("d") else result
    inverse = jnp.cos(angle) if name.startswith("asec") else jnp.sin(angle)
    # Four elementary operations plus inverse/trig composition; supplemental,
    # not a substitute for any original or captured native predicate.
    bound = 16*jnp.finfo(jnp.real(x).dtype).eps
    assert jnp.max(jnp.abs(inverse*x-1)) < bound
    if complex_input and name.startswith("acsc"):
        axis_real = np.asarray(jnp.real(result[1]))
        assert np.all(axis_real == 0) and not np.any(np.signbit(axis_real))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("name", ["acsc", "acscd", "asec", "asecd"])
def test_reciprocal_circular_real_storage(dtype, name):
    x = jnp.asarray([2., -2., 3.], dtype=dtype)
    result = _UNARY_FNS[name](x)
    assert result.dtype == dtype and result.shape == x.shape
