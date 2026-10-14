"""TreeVar direct degree kernels against existing R2025b raw36 capture.

These supplemental comparisons retain the native packet component/storage
bounds. They do not replace the 49 original TreeVar predicates.
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
                   "direct_degree_native.json").read_text())
NAMES = tuple(DATA["outputs_by_method"])
FRONTIER = json.loads((Path(__file__).parent / "data" /
                       "direct_degree_frontier_native.json").read_text())
RECORDS = [
    {"method": method, "input": DATA["inputs"][i], "expected": expected}
    for method in NAMES
    for i, expected in enumerate(DATA["outputs_by_method"][method])
]
RECORDS += [
    {"method": expected["method"],
     "input": dict(FRONTIER["inputs"][expected["input_index"]],
                   label="frontier_"+str(expected["input_index"])),
     "expected": expected}
    for expected in FRONTIER["records"]
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
    elif np.isfinite(target):
        # Adjacent quadrants must stay distinct even when the ordinary
        # absolute component bound would permit snapping them to zero.
        assert value != 0
        assert np.signbit(value) == np.signbit(target)


@pytest.mark.parametrize("row", RECORDS,
                         ids=[r["method"]+"-"+r["input"]["label"]
                              for r in RECORDS])
def test_treevar_direct_degree_native_boundaries(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    _check_component(float(np.real(got)), _decode(expected["real_hex"]))
    _check_component(float(np.imag(got)), _decode(expected["imag_hex"]))




@pytest.mark.parametrize('method', NAMES)
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
def test_degree_precision_and_matrix_shape(method, dtype):
    """Independent ordinary-angle reference; supplemental dtype adaptation."""
    iscomplex = jnp.issubdtype(dtype, jnp.complexfloating)
    values = [[.4+.3j, -.7+.2j], [3j, -.25j]] if iscomplex else [[.4, -.7], [2., -3.]]
    x = jnp.asarray(values, dtype=dtype)
    got = _UNARY_FNS[method](x)
    assert got.shape == x.shape and got.dtype == dtype
    host = np.asarray(x).astype(complex if iscomplex else float)
    op = {'sind': np.sin, 'cosd': np.cos, 'tand': np.tan,
          'cscd': np.sin, 'secd': np.cos, 'cotd': np.tan}[method]
    expected = op(host*(np.pi/180))
    if method in ('cscd', 'secd', 'cotd'):
        expected = 1/expected
    bound = 16*np.finfo(np.asarray(jnp.real(x)).dtype).eps
    assert np.max(np.abs(np.asarray(got)-expected)/np.maximum(1., np.abs(expected))) < bound


@pytest.mark.parametrize('method', NAMES)
def test_degree_integer_input(method):
    got = _UNARY_FNS[method](jnp.asarray([1, 2], dtype=jnp.int32))
    expected = _UNARY_FNS[method](jnp.asarray([1., 2.]))
    assert jnp.issubdtype(got.dtype, jnp.floating)
    np.testing.assert_array_equal(got, expected)


def test_treevar_tand_retains_native_tiny_complex_component():
    """Retained independent MATLAB TAND capture, reused by degree_tangent.

    Same input/output as test_chebfun3t_tand_contract.py; this checks the
    public TreeVar mapping does not erase the kernel's subnormal component.
    """
    x = jnp.asarray(complex(np.finfo(float).eps, np.finfo(float).tiny))
    got = np.asarray(_UNARY_FNS['tand'](x))
    expected = np.asarray(complex(3.8754094422318132e-18, 3.88348649310052e-310))
    assert np.iscomplexobj(got)
    np.testing.assert_array_equal(got.reshape(1).view(np.uint64),
                                  expected.reshape(1).view(np.uint64))
