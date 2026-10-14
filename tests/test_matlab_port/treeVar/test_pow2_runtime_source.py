"""TreeVar POW2 native boundary-storage controls.

Provenance
----------
MATLAB source: @treeVar/treeVar.m dispatches unary POW2 to the numeric
    builtin; @chebfun/pow2.m defines the one-argument function wrapper as
    ``power(2,f)``. Chebfun commit 7574c77, MATLAB R2025b.
Fixture: tests/fixtures/treevar_unary_boundary_native_20261014.json,
    captured 49 unary methods over 36 scalar real/complex boundary inputs.
This test checks numeric scalar storage classes and nonfinite component
classes. Finite values use the established 8-eps scaled primitive-boundary
guard used by adjacent TreeVar log-family controls; this capture itself did
not define a separate tolerance predicate.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS
from chebfunjax.utils.numeric_sign import real_sign

FIXTURE = (Path(__file__).parents[2] / "fixtures" /
           "treevar_unary_boundary_native_20261014.json")
_DATA = json.loads(FIXTURE.read_text())
_POW2 = _DATA["names"].index("pow2")
_METHODS = len(_DATA["names"])
_CASES = [(i, item, _DATA["records"][i*_METHODS + _POW2])
          for i, item in enumerate(_DATA["inputs"])]


def _float_from_hex(value):
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _input(item):
    real = _float_from_hex(item["real_hex"])
    imag = _float_from_hex(item["imag_hex"])
    return jnp.asarray(real if item["isreal"] else complex(real, imag))


def _component_class(value):
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    if value == 0:
        return "-zero" if np.signbit(value) else "+zero"
    return "finite"


@pytest.mark.parametrize("index,item,expected", _CASES,
                         ids=[item["label"] for item in _DATA["inputs"]])
def test_pow2_native_storage_boundary(index, item, expected):
    """Preserve captured scalar real/complex and nonfinite component storage."""
    got = np.asarray(_UNARY_FNS["pow2"](_input(item)))
    assert got.shape == ()
    assert got.dtype in (np.dtype("float64"), np.dtype("complex128"))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    expected_real = _float_from_hex(expected["real_hex"])
    expected_imag = _float_from_hex(expected["imag_hex"])
    for part, target in (("real", expected_real), ("imag", expected_imag)):
        value = float(getattr(got, part))
        assert _component_class(value) == _component_class(target)
        if np.isfinite(target):
            assert abs(value - target) <= (
                8 * np.finfo(float).eps * max(1., abs(target)))
        # Captured exact zeros retain their IEEE sign bit.
        if target == 0:
            assert np.signbit(value) == np.signbit(target)


def test_pow2_complex_matrix_storage_and_subnormal_axis():
    """Keep array storage complex when only some entries lie on the real axis."""
    mixed = jnp.asarray([[1. + 0.j, 2. + .5j], [-1. - 0.j, 0. + 1.j]],
                        dtype=jnp.complex128)
    got = _UNARY_FNS["pow2"](mixed)
    assert got.shape == mixed.shape
    assert got.dtype == jnp.complex128
    assert got[0, 0] == 2. + 0.j
    assert got[1, 0] == .5 + 0.j
    assert jnp.all(jnp.isfinite(jnp.real(got)))
    assert jnp.all(jnp.isfinite(jnp.imag(got)))
    assert got[0, 1].imag != 0
    assert got[1, 1].imag != 0

    # CPU floating comparisons can flush subnormal values. The axis decision
    # uses the existing IEEE-word sign helper so this input remains complex.
    tiny = jnp.nextafter(jnp.float64(0.), jnp.float64(1.))
    near_axis = jnp.asarray(complex(1., float(tiny)), dtype=jnp.complex128)
    assert real_sign(jnp.imag(near_axis)) != 0
    result = _UNARY_FNS["pow2"](near_axis)
    assert result.dtype == jnp.complex128
