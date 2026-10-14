"""Native numeric Ai boundary/storage controls for TreeVar.

Provenance
----------
MATLAB source: @treeVar/treeVar.m dispatch to numeric AIRY, Chebfun commit
7574c77680d7e82b79626300bf255498271a72df. AIRY is an opaque MATLAB builtin;
these are independently captured R2025b boundary conventions, not an
implementation-source claim. Finite components retain the established
8eps*max(1,abs(expected)) boundary guard, with exact zero/nonfinite classes.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

DATA=Path(__file__).with_name('data')
CASES=json.loads((DATA/'airy_native36.json').read_text())['records']
CASES+=json.loads((DATA/'airy_nonfinite_native.json').read_text())['records']


def decode(item):
    def words(key):
        return np.asarray([struct.unpack('>d',bytes.fromhex(x))[0]
                           for x in item[key] if x])
    re,im=words('real_hex'),words('imag_hex')
    value=re if item['isreal'] else np.asarray([complex(r,i) for r,i in zip(re,im)],dtype=np.complex128)
    return value.reshape(item['shape'],order='F')


@pytest.mark.parametrize('row',[
    pytest.param(row, marks=pytest.mark.xfail(strict=True, reason=
        "Native Ai(-2+/-2i) differs beyond8eps;80-digit reference shows native rounding dominates, finite JAX kernel unchanged"))
    if row['label'] in ('complex_2_2','complex_2_5') else row
    for row in CASES],ids=lambda r:r['label'])
def test_native_airy_boundary(row):
    assert row['ok']
    x=decode(row['input'])
    expected=decode(row['output'])
    result=np.asarray(_UNARY_FNS['airy'](jnp.asarray(x)))
    assert result.shape==expected.shape
    assert (not np.iscomplexobj(result))==row['output']['isreal']
    for value,target in [(result.real,expected.real),(result.imag,expected.imag)]:
        assert np.array_equal(np.isnan(value),np.isnan(target))
        assert np.array_equal(np.isposinf(value),np.isposinf(target))
        assert np.array_equal(np.isneginf(value),np.isneginf(target))
        finite=np.isfinite(target)
        assert np.all(np.abs(value[finite]-target[finite])<=8*np.finfo(float).eps*np.maximum(1.,np.abs(target[finite])))
        zero=target==0
        assert np.all(value[zero]==0)
        assert np.array_equal(np.signbit(value[zero]),np.signbit(target[zero]))


def test_finite_airy_kernel_values_preserved():
    from chebfunjax.utils.airy_general import airy_all
    for x in (jnp.asarray([-2., -.4, 0., .4, 2.]),
              jnp.asarray([-.5+.25j, .4-.3j, 2.+.3j])):
        got=_UNARY_FNS['airy'](x)
        expected=airy_all(x)[0]
        assert jnp.array_equal(got,expected)


def test_airy_rhs_nonfinite_conventions():
    from chebfunjax.operators.treevar import to_first_order
    converted,*_=to_first_order(lambda u:u.diff()-u.airy(),0.,(-1.,1.))
    for x,expected in [(jnp.inf,0.),(-jnp.inf,jnp.nan),
                       (complex(1,jnp.inf),jnp.inf)]:
        value=np.asarray(converted(0.,jnp.asarray([x]))).reshape(-1)[0]
        if np.isnan(expected):
            assert np.isnan(value)
        else:
            assert value==expected


@pytest.mark.parametrize('row',json.loads((DATA/'airy_highprecision_controls.json').read_text())['records'])
def test_finite_airy_independent_highprecision(row):
    # Separate mathematical control, not a replacement for the strict native
    # comparisons above. Decimal references were evaluated at80digits.
    x=jnp.asarray([[complex(*row['input'])]])
    value=np.asarray(_UNARY_FNS['airy'](x))
    assert np.iscomplexobj(value) and value.shape==(1,1)
    for part,reference in zip((value.real,value.imag),row['reference80']):
        target=float(reference)
        assert np.all(np.abs(part-target)<=8*np.finfo(float).eps*max(1.,abs(target)))
