"""Exact native atan axes, independent off-axis references and analytic AD.

Provenance
----------
MATLAB source: @treeVar/treeVar.m and @chebfun/atan.m, atand.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Opaque R2025b builtin conventions are captured independently. Off-axis
80-digit mathematical references were independently verified at500digits.
No magnitude floor inferred from native tiny-pole output is implemented.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS, to_first_order

from .test_unary_eager_storage_source import decode

DATA = Path(__file__).with_name('data')
ROWS = json.loads((DATA/'atan_native198.json').read_text())['records']
ROWS += json.loads((DATA/'atan_native72.json').read_text())['records']
OPEN = {(r['method'], r['label']):r['flags'] for r in json.loads(
    (DATA/'atan_open.json').read_text())['records']}


def check_components(got, expected):
    for value, target in [(got.real, expected.real), (got.imag, expected.imag)]:
        assert np.array_equal(np.isnan(value), np.isnan(target))
        assert np.array_equal(np.isposinf(value), np.isposinf(target))
        assert np.array_equal(np.isneginf(value), np.isneginf(target))
        finite = np.isfinite(target)
        assert np.all(np.abs(value[finite]-target[finite]) <=
                      8*np.finfo(float).eps*np.maximum(1., np.abs(target[finite])))
        assert np.array_equal(value == 0, target == 0)
        zeros = target == 0
        assert np.array_equal(np.signbit(value[zeros]), np.signbit(target[zeros]))


@pytest.mark.parametrize('row', [
    pytest.param(r, marks=pytest.mark.xfail(strict=True, reason=
        'Retained off-axis subnormal primitive gap: '+','.join(OPEN[(r['method'], r['label'])])))
    if (r['method'], r['label']) in OPEN else r for r in ROWS
], ids=lambda r:r['method']+'_'+r['label'])
def test_native_atan_axes_and_neighbors(row):
    assert row['ok']
    got = np.asarray(_UNARY_FNS[row['method']](jnp.asarray(decode(row['input']))))
    expected = decode(row['output'])
    assert got.shape == expected.shape
    assert (not np.iscomplexobj(got)) == row['output']['isreal']
    check_components(got, expected)


HIGHPRECISION = json.loads((DATA/'atan_highprecision.json').read_text())['records']
HP_OPEN = {r['label'] for r in json.loads((DATA/'atan_open.json').read_text())['records']
           if r['method'] == 'atan' and not r['label'].endswith(('ninf', 'pinf'))}


@pytest.mark.parametrize('row', [
    pytest.param(r, marks=pytest.mark.xfail(strict=True, reason=
        'JAX off-axis minimum-subnormal arithmetic remains inaccurate; independent mathematical reference'))
    if r['label'] in HP_OPEN else r for r in HIGHPRECISION
], ids=lambda r:r['label'])
def test_independent_highprecision_atan(row):
    got = np.asarray(_UNARY_FNS['atan'](jnp.asarray(decode(row['input']))))
    expected = np.asarray([[complex(*(float(v) for v in row['reference80']))]])
    check_components(got, expected)


@pytest.mark.parametrize('name', ['atan', 'atand'])
@pytest.mark.parametrize('z', [.4+0j, .4+.3j, .5j, -.5j])
def test_regular_axis_jit_and_analytic_transverse_tangent(name, z):
    value = jnp.asarray(z)
    dz = jnp.asarray(.2+.3j)
    scale = 1. if name == 'atan' else 180/jnp.pi
    expected = dz/(1+value*value)*scale
    fn = _UNARY_FNS[name]
    for op in (fn, jax.jit(fn)):
        primal, tangent = jax.jvp(op, (value,), (dz,))
        assert primal.dtype == jnp.complex128
        assert jnp.abs(tangent-expected) < 10*jnp.finfo(float).eps
    rhs = to_first_order(lambda u:u.diff()-getattr(u, name)(), 0., (-1., 1.))[0]
    for op in (rhs, jax.jit(rhs)):
        _, tangent = jax.jvp(lambda x:op(0., x), (jnp.asarray([z]),), (jnp.asarray([dz]),))
        assert jnp.linalg.norm(tangent-expected) < 10*jnp.finfo(float).eps


@pytest.mark.parametrize('name', ['atan', 'atand'])
def test_jit_value_convention_on_cut_without_derivative_claim(name):
    x = jnp.asarray([2j, -2j, 1j, -1j])
    got = np.asarray(jax.jit(_UNARY_FNS[name])(x))
    scale = 1. if name == 'atan' else 180/np.pi
    expected = np.asarray([complex(-np.pi/2, .5*np.log(3)),
                           complex(np.pi/2, -.5*np.log(3)),
                           complex(0, np.inf), complex(0, -np.inf)])
    expected.real *= scale
    expected.imag *= scale
    check_components(got, expected)
