"""Supplemental native matrix FUNQUI regression; source smoothfun.m, 7574c77."""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.interpolation import funqui

CASES = json.loads((Path(__file__).parent / 'funqui_native_matrix.json').read_text())['cases']


def _array(value):
    if isinstance(value, dict):
        return jnp.asarray(value['real']) + 1j * jnp.asarray(value['imag'])
    return jnp.asarray(value)


@pytest.mark.parametrize('record', CASES, ids=lambda r: r['name'])
def test_native_matrix_interpolant(record):
    values = _array(record['input_values'])
    x = _array(record['query_points'])
    a, b = record['domain']
    expected = _array(record['query_values'])
    # Supplemental machine-precision control using the original constructor
    # test's 10-epsilon scale convention; no CV tie or degree is prescribed.
    bound = 10 * jnp.finfo(jnp.float64).eps * jnp.max(jnp.abs(values))
    got = funqui(values)(2 * (x - a) / (b - a) - 1)
    assert jnp.max(jnp.abs(got - expected)) < bound


@pytest.mark.parametrize('record', CASES[:3], ids=lambda r: r['name'])
def test_native_matrix_public_constructor(record):
    values = _array(record['input_values'])
    pref = ChebfunPref({'enableFunqui': True, 'tech': record['tech']})
    f = chebfun(values, domain=tuple(record['domain']), pref=pref)
    assert f.funs[0].tech.coeffs.shape[0] == record['length'] == 4
    bound = 10 * jnp.finfo(jnp.float64).eps * jnp.max(jnp.abs(values))
    assert jnp.max(jnp.abs(f(_array(record['query_points'])) - _array(record['query_values']))) < bound
