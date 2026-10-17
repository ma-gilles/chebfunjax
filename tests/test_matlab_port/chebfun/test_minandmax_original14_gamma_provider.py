"""Numeric string gamma uses the retained native real-pole convention.

Provenance
----------
MATLAB source : built-in gamma, actual R2025b primitive capture
Chebfun commit: 7574c77
"""

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import _string_op


def _from_hex(value):
    return struct.unpack('>d', bytes.fromhex(value))[0]


def test_native_gamma_primitive_fixture():
    fixture = Path(__file__).parents[2] / 'test_utils/gamma_matlab_primitive_fixture.json'
    data = json.loads(fixture.read_text())
    inputs = jnp.asarray([_from_hex(item['input_hex']) for item in data['inputs']])
    actual = np.asarray(_string_op('gamma(x)')(inputs))
    expected = np.array([_from_hex(item['output_hex']) for item in data['expected']])
    for value, reference in zip(actual, expected):
        if np.isinf(reference):
            assert struct.pack('>d', value).hex() == struct.pack('>d', reference).hex()
        else:
            assert abs(value-reference) <= 10*np.spacing(reference)


def test_reciprocal_pole_jumps_do_not_create_quotient_poles():
    # @chebfun/getRootsForBreaks.m uses nojump/nozerofun: a pole's sign
    # change is not a denominator zero. Reciprocal simple poles become zeros.
    from chebfunjax.chebfun1d.chebfun import chebfun

    f = chebfun(lambda x: 1/x, domain=[-1, 0, 1], exps=[0, -1, 0])
    q = 1/f
    (_, low), (_, high) = q.minandmax()
    assert abs(float(low) + 1) < 2e-14
    assert abs(float(high) - 1) < 2e-14
    points = jnp.linspace(-1, 1, 17)
    assert float(jnp.max(jnp.abs(q(points)-points))) < 2e-14
