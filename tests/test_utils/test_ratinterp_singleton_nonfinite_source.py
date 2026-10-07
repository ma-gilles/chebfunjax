"""Literal copied-source singleton outputs, expected-only MATLAB binary64 data.

Provenance
----------
MATLAB source : ratinterp.m/constructRatApproxCheb2,ratbary
Chebfun commit: 7574c77
MATLAB runtime: R2025b; raw source capture, not public fitting evidence.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_type2_rational_handle


@pytest.mark.parametrize("name,coefficient", [
    ("explicit_zero_imag", complex(2, 0)),
    ("genuine_complex", 2+.25j),
    ("pure_imaginary", 2j),
    ("tiny_imaginary", 2+1e-300j),
])
def test_singleton_both_components_against_matlab(name, coefficient):
    record = json.loads((Path(__file__).parents[1]/'fixtures'/
                         'ratinterp_type2_singleton_2025b'/f'{name}.json').read_text())
    def decode(words):
        return np.array([struct.unpack('>d', bytes.fromhex(word))[0] for word in words])
    expected_real = decode(record['value']['real_num2hex'])
    expected_imag = decode(record['value']['imag_num2hex'])
    # Independent literal inputs; only EXPECTED outputs are decoded.
    r = _source_type2_rational_handle(jnp.array([coefficient]), jnp.array([2., 1.]), (-1., 1.))
    x = jnp.array([-.5, 0., .5])
    for evaluate in (r, jax.jit(r)):
        values = np.asarray(evaluate(x))
        assert np.iscomplexobj(values) == (not record['value']['isreal'])
        np.testing.assert_array_equal(values.real, expected_real)
        np.testing.assert_array_equal(values.imag, expected_imag)
