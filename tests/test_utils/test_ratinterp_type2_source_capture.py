"""Source-output comparisons and public integration; expected data only.

Provenance
----------
MATLAB source : ratinterp.m/constructRatApproxCheb2 and ratbary
Chebfun commit: 7574c77
Captured helper archive: b0fd61e4b4e6d1b9e8612f92aac8f958417ad92f782d2d9bd8d5b19199df94c5
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_type2_rational_handle
from chebfunjax.utils.ratapprox import ratinterp


@pytest.mark.parametrize("name,query", [
    ("complex_scalar", .2+.1j),
    ("complex_vector", [-.5+.1j, 0, .2-.1j, .5+.25j]),
    ("complex_matrix", [[-.5+.1j, .1-.2j, .4], [-.1+.3j, .25+.05j, .75-.2j]]),
])
def test_complete_complex_source_output(name, query):
    record = json.loads((Path(__file__).parents[1]/'fixtures'/'ratinterp_type2_source_2025b'/f'{name}.json').read_text())
    def decode(words):
        return np.array([struct.unpack('>d', bytes.fromhex(word))[0] for word in words])
    expected = decode(record['value_real_num2hex']) + 1j*decode(record['value_imag_num2hex'])
    # MATLAB serializes flattened arrays in column-major order. Inputs are
    # independent literal expressions above, never decoded source query arrays.
    shape = np.shape(query)
    expected = expected.reshape(shape, order='F')
    r = _source_type2_rational_handle(jnp.array([1+.25j, .2-.1j]),
        jnp.array([3+.5j, -.3+.2j, .1-.05j]), (-1., 1.))
    np.testing.assert_allclose(r(jnp.asarray(query)), expected, rtol=0, atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(jax.jit(r)(jnp.asarray(query)), expected, rtol=0, atol=100*np.finfo(float).eps)


@pytest.mark.parametrize("disabled", [False, True])
def test_public_mu0_type2_source_node_and_nonfinite(disabled):
    with jax.disable_jit(disabled):
        r, _, _, mu, nu, _, _ = ratinterp(lambda x: 2/(2+x), 0, 1, NN=8, xi='type2')
        assert (mu, nu) == (0, 1)
        values = np.asarray(r(jnp.array([-.5, 0., .5])))
        assert np.isposinf(values.real[[0, 2]]).all()
        assert np.all(values.imag == 0)
        assert abs(values[1]+1) <= 100*np.finfo(float).eps


def test_disabled_jit_finite_shape_and_query_gradient():
    with jax.disable_jit():
        r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (-1., 1.))
        x = jnp.array([-.5, 0., .5])
        np.testing.assert_allclose(r(x), (1+x)/(3+x+x*x), rtol=0, atol=100*np.finfo(float).eps)
        actual = jax.vmap(jax.grad(r))(x)
        expected = ((3+x+x*x)-(1+x)*(1+2*x))/(3+x+x*x)**2
        np.testing.assert_allclose(actual, expected, rtol=0, atol=200*np.finfo(float).eps)
