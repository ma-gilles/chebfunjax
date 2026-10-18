"""Piecewise local extrema must include derivative-jump contributions.

Provenance
----------
MATLAB source: @chebfun/min.m localMin; @chebfun/max.m localMax;
@chebfun/diff.m diffContinuousDim.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Expected extrema follow the independent piecewise quadratic formulas.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("center,halfwidth", [(0.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize("amplitude", [1.0, 1e-6])
def test_local_piecewise_quadratic_extrema(sign, center, halfwidth, amplitude):
    x = cj.chebfun('x', domain=(center-halfwidth, center+halfwidth))
    t = (x-center)/halfwidth
    f = amplitude*sign*(t.abs()-t*t)
    for operation in ['min', 'max']:
        cusp_selected = (operation == 'min') == (sign > 0)
        nodes = np.array([-1.0, 0.0, 1.0] if cusp_selected else [-0.5, 0.5])
        positions = center+halfwidth*nodes
        values = amplitude*sign*(np.abs(nodes)-nodes*nodes)
        actual_positions, actual_values = getattr(f, operation)('local')
        assert actual_positions.shape == positions.shape
        np.testing.assert_allclose(actual_positions, positions, rtol=0,
                                   atol=128*np.finfo(float).eps*max(1, abs(center), halfwidth))
        np.testing.assert_allclose(actual_values, values, rtol=0,
                                   atol=128*np.finfo(float).eps*amplitude)
        np.testing.assert_allclose(f(jnp.asarray(positions)), values, rtol=0,
                                   atol=128*np.finfo(float).eps*amplitude)


_NATIVE = json.loads((Path(__file__).parent/'fixtures'/
                      'local_extrema_delta_native_20261018.json').read_text())['rows']


def _decode_native(record):
    assert record['class'] == 'double' and record['isreal']
    values = np.asarray([int(word, 16) for word in record['real_hex']],
                        dtype=np.uint64).view(np.float64)
    return values.reshape(record['shape'], order='F')


@pytest.mark.parametrize('record', _NATIVE, ids=lambda r: r['name'])
def test_native_derivative_jump_local_minima(record):
    from chebfunjax.domain import Domain

    domain = _decode_native(record['domain']).ravel()
    pieces = [cj.Chebfun.from_coeffs(
        jnp.asarray(_decode_native(piece['coeffs']).ravel()), domain=(a, b)).funs[0]
        for piece, a, b in zip(record['pieces'], domain[:-1], domain[1:])]
    f = cj.Chebfun(funs=pieces, domain=Domain(tuple(domain))).set_point_values(
        jnp.asarray(_decode_native(record['point_values']).ravel()))
    expected_positions = _decode_native(record['local_min_positions']).ravel()
    expected_values = _decode_native(record['local_min_values']).ravel()
    positions, values = f.min('local')
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(values, expected_values)
    queries = jnp.asarray(_decode_native(record['all_extrema_positions']).ravel())
    curvature = np.asarray(f.diff(2)(queries))
    expected_curvature = _decode_native(record['d2_at_extrema']).ravel()
    np.testing.assert_array_equal(np.isposinf(curvature), np.isposinf(expected_curvature))
    np.testing.assert_array_equal(np.isneginf(curvature), np.isneginf(expected_curvature))
