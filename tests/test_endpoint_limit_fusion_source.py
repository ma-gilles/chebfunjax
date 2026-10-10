"""Native lval/rval operations; eager compatibility and compilation controls."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._construction import (
    _polynomial_endpoint_limit,
    endpoint_limit,
)
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

TECHS = [Chebtech1, Chebtech2]
DTYPES = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]


def eager(coeffs, right):
    # Exact pre-extraction branch, independently retained as compatibility oracle.
    if not right:
        coeffs = coeffs.at[1::2].set(-coeffs[1::2])
    return jnp.sum(coeffs, axis=0)


def check_bits(actual, expected, tag):
    a, b = jax.device_get(actual), jax.device_get(expected)
    row = {'tag': tag, 'shape': list(a.shape), 'dtype': str(a.dtype),
           'actual_hex': a.tobytes().hex(), 'eager_hex': b.tobytes().hex()}
    report_directory = os.environ.get('CHEBFUN_RUNTIME_REPORT')
    if report_directory:
        output = Path(report_directory).parent / 'comparisons.jsonl'
        with output.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), row


def piece(tech, coeffs):
    return _Piece(tech=tech(coeffs=coeffs), interval=(2., 7.))


@pytest.mark.parametrize('tech', TECHS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(1,), (6, 3), (7,)])
def test_finite_source_operations(tech, dtype, shape):
    count = shape[0]
    columns = shape[1] if len(shape) == 2 else 1
    rows = [[(i + 1) * (j + 1) / 8 for j in range(columns)]
            for i in range(count)]
    coeffs = jnp.asarray(rows, dtype=dtype).reshape(shape)
    complex_input = jnp.issubdtype(dtype, jnp.complexfloating)
    if complex_input:
        coeffs = coeffs * (1 + 2j)
        assert bool(jnp.any(jnp.imag(coeffs) != 0))
    f = piece(tech, coeffs)
    for right in (False, True):
        actual = endpoint_limit(f, right)
        check_bits(actual, eager(coeffs, right),
                   f'{tech.__name__}/{dtype}/{shape}/{right}')
        oracle = [sum((1 if right or i % 2 == 0 else -1) * rows[i][j]
                      for i in range(count)) for j in range(columns)]
        expected = jnp.asarray(oracle, dtype=dtype)
        if complex_input:
            expected = expected * (1 + 2j)
        expected = expected.reshape(actual.shape)
        assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize('tech', TECHS)
@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
@pytest.mark.parametrize('shape', [(0,), (0, 3)])
def test_empty_sum_shape_dtype(tech, dtype, shape):
    coeffs = jnp.empty(shape, dtype=dtype)
    for right in (False, True):
        actual = endpoint_limit(piece(tech, coeffs), right)
        check_bits(actual, eager(coeffs, right), f'empty/{tech.__name__}/{dtype}/{shape}/{right}')
        assert actual.shape == shape[1:]
        assert actual.dtype == coeffs.dtype
        assert jnp.array_equal(actual, jnp.zeros(shape[1:], dtype=dtype))


@pytest.mark.parametrize('tech', TECHS)
@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_nonfinite_source_classification(tech, dtype):
    coeffs = jnp.asarray([[jnp.nan, jnp.inf, -jnp.inf, jnp.inf],
                          [1., 2., 3., -jnp.inf],
                          [0., 0., 0., 0.]], dtype=dtype)
    for right in (False, True):
        actual = endpoint_limit(piece(tech, coeffs), right)
        expected = eager(coeffs, right)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        # NaN payloads are diagnostic rather than native validity predicates.
        for a, b in [(jnp.real(actual), jnp.real(expected)),
                     (jnp.imag(actual), jnp.imag(expected))]:
            assert jnp.array_equal(jnp.isnan(a), jnp.isnan(b))
            assert jnp.array_equal(jnp.isposinf(a), jnp.isposinf(b))
            assert jnp.array_equal(jnp.isneginf(a), jnp.isneginf(b))
            assert jnp.array_equal(jnp.where(jnp.isfinite(a), a, 0.),
                                   jnp.where(jnp.isfinite(b), b, 0.))


@pytest.mark.parametrize('tech', TECHS)
def test_signed_zero_eager_bits(tech):
    coeffs = jnp.asarray([[0., -0.], [-0., 0.], [0., -0.], [-0., 0.]])
    for right in (False, True):
        check_bits(endpoint_limit(piece(tech, coeffs), right), eager(coeffs, right),
                   f'zero/{tech.__name__}/{right}')


@pytest.mark.parametrize('dtype', DTYPES)
def test_cancellation_exact_eager_operation_sequence(dtype):
    numbers = [float.fromhex(s) for s in ['0x1.000002p+12', '0x1.3p-8',
               '-0x1.000000p+12', '0x1.555556p-3', '-0x1.4p-9']]
    coeffs = jnp.asarray((numbers * 7)[:33], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        coeffs = coeffs + 1j * jnp.roll(coeffs, 3)
    for right in (False, True):
        check_bits(_polynomial_endpoint_limit(coeffs, right=right),
                   eager(coeffs, right), f'cancellation/{dtype}/{right}')


@pytest.mark.parametrize('tech', TECHS)
def test_public_dispatch_jit_and_derivatives(tech):
    coeffs = jnp.asarray([1., 2., 3., 4., 5.])
    direction = jnp.asarray([.5, -.25, .125, .25, -.5])
    for right in (False, True):
        def operation(c):
            return endpoint_limit(piece(tech, c), right)
        compiled = jax.jit(operation)
        check_bits(compiled(coeffs), eager(coeffs, right), f'jit/{tech.__name__}/{right}')
        signs = jnp.asarray([1. if right or i % 2 == 0 else -1. for i in range(5)])
        assert jnp.array_equal(jax.grad(operation)(coeffs), signs)
        c = coeffs.astype(jnp.complex128) * (1 + 2j)
        tangent = direction.astype(jnp.complex128) * (2 - 1j)
        _, derivative = jax.jvp(operation, (c,), (tangent,))
        assert jnp.array_equal(derivative, eager(tangent, right))
