"""Literal separately rounded source order for retained binary32/binary64 inputs.

Exact represented coefficients retain the regression exposed by count50,
negative scale. The expected column uses Fraction with explicit dtype rounding;
no filesystem captures, compiler flags or alternate numerical solver are used.
"""
import struct
from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module

_RETAINED_INPUT_BITS = {'float32': [['bf400000', '00000000'], ['bf4bc14e', '00000000'], ['bf57829c', '00000000'], ['bf6343ec', '00000000'], ['bf6f053a', '00000000'], ['bf7ac688', '00000000'], ['bf8343eb', '00000000'], ['bf892492', '00000000'], ['bf8f053a', '00000000'], ['bf94e5e0', '00000000'], ['bf9ac688', '00000000'], ['bfa0a72f', '00000000'], ['bfa687d6', '00000000'], ['bfac687d', '00000000'], ['bfb24924', '00000000'], ['bfb829cc', '00000000'], ['bfbe0a73', '00000000'], ['bfc3eb1a', '00000000'], ['bfc9cbc1', '00000000'], ['bfcfac68', '00000000'], ['bfd58d0f', '00000000'], ['bfdb6db8', '00000000'], ['bfe14e5e', '00000000'], ['bfe72f06', '00000000'], ['bfed0fac', '00000000'], ['bff2f054', '00000000'], ['bff8d0fa', '00000000'], ['bffeb1a2', '00000000'], ['c0024924', '00000000'], ['c0053978', '00000000'], ['c00829cc', '00000000'], ['c00b1a20', '00000000'], ['c00e0a73', '00000000'], ['c010fac6', '00000000'], ['c013eb1a', '00000000'], ['c016db6e', '00000000'], ['c019cbc1', '00000000'], ['c01cbc15', '00000000'], ['c01fac69', '00000000'], ['c0229cbc', '00000000'], ['c0258d10', '00000000'], ['c0287d63', '00000000'], ['c02b6db7', '00000000'], ['c02e5e0a', '00000000'], ['c0314e5e', '00000000'], ['c0343eb2', '00000000'], ['c0372f06', '00000000'], ['c03a1f59', '00000000'], ['c03d0fac', '00000000'], ['c0400000', '00000000']], 'float64': [['bfe8000000000000', '0000000000000000'], ['bfe97829cbc14e5e', '0000000000000000'], ['bfeaf05397829cbc', '0000000000000000'], ['bfec687d6343eb1a', '0000000000000000'], ['bfede0a72f053978', '0000000000000000'], ['bfef58d0fac687d6', '0000000000000000'], ['bff0687d6343eb1a', '0000000000000000'], ['bff1249249249249', '0000000000000000'], ['bff1e0a72f053978', '0000000000000000'], ['bff29cbc14e5e0a7', '0000000000000000'], ['bff358d0fac687d6', '0000000000000000'], ['bff414e5e0a72f05', '0000000000000000'], ['bff4d0fac687d634', '0000000000000000'], ['bff58d0fac687d63', '0000000000000000'], ['bff6492492492492', '0000000000000000'], ['bff705397829cbc1', '0000000000000000'], ['bff7c14e5e0a72f0', '0000000000000000'], ['bff87d6343eb1a20', '0000000000000000'], ['bff9397829cbc14e', '0000000000000000'], ['bff9f58d0fac687c', '0000000000000000'], ['bffab1a1f58d0fac', '0000000000000000'], ['bffb6db6db6db6db', '0000000000000000'], ['bffc29cbc14e5e0a', '0000000000000000'], ['bffce5e0a72f053a', '0000000000000000'], ['bffda1f58d0fac68', '0000000000000000'], ['bffe5e0a72f05396', '0000000000000000'], ['bfff1a1f58d0fac6', '0000000000000000'], ['bfffd6343eb1a1f5', '0000000000000000'], ['c000492492492492', '0000000000000000'], ['c000a72f0539782a', '0000000000000000'], ['c00105397829cbc1', '0000000000000000'], ['c0016343eb1a1f58', '0000000000000000'], ['c001c14e5e0a72f0', '0000000000000000'], ['c0021f58d0fac688', '0000000000000000'], ['c0027d6343eb1a1f', '0000000000000000'], ['c002db6db6db6db7', '0000000000000000'], ['c003397829cbc14e', '0000000000000000'], ['c00397829cbc14e5', '0000000000000000'], ['c003f58d0fac687d', '0000000000000000'], ['c0045397829cbc14', '0000000000000000'], ['c004b1a1f58d0fac', '0000000000000000'], ['c0050fac687d6344', '0000000000000000'], ['c0056db6db6db6db', '0000000000000000'], ['c005cbc14e5e0a72', '0000000000000000'], ['c00629cbc14e5e0a', '0000000000000000'], ['c00687d6343eb1a2', '0000000000000000'], ['c006e5e0a72f0539', '0000000000000000'], ['c00743eb1a1f58d1', '0000000000000000'], ['c007a1f58d0fac68', '0000000000000000'], ['c008000000000000', '0000000000000000']]}


def component_bits(a):
    # Host serialization only; numerical operations remain in JAX.
    values = jax.device_get(jnp.ravel(a)).tolist()
    return [[struct.pack('>d', float(x.real)).hex(),
             struct.pack('>d', float(x.imag)).hex()] for x in values]


def round_rational(value, dtype):
    if dtype == 'float64':
        return float(value)
    # The float64 approximation only locates neighbors. The final decision is
    # direct rational distance, with ties-to-even, so no double rounding is used.
    approximate_bits = struct.unpack('>I', struct.pack('>f', float(value)))[0]
    candidates = []
    for word in range(approximate_bits-2, approximate_bits+3):
        candidate = struct.unpack('>f', struct.pack('>I', word))[0]
        rational = Fraction.from_float(candidate)
        candidates.append((abs(rational-value), word & 1, candidate, rational))
    assert min(x[3] for x in candidates) <= value <= max(x[3] for x in candidates)
    return min(candidates, key=lambda x: (x[0], x[1]))[2]


def source_sequence_bits(coefficient_bits, dtype):
    fmt = '>f' if dtype == 'float32' else '>d'
    c = [Fraction.from_float(struct.unpack(fmt, bytes.fromhex(x[0]))[0])
         for x in coefficient_bits]
    expected = []
    for index, coefficient in enumerate(c[:-1]):
        product = round_rational(-Fraction(1, 2)*coefficient, dtype)
        quotient = round_rational(Fraction.from_float(product)/c[-1], dtype)
        amended = (round_rational(Fraction.from_float(quotient)+Fraction(1, 2), dtype)
                   if index == len(c)-3 else quotient)
        expected.append([struct.pack('>d', amended).hex(), '0000000000000000'])
    return expected[::-1]


def source_matrix_bits(coefficient_bits, dtype):
    column = source_sequence_bits(coefficient_bits, dtype)
    n = len(column)
    zero = ['0000000000000000', '0000000000000000']
    half = ['3fe0000000000000', '0000000000000000']
    one = ['3ff0000000000000', '0000000000000000']
    matrix = [list(zero) for _ in range(n*n)]
    for row in range(n):
        if row:
            matrix[row*n+row-1] = list(half)
        if row+1 < n:
            matrix[row*n+row+1] = list(half)
    matrix[(n-2)*n+n-1] = list(one)
    for row, value in enumerate(column):
        matrix[row*n] = value
    return matrix


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_colleague_literal_source_operation_sequence(dtype):
    fixture = _RETAINED_INPUT_BITS[dtype]
    fmt = '>f' if dtype == 'float32' else '>d'
    c = jnp.asarray([struct.unpack(fmt, bytes.fromhex(x[0]))[0]
                     for x in fixture], dtype=getattr(jnp, dtype))
    actual = component_bits(module._roots_colleague_matrix_jax(c))
    assert actual == source_matrix_bits(fixture, dtype)
