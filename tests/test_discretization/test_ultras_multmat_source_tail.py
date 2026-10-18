"""Native/source controls for @ultraS/multmat.m tail stopping, Chebfun7574c77.

Original ultras and disk fixture bounds are retained in their original files.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.discretization import ultras

FIXTURE = Path(__file__).parents[1]/'fixtures'/'ultras_multmat_tail_native_20261018.json'


def _decode(words):
    return [struct.unpack('>d', bytes.fromhex(word))[0] for word in words]


def _case(index):
    case = json.loads(FIXTURE.read_text())['cases'][index]
    a = jnp.asarray(_decode(case['input_re_hex']))+1j*jnp.asarray(_decode(case['input_im_hex']))
    if not bool(jnp.any(jnp.imag(a))):
        a = jnp.real(a)
    return case, a


def _record_recurrence_steps(monkeypatch):
    steps = []
    # convertmat uses range(start,stop); multmat's recurrence uses range(count).
    # Count actual evaluated steps so sub-eps omitted terms remain observable
    # even when their contribution rounds below a matrix-value comparison bound.
    def tracked_range(*args):
        for value in range(*args):
            if len(args) == 1:
                steps.append(value)
            yield value
    monkeypatch.setattr(ultras, 'range', tracked_range, raising=False)
    return steps


@pytest.mark.parametrize('index', range(10))
def test_unmodified_native_finite_matrix_and_source_recurrence_steps(index, monkeypatch):
    case, a = _case(index)
    assert case['error_id'] == ''
    steps = _record_recurrence_steps(monkeypatch)
    actual = ultras.multmat(case['n'],a,case['lambda'])
    expected = jnp.asarray(_decode(case['output_re_hex']))+1j*jnp.asarray(_decode(case['output_im_hex']))
    expected = expected.reshape(case['n'],case['n']).T  # Native column-major.
    # New matrix control: fixed binary64 operation-count/scale allowance;
    # there is no original MATLAB test tolerance for this capture.
    scale = jnp.linalg.norm(actual,ord=jnp.inf)+jnp.linalg.norm(expected,ord=jnp.inf)
    bound = 64*case['n']*jnp.finfo(jnp.float64).eps*scale
    assert float(jnp.linalg.norm(actual-expected,ord=jnp.inf)) <= float(bound)
    decisions = case['tail_decisions']
    expected_steps = len(decisions) if isinstance(decisions,list) else 1
    assert len(steps) == expected_steps


@pytest.mark.parametrize('index', [10,11])
@pytest.mark.xfail(strict=True, reason='Native sparse conversion localizes NaN/Inf; dense conversion gap remains')
def test_native_nonfinite_conversion_and_source_tail(index, monkeypatch):
    case, a = _case(index)
    steps = _record_recurrence_steps(monkeypatch)
    actual = ultras.multmat(case['n'],a,case['lambda'])
    # Counts are derived from the captured native converted coefficients and
    # literal source tail policy, not instrumented native loop observations.
    assert len(steps) == len(case['tail_decisions']) == 2
    expected = jnp.asarray(_decode(case['output_re_hex'])).reshape(case['n'],case['n']).T
    assert bool(jnp.array_equal(jnp.isnan(actual),jnp.isnan(expected)))
    assert bool(jnp.array_equal(jnp.isposinf(actual),jnp.isposinf(expected)))
    assert bool(jnp.array_equal(jnp.isneginf(actual),jnp.isneginf(expected)))


def _gegenbauer_values(n, lam, x):
    values = [1.,2*lam*x]
    for k in range(1,n-1):
        values.append(2*(k+lam)/(k+1)*x*values[-1]-(k+2*lam-1)/(k+1)*values[-2])
    return values[:n]


def _chebyshev_value(coefficients,x):
    values = [1.,x]
    for _ in range(1,len(coefficients)-1):
        values.append(2*x*values[-1]-values[-2])
    return sum(c*v for c,v in zip(coefficients,values))


@pytest.mark.parametrize('lam', [2,3])
@pytest.mark.parametrize('complex_values', [False,True])
def test_independent_polynomial_product_without_truncation(lam, complex_values):
    n=12
    coefficients=[.5,0.,.125,0.,0.,.25]
    if complex_values:
        coefficients=[(.75+.25j)*value for value in coefficients]
    matrix=ultras.multmat(n,jnp.asarray(coefficients),lam)
    # Multipliers degree5 and basis degree<=4 give degree<=9<n: no truncated
    # tail is involved. Independent scalar Chebyshev/Gegenbauer recurrences
    # verify physical multiplication rather than reusing matrix construction.
    for x in [-.8,-.2,.4,.9]:
        basis=jnp.asarray(_gegenbauer_values(n,lam,x))
        f=_chebyshev_value(coefficients,x)
        for column in range(5):
            actual=jnp.sum(basis*matrix[:,column])
            expected=f*basis[column]
            scale=jnp.sum(jnp.abs(basis*matrix[:,column]))+jnp.abs(expected)
            bound=64*n*jnp.finfo(jnp.float64).eps*scale
            assert float(jnp.abs(actual-expected)) <= float(bound)


@pytest.mark.parametrize('index', [10,11])
def test_native_converted_nonfinite_tail_condition(index, monkeypatch):
    case, a = _case(index)
    converted = jnp.asarray(_decode(case['converted_re_hex']))+1j*jnp.asarray(_decode(case['converted_im_hex']))
    # Explicit input adapter isolates the source tail conditional from the
    # inherited sparse-vs-dense conversion gap documented above. These are
    # captured native converted words; public input assertions are separate.
    class CapturedConversion:
        def __matmul__(self, operand):
            return converted
    monkeypatch.setattr(ultras, 'convertmat', lambda *args:CapturedConversion())
    steps = _record_recurrence_steps(monkeypatch)
    actual = ultras.multmat(case['n'],a,case['lambda'])
    assert len(steps) == len(case['tail_decisions']) == 2
    assert bool(jnp.any(~jnp.isfinite(actual)))
