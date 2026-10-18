"""Exact sparse-conversion controls, @ultraS/multmat.m Chebfun7574c77.

Twenty unmodified native calls, exact source inputs and CSC support/weights.
Complex final recurrence classes beyond conversion remain separately recorded.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.discretization import ultras

FIXTURE=Path(__file__).parents[1]/'fixtures'/'ultras_multmat_nonfinite_native_20261018.json'


def _decode(words):
    return [struct.unpack('>d',bytes.fromhex(word))[0] for word in words]


def _case(index):
    case=json.loads(FIXTURE.read_text())['cases'][index]
    real=jnp.asarray(_decode(case['input_re_hex']))
    imag=jnp.asarray(_decode(case['input_im_hex']))
    a=real if case['input_isreal'] else jax.lax.complex(real,imag)
    return case,a


def _assert_component(actual,words):
    expected=jnp.asarray(_decode(words)).reshape(actual.shape)
    assert bool(jnp.array_equal(jnp.isnan(actual),jnp.isnan(expected)))
    assert bool(jnp.array_equal(jnp.isposinf(actual),jnp.isposinf(expected)))
    assert bool(jnp.array_equal(jnp.isneginf(actual),jnp.isneginf(expected)))
    # Native exact binary64 words for every finite coefficient, signed zeros
    # included; NaN payload bits are outside this component-class contract.
    finite=jnp.isfinite(expected)
    assert bool(jnp.all(jnp.where(finite,actual.view(jnp.uint64)==expected.view(jnp.uint64),True)))


def _source_matrix(n,lam):
    # Independent scalar implementation of sparse @ultraS/spconvert.m and
    # convertmat.m. Skip implicit zeros before multiplication, no thresholds.
    S=[[float(i==j) for j in range(n)] for i in range(n)]
    for degree in range(lam):
        T=[[0.]*n for _ in range(n)]
        for i in range(n):
            T[i][i]=(1. if i==0 else .5) if degree==0 else degree/(degree+i)
            if i+2<n:
                T[i][i+2]=-.5 if degree==0 else -degree/(degree+i+2)
        S=[[sum(T[i][k]*S[k][j] for k in range(n) if T[i][k]!=0 and S[k][j]!=0)
            for j in range(n)] for i in range(n)]
    return S


@pytest.mark.parametrize('index',range(20))
def test_native_and_independent_csc_conversion_components(index):
    case,a=_case(index)
    n,lam=case['n'],case['lambda']
    matrix=ultras.convertmat(n,0,lam-1)
    native_matrix=jnp.zeros((n,n)).at[jnp.asarray(case['conversion_i'])-1,jnp.asarray(case['conversion_j'])-1].set(jnp.asarray(_decode(case['conversion_v_hex'])))
    source=jnp.asarray(_source_matrix(n,lam))
    assert bool(jnp.array_equal(source!=0,native_matrix!=0))
    assert int(jnp.count_nonzero(matrix))==case['conversion_nnz']
    # Existing dense convertmat can differ from native sparse products by
    # binary64 rounding (observed at lambda3). Its matrix values retain a
    # fixed operation-count/scale bound; converted coefficients below still
    # require exact native finite words and nonfinite component classes.
    scale=jnp.linalg.norm(native_matrix,ord=jnp.inf)
    assert float(jnp.linalg.norm(matrix-native_matrix,ord=jnp.inf))<=float(64*n*jnp.finfo(jnp.float64).eps*scale)
    assert float(jnp.linalg.norm(source-native_matrix,ord=jnp.inf))<=float(64*n*jnp.finfo(jnp.float64).eps*scale)
    actual=ultras._sparse_nonfinite_conversion(matrix,a)
    _assert_component(jnp.real(actual),case['converted_re_hex'])
    _assert_component(jnp.imag(actual),case['converted_im_hex'])
    # Independent CSC scalar interpreter uses captured source-matrix weights,
    # visits columns in ascending order, and rounds multiply before addition.
    # Real/imaginary components scale independently by the real sparse matrix.
    results=[]
    for values in [_decode(case['input_re_hex']),_decode(case['input_im_hex'])]:
        result=[0.]*n
        for row,col,weight in zip(case['conversion_i'],case['conversion_j'],_decode(case['conversion_v_hex'])):
            result[row-1]+=weight*values[col-1]
        results.append(jnp.asarray(result))
    _assert_component(results[0],case['converted_re_hex'])
    _assert_component(results[1],case['converted_im_hex'])


@pytest.mark.parametrize('index',range(9))
def test_native_real_nonfinite_matrix_and_source_tail(index,monkeypatch):
    case,a=_case(index)
    assert case['input_isreal']
    steps=[]
    def tracked_range(*args):
        for value in range(*args):
            if len(args)==1:
                steps.append(value)
            yield value
    monkeypatch.setattr(ultras,'range',tracked_range,raising=False)
    actual=ultras.multmat(case['n'],a,case['lambda'])
    decisions=case['source_derived_tail_decisions']
    # Counts are derived from captured normalization and source tail policy;
    # native multmat itself was not instrumented.
    assert len(steps)==(len(decisions) if isinstance(decisions,list) else 1)
    _assert_component(jnp.real(actual).T,case['output_re_hex'])
    _assert_component(jnp.imag(actual).T,case['output_im_hex'])


@pytest.mark.parametrize('lam',[2,3])
@pytest.mark.parametrize('complex_values',[False,True])
def test_finite_path_does_not_use_sparse_nonfinite_kernel(lam,complex_values,monkeypatch):
    def forbidden(*args):
        pytest.fail('finite multmat entered nonfinite sparse conversion kernel')
    monkeypatch.setattr(ultras,'_sparse_nonfinite_conversion',forbidden)
    coefficients=jnp.array([.5,0.,.125,0.,0.,.25])
    if complex_values:
        coefficients=(.75+.25j)*coefficients
    actual=ultras.multmat(9,coefficients,lam)
    assert bool(jnp.all(jnp.isfinite(actual)))
