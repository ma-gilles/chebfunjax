"""Observable source grids for oversized minSamples and raw maxLength.

Provenance
----------
MATLAB source: @chebfun/constructor.m and @chebtech{1,2}/refine.m.
Chebfun commit: 7574c77
These supplemental analytic controls retain existing source assertion bounds.
Grid tolerance1e-14 is the existing adaptive callback-grid bound. Polynomial
value bound100*binary64 eps is the existing independent constructor bound.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.tech.chebtech import Chebtech2, _refine_chebtech2_nested
from chebfunjax.utils.quadrature import chebpts


def _polynomial(x):
    return 1 + x + x*x


def _same_grid(actual, expected):
    assert actual.shape == expected.shape
    assert float(jnp.max(jnp.abs(actual-expected), initial=0.0)) < 1e-14


@pytest.mark.parametrize("extrapolate", [False, True])
def test_source_initial_clipping_and_later_nested_giveup(extrapolate):
    calls=[]
    def op(x):
        calls.append(x)
        return _polynomial(x)
    sampled, gave_up = _refine_chebtech2_nested(
        op, max_length=160, min_samples=226, extrapolate=extrapolate)
    assert not gave_up and sampled.size==160
    expected=chebpts(160,kind=2)
    if extrapolate:
        expected=expected[1:-1]
    _same_grid(calls[0],expected)
    unchanged,gave_up=_refine_chebtech2_nested(
        op,sampled,max_length=160,min_samples=226,extrapolate=extrapolate)
    assert gave_up and len(calls)==1  # next nested319 exceeds raw160.
    assert unchanged.shape == sampled.shape
    assert bool(jnp.array_equal(unchanged,sampled,equal_nan=True))


@pytest.mark.parametrize("kind", [1,2])
@pytest.mark.parametrize("cap", [20,160])
def test_public_ordinary_constructor_preserves_raw_cap(kind,cap):
    calls=[]
    def op(x):
        calls.append(x)
        return _polynomial(x)
    built=cj.chebfun(op,min_samples=226,max_length=cap,
                     chebkind=kind,sample_test=False)
    # _vector_check can make small preflight probes; identify the adaptive batch.
    adaptive=[x for x in calls if x.size>8]
    assert len(adaptive)==1
    _same_grid(adaptive[0],chebpts(cap,kind=kind))
    probes=jnp.linspace(-1,1,11)
    assert float(jnp.max(jnp.abs(built(probes)-_polynomial(probes)))) < 100*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("ordinary_cap", [None,33])
@pytest.mark.parametrize("turbo", [False,True])
def test_split_probe_and_piece_fits_keep_raw_split_cap(monkeypatch,ordinary_cap,turbo):
    observed=[]
    original=Chebtech2.from_function.__func__
    def observe(cls,f,*args,**kwargs):
        batches=[]
        def record(x):
            batches.append(x)
            return f(x)
        result=original(cls,record,*args,**kwargs)
        observed.append(batches)
        return result
    monkeypatch.setattr(Chebtech2,"from_function",classmethod(observe))
    built=cj.chebfun(_polynomial,splitting=True,split_length=160,
                     min_samples=226,max_length=ordinary_cap,
                     sample_test=False,turbo=turbo)
    assert built.ishappy and observed
    expected=chebpts(160,kind=2)[1:-1]
    for batches in observed:
        # Source constructorSplit sets extrapolate on for every fit, including
        # the first full-interval attempt and later plain/turbo piece fits.
        assert batches
        _same_grid(batches[0],expected)
