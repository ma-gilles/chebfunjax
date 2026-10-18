"""Native Chebtech coefficient normalization and linear root words,7574c77.

Nine actual native controls use documented coefficient-cell construction.
The observer records the real recursive engine input without replacing its
output; mapped and public roots are checked against independent native words.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech import chebtech as module

_ROWS=json.loads((Path(__file__).parent/'fixtures/roots_normalization_native_20261018.json').read_text())['cases']

def _array(words):
    return jax.lax.bitcast_convert_type(jnp.asarray([int(w,16) for w in words],dtype=jnp.uint64),jnp.float64)

def _words(value):
    return jax.lax.bitcast_convert_type(jnp.asarray(value).reshape(-1),jnp.uint64)

@pytest.mark.parametrize('case',_ROWS,ids=lambda q:q['alpha']['real'][0])
def test_native_normalization_and_roots(case,monkeypatch):
    coeffs=_array(case['coeffs']['real'])
    tech=module.Chebtech2.from_coeffs(coeffs)
    assert jnp.array_equal(_words(tech.vscale),_words(_array(case['vscale']['real'])))
    observed=[]
    original=module._roots_main
    def observe(c,*args,**kwargs):
        observed.append(jnp.asarray(c))
        return original(c,*args,**kwargs)
    monkeypatch.setattr(module,'_roots_main',observe)
    result=tech.roots()
    assert observed
    assert jnp.array_equal(_words(observed[0]),_words(_array(case['coeffs_mrdivide_vscale']['real'])))
    assert jnp.array_equal(_words(result),_words(_array(case['roots']['real'])))
    piece=_Piece(tech,(-1.,1.))
    expected=_words(_array(case['mapping_For_roots']['real']))
    assert jnp.array_equal(_words(piece.roots()),expected)
    assert jnp.array_equal(_words(Chebfun([piece],Domain((-1.,1.))).roots()),expected)
