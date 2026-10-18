"""Source-scaled finite hyperbolic SIN factors and pinned native components."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.elementary import _sin_cos_numeric

DATA=json.loads((Path(__file__).with_name('data')/'sin_large_hyperbolic_native_20261018.json').read_text())
def _words(v):
    return [f'{int(w):016x}' for w in np.asarray(v,dtype=np.float64).ravel(order='F').view(np.uint64)]
def _input(row):
    def component(key):
        return jnp.asarray(np.array([int(w,16) for w in row['input'][key]],dtype=np.uint64).view(np.float64).reshape(tuple(row['input']['shape']),order='F'))
    x,y=component('real'),component('imag')
    assert _words(x)==row['input']['real'] and _words(y)==row['input']['imag']
    return jax.lax.complex(x,y)
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['name'])
def test_sin_large_finite_hyperbolic_native_components(row,mode):
    fn=_sin_cos_numeric if mode=='eager' else jax.jit(_sin_cos_numeric)
    actual=fn(_input(row))
    assert jnp.iscomplexobj(actual) and actual.shape==tuple(row['native']['shape'])
    assert sum(sum(r['mask_F']) for r in DATA['rows'])==864
    for component,value in [('real',jnp.real(actual)),('imag',jnp.imag(actual))]:
        assert [w for w,k in zip(_words(value),row['mask_F']) if k]==[w for w,k in zip(row['native'][component],row['mask_F']) if k]
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['name'])
def test_sin_large_finite_hyperbolic_outside_words_preserved(row,mode):
    fn=_sin_cos_numeric if mode=='eager' else jax.jit(_sin_cos_numeric)
    actual=fn(_input(row))
    for component,value in [('real',jnp.real(actual)),('imag',jnp.imag(actual))]:
        assert [w for w,k in zip(_words(value),row['mask_F']) if not k]==[w for w,k in zip(row['prior_modes'][mode][component],row['mask_F']) if not k]
def test_native_large_hyperbolic_factor_and_zero_axis_controls():
    zeros=0
    for row in DATA['rows']:
        for x,y,h,s,real,keep in zip(row['input']['real'],row['input']['imag'],row['native_cosh'],row['native_sinh'],row['native']['real'],row['mask_F']):
            if not keep:
                continue
            assert 0x0010000000000000<=int(h,16)<0x7ff0000000000000
            assert int(s,16)==int(h,16)^(int(y,16)&(1<<63))
            if int(x,16)&0x7fffffffffffffff==0:
                assert real=='0000000000000000'
                zeros+=1
    assert zeros==48
