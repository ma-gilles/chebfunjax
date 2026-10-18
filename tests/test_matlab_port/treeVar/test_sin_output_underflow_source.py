"""Pinned native near-underflow SIN controls, with prior-word preservation."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.elementary import _sin_cos_numeric

DATA = json.loads((Path(__file__).with_name('data') / 'sin_output_underflow_native_20261018.json').read_text())
def _words(value):
    return [f'{int(w):016x}' for w in np.asarray(value,dtype=np.float64).ravel(order='F').view(np.uint64)]
def _input(row):
    def component(key):
        return jnp.asarray(np.array([int(w,16) for w in row['input'][key]],dtype=np.uint64).view(np.float64).reshape(tuple(row['input']['shape']),order='F'))
    x,y=component('real'),component('imag')
    assert _words(x)==row['input']['real'] and _words(y)==row['input']['imag']
    return jax.lax.complex(x,y)
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['name'])
def test_sin_near_underflow_native_words(row,mode):
    fn=_sin_cos_numeric if mode=='eager' else jax.jit(_sin_cos_numeric)
    actual=_words(jnp.imag(fn(_input(row))))
    assert sum(sum(r['mask_F']) for r in DATA['rows'])==904
    assert [w for w,k in zip(actual,row['mask_F']) if k]==[w for w,k in zip(row['native']['imag'],row['mask_F']) if k]
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['name'])
def test_sin_near_underflow_preserves_prior_words(row,mode):
    fn=_sin_cos_numeric if mode=='eager' else jax.jit(_sin_cos_numeric)
    actual=fn(_input(row))
    prior=row['prior_modes'][mode]
    assert _words(jnp.real(actual))==prior['actual_real_words']
    imag=_words(jnp.imag(actual))
    assert [w for w,k in zip(imag,row['mask_F']) if not k]==[w for w,k in zip(prior['actual_imaginary_words'],row['mask_F']) if not k]
