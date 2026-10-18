"""Whole-subnormal unary source staging and independent analytic derivatives."""
import cmath
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

DATA=json.loads((Path(__file__).with_name('data')/'subnormal_staged_native_20261018.json').read_text())
METHODS=['atan','exp','expm1','sinh']
def _words(v):
    return [f'{int(w):016x}' for w in np.asarray(v,dtype=np.float64).ravel(order='F').view(np.uint64)]
def _input(row):
    def component(part):
        return jnp.asarray(np.array([int(w,16) for w in row['input'][part]],dtype=np.uint64).view(np.float64).reshape(tuple(row['input']['shape']),order='F'))
    x,y=component('real'),component('imag')
    assert _words(x)==row['input']['real'] and _words(y)==row['input']['imag']
    return jax.lax.complex(x,y)
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['method'])
def test_subnormal_staged_native_imaginary_words(row,mode):
    fn=_UNARY_FNS[row['method']]
    actual=(fn if mode=='eager' else jax.jit(fn))(_input(row))
    assert sum(sum(r['mask_F']) for r in DATA['rows'])==1152
    assert jnp.iscomplexobj(actual) and actual.shape==tuple(row['native']['shape'])
    assert [w for w,k in zip(_words(jnp.imag(actual)),row['mask_F']) if k]==[w for w,k in zip(row['native']['imag'],row['mask_F']) if k]
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['method'])
def test_subnormal_staged_real_and_outside_words_preserved(row,mode):
    fn=_UNARY_FNS[row['method']]
    actual=(fn if mode=='eager' else jax.jit(fn))(_input(row))
    assert _words(jnp.real(actual))==row['prior_modes'][mode]['real']
    assert [w for w,k in zip(_words(jnp.imag(actual)),row['mask_F']) if not k]==[w for w,k in zip(row['prior_modes'][mode]['imag'],row['mask_F']) if not k]
def _derivatives(method,z):
    if method in ['exp','expm1']:
        return cmath.exp(z),cmath.exp(z)
    if method=='sinh':
        return cmath.cosh(z),cmath.sinh(z)
    return 1/(1+z*z),-2*z/(1+z*z)**2
@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('point',[(.4,0.),(-.4,-0.),(0.,0.),(1.,0.),(-1.,.3),(.4,.3)])
@pytest.mark.parametrize('reverse',[False,True])
def test_subnormal_staged_ordinary_real_coordinate_first_second(method,point,reverse):
    def fn(p):
        value=_UNARY_FNS[method](jax.lax.complex(p[0],p[1]))
        return jnp.stack((jnp.real(value),jnp.imag(value)))
    d1,d2=_derivatives(method,complex(*point))
    jac=np.array([[d1.real,-d1.imag],[d1.imag,d1.real]])
    hess=np.array([[[d2.real,-d2.imag],[-d2.imag,-d2.real]],[[d2.imag,d2.real],[d2.real,-d2.imag]]])
    d=jax.jacrev if reverse else jax.jacfwd
    p=jnp.asarray(point)
    np.testing.assert_allclose(d(fn)(p),jac,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(d(d(fn))(p),hess,rtol=2e-14,atol=2e-14)
@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('tangent',[1.+0j,1j,.3-.7j])
def test_subnormal_staged_transverse_jvp(method,tangent):
    fn=_UNARY_FNS[method]
    _,actual=jax.jvp(fn,(jnp.asarray(.4+0j),),(jnp.asarray(tangent),))
    expected=_derivatives(method,.4+0j)[0]*tangent
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)

BOUNDARY=json.loads((Path(__file__).with_name('data')/'subnormal_staged_boundaries_native_20261018.json').read_text())
@pytest.mark.parametrize('mode',['eager','jit'])
@pytest.mark.parametrize('row',BOUNDARY['rows'],ids=lambda r:r['method'])
def test_subnormal_staged_native_helper_boundaries(row,mode):
    fn=_UNARY_FNS[row['method']]
    actual=(fn if mode=='eager' else jax.jit(fn))(_input(row))
    assert actual.shape==tuple(row['native']['shape']) and jnp.iscomplexobj(actual)
    assert [w for w,k in zip(_words(jnp.imag(actual)),row['mask_F']) if k]==[w for w,k in zip(row['native']['imag'],row['mask_F']) if k]
@pytest.mark.parametrize('kind',['exp','cosh'])
def test_subnormal_coefficient_source_boundaries(kind):
    from chebfunjax.utils.elementary import _subnormal_cosh_coefficient, _subnormal_exp_coefficient
    row=BOUNDARY['providers'][kind]
    x=jnp.asarray(np.array([int(w,16) for w in row['input']['real']],dtype=np.uint64).view(np.float64))
    fn=_subnormal_exp_coefficient if kind=='exp' else _subnormal_cosh_coefficient
    target=row['output']['real']
    selected=[0x0010000000000000<=int(w,16)&0x7fffffffffffffff<0x7ff0000000000000 for w in target]
    for f in [fn,jax.jit(fn)]:
        actual=_words(f(x))
        assert [w for w,k in zip(actual,selected) if k]==[w for w,k in zip(target,selected) if k]
        if kind=='exp':
            # Four captured subnormal-coefficient provider losses stay outside
            # the normal-coefficient repair; its fallback is unchanged.
            subnormal=[0 < int(w,16)&0x7fffffffffffffff < 0x0010000000000000 for w in target]
            infinite=[int(w,16)&0x7fffffffffffffff == 0x7ff0000000000000 for w in target]
            assert sum(subnormal)==4 and sum(infinite)==4
            assert [w for w,k in zip(actual,subnormal) if k]==['0000000000000000']*4
            assert [w for w,k in zip(actual,infinite) if k]==[w for w,k in zip(target,infinite) if k]
