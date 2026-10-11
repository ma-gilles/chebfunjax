"""Supplemental shape, dispatch and differentiability controls.

Provenance
----------
MATLAB source : @chebfun/dct.m, idct.m, dst.m, idst.m
Chebfun commit: 7574c77
These API/JAX tests supplement the separately restored original 64 slots.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('name', ['dct','idct','dst','idst'])
def test_public_static_and_factory_dispatch(name):
    assert getattr(cj.Chebfun,name) is getattr(cj,name)
    assert getattr(cj.chebfun,name) is getattr(cj,name)
    x = jnp.asarray([.2,.4,.6,.8])
    kind = 2 if 'dct' in name else 1
    np.testing.assert_array_equal(getattr(cj,name)(x),getattr(cj,name)(x,kind))


@pytest.mark.parametrize('name', ['dct','dst'])
@pytest.mark.parametrize('kind', range(1,9))
def test_jit_complex_columns_and_linear_jvp(name,kind):
    def fn(x):
        return getattr(cj,name)(x,kind)
    x = jnp.arange(18,dtype=jnp.float64).reshape(9,2)/17
    x = x + 1j*(.7+x[::-1])
    tangent = jnp.cos(jnp.real(x)) + 1j*jnp.sin(.2+jnp.real(x))
    eager = fn(x)
    traced = jax.jit(fn)(x)
    columns = jnp.stack([fn(x[:,0]),fn(x[:,1])],axis=1)
    _,derivative = jax.jvp(fn,(x,),(tangent,))
    for actual,expected in [(traced,eager),(columns,eager),(derivative,fn(tangent))]:
        np.testing.assert_allclose(actual,expected,rtol=0,atol=100*np.finfo(float).eps)
    assert eager.shape == x.shape and jnp.iscomplexobj(eager)


@pytest.mark.parametrize('name', ['dct','idct','dst','idst'])
def test_invalid_type_source_identifier(name):
    with pytest.raises(ValueError,match='CHEB.*:.*:type'):
        getattr(cj,name)(jnp.ones(4),9)


def test_idst8_preserves_native_unsupported_error():
    with pytest.raises(ValueError,match='CHEBFUN:CHEBFUN:IDST:EIGHT'):
        cj.idst(jnp.ones(4),8)


@pytest.mark.parametrize('kind', range(1,9))
def test_real_matrix_shape_and_dtype(kind):
    x = jnp.arange(8).reshape(4,2)
    for fn in [cj.dct,cj.dst]:
        y = fn(x,kind)
        assert y.shape == x.shape and y.dtype == jnp.float64


@pytest.mark.parametrize("values, expected", [
    (1j, 1.),
    ([1j, 2j, 3j], [4.82842712474619, -2., .8284271247461903]),
    (0j, 0.),
    ([0j, 0j, 0j], [0., 0., 0.]),
])
def test_dst1_native_pure_imaginary_and_zero_branch(values, expected):
    # Fresh R2025b source capture; supplemental comparison, not a new native
    # test predicate. The source intentionally returns real values here.
    x = jnp.asarray(values, dtype=jnp.complex128)
    result = cj.dst(x, 1)
    bound = 4*np.finfo(float).eps*max(1.,float(np.max(np.abs(expected))))
    np.testing.assert_allclose(result, expected, rtol=0, atol=bound)
    np.testing.assert_allclose(jax.jit(lambda v:cj.dst(v,1))(x), expected,
                               rtol=0, atol=bound)
    assert not jnp.iscomplexobj(result)


def test_dst1_nonzero_complex_storage_uses_general_branch():
    # isreal(u) is false for explicitly complex storage, even with zero imag.
    result = cj.dst(jnp.asarray([1+0j,2+0j,3+0j]),1)
    assert jnp.iscomplexobj(result)
