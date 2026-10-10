"""Supplemental boundary/orientation/complex controls for source sign route."""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('op,expected', [('lt',0),('le',1),('gt',0),('ge',1)])
def test_identically_equal_intervals(op, expected):
    f = cj.chebfun(jnp.exp)
    h = getattr(f, op)(f)
    assert len(h.funs) == 1
    np.testing.assert_array_equal(h(jnp.array([-1., 0., 1.])), [expected]*3)
    np.testing.assert_array_equal(h.point_values, [expected]*2)

def test_sign_keeps_complex_phase():
    f = cj.chebfun(lambda x: 2*jnp.exp(1j*x))
    g = f.sign()
    x = jnp.linspace(-1., 1., 31)
    np.testing.assert_allclose(g(x), jnp.exp(1j*x), rtol=0, atol=1e-13)
    np.testing.assert_allclose(g.point_values, jnp.exp(1j*jnp.asarray(g.domain.breakpoints)), rtol=0, atol=1e-13)

def test_row_orientation():
    f = cj.chebfun(lambda x: x-.25).transpose()
    assert f.sign().is_transposed
    for op in ('lt','le','gt','ge'):
        assert getattr(f, op)(0.).is_transposed

def test_sign_empty():
    assert cj.chebfun().sign().isempty()

def test_sign_array_columns():
    f = cj.chebfun(lambda x: jnp.stack((x-.25, x+.5), axis=-1))
    g = f.sign()
    x = jnp.asarray([-.8, -.2, .6])
    np.testing.assert_array_equal(g(x), jnp.sign(jnp.stack((x-.25,x+.5),axis=-1)))

def test_singfun_sign_direct():
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2
    f = Singfun(Chebtech2.from_function(lambda x: 2+x), (-1., -.5))
    g = f.sign()
    assert isinstance(g, Chebtech2)
    np.testing.assert_array_equal(g(jnp.asarray([-1., 0., 1.])), [1.,1.,1.])
