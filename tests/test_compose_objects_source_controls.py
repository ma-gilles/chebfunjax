"""Controls for source typed dispatch and stored point values.

Provenance
----------
MATLAB source : @chebfun/compose.m, extractColumns.m, isPeriodicTech.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


@pytest.mark.parametrize('dim, vector', [(2, False), (2, True), (3, False), (3, True)])
def test_multidimensional_operator_rejects_row_inner(dim, vector):
    f = cj.chebfun(lambda t: jnp.stack([t]*dim, axis=-1)).T
    if dim == 2:
        op = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y) if vector else Chebfun2.from_function(lambda x, y: x+y)
    else:
        op = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y) if vector else chebfun3(lambda x, y, z: x+y+z)
    label = f'Cheb{dim}'+('V' if vector else '')+'ofCheb'
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:compose:'+label):
        f.compose(op)


def test_scalar_quasi_composition_preserves_exact_point_values():
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    f = cj.chebfun(lambda x: .2*x).set_point_values(jnp.array([.37, .51]))
    a, b = cj.chebfun(jnp.sin), cj.chebfun(jnp.cos)
    outer = Quasimatrix([a, b], a.domain)
    h = outer(f)
    assert jnp.array_equal(h(jnp.array([-1., 1.])), outer(f(jnp.array([-1., 1.]))))


def test_multicolumn_typed_composition_preserves_point_values():
    f = cj.chebfun(lambda x: jnp.stack((jnp.sin(x-.13), jnp.cos(x+.31)), axis=-1))
    op = Chebfun2.from_function(lambda x, y: x+y)
    h = f.compose(op)
    endpoints = f(jnp.array([-1., 1.]))
    expected = op(endpoints[:, 0], endpoints[:, 1])
    assert jnp.array_equal(h(jnp.array([-1., 1.])), expected)


def test_domain_dimension_and_orientation_errors():
    f, g = cj.chebfun(lambda x: 2*x), cj.chebfun(jnp.exp)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:compose:domain'):
        f.compose(g)
    with pytest.raises(ValueError, match='composeTwoChebfuns:trans'):
        g.compose(g.T)
    a = cj.chebfun(lambda x: jnp.stack((x, x), axis=-1))
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:compose:trans'):
        a.compose(a)


def test_periodic_call_and_direct_compose_match_and_jit_eval():
    f = cj.chebfun(lambda x: .5*jnp.sin(jnp.pi*x), trig=True)
    g = cj.chebfun(jnp.exp)
    h = g(f)
    direct = f.compose(g)
    assert h.isPeriodicTech()
    assert jnp.array_equal(h.funs[0].coeffs, direct.funs[0].coeffs)
    x = jnp.linspace(-1., 1., 51)
    assert jnp.max(jnp.abs(jax.jit(lambda t: h(t))(x)-jnp.exp(.5*jnp.sin(jnp.pi*x)))) < 2e-14
