"""Greedy Loewner controls for aaatrig.m160-190, Chebfun7574c77.

Analytic controls supplement native tests. A multidimensional nullspace
has no unique weight basis; tests constrain its residual and public values.
"""
import importlib

import jax.numpy as jnp
import pytest

module = importlib.import_module('chebfunjax.utils.aaa')


def test_real_wide_unique_nullspace():
    # A=[1,2]; its unique nullspace is span([2,-1]). Reduced V misses it.
    w = module._trig_greedy_weights_source(
        jnp.ones((1, 2)), jnp.zeros(1), jnp.array([-1., -2.]))
    expected = jnp.array([2., -1.])/jnp.sqrt(5.)
    assert w.shape == (2,)
    assert abs(abs(jnp.vdot(expected, w))-1) < 1e-14
    assert abs(jnp.array([1., 2.])@w) < 1e-14


def test_complex_wide_nullspace():
    c = jnp.array([[1., 2j, 3., 1-2j], [2+1j, -1., 3j, 2.]])
    f = jnp.array([1+2j, -.5j])
    fj = jnp.array([-1., .2+1j, 2., -3j])
    a = f[:, None]*c-c*fj[None, :]
    w = module._trig_greedy_weights_source(c, f, fj)
    assert w.shape == (4,)
    assert abs(jnp.linalg.norm(w)-1) < 1e-14
    assert jnp.linalg.norm(a@w) < 1e-13*jnp.linalg.norm(a)


@pytest.mark.parametrize('rows', [2, 3])
def test_square_and_tall_smallest_singular_direction(rows):
    # Distinct singular values 1 and 2 specify the first coordinate uniquely.
    c = jnp.zeros((rows, 2), dtype=jnp.complex128)
    c = c.at[0, 0].set(1j).at[1, 1].set(2.)
    w = module._trig_greedy_weights_source(c, jnp.ones(rows), jnp.zeros(2))
    assert w.shape == (2,)
    assert abs(abs(w[0])-1) < 1e-14
    assert abs(w[1]) < 1e-14


@pytest.mark.parametrize('columns', [1, 3])
def test_zero_row_full_v_shape(columns):
    # Native requests V(:,m) even for a0-by-m matrix. No unique basis claimed.
    w = module._trig_greedy_weights_source(
        jnp.empty((0, columns), dtype=jnp.complex128), jnp.empty(0), jnp.ones(columns))
    assert w.shape == (columns,)
    assert jnp.all(jnp.isfinite(w))
    assert abs(jnp.linalg.norm(w)-1) < 1e-14


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_three_samples_uses_wide_nullspace(form):
    z = jnp.array([.1, .8, 1.7])
    f = jnp.array([2., 3., 5.])
    out = module.aaatrig(f, z, form=form, lawson=0, cleanup=False)
    r, _, _, _, zj, fj, wj, errors = out
    # Native greedy support order: value5 farthest from mean, then value2.
    # At m=2 the1x2 Loewner nullspace interpolates the remaining value3.
    assert len(zj) == 2
    assert jnp.array_equal(zj, z[jnp.array([2, 0])])
    assert jnp.array_equal(fj, f[jnp.array([2, 0])])
    assert len(errors) == 2
    assert jnp.max(jnp.abs(r(z)-f)) < 1e-12
    c = 1/jnp.tan((z[1]-zj)/2) if form == 'even' else 1/jnp.sin((z[1]-zj)/2)
    a = (f[1]-fj)*c
    expected = jnp.array([a[1], -a[0]])
    expected = expected/jnp.linalg.norm(expected)
    assert abs(abs(jnp.vdot(expected, wj))-1) < 1e-13
    x = jnp.linspace(.2, 1.5, 17)
    c = 1/jnp.tan((x[:, None]-zj)/2) if form == 'even' else 1/jnp.sin((x[:, None]-zj)/2)
    reference = (c@(expected*fj))/(c@expected)
    assert jnp.max(jnp.abs(r(x)-reference)) < 1e-12


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_four_samples_multidimensional_nullspace(form):
    z = jnp.array([.1, .8, 1.7, 2.9])
    f = jnp.array([2., 3., 5., 7.])
    r, _, _, _, zj, _, wj, _ = module.aaatrig(f, z, form=form, lawson=0, cleanup=False)
    assert len(zj) == 3
    assert jnp.all(jnp.isfinite(wj))
    assert jnp.max(jnp.abs(r(z)-f)) < 1e-12


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_single_sample_zero_row(form):
    r, _, _, _, zj, fj, wj, errors = module.aaatrig(
        jnp.array([2+3j]), jnp.array([.4]), form=form, lawson=0, cleanup=False)
    assert len(zj) == len(fj) == len(wj) == len(errors) == 1
    assert jnp.max(jnp.abs(r(jnp.array([.2, .4, .8]))-(2+3j))) < 1e-13
