"""Source interpolation aliases and JAX real/complex interpolation controls.

Native source: @chebfun/interp1.m, Chebfun7574c77. These authored controls
supplement, and do not replace, the35literal original predicates.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('method', [None, [], ''])
def test_empty_method_defaults_to_polynomial(method):
    x = jnp.array([-1., -.3, .4, 1.])
    y = x**3-2*x+1
    f = cj.Chebfun.interp1(x, y, method)
    queries = jnp.linspace(-1, 1, 17)
    expected = queries**3-2*queries+1
    assert float(jnp.max(jnp.abs(f(queries)-expected))) < 64*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('method', ['spline', 'pchip', 'cubic'])
def test_piecewise_cubic_aliases_hit_data(method):
    x = jnp.array([-1., -.3, .4, 1.])
    y = (1+2j)*x**3+(2-1j)*x
    f = cj.Chebfun.interp1(x, y, method)
    assert float(jnp.max(jnp.abs(f(x)-y))) < 64*jnp.finfo(jnp.float64).eps*float(jnp.max(jnp.abs(y)))
    assert len(f.funs) == 3
    assert len(f) == 12


@pytest.mark.parametrize('array', [False, True])
def test_complex_linear_preserves_physical_affine_values(array):
    x = jnp.array([2., 3., 5.])
    y = (1+2j)*x+(2-1j)
    if array:
        y = jnp.stack([y, (2-1j)*x-3j], axis=1)
    f = cj.Chebfun.interp1(x, y, 'linear')
    queries = jnp.linspace(2., 5., 17)
    expected = (1+2j)*queries+(2-1j)
    if array:
        expected = jnp.stack([expected, (2-1j)*queries-3j], axis=1)
    assert jnp.iscomplexobj(f(queries))
    assert float(jnp.max(jnp.abs(f(queries)-expected))) < 32*jnp.finfo(jnp.float64).eps*float(jnp.max(jnp.abs(expected)))
    assert len(f.funs) == 2 and len(f) == 4


@pytest.mark.parametrize('method', ['trig', 'periodic'])
def test_nonuniform_complex_trigonometric_interpolation(method):
    x = jnp.array([-.9, -.5, .05, .4, .75])
    y = jnp.exp(1j*jnp.pi*x)+2
    f = cj.Chebfun.interp1(x, y, method, (-1., 1.))
    queries = jnp.linspace(-1., 1., 19)
    expected = jnp.exp(1j*jnp.pi*queries)+2
    assert float(jnp.max(jnp.abs(f(queries)-expected))) < 128*jnp.finfo(jnp.float64).eps*float(jnp.max(jnp.abs(expected)))
    assert jnp.iscomplexobj(f(queries))


@pytest.mark.parametrize('row_x,row_y', [(True, False), (False, True), (True, True)])
def test_source_row_vectors_form_scalar_interpolants(row_x, row_y):
    x = jnp.arange(11.)
    y = jnp.sin(x)
    f = cj.Chebfun.interp1(x[None, :] if row_x else x,
                         y[None, :] if row_y else y, 'linear')
    result = f(x)
    print('row input diagnostics:', row_x, row_y, result.shape,
          'flattened error:', float(jnp.linalg.norm(result.reshape(-1)-y)))
    assert result.shape == x.shape
    assert float(jnp.linalg.norm(result-y)) < 10*float(f.vscale)*jnp.finfo(jnp.float64).eps
