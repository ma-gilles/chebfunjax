"""Public Spherefun evaluation: source mapping, genuine factors and JAX AD.

Provenance
----------
MATLAB source: @spherefun/feval.m, @separableApprox/feval.m,
@mapping/mapping.m, @trigtech/horner.m
Chebfun commit: 7574c77
New analytic controls; original MATLAB assertions remain separate/unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech import trigtech
from chebfunjax.tech.trigtech import Trigtech


def _field():
    def tech(c):
        return Trigtech.from_coeffs(jnp.asarray(c, dtype=jnp.complex128), is_real=True)
    one = tech([1.])
    return Spherefun(cols=[tech([2.]), tech([.5, 0., .5]),
                           tech([.5j, 0., -.5j])],
                     rows=[one, one, tech([.125, 0., .125])],
                     pivots=jnp.ones(3), idx_plus=(0, 1, 2), idx_minus=())


def _formula(lam, theta):
    return 2+jnp.cos(theta)+.25*jnp.sin(theta)*jnp.cos(lam)


@pytest.mark.parametrize('shape', [(), (7,), (7, 1), (1, 7), (2, 3), (2, 2, 3)])
def test_analytic_scalar_vector_matrix_tensor_and_outer_jit(shape, monkeypatch):
    f = _field()
    def forbidden(*args, **kwargs):
        raise AssertionError('host Horner or coefficient padding reached')
    monkeypatch.setattr(trigtech, '_trig_eval_np', forbidden)
    monkeypatch.setattr(trigtech, '_alias_trigtech', forbidden)
    n = int(np.prod(shape)) if shape else 1
    lam = jnp.linspace(-2., 1., n).reshape(shape)
    theta = jnp.linspace(.2, 2.7, n).reshape(shape)
    expected = _formula(lam, theta)
    for actual in (f(lam, theta), jax.jit(lambda x, y: f(x, y))(lam, theta)):
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, rtol=0, atol=32*np.finfo(float).eps)
    assert [g.coeffs.shape[0] for g in f.cols] == [1, 3, 3]
    assert [g.coeffs.shape[0] for g in f.rows] == [1, 1, 3]


@pytest.mark.parametrize('transpose', [False, True])
def test_tensor_grid_and_arbitrary_coordinate_tangents(transpose):
    f = _field()
    lam, theta = jnp.meshgrid(jnp.linspace(-2., 2., 5), jnp.linspace(.2, 2.7, 4))
    if transpose:
        lam, theta = lam.T, theta.T
    np.testing.assert_allclose(f(lam, theta), _formula(lam, theta), rtol=0, atol=32e-16)
    dl = jnp.sin(jnp.arange(lam.size).reshape(lam.shape))
    dt = jnp.cos(jnp.arange(theta.size).reshape(theta.shape))
    value, tangent = jax.jvp(jax.jit(lambda x, y: f(x, y)), (lam, theta), (dl, dt))
    expected = -.25*jnp.sin(theta)*jnp.sin(lam)*dl
    expected += (-jnp.sin(theta)+.25*jnp.cos(theta)*jnp.cos(lam))*dt
    np.testing.assert_allclose(value, _formula(lam, theta), rtol=0, atol=32e-16)
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=32e-16)


def test_coefficient_and_pivot_tangents():
    f = _field()
    def evaluate(weight):
        candidate = Spherefun(cols=f.cols, rows=f.rows, pivots=weight,
                              idx_plus=f.idx_plus, idx_minus=())
        return candidate(.4, .8)
    expected = -jnp.array([2., jnp.cos(.8), .25*jnp.sin(.8)*jnp.cos(.4)])
    np.testing.assert_allclose(jax.grad(evaluate)(jnp.ones(3)), expected,
                               rtol=0, atol=32e-16)
    def constant(c):
        tech = Trigtech.from_coeffs(c[None], is_real=True)
        return Spherefun(cols=[tech], rows=[f.rows[0]], pivots=jnp.ones(1),
                         idx_plus=(0,), idx_minus=())(.4, .8)
    assert float(jax.grad(constant)(jnp.asarray(2.))) == 1.


def test_source_empty_evaluation_shape():
    f = Spherefun.empty()
    assert f(.2, .3).shape == (0, 0)
    assert jax.jit(lambda x: f(x, x))(jnp.ones((2, 3))).shape == (0, 0)


@pytest.mark.parametrize('constant_longitude', [False, True])
def test_constant_coordinate_lines_evaluate_constant_factor_once(monkeypatch, constant_longitude):
    from chebfunjax.spherefun import _evaluation

    f = _field()
    original = _evaluation.factor_values
    counts = []
    def observe(factors, points):
        counts.append(points.size)
        return original(factors, points)
    monkeypatch.setattr(_evaluation, 'factor_values', observe)
    axis = jnp.linspace(.2, 2., 11)
    lam, theta = (.3, axis) if constant_longitude else (axis, .3)
    np.testing.assert_allclose(f(lam, theta), _formula(lam, theta), rtol=0, atol=32e-16)
    assert sorted(counts) == [1, 11]


def test_empty_cartesian_source_radius_check_precedes_empty_result():
    f = Spherefun.empty()
    assert f(.5, 0., 0.).shape == (0, 0)
    with pytest.raises(ValueError, match='pointsNotOnSphere'):
        f(2., 0., 0.)
