"""Independent dyadic controls for R2017a QP primitives; not full SQP parity."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_qp import (
    NEGATIVE_CURVATURE,
    NEWTON,
    STEEPEST_DESCENT,
    choltrap,
    compdir,
    find_blocking_constraint,
)


def f64(value):
    return jnp.asarray(value, dtype=jnp.float64)


# Expected answers come from independent 2x2 quadratic/null-space algebra.
DIRECTIONS = [
    ('positive', [[4., 0.], [0., 16.]], [8., -16.], [[1., 0.], [0., 1.]], [-2., 1.], NEWTON),
    ('projected', [[4., 0.], [0., 16.]], [8., -16.], [[0.], [1.]], [0., 1.], NEWTON),
    ('negative_flip', [[-1., 0.], [0., 2.]], [1., 0.], [[1., 0.], [0., 1.]], [-1., 0.], NEGATIVE_CURVATURE),
    ('negative_keep', [[-1., 0.], [0., 2.]], [-1., 0.], [[1., 0.], [0., 1.]], [1., 0.], NEGATIVE_CURVATURE),
    ('negative_last', [[1., 2.], [2., 1.]], [1., 0.], [[1., 0.], [0., 1.]], [-2., 1.], NEGATIVE_CURVATURE),
    ('semidefinite', [[0., 0.], [0., 1.]], [2., 3.], [[1., 0.], [0., 1.]], [-2., -3.], STEEPEST_DESCENT),
    ('upper_triangle', [[4., 2.], [100., 5.]], [6., 7.], [[1., 0.], [0., 1.]], [-1., -1.], NEWTON),
    ('threshold_equal', [[-2.**-26, 0.], [0., 1.]], [2., 3.], [[1., 0.], [0., 1.]], [-2., -3.], STEEPEST_DESCENT),
    ('shallow_negative', [[-2.**-28, 0.], [0., 1.]], [2., 3.], [[1., 0.], [0., 1.]], [-2., -3.], STEEPEST_DESCENT),
]


@pytest.mark.parametrize('name,h,g,z,expected,kind', DIRECTIONS, ids=[v[0] for v in DIRECTIONS])
def test_direction(name, h, g, z, expected, kind):
    h, g, z = f64(h), f64(g), f64(z)
    for run in (compdir, jax.jit(compdir)):
        result = run(z, h, g)
        assert jnp.array_equal(result.step, f64(expected))
        assert result.kind == kind
        assert g @ result.step <= 0
        if kind == NEWTON:
            # Native chol reads only the upper triangle. The asymmetric fixture
            # tests that contract, not stationarity for the nonsymmetric input.
            model = f64([[4., 2.], [2., 5.]]) if name == 'upper_triangle' else h
            assert jnp.array_equal(z.T @ (model @ result.step + g), f64([0.]*z.shape[1]))
        elif kind == NEGATIVE_CURVATURE:
            assert result.step @ h @ result.step < 0
    # Direct choltrap branches not necessarily entered after successful chol.
    if name == 'positive':
        for run in (choltrap, jax.jit(choltrap)):
            ell, negative, present = run(h)
            assert not present and jnp.array_equal(negative, f64([0., 0.]))
            assert jnp.array_equal(ell, f64([[2., 0.], [0., 4.]]))
        with pytest.raises(ValueError):
            compdir(f64([[1.]]), h, g)
        with pytest.raises(ValueError):
            choltrap(jnp.eye(3, dtype=jnp.float64))
        with pytest.raises(TypeError):
            compdir(z.astype(jnp.complex128), h, g)
    elif name == 'projected':
        for run in (choltrap, jax.jit(choltrap)):
            ell, negative, present = run(f64([[16.]]))
            assert not present and jnp.array_equal(ell, f64([[4.]]))
            assert jnp.array_equal(negative, f64([0.]))
    elif name in ('negative_flip', 'semidefinite'):
        # Same source failure branches also apply to a one-dimensional projection.
        for run in (compdir, jax.jit(compdir)):
            result = run(f64([[1.], [0.]]), h, g)
            target = [-1., 0.] if name == 'negative_flip' else [-2., 0.]
            assert jnp.array_equal(result.step, f64(target)) and result.kind == kind
    elif name == 'negative_last':
        for run in (choltrap, jax.jit(choltrap)):
            ell, negative, present = run(h)
            assert present
            assert jnp.array_equal(ell, f64([[1., 0.], [2., 1.]]))
            assert jnp.array_equal(negative, f64([-2., 1.]))
            assert negative @ h @ negative == -3


BLOCKERS = ['zero', 'unique', 'tie', 'active', 'threshold_equal',
            'threshold_above', 'positive_violation']


@pytest.mark.parametrize('name', BLOCKERS)
def test_blocking(name):
    a = f64([[-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
    step, residual = f64([1., 0.]), f64([-1., -1., -0.5, -1.])
    active = jnp.zeros(4, dtype=jnp.bool_)
    threshold = f64(0.125)
    eligible, index, distance = [False, False, True, False], 2, 0.5
    if name == 'zero':
        step = f64([0., 0.])
        eligible, index, distance = [False]*4, -1, 1e16
    elif name in ('tie', 'active'):
        step, residual = f64([1., 1.]), f64([-1., -1., -0.5, -0.5])
        eligible = [False, False, True, True]
        if name == 'active':
            active = active.at[2].set(True)
            eligible, index = [False, False, False, True], 3
    elif name in ('threshold_equal', 'threshold_above'):
        # General four-row native helper: compare exact threshold without sqrt rounding.
        coefficient = 0.125 if name == 'threshold_equal' else float.fromhex('0x1.0000000000001p-3')
        a = a.at[2, 0].set(coefficient)
        residual = residual.at[2].set(-coefficient)
        if name == 'threshold_equal':
            eligible, index, distance = [False]*4, -1, 1e16
        else:
            distance = 1.
    elif name == 'positive_violation':
        residual = residual.at[2].set(0.125)
        distance = 0.125
    for run in (find_blocking_constraint, jax.jit(find_blocking_constraint)):
        result = run(a, step, residual, threshold, active)
        assert jnp.array_equal(result.eligible, jnp.asarray(eligible))
        assert result.index == index
        assert result.distance == distance
    if name == 'zero':
        with pytest.raises(ValueError):
            find_blocking_constraint(a[:3], step, residual, threshold, active)
        with pytest.raises(TypeError):
            find_blocking_constraint(a, step, residual, threshold, active.astype(jnp.int32))
