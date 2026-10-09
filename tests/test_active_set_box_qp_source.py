"""Unrun source/analytic controls for the actual finite-box QP state machine."""
import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_qp import _delete, _insert, box_qp


def f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


@pytest.mark.parametrize('name,g,expected,multipliers,active', [
    ('interior', [-4., 16.], [1., -1.], [0., 0., 0., 0.], ()),
    ('upper_edge', [-16., 0.], [2., 0.], [0., 0., 8., 0.], ()),
    ('lower_edge', [16., 0.], [-2., 0.], [8., 0., 0., 0.], ()),
    ('corner', [-16., -64.], [2., 2.], [0., 0., 8., 32.], ()),
    ('wrong_warm_face', [-24., 0.], [2., 0.], [0., 0., 16., 0.], (0,)),
    ('warm_truncate', [-24., -64.], [2., 2.], [0., 0., 16., 32.], (0, 1)),
])
def test_quadratic_solve(name, g, expected, multipliers, active):
    # Independent separable quadratic optimum: clip(-g_i/H_ii,-2,2).
    # H has exact Cholesky diag(2,4). Axis QR is a signed permutation;
    # selected blockers use half steps (or zero at the second corner face).
    # Thus these controls require exact answers through the algorithm.
    result = box_qp(f64([[4., 0.], [0., 16.]]), f64(g), f64([-2., -2.]),
                    f64([2., 2.]), active=active)
    assert result.exitflag == 1
    assert jnp.array_equal(result.x, f64(expected))
    assert jnp.array_equal(result.multipliers, f64(multipliers))
    a = f64([[-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
    assert jnp.array_equal(f64([[4., 0.], [0., 16.]])@result.x+f64(g)+a.T@result.multipliers,
                           f64([0., 0.]))


@pytest.mark.parametrize('initial', [[4., 0.], [-4., 4.]])
def test_actual_phase_one_then_quadratic(initial):
    events = []
    result = box_qp(jnp.eye(2, dtype=jnp.float64), f64([0., 0.]),
                    f64([-1., -1.]), f64([1., 1.]), initial=f64(initial),
                    observer=lambda event, iteration, active, values: events.append(event))
    assert 'phase_one_enter' in events and 'phase_one_return' in events
    assert result.exitflag == 1
    assert jnp.array_equal(result.x, f64([0., 0.]))
    assert jnp.array_equal(result.multipliers, f64([0., 0., 0., 0.]))


def test_singular_convex_compatible_system():
    result = box_qp(f64([[4., 0.], [0., 0.]]), f64([0., 0.]),
                    f64([-1., -1.]), f64([1., 1.]))
    assert result.exitflag == 1 and jnp.array_equal(result.x, f64([0., 0.]))


def test_negative_curvature_box_minimum():
    # -2*x^2+2*y^2 on the unit box has two minima. Native choltrap chooses +e1.
    result = box_qp(f64([[-4., 0.], [0., 4.]]), f64([0., 0.]),
                    f64([-1., -1.]), f64([1., 1.]))
    assert result.exitflag == 1 and jnp.array_equal(result.x, f64([1., 0.]))
    assert jnp.array_equal(result.multipliers, f64([0., 0., 4., 0.]))


def test_iteration_limit_returns_without_extra_step():
    result = box_qp(jnp.eye(2, dtype=jnp.float64), f64([-1., -1.]),
                    f64([-2., -2.]), f64([2., 2.]), max_iterations=0)
    assert result.exitflag == 0 and result.how == 'MaxSQPIter'
    assert result.iterations == 0 and jnp.array_equal(result.x, f64([0., 0.]))


def test_qr_empty_insert_and_column_delete_source_order():
    # Empty insert must discard the input Q, including native post-phase-I zero Q.
    q, r = _insert(jnp.zeros((2, 2)), jnp.empty((2, 0)), 0, f64([0., 1.]))
    assert jnp.array_equal(q@r, f64([[0.], [1.]]))
    q, r = _insert(q, r, 0, f64([1., 0.]))
    assert jnp.array_equal(q@r, jnp.eye(2))
    q, r = _delete(q, r, 0)
    assert jnp.array_equal(q@r, f64([[0.], [1.]]))


def test_invalid_scope_is_explicit():
    with pytest.raises(ValueError):
        box_qp(jnp.eye(3, dtype=jnp.float64), f64([0., 0., 0.]),
               f64([-1., -1., -1.]), f64([1., 1., 1.]))
    with pytest.raises(ValueError):
        box_qp(jnp.eye(2, dtype=jnp.float64), f64([0., 0.]),
               f64([1., -1.]), f64([1., 1.]))
