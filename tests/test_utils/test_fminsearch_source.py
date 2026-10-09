"""Analytic and source-clause controls; no MATLAB capture claim."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._fminsearch import _converged, _threshold, fminsearch


def test_initial_simplex_and_stable_ties():
    seen = []

    def objective(x):
        seen.append(x)
        return jnp.asarray(0.)

    result = fminsearch(objective, [2., -3.], max_iterations=1)
    expected = jnp.asarray([[2., -3.], [2.1, -3.], [2., -3.15]])
    assert bool(jnp.all(jnp.abs(jnp.stack(seen)-expected)
                        <= 4*jnp.finfo(jnp.float64).eps*3.15))
    assert bool(jnp.array_equal(result.x, jnp.asarray([2., -3.])))
    assert (result.iterations, result.evaluations, result.exitflag) == (1, 3, 0)


@pytest.mark.parametrize('tail,expected_points', [
    ([-1., -2.], [[.00025, -.00025], [.000375, -.0005]]),  # expand
    ([-1., -.5], [[.00025, -.00025], [.000375, -.0005]]),  # reflected wins
    ([.5], [[.00025, -.00025]]),  # reflect between best and second
    ([1.5, 1.5], [[.00025, -.00025], [.0001875, -.000125]]),  # outside <=
    ([3., 1.9], [[.00025, -.00025], [.0000625, .000125]]),  # inside <
    ([1.5, 1.6, .3, .4], [[.00025, -.00025], [.0001875, -.000125],
                          [.000125, 0.], [0., .000125]]),  # outside shrink
    ([3., 2., .3, .4], [[.00025, -.00025], [.0000625, .000125],
                       [.000125, 0.], [0., .000125]]),  # inside equality shrinks
])
def test_literal_branch_evaluation_order(tail, expected_points):
    values = iter([0., 1., 2., *tail])
    seen = []

    def objective(x):
        seen.append(x)
        return jnp.asarray(next(values))

    result = fminsearch(objective, [0., 0.], max_iterations=2)
    # Values script isolates branch predicates; it is not an optimization oracle.
    assert bool(jnp.all(jnp.abs(jnp.stack(seen[3:])-jnp.asarray(expected_points))
                        <= 4*jnp.finfo(jnp.float64).eps*.0005))
    assert result.evaluations == len(seen) == 3+len(tail)
    assert result.exitflag == 0


def test_stop_requires_both_spreads_and_native_eps_argument():
    eps = jnp.finfo(jnp.float64).eps
    assert _threshold(jnp.asarray(0.)) == eps
    assert _threshold(jnp.asarray(-1.)) == 10*eps
    assert _threshold(jnp.asarray(-.5)) == 5*eps
    # Best coordinates[-16,-1]: native eps(max(...)) is eps(-1), not eps(16).
    vertices = jnp.asarray([[-16., -16.+16*eps, -16.], [-1., -1., -1.]])
    assert not _converged(vertices, jnp.zeros(3))
    assert not _converged(jnp.zeros((2, 3)), jnp.asarray([0., 2*eps, 0.]))
    assert _converged(jnp.zeros((2, 3)), jnp.zeros(3))


def test_evaluation_limit_can_be_crossed_by_complete_shrink():
    values = iter([0., 1., 2., 3., 2., .3, .4])
    result = fminsearch(lambda x: jnp.asarray(next(values)), [0., 0.],
                        max_evaluations=4)
    assert (result.evaluations, result.iterations, result.exitflag) == (7, 2, 0)


def test_analytic_quadratic_default_limits():
    target = jnp.asarray([.25, -.5])
    objective = jax.jit(lambda x: jnp.sum((x-target)**2))
    result = fminsearch(objective, [1., -1.])
    # Hessian2I with unit quadratic coefficients: coordinate1e-12 is a declared analytic diagnostic
    # requirement, not an optimizer tolerance or fitted historical constant.
    assert bool(jnp.all(jnp.abs(result.x-target) <= 1e-12))
    assert result.fun <= 2e-24
    assert result.exitflag == 1
    assert result.iterations < 400 and result.evaluations < 400


def test_objective_exception_is_not_swallowed():
    def objective(x):
        raise RuntimeError('native caller owns fallback')

    with pytest.raises(RuntimeError, match='native caller owns fallback'):
        fminsearch(objective, [0., 0.])
