"""SQP mechanics controls with explicit analytic injection, NOT native FD parity."""
from fractions import Fraction

import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_sqp import active_set_box


def f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def test_missing_native_difference_provider_stays_explicit():
    calls = []
    with pytest.raises(RuntimeError, match='provider is unavailable'):
        active_set_box(lambda x: calls.append(x), f64([0., 0.]),
                       f64([-3., -3.]), f64([3., 3.]), finite_difference=None)
    assert calls == []


def test_actual_sqp_with_explicit_analytic_provider():
    calls = []
    target = f64([1., -2.])

    def objective(x):
        calls.append(x)
        return ((x-target)@(x-target))/2

    def analytic_provider(x, fun, lower, upper, base, options):
        assert options['FinDiffType'] == 'forward'
        assert jnp.array_equal(options['TypicalX'], f64([1., 1.]))
        # Two actual provider calls exercise accounting; analytic gradient
        # injection still does not assert protected finite-difference semantics.
        fun(x)
        fun(x)
        return x-target, 2

    result = active_set_box(objective, f64([0., 0.]), f64([-3., -3.]),
                            f64([3., 3.]), finite_difference=analytic_provider)
    assert result.exitflag == 1
    assert jnp.array_equal(result.x, target)
    assert result.value == 0
    assert result.iterations == 1 and result.evaluations == len(calls) == 6


def test_literal_equal_merit_acceptance_and_bfgs_transition():
    class Captured(Exception):
        pass

    records = []

    def observer(event, iteration, state):
        records.append((event, iteration, state))
        if event == 'bfgs':
            raise Captured

    # f=(x-1)^2+y^2: first identity-Hessian step is(2,0).
    # Equal objective at x=0 and x=2 is accepted by the native strict-worsening
    # AND condition. Next BFGS has s=(2,0),y=(4,0), giving diag(2,1) exactly.
    def objective(x):
        return (x[0]-1)**2+x[1]**2

    def analytic_provider(x, fun, lower, upper, base, options):
        return 2*(x-f64([1., 0.])), 0

    with pytest.raises(Captured):
        active_set_box(objective, f64([0., 0.]), f64([-3., -3.]),
                       f64([3., 3.]), finite_difference=analytic_provider, observer=observer)
    searches = [state for event, _, state in records if event == 'line_search']
    assert len(searches) == 1
    assert searches[0]['trial'] == 1
    assert jnp.array_equal(searches[0]['x'], f64([2., 0.]))
    update = records[-1][2]
    assert jnp.array_equal(update['displacement'], f64([2., 0.]))
    assert jnp.array_equal(update['corrected_y'], f64([4., 0.]))
    assert jnp.array_equal(update['hessian'], f64([[2., 0.], [0., 1.]]))


def test_rejected_full_trial_then_half_step_acceptance():
    class Captured(Exception):
        pass

    records = []

    def observer(event, iteration, state):
        records.append((event, iteration, state))
        if event == 'bfgs':
            raise Captured

    def objective(x):
        return 2*(x[0]-1)**2+x[1]**2

    def analytic_provider(x, fun, lower, upper, base, options):
        return f64([4*(x[0]-1), 2*x[1]]), 0

    with pytest.raises(Captured):
        active_set_box(objective, f64([0., 0.]), f64([-8., -8.]),
                       f64([8., 8.]), finite_difference=analytic_provider, observer=observer)
    trials = [state for event, _, state in records if event == 'line_search']
    assert len(trials) == 2
    assert [float(state['trial']) for state in trials] == [1., 0.5]
    assert [float(state['value']) for state in trials] == [18., 2.]
    assert jnp.array_equal(trials[0]['x'], f64([4., 0.]))
    assert jnp.array_equal(trials[1]['x'], f64([2., 0.]))
    update = records[-1][2]
    assert jnp.array_equal(update['displacement'], f64([2., 0.]))
    assert jnp.array_equal(update['corrected_y'], f64([8., 0.]))
    assert jnp.array_equal(update['hessian'], f64([[4., 0.], [0., 1.]]))


def test_actual_negative_curvature_repair_sequence():
    class Captured(Exception):
        pass

    records = []

    def observer(event, iteration, state):
        records.append((event, iteration, state))
        if event == 'bfgs':
            raise Captured

    def objective(x):
        return -x[0]**2

    def analytic_provider(x, fun, lower, upper, base, options):
        return f64([-2*x[0], 0.]), 0

    with pytest.raises(Captured):
        active_set_box(objective, f64([1., 0.]), f64([-8., -8.]),
                       f64([8., 8.]), finite_difference=analytic_provider, observer=observer)
    trials = [state for event, _, state in records if event == 'line_search']
    assert len(trials) == 1 and trials[0]['trial'] == 1
    assert jnp.array_equal(trials[0]['x'], f64([3., 0.]))
    update = records[-1][2]
    # y=(-4,0), s=(2,0): twenty first-minimum halvings stop at
    # y0=-2^-18, curvature=-2^-17 >= -1e-5. Native factor is(4,0).
    # The first second-loop addition is exactly represented .01*4 then
    # one rounded subtraction of2^-18. Fraction is independent of JAX.
    expected = float(4*Fraction.from_float(0.01)-Fraction(1, 2**18))
    assert jnp.array_equal(update['displacement'], f64([2., 0.]))
    assert jnp.array_equal(update['corrected_y'], f64([expected, 0.]))
    assert 0 < update['hessian'][0, 0] < 1
    assert update['hessian'][1, 1] == 1


def test_positive_provider_count_cap_restores_original_state():
    calls, providers, trials = [], [], []

    def objective(x):
        calls.append(x)
        return 2.5*(x[0]-1)**2+x[1]**2

    def provider(x, fun, lower, upper, base, options):
        # Synthetic provider spends actual calls, not fake FD results/counts.
        # Native FD behavior is still explicitly unqualified.
        count = 196 if not providers else 1
        providers.append((x, base, count))
        for _ in range(count):
            fun(x)
        return f64([5*(x[0]-1), 2*x[1]]), count

    def observer(event, iteration, state):
        if event == 'line_search':
            trials.append(state)

    result = active_set_box(objective, f64([0., 0.]), f64([-8., -8.]),
                            f64([8., 8.]), finite_difference=provider, observer=observer)
    assert [float(state['trial']) for state in trials] == [1., 0.5, 0.25]
    assert [float(state['value']) for state in trials] == [40., 5.625, 0.15625]
    assert [state['evaluations'] for state in trials] == [198, 199, 200]
    assert len(providers) == 2
    assert jnp.array_equal(providers[1][0], f64([1.25, 0.]))
    assert providers[1][1] == 0.15625
    assert result.exitflag == 0 and result.iterations == 1
    assert result.evaluations == len(calls) == 201
    assert jnp.array_equal(result.x, f64([0., 0.]))
    assert result.value == 2.5
    assert jnp.array_equal(result.gradient, f64([-5., 0.]))
