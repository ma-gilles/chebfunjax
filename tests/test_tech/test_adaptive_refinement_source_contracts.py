"""Source callback-order fixtures for adaptive refinement.

MATLAB provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
@chebtech1/refine.m, @chebtech2/refine.m, @chebtech/populate.m and
@chebtech/techPref.m. Sampling batches are compared to the literal source grids.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    _next_resampling_n,
    _refine_chebtech1_nested,
    _refine_chebtech1_resampling,
    _refine_chebtech2_nested,
    _refine_chebtech2_resampling,
    _update_running_vscale,
)
from chebfunjax.utils.quadrature import chebpts


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, x):
        x = jnp.asarray(x)
        self.calls.append(x)
        return jnp.stack((x, x**2), axis=1)


def _assert_same(actual, expected):
    expected = jnp.asarray(expected)
    assert actual.shape == expected.shape
    assert jnp.max(jnp.abs(actual - expected), initial=0.0) < 1e-14


def test_matlab_default_initial_and_resampling_length_schedule():
    assert _next_resampling_n(17) == 33
    assert _next_resampling_n(33) == 65
    assert _next_resampling_n(65) == 93
    assert _next_resampling_n(93) == 129
    assert _next_resampling_n(129) == 183
    # MATLAB minSamples=17 gives 2^ceil(log2(16))+1=17.
    assert _refine_chebtech2_resampling(_Recorder(), None)[0].shape == (17, 2)
    # minSamples need not itself be a 2^q+1 count: 18 rounds up to 33.
    assert _refine_chebtech1_resampling(_Recorder(), None, min_samples=18)[0].shape == (33, 2)


def test_chebtech2_nested_reuses_old_values_and_only_calls_new_nodes():
    recorder = _Recorder()
    x17 = chebpts(17, kind=2)
    values17, gave_up = _refine_chebtech2_nested(recorder)
    assert not gave_up
    assert recorder.calls[0].shape == (17,)
    _assert_same(recorder.calls[0], x17)

    values33, gave_up = _refine_chebtech2_nested(recorder, values17)
    assert not gave_up
    x33 = chebpts(33, kind=2)
    expected_new_nodes = x33[1:-1:2]
    assert len(recorder.calls) == 2
    _assert_same(recorder.calls[1], expected_new_nodes)
    _assert_same(values33, jnp.stack((x33, x33**2), axis=1))


def test_chebtech1_nested_uses_triples_and_prescribed_two_call_order():
    recorder = _Recorder()
    values17, gave_up = _refine_chebtech1_nested(recorder)
    assert not gave_up
    _assert_same(recorder.calls[0], chebpts(17, kind=1))

    values51, gave_up = _refine_chebtech1_nested(recorder, values17)
    assert not gave_up
    x51 = chebpts(51, kind=1)
    assert [call.shape[0] for call in recorder.calls] == [17, 17, 17]
    _assert_same(recorder.calls[1], x51[::3])
    _assert_same(recorder.calls[2], x51[2::3])
    _assert_same(values51, jnp.stack((x51, x51**2), axis=1))


def test_resampling_reinvokes_full_grid_and_vscale_is_columnwise_finite_only():
    recorder = _Recorder()
    values17, gave_up = _refine_chebtech1_resampling(recorder)
    assert not gave_up
    values33, gave_up = _refine_chebtech1_resampling(recorder, values17)
    assert not gave_up
    assert [call.shape[0] for call in recorder.calls] == [17, 33]
    _assert_same(recorder.calls[1], chebpts(33, kind=1))
    sampled = jnp.asarray([[2.0, 9.0], [jnp.nan, 3.0], [jnp.inf, 5.0]])
    _assert_same(_update_running_vscale(sampled, jnp.array([1.0, 10.0])), [2.0, 10.0])


def test_max_length_source_policies_differ_by_tech_and_refinement():
    f = _Recorder()
    # Chebtech2 nested gives up rather than snapping 17 -> 33 to maxLength 20.
    old2, _ = _refine_chebtech2_nested(f, max_length=20)
    unchanged2, give_up2 = _refine_chebtech2_nested(f, old2, max_length=20)
    assert give_up2 and unchanged2.shape == (17, 2)

    # Chebtech1 nested takes one final full sample exactly at maxLength=40.
    f1 = _Recorder()
    old1, _ = _refine_chebtech1_nested(f1, max_length=40)
    final1, give_up1 = _refine_chebtech1_nested(f1, old1, max_length=40)
    assert not give_up1 and final1.shape == (40, 2)
    _assert_same(f1.calls[-1], chebpts(40, kind=1))


def test_chebtech2_extrapolate_resampling_marks_endpoints_for_populate():
    f = _Recorder()
    values, gave_up = _refine_chebtech2_resampling(f, extrapolate=True)
    assert not gave_up
    assert values.shape == (17, 2)
    assert jnp.all(jnp.isnan(values[jnp.array([0, -1]), :]))
    assert f.calls[0].shape == (15,)
    _assert_same(f.calls[0], chebpts(17, kind=2)[1:-1])


@pytest.mark.parametrize("tech,cap,batches,n", [
    (Chebtech1, 40, [17, 40], 40),
    (Chebtech2, 40, [17, 16], 33),
])
def test_public_constructor_respects_nested_callback_and_cap_contract(tech, cap, batches, n):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.cos(80 * x)

    with pytest.warns(UserWarning, match="did not converge"):
        result = tech.from_function(op, max_length=cap, sample_test=False)
    assert not result.ishappy
    assert result.n == n
    assert [x.size for x in calls] == batches
    kind = 1 if tech is Chebtech1 else 2
    _assert_same(calls[0], chebpts(17, kind=kind))
    expected = chebpts(n, kind=kind)
    _assert_same(calls[1], expected if kind == 1 else expected[1:-1:2])


@pytest.mark.parametrize("tech,kind", [(Chebtech1, 1), (Chebtech2, 2)])
def test_public_constructor_resampling_preferences_use_full_grids(tech, kind):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.cos(80 * x)

    with pytest.warns(UserWarning, match="did not converge"):
        result = tech.from_function(op, refinement_function="ReSaMpLiNg",
                                   min_samples=18, max_length=65, sample_test=False)
    assert not result.ishappy
    assert result.n == 65
    assert [x.size for x in calls] == [33, 65]
    for x in calls:
        _assert_same(x, chebpts(x.size, kind=kind))


@pytest.mark.parametrize("tech,kind", [(Chebtech1, 1), (Chebtech2, 2)])
def test_custom_refiner_receives_restored_whole_nonfinite_rows(tech, kind):
    calls = []

    def op(x):
        y = jnp.cos(80 * x)
        return jnp.stack((y, 2 * y), axis=1)

    def refine(function, old, pref):
        calls.append(old)
        assert pref["maxLength"] == 17
        assert pref["minSamples"] == 17
        assert pref["chebfuneps"] == float(jnp.finfo(jnp.float64).eps)
        assert jnp.isnan(pref["fixedLength"])
        assert pref["sampleTest"] is False
        assert pref["refinementFunction"] is refine
        assert pref["happinessCheck"] == "standard"
        assert pref["useTurbo"] is False
        assert pref["extrapolate"] is False
        if old is None:
            samples = function(chebpts(17, kind=kind))
            # Source row masks restore both columns. Inf restoration discards
            # the original sign, and Inf wins when a row also held NaN.
            samples = samples.at[0, 0].set(jnp.nan)
            samples = samples.at[1, 1].set(-jnp.inf)
            samples = samples.at[2, 0].set(jnp.nan)
            samples = samples.at[2, 1].set(-jnp.inf)
            return samples, False
        assert jnp.all(jnp.isnan(old[0, :]))
        assert jnp.all(jnp.isposinf(old[1:3, :]))
        return old, True

    with pytest.warns(UserWarning, match="did not converge"):
        result = tech.from_function(op, refinement_function=refine,
                                   max_length=17, sample_test=False)
    assert not result.ishappy
    assert len(calls) == 2
    assert jnp.all(jnp.isfinite(result.coeffs))


@pytest.mark.parametrize("tech,kind", [(Chebtech1, 1), (Chebtech2, 2)])
def test_happy_constructor_runs_source_probes_after_initial_grid(tech, kind):
    calls = []

    def op(x):
        calls.append(x)
        return 1 + x + x**2

    result = tech.from_function(op)
    assert result.ishappy
    assert [x.size for x in calls] == [17, 2]
    _assert_same(calls[0], chebpts(17, kind=kind))
    _assert_same(calls[1], [-0.357998918959666, 0.036785641195074])
    _assert_same(result(jnp.array([-0.5, 0.5])), [0.75, 1.75])


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_immediate_custom_giveup_has_controlled_python_error(tech):
    def refine(op, old, pref):
        assert old is None
        return old, True

    with pytest.raises(ValueError, match="gave up before sampling"):
        tech.from_function(jnp.sin, refinement_function=refine)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_single_sample_constant_has_bounded_unhappy_python_exit(tech):
    calls = []

    def op(x):
        calls.append(x)
        return 3 * jnp.ones_like(x)

    # Source standardCheck requires cutoff<n. At n=1 it is unhappy;
    # Tech2's source nested loop would then never increase its grid. Python
    # bounds that degenerate case by giving up, preserving the sampled tech.
    with pytest.warns(UserWarning, match="did not converge"):
        result = tech.from_function(op, min_samples=1, max_length=1)
    assert not result.ishappy
    assert result.n == 1
    assert [x.size for x in calls] == [1]
    _assert_same(calls[0], [0.0])
    _assert_same(result.coeffs, [3.0])
