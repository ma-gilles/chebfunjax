"""Fixed callable order from @trigtech/trigtech.m136-150, pin7574c77.

Adaptive probe contracts: @trigtech/populate.m69-80. No preference redesign.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

PROBE = 2.0 * 0.376989633393435 - 1.0
EPS = jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('n', [1, 4, 5])
@pytest.mark.parametrize('complex_output', [False, True])
def test_fixed_grid_only_and_endpoint_average(n, complex_output):
    calls = []

    def op(x):
        calls.append(x)
        # Asymmetric ends make the average observable, including n=1.
        return (2 + x) * (1 + 2j if complex_output else 1)

    f = Trigtech.from_function(op, n=n)
    assert len(calls) == 1
    expected_points = jnp.concatenate((trigpts(n), jnp.ones(1)))
    assert jnp.array_equal(calls[0], expected_points)
    expected = (2 + trigpts(n)) * (1 + 2j if complex_output else 1)
    expected = expected.at[0].set(2 * (1 + 2j if complex_output else 1))
    assert f.n == n and f.ishappy
    assert f.is_real is (not complex_output)
    assert jnp.max(jnp.abs(f.values - expected)) <= 10 * EPS * jnp.max(jnp.abs(expected))


@pytest.mark.parametrize('bad', [jnp.nan, jnp.inf])
def test_fixed_ignores_nonfinite_only_at_adaptive_probe(bad):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.where(x == PROBE, bad, jnp.ones_like(x))

    f = Trigtech.from_function(op, n=4)
    assert len(calls) == 1
    assert jnp.all(jnp.isfinite(f.coeffs))
    assert jnp.max(jnp.abs(f.values - 1)) <= EPS


@pytest.mark.parametrize('bad', [jnp.nan, jnp.inf])
def test_adaptive_nonfinite_probe_still_errors_first(bad):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.full_like(x, bad)

    with pytest.raises(ValueError, match='Inf or NaN'):
        Trigtech.from_function(op)
    assert len(calls) == 1
    assert jnp.array_equal(calls[0], jnp.asarray([PROBE]))


@pytest.mark.parametrize('complex_output', [False, True])
def test_adaptive_probe_then_original_grid(complex_output):
    from chebfunjax.utils._trigpts import global_trigpts_nodes

    calls = []

    def op(x):
        calls.append(x)
        return jnp.ones_like(x) * (1 + 2j if complex_output else 1)

    f = Trigtech.from_function(op)
    assert jnp.array_equal(calls[0], jnp.asarray([PROBE]))
    assert jnp.array_equal(calls[1], jnp.concatenate((global_trigpts_nodes(16), jnp.ones(1))))
    assert f.ishappy and f.is_real is (not complex_output)
    assert jnp.max(jnp.abs(f.values - (1 + 2j if complex_output else 1))) <= 10 * EPS


def test_fixed_rejects_no_unrequested_point_evaluation():
    def grid_operator(x):
        if x.shape != (6,):
            raise RuntimeError('operator is defined on its complete fixed grid only')
        return jnp.cos(jnp.pi * x)

    f = Trigtech.from_function(grid_operator, n=5)
    assert f.n == 5 and f.ishappy
