"""Source PhaseOne rank-budget stopping controls on exact finite matrices.

Provenance
----------
MATLAB source : @spherefun/constructor.m (PhaseOne rankCount >= width override)
Chebfun commit: 7574c77
These are independent algorithm controls; original source bounds are unchanged.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import _phase_one_sphere


@pytest.mark.parametrize("factor,dual,expected_happy", [
    (8.0, False, False),  # exact rank1 reaches width1
    (4.0, False, True),   # exact rank1 remains below width2
    (8.0, True, False),   # rank2 block crosses width1
    (2.0, True, True),    # exact rank2 remains below width4
])
def test_exact_residual_does_not_override_rank_budget(factor, dual, expected_happy):
    plus = jnp.zeros((5, 4), dtype=jnp.float64).at[2, 1].set(1.0)
    minus = plus if dual else jnp.zeros_like(plus)
    samples = jnp.concatenate((plus-minus, plus+minus), axis=1)
    indices, pivots, removed_poles, happy = _phase_one_sphere(
        samples, 1e-14, 100.0, factor)
    assert not removed_poles
    assert indices.shape == (1, 2)
    assert int(jnp.count_nonzero(jnp.asarray(pivots))) == (2 if dual else 1)
    assert bool(happy) is expected_happy


def test_source_zero_early_return_stays_happy():
    _, _, removed_poles, happy = _phase_one_sphere(
        jnp.zeros((5, 8), dtype=jnp.float64), 1e-14, 100.0, 8.0)
    assert not removed_poles and happy
