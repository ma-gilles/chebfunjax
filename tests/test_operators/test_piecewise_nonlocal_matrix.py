import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize("two_delays, pieces", [(False, 6), (True, 11)])
def test_piecewise_nonlocal_matrix_source_dimensions(two_delays, pieces):
    # Native DDE example matrix(N,10): equation size10 plus one input
    # point per interval, with continuity constraints before the endpoint BC.
    p, q = 0.5, 1 / 3
    domain = (
        tuple(sorted(set([k * p for k in range(5)] + [k * q for k in range(7)])))
        if two_delays
        else tuple(k * p for k in range(7))
    )
    operator = (
        (lambda t, x: x.diff() - x((t - p).maximum(0.0)) - x((t - q).maximum(0.0)) - x)
        if two_delays
        else (lambda t, x: x.diff() - x((t - p).maximum(0.0)))
    )
    N = Chebop(operator, domain=domain)
    N.lbc = 1.0
    matrix = N.matrix(10)
    assert matrix.shape == (11 * pieces, 11 * pieces)
    assert bool(jnp.all(jnp.isfinite(matrix)))
    for k in range(pieces - 1):
        expected = jnp.zeros(11 * pieces).at[11 * (k + 1) - 1].set(1.0)
        expected = expected.at[11 * (k + 1)].set(-1.0)
        assert bool(jnp.array_equal(matrix[k], expected))
    expected_bc = jnp.zeros(11 * pieces).at[0].set(1.0)
    assert bool(jnp.array_equal(matrix[pieces - 1], expected_bc))
    # Added analytic control: apply the discretized operator to constant one.
    expected_equation = -3.0 if two_delays else -1.0
    assert (
        float(jnp.max(jnp.abs(matrix[pieces:] @ jnp.ones(11 * pieces) - expected_equation))) < 1e-10
    )
