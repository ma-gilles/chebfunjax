"""Independent equation check, supplementary to the unchanged source assertion.

Provenance
----------
MATLAB source : tests/chebop/test_jump_scaled.m, @chebfun/jump.m
Chebfun commit: 7574c77

The fixed absolute 1e-10 target is retained from the source interface test,
now also required of the independent analytic solution and all constraints.
No saved oracle solution is a constructor input; no relative tolerance.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import jump
from chebfunjax.operators.chebop import Chebop


def test_scaled_jump_all_constraints_and_analytic_solution():
    operator = Chebop(lambda x, u: u.diff(2) - u + x, (-1.0, 1.0))
    operator.lbc = 0.2
    operator.rbc = 0.0
    operator.bc = lambda x, u: [u(0.1, 'left') - 4 * u(0.1, 'right') - 2.2,
                               jump(u.diff(), 0.1, 1)]
    u = operator.solve(0.0)
    t = jnp.asarray(0.1)
    e, q = jnp.exp(t), jnp.exp(-t)
    matrix = jnp.asarray([[jnp.exp(-1.0), jnp.exp(1.0), 0, 0],
                          [0, 0, jnp.exp(1.0), jnp.exp(-1.0)],
                          [e, q, -4 * e, -4 * q], [-e, q, e, -q]])
    constants = jnp.linalg.solve(matrix, jnp.asarray([1.2, -1.0, 2.5, 1.0]))
    x = jnp.asarray([-0.9, -0.5, 0.0, 0.2, 0.5, 0.9])
    aa = jnp.where(x < t, constants[0], constants[2])
    bb = jnp.where(x < t, constants[1], constants[3])
    expected = x + aa * jnp.exp(x) + bb * jnp.exp(-x)
    np.testing.assert_allclose(u(x), expected, atol=1e-10, rtol=0)
    residuals = jnp.asarray([u(-1.0) - 0.2, u(1.0),
                             u(t, 'left') - 4 * u(t, 'right') - 2.2,
                             jump(u.diff(), t, 1)])
    assert bool(jnp.all(jnp.isfinite(residuals)))
    assert bool(jnp.max(jnp.abs(residuals)) < 1e-10)
