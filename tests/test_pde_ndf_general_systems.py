"""Real multicomponent PDE manufactured solutions and initializer scope.

MATLAB source: @chebfun/pdeSolve.m, @chebfun/pde15s.m; R2025b daeic12.m.
Chebfun commit: 7574c77. Existing two-component manufactured controls use
1e-7 at RelTol1e-9/AbsTol1e-11; these controls keep that policy.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d._pde_ndf.dae_init import initialize_full_mass
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


def test_three_component_coupled_neumann_manufactured_solution():
    initial = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0, 4.0]]))

    def pde(u, v, w):
        return (0.1 * u.diff(2) - u + v, 0.2 * v.diff(2) + u - 2 * v + w, 0.05 * w.diff(2) + v - w)

    def boundary(u, v, w):
        return (u.diff(), v.diff(), w.diff())

    times = [0.0, 0.005, 0.01]
    outputs = pde15s(pde, times, initial, lbc=boundary, rbc=boundary, n=9, rtol=1e-9, atol=1e-11)
    assert len(outputs) == len(times)
    for t, f in zip(times, outputs):
        fast, slow = jnp.exp(-3 * t), jnp.exp(-t)
        amplitudes = jnp.asarray(
            [7 / 3 - 1.5 * slow + fast / 6, 7 / 3 - fast / 3, 7 / 3 + 1.5 * slow + fast / 6]
        )
        actual = f(jnp.asarray([-0.75, 0.0, 0.625]))
        assert actual.shape == (3, 3)
        assert float(jnp.max(jnp.abs(actual - amplitudes))) < 1e-7


def test_five_component_independent_neumann_manufactured_solution():
    initial = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]]))

    def pde(u, v, w, z, q):
        return (
            0.1 * u.diff(2) - u,
            0.1 * v.diff(2) - 2 * v,
            0.1 * w.diff(2) - 3 * w,
            0.1 * z.diff(2) - 4 * z,
            0.1 * q.diff(2) - 5 * q,
        )

    def boundary(u, v, w, z, q):
        return (u.diff(), v.diff(), w.diff(), z.diff(), q.diff())

    times = [0.0, 0.005, 0.01]
    outputs = pde15s(pde, times, initial, lbc=boundary, rbc=boundary, n=9, rtol=1e-9, atol=1e-11)
    assert len(outputs) == len(times)
    k = jnp.arange(1, 6, dtype=jnp.float64)
    for t, f in zip(times, outputs):
        expected = k * jnp.exp(-k * t)
        actual = f(jnp.asarray([-0.75, 0.625]))
        assert actual.shape == (2, 5)
        assert float(jnp.max(jnp.abs(actual - expected))) < 1e-7


def test_six_algebraic_consistent_zero_state_uses_source_condition_check():
    mass = jnp.diag(jnp.asarray([0.0] * 6 + [1.0] * 6))
    result = initialize_full_mass(lambda y: y, mass, jnp.zeros(12))
    assert result["reason"] == "initial_consistent"
    assert result["algebraic"].size == 6
    np.testing.assert_array_equal(result["y"], jnp.zeros(12))
    np.testing.assert_array_equal(result["yp"], jnp.zeros(12))
