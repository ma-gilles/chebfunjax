"""Independent manufactured controls for source-shaped two-component systems."""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


def test_neumann_coupled_reaction():
    u0 = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0]]))

    def pde(t, x, u, v):
        return (-u + v + 0.1 * u.diff(2), u - v + 0.2 * v.diff(2))

    def boundary(u, v):
        return (u.diff(), v.diff())

    times = [0.0, 0.05, 0.1]
    outputs = pde15s(
        pde, times, u0, lbc=boundary, rbc=boundary, n=9, rtol=1e-9, atol=1e-11
    )
    assert len(outputs) == 3
    for t, u in zip(times, outputs):
        expected = jnp.asarray([1.5 - 0.5 * jnp.exp(-2 * t), 1.5 + 0.5 * jnp.exp(-2 * t)])
        assert float(jnp.max(jnp.abs(u(jnp.asarray([-0.7, 0.4])) - expected))) < 1e-7


def test_coupled_diffusive_eigenmode():
    u0 = chebfun(lambda x: jnp.column_stack((jnp.sin(jnp.pi * x), 2 * jnp.sin(jnp.pi * x))))

    def pde(t, x, u, v):
        return (0.1 * u.diff(2) - u + v, 0.1 * v.diff(2) + u - v)

    def boundary(u, v):
        return (u, v)

    times = [0.0, 0.05, 0.1]
    outputs = pde15s(
        pde, times, u0, lbc=boundary, rbc=boundary, n=17, rtol=1e-9, atol=1e-11
    )
    x = jnp.asarray([-0.7, 0.4])
    for t, u in zip(times, outputs):
        amplitudes = jnp.asarray([1.5 - 0.5 * jnp.exp(-2 * t), 1.5 + 0.5 * jnp.exp(-2 * t)])
        expected = jnp.sin(jnp.pi * x)[:, None] * jnp.exp(-0.1 * jnp.pi**2 * t) * amplitudes
        assert float(jnp.max(jnp.abs(u(x) - expected))) < 1e-7
