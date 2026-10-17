"""Source full-mass controls for empty and absent system boundaries."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._pde_ndf import pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


@pytest.mark.parametrize("empty_handle", [False, True])
def test_zero_order_system_keeps_full_mass_dae_route(monkeypatch, empty_handle):
    # Native pdeSolve774 always sets MassSingular=yes; ode15s222-227 honors it
    # even for full-rank M. This is not the scalar no-BC compatibility route.
    initial = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0, 3.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]))
    original_init, original_start = pde_driver.initialize_full_mass, pde_driver.startup
    observed = []

    def initialize(fun, mass, y, **kwargs):
        assert mass.shape == (27, 27)
        assert not bool(jnp.array_equal(mass, jnp.eye(27)))
        result = original_init(fun, mass, y, **kwargs)
        assert result["reason"] == "nonsingular" and result["algebraic"].size == 0
        observed.append("full_mass_init")
        return result

    def startup(*args, **kwargs):
        assert kwargs["dae"] is True
        observed.append("dae_startup")
        return original_start(*args, **kwargs)

    def rhs(u, v, w):
        return (-u, -2 * v, -3 * w)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", initialize)
    monkeypatch.setattr(pde_driver, "startup", startup)
    times = [0.0, 0.005, 0.01]
    boundary = (lambda u, v, w: ()) if empty_handle else None
    outputs = pde15s(rhs, times, initial, lbc=boundary, rbc=boundary, n=9, rtol=1e-9, atol=1e-11)
    assert observed == ["full_mass_init", "dae_startup"]
    assert len(outputs) == 3
    x = jnp.asarray([-0.75, 0.0, 0.625])
    for t, solution in zip(times, outputs):
        expected = jnp.column_stack(
            (
                jnp.exp(-t) * jnp.ones_like(x),
                2 * jnp.exp(-2 * t) * jnp.ones_like(x),
                (3 + x * x) * jnp.exp(-3 * t),
            )
        )
        assert float(jnp.max(jnp.abs(solution(x) - expected))) < 1e-7


def test_first_order_system_with_absent_right_boundary(monkeypatch):
    initial = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0, 3.5], [1.0, 2.0, 0.0], [0.0, 0.0, 0.5]]))
    original = pde_driver.initialize_full_mass
    observed = []

    def initialize(fun, mass, y, **kwargs):
        assert mass.shape == (27, 27) and bool(jnp.all(mass[:2] == 0))
        result = original(fun, mass, y, **kwargs)
        assert result["algebraic"].size == 2
        observed.append(True)
        return result

    def rhs(u, v, w):
        return (-u.diff(), -v.diff(), -w)

    def left(t, x, u, v, w):
        # Left inflow for u_t+u_x=0 and v_t+v_x=0.
        return (u + t, v + 2 * t)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", initialize)
    times = [0.0, 0.005, 0.01]
    outputs = pde15s(rhs, times, initial, lbc=left, rbc=None, n=9, rtol=1e-9, atol=1e-11)
    assert observed == [True] and len(outputs) == 3
    x = jnp.asarray([-0.75, 0.0, 0.625])
    for t, solution in zip(times, outputs):
        expected = jnp.column_stack((1 + x - t, 2 + 2 * x - 2 * t, (3 + x * x) * jnp.exp(-t)))
        assert float(jnp.max(jnp.abs(solution(x) - expected))) < 1e-7
