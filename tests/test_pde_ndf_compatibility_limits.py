"""Concrete preexisting input behavior and source termination controls."""

import jax.numpy as jnp

from chebfunjax.chebfun1d._pde_ndf.ndf import startup
from chebfunjax.chebfun1d._pde_ndf.ndf_segment import segment
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


def test_scalar_rhs_is_broadcast():
    u0 = chebfun(lambda x: 1 + x)
    out = pde15s(lambda u: 2.0, [0.0, 0.1], u0, n=9)
    x = jnp.array([-0.8, 0.3])
    assert float(jnp.max(jnp.abs(out[-1](x) - (1 + x + 0.2)))) < 1e-8


def test_smooth_piecewise_initial_data_merges():
    u0 = chebfun(lambda x: x * x, domain=[-1.0, 0.0, 1.0])
    assert len(u0.funs) == 2
    out = pde15s(lambda u: 0 * u, [0.0, 0.1], u0, n=9)
    assert all(len(u.funs) == 1 for u in out)
    x = jnp.array([-0.8, 0.3])
    assert float(jnp.max(jnp.abs(out[-1](x) - x * x))) < 1e-8


def test_trig_initial_representation_uses_polynomial_collocation():
    u0 = chebfun(lambda x: jnp.cos(jnp.pi * x), trig=True)
    out = pde15s(lambda u: 0 * u, [0.0, 0.1], u0, n=33)
    x = jnp.array([-0.8, 0.3])
    assert len(out) == 2
    assert float(jnp.max(jnp.abs(out[-1](x) - jnp.cos(jnp.pi * x)))) < 1e-8


def test_endpoint_termination_after_more_than_1000_attempts():
    y = jnp.array([1.0])

    def fun(t, y):
        return jnp.zeros_like(y)

    state = startup(
        y,
        jnp.zeros(1),
        jnp.zeros((1, 1)),
        jnp.eye(1),
        rtol=1e-6,
        threshold=1.0,
        htspan=0.001,
        userhmax=0.001,
        dae=False,
        rhs=fun,
    )
    state.update(tfinal=jnp.asarray(1.01), jac_threshold=jnp.asarray(1e-6))
    outputs = []
    final, rows = segment(
        state,
        fun,
        None,
        jnp.array([1.01]),
        lambda t, y: outputs.append(y) or False,
        record_trace=False,
    )
    assert final["attempt_count"] > 1000
    assert rows == []
    assert float(final["t"]) == 1.01
    assert len(outputs) == 1
    assert bool(jnp.array_equal(outputs[0], y))


def test_coordinate_proxy_differentiation_is_preserved():
    u0 = chebfun(lambda x: 0 * x)
    out = pde15s(lambda t, x, u: x.diff() + 0 * u, [0.0, 0.1], u0, n=9)
    assert float(jnp.max(jnp.abs(out[-1](jnp.array([-0.8, 0.3])) - 0.1))) < 1e-8
