"""Literal CV weight branches from native smoothfun.m, Chebfun 7574c77.

Expected weights are raw native case01 trace values, independently captured at
funqui_equi_native_trace_20261014/native_trace.mat SHA256
47dca77d6e69232fa6f9821a99268d0c45d0ad72f24616ce69146e4c8d35a55d.
"""
import jax.numpy as jnp

import chebfunjax.utils.interpolation as interpolation


def test_native_cv_signed_weight_branches(monkeypatch):
    original = interpolation.bary
    weights = []

    def record(x, values, nodes, w):
        weights.append(w)
        return original(x, values, nodes, w)

    monkeypatch.setattr(interpolation, 'bary', record)
    values = jnp.asarray([[11., -21.], [5.6875, -10.375], [2.5, -4.],
                          [1.0625, -1.125], [1., -1.], [1.9375, -2.875],
                          [3.5, -6.]])
    interpolation.funqui(values)
    assert jnp.array_equal(weights[0], jnp.asarray([-1., 1., -1., 1., -1.]))
    expected = jnp.asarray([-1.5, 4.5, -4.5, 4.5, -3.])
    assert jnp.max(jnp.abs(weights[1] - expected)) < 10 * jnp.finfo(jnp.float64).eps * jnp.max(jnp.abs(expected))
