"""Actual test_merge assertion13, pinned Chebfun 7574c77, original bound.

Provenance
----------
MATLAB source : tests/chebfun/test_merge.m, @fun/merge.m,
    @singfun/singfun.m, @unbndfun/unbndfun.m
Chebfun commit: 7574c77
"""
import json
from pathlib import Path

import jax.numpy as jnp

import chebfunjax as cj


def test_original_source_assertion13():
    fixture = json.loads((Path(__file__).with_name('data') / 'merge_assertion13_rng6178.json').read_text())
    x = jnp.asarray([float.fromhex(v) for v in fixture['x_hex']])
    assert x.shape == (100,)
    domain = tuple(float(x) for x in range(0, 101, 10)) + (float('inf'),)
    def op(x):
        return .75 + jnp.sin(10*x)/jnp.exp(x)
    f = cj.chebfun(op, domain=domain, splitting=True)
    g = f.merge()
    error = jnp.max(jnp.abs(g(x)-op(x)))
    assert float(error) < 1e3*float(jnp.finfo(jnp.float64).eps)*f.vscale
