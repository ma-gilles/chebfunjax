"""Original test_merge assertion12; exact captured MATLAB RNG points.

Chebfun 7574c77 tests/chebfun/test_merge.m. Coefficients are constructed in
Python; the fixture supplies only exact evaluation points, never fitted data.


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


def test_original_source_assertion12():
    fixture = json.loads((Path(__file__).with_name('data') / 'merge_assertion12_rng6178.json').read_text())
    x = jnp.asarray([float.fromhex(v) for v in fixture['x_hex']])
    assert x.shape == (100,)
    dom = (-2., 7.)
    def op(x):
        return (x-dom[0])**-1 * jnp.sin(10*x) * (x-dom[1])**-1
    f = cj.chebfun(op, domain=dom, exps=[-1, -1])
    g = f.addBreaksAtRoots()
    h = g.merge()
    error = jnp.max(jnp.abs(h(x)-op(x)))
    assert float(error) < 2e4*h.vscale*float(jnp.finfo(jnp.float64).eps)
