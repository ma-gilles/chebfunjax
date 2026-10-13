"""Exact native affine probes and explicit-state uniform stream controls.

Provenance
----------
MATLAB source: tests/chebfun/test_rdivide.m clauses16–17.
Chebfun commit: 7574c77.
Fixture records native release, source formulas, MAT hash and RNG assumptions.
"""
import json
from pathlib import Path

import jax.numpy as jnp

from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform


def test_native_raw_uniforms_and_affine_probe_words():
    fixture = json.loads((Path(__file__).parent/'fixtures/rdivide_uniform_probes_native.json').read_text())
    u16, state = matlab_rand_uniform(init_matlab_twister(6178), (100,))
    u17, state = matlab_rand_uniform(state, (100,))
    for name, actual in [('u16',u16),('u17',u17),('x16',9*u16-2),('x17',98*u17+2)]:
        expected = jnp.asarray(fixture[name],dtype=jnp.float64)
        assert jnp.array_equal(actual.view(jnp.uint64),expected.view(jnp.uint64))
