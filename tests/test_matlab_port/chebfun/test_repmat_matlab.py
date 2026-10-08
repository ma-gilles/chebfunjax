"""Both active assertions in pinned tests/chebfun/test_repmat.m.

Chebfun7574c77680d7e82b79626300bf255498271a72df; MATLAB R2025b
seedRNG(7681) points captured from that source. Original 10*vscale*eps bound.
The source's third row assertion is commented out, so it is not a skipped test.
Row behavior is independently qualified in test_repmat_source.py.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize('args', [(1, 3), ([1, 3],)])
def test_source_repmat(args):
    fixture = json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())
    xr = jnp.asarray(fixture['xr'])
    f = chebfun(jnp.sin, domain=(-1, 0, 1))
    q = f.repmat(*args)
    exact = jnp.stack([jnp.sin(xr)]*3, axis=-1)
    assert float(jnp.max(jnp.abs(q(xr)-exact))) < 10*q.vscale*jnp.finfo(jnp.float64).eps
