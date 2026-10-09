"""Literal nine slots from tests/chebfun/test_horzcat.m, pin7574c77.

RNG primitive fixture is the prior native seed7681 capture, unchanged.
Numel translates source storage count, not function column count.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.tech.trigtech import Trigtech


@pytest.fixture(scope='module')
def source_inputs():
    fixture = Path(__file__).parent/'test_matlab_port/chebfun/repmat_matlab_inputs.json'
    xr = jnp.asarray(json.loads(fixture.read_text())['xr'])
    assert xr.shape == (1000,)
    f = chebfun(jnp.sin, domain=(-1., 0., 1.))
    g = chebfun(jnp.cos, domain=(-1., .5, 1.))
    h = chebfun(jnp.exp, domain=(-1., -.5, 0., .5, 1.))
    periodic = chebfun(lambda x: jnp.cos(jnp.pi*x), trig=True)
    mixed = Chebfun.horzcat([periodic, chebfun('x')])
    return xr, f, g, h, mixed


def numel(q):
    return len(q.cols) if isinstance(q, Quasimatrix) else 1


@pytest.mark.parametrize('slot', range(1, 10))
def test_native_slot(slot, source_inputs):
    xr, f, g, h, mixed = source_inputs
    if slot <= 5:
        if slot == 1:
            q = Chebfun.horzcat([1., f])
            exact = jnp.stack([jnp.ones_like(xr), jnp.sin(xr)], axis=1)
            storage_ok = numel(f) == 1  # Literal source quirk, not numel(q).
        elif slot == 2:
            q = Chebfun.horzcat([f, f])
            exact = jnp.stack([jnp.sin(xr), jnp.sin(xr)], axis=1)
            storage_ok = numel(f) == 1  # Literal source quirk.
        elif slot == 3:
            q = Chebfun.horzcat([f, g])
            exact = jnp.stack([jnp.sin(xr), jnp.cos(xr)], axis=1)
            storage_ok = numel(q) == 2
        elif slot == 4:
            q = Chebfun.horzcat([f, g, h])
            exact = jnp.stack([jnp.sin(xr), jnp.cos(xr), jnp.exp(xr)], axis=1)
            storage_ok = numel(q) == 3
        else:
            q = Chebfun.horzcat([Chebfun.horzcat([f, f]), f])
            exact = jnp.stack([jnp.sin(xr)]*3, axis=1)
            storage_ok = numel(q) == 1
        scale = max(c.vscale for c in q.cols) if isinstance(q, Quasimatrix) else q.vscale
        error = jnp.max(jnp.abs(q(xr)-exact))
        assert storage_ok and float(error) < 10*scale*jnp.finfo(jnp.float64).eps
        if slot <= 2:
            # Independent structural companion, beyond literal source slot.
            assert isinstance(q, Chebfun) and q.n_columns == 2
    elif slot == 6:
        assert numel(mixed) == 2 and numel(Chebfun.horzcat([mixed, mixed])) == 4
    elif slot == 7:
        assert all(isinstance(p.tech, Trigtech) for p in mixed.cols[0].funs)
    elif slot == 8:
        assert not all(isinstance(p.tech, Trigtech) for p in mixed.cols[1].funs)
    else:
        with pytest.warns(UserWarning, match='vertcat:join'):
            q = Chebfun.horzcat([chebfun('x').T, [0., 0.]])
        assert isinstance(q, ChebMatrix) and (q.nrows, q.ncols) == (1, 3)
