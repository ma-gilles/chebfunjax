"""Original seven default predicates of MATLAB tests/fun/test_detectEdge.m.

The ten seedRNG(13) inputs are captured component words from the pinned
MATLAB test. Preserve its full trial count, operators and error bounds.
The old loose shortcut did not exercise the second predicate. Optional nondefault preference runs are not qualified here.

Provenance
----------
MATLAB source : tests/fun/test_detectEdge.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import _detect_edge_matlab

_NATIVE = json.loads(
    (Path(__file__).parents[2] / 'fixtures/fun_detect_edge_native_20261018.json').read_text()
)
assert _NATIVE['pin'] == '7574c77680d7e82b79626300bf255498271a72df'
assert _NATIVE['pass'] == [True] * 7
_LOCATIONS = tuple(struct.unpack('>d', bytes.fromhex(w))[0]
                   for w in _NATIVE['location_words'])


def _edge(op, domain=(0., 1.), *, blowup=False):
    raw = _detect_edge_matlab(op, *domain, vscale=1., hscale=1., blowup=blowup)
    # Native fun.detectEdge replaces an empty raw result with the midpoint.
    return (domain[0] + domain[1]) / 2 if raw is None else raw


@pytest.mark.parametrize('slot', range(1, 8))
def test_original_predicate(slot):
    if slot <= 5:
        edges = []
        for x0 in _LOCATIONS:
            def op(x):
                z = x - x0
                if slot == 1:
                    return jnp.exp(x) + jnp.cos(7*x) + .1*jnp.sign(z)
                if slot == 2:
                    return jnp.exp(x) + jnp.cos(7*x) + .1*jnp.abs(z)
                if slot == 3:
                    return jnp.sign(z)*z
                if slot == 4:
                    return jnp.abs(z)**3
                return jnp.sign(z)*z**3
            edges.append(_edge(op))
        err = jnp.linalg.norm(jnp.asarray(edges) - jnp.asarray(_LOCATIONS), ord=jnp.inf)
        assert err < 5e-14
    elif slot == 6:
        x0 = struct.unpack('>d', bytes.fromhex('3fe55ec001aabd80'))[0]
        edge = _edge(lambda x: jnp.exp(x) + jnp.cos(7*x) + .1*jnp.sign(x-x0))
        assert abs(edge-x0) < 5*jnp.finfo(jnp.float64).eps
    else:
        edge = _edge(jnp.tan, (1., 2.), blowup=True)
        assert abs(edge-jnp.pi/2) < 5*jnp.finfo(jnp.float64).eps
