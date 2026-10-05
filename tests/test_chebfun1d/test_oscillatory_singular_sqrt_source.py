"""Source24 oscillatory singular square root, isolated CPU gate.

The sampling
expression and original bounds are copied from the pinned MATLAB test. NumPy
RandomState/MT19937 seed 6178 is only a reproducible Python adapter; it is not
claimed to match MATLAB's ``seedRNG`` stream.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, pass 24
Chebfun commit: 7574c77
MATLAB APIs  : @chebfun/power.m, @singfun/power.m
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import json
import time

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
DOMAIN = (-2.0, 7.0)


def _source_singular_samples(seed=6178):
    # Literal local test_power.m: domCheck = [dom(1)+.1, dom(2)-.1],
    # x = diff(domCheck)*rand(100,1)+domCheck(1).
    dom_check = (DOMAIN[0] + 0.1, DOMAIN[1] - 0.1)
    rng = np.random.RandomState(seed)
    return (dom_check[1] - dom_check[0]) * rng.rand(100) + dom_check[0]


def _max_abs_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _max_abs_exact(expected):
    return float(np.max(np.abs(np.asarray(expected))))




def test_source_pass24_positive_oscillatory_singfun_square_root():
    endpoint_power = -1.5
    started = time.perf_counter()
    f = cj.chebfun(
        lambda x: (jnp.sin(100.0 * x) ** 2 + 1.0)
        * (x - DOMAIN[0]) ** endpoint_power,
        domain=DOMAIN,
        exps=(endpoint_power, 0.0),
        splitting=True,
    )
    print(json.dumps({"phase": "constructed", "seconds": time.perf_counter()-started,
                      "pieces": len(f.funs),
                      "max_smooth_length": max(piece.tech.n for piece in f.funs),
                      "all_happy": all(piece.ishappy for piece in f.funs)}), flush=True)
    started_power = time.perf_counter()
    result = f**0.5
    print(json.dumps({"phase": "powered", "seconds": time.perf_counter()-started_power,
                      "pieces": len(result.funs),
                      "max_smooth_length": max(piece.tech.n for piece in result.funs),
                      "all_happy": all(piece.ishappy for piece in result.funs)}), flush=True)
    x = jnp.asarray(_source_singular_samples())
    exact = jnp.sqrt(jnp.sin(100.0 * x) ** 2 + 1.0) * (
        x - DOMAIN[0]
    ) ** (endpoint_power / 2.0)
    assert _max_abs_error(result(x), exact) < 1e2 * EPS * _max_abs_exact(exact)
