"""Source23 oscillatory singular cube, qualified separately on CPU.

The sampling
expression and original bounds are copied from the pinned MATLAB test. The
existing NumPy RandomState/MT19937 adapter is retained. Its first100 sites
are checked bit for bit against a fresh pinned MATLAB ``seedRNG(6178)`` run.
This qualifies these sites, not the remaining random streams.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, pass 23
Chebfun commit: 7574c77
MATLAB APIs  : @chebfun/power.m, @singfun/power.m
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def test_source_pass23_split_oscillatory_singfun_cube():
    endpoint_power = -0.5
    integer_power = 3
    f = cj.chebfun(
        lambda x: jnp.sin(100.0 * x) * (x - DOMAIN[0]) ** endpoint_power,
        domain=DOMAIN,
        exps=(endpoint_power, 0.0),
        splitting=True,
    )
    result = f**integer_power
    x = jnp.asarray(_source_singular_samples())
    exact = jnp.sin(100.0 * x) ** integer_power * (
        x - DOMAIN[0]
    ) ** (integer_power * endpoint_power)
    assert _max_abs_error(result(x), exact) < 1e2 * EPS * _max_abs_exact(exact)



def test_source_pass23_adapter_sites_match_captured_matlab_binary64():
    fixture_path = Path(__file__).with_name("test_power_singular_sites_6178.json")
    fixture = json.loads(fixture_path.read_text())
    assert fixture["matlab_source_commit"] == "7574c77680d7e82b79626300bf255498271a72df"
    expected = np.asarray([int(value, 16) for value in fixture["hex_values"]], dtype=np.uint64)
    actual = _source_singular_samples()
    assert actual.shape == expected.shape == (100,)
    assert np.array_equal(actual.view(np.uint64), expected)
