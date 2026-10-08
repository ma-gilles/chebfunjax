"""All seven original plain Jacobi-function clauses, unchanged source bounds.

Provenance
----------
MATLAB source : tests/chebfun/test_ellipj.m, @chebfun/ellipj.m
Chebfun commit: 7574c77
Source seed6178 sites come from the native logical-test fixture. SciPy is an
independent numeric oracle for default tolerance. The explicit-tolerance case
also checks the public numeric dispatch, as the original calls the builtin.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ellipj as scipy_ellipj

import chebfunjax as cj

DOM = (-1., -.5, 0., .5, 1.)
EPS = np.finfo(np.float64).eps


@pytest.fixture(scope='module')
def sites():
    return jnp.asarray(json.loads(Path(__file__).with_name('fixtures').joinpath(
        'logical_source_matlab.json').read_text())['x']).reshape(-1)


def _u_array(x):
    return jnp.stack((jnp.exp(x), jnp.sin(jnp.pi*x)), axis=-1)


def _m_array(x):
    return .05+jnp.abs(.9*jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x)), axis=-1))


@pytest.mark.parametrize('clause', range(1, 7))
def test_original_default_cases(sites, clause):
    cases = {
        1: (jnp.exp, .6),
        2: (_u_array, .6),
        3: (.3, lambda x: .99*jnp.abs(x)),
        4: (.3, _m_array),
        5: (jnp.exp, lambda x: .99*jnp.abs(x)),
        6: (_u_array, _m_array),
    }
    u_op, m_op = cases[clause]
    u = cj.chebfun(u_op, domain=DOM) if callable(u_op) else u_op
    m = cj.chebfun(m_op, domain=DOM) if callable(m_op) else m_op
    expected = scipy_ellipj(np.asarray(u_op(sites) if callable(u_op) else u_op),
                           np.asarray(m_op(sites) if callable(m_op) else m_op))[:3]
    for actual, exact in zip(cj.ellipj(u, m), expected):
        error = np.max(np.abs(np.asarray(actual(sites))-exact))
        assert error < 100*actual.vscale*EPS


def test_original_explicit_tolerance_case7(sites):
    u = cj.chebfun(lambda x: x, domain=DOM)
    actual = cj.ellipj(u, .9, 1e-2)
    expected = cj.ellipj(sites, .9, 1e-2)
    for value, exact in zip(actual, expected):
        error = float(jnp.max(jnp.abs(value(sites)-exact)))
        assert error < 100*value.vscale*EPS
