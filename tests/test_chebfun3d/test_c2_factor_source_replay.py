"""Native numeric-factor replays from the exposed Laplacian input.

Provenance
----------
MATLAB source : @chebfun3/chebfun3f.m, phase3
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d._chebfun3f_source import c2_factors

FIXTURE = json.loads(Path(__file__).with_name(
    'c2_factor_native_replay_20261015.json').read_text())


@pytest.mark.parametrize('axis', range(3))
def test_native_numeric_factor_replay(axis):
    case = FIXTURE['matrices'][axis]
    factors = c2_factors(jnp.asarray(case['values']),
                         FIXTURE['relative_tol'], FIXTURE['session_tol'])
    actual = jnp.stack([f.coeffs for f in factors], axis=1)
    expected = jnp.asarray(case['expected_coefficients'])
    assert actual.shape == expected.shape
    # Supplemental FFT/roundoff comparison; native output dimensions are exact.
    scale = jnp.max(jnp.abs(expected))
    assert jnp.max(jnp.abs(actual-expected)) < 100*jnp.finfo(jnp.float64).eps*scale
