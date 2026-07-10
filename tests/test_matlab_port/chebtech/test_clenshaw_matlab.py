"""Port of MATLAB Chebfun tests/chebtech/test_clenshaw.m (Opus 4.8).

Tests the Clenshaw evaluator directly.  MATLAB's static
``chebtech.clenshaw(x, c)`` maps to chebfunjax ``_clenshaw(c, x)`` (arguments
are REVERSED).  This MATLAB test is NOT looped over the tech classes.

Notes on gaps (see the report):
* The array-of-columns coefficient cases (pass 3 and pass 5, where ``c`` is a
  matrix) require array-valued coefficients; chebfunjax ``_clenshaw`` takes a
  1-D coefficient vector only, so those are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_clenshaw.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _clenshaw

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechClenshaw:
    def test_single_coefficient_scalar(self):
        # pass(1): chebtech.clenshaw(0, sqrt(2)) == sqrt(2)
        c = jnp.array([np.sqrt(2.0)])
        v = _clenshaw(c, jnp.asarray(0.0))
        assert float(v) == np.sqrt(2.0)

    def test_single_coefficient_vector(self):
        # pass(2): clenshaw([-0.5; 1], sqrt(2)) is [sqrt(2); sqrt(2)]
        c = jnp.array([np.sqrt(2.0)])
        v = _clenshaw(c, jnp.array([-0.5, 1.0]))
        assert v.shape == (2,)
        assert bool(np.all(np.asarray(v) == np.sqrt(2.0)))

    def test_row_vector_column_coeffs_skipped(self):
        # pass(3): c = [c, c, c] (matrix of columns) -> array-valued coeffs.
        pytest.skip(
            "chebfunjax _clenshaw takes a 1-D coefficient vector; no "
            "array-of-columns (matrix) coefficients"
        )

    def test_vector_coefficient(self):
        # pass(4): c = (5:-1:1).' , x = [-0.5; -0.1; 1]
        c = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        x = jnp.array([-0.5, -0.1, 1.0])
        v = _clenshaw(c, x)
        vTrue = np.array([3.0, 3.1728, 15.0])  # exact for these x
        assert _ninf(np.asarray(v) - vTrue) < TOL

    def test_array_coefficient_skipped(self):
        # pass(5): clenshaw(x, [c, c(end:-1:1)]) -> array-valued coeffs.
        pytest.skip(
            "chebfunjax _clenshaw takes a 1-D coefficient vector; no "
            "array-of-columns (matrix) coefficients"
        )
