"""Port of MATLAB Chebfun tests/chebtech/test_max.m (Opus 4.8).

Self-validating: each maximum value and its location are checked against an
analytic exact at the SAME tolerance MATLAB uses (10 * vscale(f) * eps).
The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``.

``max`` exists ONLY on Chebtech2 in chebfunjax (Chebtech1 lacks it), so every
method xfails the Chebtech1 parametrization with a precise reason.  MATLAB
``[y, x] = max(f)`` maps to ``y, x = f.max()``.

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 has no ``max``.
- complex-valued ``max``: chebfunjax uses real-valued rootfinding/argmin, not
  MATLAB's complex ordering.
- array-valued ``max``: chebfunjax Chebtech is scalar-valued.

Provenance
----------
MATLAB source : tests/chebtech/test_max.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _eval(fun, xpos):
    """Evaluate the (possibly scipy-backed) test function at a scalar."""
    return float(np.asarray(fun(jnp.asarray([float(xpos)])))[0])


def _spotcheck_max(fun, exact):
    f = Chebtech2.from_function(fun)
    y, xpos = f.max()
    tol = 10 * f.vscale * EPS
    return abs(float(y) - exact), abs(_eval(fun, xpos) - exact), tol


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechMax:
    def _skip_c1(self, Tech):
        if Tech is Chebtech1:
            pytest.xfail("Chebtech1 lacks .max (Chebtech2-only method)")

    def test_max_secant_cubic(self, Tech):
        # pass(n, 1)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_max(
            lambda x: ((x - 0.2) ** 3 - (x - 0.2) + 1) * (1.0 / jnp.cos(x - 0.2)),
            1.884217141925336,
        )
        assert ey < tol and efx < tol

    def test_max_sin10(self, Tech):
        # pass(n, 2)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_max(lambda x: jnp.sin(10 * x), 1.0)
        assert ey < tol and efx < tol

    def test_max_airy(self, Tech):
        # pass(n, 3)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_max(
            lambda x: sp.airy(np.asarray(x))[0], float(sp.airy(-1.0)[0])
        )
        assert ey < tol and efx < tol

    def test_max_neg_runge(self, Tech):
        # pass(n, 4)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_max(lambda x: -1.0 / (1.0 + x**2), -0.5)
        assert ey < tol and efx < tol

    def test_max_cubic_cosh(self, Tech):
        # pass(n, 5)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_max(
            lambda x: (x - 0.25) ** 3 * jnp.cosh(x),
            0.75**3 * float(np.cosh(1.0)),
        )
        assert ey < tol and efx < tol

    def test_max_array_valued(self, Tech):
        # pass(n, 6): array-valued max.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_max_complex(self, Tech):
        # pass(n, 7): max of (x-0.2)*(exp(1i(x-0.2)) + 1i sin(x-0.2)).
        self._skip_c1(Tech)
        pytest.xfail(
            "chebfunjax max on complex-valued functions uses real-valued "
            "rootfinding/argmin, not MATLAB's complex ordering"
        )

    def test_max_complex_array(self, Tech):
        # pass(n, 8): complex array-valued max.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )
