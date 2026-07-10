"""Port of MATLAB Chebfun tests/chebtech/test_min.m (Opus 4.8).

Self-validating: each minimum value and its location are checked against an
analytic exact at the SAME tolerance MATLAB uses (1e2 * vscale(f) * eps).
The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``.

``min`` exists ONLY on Chebtech2 in chebfunjax (Chebtech1 lacks it), so every
method xfails the Chebtech1 parametrization with a precise reason.  MATLAB
``[y, x] = min(f)`` maps to ``y, x = f.min()``.

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 has no ``min``.
- complex-valued ``min``: chebfunjax uses real-valued rootfinding/argmin, not
  MATLAB's complex ordering.
- array-valued ``min``: chebfunjax Chebtech is scalar-valued.

Provenance
----------
MATLAB source : tests/chebtech/test_min.m
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


def _spotcheck_min(fun, exact):
    f = Chebtech2.from_function(fun)
    y, xpos = f.min()
    tol = 1e2 * f.vscale * EPS
    return abs(float(y) - exact), abs(_eval(fun, xpos) - exact), tol


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechMin:
    def _skip_c1(self, Tech):
        if Tech is Chebtech1:
            pytest.xfail("Chebtech1 lacks .min (Chebtech2-only method)")

    def test_min_secant_cubic(self, Tech):
        # pass(n, 1)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_min(
            lambda x: -(((x - 0.2) ** 3 - (x - 0.2) + 1) * (1.0 / jnp.cos(x - 0.2))),
            -1.884217141925336,
        )
        assert ey < tol and efx < tol

    def test_min_neg_sin10(self, Tech):
        # pass(n, 2)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_min(lambda x: -jnp.sin(10 * x), -1.0)
        assert ey < tol and efx < tol

    def test_min_neg_airy(self, Tech):
        # pass(n, 3)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_min(
            lambda x: -sp.airy(np.asarray(x))[0], -float(sp.airy(-1.0)[0])
        )
        assert ey < tol and efx < tol

    def test_min_runge(self, Tech):
        # pass(n, 4)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_min(lambda x: 1.0 / (1.0 + x**2), 0.5)
        assert ey < tol and efx < tol

    def test_min_neg_cubic_cosh(self, Tech):
        # pass(n, 5)
        self._skip_c1(Tech)
        ey, efx, tol = _spotcheck_min(
            lambda x: -((x - 0.25) ** 3) * jnp.cosh(x),
            -(0.75**3) * float(np.cosh(1.0)),
        )
        assert ey < tol and efx < tol

    def test_min_array_valued(self, Tech):
        # pass(n, 6): array-valued min.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_min_complex(self, Tech):
        # pass(n, 7): min of exp(1i x) - .5i sin(x) + x.
        self._skip_c1(Tech)
        pytest.xfail(
            "chebfunjax min on complex-valued functions uses real-valued "
            "rootfinding/argmin, not MATLAB's complex ordering"
        )

    def test_min_complex_array(self, Tech):
        # pass(n, 8): complex array-valued min.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )
