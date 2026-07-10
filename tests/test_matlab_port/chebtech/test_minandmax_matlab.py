"""Port of MATLAB Chebfun tests/chebtech/test_minandmax.m (Opus 4.8).

Self-validating: the global min and max values and their locations are checked
against analytic exacts at the SAME tolerance MATLAB uses (10*vscale(f)*eps).
The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``.

``minandmax`` exists ONLY on Chebtech2 in chebfunjax (Chebtech1 lacks it), so
every method xfails the Chebtech1 parametrization with a precise reason.
MATLAB ``[y, x] = minandmax(f)`` returns ``y = [ymin; ymax]``; chebfunjax
``f.minandmax()`` returns ``((minval, minpos), (maxval, maxpos))``.

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 has no ``minandmax``.
- array-valued / complex-array-valued: chebfunjax Chebtech is scalar-valued.

Provenance
----------
MATLAB source : tests/chebtech/test_minandmax.m
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


def _spotcheck_minmax(fun, exact_min, exact_max):
    f = Chebtech2.from_function(fun)
    (mn, xmn), (mx, xmx) = f.minandmax()
    tol = 10 * f.vscale * EPS
    # Value errors and position errors (MATLAB checks both y and fun_op(x)).
    return (
        abs(float(mn) - exact_min),
        abs(float(mx) - exact_max),
        abs(_eval(fun, xmn) - exact_min),
        abs(_eval(fun, xmx) - exact_max),
        tol,
    )


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechMinAndMax:
    def _skip_c1(self, Tech):
        if Tech is Chebtech1:
            pytest.xfail("Chebtech1 lacks .minandmax (Chebtech2-only method)")

    def test_minmax_secant_cubic(self, Tech):
        # pass(n, 1)
        self._skip_c1(Tech)
        e = _spotcheck_minmax(
            lambda x: ((x - 0.2) ** 3 - (x - 0.2) + 1) * (1.0 / jnp.cos(x - 0.2)),
            0.710869767377087,
            1.884217141925336,
        )
        assert all(v < e[-1] for v in e[:-1])

    def test_minmax_sin10(self, Tech):
        # pass(n, 2)
        self._skip_c1(Tech)
        e = _spotcheck_minmax(lambda x: jnp.sin(10 * x), -1.0, 1.0)
        assert all(v < e[-1] for v in e[:-1])

    def test_minmax_airy(self, Tech):
        # pass(n, 3)
        self._skip_c1(Tech)
        e = _spotcheck_minmax(
            lambda x: sp.airy(np.asarray(x))[0],
            float(sp.airy(1.0)[0]),
            float(sp.airy(-1.0)[0]),
        )
        assert all(v < e[-1] for v in e[:-1])

    def test_minmax_neg_runge(self, Tech):
        # pass(n, 4)
        self._skip_c1(Tech)
        e = _spotcheck_minmax(lambda x: -1.0 / (1.0 + x**2), -1.0, -0.5)
        assert all(v < e[-1] for v in e[:-1])

    def test_minmax_cubic_cosh(self, Tech):
        # pass(n, 5)
        self._skip_c1(Tech)
        e = _spotcheck_minmax(
            lambda x: (x - 0.25) ** 3 * jnp.cosh(x),
            (-1.25) ** 3 * float(np.cosh(-1.0)),
            0.75**3 * float(np.cosh(1.0)),
        )
        assert all(v < e[-1] for v in e[:-1])

    def test_minmax_array_valued(self, Tech):
        # pass(n, 6): array-valued minandmax.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_minmax_complex_array_vals(self, Tech):
        # pass(n, 7): complex array-valued minandmax (|vals| comparison).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_minmax_complex_array_pos(self, Tech):
        # pass(n, 8): complex array-valued minandmax (position comparison).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )
