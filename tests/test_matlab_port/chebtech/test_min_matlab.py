"""Port of MATLAB Chebfun tests/chebtech/test_min.m (Opus 4.8).

Self-validating: each minimum value and its location are checked against an
analytic exact at the SAME tolerance MATLAB uses (1e2 * vscale(f) * eps).
The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``.

``min`` now exists on BOTH tech classes; every method is exercised on
Chebtech1 and Chebtech2.  MATLAB ``[y, x] = min(f)`` maps to ``y, x = f.min()``.

Gaps vs MATLAB (honest xfail/skip):
- complex-valued / complex-array-valued ``min``: FIXED (Fable 5) --
  minandmax follows MATLAB's complex path (extrema of |f|^2), so the
  complex spot-checks port at the same tolerances.

Real array-valued ``min`` (pass(n, 6)) is now supported: techs carry (n, m)
coeffs and ``min`` returns per-column ``(m,)`` values/positions.

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


def airy(x):
    """Airy Ai at ``x`` (MATLAB ``airy(x)`` -> ``Ai``), as a NumPy array."""
    return sp.airy(np.asarray(x))[0]


def _eval(fun, xpos):
    """Evaluate the (possibly scipy-backed) test function at a scalar."""
    return float(np.asarray(fun(jnp.asarray([float(xpos)])))[0])


def _spotcheck_min(Tech, fun, exact):
    f = Tech.from_function(fun)
    y, xpos = f.min()
    tol = 1e2 * f.vscale * EPS
    return abs(float(y) - exact), abs(_eval(fun, xpos) - exact), tol


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechMin:


    def test_min_secant_cubic(self, Tech):
        # pass(n, 1)
        ey, efx, tol = _spotcheck_min(
            Tech,
            lambda x: -(((x - 0.2) ** 3 - (x - 0.2) + 1) * (1.0 / jnp.cos(x - 0.2))),
            -1.884217141925336,
        )
        assert ey < tol and efx < tol

    def test_min_neg_sin10(self, Tech):
        # pass(n, 2)
        ey, efx, tol = _spotcheck_min(Tech, lambda x: -jnp.sin(10 * x), -1.0)
        assert ey < tol and efx < tol

    def test_min_neg_airy(self, Tech):
        # pass(n, 3)
        ey, efx, tol = _spotcheck_min(
            Tech,
            lambda x: -sp.airy(np.asarray(x))[0], -float(sp.airy(-1.0)[0])
        )
        assert ey < tol and efx < tol

    def test_min_runge(self, Tech):
        # pass(n, 4)
        ey, efx, tol = _spotcheck_min(Tech, lambda x: 1.0 / (1.0 + x**2), 0.5)
        assert ey < tol and efx < tol

    def test_min_neg_cubic_cosh(self, Tech):
        # pass(n, 5)
        ey, efx, tol = _spotcheck_min(
            Tech,
            lambda x: -((x - 0.25) ** 3) * jnp.cosh(x),
            -(0.75**3) * float(np.cosh(1.0)),
        )
        assert ey < tol and efx < tol

    def test_min_array_valued(self, Tech):
        # pass(n, 6): array-valued min.
        # FIXED (Fable 5, Big-Three array-valued epic): techs now carry (n, m)
        # coeffs and min returns per-column (m,) values/positions.
        fun = lambda x: -jnp.stack(
            [jnp.sin(10 * x), airy(x), (x - 0.25) ** 3 * jnp.cosh(x)], axis=-1
        )
        f = Tech.from_function(fun)
        y, xpos = f.min()
        exact = -jnp.array(
            [1.0, float(sp.airy(-1.0)[0]), 0.75**3 * float(np.cosh(1.0))],
            dtype=jnp.float64,
        )
        # MATLAB pass(n, 6) tol: 10*max(vscale(f)*eps).
        tol = 10 * f.vscale * EPS
        assert float(jnp.max(jnp.abs(y - exact))) < tol
        assert float(jnp.max(jnp.abs(jnp.diagonal(fun(xpos)) - exact))) < tol

    # FIXED (Fable 5, Big-Three array-valued epic): min now follows
    # MATLAB's complex path (|f|^2 extrema), so pass(n, 7)-(8) port.
    def test_min_complex(self, Tech):
        # pass(n, 7): min of exp(1i x) - .5i sin(x) + x.
        import warnings

        def fun(x):
            return jnp.exp(1j * x) - 0.5j * jnp.sin(x) + x

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(fun)
        y, x = f.min()
        exact = 0.074968381369117 - 0.319744137826069j
        tol = 1e2 * f.vscale * EPS
        assert abs(complex(y) - exact) < tol
        assert abs(complex(fun(x)) - exact) < tol

    def test_min_complex_array(self, Tech):
        # pass(n, 8): complex array-valued min; fx picks each column at
        # its own position (MATLAB fx([1 4]) diagonal trick).
        import warnings

        def fun(x):
            return jnp.stack(
                [jnp.exp(1j * x) - 0.5j * jnp.sin(x) + x,
                 (1 + 0.5 * (x - 0.1) ** 2) * jnp.exp(1j * x)],
                axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(fun)
        y, x = f.min()
        exact = np.array([0.074968381369117 - 0.319744137826069j,
                          0.995004165278026 + 0.099833416646827j])
        fx = np.asarray(jnp.diagonal(fun(x)))
        tol = 1e2 * f.vscale * EPS
        assert np.max(np.abs(np.asarray(y) - exact)) < tol
        assert np.max(np.abs(fx - exact)) < tol
