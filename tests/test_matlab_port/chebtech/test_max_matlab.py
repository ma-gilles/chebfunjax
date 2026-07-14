"""Port of MATLAB Chebfun tests/chebtech/test_max.m (Opus 4.8).

Self-validating: each maximum value and its location are checked against an
analytic exact at the SAME tolerance MATLAB uses (10 * vscale(f) * eps).
The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``.

``max`` exists ONLY on Chebtech2 in chebfunjax (Chebtech1 lacks it), so every
method xfails the Chebtech1 parametrization with a precise reason.  MATLAB
``[y, x] = max(f)`` maps to ``y, x = f.max()``.

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 has no ``max``.
- complex-valued / complex-array-valued ``max``: FIXED (Fable 5) --
  minandmax follows MATLAB's complex path (extrema of |f|^2), so the
  complex spot-checks port at the same tolerances.

Real array-valued ``max`` (pass(n, 6)) is now supported: techs carry (n, m)
coeffs and ``max`` returns per-column ``(m,)`` values/positions.

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


def airy(x):
    """Airy Ai at ``x`` (MATLAB ``airy(x)`` -> ``Ai``), as a NumPy array."""
    return sp.airy(np.asarray(x))[0]


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
        # FIXED (Fable 5, Big-Three array-valued epic): techs now carry (n, m)
        # coeffs and max returns per-column (m,) values/positions.
        self._skip_c1(Tech)
        fun = lambda x: jnp.stack(
            [jnp.sin(10 * x), airy(x), (x - 0.25) ** 3 * jnp.cosh(x)], axis=-1
        )
        f = Tech.from_function(fun)
        y, xpos = f.max()
        exact = jnp.array(
            [1.0, float(sp.airy(-1.0)[0]), 0.75**3 * float(np.cosh(1.0))],
            dtype=jnp.float64,
        )
        # MATLAB uses the plain tol 10*eps for pass(n, 6).
        assert float(jnp.max(jnp.abs(y - exact))) < 10 * EPS
        assert float(jnp.max(jnp.abs(jnp.diagonal(fun(xpos)) - exact))) < 10 * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): max now follows
    # MATLAB's complex path (|f|^2 extrema), so pass(n, 7)-(8) port.
    def test_max_complex(self, Tech):
        # pass(n, 7): max of (x-0.2)*(exp(1i(x-0.2)) + 1i sin(x-0.2)).
        self._skip_c1(Tech)
        import warnings

        def fun(x):
            return (x - 0.2) * (jnp.exp(1j * (x - 0.2))
                                + 1j * jnp.sin(x - 0.2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(fun)
        y, x = f.max()
        exact = -0.434829305372008 + 2.236893806321343j
        tol = 10 * f.vscale * EPS
        assert abs(complex(y) - exact) < tol
        assert abs(complex(fun(x)) - exact) < tol

    def test_max_complex_array(self, Tech):
        # pass(n, 8): complex array-valued max of [sin(z) sinh(z)],
        # z = (x - 0.3 + 1i)^3 - 2i; tol = 10*max(vscale(f), eps).
        self._skip_c1(Tech)
        import warnings

        def z(x):
            return (x - 0.3 + 1j) ** 3 - 2j

        def fun(x):
            return jnp.stack([jnp.sin(z(x)), jnp.sinh(z(x))], axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(fun)
        y, x = f.max()
        exact = np.array([-10.017874927409903j, 3.626860407847019])
        fx = np.array([complex(jnp.sin(z(x[0]))),
                       complex(jnp.sinh(z(x[1])))])
        tol = 10 * max(f.vscale, EPS)
        assert np.max(np.abs(np.asarray(y) - exact)) < tol
        assert np.max(np.abs(fx - exact)) < tol
