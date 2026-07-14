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
- complex-array-valued: FIXED (Fable 5) -- ``minandmax`` follows
  MATLAB's complex path (extrema of |f|^2, values = f at those
  positions), so pass(n, 7)-(8) port at the same tolerances.

Real array-valued ``minandmax`` (pass(n, 6)) is now supported: techs carry
(n, m) coeffs and ``minandmax`` returns per-column ``(m,)`` values/positions.

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


def airy(x):
    """Airy Ai at ``x`` (MATLAB ``airy(x)`` -> ``Ai``), as a NumPy array."""
    return sp.airy(np.asarray(x))[0]


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
        # FIXED (Fable 5, Big-Three array-valued epic): techs now carry (n, m)
        # coeffs and minandmax returns per-column (m,) values/positions.
        self._skip_c1(Tech)
        fun = lambda x: jnp.stack(
            [jnp.sin(10 * x), airy(x), (x - 0.25) ** 3 * jnp.cosh(x)], axis=-1
        )
        f = Tech.from_function(fun)
        (mn, xmn), (mx, xmx) = f.minandmax()
        min_exact = jnp.array(
            [-1.0, float(sp.airy(1.0)[0]), (-1.25) ** 3 * float(np.cosh(1.0))],
            dtype=jnp.float64,
        )
        max_exact = jnp.array(
            [1.0, float(sp.airy(-1.0)[0]), 0.75**3 * float(np.cosh(1.0))],
            dtype=jnp.float64,
        )
        # MATLAB value tol: 10*max(vscale(f)*eps); vscale is the scalar global max.
        tol = 10 * f.vscale * EPS
        assert float(jnp.max(jnp.abs(mn - min_exact))) < tol
        assert float(jnp.max(jnp.abs(mx - max_exact))) < tol
        # MATLAB position check: each column's function at its own extremum
        # matches the exact value (tol 1e1*max(vscale(f)*eps)).
        tol_pos = 1e1 * f.vscale * EPS
        assert float(jnp.max(jnp.abs(jnp.diagonal(fun(xmn)) - min_exact))) < tol_pos
        assert float(jnp.max(jnp.abs(jnp.diagonal(fun(xmx)) - max_exact))) < tol_pos

    # FIXED (Fable 5, Big-Three array-valued epic): minandmax now takes
    # MATLAB's complex path (extrema of |f|^2, values = f at those
    # positions), so pass(n, 7)-(8) port at the same tolerances.
    def test_minmax_complex_array_vals(self, Tech):
        # pass(n, 7): complex array-valued minandmax (|vals| comparison).
        self._skip_c1(Tech)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(
                lambda x: jnp.stack(
                    [jnp.exp(jnp.sin(2 * x)), 1j * jnp.cos(20 * x)],
                    axis=-1))
            f1 = Tech.from_function(lambda x: jnp.exp(jnp.sin(2 * x)))
            f2 = Tech.from_function(lambda x: 1j * jnp.cos(20 * x))
        (mn, _), (mx, _) = f.minandmax()
        (mn1, _), (mx1, _) = f1.minandmax()
        (mn2, _), (mx2, _) = f2.minandmax()
        vals = np.abs(np.array(
            [[complex(mn[0]), complex(mn[1])],
             [complex(mx[0]), complex(mx[1])]]))
        ref = np.abs(np.array(
            [[complex(mn1), complex(mn2)],
             [complex(mx1), complex(mx2)]]))
        assert np.max(np.abs(vals - ref)) < 1e2 * f.vscale * EPS

    def test_minmax_complex_array_pos(self, Tech):
        # pass(n, 8): complex array-valued minandmax (position of the
        # FIRST column only -- the second column's extrema are not
        # unique, as MATLAB's own comment notes).
        self._skip_c1(Tech)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(
                lambda x: jnp.stack(
                    [jnp.exp(jnp.sin(2 * x)), 1j * jnp.cos(20 * x)],
                    axis=-1))
            f1 = Tech.from_function(lambda x: jnp.exp(jnp.sin(2 * x)))
        (_, xmn), (_, xmx) = f.minandmax()
        (_, p1), (_, P1) = f1.minandmax()
        pos = np.array([float(xmn[0]), float(xmx[0])])
        pos1 = np.array([float(p1), float(P1)])
        assert np.max(np.abs(pos - pos1)) < 500 * f.vscale * EPS
