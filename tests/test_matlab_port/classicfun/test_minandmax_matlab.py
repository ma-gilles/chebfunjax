"""Port of MATLAB Chebfun tests/classicfun/test_minandmax.m (Opus 4.8).

Self-validating: the simultaneous global min and max of a Bndfun are
spot-checked against known extreme values at the SAME tolerances MATLAB uses
(values within 100*eps, operator-at-position within 10*eps).  Airy is
evaluated with SciPy inside the constructor sampling (test-only).

Provenance
----------
MATLAB source : tests/classicfun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
INF = np.inf


def _bf(op):
    return Bndfun.from_function(op, DOM)


def _spotcheck_minmax(op, exact_min, exact_max):
    f = _bf(op)
    (min_val, min_pos), (max_val, max_pos) = f.minandmax()
    y = np.array([complex(min_val), complex(max_val)])
    y_exact = np.array([exact_min, exact_max])
    fx = np.array([complex(np.asarray(op(min_pos))),
                   complex(np.asarray(op(max_pos)))])
    assert np.max(np.abs(y - y_exact)) < 100 * EPS
    assert np.max(np.abs(fx - y_exact)) < 10 * EPS


class TestClassicfunMinAndMax:
    def test_sine(self):
        _spotcheck_minmax(lambda x: jnp.sin(10 * x), -1.0, 1.0)

    def test_airy(self):
        _spotcheck_minmax(
            lambda x: jnp.asarray(sp.airy(np.asarray(x))[0]),
            7.492128863997157e-07,
            0.535656656015700,
        )

    def test_neg_lorentzian(self):
        _spotcheck_minmax(lambda x: -1.0 / (1 + x ** 2), -1.0, -0.02)

    def test_cubic_cosh(self):
        _spotcheck_minmax(
            lambda x: (x / 10) ** 3 * jnp.cosh(x / 10),
            (-0.2) ** 3 * np.cosh(-0.2),
            0.7 ** 3 * np.cosh(0.7),
        )

    _ARR = staticmethod(
        lambda x: jnp.stack(
            [
                jnp.sin(10 * x),
                jnp.asarray(np.real(sp.airy(np.asarray(x))[0])),
                (x / 10) ** 3 * jnp.cosh(x / 10),
            ],
            axis=-1,
        )
    )
    _YEX = np.array(
        [[-1.0, 7.492128863997157e-07, (-0.2) ** 3 * np.cosh(-0.2)],
         [1.0, 0.535656656015700, 0.7 ** 3 * np.cosh(0.7)]]
    )

    def test_array_valued_values(self):
        # pass(5): minandmax of [sin(10x) real(airy(x)) (x/10)^3 cosh(x/10)]
        # gives the 2x3 matrix of per-column extrema, tol 100*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(self._ARR)
        (mn, _), (mx, _) = f.minandmax()
        y = np.vstack([np.asarray(mn), np.asarray(mx)])
        assert np.max(np.abs(y - self._YEX)) < 100 * EPS

    def test_array_valued_positions(self):
        # pass(6): op evaluated at the per-column extreme positions matches the
        # extrema, tol 10*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(self._ARR)
        (mn, mnp), (mx, mxp) = f.minandmax()
        m = 3
        fx_min = np.asarray(self._ARR(mnp))[np.arange(m), np.arange(m)]
        fx_max = np.asarray(self._ARR(mxp))[np.arange(m), np.arange(m)]
        assert np.max(np.abs(fx_min - self._YEX[0])) < 10 * EPS
        assert np.max(np.abs(fx_max - self._YEX[1])) < 10 * EPS

    def test_complex_array_valued(self):
        # pass(7): abs(minandmax([exp(sin 2x), 1i cos 20x])) equals the abs of
        # the per-column scalar minandmax, tol 100*vscale*eps.
        # FIXED (Fable 5, Big-Three array-valued epic): complex extrema work.
        fop = lambda x: jnp.stack([jnp.exp(jnp.sin(2 * x)), 1j * jnp.cos(20 * x)], axis=-1)
        f = _bf(fop)
        (cmn, _), (cmx, _) = f.minandmax()
        vals_abs = np.abs(np.vstack([np.asarray(cmn), np.asarray(cmx)]))
        f1 = _bf(lambda x: jnp.exp(jnp.sin(2 * x)))
        f2 = _bf(lambda x: 1j * jnp.cos(20 * x))
        (a1, _), (a2, _) = f1.minandmax()
        (b1, _), (b2, _) = f2.minandmax()
        ref_abs = np.abs(
            np.array([[complex(a1), complex(b1)], [complex(a2), complex(b2)]])
        )
        assert np.max(np.abs(vals_abs - ref_abs)) < 100 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="Singular Bndfun (exponents (-0.5,0)) now BUILDS, but "
        "Singfun.minandmax ignores the algebraic blowup: for (x-a)^-0.5 "
        "sin(x)^2 it returns the smooth-part max (~0.14) instead of MATLAB's "
        "+Inf.  Needs @singfun/minandmax blowup handling (tech/Singfun layer)."
    )
    def test_singular(self):
        pow_ = -0.5
        op = lambda x: (x - DOM.a) ** pow_ * jnp.sin(x) ** 2
        f = Bndfun.from_function(op, DOM, exponents=(pow_, 0.0))
        (min_val, _), (max_val, _) = f.minandmax()
        assert float(min_val) < 1e-10
        assert not np.isfinite(float(max_val))  # MATLAB: +Inf

    def test_complex_array_valued_2(self):
        # MATLAB records the complex-array-valued minandmax assertion twice
        # (identical); covered by test_complex_array_valued.
        pytest.skip(
            "duplicate of test_complex_array_valued (MATLAB records the same "
            "assertion twice)"
        )

    def test_unbndfun(self):  # pass(10): (1-e^{-x^2})/x on (-inf, inf)
        f = Unbndfun.from_function(
            lambda x: (1 - jnp.exp(-x ** 2)) / x, Domain((-INF, INF))
        )
        (min_val, min_pos), (max_val, max_pos) = f.minandmax()
        v = np.array([complex(min_val), complex(max_val)])
        p = np.array([float(min_pos), float(max_pos)])
        v_exact = np.array([-0.6381726863389515, 0.6381726863389515])
        p_exact = np.array([-1.120906422778534, 1.120906422778534])
        vscale = float(f.vscale)
        assert np.max(np.abs(v - v_exact)) < 1e1 * EPS * vscale
        assert np.max(np.abs(p - p_exact)) < 1e2 * EPS * vscale
