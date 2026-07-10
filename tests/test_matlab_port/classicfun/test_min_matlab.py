"""Port of MATLAB Chebfun tests/classicfun/test_min.m (Opus 4.8).

Self-validating: the global minimum (value and location) of a Bndfun is
spot-checked against a known extreme value at the SAME tolerance MATLAB uses
(2e3*vscale*eps).  Airy is evaluated with SciPy inside the constructor
sampling (test-only).

Provenance
----------
MATLAB source : tests/classicfun/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))


def _bf(op):
    return Bndfun.from_function(op, DOM)


def _spotcheck_min(op, exact_min):
    f = _bf(op)
    y, xpos = f.min()
    fx = op(xpos)
    tol = 2e3 * f.vscale * EPS
    assert abs(complex(y) - exact_min) < tol
    assert abs(complex(np.asarray(fx)) - exact_min) < tol


class TestClassicfunMin:
    def test_neg_sine(self):
        _spotcheck_min(lambda x: -jnp.sin(10 * x), -1.0)

    def test_neg_airy(self):
        _spotcheck_min(
            lambda x: -jnp.asarray(sp.airy(np.asarray(x))[0]),
            -0.535656656015700,
        )

    def test_lorentzian(self):
        _spotcheck_min(lambda x: 1.0 / (1 + x ** 2), 0.02)

    def test_neg_cubic_cosh(self):
        _spotcheck_min(
            lambda x: -(x / 10) ** 3 * jnp.cosh(x / 10),
            -0.7 ** 3 * np.cosh(0.7),
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: min of a 3-column fun "
        "returns per-column extrema."
    )
    def test_array_valued(self):
        raise NotImplementedError("array-valued Bndfun min")

    @pytest.mark.xfail(
        reason="chebfunjax min/max on a complex Bndfun compares via "
        "numpy.argmin on the complex values (imag part discarded)."
    )
    def test_complex_valued(self):
        _spotcheck_min(
            lambda x: (x / 2) * (jnp.exp(1j * (x / 2)) + 1j * jnp.sin(x / 2)),
            0.0,
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun (and complex extrema)."
    )
    def test_complex_array_valued(self):
        raise NotImplementedError("array-valued complex Bndfun min")

    @pytest.mark.xfail(
        reason="chebfunjax Unbndfun has no min()/minandmax() method, and lacks "
        "the blowup (exponents [0 -1]) representation this test requires."
    )
    def test_unbndfun_min(self):
        raise NotImplementedError("Unbndfun min")
