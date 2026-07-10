"""Port of MATLAB Chebfun tests/classicfun/test_max.m (Opus 4.8).

Self-validating: the global maximum (value and location) of a Bndfun is
spot-checked against a known extreme value at the SAME tolerance MATLAB uses
(100*vscale*eps).  Airy is evaluated with SciPy inside the constructor
sampling (test-only).

Provenance
----------
MATLAB source : tests/classicfun/test_max.m
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


def _spotcheck_max(op, exact_max):
    f = _bf(op)
    y, xpos = f.max()
    fx = op(xpos)
    tol = 100 * f.vscale * EPS
    assert abs(complex(y) - exact_max) < tol
    assert abs(complex(np.asarray(fx)) - exact_max) < tol


class TestClassicfunMax:
    def test_sine(self):
        _spotcheck_max(lambda x: jnp.sin(10 * x), 1.0)

    def test_airy(self):
        _spotcheck_max(
            lambda x: jnp.asarray(sp.airy(np.asarray(x))[0]),
            0.535656656015700,
        )

    def test_neg_lorentzian(self):
        _spotcheck_max(lambda x: -1.0 / (1 + x ** 2), -0.02)

    def test_cubic_cosh(self):
        _spotcheck_max(
            lambda x: (x / 10) ** 3 * jnp.cosh(x / 10),
            0.7 ** 3 * np.cosh(0.7),
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: max of a 3-column fun "
        "returns per-column extrema."
    )
    def test_array_valued(self):
        raise NotImplementedError("array-valued Bndfun max")

    @pytest.mark.xfail(
        reason="chebfunjax min/max on a complex Bndfun compares via "
        "numpy.argmax on the complex values (imag part discarded), so the "
        "complex extremum is not the MATLAB one."
    )
    def test_complex_valued(self):
        _spotcheck_max(
            lambda x: (x / 2) * (jnp.exp(1j * (x / 2)) + 1j * jnp.sin(x / 2)),
            -3.277598405517787 - 2.455482593827339j,
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun (and complex extrema)."
    )
    def test_complex_array_valued(self):
        raise NotImplementedError("array-valued complex Bndfun max")

    @pytest.mark.xfail(
        reason="chebfunjax Unbndfun has no max()/minandmax() method."
    )
    def test_unbndfun_max(self):
        raise NotImplementedError("Unbndfun max")
