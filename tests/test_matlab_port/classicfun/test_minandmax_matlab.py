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

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))


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

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: minandmax of a 3-column "
        "fun returns a 2xN matrix of extrema."
    )
    def test_array_valued_values(self):
        raise NotImplementedError("array-valued Bndfun minandmax")

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: cannot check per-column "
        "extreme positions."
    )
    def test_array_valued_positions(self):
        raise NotImplementedError("array-valued Bndfun minandmax positions")

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun and complex extrema "
        "(minandmax compares by numpy argmin/argmax, not abs)."
    )
    def test_complex_array_valued(self):
        raise NotImplementedError("array-valued complex Bndfun minandmax")

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (exponents [-0.5 0]) Bndfun."
    )
    def test_singular(self):
        raise NotImplementedError("singular Bndfun minandmax")

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun and complex extrema."
    )
    def test_complex_array_valued_2(self):
        raise NotImplementedError("array-valued complex Bndfun minandmax (2)")

    @pytest.mark.xfail(
        reason="chebfunjax Unbndfun has no minandmax() method."
    )
    def test_unbndfun(self):
        raise NotImplementedError("Unbndfun minandmax")
