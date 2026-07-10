"""Port of MATLAB Chebfun tests/bndfun/test_poly.m (Opus 4.8).

MATLAB ``poly(f)`` returns the monomial (power-basis) coefficients of the
polynomial represented by the Bndfun.  chebfunjax has no ``poly`` method on
Bndfun/Classicfun (nor a cheb2poly conversion exposed at the fun level), so
every assertion here is xfail with that precise reason.

Provenance
----------
MATLAB source : tests/bndfun/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
_POLY_MISSING = "chebfunjax has no Bndfun.poly() (power-basis coefficients)"


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


class TestBndfunPoly:
    @pytest.mark.xfail(reason=_POLY_MISSING, raises=AttributeError)
    def test_zeros(self):
        f = _bf(lambda x: jnp.zeros_like(x))
        p = np.asarray(f.poly())
        assert float(np.max(np.abs(p))) <= f.vscale * EPS

    @pytest.mark.xfail(reason=_POLY_MISSING, raises=AttributeError)
    def test_constant(self):
        f = _bf(lambda x: 3 * jnp.ones_like(x))
        p = np.asarray(f.poly())
        assert float(np.max(np.abs(p - 3))) < f.vscale * EPS

    @pytest.mark.xfail(reason=_POLY_MISSING, raises=AttributeError)
    def test_linear_complex(self):
        f = _bf(lambda x: 6.4 * x - 3j)
        p = np.asarray(f.poly())
        assert float(np.max(np.abs(p - np.array([6.4, -3j])))) < f.vscale * EPS

    @pytest.mark.xfail(reason=_POLY_MISSING, raises=AttributeError)
    def test_quintic_complex(self):
        f = _bf(lambda x: 2j * x ** 5 - 3.2 * x ** 4 + 2 * x ** 2 - (1.2 + 3j))
        p = np.asarray(f.poly())
        exact = np.array([2j, -3.2, 0, 2, 0, -(1.2 + 3j)])
        assert float(np.max(np.abs(p - exact))) < f.vscale * EPS

    @pytest.mark.xfail(
        reason=_POLY_MISSING + "; also needs array-valued Bndfun.",
        raises=AttributeError,
    )
    def test_array_valued(self):
        f = _bf(
            lambda x: jnp.stack(
                [3 * jnp.ones_like(x), 6.4 * x - 3j, 4 * x ** 2 - 2j * x + 3.7],
                axis=-1,
            ),
            n=17,
        )
        f.poly()
