"""Port of MATLAB Chebfun tests/bndfun/test_mldivide.m (Opus 4.8).

MATLAB ``f \\ g`` (mldivide) solves the least-squares problem for Bndfun
quasimatrices: given array-valued Bndfuns, it returns the numeric matrix of
coefficients that best expands ``g`` in the columns of ``f``.  chebfunjax has
no ``mldivide``/backslash on Bndfun/Classicfun, so every assertion is xfail
with that precise reason.  (Array-valued Bndfun itself now works.)

Provenance
----------
MATLAB source : tests/bndfun/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

TOL = 10 * float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
_MLDIV_MISSING = "chebfunjax has no Bndfun mldivide (\\, least-squares) operator"


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


class TestBndfunMldivide:
    def test_scalar_self_divide_coefficient(self):  # pass(1)
        f = _bf(jnp.sin)
        g = f.mldivide(f)
        assert abs(float(g) - 1) < TOL

    def test_scalar_self_divide_residual(self):  # pass(2)
        f = _bf(jnp.sin)
        g = f.mldivide(f)
        err = f - g * f
        assert float(jnp.max(jnp.abs(err(jnp.asarray(np.linspace(-2, 7, 100)))))) < TOL

    def test_array_valued_solution(self):  # pass(3)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = _bf(lambda x: jnp.sin(x + np.pi / 4))
        h = f.mldivide(g)
        assert float(np.max(np.abs(np.asarray(h) - np.array([1, 1]) / np.sqrt(2)))) < TOL

    def test_array_valued_residual(self):  # pass(4)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = _bf(lambda x: jnp.sin(x + np.pi / 4))
        h = f.mldivide(g)
        err = g - f @ h  # mtimes: mix the columns of f by h
        assert float(jnp.max(jnp.abs(err(jnp.asarray(np.linspace(-2, 7, 100)))))) < TOL

    def test_least_squares_solution(self):  # pass(5)
        f = _bf(lambda x: jnp.stack([jnp.ones_like(x), x, x ** 2, x ** 3], axis=-1))
        g = _bf(lambda x: x ** 4 + x ** 3 + x + 1)
        sol = np.asarray(f.mldivide(g))
        exact = np.array([2469 / 70, -163 / 7, -141 / 7, 11])
        assert float(np.max(np.abs(sol - exact))) < 1000 * TOL

    def test_error_on_non_bndfun(self):  # pass(6)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        with pytest.raises(TypeError):
            f.mldivide(2)
