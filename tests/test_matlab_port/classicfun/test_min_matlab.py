"""Port of MATLAB Chebfun tests/classicfun/test_min.m (Fable 5).

Self-validating: the global minimum (value and location) of a Bndfun is
spot-checked against a known extreme value at the SAME tolerance MATLAB uses
(2e3*vscale*eps).  Airy is evaluated with SciPy inside the constructor
sampling (test-only).

Scalar, array-valued, complex and complex-array-valued cases all port.


Provenance
----------
MATLAB source : tests/classicfun/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

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

    def test_array_valued(self):
        # pass(5): min of -[sin(10x) real(airy(x)) (x/10)^3 cosh(x/10)].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        fun_op = lambda x: -jnp.stack(
            [
                jnp.sin(10 * x),
                jnp.asarray(np.real(sp.airy(np.asarray(x))[0])),
                (x / 10) ** 3 * jnp.cosh(x / 10),
            ],
            axis=-1,
        )
        f = _bf(fun_op)
        y, xpos = f.min()
        y = np.asarray(y)
        fx = np.asarray(fun_op(xpos))[np.arange(3), np.arange(3)]
        exact = -np.array([1.0, 0.535656656015700, 0.7 ** 3 * np.cosh(0.7)])
        tol = 10 * np.asarray(f.onefun.vscale_columns) * EPS
        assert np.all(np.abs(y - exact) < 10 * tol)
        assert np.all(np.abs(fx - exact) < tol)

    def test_complex_valued(self):
        # pass(6): min of a complex-valued Bndfun (exact 0).
        # FIXED (Fable 5, Big-Three array-valued epic): complex extrema now work.
        _spotcheck_min(
            lambda x: (x / 2) * (jnp.exp(1j * (x / 2)) + 1j * jnp.sin(x / 2)),
            0.0,
        )

    def test_complex_array_valued(self):
        # pass(7): min of a complex array-valued Bndfun, per column.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fun_op = lambda x: jnp.stack(
            [
                ((x - 2) ** 2 / 4 + 1) * jnp.exp(1j * (x / 2)),
                -((x + 1) ** 2 / 4 + 1) * jnp.exp(1j * (x / 2)),
            ],
            axis=-1,
        )
        f = _bf(fun_op)
        y, xpos = f.min()
        y = np.asarray(y)
        fx = np.asarray(fun_op(xpos))[np.arange(2), np.arange(2)]
        exact = np.array([np.exp(1j), -np.exp(-1j / 2)])
        tol = 10 * f.vscale * EPS
        assert np.max(np.abs(y - exact)) < 10 * tol
        assert np.max(np.abs(fx - exact)) < tol

    def test_unbndfun_min(self):
        domain = Domain((-np.inf, -3 * np.pi))
        op = lambda x: x * (5 + jnp.exp(x ** 3)) / (domain.b - x)
        f = Unbndfun.from_function(op, domain, exps=(0.0, -1.0))
        y, x = f.min()
        assert abs(float(x) - domain.b) < EPS * f.vscale
        assert float(y) == -np.inf
