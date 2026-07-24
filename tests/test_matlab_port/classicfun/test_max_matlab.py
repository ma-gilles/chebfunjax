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
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
INF = np.inf


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

    def test_array_valued(self):
        # pass(5): max of [sin(10x) airy(x) (x/10)^3 cosh(x/10)] per column.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        fun_op = lambda x: jnp.stack(
            [
                jnp.sin(10 * x),
                jnp.asarray(sp.airy(np.asarray(x))[0]),
                (x / 10) ** 3 * jnp.cosh(x / 10),
            ],
            axis=-1,
        )
        f = _bf(fun_op)
        y, xpos = f.max()
        y = np.asarray(y)
        fx = np.asarray(fun_op(xpos))[np.arange(3), np.arange(3)]
        exact = np.array([1.0, 0.535656656015700, 0.7 ** 3 * np.cosh(0.7)])
        tol = 10 * f.vscale * EPS
        assert np.max(np.abs(y - exact)) < 10 * tol
        assert np.max(np.abs(fx - exact)) < tol

    def test_complex_valued(self):
        # pass(6): max of a complex-valued Bndfun.
        # FIXED (Fable 5, Big-Three array-valued epic): complex extrema now work.
        _spotcheck_max(
            lambda x: (x / 2) * (jnp.exp(1j * (x / 2)) + 1j * jnp.sin(x / 2)),
            -3.277598405517787 - 2.455482593827339j,
        )

    def test_complex_array_valued(self):
        # pass(7): max of a complex array-valued Bndfun, per column.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fun_op = lambda x: jnp.stack(
            [
                ((x - 2) ** 2 / 4 + 1) * jnp.exp(1j * (x / 2)),
                -((x + 1) ** 2 / 4 + 1) * jnp.exp(1j * (x / 2)),
            ],
            axis=-1,
        )
        f = _bf(fun_op)
        y, xpos = f.max()
        y = np.asarray(y)
        fx = np.asarray(fun_op(xpos))[np.arange(2), np.arange(2)]
        exact = np.array(
            [-6.789310982858273 - 2.543178400749744j,
             15.919763683943538 + 5.963314870723537j]
        )
        tol = f.vscale * EPS
        assert np.max(np.abs(y - exact)) < 10 * tol
        assert np.max(np.abs(fx - exact)) < tol

    def test_unbndfun_max(self):  # pass(8): x e^{-x} on [1, inf)
        f = Unbndfun.from_function(lambda x: x * jnp.exp(-x), Domain((1.0, INF)))
        y, xpos = f.max()
        err = np.abs(complex(y) - np.exp(-1.0)) + np.abs(float(xpos) - 1.0)
        assert err < 1e3 * EPS * float(f.vscale)
