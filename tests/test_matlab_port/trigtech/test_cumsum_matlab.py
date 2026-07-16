"""Port of MATLAB Chebfun tests/trigtech/test_cumsum.m (Opus 4.8).

trigtech's cumsum only works when the mean of the function is zero (the
antiderivative of a non-zero-mean periodic function is not periodic), so
every test function has zero mean; the last test checks the error raised
otherwise.  Antiderivatives are pinned by feval(cumsum(f), -1) == 0.

Provenance
----------
MATLAB source : tests/trigtech/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _std(a):
    return float(jnp.std(jnp.asarray(a)))


class TestTrigtechCumsum:
    def test_antiderivative_k2(self):
        k, a = 2, 0.0
        f = _tt(lambda x: jnp.sin(k * jnp.pi * (x - a)) * jnp.cos((k + 1) * jnp.pi * (x - a)))
        F = f.cumsum()
        F_ex = ((2 * k + 1) * jnp.cos(jnp.pi * (X - a)) - jnp.cos((2 * k + 1) * jnp.pi * (X - a))) / (
            2 * (jnp.pi + 2 * k * jnp.pi)
        )
        err = F(X) - F_ex
        tol = 10 * F.vscale * EPS
        assert _std(err) < tol
        assert abs(float(F(jnp.array(-1.0)))) < tol

    def test_antiderivative_k200(self):
        k, a = 200, 0.17
        f = _tt(lambda x: jnp.sin(k * jnp.pi * (x - a)) * jnp.cos((k + 1) * jnp.pi * (x - a)))
        F = f.cumsum()
        F_ex = ((2 * k + 1) * jnp.cos(jnp.pi * (X - a)) - jnp.cos((2 * k + 1) * jnp.pi * (X - a))) / (
            2 * (jnp.pi + 2 * k * jnp.pi)
        )
        err = F(X) - F_ex
        tol = 100 * F.vscale * EPS
        assert _std(err) < tol
        assert abs(float(F(jnp.array(-1.0)))) < tol

    def test_antiderivative_complex(self):
        k1, a1, k2, a2 = 5, -0.33, 40, 0.17
        f = _tt(
            lambda x: jnp.sin(k1 * jnp.pi * (x - a1)) * jnp.cos((k1 + 1) * jnp.pi * (x - a1))
            + 1j * jnp.sin(k2 * jnp.pi * (x - a2)) * jnp.cos((k2 + 1) * jnp.pi * (x - a2))
        )
        F = f.cumsum()
        F_ex = (
            (2 * k1 + 1) * jnp.cos(jnp.pi * (X - a1)) - jnp.cos((2 * k1 + 1) * jnp.pi * (X - a1))
        ) / (2 * (jnp.pi + 2 * k1 * jnp.pi)) + 1j * (
            (2 * k2 + 1) * jnp.cos(jnp.pi * (X - a2)) - jnp.cos((2 * k2 + 1) * jnp.pi * (X - a2))
        ) / (2 * (jnp.pi + 2 * k2 * jnp.pi))
        err = F(X) - F_ex
        tol = 100 * F.vscale * EPS
        assert _std(err) < tol
        assert abs(complex(F(jnp.array(-1.0)))) < tol

    def test_diff_of_cumsum_is_identity(self):
        f = _tt(lambda x: jnp.sin(4 * jnp.pi * jnp.cos(jnp.pi * x)))
        g = f.cumsum().diff()
        err = f(X) - g(X)
        tol = 10 * g.vscale * EPS
        assert _ninf(err) < 100 * tol

    def test_cumsum_of_diff_is_identity_up_to_const(self):
        f = _tt(lambda x: jnp.sin(4 * jnp.pi * jnp.cos(jnp.pi * x)))
        h = f.diff().cumsum()
        err = f(X) - h(X)
        tol = 10 * h.vscale * EPS
        assert _std(err) < tol
        assert abs(float(h(jnp.array(-1.0)))) < tol

    def test_error_when_mean_not_zero(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        with pytest.raises(ValueError):
            f.cumsum()

    def test_array_valued_cumsum(self):
        # pass(6, 7): diff(cumsum(f)) == f and cumsum(diff(f)) == f (up to a
        # constant per column) for f = [sin(4pi cos(2pi x)) sin(3pi x)].
        # FIXED (Fable 5, Big-Three array-valued epic): column-wise cumsum.
        f = _tt(
            lambda x: jnp.stack(
                [jnp.sin(4 * jnp.pi * jnp.cos(2 * jnp.pi * x)), jnp.sin(3 * jnp.pi * x)],
                axis=-1,
            )
        )
        # pass(6): all(max(abs(err)) < 100*tol)
        g = f.cumsum().diff()
        err = np.asarray(f(X) - g(X))
        tol = 10 * g.vscale * EPS
        assert bool(np.all(np.max(np.abs(err), axis=0) < 100 * tol))
        # pass(7): all(std(err) < tol) && all(abs(feval(h, -1)) < tol)
        h = f.diff().cumsum()
        errh = np.asarray(f(X) - h(X))
        tolh = 10 * h.vscale * EPS
        assert bool(np.all(np.std(errh, axis=0) < tolh))
        assert bool(np.all(np.abs(np.asarray(h(jnp.array(-1.0)))) < tolh))

    def test_array_valued_mean_check(self):
        # pass(9): error when just one column of an array-valued trigtech has
        # nonzero mean.
        # FIXED (Fable 5, Big-Three array-valued epic): the meanNotZero guard
        # is enforced per column.
        f = _tt(
            lambda x: jnp.stack(
                [jnp.sin(4 * jnp.pi * jnp.cos(jnp.pi * x)), jnp.exp(jnp.cos(jnp.pi * x))],
                axis=-1,
            )
        )
        with pytest.raises(ValueError):
            f.cumsum()
