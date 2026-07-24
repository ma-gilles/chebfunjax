"""Port of MATLAB Chebfun tests/trigtech/test_diffmat.m (Opus 4.8[1m]).

diffmat builds the trigcolloc Fourier spectral differentiation matrix D
mapping values on the equispaced grid to derivative values.

Provenance
----------
MATLAB source : tests/trigtech/test_diffmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.discretization.trigcolloc import trig_diffmat
from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)


def _tt(op):
    return Trigtech.from_function(op)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _grid_and_vals(op, force_odd):
    """Mirror the MATLAB parity handling: pick n = length(f), bump it by
    one if its parity is wrong, and sample the operator on that grid."""
    f = _tt(op)
    n = f.n
    want_odd = force_odd
    if (n % 2 == 1) != want_odd:
        n += 1
    x = trigpts(n)
    v = op(x)
    return f, n, x, v


class TestTrigtechDiffmat:
    def test_d1_odd(self):
        op = lambda x: jnp.exp(jnp.cos(jnp.pi * x))  # noqa: E731
        f, n, x, v = _grid_and_vals(op, force_odd=True)
        df = trig_diffmat(n, 1) @ v
        exact = -jnp.pi * jnp.sin(jnp.pi * x) * jnp.exp(jnp.cos(jnp.pi * x))
        assert _ninf(exact - df) < 1e3 * f.vscale * EPS

    def test_d1_odd_hard(self):
        a, b = 10, 20
        op = lambda x: jnp.cos(a * jnp.pi * jnp.sin(b * jnp.pi * x))  # noqa: E731
        f, n, x, v = _grid_and_vals(op, force_odd=True)
        df = trig_diffmat(n, 1) @ v
        exact = (-jnp.pi ** 2 * a * b * jnp.cos(b * jnp.pi * x)
                 * jnp.sin(a * jnp.pi * jnp.sin(b * jnp.pi * x)))
        assert _ninf(exact - df) < 1e7 * f.vscale * EPS

    def test_d1_even(self):
        op = lambda x: jnp.exp(-50 * x ** 2)  # noqa: E731
        f, n, x, v = _grid_and_vals(op, force_odd=False)
        df = trig_diffmat(n, 1) @ v
        exact = -100 * x * jnp.exp(-50 * x ** 2)
        assert _ninf(exact - df) < 1e3 * f.vscale * EPS

    def test_d1_even_complex(self):
        a1, b1, a2, b2 = 4, 3, 6, 4
        op = lambda x: (jnp.cos(a1 * jnp.pi * jnp.sin(b1 * jnp.pi * x))  # noqa: E731
                        + 1j * jnp.cos(a2 * jnp.pi * jnp.sin(b2 * jnp.pi * x)))
        f, n, x, v = _grid_and_vals(op, force_odd=False)
        df = trig_diffmat(n, 1) @ v
        exact = (-jnp.pi ** 2 * a1 * b1 * jnp.cos(b1 * jnp.pi * x)
                 * jnp.sin(a1 * jnp.pi * jnp.sin(b1 * jnp.pi * x))
                 - 1j * jnp.pi ** 2 * a2 * b2 * jnp.cos(b2 * jnp.pi * x)
                 * jnp.sin(a2 * jnp.pi * jnp.sin(b2 * jnp.pi * x)))
        assert _ninf(exact - df) < 1e5 * f.vscale * EPS

    def test_d2_odd(self):
        op = lambda x: jnp.exp(jnp.cos(4 * jnp.pi * x)) - 1  # noqa: E731
        f = _tt(op)
        n = f.n
        x = trigpts(n)
        v = op(x)
        exact = (-16 * jnp.pi ** 2 * jnp.exp(jnp.cos(4 * jnp.pi * x))
                 * (jnp.cos(4 * jnp.pi * x)
                    + jnp.cos(4 * jnp.pi * x) ** 2 - 1))
        assert _ninf(exact - trig_diffmat(n, 2) @ v) < 5e5 * f.vscale * EPS
        # And with n bumped by one (opposite parity).
        n2 = n + 1
        x2 = trigpts(n2)
        v2 = op(x2)
        exact2 = (-16 * jnp.pi ** 2 * jnp.exp(jnp.cos(4 * jnp.pi * x2))
                  * (jnp.cos(4 * jnp.pi * x2)
                     + jnp.cos(4 * jnp.pi * x2) ** 2 - 1))
        assert _ninf(exact2 - trig_diffmat(n2, 2) @ v2) < 5e5 * f.vscale * EPS

    def test_d2_even(self):
        # Covered together with the odd case above (kept for the MATLAB
        # pass(6) parity check).
        op = lambda x: jnp.exp(jnp.cos(4 * jnp.pi * x)) - 1  # noqa: E731
        f = _tt(op)
        n = f.n + 1
        x = trigpts(n)
        v = op(x)
        exact = (-16 * jnp.pi ** 2 * jnp.exp(jnp.cos(4 * jnp.pi * x))
                 * (jnp.cos(4 * jnp.pi * x)
                    + jnp.cos(4 * jnp.pi * x) ** 2 - 1))
        assert _ninf(exact - trig_diffmat(n, 2) @ v) < 5e5 * f.vscale * EPS

    def test_d5_odd(self):
        op = lambda x: jnp.sin(jnp.pi * x)  # noqa: E731
        f = _tt(op)
        n = f.n
        x = trigpts(n)
        v = op(x)
        exact = jnp.pi ** 5 * jnp.cos(jnp.pi * x)
        assert _ninf(exact - trig_diffmat(n, 5) @ v) < 1e3 * f.vscale * EPS

    def test_d5_even(self):
        op = lambda x: jnp.sin(jnp.pi * x)  # noqa: E731
        f = _tt(op)
        n = f.n + 1
        x = trigpts(n)
        v = op(x)
        exact = jnp.pi ** 5 * jnp.cos(jnp.pi * x)
        assert _ninf(exact - trig_diffmat(n, 5) @ v) < 1e3 * f.vscale * EPS

    def test_d6_odd(self):
        op = lambda x: jnp.sin(jnp.pi * x)  # noqa: E731
        f = _tt(op)
        n = f.n
        x = trigpts(n)
        v = op(x)
        exact = -jnp.pi ** 6 * jnp.sin(jnp.pi * x)
        assert _ninf(exact - trig_diffmat(n, 6) @ v) < 1e4 * f.vscale * EPS

    def test_d6_even(self):
        op = lambda x: jnp.sin(jnp.pi * x)  # noqa: E731
        f = _tt(op)
        n = f.n + 1
        x = trigpts(n)
        v = op(x)
        exact = -jnp.pi ** 6 * jnp.sin(jnp.pi * x)
        assert _ninf(exact - trig_diffmat(n, 6) @ v) < 1e5 * f.vscale * EPS
