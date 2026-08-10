"""Port of MATLAB Chebfun tests/trigtech/test_trigcoeffs.m (Opus 4.8).

``trigcoeffs(f)`` returns the Fourier coefficients (== ``f.coeffs`` for a
resolved trigtech), and ``trigcoeffs(f, N)`` returns exactly N of them,
zero-padding or truncating symmetrically.  chebfunjax implements
``Trigtech.trigcoeffs`` as an exact port of ``@trigtech/trigcoeffs.m``,
including the even-``N`` Nyquist fold on truncation.

Provenance
----------
MATLAB source : tests/trigtech/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _trigcoeffs(f, n=None):
    return f.trigcoeffs(n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechTrigcoeffs:
    def test_zeros(self):
        f = _tt(lambda x: jnp.zeros_like(x))
        p = _trigcoeffs(f)
        assert _ninf(p) <= 10 * f.vscale * EPS

    def test_constant(self):
        f = _tt(lambda x: 3 * jnp.ones_like(x))
        p = _trigcoeffs(f)
        assert _ninf(p - 3) < 10 * f.vscale * EPS

    # Odd tests: f = 1 + cos(pi x) -> [.5 1 .5]
    def test_odd_default(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f)
        assert _ninf(p - jnp.array([0.5, 1, 0.5])) < 10 * f.vscale * EPS

    def test_odd_pad_to_five(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 5)
        assert _ninf(p - jnp.array([0, 0.5, 1, 0.5, 0])) < 10 * f.vscale * EPS

    def test_odd_truncate_to_one(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 1)
        assert _ninf(p - 1) < 10 * f.vscale * EPS

    # f = 1 + exp(2i pi x) + exp(-i pi x) -> [0 1 1 0 1]
    def test_complex_default(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f)
        assert _ninf(p - jnp.array([0, 1, 1, 0, 1], dtype=jnp.complex128)) < 10 * f.vscale * EPS

    def test_complex_pad_to_nine(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 9)
        exact = jnp.array([0, 0, 0, 1, 1, 0, 1, 0, 0], dtype=jnp.complex128)
        assert _ninf(p - exact) < 10 * f.vscale * EPS

    def test_complex_truncate_to_three(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 3)
        assert _ninf(p - jnp.array([1, 1, 0], dtype=jnp.complex128)) < 10 * f.vscale * EPS

    # Even tests
    def test_even_cos(self):
        f = _tt(lambda x: 2 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 2)
        assert _ninf(p - jnp.array([1, 2])) < 10 * f.vscale * EPS

    def test_even_sin(self):
        f = _tt(lambda x: 2 + jnp.sin(jnp.pi * x))
        p = _trigcoeffs(f, 2)
        assert _ninf(p - jnp.array([0, 2])) < 10 * f.vscale * EPS

    def test_even_cos2(self):
        f = _tt(lambda x: 2 + jnp.cos(2 * jnp.pi * x))
        p = _trigcoeffs(f, 4)
        assert _ninf(p - jnp.array([1, 0, 2, 0])) < 10 * f.vscale * EPS

    def _array_f(self):
        return _tt(lambda x: jnp.stack(
            [3 * jnp.ones_like(x),
             1 + jnp.cos(jnp.pi * x),
             1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)],
            axis=-1))

    def test_array_default(self):
        f = self._array_f()
        p = _trigcoeffs(f)
        p_exact = jnp.array([[0, 0, 0],
                             [0, 0.5, 1],
                             [3, 1, 1],
                             [0, 0.5, 0],
                             [0, 0, 1]], dtype=jnp.complex128)
        assert _ninf(p - p_exact) < 10 * f.vscale * EPS

    def test_array_pad(self):
        f = self._array_f()
        p = _trigcoeffs(f, 7)
        p_exact = jnp.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0.5, 1],
                             [3, 1, 1],
                             [0, 0.5, 0],
                             [0, 0, 1],
                             [0, 0, 0]], dtype=jnp.complex128)
        assert _ninf(p - p_exact) < 10 * f.vscale * EPS

    def test_array_truncate(self):
        f = self._array_f()
        p = _trigcoeffs(f, 3)
        p_exact = jnp.array([[0, 0.5, 1],
                             [3, 1, 1],
                             [0, 0.5, 0]], dtype=jnp.complex128)
        assert _ninf(p - p_exact) < 10 * f.vscale * EPS

    def test_zero_length(self):
        # trigcoeffs(f, 0) is empty.
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 0)
        assert p.shape[0] == 0
