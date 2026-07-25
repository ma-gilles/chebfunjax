"""Port of MATLAB Chebfun tests/trigtech/test_coeffs2vals.m (Opus 4.8).

Static transform from Fourier coefficients to equispaced values (inverse
of vals2coeffs).  chebfunjax's transform returns exactly-real values for
purely real symmetric coefficient input (and exactly-imaginary for the
imaginary branch), matching MATLAB's ``~any(imag(v))`` / ``~any(real(v))``
checks.

Provenance
----------
MATLAB source : tests/trigtech/test_coeffs2vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _c2v(c):
    return Trigtech.coeffs2vals(c)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _anynonzero(a):
    return bool(jnp.any(jnp.abs(jnp.asarray(a)) > 0))


class TestTrigtechCoeffs2Vals:
    def test_single_coeff(self):
        c = jnp.array([np.sqrt(2.0)], dtype=jnp.complex128)
        assert _ninf(_c2v(c) - np.sqrt(2.0)) == 0.0

    # Even case: c = [0 .5 0 1 0 .5], f = 1 + cos(2 pi x), n = 6
    def _even_setup(self):
        c = jnp.array([0, 0.5, 0, 1, 0, 0.5], dtype=jnp.complex128)
        vTrue = 1 + jnp.cos(2 * jnp.pi * trigpts(6))
        return c, vTrue

    def test_even_real_branch(self):
        c, vTrue = self._even_setup()
        v = _c2v(c)
        assert _ninf(v - vTrue) < TOL

    def test_even_real_no_imag(self):
        c, _ = self._even_setup()
        v = _c2v(c)
        assert not _anynonzero(jnp.imag(v))

    def test_even_imag_branch(self):
        c, vTrue = self._even_setup()
        v = _c2v(1j * c)
        assert _ninf(v - 1j * vTrue) < TOL

    def test_even_imag_no_real(self):
        c, _ = self._even_setup()
        v = _c2v(1j * c)
        assert not _anynonzero(jnp.real(v))

    def test_even_general_branch(self):
        c, vTrue = self._even_setup()
        v = _c2v((1 + 1j) * c)
        assert _ninf(v - (1 + 1j) * vTrue) < TOL

    # Odd case: c = [.5 .5i 0 1 0 -.5i .5], f = 1 + sin(2 pi x) + cos(3 pi x), n = 7
    def _odd_setup(self):
        c = jnp.array([0.5, 0.5j, 0, 1, 0, -0.5j, 0.5], dtype=jnp.complex128)
        vTrue = 1 + jnp.sin(2 * jnp.pi * trigpts(7)) + jnp.cos(3 * jnp.pi * trigpts(7))
        return c, vTrue

    def test_odd_real_branch(self):
        c, vTrue = self._odd_setup()
        v = _c2v(c)
        assert _ninf(v - vTrue) < TOL

    def test_odd_real_no_imag(self):
        c, _ = self._odd_setup()
        v = _c2v(c)
        assert not _anynonzero(jnp.imag(v))

    def test_odd_imag_branch(self):
        c, vTrue = self._odd_setup()
        v = _c2v(1j * c)
        assert _ninf(v - 1j * vTrue) < TOL

    def test_odd_imag_no_real(self):
        c, _ = self._odd_setup()
        v = _c2v(1j * c)
        assert not _anynonzero(jnp.real(v))

    def test_odd_general_branch(self):
        c, vTrue = self._odd_setup()
        v = _c2v((1 + 1j) * c)
        assert _ninf(v - (1 + 1j) * vTrue) < TOL

    def test_even_array_input(self):
        c, vTrue = self._even_setup()
        v = _c2v(jnp.stack([c, -c], axis=-1))
        assert _ninf(v[:, 0] - vTrue) < TOL and _ninf(v[:, 1] + vTrue) < TOL

    def test_odd_array_input(self):
        c, vTrue = self._odd_setup()
        v = _c2v(jnp.stack([c, -c], axis=-1))
        assert _ninf(v[:, 0] - vTrue) < TOL and _ninf(v[:, 1] + vTrue) < TOL

    def test_symmetry_real(self):
        # Real coeffs -> exactly Hermitian values (MATLAB real branch).
        c = jnp.ones(100, dtype=jnp.complex128)
        v = _c2v(c)
        v = jnp.concatenate([v, v[:1]])
        assert float(jnp.linalg.norm(v - jnp.flip(jnp.conj(v)))) == 0.0

    def test_symmetry_imag(self):
        # Imaginary coeffs -> exactly skew-Hermitian values.
        c = jnp.ones(100, dtype=jnp.complex128)
        v = _c2v(1j * c)
        v = jnp.concatenate([v, v[:1]])
        assert float(jnp.linalg.norm(v + jnp.flip(jnp.conj(v)))) == 0.0
