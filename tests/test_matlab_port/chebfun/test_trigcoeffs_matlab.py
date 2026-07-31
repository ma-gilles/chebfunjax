"""Port of MATLAB Chebfun tests/chebfun/test_trigcoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunTrigcoeffs:
    def test_complex_exponential_modes(self):
        # pass(1): 1 + (2+2i)e^{2pi i x} + (5+5i)e^{-5pi i x} on [-1,1].
        f = cj.chebfun(
            lambda x: 1 + (2 + 2j) * jnp.exp(1j * 2 * np.pi * x)
            + (5 + 5j) * jnp.exp(-1j * 5 * np.pi * x), trig=True)
        c = np.asarray(f.trigcoeffs())
        c_exact = np.zeros(11, dtype=complex)
        c_exact[0] = 5 + 5j     # mode -5
        c_exact[5] = 1          # mode 0
        c_exact[7] = 2 + 2j     # mode +2
        tol = 1e2 * float(f.vscale) * EPS
        assert float(np.max(np.abs(c - c_exact))) < tol

    def test_cos_sin_form(self):
        # pass(2): cosine/sine coefficient extraction.
        f = cj.chebfun(
            lambda x: 1 + 5 * jnp.cos(5 * np.pi * x)
            + 10 * jnp.cos(10 * np.pi * x) - 7 * jnp.sin(7 * np.pi * x)
            + 8 * jnp.sin(8 * np.pi * x), trig=True)
        a, b = f.trigcoeffs(form="cos_sin")
        a_exact = np.zeros(11)
        a_exact[0], a_exact[5], a_exact[10] = 1, 5, 10
        b_exact = np.zeros(10)
        b_exact[6], b_exact[7] = -7, 8
        tol = 1e2 * float(f.vscale) * EPS
        assert float(np.max(np.abs(np.asarray(a) - a_exact))) < tol
        assert float(np.max(np.abs(np.asarray(b) - b_exact))) < tol

    def test_truncation_on_pi_domain(self):
        # pass(3)-ish: 2 + 2cos(2x) + sin(x) on [-pi, pi].
        f = cj.chebfun(lambda x: 2 + 2 * jnp.cos(2 * x) + jnp.sin(x),
                       domain=[-np.pi, np.pi], trig=True)
        c = np.asarray(f.trigcoeffs(5))
        c_exact = np.array([1.0, 0.5j, 2.0, -0.5j, 1.0], dtype=complex)
        tol = 1e2 * float(f.vscale) * EPS
        assert float(np.max(np.abs(c - c_exact))) < tol

    def test_nonperiodic_with_n(self):
        # Square wave via splitting: b_k = 4/(pi*k) for odd k.
        sq = cj.chebfun(lambda x: jnp.sign(jnp.sin(x)),
                        domain=[-np.pi, np.pi], splitting=True)
        a, b = sq.trigcoeffs(15, form="cos_sin")
        k = np.arange(1, 8)
        b_exact = np.where(k % 2 == 1, 4.0 / (np.pi * k), 0.0)
        assert float(np.max(np.abs(np.asarray(b) - b_exact))) < 1e-12
        assert float(np.max(np.abs(np.asarray(a)))) < 1e-12
