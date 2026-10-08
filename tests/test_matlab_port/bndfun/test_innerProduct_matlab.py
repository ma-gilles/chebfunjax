"""Port of MATLAB Chebfun tests/bndfun/test_innerProduct.m (Opus 4.8).

Self-validating: known inner products and algebraic properties, checked at
the SAME tolerances MATLAB uses.  chebfunjax's inner product is
conjugate-linear in the first argument (matches @chebtech/innerProduct.m).

Provenance
----------
MATLAB source : tests/bndfun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.utils.airy_general import airy_all

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS
DOM = Domain((-2.0, 7.0))
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


def _bf(f):
    return Bndfun.from_function(f, DOM)


def _ip(f, g):
    return complex(f.inner(g))


class TestBndfunInnerProduct:
    def test_orthogonal_sin_cos_same_freq(self):
        f = _bf(lambda x: jnp.sin(2 * np.pi * x))
        g = _bf(lambda x: jnp.cos(2 * np.pi * x))
        assert abs(_ip(f, g)) < 10 * EPS

    def test_orthogonal_sin_cos_diff_freq(self):
        f = _bf(lambda x: jnp.sin(2 * np.pi * x))
        g = _bf(lambda x: jnp.cos(4 * np.pi * x))
        assert abs(_ip(f, g)) < 10 * EPS

    def test_exp_times_exp_neg(self):
        f = _bf(jnp.exp)
        g = _bf(lambda x: jnp.exp(-x))
        assert abs(_ip(f, g) - 9) < 1e2 * max(f.vscale, g.vscale) * EPS

    def test_exp_times_sin_exact(self):
        f = _bf(jnp.exp)
        g = _bf(jnp.sin)
        exact = (
            np.exp(7) * (np.sin(7) - np.cos(7)) / 2
            - np.exp(-2) * (np.sin(-2) - np.cos(-2)) / 2
        )
        assert abs(_ip(f, g) - exact) < max(f.vscale, g.vscale) * 10 * EPS

    def test_conjugate_linearity(self):
        f = _bf(lambda x: jnp.exp(1j * x) - 1)
        g = _bf(lambda x: 1.0 / (1 + 1j * x ** 2))
        ip1 = _ip(ALPHA * f, BETA * g)
        ip2 = np.conj(ALPHA) * BETA * _ip(f, g)
        assert abs(ip1 - ip2) < 10 * TOL

    def test_hermitian_symmetry(self):
        g = _bf(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = _bf(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        assert abs(_ip(g, h) - np.conj(_ip(h, g))) < TOL

    def test_left_additivity(self):
        f = _bf(lambda x: jnp.exp(1j * x) - 1)
        g = _bf(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = _bf(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        ip1 = _ip(f + g, h)
        ip2 = _ip(f, h) + _ip(g, h)
        assert abs(ip1 - ip2) < max((f.vscale + g.vscale), h.vscale) * TOL

    def test_right_additivity(self):
        f = _bf(lambda x: jnp.exp(1j * x) - 1)
        g = _bf(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = _bf(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        ip1 = _ip(f, g + h)
        ip2 = _ip(f, g) + _ip(f, h)
        assert abs(ip1 - ip2) < max((g.vscale + h.vscale), f.vscale) * TOL

    def test_self_inner_products_real_nonneg(self):
        f = _bf(lambda x: jnp.exp(1j * x) - 1)
        g = _bf(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = _bf(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        n2vals = np.array([_ip(f, f), _ip(g, g), _ip(h, h)])
        assert np.all(n2vals.imag == 0)
        assert np.all(n2vals.real >= 0)

    def test_array_valued(self):
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = _bf(lambda x: jnp.stack([jnp.exp(x), 1/(1+x**2), airy_all(x)[0]], axis=-1))
        exact = np.array([[-53.1070904269318222, 0.0025548835039100, -0.4683303433821355],
                          [773.70343924989359096771, 1.3148120368924471, 0.6450791915572742]])
        assert np.max(np.abs(np.asarray(f.inner(g))-exact)) < 10*EPS*max(f.vscale,g.vscale)

    def test_error_on_non_bndfun(self):
        f = _bf(jnp.sin)
        with pytest.raises(TypeError, match="CHEBFUN:BNDFUN:innerProduct:input"):
            f.inner(2)

    def test_singular_function(self):
        pow1, pow2 = -0.3, -0.5
        f = Bndfun.from_function(lambda x: (x - DOM.b + 0j) ** pow1 * jnp.sin(x), DOM, exponents=(0.0, pow1))
        g = Bndfun.from_function(lambda x: (x - DOM.b + 0j) ** pow2 * jnp.cos(3 * x), DOM, exponents=(0.0, pow2))
        I = _ip(f, g)
        I_exact = -0.65182492763883119 + 0.47357853074362785j
        assert abs(I - I_exact) < 5e2 * EPS * abs(I_exact)
