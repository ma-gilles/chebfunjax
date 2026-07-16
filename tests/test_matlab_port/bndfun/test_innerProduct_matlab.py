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

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS
DOM = Domain((-2.0, 7.0))
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


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
        assert np.all(np.abs(n2vals.imag) < 10 * EPS)
        assert np.all(n2vals.real >= 0)

    def test_array_valued(self):
        # pass(10): innerProduct of a 2-column f with a 3-column g yields the
        # 2x3 Gram matrix of pairwise inner products.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun; inner
        # returns the full Gram matrix.  MATLAB's third g column is airy(x),
        # which chebfunjax has no special function for, so we substitute cos(x)
        # and validate against the matrix of the corresponding SCALAR inner
        # products (the exact array-valued property MATLAB asserts).
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = _bf(
            lambda x: jnp.stack(
                [jnp.exp(x), 1.0 / (1 + x ** 2), jnp.cos(x)], axis=-1
            )
        )
        ip = np.asarray(f.inner(g))
        assert ip.shape == (2, 3)
        f_cols = [_bf(jnp.sin), _bf(jnp.cos)]
        g_cols = [_bf(jnp.exp), _bf(lambda x: 1.0 / (1 + x ** 2)), _bf(jnp.cos)]
        ref = np.array(
            [[complex(f_cols[i].inner(g_cols[j])) for j in range(3)] for i in range(2)]
        )
        assert float(np.max(np.abs(ip - ref))) < 10 * max(f.vscale, g.vscale) * EPS

    def test_error_on_non_bndfun(self):
        # MATLAB raises CHEBFUN:BNDFUN:innerProduct:input.  chebfunjax reaches
        # into `other.onefun`, so a non-Bndfun argument raises AttributeError.
        f = _bf(jnp.sin)
        with pytest.raises(AttributeError):
            f.inner(2)

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: (x-b)^p factors "
        "cannot be constructed via Bndfun.from_function."
    )
    def test_singular_function(self):
        pow1, pow2 = -0.3, -0.5
        f = _bf(lambda x: (x - DOM.b) ** pow1 * jnp.sin(x), n=17)
        g = _bf(lambda x: (x - DOM.b) ** pow2 * jnp.cos(3 * x), n=17)
        I = _ip(f, g)
        I_exact = -0.65182492763883119 + 0.47357853074362785j
        assert abs(I - I_exact) < 5e2 * EPS * abs(I_exact)
