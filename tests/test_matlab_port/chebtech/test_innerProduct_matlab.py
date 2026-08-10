"""Port of MATLAB Chebfun tests/chebtech/test_innerProduct.m (Opus 4.8; marker
audit Fable 5).

Self-validating: inner products are checked against analytic exacts and known
algebraic properties at the SAME tolerances MATLAB uses.  The MATLAB test loops
``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we parametrize over
``[Chebtech1, Chebtech2]``.

chebfunjax ``f.inner(g)`` is the L2 inner product, conjugate-linear in ``f``
(matching MATLAB's ``innerProduct``).

Every MATLAB assertion (pass 1-11) is ported on BOTH tech kinds; there are no
gaps:

* Complex-valued construction works on Chebtech1 as well as Chebtech2, so
  sub-tests 5-8 (complex ``g``/``h``) run on both.
* Sub-test 9 (``isreal(<f,f>) && <f,f> >= 0``) passes because ``inner`` now
  carries MATLAB's ``isequal(f, g)`` "force the diagonal to real, non-negative"
  branch (innerProduct.m lines 37-41).
* The array-valued exact-matrix case (pass 10) is a real test now that
  ``inner`` returns the (m, m') pairwise column Gram matrix.
* Sub-test 11 checks that ``<f, non-tech>`` errors.  chebfunjax has no MATLAB
  error identifiers (``CHEBFUN:CHEBTECH:innerProduct:input``), so the ported
  test asserts only that an exception is raised.

Provenance
----------
MATLAB source : tests/chebtech/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

BOTH = [Chebtech1, Chebtech2]

# Fixed arbitrary multiplicative constants (from the MATLAB test).
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


class TestChebtechInnerProduct:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_orthogonal_sin_cos(self, Tech):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.sin(2 * np.pi * x))
        g = Tech.from_function(lambda x: jnp.cos(2 * np.pi * x))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        assert abs(complex(f.inner(g))) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_orthogonal_sin_cos4(self, Tech):
        # pass(n, 2)
        f = Tech.from_function(lambda x: jnp.sin(2 * np.pi * x))
        g = Tech.from_function(lambda x: jnp.cos(4 * np.pi * x))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        assert abs(complex(f.inner(g))) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_exp_times_exp_neg(self, Tech):
        # pass(n, 3): <exp(x), exp(-x)> == 2
        f = Tech.from_function(lambda x: jnp.exp(x))
        g = Tech.from_function(lambda x: jnp.exp(-x))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        assert abs(complex(f.inner(g)) - 2) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_exp_times_sin(self, Tech):
        # pass(n, 4)
        f = Tech.from_function(lambda x: jnp.exp(x))
        g = Tech.from_function(lambda x: jnp.sin(x))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        exact = (
            np.exp(1) * (np.sin(1) - np.cos(1)) / 2
            - np.exp(-1) * (np.sin(-1) - np.cos(-1)) / 2
        )
        assert abs(complex(f.inner(g)) - exact) < max(tol_f, tol_g)

    # pass(n, 5): conjugate-linearity in the first argument.  g is complex;
    # both tech kinds handle complex data, so this runs on n = 1:2.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_conjugate_linearity(self, Tech):
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        ip1 = complex((ALPHA * f).inner(BETA * g))
        ip2 = complex(np.conj(ALPHA) * BETA * f.inner(g))
        assert abs(ip1 - ip2) < max(tol_f, tol_g)

    # pass(n, 6): conjugate symmetry <g,h> == conj(<h,g>).  g, h complex.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_conjugate_symmetry(self, Tech):
        g = Tech.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = Tech.from_function(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        tol_g = 10 * EPS * g.vscale
        tol_h = 10 * EPS * h.vscale
        assert abs(complex(g.inner(h)) - np.conj(complex(h.inner(g)))) < max(
            tol_g, tol_h
        )

    # pass(n, 7): additivity in the first argument.  g, h complex.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_additivity_first_arg(self, Tech):
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = Tech.from_function(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        tol_h = 10 * EPS * h.vscale
        ip1 = complex((f + g).inner(h))
        ip2 = complex(f.inner(h) + g.inner(h))
        assert abs(ip1 - ip2) < max(tol_f, tol_g, tol_h)

    # pass(n, 8): additivity in the second argument.  g, h complex.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_additivity_second_arg(self, Tech):
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = Tech.from_function(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        tol_f = 10 * EPS * f.vscale
        tol_g = 10 * EPS * g.vscale
        tol_h = 10 * EPS * h.vscale
        ip1 = complex(f.inner(g + h))
        ip2 = complex(f.inner(g) + f.inner(h))
        assert abs(ip1 - ip2) < max(tol_f, tol_g, tol_h)

    # pass(n, 9): <f,f> real & non-negative.  Passes since the Fable 5
    # audit added MATLAB's isequal(f,g) force-real-nonnegative branch
    # (innerProduct.m lines 37-41) to Chebtech{1,2}.inner.
    @pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
    def test_self_inner_products_real_nonneg(self, Tech):
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
        h = Tech.from_function(lambda x: jnp.sinh(x * np.exp(np.pi * 1j / 6)))
        n2vals = [f.inner(f), g.inner(g), h.inner(h)]
        # isreal(n2vals): imaginary part exactly zero, as MATLAB guarantees.
        assert all(float(jnp.imag(v)) == 0.0 for v in n2vals)
        assert all(float(jnp.real(v)) >= 0.0 for v in n2vals)

    # FIXED (Fable 5, Big-Three array-valued epic): pass 10 ports now
    # that inner returns the pairwise column Gram matrix.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_inner_product_matrix(self, Tech):
        from scipy.special import airy

        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.exp(x), 1.0 / (1 + x ** 2),
                 jnp.asarray(airy(np.asarray(x))[0])], axis=-1))
        tol_f = 10 * f.vscale * EPS
        tol_g = 10 * g.vscale * EPS
        ip = np.asarray(f.inner(g))
        exact = np.array(
            [[0.663493666631241, 0.0, -0.135033172317858],
             [1.933421496200713, 1.365866063614065, 0.592109441404267]])
        assert np.max(np.abs(ip - exact)) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_inner_product_with_non_tech_errors(self, Tech):
        # pass(n, 11): innerProduct(f, 2) must error.  MATLAB checks the id
        # 'CHEBFUN:CHEBTECH:innerProduct:input'; chebfunjax has no MATLAB error
        # identifiers, so we assert only that it raises (it currently surfaces
        # as AttributeError from the missing .coeffs attribute).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        with pytest.raises(Exception):
            f.inner(2)


def test_chebtech1_rejects_complex():
    # FIXED (Fable 5): Chebtech1 now splits complex data into re/im
    # in vals2coeffs/coeffs2vals; this sentinel now passes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Chebtech1.from_function(lambda x: 1.0 / (1 + 1j * x ** 2))
    assert f.ishappy
