"""Port of MATLAB Chebfun tests/bndfun/test_mrdivide.m (Fable 5).

MATLAB ``f / A`` (mrdivide) divides a Bndfun quasimatrix by a scalar or
numeric matrix, and ``c / f`` solves a least-squares problem giving a Bndfun.
chebfunjax exposes these as the named methods ``Bndfun.mrdivide`` and
``Bndfun.rmrdivide``, which delegate to the onefun exactly as MATLAB's
``@bndfun/mrdivide.m`` does; Python's ``/`` operator stays pointwise division
(MATLAB ``./``).  With those in place every MATLAB assertion is ported at
MATLAB's tolerances.

The ``double / bndfun`` branch carries MATLAB's ``X / (0.5*diff(domain))``
rescale, which is what produces the ``2/9`` factor in pass 6 on ``[-2, 7]``.

No gaps: all nine MATLAB passes are exercised.  chebfunjax raises ``ValueError``
carrying the MATLAB identifier text (``mrdivide:size`` /
``mrdivide:bndfunDivBndfun`` / ``mrdivide:badArg``) rather than a MATLAB
exception object.

Provenance
----------
MATLAB source : tests/bndfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)

# MATLAB: dom = [-2 7]; x = diff(dom)*rand(100,1) + dom(1).
DOM = Domain((-2.0, 7.0))
XR = np.linspace(-2.0, 7.0, 100)
X = jnp.asarray(XR)

# MATLAB: alpha = -0.194758928283640 + 0.075474485412665i.
ALPHA = -0.194758928283640 + 0.075474485412665j


def _bf(f, n=None):
    return Bndfun.from_function(f, DOM, n=n)


def _sin_cos(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)


def _max_abs(a):
    return float(np.max(np.abs(np.asarray(a))))


class TestBndfunMrdivide:
    def test_divide_by_zero_is_nan(self):
        # pass(1): isnan(f / 0).
        f = _bf(_sin_cos)
        assert bool(np.all(np.isnan(np.asarray(f.mrdivide(0).onefun.coeffs))))

    def test_divide_by_scalar_array_valued(self):
        # pass(2): f / alpha.
        f = _bf(_sin_cos)
        g = f.mrdivide(ALPHA)
        exact = np.stack([np.sin(XR), np.cos(XR)], axis=-1) / ALPHA
        assert _max_abs(g(X) - exact) < 10 * g.vscale * EPS

    def test_divide_by_identity_matrix(self):
        # pass(3): g = f / eye(2); g*eye(2) == f.
        f = _bf(_sin_cos)
        eye2 = jnp.eye(2)
        g = f.mrdivide(eye2)
        err = (g @ eye2) - f
        assert _max_abs(err(X)) < 1e2 * g.vscale * EPS

    def test_divide_by_row_vector_least_squares(self):
        # pass(4): g = f / [1 1] == (sin + cos)/2.
        f = _bf(_sin_cos)
        g = f.mrdivide(jnp.array([[1.0, 1.0]]))
        exact = (np.sin(XR) + np.cos(XR)) / 2
        assert _max_abs(g(X) - exact) < 1e2 * g.vscale * EPS

    def test_scalar_divided_by_bndfun(self):
        # pass(5): g = alpha / f; innerProduct(f, g) == alpha.
        f = _bf(jnp.sin)
        g = Bndfun.rmrdivide(ALPHA, f)
        ip = complex(jnp.reshape(jnp.asarray(f.inner(g)), ()))
        assert abs(ip - ALPHA) < 10 * g.vscale * EPS

    def test_row_vector_divided_by_bndfun(self):
        # pass(6): [1 1] / [sin(2*pi*x) cos(2*pi*x)] == (2/9)*(sin + cos).
        # The 2/9 is MATLAB's X / (0.5*diff(dom)) rescale with diff(dom) = 9.
        f = _bf(lambda x: jnp.stack(
            [jnp.sin(2 * np.pi * x), jnp.cos(2 * np.pi * x)], axis=-1))
        g = Bndfun.rmrdivide(jnp.array([[1.0, 1.0]]), f)
        exact = (2 / 9) * (np.sin(2 * np.pi * XR) + np.cos(2 * np.pi * XR))
        assert _max_abs(g(X) - exact) < 1e2 * g.vscale * EPS

    def test_error_dimension_mismatch(self):
        # pass(7): f / [1 2 3] raises mrdivide:size.
        f = _bf(_sin_cos)
        with pytest.raises(ValueError, match="mrdivide:size"):
            f.mrdivide(jnp.array([[1.0, 2.0, 3.0]]))

    def test_error_bndfun_div_bndfun(self):
        # pass(8): f / g raises mrdivide:bndfunDivBndfun.
        f = _bf(jnp.sin)
        g = _bf(jnp.cos)
        with pytest.raises(ValueError, match="bndfunDivBndfun"):
            f.mrdivide(g)

    def test_error_bad_arg(self):
        # pass(9): f / true raises mrdivide:badArg.
        f = _bf(jnp.sin)
        with pytest.raises(ValueError, match="mrdivide:badArg"):
            f.mrdivide(True)
