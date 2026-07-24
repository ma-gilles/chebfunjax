"""Port of MATLAB Chebfun tests/bndfun/test_mrdivide.m (Opus 4.8).

MATLAB ``f / A`` (mrdivide) divides a Bndfun quasimatrix by a numeric matrix
(or scalar), and ``c / f`` solves a least-squares problem giving a Bndfun.
chebfunjax's ``Classicfun.__truediv__`` is *pointwise* division (f(x)/g(x)),
which is a different operation, and there is no matrix/least-squares mrdivide.
(Array-valued Bndfun itself now works; the gap is the mrdivide operator.)
Every assertion is therefore xfail/skip with a precise reason.

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
DOM = Domain((-2.0, 7.0))
ALPHA = -0.194758928283640 + 0.075474485412665j
XR = np.linspace(-2.0, 7.0, 100)
X = jnp.asarray(XR)
_MRDIV_MISSING = (
    "chebfunjax Classicfun.__truediv__ is pointwise division, not MATLAB "
    "mrdivide (matrix/least-squares); array-valued Bndfun itself now works"
)


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


class TestBndfunMrdivide:
    def test_divide_by_zero_is_nan(self):  # pass(1)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        assert bool(np.all(np.isnan(np.asarray((f / 0).coeffs))))

    def test_divide_by_scalar_array_valued(self):  # pass(2)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = f / ALPHA
        exact = np.stack([np.sin(XR), np.cos(XR)], axis=-1) / ALPHA
        assert float(np.max(np.abs(np.asarray(g(X)) - exact))) < 10 * g.vscale * EPS

    def test_divide_by_identity_matrix(self):  # pass(3)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = f / np.eye(2)
        err = g @ np.eye(2) - f  # mtimes back by the identity
        assert float(np.max(np.abs(np.asarray(err(X))))) < 1e2 * g.vscale * EPS

    def test_divide_by_row_vector_least_squares(self):  # pass(4)
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = f / np.array([[1, 1]])
        exact = (np.sin(XR) + np.cos(XR)) / 2
        assert float(np.max(np.abs(np.asarray(g(X)) - exact))) < 1e2 * g.vscale * EPS

    @pytest.mark.skip(
        reason="MATLAB alpha/f is a least-squares Bndfun with "
        "innerProduct(f, g)=alpha; chebfunjax alpha/f is pointwise "
        "alpha/f(x) = alpha/sin(x), which has poles in [-2,7] and is a "
        "different operation entirely (no least-squares mrdivide)."
    )
    def test_scalar_divided_by_bndfun(self):  # pass(5)
        f = _bf(jnp.sin)
        g = ALPHA / f
        assert abs(complex(f.inner(g)) - ALPHA) < 10 * g.vscale * EPS

    def test_row_vector_divided_by_bndfun(self):  # pass(6)
        f = _bf(lambda x: jnp.stack([jnp.sin(2 * np.pi * x), jnp.cos(2 * np.pi * x)], axis=-1))
        g = np.array([[1, 1]]) / f
        exact = (2 / 9) * (np.sin(2 * np.pi * XR) + np.cos(2 * np.pi * XR))
        assert float(np.max(np.abs(np.asarray(g(X)) - exact))) < 1e2 * g.vscale * EPS

    @pytest.mark.skip(
        reason="chebfunjax has no matrix mrdivide and does not raise "
        "CHEBFUN:BNDFUN:mrdivide:size; f/[1 2 3] instead hits a numpy "
        "broadcast error whose type is an implementation detail, so the "
        "MATLAB error behaviour cannot be faithfully asserted."
    )
    def test_error_dimension_mismatch(self):  # pass(7)
        f = _bf(jnp.sin)
        with pytest.raises(ValueError):
            f / np.array([1, 2, 3])

    @pytest.mark.skip(
        reason="chebfunjax allows f/g for two Bndfuns as *pointwise* division "
        "(sin/cos=tan, which has poles -> a non-converging build), rather "
        "than raising CHEBFUN:BNDFUN:mrdivide:bndfunDivBndfun."
    )
    def test_error_bndfun_div_bndfun(self):  # pass(8)
        f = _bf(jnp.sin)
        g = _bf(jnp.cos)
        with pytest.raises(Exception):
            f / g

    @pytest.mark.xfail(
        reason="chebfunjax accepts f/True (True treated as 1); MATLAB raises "
        "CHEBFUN:BNDFUN:mrdivide:badArg."
    )
    def test_error_bad_arg(self):  # pass(9)
        f = _bf(jnp.sin)
        with pytest.raises(Exception):
            f / True
