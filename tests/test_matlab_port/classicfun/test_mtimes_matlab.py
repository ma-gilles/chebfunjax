"""Port of MATLAB Chebfun tests/classicfun/test_mtimes.m (Fable 5).

Scalar, array-valued and matrix cases at MATLAB tolerances on the MATLAB
domain [-2, 7], plus the Unbndfun matrix product on [-Inf, -3*pi].  MATLAB's
``mtimes`` (``*``) maps to Python's ``@`` here, because chebfunjax follows the
Python convention that ``*`` is elementwise (MATLAB's ``.*``/``times``).

Gap: MATLAB raises typed errors (``CHEBFUN:CLASSICFUN:mtimes:size``,
``CHEBFUN:CHEBTECH:mtimes:size2``) with fixed messages; chebfunjax raises
plain ``TypeError``/``ValueError``, so the error-path assertions check only
that the operation raises.  pass(12) (rejecting a ``uint8`` multiplier) has no
counterpart at all and is skipped.

Provenance
----------
MATLAB source : tests/classicfun/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))
ALPHA = 0.3 + 0.7j
INF = np.inf

# Stand-in for MATLAB's ``A = randn(3, 3)``; any well-conditioned 3x3 real
# matrix exercises the same code path.
A = jnp.asarray(
    np.array(
        [
            [0.537667139546100, -2.258846861003648, 0.318765239858981],
            [1.833885014595086, 0.862173320368121, -1.307688296305273],
            [-2.258846861003648, 0.318765239858981, -0.433592022305684],
        ]
    )
)


class TestClassicfunMtimes:
    def test_empty_cases(self):
        # pass(1): isempty(f*[]) && isempty([]*f) && isempty(2*g) && isempty(g*2)
        f = Bndfun.from_function(jnp.sin, DOM)
        e = Bndfun.empty()
        assert (f * e).isempty()
        assert (e * f).isempty()
        assert (2.0 * e).isempty()
        assert (e * 2.0).isempty()

    def test_scalar_left_equals_right(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA * f
        g2 = f * ALPHA
        err = jnp.abs(jnp.asarray(g1(X)) - jnp.asarray(g2(X)))
        assert float(jnp.max(err)) == 0.0

    def test_scalar_multiplication_values(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA * f
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * jnp.sin(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS

    def test_zero_scalar_gives_zero(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g = 0 * f
        assert bool(jnp.all(jnp.asarray(g(X)) == 0))

    def test_array_valued_scalar_mult(self):
        # pass(5,6,7): scalar mtimes of an array-valued fun -- alpha*f == f*alpha,
        # values match, and 0*f is all-zero.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g1 = ALPHA * f
        g2 = f * ALPHA
        assert bool(jnp.all(jnp.asarray(g1(X)) == jnp.asarray(g2(X))))
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * fop(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS
        assert bool(jnp.all(jnp.asarray((0 * f)(X)) == 0))

    def test_matrix_mtimes(self):
        # pass(8): f*A mixes the columns of an array-valued fun, tol
        # 10*max(vscale)*eps.  MATLAB's '*' is Python's '@'.
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g = f @ A
        err = np.abs(np.asarray(g(X)) - np.asarray(fop(X)) @ np.asarray(A))
        assert float(np.max(err)) < 10 * float(np.max(np.asarray(g.vscale))) * EPS

    def test_dimension_error(self):
        # pass(9)/pass(10): inner dimensions must agree, in both operand
        # orders.  chebfunjax does not carry MATLAB's typed identifiers
        # (CHEBFUN:CLASSICFUN:mtimes:size / CHEBFUN:CHEBTECH:mtimes:size2),
        # so only the raise itself is asserted.
        f = Bndfun.from_function(jnp.exp, DOM)
        with pytest.raises((TypeError, ValueError)):
            jnp.asarray([[1.0, 2.0, 3.0]]) @ f
        f2 = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), DOM
        )
        with pytest.raises((TypeError, ValueError)):
            f2 @ jnp.asarray([[1.0], [2.0], [3.0]])

    def test_mtimes_two_funs_raises(self):
        # pass(11): MATLAB rejects f*g for two CLASSICFUNs ('Use .* to
        # multiply CLASSICFUN objects.').  chebfunjax's mtimes analogue '@'
        # likewise rejects a fun operand, without MATLAB's typed message.
        # (chebfunjax's '*' IS elementwise times, per Python convention.)
        f = Bndfun.from_function(jnp.exp, DOM)
        g = Bndfun.from_function(lambda x: x, DOM)
        with pytest.raises((TypeError, ValueError)):
            f @ g

    def test_uint8_multiplier(self):
        # pass(12): MATLAB rejects f*uint8(128) ('mtimes does not know how to
        # multiply a CLASSICFUN and a uint8.').
        pytest.skip(
            "chebfunjax's Bndfun.__mul__ accepts any numeric scalar (numpy "
            "integer types included) and has no double-only type check, so "
            "there is no 'mtimes does not know how to multiply a CLASSICFUN "
            "and a uint8' error path"
        )

    def test_unbndfun_matrix_mtimes(self):
        # pass(13): array-valued Unbndfun on [-Inf, -3*pi] times A,
        # tol 1e2*max(eps*vscale).
        dom = Domain((-INF, -3 * np.pi))
        op = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
        )
        f = Unbndfun.from_function(op, dom)
        g = f @ A
        x = jnp.asarray(np.linspace(-1e6, -3 * np.pi, 100))
        gexact = np.asarray(op(x)) @ np.asarray(A)
        err = float(np.max(np.abs(np.asarray(g(x)) - gexact)))
        assert err < 1e2 * EPS * float(np.max(np.asarray(g.vscale)))
