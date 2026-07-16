"""Port of MATLAB Chebfun tests/trigtech/test_mtimes.m (Opus 4.8).

mtimes (*) with a scalar is elementwise scaling (commutes).  MATLAB's matrix
multiply ``f*A`` maps to chebfunjax ``f @ A`` (matmul); array-valued trigtechs
are now supported, so the array scalar (pass 5-7) and matrix (pass 8-10) cases
are real assertions (FIXED, Fable 5, Big-Three array-valued epic).

The MATLAB size/type error paths (pass 11-13) do raise in chebfunjax, but as a
generic ``TypeError`` rather than MATLAB's typed identifiers/messages
(``CHEBFUN:TRIGTECH:mtimes:size`` etc.), so those are ported as assert-raises.
pass 1 (empty-argument arithmetic) and pass 14 (``TRIGTECH*uint8`` message) have
no chebfunjax analogue and stay xfail with precise reasons.

Provenance
----------
MATLAB source : tests/trigtech/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))
rng = np.random.default_rng(6178)
ALPHA = complex(rng.standard_normal(), rng.standard_normal())


def _arr3():
    return Trigtech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.cos(20 * jnp.pi * x), jnp.cos(jnp.sin(jnp.pi * x))],
            axis=-1,
        )
    )


def _carr3():
    return Trigtech.from_function(
        lambda x: jnp.stack(
            [jnp.exp(1j * 11 * jnp.pi * x), jnp.cos(20 * jnp.pi * x), jnp.cos(jnp.sin(jnp.pi * x))],
            axis=-1,
        )
    )


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechMtimes:
    def test_scalar_mult_commutes(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g1, g2 = ALPHA * f, f * ALPHA
        assert bool(jnp.all(g1.coeffs == g2.coeffs))

    def test_scalar_mult_value(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g1 = ALPHA * f
        exact = ALPHA * jnp.sin(10 * jnp.pi * X)
        assert _ninf(g1(X) - exact) < 100 * g1.vscale * EPS

    def test_mult_by_zero(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g = 0.0 * f
        assert bool(jnp.all(g.coeffs == 0))

    @pytest.mark.xfail(reason="chebfunjax lacks empty-argument arithmetic")
    def test_empty_arguments(self):
        raise AssertionError("empty trigtech arithmetic not implemented")

    def test_array_scalar_mult_commutes(self):
        # pass(5): alpha*f == f*alpha for array-valued f.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _arr3()
        assert bool(jnp.all((ALPHA * f).coeffs == (f * ALPHA).coeffs))

    def test_array_scalar_mult_value(self):
        # pass(6): (alpha*f)(x) == alpha*[...].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _arr3()
        g1 = ALPHA * f
        exact = ALPHA * jnp.stack(
            [jnp.sin(10 * jnp.pi * X), jnp.cos(20 * jnp.pi * X), jnp.cos(jnp.sin(jnp.pi * X))],
            axis=-1,
        )
        assert _ninf(g1(X) - exact) < 100 * g1.vscale * EPS

    def test_array_mult_by_zero(self):
        # pass(7): 0*f has all-zero coeffs for array-valued f.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _arr3()
        assert bool(jnp.all((0.0 * f).coeffs == 0))

    def test_matrix_mult_real(self):
        # pass(8): f*A (real A) == [...]*A.  MATLAB f*A maps to f @ A.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _arr3()
        A = jnp.asarray(np.random.default_rng(1).standard_normal((3, 3)))
        g = f @ A
        gex = jnp.stack(
            [jnp.sin(10 * jnp.pi * X), jnp.cos(20 * jnp.pi * X), jnp.cos(jnp.sin(jnp.pi * X))],
            axis=-1,
        ) @ A
        assert _ninf(g(X) - gex) < 100 * g.vscale * EPS

    def test_matrix_mult_complex_trigtech(self):
        # pass(9): f*A with a complex-valued f and real A.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _carr3()
        A = jnp.asarray(np.random.default_rng(2).standard_normal((3, 3)))
        g = f @ A
        gex = jnp.stack(
            [jnp.exp(1j * 11 * jnp.pi * X), jnp.cos(20 * jnp.pi * X), jnp.cos(jnp.sin(jnp.pi * X))],
            axis=-1,
        ) @ A
        assert _ninf(g(X) - gex) < 100 * g.vscale * EPS

    def test_matrix_mult_complex_matrix(self):
        # pass(10): f*A with a complex-valued f and complex A.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _carr3()
        r = np.random.default_rng(3)
        A = jnp.asarray(r.standard_normal((3, 3)) + 1j * r.standard_normal((3, 3)))
        g = f @ A
        gex = jnp.stack(
            [jnp.exp(1j * 11 * jnp.pi * X), jnp.cos(20 * jnp.pi * X), jnp.cos(jnp.sin(jnp.pi * X))],
            axis=-1,
        ) @ A
        assert _ninf(g(X) - gex) < 100 * g.vscale * EPS

    def test_error_nonscalar_double_times_trigtech(self):
        # pass(11): [1 2 3]*f (non-scalar double times trigtech) is a dimension
        # mismatch.  FIXED (Fable 5, Big-Three array-valued epic): chebfunjax
        # raises a generic TypeError (no MATLAB 'mtimes:size' identifier).
        f = Trigtech.from_function(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        with pytest.raises(Exception):
            jnp.array([1.0, 2.0, 3.0]) @ f

    def test_error_trigtech_times_mismatched_double(self):
        # pass(12): f*[1;2;3] with a 2-column f is a dimension mismatch.
        # FIXED (Fable 5, Big-Three array-valued epic): raises (see pass 11).
        f = Trigtech.from_function(
            lambda x: jnp.stack([jnp.sin(10 * jnp.pi * x), jnp.cos(20 * jnp.pi * x)], axis=-1)
        )
        with pytest.raises(Exception):
            f @ jnp.array([1.0, 2.0, 3.0])

    def test_error_trigtech_times_trigtech(self):
        # pass(13): f*g of two trigtechs via mtimes (@) is forbidden.
        # FIXED (Fable 5, Big-Three array-valued epic): chebfunjax raises a
        # TypeError (MATLAB emits a "Use .* to multiply" message instead).
        f = Trigtech.from_function(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = Trigtech.from_function(lambda x: jnp.cos(20 * jnp.pi * x))
        with pytest.raises(Exception):
            f @ g

    @pytest.mark.xfail(
        reason="chebfunjax has no MATLAB-style typed message for TRIGTECH*<unknown "
        "type>; matmul against an unsupported operand simply raises TypeError"
    )
    def test_error_unknown_type(self):
        # pass(14): MATLAB emits a specific 'mtimes does not know how to multiply
        # a TRIGTECH and a uint8' message; chebfunjax has no such typed message.
        raise NotImplementedError("no typed mtimes message for unknown operands")
