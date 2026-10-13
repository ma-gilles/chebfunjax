"""Port of MATLAB Chebfun tests/chebfun/test_feval.m (Fable 5).

String endpoint syntaxes ('left','start','-') are skipped; numeric
evaluation incl. endpoint one-sided values ported.

Provenance
----------
MATLAB source : tests/chebfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import (
    Chebfun,
    _exact_breakpoint_equal,
    _float32_to_float64_bits,
    _Piece,
)
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunFeval:
    def test_empty_input_shape(self):
        f = cj.chebfun(lambda x: x)
        fx = f(jnp.zeros((0,)))
        assert np.asarray(fx).shape == (0,)

    def test_string_endpoint_syntax(self):
        # MATLAB pass(2,3): feval(f, 'left'/'start'/'-') and
        # feval(f, 'right'/'end'/'+').
        f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))))
        lvals = [float(f(s)) for s in ("left", "start", "-")]
        rvals = [float(f(s)) for s in ("right", "end", "+")]
        assert abs(lvals[0] - erf(-1.0)) < 10 * EPS * f.vscale
        assert lvals[0] == lvals[1] == lvals[2]
        assert abs(rvals[0] - erf(1.0)) < 10 * EPS * f.vscale
        assert rvals[0] == rvals[1] == rvals[2]
        with pytest.raises(ValueError):
            f("middle")

    def test_side_string_one_sided_limits(self):
        # MATLAB pass(6,7): feval(f, x, 'left'/'-') and 'right'/'+' at a
        # jump.  f = sign under splitting; x = [-0.5, 0, 0.5].
        # Use native numeric sign semantics: CPU jnp.sign flushes subnormals
        # during adaptive jump localization and constructs a different split.
        f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
        x = jnp.asarray([-0.5, 0.0, 0.5])
        for s in ("left", "-"):
            got = np.asarray(f(x, side=s))
            assert np.allclose(got, [-1.0, -1.0, 1.0])
        for s in ("right", "+"):
            got = np.asarray(f(x, side=s))
            assert np.allclose(got, [-1.0, 1.0, 1.0])

    def test_endpoint_values(self):
        f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))))
        lval = float(f(jnp.asarray(-1.0)))
        rval = float(f(jnp.asarray(1.0)))
        assert abs(lval - erf(-1.0)) < 10 * EPS * f.vscale
        assert abs(rval - erf(1.0)) < 10 * EPS * f.vscale

    def test_vectorized_evaluation(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        err = jnp.abs(f(XR) - jnp.exp(XR) * jnp.sin(3 * XR))
        assert float(jnp.max(err)) < 100 * f.vscale * EPS

    def test_matrix_shaped_input(self):
        f = cj.chebfun(lambda x: x ** 2)
        xm = jnp.asarray(np.linspace(-1, 1, 12).reshape(3, 4))
        out = f(xm)
        assert np.asarray(out).shape == (3, 4)
        assert float(jnp.max(jnp.abs(out - xm ** 2))) < 100 * EPS

    def test_complex_points_just_outside_domain(self):
        # MATLAB test_feval.m pass(20): evaluate a real chebfun at complex
        # points (and points just outside the domain), comparing to the
        # analytic function.  f_exact = cos(x - 0.2) on [-1, 1].
        def f_exact(x):
            return np.cos(x - 0.2)

        f = cj.chebfun(lambda x: jnp.cos(x - 0.2), domain=[-1, 1])
        x = np.array([-1 - 1e-6, 1e-6 + 1e-6j, 1 + 1e-6])
        err = np.asarray(f(jnp.asarray(x))) - f_exact(x)
        assert float(np.max(np.abs(err))) < 10 * EPS * f.vscale

    def test_complex_valued_chebfun_at_real_points(self):
        # MATLAB test_feval.m pass(13): a complex-VALUED chebfun (complex
        # coefficients) evaluated at real points.  f_exact = sinh(t*z),
        # z = exp(2*pi*1i/6).
        z = np.exp(2 * np.pi * 1j / 6)

        def f_exact(t):
            return np.sinh(t * z)

        f = cj.chebfun(lambda t: jnp.sinh(t * z), domain=[-1, 1])
        x = np.asarray(XR)
        err = np.asarray(f(jnp.asarray(x))) - f_exact(x)
        assert float(np.max(np.abs(err))) < 10 * EPS * f.vscale

    def test_explicit_point_values_at_breakpoints_eager_and_jit(self):
        # MATLAB @chebfun/feval.m substitutes pointValues at exact members
        # of the domain breakpoint vector (test_feval.m passes 28-30).
        tiny_word = np.uint64(0x0008000000000000)  # 2**-1023, subnormal
        middle = float(np.asarray(tiny_word).view(np.float64))
        domain = Domain((-1.0, middle, 1.0))
        base = Chebfun(
            funs=[_Piece.from_coeffs(jnp.asarray([1.0]), -1.0, middle),
                  _Piece.from_coeffs(jnp.asarray([2.0]), middle, 1.0)],
            domain=domain)
        point_values = jnp.asarray([-3.0, 7.0, 4.0])
        f = base.set_point_values(point_values)
        breakpoint_words = np.asarray(f.domain.breakpoints, dtype=np.float64).view(np.uint64)
        x = jax.lax.bitcast_convert_type(
            jnp.asarray(breakpoint_words, dtype=jnp.uint64), jnp.float64)
        expected = np.asarray(point_values)

        np.testing.assert_array_equal(np.asarray(f(x)), expected)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lambda z: f(z))(x)), expected)
        # The exact subnormal breakpoint is distinct from x=0.
        assert float(f(0.0)) == float(base(0.0))
        assert float(jax.jit(lambda z: f(z))(jnp.asarray(0.0))) == \
            float(jax.jit(lambda z: base(z))(jnp.asarray(0.0)))
        assert float(f(middle)) == 7.0
        assert float(jax.jit(lambda z: f(z))(x[1])) == 7.0

    def test_explicit_point_values_do_not_snap_nearby_complex_points(self):
        domain = Domain((-1.0, 0.0, 1.0))
        base = Chebfun(
            funs=[_Piece.from_coeffs(jnp.asarray([1.0]), -1.0, 0.0),
                  _Piece.from_coeffs(jnp.asarray([2.0]), 0.0, 1.0)],
            domain=domain)
        f = base.set_point_values(jnp.asarray([-3.0, 7.0, 4.0]))
        # MATLAB uses exact ismember(x, dom): nonzero imaginary offsets are
        # not breakpoint hits, even when real(x) is an exact breakpoint.
        x = jnp.asarray([-1.0 + 1e-8j, 1e-8j, 1.0 + 1e-8j])
        np.testing.assert_allclose(np.asarray(f(x)), np.asarray(base(x)),
                                   rtol=0, atol=0)
        np.testing.assert_allclose(
            np.asarray(jax.jit(lambda z: f(z))(x)),
            np.asarray(jax.jit(lambda z: base(z))(x)), rtol=0, atol=0)
        tiny = np.nextafter(0.0, 1.0)
        middle = 0.0
        subnormal_imaginary = jnp.asarray([complex(middle, tiny)])
        np.testing.assert_allclose(
            np.asarray(jax.jit(lambda z: f(z))(subnormal_imaginary)),
            np.asarray(jax.jit(lambda z: base(z))(subnormal_imaginary)),
            rtol=0, atol=0)

    def test_explicit_point_values_keep_transposed_array_orientation(self):
        base = cj.chebfun(
            lambda x: jnp.stack([x, 1 + x], axis=-1), domain=[-1.0, 1.0])
        point_values = jnp.asarray([[10.0, -10.0], [20.0, -20.0]])
        f = base.set_point_values(point_values).transpose()
        x = jnp.asarray([-1.0, 1.0])
        expected = np.asarray(point_values).T

        assert np.asarray(f(x)).shape == (2, 2)
        np.testing.assert_array_equal(np.asarray(f(x)), expected)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lambda z: f(z))(x)), expected)

    def test_exact_breakpoint_equality_preserves_ieee_cases(self):
        # Compare IEEE words so subnormals survive CPU XLA float comparisons.
        real_a_bits = jnp.asarray([
            0x8000000000000000,  # -0
            0x0000000000000001,  # smallest positive subnormal
            0x0000000000000001,
            0x7ff0000000000000,  # +inf
            0xfff0000000000000,  # -inf
            0x7ff8000000000001,  # NaN
        ], dtype=jnp.uint64)
        real_b_bits = jnp.asarray([
            0x0000000000000000,  # +0 compares equal to -0
            0x0000000000000001,
            0x0000000000000002,  # adjacent but unequal subnormal
            0x7ff0000000000000,
            0x7ff0000000000000,
            0x7ff8000000000001,  # NaN never compares equal
        ], dtype=jnp.uint64)
        real_a = jax.lax.bitcast_convert_type(real_a_bits, jnp.float64)
        real_b = jax.lax.bitcast_convert_type(real_b_bits, jnp.float64)
        expected = np.asarray([True, True, False, True, False, False])
        np.testing.assert_array_equal(
            np.asarray(jax.jit(_exact_breakpoint_equal)(real_a, real_b)),
            expected)

        # Complex equality also requires exact zero imaginary part; either
        # signed zero is accepted and a subnormal imaginary part is not.
        complex_values = jnp.asarray([
            complex(0.0, 0.0), complex(-0.0, -0.0),
            complex(0.0, np.nextafter(0.0, 1.0)),
        ], dtype=jnp.complex128)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(_exact_breakpoint_equal)(
                complex_values, jnp.asarray(0.0))),
            np.asarray([True, True, False]))

        f32 = jnp.asarray([0.0, 1.0e-40, 1.1e-40], dtype=jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(_exact_breakpoint_equal)(
                f32, jnp.asarray(1.0e-40, dtype=jnp.float32))),
            np.asarray([False, True, False]))

        # Validate binary32-to-binary64 widening independently across zero,
        # both subnormal boundaries, normal boundaries, infinities and NaNs.
        f32_words = np.asarray([
            0x00000000, 0x80000000, 0x00000001, 0x007fffff,
            0x00800000, 0x3f800000, 0x7f7fffff, 0x7f800000,
            0xff800000, 0x7fc00001,
        ], dtype=np.uint32)
        f32_values = jax.lax.bitcast_convert_type(
            jnp.asarray(f32_words), jnp.float32)
        widened_bits = np.asarray(jax.jit(_float32_to_float64_bits)(f32_values))
        f64_expected = f32_words.view(np.float32).astype(np.float64).view(np.uint64)
        np.testing.assert_array_equal(widened_bits[:-1], f64_expected[:-1])
        np.testing.assert_array_equal(
            np.asarray(jax.jit(_exact_breakpoint_equal)(
                f32_values, f32_values)),
            np.asarray([True, True, True, True, True, True, True, True,
                        True, False]))

        # Mixed float32 and complex64 query arrays are compared before their
        # promoted evaluation values can flush a nonzero subnormal to zero.
        zero_domain = Domain((-1.0, 0.0, 1.0))
        zero_base = Chebfun(
            funs=[_Piece.from_coeffs(jnp.asarray([1.0]), -1.0, 0.0),
                  _Piece.from_coeffs(jnp.asarray([2.0]), 0.0, 1.0)],
            domain=zero_domain)
        zero_f = zero_base.set_point_values(jnp.asarray([-3.0, 7.0, 4.0]))
        f32_queries = jax.lax.bitcast_convert_type(
            jnp.asarray([0x00000001, 0x80000001, 0x00000000, 0x80000000],
                        dtype=jnp.uint32), jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lambda z: zero_f(z))(f32_queries[:2])),
            np.asarray([2.0, 2.0]))
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lambda z: zero_f(z))(f32_queries[2:])),
            np.asarray([7.0, 7.0]))

        complex64_query = jax.lax.complex(
            jnp.asarray([0.0], dtype=jnp.float32), f32_queries[:1])
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lambda z: zero_f(z))(complex64_query)),
            np.asarray([2.0]))
