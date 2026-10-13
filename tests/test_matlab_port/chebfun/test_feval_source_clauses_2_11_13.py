"""Focused qualification of MATLAB test_feval clauses 2-13.

Seeded input is the captured MATLAB seedRNG(7681) vector, not a Python
regeneration. Predicates, functions, domains and error identifiers follow the
pinned source test literally.

Provenance
----------
MATLAB source : tests/chebfun/test_feval.m, Chebfun commit 7574c77
Input capture : feval_seedRNG_7681_xr.mat, captured from MATLAB seedRNG(7681)
Capture source: feval_finite_source_20261013

The sign callbacks use the public ``cj.sign`` numeric adapter. On the pinned
CPU JAX/XLA runtime, ``jnp.sign`` classifies subnormals as zero, while the
MATLAB source test's ``sign`` distinguishes them. This adapter fixes that
qualified callback behavior; arbitrary user callbacks are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat
from scipy.special import erf

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
DATA = Path(__file__).parent / "data" / "feval_seedRNG_7681_xr.mat"


@pytest.fixture(scope="module")
def xr():
    values = loadmat(DATA)["xr"]
    assert values.shape == (1000, 1)
    return jnp.asarray(values)


def _source_identifier(exc):
    return getattr(exc, "identifier", None)


def test_clause_2_endpoint_aliases_and_erf_bound():
    f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))),
                   domain=[-1.0, 1.0], splitting=False)
    lvals = [float(f(label)) for label in ("left", "start", "-")]
    assert abs(lvals[0] - erf(-1.0)) < 10 * EPS * f.vscale
    assert lvals[0] == lvals[1] == lvals[2]


def test_clause_3_endpoint_aliases_and_erf_bound():
    f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))),
                   domain=[-1.0, 1.0], splitting=False)
    rvals = [float(f(label)) for label in ("right", "end", "+")]
    assert abs(rvals[0] - erf(1.0)) < 10 * EPS * f.vscale
    assert rvals[0] == rvals[1] == rvals[2]


def test_clause_4_bad_endpoint_source_identifier():
    f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))),
                   domain=[-1.0, 1.0], splitting=False)
    with pytest.raises(Exception) as caught:
        f("???")
    assert _source_identifier(caught.value) == "CHEBFUN:CHEBFUN:feval:strInput"


def test_clause_5_sign_value_on_source_points():
    # cj.sign uses IEEE words for real inputs because CPU XLA's jnp.sign
    # treats subnormals as zero; this callback matches MATLAB sign there.
    f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
    x = jnp.asarray([[-0.5], [0.0], [0.5]])
    err = np.asarray(f(x)) - np.asarray([[-1.0], [0.0], [1.0]])
    assert np.linalg.norm(err, ord=np.inf) < 10 * EPS * f.vscale


def test_clause_6_left_and_minus_one_sided_aliases():
    f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
    x = jnp.asarray([[-0.5], [0.0], [0.5]])
    left = np.asarray(f(x, side="left"))
    minus = np.asarray(f(x, side="-"))
    expected = np.asarray([[-1.0], [-1.0], [1.0]])
    assert np.linalg.norm(left - expected, ord=np.inf) < 10 * EPS * f.vscale
    assert np.linalg.norm(left - minus, ord=np.inf) < 10 * EPS * f.vscale


def test_clause_7_right_and_plus_one_sided_aliases():
    f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
    x = jnp.asarray([[-0.5], [0.0], [0.5]])
    right = np.asarray(f(x, side="right"))
    plus = np.asarray(f(x, side="+"))
    expected = np.asarray([[-1.0], [1.0], [1.0]])
    assert np.linalg.norm(right - expected, ord=np.inf) < 10 * EPS * f.vscale
    assert np.linalg.norm(right - plus, ord=np.inf) < 10 * EPS * f.vscale


def test_clause_8_bad_side_string_source_identifier():
    f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
    x = jnp.asarray([[-0.5], [0.0], [0.5]])
    with pytest.raises(Exception) as caught:
        f(x, side="???")
    assert _source_identifier(caught.value) == "CHEBFUN:CHEBFUN:feval:leftRightChar"


def test_clause_9_empty_side_source_identifier():
    f = cj.chebfun(cj.sign, domain=[-1.0, 1.0], splitting=True)
    x = jnp.asarray([[-0.5], [0.0], [0.5]])
    with pytest.raises(Exception) as caught:
        f(x, side=[])
    assert _source_identifier(caught.value) == "CHEBFUN:CHEBFUN:feval:leftRight"


def test_clause_10_exp_minus_one_on_captured_matlab_points(xr):
    exact = lambda x: jnp.exp(x) - 1
    f = cj.chebfun(exact, domain=[-1.0, 1.0], splitting=False)
    err = np.asarray(f(xr)) - np.asarray(exact(xr))
    assert np.linalg.norm(err, ord=np.inf) < 10 * EPS * f.vscale


def test_clause_11_abs_over_quadratic_on_captured_matlab_points(xr):
    exact = lambda x: jnp.abs(x) / (1 + x**2)
    f = cj.chebfun(exact, domain=[-2.0, 7.0], splitting=True)
    x = 4.5 * xr + 2.5
    err = np.asarray(f(x)) - np.asarray(exact(x))
    assert np.linalg.norm(err, ord=np.inf) < 100 * EPS * f.vscale


def test_clause_12_high_frequency_cosine_on_captured_matlab_points(xr):
    exact = lambda x: jnp.cos(1e4 * x)
    f = cj.chebfun(exact, domain=[1.0, 5.0], splitting=False)
    x = 2 * xr + 3
    err = np.asarray(f(x)) - np.asarray(exact(x))
    assert np.linalg.norm(err, ord=np.inf) < 1e5 * EPS * f.vscale


def test_clause_13_complex_sinh_on_captured_matlab_points(xr):
    z = np.exp(2 * np.pi * 1j / 6)
    exact = lambda t: jnp.sinh(t * z)
    f = cj.chebfun(exact, domain=[-1.0, 1.0], splitting=False)
    err = np.asarray(f(xr)) - np.asarray(exact(xr))
    assert np.linalg.norm(err, ord=np.inf) < 10 * EPS * f.vscale


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_numeric_sign_eager_and_jit_preserve_real_ieee_sign(dtype):
    if dtype == jnp.bfloat16:
        raw = np.asarray([0x0001, 0x8001, 0x0000, 0x8000, 0x3F80, 0xBF80,
                          0x7F80, 0xFF80, 0x7FC1],
                         dtype=np.uint16)
        values = jax.lax.bitcast_convert_type(jnp.asarray(raw), dtype)
        expected = np.asarray([1, -1, 0, 0, 1, -1, 1, -1, np.nan], dtype=np.float32)
    elif dtype == jnp.float16:
        raw = np.asarray([0x0001, 0x8001, 0x0000, 0x8000, 0x3C00, 0xBC00,
                          0x7C00, 0xFC00, 0x7E01],
                         dtype=np.uint16)
        values = jax.lax.bitcast_convert_type(jnp.asarray(raw), dtype)
        expected = np.asarray([1, -1, 0, 0, 1, -1, 1, -1, np.nan], dtype=np.float32)
    elif dtype == jnp.float32:
        raw = np.asarray([1, 0x80000001, 0, 0x80000000, 0x3F800000, 0xBF800000,
                          0x7F800000, 0xFF800000, 0x7FC00001],
                         dtype=np.uint32)
        values = jax.lax.bitcast_convert_type(jnp.asarray(raw), dtype)
        expected = np.asarray([1, -1, 0, 0, 1, -1, 1, -1, np.nan], dtype=np.float32)
    else:
        raw = np.asarray([1, 0x8000000000000001, 0, 0x8000000000000000,
                          0x3FF0000000000000, 0xBFF0000000000000,
                          0x7FF0000000000000, 0xFFF0000000000000,
                          0x7FF8000000000001], dtype=np.uint64)
        values = jax.lax.bitcast_convert_type(jnp.asarray(raw), dtype)
        expected = np.asarray([1, -1, 0, 0, 1, -1, 1, -1, np.nan], dtype=np.float64)

    for result in (cj.sign(values), jax.jit(cj.sign)(values)):
        assert result.dtype == dtype
        np.testing.assert_array_equal(np.asarray(result, dtype=np.float32), expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_numeric_sign_fused_with_arithmetic_preserves_subnormal_sign(dtype):
    if dtype == jnp.bfloat16:
        raw = np.asarray([0x0001, 0x007F, 0x8001, 0x807F, 0x0000, 0x8000,
                          0x0080, 0x8080], dtype=np.uint16)
    elif dtype == jnp.float16:
        raw = np.asarray([0x0001, 0x03FF, 0x8001, 0x83FF, 0x0000, 0x8000,
                          0x0400, 0x8400], dtype=np.uint16)
    elif dtype == jnp.float32:
        raw = np.asarray([0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF,
                          0x00000000, 0x80000000, 0x00800000, 0x80800000],
                         dtype=np.uint32)
    else:
        raw = np.asarray([0x0000000000000001, 0x000FFFFFFFFFFFFF,
                          0x8000000000000001, 0x800FFFFFFFFFFFFF,
                          0x0000000000000000, 0x8000000000000000,
                          0x0010000000000000, 0x8010000000000000], dtype=np.uint64)
    values = jax.lax.bitcast_convert_type(jnp.asarray(raw), dtype)
    expected = np.asarray([2, 2, 0, 0, 1, 1, 2, 0], dtype=np.float32)
    fused = jax.jit(lambda x: cj.sign(x) + (x + jnp.ones_like(x)))
    np.testing.assert_array_equal(np.asarray(fused(values), dtype=np.float32), expected)


def test_numeric_sign_complex_delegates_and_chebfun_dispatch_is_preserved():
    z = jnp.asarray([1 + 2j, -3 + 4j], dtype=jnp.complex128)
    np.testing.assert_array_equal(np.asarray(cj.sign(z)), np.asarray(jnp.sign(z)))
    f = cj.chebfun(lambda x: x, domain=[-1.0, 1.0], splitting=False)
    assert isinstance(cj.sign(f), cj.Chebfun)
    tech = f.funs[0].tech
    assert isinstance(cj.sign(tech), type(tech))


def test_chebfun_sign_preserves_subnormal_breakpoint_point_values():
    tiny = jax.lax.bitcast_convert_type(jnp.asarray(1, dtype=jnp.uint64), jnp.float64)
    f = cj.chebfun(lambda x: jnp.ones_like(x), domain=[-1.0, 1.0], splitting=False)
    f = f.set_point_values(jnp.asarray([-tiny, tiny]))
    np.testing.assert_array_equal(np.asarray(f.sign().point_values), [-1.0, 1.0])


def test_chebfun_sign_preserves_complex_and_integer_point_values():
    complex_f = cj.chebfun(lambda x: (1 + 1j) * x, domain=(-1.0, 0.0, 1.0))
    complex_sign = complex_f.sign()
    np.testing.assert_array_equal(
        np.asarray(complex_sign.point_values),
        np.asarray(jnp.sign(complex_f.point_values)),
    )

    real_f = cj.chebfun(lambda x: jnp.ones_like(x), domain=[-1.0, 1.0], splitting=False)
    integer_values = real_f.set_point_values(jnp.asarray([-2, 3], dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(integer_values.sign().point_values), [-1, 1])
