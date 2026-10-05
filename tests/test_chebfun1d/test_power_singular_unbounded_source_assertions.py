"""Source assertion port of MATLAB Chebfun test_power.m passes 32--38.

The NumPy MT19937 generator below is a deterministic Python sampling adapter;
it preserves the source's explicit call count/order, not MATLAB seedRNG values.
In particular source passes 32/33 share the first 100 samples, 34/35 share the
next 100, and 36/37/38 share the final 100 after one seedRNG(6178) call.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, passes 32--38
MATLAB APIs  : @chebfun/power.m, @singfun/power.m, @chebtech/power.m
Chebfun commit: 7574c77
Original copyright: 2017 The University of Oxford and Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
FINITE_DOM = (-2.0, 7.0)
SINGULAR_SAMPLE_DOM = (-1.9, 6.9)


def _source_sample_blocks():
    """Adapt the three sequential `rand(100,1)` calls after seedRNG(6178)."""
    rng = np.random.RandomState(6178)
    x_singular = np.diff(SINGULAR_SAMPLE_DOM)[0] * rng.rand(100) + \
        SINGULAR_SAMPLE_DOM[0]
    x_whole = 200.0 * rng.rand(100) - 100.0
    x_right = 99.0 * rng.rand(100) + 1.0
    return x_singular, x_whole, x_right


_X_SINGULAR, _X_WHOLE, _X_RIGHT = _source_sample_blocks()


def _err_inf(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected), ord=np.inf))


def _scale_inf(values):
    return float(np.linalg.norm(np.asarray(values), ord=np.inf))


def _assert_original_bound(actual, exact, multiplier, scale=None):
    error = _err_inf(actual, exact)
    bound = multiplier * EPS * (_scale_inf(exact) if scale is None else scale)
    assert np.isfinite(bound) and bound > 0
    print(f"source sample error={error:.17g} original_bound={bound:.17g}")
    assert error < bound


def _complex_real_power(values, exponent):
    """MATLAB noninteger real POWER uses the principal complex branch."""
    return np.asarray(values, dtype=np.complex128) ** exponent


def test_source_pass32_singular_endpoint_varying_sign_fractional_power():
    exponent = 0.77
    f = cj.chebfun(
        lambda x: jnp.sin(20.0 * x) * (x - FINITE_DOM[0]) ** -1.2,
        domain=FINITE_DOM,
        exps=(-1.2, 0.0),
        splitting=True,
    )
    result = f**exponent
    exact = _complex_real_power(np.sin(20.0 * _X_SINGULAR), exponent)
    exact *= (_X_SINGULAR - FINITE_DOM[0]) ** (-1.2 * exponent)
    _assert_original_bound(result(_X_SINGULAR), exact, 1e3)


def test_source_pass33_two_sided_singularity_complex_fractional_power():
    exponent = 0.69

    def op(x):
        return ((jnp.sin(20.0 * x) + 1j * jnp.cos(3.0 * x))
                * (x - FINITE_DOM[0]) ** -1.2
                * (FINITE_DOM[1] - x) ** -0.49)

    f = cj.chebfun(op, domain=FINITE_DOM, exps=(-1.2, -0.49), splitting=True)
    result = f**exponent
    exact = (np.sin(20.0 * _X_SINGULAR)
             + 1j * np.cos(3.0 * _X_SINGULAR)) ** exponent
    exact *= (_X_SINGULAR - FINITE_DOM[0]) ** (-1.2 * exponent)
    exact *= (FINITE_DOM[1] - _X_SINGULAR) ** (-0.49 * exponent)
    _assert_original_bound(result(_X_SINGULAR), exact, 1e3)
    # Additional representation control: the source casts boundary zeros to
    # Singfun before powering; a sampled value pass alone misses that gap.
    print(f"complex singular power pieces={len(result.funs)} "
          f"happy={result.ishappy} max_length={max(p.tech.n for p in result.funs)}")
    assert result.ishappy


def test_source_pass34_whole_line_bounded_positive_fractional_power():
    domain = (-jnp.inf, jnp.inf)
    exponent = 0.6
    op = lambda x: x**2 * jnp.exp(-x**2) + 2.0  # noqa: E731
    f = cj.chebfun(op, domain=domain)
    result = f**exponent
    x = _X_WHOLE
    exact = (x**2 * np.exp(-x**2) + 2.0) ** exponent
    _assert_original_bound(result(x), exact, 1e2, float(result.vscale))


def test_source_pass35_whole_line_blowup_positive_fractional_power():
    domain = (-jnp.inf, jnp.inf)
    exponent = 1.5
    op = lambda x: x**2 * (1.0 - jnp.exp(-x**2)) + 2.0  # noqa: E731
    f = cj.chebfun(op, domain=domain, exps=(2.0, 2.0))
    result = f**exponent
    x = _X_WHOLE
    exact = (x**2 * (1.0 - np.exp(-x**2)) + 2.0) ** exponent
    _assert_original_bound(result(x), exact, 1e7, float(result.vscale))


def test_source_pass36_right_halfline_integer_power():
    domain = (1.0, jnp.inf)
    exponent = 2
    op = lambda x: x * jnp.exp(-x) + 3.0  # noqa: E731
    f = cj.chebfun(op, domain=domain)
    result = f**exponent
    x = _X_RIGHT
    exact = (x * np.exp(-x) + 3.0) ** exponent
    _assert_original_bound(result(x), exact, 1e1, float(result.vscale))


def test_source_pass37_right_halfline_varying_sign_integer_power():
    domain = (1.0, jnp.inf)
    exponent = 3
    op = lambda x: 0.1 + jnp.sin(10.0 * x) / jnp.exp(x)  # noqa: E731
    f = cj.chebfun(op, domain=domain, splitting=True)
    result = f**exponent
    x = _X_RIGHT
    exact = (0.1 + np.sin(10.0 * x) / np.exp(x)) ** exponent
    _assert_original_bound(result(x), exact, 1e5, float(result.vscale))


def test_source_pass38_right_halfline_blowup_negative_integer_power():
    domain = (1.0, jnp.inf)
    exponent = -3
    op = lambda x: x * (5.0 + jnp.exp(-x**3))  # noqa: E731
    f = cj.chebfun(op, domain=domain, exps=(0.0, 1.0))
    result = f**exponent
    x = _X_RIGHT
    exact = (x * (5.0 + np.exp(-x**3))) ** exponent
    _assert_original_bound(result(x), exact, 1e2, float(result.vscale))
