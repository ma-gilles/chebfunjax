"""Port of MATLAB Chebfun tests/trigtech/test_rdivide.m (Opus 4.8).

Pointwise division (./).  Division of two resolved real functions and
scalar/function division with a real result match direct evaluation.
Array-valued division works, and a column-count mismatch (2-col ./ [1 2 3])
is rejected.  Remaining gaps: complex-scalar division (is_real not cleared,
dropping the imaginary part -- affects the scalar and array cases), division
by zero (needs isnan(), and f./0 yields inf not all-NaN coeffs), and the
column-vector size check (silently broadcasts instead of raising).

Provenance
----------
MATLAB source : tests/trigtech/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 10, endpoint=False))
XX = jnp.asarray(np.linspace(-1.0, 1.0, 10))


# MATLAB's arbitrary constants (seedRNG(6178) draws).
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechRdivide:
    def test_scalar_over_function(self):
        # 2 ./ exp(cos(pi x)) : real scalar over real function.
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = 2.0 / f
        exact = 2.0 / jnp.exp(jnp.cos(jnp.pi * X))
        assert _ninf(g(X) - exact) < 100 * g.vscale * EPS

    def test_function_over_function_expcos(self):
        g = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)))
        f = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)) - 1)
        h = f / g
        exact = (jnp.exp(jnp.cos(20 * jnp.pi * X)) - 1) / jnp.exp(jnp.cos(20 * jnp.pi * X))
        assert _ninf(h(X) - exact) < 1e3 * h.vscale * EPS

    def test_function_over_function_cos(self):
        g = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)))
        f = _tt(lambda x: jnp.cos(1e3 * jnp.pi * x))
        h = f / g
        exact = jnp.cos(1e3 * jnp.pi * X) / jnp.exp(jnp.cos(20 * jnp.pi * X))
        assert _ninf(h(X) - exact) < 1e3 * h.vscale * EPS

    def test_direct_construction_matches(self):
        # sin(10 pi x) ./ exp(cos(pi x)) : real.
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        h1 = f / g
        h2 = _tt(lambda x: jnp.sin(10 * jnp.pi * x) / jnp.exp(jnp.cos(jnp.pi * x)))
        assert _ninf(h1(XX) - h2(XX)) < 100 * EPS

    # FIXED (Fable 5): division by a complex scalar clears is_real,
    # so pass(1) ports at MATLAB's tolerance (100*vscale*eps).
    def test_scalar_division_complex(self):
        fop = lambda x: jnp.sin(10 * jnp.pi * x)
        f = _tt(fop)
        g = f / ALPHA
        assert _ninf(g(X) - fop(X) / ALPHA) < 100 * g.vscale * EPS

    # FIXED (Fable 5): __rtruediv__ keeps a complex numerator, so
    # pass(7) ports at MATLAB's tolerance.
    def test_complex_scalar_over_function(self):
        import warnings

        fop = lambda x: jnp.exp(jnp.cos(jnp.pi * x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = _tt(fop)
            g = ALPHA / f
        assert _ninf(g(X) - ALPHA / fop(X)) < 100 * g.vscale * EPS

    def test_scalar_division_by_zero_is_nan(self):
        # pass(2): f ./ 0 -> isnan(g).  FIXED (Fable 5): Trigtech.isnan()
        # ports @trigtech/isnan.m; f/0 makes NaN coeffs -> NaN values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
            g = f / 0.0
        assert g.isnan()

    # FIXED (Fable 5): complex scalars clear is_real, so pass(3)
    # ports directly.
    def test_array_scalar_division(self):
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(20 * jnp.pi * x)],
            axis=-1)
        f = _tt(fop)
        g = f / ALPHA
        assert _ninf(g(X) - fop(X) / ALPHA) < 100 * g.vscale * EPS

    def test_array_division_by_zero(self):
        # pass(4): array-valued f ./ 0 -> isnan(g).  FIXED (Fable 5).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = _tt(lambda x: jnp.stack(
                [jnp.sin(10 * jnp.pi * x), jnp.sin(20 * jnp.pi * x)], axis=-1))
            g = f / 0.0
        assert g.isnan()

    # FIXED (Fable 5): a complex row divisor clears is_real, so
    # pass(5) ports at MATLAB's absolute 1e3*eps tolerance.
    def test_array_division_by_row(self):
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(20 * jnp.pi * x)],
            axis=-1)
        f = _tt(fop)
        g = f / jnp.asarray([ALPHA, BETA])
        exact = jnp.stack(
            [jnp.sin(10 * jnp.pi * X) / ALPHA,
             jnp.sin(20 * jnp.pi * X) / BETA], axis=-1)
        assert _ninf(g(X) - exact) < 1e3 * EPS

    @pytest.mark.xfail(
        reason="isnan() now exists, but MATLAB divides VALUES by the row [alpha 0] "
        "(the zero column's values become 0/0=NaN and Inf, whose transform is all-NaN "
        "coeffs); chebfunjax rdivide divides COEFFS, so the zero column keeps Inf (not "
        "all-NaN) coeffs and the per-column all(isnan(coeffs)) pattern differs"
    )
    def test_array_division_by_row_with_zero(self):
        raise AssertionError("value-space rdivide NaN-pattern not reproducible")

    @pytest.mark.xfail(
        reason="chebfunjax does not validate division by a column vector [1;2]; it silently "
        "broadcasts instead of raising CHEBFUN:TRIGTECH:rdivide:size"
    )
    def test_size_error_column_vector(self):
        raise AssertionError("column-vector size check not implemented")

    def test_size_error_row_mismatch(self):
        # pass(11): 2-column f ./ [1 2 3] is rejected with a dimension error.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        with pytest.raises((ValueError, TypeError)):
            _ = f / jnp.array([1.0, 2.0, 3.0])
