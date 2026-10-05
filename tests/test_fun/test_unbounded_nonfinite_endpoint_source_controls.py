"""Independent controls for source unbounded endpoint construction.

Provenance
----------
MATLAB source : @unbndfun/unbndfun.m, @onefun/onefun.m,
                @chebtech/populate.m, @chebtech/extrapolate.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original: Copyright 2017 by The University of Oxford and Chebfun Developers.

These analytic controls supplement the original constructor/power assertions;
their 100*eps bounds are independent binary64 checks. Adaptive endpoint bounds
use the returned function vscale, following the source scale convention. Fixed
construction checks
endpoint limits, not unresolved interior accuracy. Adaptive controls also check
finite physical points and absence of an exhausted refinement warning. The
Python callable accepts arrays, including singleton endpoint probe arrays.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)


def _end_values(fun, exact, scale=None):
    actual = np.asarray(fun.onefun(jnp.asarray([-1.0, 1.0])))
    expected = np.asarray(exact)
    assert actual.shape == expected.shape
    bound_scale = np.max(np.abs(expected)) if scale is None else scale
    assert np.max(np.abs(actual - expected)) < 100 * EPS * bound_scale


def test_fixed_whole_line_removable_nan_samples_keep_nonzero_constant_limit():
    fun = Unbndfun.from_function(
        lambda x: 2.0 * x / x,
        Domain((-jnp.inf, jnp.inf)), n=65)
    _end_values(fun, [2.0, 2.0])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_fixed_half_line_nan_endpoint_keeps_offset_limit(side):
    if side == 'right':
        dom = Domain((0.0, jnp.inf))
        def op(x):
            return x * jnp.exp(-x) + 3.0
    else:
        dom = Domain((-jnp.inf, 0.0))
        def op(x):
            return x * jnp.exp(x) + 3.0
    fun = Unbndfun.from_function(op, dom, n=65)
    _end_values(fun, [3.0, 3.0])


def test_fixed_complex_array_extrapolates_entire_partly_nonfinite_row():
    def op(x):
        # Only the first column is NaN at +inf; the second remains finite.
        return jnp.stack([x * jnp.exp(-x) + 3.0,
                          jnp.exp(-x) + 1j], axis=-1)
    fun = Unbndfun.from_function(op, Domain((0.0, jnp.inf)), n=65)
    _end_values(fun, [[3.0, 1.0+1j], [3.0, 1j]])


@pytest.mark.parametrize('n', [17, None])
def test_all_nan_callback_raises_source_extrapolation_error(n):
    with pytest.raises(ValueError, match=r'Too many NaNs/Infs to handle\.'):
        Unbndfun.from_function(lambda x: jnp.full_like(x, jnp.nan),
                              Domain((-jnp.inf, jnp.inf)), n=n)


@pytest.mark.parametrize('side', ['whole', 'left', 'right'])
def test_adaptive_finite_offset_limits_resolve_without_exhausting_refinement(side):
    if side == 'whole':
        dom = Domain((-jnp.inf, jnp.inf))
        def op(x):
            return x**2 * jnp.exp(-x**2) + 2.0
        points = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        exact = points**2 * np.exp(-points**2) + 2.0
        ends = [2.0, 2.0]
    elif side == 'right':
        dom = Domain((0.0, jnp.inf))
        def op(x):
            return x * jnp.exp(-x) + 3.0
        points = np.array([0.0, 0.25, 1.0, 4.0, 30.0])
        exact = points * np.exp(-points) + 3.0
        ends = [3.0, 3.0]
    else:
        dom = Domain((-jnp.inf, 0.0))
        def op(x):
            return x * jnp.exp(x) + 3.0
        points = np.array([-30.0, -4.0, -1.0, -0.25, 0.0])
        exact = points * np.exp(points) + 3.0
        ends = [3.0, 3.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fun = Unbndfun.from_function(op, dom)
    assert fun.ishappy
    _end_values(fun, ends, scale=float(fun.vscale))
    np.testing.assert_allclose(np.asarray(fun(jnp.asarray(points))), exact,
                               rtol=100 * EPS, atol=0.0)
    assert not any('did not converge' in str(w.message) for w in caught)


@pytest.mark.parametrize('side', ['whole', 'left', 'right'])
@pytest.mark.parametrize('n', [None, 65])
def test_endpoint_infinity_selects_source_singular_constructor(side, n):
    if side == 'whole':
        dom = Domain((-jnp.inf, jnp.inf))
        def op(x):
            return x**2 + 2.0
        points = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        exact = points**2 + 2.0
        exponents = [-2.0, -2.0]
    elif side == 'right':
        dom = Domain((0.0, jnp.inf))
        def op(x):
            return x + 3.0
        points = np.array([0.0, 0.25, 1.0, 4.0, 30.0])
        exact = points + 3.0
        exponents = [0.0, -1.0]
    else:
        dom = Domain((-jnp.inf, 0.0))
        def op(x):
            return -x + 3.0
        points = np.array([-30.0, -4.0, -1.0, -0.25, 0.0])
        exact = -points + 3.0
        exponents = [-1.0, 0.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fun = Unbndfun.from_function(op, dom, n=n)
    assert isinstance(fun.onefun, Singfun)
    assert fun.ishappy
    np.testing.assert_allclose(fun.onefun.exponents, exponents, rtol=0.0, atol=20 * EPS)
    np.testing.assert_allclose(np.asarray(fun(jnp.asarray(points))), exact,
                               rtol=100 * EPS, atol=0.0)
    assert not any('did not converge' in str(w.message) for w in caught)
