"""Pinned empty and rank-zero outcomes for native subspace().

Provenance
----------
MATLAB source : @chebfun/subspace.m, @chebfun/domainCheck.m,
                @chebfun/orth.m, @chebfun/svd.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Native capture: X/subspace_empty_rank0_20261017/native/result.json
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj


def _raises_source(identifier, function):
    with pytest.raises(ValueError) as caught:
        function()
    assert identifier in str(caught.value)


def test_zero_scalar_self_subspace_matches_native_rank_zero_result():
    zero = cj.chebfun(0.0)
    assert zero.isempty() is False
    assert abs(float(cj.subspace(zero, zero))) == 0.0


@pytest.mark.parametrize("reverse", [False, True])
def test_zero_scalar_against_nonzero_preserves_native_overlap_failure(reverse):
    zero = cj.chebfun(0.0)
    x = cj.chebfun(lambda t: t)
    left, right = (x, zero) if reverse else (zero, x)
    _raises_source(
        "CHEBFUN:CHEBFUN:overlap:domains",
        lambda: cj.subspace(left, right),
    )


@pytest.mark.parametrize("factory", [cj.chebfun, lambda: cj.chebfun([])])
def test_empty_chebfun_pair_preserves_native_empty_qr_error(factory):
    empty = factory()
    _raises_source(
        "MATLAB:spdiags:InvalidSizeBFourInput",
        lambda: cj.subspace(empty, empty),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_one_empty_chebfun_fails_native_domain_check_first(reverse):
    empty = cj.chebfun()
    x = cj.chebfun(lambda t: t, domain=(0.0, 1.0))
    left, right = (x, empty) if reverse else (empty, x)
    _raises_source(
        "CHEBFUN:CHEBFUN:subspace:domain",
        lambda: cj.subspace(left, right),
    )


def test_rank_zero_route_returns_scalar_jax_value():
    zero = cj.chebfun(lambda t: jnp.zeros_like(t))
    result = cj.subspace(zero, zero)
    assert isinstance(result, jax.Array)
    assert float(result) == 0.0


def test_zero_column_in_rank_deficient_array_preserves_native_span():
    zero = cj.chebfun(0.0)
    f = cj.chebfun(lambda t: t)
    # The native probe returned roundoff-sized angles for [zero,f] and
    # [f,zero] against f, in both operand orders.
    for columns in ([zero, f], [f, zero]):
        assert abs(float(cj.subspace(columns, [f]))) < 1e-12
        assert abs(float(cj.subspace([f], columns))) < 1e-12
