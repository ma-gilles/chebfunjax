"""Source-backed controls for bounded Chebfun/Chebfun division compatibility.

Provenance
----------
MATLAB source: @chebfun/rdivide.m, @chebfun/domainCheck.m,
    @chebfun/hscale.m, @chebfun/overlap.m, @chebfun/tweakDomain.m,
    tests/chebfun/test_rdivide.m (commit 7574c77680d7e82b79626300bf255498271a72df).
Literal MATLAB error assertions14/15 are included below; their identifiers
and original fixtures are preserved. Existing numerical bounds are unchanged.
These controls exercise source error ordering, endpoint scale handling,
`tweakDomain` map/metadata preservation, and a constant quotient. The
128-epsilon value bound is a diagnostic control, not a MATLAB source bound.
Chebfun commit: 7574c77
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import (
    Chebfun,
    _Piece,
    tweak_domain,
)
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

_DOMAIN_ID = "CHEBFUN:CHEBFUN:rdivide:columnRdivide:domain"
_DOMAIN_ERROR = _DOMAIN_ID + ": Inconsistent domains."
_DIM_ID = "CHEBFUN:CHEBFUN:rdivide:columnRdivide:dim"
_DIM_ERROR = _DIM_ID + ": Matrix dimension do not agree (transposed)"


class QuotientBuilderReached(RuntimeError):
    """Sentinel proving compatibility validation preceded quotient building."""


def _cf(coeffs, domain=(-1.0, 1.0), *, row=False, point_values=None,
        deltas=()):
    """Construct one bounded piece directly from Chebyshev coefficients."""
    a, b = map(float, domain)
    tech = Chebtech2.from_coeffs(jnp.asarray(coeffs))
    result = Chebfun(
        funs=[_Piece(tech=tech, interval=(a, b))],
        domain=Domain((a, b)),
        deltas=deltas,
    )
    if point_values is not None:
        result = result.set_point_values(jnp.asarray(point_values))
    return result.T if row else result


def _record_root_stage(monkeypatch, *, pole_count):
    """Install bounded roots/builders, and return a call-record list."""
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    calls = []

    def roots(denominator):
        calls.append(("roots", denominator))
        if pole_count:
            return jnp.asarray([0.25], dtype=jnp.float64)
        return jnp.zeros((0,), dtype=jnp.float64)

    def pole_builder(*_args, **_kwargs):
        calls.append(("pole_builder",))
        raise QuotientBuilderReached("pole quotient builder reached")

    def smooth_builder(*_args, **_kwargs):
        calls.append(("smooth_builder",))
        raise QuotientBuilderReached("smooth quotient builder reached")

    monkeypatch.setattr(module, "_real_simple_roots", roots)
    monkeypatch.setattr(module, "_divide_with_poles", pole_builder)
    monkeypatch.setattr(Chebfun, "_binary_op", staticmethod(smooth_builder))
    return calls


def _mismatch_pair(mismatch):
    numerator = _cf([6.0])
    denominator = _cf([3.0])
    if mismatch == "domain":
        numerator = _cf([6.0], domain=(-2.0, 2.0))
    elif mismatch == "orientation":
        numerator = numerator.T
    elif mismatch == "domain_and_orientation":
        numerator = _cf([6.0], domain=(-2.0, 2.0), row=True)
    else:
        raise AssertionError(f"unknown mismatch {mismatch}")
    return numerator, denominator


@pytest.mark.parametrize("pole_count", [0, 1], ids=["root-free", "root-bearing"])
@pytest.mark.parametrize(
    ("mismatch", "expected"),
    [
        ("domain", _DOMAIN_ERROR),
        ("orientation", _DIM_ERROR),
        # MATLAB domainCheck is evaluated before the transposed-dimension check.
        ("domain_and_orientation", _DOMAIN_ERROR),
    ],
)
def test_rdivide_checks_domain_then_orientation_after_root_stage(
        monkeypatch, pole_count, mismatch, expected):
    calls = _record_root_stage(monkeypatch, pole_count=pole_count)
    numerator, denominator = _mismatch_pair(mismatch)
    if pole_count:
        # The independent affine denominator has its genuine root at0.25.
        denominator = _cf([-0.25, 1.0])

    with pytest.raises(ValueError) as exc_info:
        numerator / denominator

    assert str(exc_info.value) == expected
    assert calls and calls[0][0] == "roots"
    assert not any(call[0] in {"pole_builder", "smooth_builder"}
                   for call in calls)


@pytest.mark.parametrize("row", [False, True], ids=["column", "row"])
def test_tweak_domain_snaps_near_integer_endpoint_and_preserves_metadata(row):
    """Direct map control: retain coefficients, point values, deltas, row flag."""
    base = 1_000_000.0
    shifted = base + 2 * np.spacing(base)
    delta = (base + 0.5, 2.0)
    f = _cf(
        [3.0, 0.25],
        domain=(shifted, base + 1.0),
        row=row,
        point_values=[6.0, 8.0],
        deltas=(delta,),
    )
    g = _cf([2.0], domain=(base, base + 1.0))
    original_coeffs = np.asarray(f.funs[0].tech.coeffs).copy()

    f_aligned, g_aligned, loc_f, loc_g = tweak_domain(f, g)

    assert f_aligned.domain.breakpoints == g_aligned.domain.breakpoints
    np.testing.assert_array_equal(
        np.asarray(f_aligned.domain.breakpoints), [base, base + 1.0]
    )
    assert loc_f == [0]
    assert loc_g == [0]
    np.testing.assert_array_equal(
        np.asarray(f_aligned.funs[0].tech.coeffs), original_coeffs
    )
    assert f_aligned.is_transposed is row
    np.testing.assert_array_equal(f_aligned.point_values, [6.0, 8.0])
    assert f_aligned.deltas == (delta,)


def test_rdivide_accepts_source_close_endpoint_and_maps_before_quotient():
    """Two-ULP endpoint shift is below 1e-15*hscale at hscale=1e6."""
    base = 1_000_000.0
    shifted = base + 2 * np.spacing(base)
    numerator = _cf(
        [6.0], domain=(shifted, base + 1.0), row=True,
        point_values=[12.0, 18.0],
    )
    denominator = _cf(
        [3.0], domain=(base, base + 1.0), row=True,
        point_values=[3.0, 6.0],
    )

    result = numerator / denominator

    np.testing.assert_array_equal(
        np.asarray(result.domain.breakpoints), [base, base + 1.0]
    )
    assert result.is_transposed
    np.testing.assert_allclose(
        np.asarray(result(jnp.asarray([base + 0.25, base + 0.75]))).ravel(),
        [2.0, 2.0], rtol=0.0,
        atol=128 * np.finfo(float).eps,
    )
    np.testing.assert_array_equal(result.point_values, [4.0, 3.0])


def test_rdivide_rejects_endpoint_past_source_tolerance_before_builder(monkeypatch):
    """16 ULPs exceeds 1e-15*hscale but stays below generic 1e-14*hscale."""
    base = 1_000_000.0
    shifted = base + 16 * np.spacing(base)
    numerator = _cf([6.0], domain=(shifted, base + 1.0))
    denominator = _cf([3.0], domain=(base, base + 1.0))
    calls = _record_root_stage(monkeypatch, pole_count=0)

    with pytest.raises(ValueError) as exc_info:
        numerator / denominator

    assert str(exc_info.value) == _DOMAIN_ERROR
    assert calls and calls[0][0] == "roots"
    assert not any(call[0] in {"pole_builder", "smooth_builder"}
                   for call in calls)


def test_root_free_matched_column_quotient_has_independent_value_control():
    """Constant source-independent quotient; diagnostic 128-epsilon bound."""
    result = _cf([6.0]) / _cf([3.0])
    points = jnp.asarray([-0.75, -0.1, 0.4, 0.9])
    np.testing.assert_allclose(
        np.asarray(result(points)), np.full(points.shape, 2.0),
        rtol=0.0, atol=128 * np.finfo(float).eps,
    )


@pytest.mark.parametrize("source_assertion", [14, 15])
def test_literal_source_compatibility_error_identifiers(source_assertion):
    """Original tests/chebfun/test_rdivide.m assertions14/15, no probes."""
    if source_assertion == 14:
        f = Chebfun.from_function(
            lambda x: jnp.stack([jnp.exp(x), jnp.exp(-x)], axis=-1),
            Domain((-1.0, -0.5, 0.0, 0.5, 1.0)))
        g = Chebfun.from_function(
            lambda x: jnp.stack([jnp.exp(-x), jnp.exp(x)], axis=-1),
            Domain((-1.0, 1.0))).T
        expected = _DIM_ERROR
    else:
        f = Chebfun.from_function(jnp.exp, Domain((-1.0, 1.0)))
        g = Chebfun.from_function(jnp.exp, Domain((0.0, 2.0)))
        expected = _DOMAIN_ERROR
    with pytest.raises(ValueError) as exc_info:
        f / g
    assert str(exc_info.value) == expected


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_numeric_domain_always_receives_source_preference(side):
    """Source numeric-domain recursion forces SIDE=1, preserving noninteger."""
    eps = np.finfo(float).eps
    breaks = (-1.0, 0.5 + eps, 1.0)
    f = Chebfun(funs=[
        _Piece(tech=Chebtech2.from_coeffs(jnp.asarray([2.0])),
               interval=interval)
        for interval in zip(breaks[:-1], breaks[1:])], domain=Domain(breaks))
    target = [-1.0, 0.5 - eps, 1.0]
    mapped, returned, i, j = tweak_domain(f, target, side=side)
    np.testing.assert_array_equal(mapped.domain.breakpoints, target)
    np.testing.assert_array_equal(returned, target)
    assert i == j == [1]


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_source_rounding_at_exact_half_integer(sign):
    """MATLAB round uses ties away fromzero; customtol activates this branch."""
    f = _cf([2.0], domain=(sign * 0.1, 10.0))
    g = _cf([3.0], domain=(sign * 0.9, 10.0))
    mapped_f, mapped_g, i, j = tweak_domain(f, g, tol=1.0)
    assert mapped_f.domain.breakpoints == mapped_g.domain.breakpoints
    assert mapped_f.domain.a == sign
    assert i == j == [0]
