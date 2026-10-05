"""Source-backed controls for MATLAB's exact zero-FUN right-division guard.

Source provenance: @chebfun/rdivide.m and @chebtech/iszero.m, Chebfun
commit 7574c77680d7e82b79626300bf255498271a72df. Error strings preserve the
MATLAB identifier as a Python message prefix because ValueError has no
separate MATLAB-style identifier attribute in this port.

The tests use direct Chebyshev coefficients and physical piece intervals;
there is no adaptive construction or sample-based zero oracle. Root-finder
sentinels make a missing pre-root guard fail quickly rather than attempting
an invalid quotient. The one quotient value check is an independent constant
identity with a 128-epsilon diagnostic bound, not an original MATLAB bound.
Provenance
----------
MATLAB source: @chebfun/rdivide.m, @chebtech/iszero.m,
    tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_ZERO_FUN_ID = (
    "CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun"
)
_ZERO_FUN_MESSAGE = (
    _ZERO_FUN_ID + ": Division by CHEBFUN with identically zero FUN."
)


class RootStageReached(RuntimeError):
    """Sentinel: a zero-FUN error must be raised before root finding."""


def _piece(coeffs, kind, interval):
    tech_class = Chebtech1 if kind == 1 else Chebtech2
    return _Piece(
        tech=tech_class.from_coeffs(jnp.asarray(coeffs)),
        interval=(float(interval[0]), float(interval[1])),
    )


def _cf(coeffs, kind=2, domain=(-1.0, 1.0), transposed=False):
    a, b = map(float, domain)
    f = Chebfun(
        funs=[_piece(coeffs, kind, (a, b))],
        domain=Domain((a, b)),
    )
    return f.T if transposed else f


def _denominator(kind, layout):
    if layout == "single":
        breaks = (-1.0, 1.0)
        coeffs = ([0.0],)
    elif layout == "zero_first":
        breaks = (-1.0, 0.0, 1.0)
        coeffs = ([0.0], [1.0])
    elif layout == "zero_last":
        breaks = (-1.0, 0.0, 1.0)
        coeffs = ([1.0], [0.0])
    else:
        raise AssertionError(f"unknown layout {layout}")
    funs = [
        _piece(c, kind, (breaks[k], breaks[k + 1]))
        for k, c in enumerate(coeffs)
    ]
    g = Chebfun(funs=funs, domain=Domain(breaks))
    return g, breaks


def _force_root_stage(monkeypatch):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")

    def raise_root_stage(*_args, **_kwargs):
        raise RootStageReached("root finder was reached before zero-FUN guard")

    monkeypatch.setattr(module, "_real_simple_roots", raise_root_stage)


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("layout", ["single", "zero_first", "zero_last"])
@pytest.mark.parametrize("numerator", ["chebfun", "scalar"])
@pytest.mark.parametrize("override_points", [False, True])
def test_exact_zero_fun_error_precedes_roots(
        monkeypatch, kind, layout, numerator, override_points):
    """Each zero piece errors; nonzero pointValues cannot mask a zero FUN."""
    g, breaks = _denominator(kind, layout)
    zero_piece = 0 if layout != "zero_last" else len(g.funs) - 1
    assert np.all(np.asarray(g.funs[zero_piece].tech.coeffs) == 0.0)
    if override_points:
        # MATLAB rdivide tests each smooth FUN before point-value division.
        g = g.set_point_values(jnp.arange(len(breaks), dtype=jnp.float64) + 1.0)

    _force_root_stage(monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        if numerator == "chebfun":
            _cf([1.0], kind=kind, domain=(breaks[0], breaks[-1])) / g
        else:
            1.0 / g
    assert str(exc_info.value) == _ZERO_FUN_MESSAGE


def test_zero_point_values_do_not_turn_nonzero_fun_into_zero_denominator(
        monkeypatch):
    """Endpoint values are divided later; the guard itself tests smooth FUNs."""
    g = _cf([2.0, 1.0]).set_point_values(jnp.asarray([0.0, 0.0]))
    assert not np.all(np.asarray(g.funs[0].tech.coeffs) == 0.0)
    _force_root_stage(monkeypatch)
    with pytest.raises(RootStageReached):
        1.0 / g


@pytest.mark.parametrize("mismatch", ["domain", "orientation"])
def test_zero_fun_error_precedes_domain_or_orientation_check(
        monkeypatch, mismatch):
    """MATLAB's zero-FUN loop is before domain and orientation validation."""
    g = _cf([0.0])
    if mismatch == "domain":
        numerator = _cf([1.0], domain=(-2.0, 2.0))
    else:
        numerator = _cf([1.0]).T
    _force_root_stage(monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        numerator / g
    assert str(exc_info.value) == _ZERO_FUN_MESSAGE


def test_zero_numerator_shortcut_precedes_zero_fun_guard(monkeypatch):
    """MATLAB rdivide.m lines 18-22 return 0*g before denominator checks."""
    g = _cf([0.0], kind=2)
    _force_root_stage(monkeypatch)
    result = 0.0 / g
    assert all(np.all(np.asarray(piece.tech.coeffs) == 0.0)
               for piece in result.funs)


def test_tiny_nonzero_constant_is_not_an_identically_zero_fun(monkeypatch):
    """Exact source coefficient zero logic must not impose a vscale cutoff."""
    g = _cf([2.0**-60], kind=2)
    assert np.asarray(g.funs[0].tech.coeffs)[0] != 0.0
    _force_root_stage(monkeypatch)
    with pytest.raises(RootStageReached):
        1.0 / g


def test_nonzero_constant_quotient_independent_value_control():
    """Source-independent exact constant quotient; diagnostic 128-eps bound."""
    f = _cf([6.0], kind=2)
    g = _cf([3.0], kind=2)
    result = f / g
    points = np.asarray([-0.75, -0.1, 0.4, 0.9])
    values = np.asarray(result(jnp.asarray(points)))
    np.testing.assert_allclose(
        values, np.full(points.shape, 2.0), rtol=0.0,
        atol=128 * np.finfo(float).eps,
    )
