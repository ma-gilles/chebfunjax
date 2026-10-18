"""Pole addition in pinned @diskfun/plus.m uses a continuous L2 decision.

Chebfun 7574c77680d7e82b79626300bf255498271a72df, addPoles and
@chebfun/norm.m. These manufactured controls complement original disk31/32.
"""

import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


def _disk_add_poles(*args):
    # Lazy lookup lets the same baseline module exercise public PLUS even
    # though the new private helper does not exist before this repair.
    return importlib.import_module("chebfunjax.diskfun.diskfun")._disk_add_poles(*args)


def _pole(coeffs, pivot=1.0):
    return (
        Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64)),
        Trigtech.from_coeffs(jnp.ones(1), is_real=True),
        jnp.float64(pivot),
    )


def test_empty_pair():
    assert _disk_add_poles(None, None, 0.0) is None


@pytest.mark.parametrize("left", [False, True])
def test_single_pole_passes_through_without_sampling(left):
    p = _pole([1.0, 0.0, 0.5], 3.0)
    result = _disk_add_poles(p if left else None, None if left else p, 100.0)
    assert result is p


def test_exact_cancellation():
    assert _disk_add_poles(_pole([1.0, 0.5]), _pole([1.0, 0.5], -1.0), 0.0) is None


@pytest.mark.parametrize("amplitude,cancel", [(1.02, True), (1.1, False)])
def test_continuous_norm_not_sample_peak(amplitude, cancel):
    tol = 1e-3
    # Integral(T2**2) = 14/15. At amplitude1.02, peak>tol but norm<tol.
    p = _pole([0.0, 0.0, amplitude * tol])
    result = _disk_add_poles(_pole([0.0]), p, tol)
    assert (result is None) is cancel
    if not cancel:
        assert jnp.array_equal(result[0].coeffs, p[0].coeffs)
        assert result[2] == 1.0


def test_constant_peak_below_tolerance_can_have_norm_above():
    tol = 1e-3
    # Integral(constant**2) = 2*constant**2; max alone would wrongly cancel.
    result = _disk_add_poles(_pole([0.0]), _pole([0.8 * tol]), tol)
    assert result is not None
    assert jnp.array_equal(result[0].coeffs, jnp.array([0.8 * tol]))


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_nonfinite_norm_does_not_satisfy_source_less_equal(value):
    assert _disk_add_poles(_pole([value]), _pole([0.0]), 1.0) is not None


@pytest.mark.parametrize("rest_pivot,cancel", [(1e-2, True), (1e2, False)])
def test_public_plus_scales_tolerance_after_extracting_poles(rest_pivot, cancel):
    # The physical rest scale is1/pivot; individual column scales remain1.
    radial = Chebtech2.from_coeffs(jnp.array([0.5, 0.0, 0.5]))
    row = Trigtech.from_coeffs(jnp.array([0.5, 0.0, 0.0, 0.0, 0.5]), is_real=True)
    one = Trigtech.from_coeffs(jnp.ones(1), is_real=True)
    delta = 8 * jnp.finfo(jnp.float64).eps
    a = Diskfun(
        cols=[_pole([1.0])[0], radial],
        rows=[one, row],
        pivots=jnp.array([1.0, rest_pivot]),
        idx_plus=(0, 1),
        idx_minus=(),
        nonzero_poles=True,
    )
    b = Diskfun(
        cols=[_pole([-1.0 + delta])[0], radial],
        rows=[one, row],
        pivots=jnp.array([1.0, -rest_pivot]),
        idx_plus=(0, 1),
        idx_minus=(),
        nonzero_poles=True,
    )
    result = a + b
    assert result.nonzero_poles is (not cancel)
    if not cancel:
        assert jnp.array_equal(result.cols[0].coeffs, jnp.array([delta]))


@pytest.mark.parametrize("kind,cancel", [("T2", True), ("constant", False)])
def test_public_plus_uses_continuous_pole_norm(kind, cancel):
    eps = jnp.finfo(jnp.float64).eps
    radial = Chebtech2.from_coeffs(jnp.array([0.5, 0.0, 0.5]))
    row = Trigtech.from_coeffs(jnp.array([0.5, 0.0, 0.0, 0.0, 0.5]), is_real=True)
    one = Trigtech.from_coeffs(jnp.ones(1), is_real=True)
    # The extracted remainder has vscale1, hence the source tolerance isEPS.
    # T2's norm is sqrt(14/15) times its coefficient; a constant's norm
    # is sqrt(2) times its value. Opposite peak/norm decisions matter here.
    second = [-1.0, 0.0, 1.02 * eps] if kind == "T2" else [-1.0 + 0.8 * eps]
    a = Diskfun(
        cols=[_pole([1.0])[0], radial],
        rows=[one, row],
        pivots=jnp.ones(2),
        idx_plus=(0, 1),
        idx_minus=(),
        nonzero_poles=True,
    )
    b = Diskfun(
        cols=[_pole(second)[0], radial],
        rows=[one, row],
        pivots=jnp.array([1.0, -1.0]),
        idx_plus=(0, 1),
        idx_minus=(),
        nonzero_poles=True,
    )
    result = a + b
    assert result.nonzero_poles is (not cancel)
    if not cancel:
        assert jnp.array_equal(result.cols[0].coeffs, jnp.array([eps]))
