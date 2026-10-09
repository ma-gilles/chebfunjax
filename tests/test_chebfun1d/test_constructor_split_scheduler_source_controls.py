"""Source-ordered state-machine controls for constructorSplit.

The first three controls target the candidate API extension `breakpoints=` on
`_construct_with_splitting`; that keyword is absent from the P86 baseline, so
their baseline run is expected to fail at the API boundary until the candidate
lands. The public smooth-domain case separately qualifies the existing public
constructor route. The scheduler remains production code.  Only _Piece.from_function,
_detect_edge_matlab, and Chebfun.merge are controlled; every scheduled fit is
represented by a real Chebtech2.from_coeffs fixture.

Provenance
----------
MATLAB source : @chebfun/constructor.m, @chebfunpref/chebfunpref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.tech.chebtech import Chebtech2


@dataclass(frozen=True)
class _Fit:
    length: int
    scale: float
    happy: bool


def _install_scheduler_seams(monkeypatch, fits, edges=()):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    remaining = list(fits)
    scripted_edges = dict(edges)
    fit_events = []
    edge_events = []

    def piece_from_function(cls, _f, a, b, **kwargs):
        a, b = float(a), float(b)
        fit_events.append((a, b, float(kwargs.get("vscale", 0.0))))
        fit = remaining.pop(0)
        coeffs = jnp.zeros((fit.length,), dtype=jnp.float64).at[0].set(fit.scale)
        tech = Chebtech2.from_coeffs(coeffs, ishappy=fit.happy)
        return cls(tech=tech, interval=(a, b))

    def detect_edge(_f, a, b, **_kwargs):
        interval = (float(a), float(b))
        edge_events.append(interval)
        return scripted_edges[interval]

    # Public construction now uses selected-Tech context fitting. Observe
    # that seam while retaining actual bounded_get_fun scale propagation.
    context = importlib.import_module("chebfunjax.chebfun1d._construction_context")
    original_get_fun = context.bounded_get_fun
    active_interval = [None]
    def get_fun(op, interval, data, pref):
        active_interval[0] = interval
        return original_get_fun(op, interval, data, pref)
    def tech_from_function(cls, op, **kwargs):
        return piece_from_function(_Piece, op, *active_interval[0], **kwargs).tech
    monkeypatch.setattr(context, "bounded_get_fun", get_fun)
    monkeypatch.setattr(Chebtech2, "from_function", classmethod(tech_from_function))
    monkeypatch.setattr(_Piece, "from_function", classmethod(piece_from_function))
    monkeypatch.setattr(module, "_detect_edge_matlab", detect_edge)
    monkeypatch.setattr(Chebfun, "merge", lambda self, *args, **kwargs: self)
    return module, fit_events, edge_events, remaining


def _smooth(x):
    return jnp.ones_like(jnp.asarray(x), dtype=jnp.float64)


def test_initial_intervals_are_left_to_right_and_only_happy_fits_update_scale(monkeypatch):
    fits = [
        _Fit(17, 100.0, False),  # unhappy initial left interval
        _Fit(17, 4.0, True),     # happy initial right interval
        _Fit(17, 10.0, True),    # happy left child
        _Fit(17, 200.0, False),  # unhappy right child
        _Fit(17, 11.0, True),
        _Fit(17, 12.0, True),
    ]
    module, fit_events, edge_events, remaining = _install_scheduler_seams(
        monkeypatch, fits, edges={(0.0, 1.0): 0.5, (0.5, 1.0): 0.75})

    result = module._construct_with_splitting(
        _smooth, 0.0, 2.0, maxpow2=8, split_length=33,
        split_max_length=1000, sample_test=False, vscale=5.0,
        breakpoints=(0.0, 1.0, 2.0))

    assert fit_events == [
        (0.0, 1.0, 5.0), (1.0, 2.0, 5.0),
        (0.0, 0.5, 5.0), (0.5, 1.0, 10.0),
        (0.5, 0.75, 10.0), (0.75, 1.0, 11.0),
    ]
    assert edge_events == [(0.0, 1.0), (0.5, 1.0)]
    assert result.ishappy
    assert not remaining


def test_widest_unhappy_interval_is_selected_first(monkeypatch):
    fits = [
        _Fit(17, 1.0, False),  # [0, 0.6]
        _Fit(17, 1.0, False),  # [0.6, 2]
        _Fit(17, 1.0, True),   # [0.6, 1.3]
        _Fit(17, 1.0, True),   # [1.3, 2]
        _Fit(17, 1.0, True),   # [0, 0.3]
        _Fit(17, 1.0, True),   # [0.3, 0.6]
    ]
    module, fit_events, edge_events, remaining = _install_scheduler_seams(
        monkeypatch, fits, edges={(0.6, 2.0): 1.3, (0.0, 0.6): 0.3})

    result = module._construct_with_splitting(
        _smooth, 0.0, 2.0, maxpow2=8, split_length=33,
        split_max_length=1000, sample_test=False,
        breakpoints=(0.0, 0.6, 2.0))

    assert edge_events == [(0.6, 2.0), (0.0, 0.6)]
    assert fit_events == [
        (0.0, 0.6, 0.0), (0.6, 2.0, 0.0),
        (0.6, 1.3, 0.0), (1.3, 2.0, 1.0),
        (0.0, 0.3, 1.0), (0.3, 0.6, 1.0),
    ]
    assert result.ishappy
    assert not remaining


def test_equal_width_tie_keeps_first_interval_and_fits_left_child_first(monkeypatch):
    fits = [
        _Fit(17, 1.0, False), _Fit(17, 1.0, False),
        _Fit(17, 1.0, True), _Fit(17, 1.0, True),
        _Fit(17, 1.0, True), _Fit(17, 1.0, True),
    ]
    module, fit_events, edge_events, remaining = _install_scheduler_seams(
        monkeypatch, fits, edges={(0.0, 1.0): 0.5, (1.0, 2.0): 1.5})

    result = module._construct_with_splitting(
        _smooth, 0.0, 2.0, maxpow2=8, split_length=33,
        split_max_length=1000, sample_test=False,
        breakpoints=(0.0, 1.0, 2.0))

    assert edge_events == [(0.0, 1.0), (1.0, 2.0)]
    assert fit_events == [
        (0.0, 1.0, 0.0), (1.0, 2.0, 0.0),
        (0.0, 0.5, 0.0), (0.5, 1.0, 1.0),
        (1.0, 1.5, 1.0), (1.5, 2.0, 1.0),
    ]
    assert result.ishappy
    assert not remaining


def test_actual_total_length_is_checked_after_both_children_are_inserted(monkeypatch):
    fits = [
        _Fit(4, 1.0, False),
        _Fit(3, 1.0, True),
        _Fit(3, 1.0, True),
    ]
    module, fit_events, edge_events, remaining = _install_scheduler_seams(
        monkeypatch, fits, edges={(0.0, 1.0): 0.5})

    with pytest.warns(UserWarning, match="Function not resolved using 6 pts"):
        result = module._construct_with_splitting(
            _smooth, 0.0, 1.0, maxpow2=8, split_length=17,
            split_max_length=5, sample_test=False)

    assert edge_events == [(0.0, 1.0)]
    assert fit_events == [
        (0.0, 1.0, 0.0), (0.0, 0.5, 0.0), (0.5, 1.0, 1.0)]
    assert [piece.n for piece in result.funs] == [3, 3]
    assert not remaining


def test_public_smooth_split_constructor_preserves_user_breakpoint_scale(monkeypatch):
    fits = [_Fit(17, 1.0, True), _Fit(17, 2.0, True)]
    _, fit_events, edge_events, remaining = _install_scheduler_seams(monkeypatch, fits)

    result = cj.chebfun(
        _smooth, domain=(0.0, 1.0, 2.0), splitting=True,
        split_length=33, split_max_length=100)

    assert [(a, b) for a, b, _ in fit_events] == [(0.0, 1.0), (1.0, 2.0)]
    assert [scale for _, _, scale in fit_events] == [0.0, 1.0]
    assert edge_events == []
    assert len(result.funs) == 2
    assert result.domain.breakpoints == (0.0, 1.0, 2.0)
    assert not remaining
