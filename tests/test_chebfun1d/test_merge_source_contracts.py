"""Independent bounded merge contracts from source review.

Provenance
----------
MATLAB source : @chebfun/merge.m, @fun/merge.m,
    @chebfun/getValuesAtBreakpoints.m
Chebfun commit: 7574c77
"""

import importlib

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


def _pieces(ends, values, lengths=None):
    if lengths is None:
        lengths = [1] * len(values)
    funs = []
    for a, b, value, length in zip(ends[:-1], ends[1:], values, lengths):
        value = jnp.asarray(value, dtype=jnp.float64)
        coeffs = jnp.zeros((length,) + value.shape).at[0].set(value)
        funs.append(_Piece(Chebtech2.from_coeffs(coeffs), (a, b)))
    return Chebfun(funs=funs, domain=Domain(tuple(ends)))


def _fit_seam(monkeypatch, scripted):
    pending = list(scripted)
    events = []

    def fit(cls, op, a, b, **kwargs):
        events.append((a, b, kwargs))
        happy, value = pending.pop(0)
        return cls(Chebtech2.from_coeffs(jnp.array([value]), ishappy=happy), (a, b))

    monkeypatch.setattr(_Piece, "from_function", classmethod(fit))
    return events, pending


def test_one_left_to_right_pass_does_not_retry_failed_earlier_break(monkeypatch):
    f = _pieces((0.0, 1.0, 2.0, 3.0), [1.0, 1.0, 1.0])
    events, pending = _fit_seam(monkeypatch, [(False, 1.0), (True, 1.0)])
    result = f.merge()
    assert [(a, b) for a, b, _ in events] == [(0.0, 2.0), (1.0, 3.0)]
    assert not pending
    assert result.domain.breakpoints == (0.0, 1.0, 3.0)


def test_length_guard_includes_exact_one_point_two_cap_boundary(monkeypatch):
    f = _pieces((0.0, 1.0, 2.0), [1.0, 1.0], lengths=[6, 6])
    events, pending = _fit_seam(monkeypatch, [])
    result = f.merge(max_length=10)
    assert result is f
    assert events == []
    assert pending == []


def test_point_jump_guard_uses_matrix_row_sum_and_global_scale(monkeypatch):
    f = _pieces((0.0, 1.0, 2.0), [[1.0, 1.0], [1.0, 1.0]])
    f = f.set_point_values(jnp.array([[1.0, 1.0],
                                     [1.0 + 1.5e-13, 1.0 + 1.5e-13],
                                     [1.0, 1.0]]))
    events, _ = _fit_seam(monkeypatch, [])
    assert f.merge() is f
    assert events == []


def test_selected_locations_are_exact_and_keep_nearby_user_break(monkeypatch):
    near = 1.0 + 2.0e-15
    f = _pieces((0.0, 1.0, near, 2.0), [1.0, 1.0, 1.0])
    events, pending = _fit_seam(monkeypatch, [(True, 1.0)])
    result = f.merge(index=[near])
    assert [(a, b) for a, b, _ in events] == [(1.0, 2.0)]
    assert not pending
    assert result.domain.breakpoints == (0.0, 1.0, 2.0)


def test_original_guards_scales_rows_and_raw_preferences_survive_prior_merge(monkeypatch):
    f = _pieces((0.0, 1.0, 2.0, 3.0), [1.0, 1.0, 1.0])
    f = f.set_point_values(jnp.array([7.0, 1.0, 1.0, 8.0]))
    object.__setattr__(f, "_is_transposed", True)
    # Deliberately different trial scale detects accidental recomputation of
    # source's frozen scale or use of current limits for the next jump guard.
    events, pending = _fit_seam(monkeypatch, [(True, 100.0), (True, 200.0)])
    result = f.merge(max_length=160, splitting=True, min_samples=226,
                     tol=1e-30, turbo=True, sample_test=False)
    assert [(a, b) for a, b, _ in events] == [(0.0, 2.0), (0.0, 3.0)]
    assert not pending
    assert [kw["vscale"] for _, _, kw in events] == [1.0, 1.0]
    assert [kw["hscale"] for _, _, kw in events] == [1.5, 1.0]
    assert all(kw["max_length"] == 160 for _, _, kw in events)
    assert all(kw["min_samples"] == 226 for _, _, kw in events)
    assert all(kw["extrapolate"] and kw["turbo"] for _, _, kw in events)
    assert all(not kw["sample_test"] for _, _, kw in events)
    assert all(float(kw["tol"]) == float(jnp.finfo(jnp.float64).eps)
               for _, _, kw in events)
    assert result.domain.breakpoints == (0.0, 3.0)
    assert result.is_transposed
    assert bool(jnp.array_equal(result._point_values, jnp.array([7.0, 8.0])))


def test_fun_merge_callback_uses_right_limit_not_stored_point_value(monkeypatch):
    f = _pieces((0.0, 1.0, 2.0), [1.0, 1.0 + 1e-14])
    f = f.set_point_values(jnp.array([1.0, 1.0 + 5e-15, 1.0 + 1e-14]))
    observed = []

    def fit(cls, op, a, b, **kwargs):
        observed.append(float(op(jnp.array(1.0))))
        return cls(Chebtech2.from_coeffs(jnp.array([1.0])), (a, b))

    monkeypatch.setattr(_Piece, "from_function", classmethod(fit))
    f.merge()
    assert observed == [1.0 + 1e-14]


def test_constructor_point_values_use_callback_replace_only_nan():
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    f = _pieces((0.0, 1.0, 2.0), [2.0, 4.0])
    calls = []

    def callback(x):
        calls.append(x)
        return jnp.array([jnp.inf, jnp.nan, 9.0])

    values = module._source_breakpoint_values(f.funs, f.domain.breakpoints, callback)
    assert len(calls) == 1
    assert bool(jnp.array_equal(calls[0], jnp.array([0.0, 1.0, 2.0])))
    assert bool(jnp.isinf(values[0]))
    assert float(values[1]) == 3.0
    assert float(values[2]) == 9.0
