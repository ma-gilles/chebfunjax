"""Independent constructor prerequisites and public scheduler contracts.

Provenance
----------
MATLAB source : @chebtech1/chebtech1.m, @chebtech2/chebtech2.m,
    @chebtech/constructorTurbo.m, @bndfun/bndfun.m,
    @chebfun/constructor.m (constructorSplit and getFun)
Chebfun commit: 7574c77
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech_cls", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("happy", [False, True])
def test_turbo_runs_only_after_happy_adaptation(monkeypatch, tech_cls, happy):
    module = importlib.import_module("chebfunjax.tech.chebtech")
    coeffs = jnp.array([2.0, -0.5, 0.25], dtype=jnp.float64)
    plain = tech_cls.from_coeffs(coeffs, ishappy=happy)
    adaptive_calls = []
    turbo_calls = []
    callback_calls = []

    def callback(x):
        callback_calls.append(x)
        return x

    def adaptive(cls, op, maxpow2=16, **kwargs):
        assert cls is tech_cls
        assert op is callback
        adaptive_calls.append((maxpow2, kwargs))
        return plain

    def turbo(op, input_coeffs, num):
        turbo_calls.append(num)
        assert bool(jnp.array_equal(input_coeffs, coeffs))
        op(jnp.array([1.0j], dtype=jnp.complex128))
        return jnp.arange(num, dtype=jnp.float64)

    monkeypatch.setattr(tech_cls, "_adaptive_construct", classmethod(adaptive))
    monkeypatch.setattr(module, "_turbo_coeffs", turbo)
    result = tech_cls.from_function(callback, turbo=True, max_length=160)

    assert len(adaptive_calls) == 1
    assert adaptive_calls[0][1]["max_length"] == 160
    assert adaptive_calls[0][1]["use_turbo"] is True
    assert adaptive_calls[0][1]["fixed_length"] is None
    assert result.ishappy is happy
    if happy:
        assert turbo_calls == [6]
        assert len(callback_calls) == 1
        assert result.n == 6
        assert bool(jnp.array_equal(result.coeffs, jnp.arange(6)))
    else:
        assert turbo_calls == []
        assert callback_calls == []
        assert result is plain
        assert result.n == 3
        assert bool(jnp.array_equal(result.coeffs, coeffs))


@pytest.mark.parametrize("interval", [(-1.0, 1.0), (2.0, 5.0)])
def test_piece_mapping_preserves_identity_and_affine_endpoints(monkeypatch, interval):
    points = jnp.array([-1.0, 2.0**-55, 1.0], dtype=jnp.float64)
    observed = []

    def callback(x):
        observed.append(x)
        return jnp.ones_like(x)

    def from_function(cls, op, **kwargs):
        op(points)
        return cls.from_coeffs(jnp.array([1.0]))

    monkeypatch.setattr(Chebtech2, "from_function", classmethod(from_function))
    _Piece.from_function(callback, *interval)

    assert len(observed) == 1
    if interval == (-1.0, 1.0):
        assert bool(jnp.array_equal(observed[0], points))
        assert float(observed[0][1]) == 2.0**-55
    else:
        a, b = interval
        expected = b * (points + 1) / 2 + a * (1 - points) / 2
        assert bool(jnp.array_equal(observed[0], expected))
        assert float(observed[0][0]) == a
        assert float(observed[0][-1]) == b


@pytest.mark.parametrize("shape", ["scalar", "flat", "row"])
def test_tiny_getfun_samples_once_and_builds_one_coefficient_row(monkeypatch, shape):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    seen = []
    values = jnp.array([2.0 + 3.0j, -4.0 + 1.0j], dtype=jnp.complex128)

    def callback(x):
        seen.append(x)
        if shape == "scalar":
            return values[0]
        if shape == "flat":
            return values
        return values[None, :]

    def unexpected_adaptation(*args, **kwargs):
        pytest.fail("Source getFun must skip adaptation on this tiny interval")

    monkeypatch.setattr(_Piece, "from_function", classmethod(unexpected_adaptation))
    left, right = 1.0, 1.0 + 2.0e-14
    result = module._construct_with_splitting(callback, left, right, maxpow2=8)

    assert len(seen) == 1
    assert float(seen[0]) == (left + right) / 2
    assert len(result.funs) == 1
    piece = result.funs[0]
    assert piece.n == 1
    assert piece.ishappy
    expected = values[:1] if shape == "scalar" else values[None, :]
    assert piece.tech.coeffs.shape == expected.shape
    assert bool(jnp.array_equal(piece.tech.coeffs, expected))


def test_public_unhappy_panels_share_widest_queue_scale_and_actual_budget(monkeypatch):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    # Initial left width1 and right width2 are unhappy. The right pair makes
    # the shared total4+3+3=10 exceed9; source stops with the left still sad.
    fits = [(4, 50.0, False), (4, 100.0, False),
            (3, 2.0, True), (3, 3.0, True)]
    events = []
    edges = []

    def from_function(cls, op, a, b, **kwargs):
        events.append((float(a), float(b), kwargs))
        length, scale, happy = fits.pop(0)
        coeffs = jnp.zeros(length, dtype=jnp.float64).at[0].set(scale)
        return cls(tech=Chebtech2.from_coeffs(coeffs, ishappy=happy),
                   interval=(float(a), float(b)))

    def detect_edge(op, a, b, **kwargs):
        edges.append((float(a), float(b), kwargs))
        assert (float(a), float(b)) == (1.0, 3.0)
        return 2.0

    monkeypatch.setattr(_Piece, "from_function", classmethod(from_function))
    monkeypatch.setattr(module, "_detect_edge_matlab", detect_edge)
    monkeypatch.setattr(Chebfun, "merge", lambda self, *args, **kwargs: self)
    with pytest.warns(UserWarning, match="Function not resolved using 10 pts"):
        result = cj.chebfun(lambda x: jnp.ones_like(x),
                            domain=(0.0, 1.0, 3.0), splitting=True,
                            split_length=160, split_max_length=9,
                            min_samples=226)

    assert not fits
    assert [(a, b) for a, b, _ in events] == [
        (0.0, 1.0), (1.0, 3.0), (1.0, 2.0), (2.0, 3.0)]
    assert [kw["vscale"] for _, _, kw in events] == [0.0, 0.0, 0.0, 2.0]
    assert [kw["hscale"] for _, _, kw in events] == [3.0, 1.5, 3.0, 3.0]
    assert all(kw["max_length"] == 160 for _, _, kw in events)
    assert all(kw["min_samples"] == 226 for _, _, kw in events)
    assert all(kw["extrapolate"] is True for _, _, kw in events)
    assert edges == [(1.0, 3.0, {"vscale": 0.0, "hscale": 3.0})]
    assert result.domain.breakpoints == (0.0, 1.0, 2.0, 3.0)
    assert [piece.n for piece in result.funs] == [4, 3, 3]
    assert [piece.ishappy for piece in result.funs] == [False, True, True]
