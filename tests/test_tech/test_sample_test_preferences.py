"""Bounded MATLAB sampleTest preference controls.

Provenance
----------
MATLAB source : @chebtech/happinessCheck.m, @chebtech/sampleTest.m,
    @chebfun/constructor.m
Chebfun commit: 7574c77
The ODE fitter control from the root suite is excluded from this publication
subset; this file adds no ODE adapter dependency.
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax.tech.chebtech as chebtech_module
from chebfunjax.chebfun1d.chebfun import _Piece, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_SAMPLE_POINTS = np.array([-0.357998918959666, 0.036785641195074])

def _sample_calls(build):
    calls = []

    def op(x):
        arr = np.asarray(x)
        if arr.shape == (2,) and np.array_equal(arr, _SAMPLE_POINTS):
            calls.append(arr)
        return jnp.ones_like(x)

    build(op)
    return calls

def test_chebtech1_and_2_respect_explicit_sample_test_flag():
    for tech in (Chebtech1, Chebtech2):
        assert not _sample_calls(
            lambda op: tech.from_function(op, maxpow2=8, sample_test=False))
        assert len(_sample_calls(
            lambda op: tech.from_function(op, maxpow2=8))) == 1

def test_chebfun_constructor_uses_sample_test_preference():
    ChebfunPref.setDefaults("sampleTest", False)
    try:
        assert not _sample_calls(
            lambda op: chebfun(op, domain=(-1.0, 1.0), max_length=17))
    finally:
        ChebfunPref.setDefaults("factory")

def _watch_sample_test_calls(monkeypatch):
    """Record only the canonical probe passed by happinessCheck/sampleTest."""
    calls = []
    original = chebtech_module._happiness_check_impl

    def spy(tech_cls, kind, coeffs, values, op, tol, vscale, hscale, check,
            sample_test=True):
        # Observe actual calls even if a future defect ignores the flag.
        if op is not None:
            original_op = op

            def tracked_op(x):
                arr = np.asarray(x)
                if arr.shape == (2,) and np.array_equal(arr, _SAMPLE_POINTS):
                    calls.append(arr.copy())
                return original_op(x)

            op = tracked_op
        return original(
            tech_cls, kind, coeffs, values, op, tol, vscale, hscale, check,
            sample_test,
        )

    monkeypatch.setattr(chebtech_module, "_happiness_check_impl", spy)
    return calls


def test_splitting_constructor_threads_sample_test_to_locator_and_pieces(
        monkeypatch):
    counts = _watch_sample_test_calls(monkeypatch)
    callback_pairs = []

    def op(x):
        arr = np.asarray(x)
        if arr.shape == (2,):
            callback_pairs.append(arr.copy())
        return jnp.sign(x)

    ChebfunPref.setDefaults("sampleTest", False)
    try:
        chebfun(op, domain=(-1.0, 1.0), splitting=True,
                max_length=129, split_length=32, split_max_length=256)
        assert counts == []
        # Edge detection legitimately queries a two-endpoint bracket even
        # with sampleTest off. Keep that independent constructor behavior.
        assert any(not np.array_equal(pair, _SAMPLE_POINTS)
                   for pair in callback_pairs)
    finally:
        ChebfunPref.setDefaults("factory")

    ChebfunPref.setDefaults("sampleTest", True)
    try:
        chebfun(op, domain=(-1.0, 1.0), splitting=True,
                max_length=129, split_length=32, split_max_length=256)
        assert counts
    finally:
        ChebfunPref.setDefaults("factory")

def test_splitting_constructor_keeps_per_column_vscale():
    def op(x):
        return jnp.stack((jnp.sign(x), 1e-8 * (1.0 + x + x*x)), axis=-1)

    result = chebfun(op, domain=(-1.0, 1.0), splitting=True,
                     max_length=129, split_length=32, split_max_length=256)
    points = jnp.array([-0.8, -0.2, 0.2, 0.8])
    actual = np.asarray(result(points))
    expected = np.stack((np.sign(np.asarray(points)),
                         1e-8 * (1.0 + np.asarray(points)
                                 + np.asarray(points)**2)), axis=-1)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0.0)

def test_piece_forwards_sample_test_flag():
    assert not _sample_calls(
        lambda op: _Piece.from_function(
            op, -1.0, 1.0, maxpow2=4, sample_test=False))
