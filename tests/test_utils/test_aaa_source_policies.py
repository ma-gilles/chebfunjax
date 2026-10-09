"""Pinned aaa.m singular-vector, scaling and nonfinite-query controls."""

import importlib

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.aaa import _aaa_scale_needed, _aaa_source_weights, _reval, aaa


def test_near_minima_are_not_exact_multiplicity():
    s = jnp.array([2.0, 1 + 5e-11, 1.0])
    vh = jnp.diag(jnp.array([1.0, 1j, -1j]))
    assert jnp.array_equal(_aaa_source_weights(s, vh, False), jnp.array([0.0, 0.0, 1j]))


def test_exact_minima_use_all_source_vectors():
    s = jnp.array([2.0, 1.0, 1.0])
    vh = jnp.diag(jnp.array([1.0, 1j, -1j]))
    expected = jnp.array([0.0, -1j, 1j]) / jnp.sqrt(2.0)
    assert jnp.array_equal(_aaa_source_weights(s, vh, False), expected)


def test_sign_zero_minimum_uses_final_vector():
    s = jnp.array([2.0, 0.0, 0.0])
    vh = jnp.diag(jnp.array([1.0, 1j, -1j]))
    assert jnp.array_equal(_aaa_source_weights(s, vh, True), jnp.array([0.0, 0.0, 1j]))


def test_condition_test_is_scale_invariant():
    s = jnp.array([1.0, 1e-17])
    assert _aaa_scale_needed(s)
    assert _aaa_scale_needed(s * 1e-290)
    assert not _aaa_scale_needed(jnp.zeros(2))


def test_scaling_stays_active_after_source_trigger(monkeypatch):
    module = importlib.import_module("chebfunjax.utils.aaa")
    original = module.np.linalg.svd
    calls = []

    def record(A, *args, **kwargs):
        calls.append(A.shape)
        return original(A, *args, **kwargs)

    monkeypatch.setattr(module.np.linalg, "svd", record)
    x = jnp.logspace(-15, 0, 300)
    _, _, _, _, z, _, _ = aaa(jnp.sqrt(x), x, cleanup=False)
    # One unscaled SVD per pre-trigger step, two at the trigger, and only
    # one scaled SVD per later step. This source input reaches scaling.
    assert len(calls) == len(z) + 1


@pytest.mark.parametrize("compile", [False, True])
def test_complex_infinite_queries_use_source_limit(compile):
    z = jnp.array([-1 + 1j, 2 - 1j, 3 + 2j])
    f = jnp.array([2 + 1j, 1 - 3j, 4 + 0.5j])
    w = jnp.array([1 - 2j, 3 + 1j, -0.5 + 0.3j])
    queries = jnp.array(
        [complex("inf"), complex("-inf"), complex(0, float("inf")), complex(float("nan"))]
    )

    def fun(x):
        return _reval(x, z, f, w)

    values = (jax.jit(fun) if compile else fun)(queries)
    expected = jnp.sum(w * f) / jnp.sum(w)
    assert jnp.all(values[:3] == expected)
    assert jnp.isnan(values[-1])


def test_complex_physical_poles_and_residues():
    poles = jnp.array([0.5 + 0.7j, -0.4 + 0.3j])
    residues = jnp.array([1 + 0.2j, -0.5 + 0.1j])
    z = jnp.linspace(-1, 1, 200) + 0.15j
    values = 2 - 1j + jnp.sum(residues[None, :] / (z[:, None] - poles[None, :]), axis=1)
    r, p, res, *_ = aaa(values, z)
    assert len(p) == 2
    for pole, residue in zip(poles, residues):
        k = jnp.argmin(jnp.abs(p - pole))
        assert jnp.abs(p[k] - pole) < 1e-11
        assert jnp.abs(res[k] - residue) < 1e-11
    assert jnp.abs(r(jnp.asarray(jnp.inf)) - (2 - 1j)) < 1e-12


@pytest.mark.parametrize("complex_input", [False, True])
def test_callback_receives_original_sample_type(complex_input):
    received = []

    def callback(z):
        received.append(jnp.iscomplexobj(z))
        return 1 + z + z * z

    z = jnp.linspace(-1, 1, 20)
    if complex_input:
        z = z + 0.2j
    aaa(callback, z)
    assert received == [complex_input]
