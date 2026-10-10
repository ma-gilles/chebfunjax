"""Focused analytic controls and compiler specialization observation."""

import equinox as eqx
import jax
import jax._src.compiler as compiler
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.tech.chebtech import Chebtech2


@pytest.fixture
def cold_compilation_cache():
    """Count a fresh compile even when other tests ran first; never write artifacts."""
    previous = jax.config.jax_enable_compilation_cache
    eqx.clear_caches()
    jax.clear_caches()
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_compilation_cache", previous)
        eqx.clear_caches()
        jax.clear_caches()


def test_twenty_intervals_compile_once(monkeypatch, cold_compilation_cache):
    original = compiler.backend_compile_and_load
    calls = []

    def observed(*args, **kwargs):
        module = kwargs["module"] if "module" in kwargs else args[1]
        name = str(module.operation.attributes["sym_name"])
        out = original(*args, **kwargs)
        if "_evaluate_piece_interval" in name:
            calls.append(name)
        return out

    monkeypatch.setattr(compiler, "backend_compile_and_load", observed)
    tech = Chebtech2(coeffs=jnp.array([1.0, 2.0, 0.25]), ishappy=True)
    for k in range(20):
        a = float(k - 10)
        piece = _Piece(tech=tech, interval=(a, a + 2.0))
        np.testing.assert_array_equal(piece(jnp.array([a, a + 1.0, a + 2.0])), [-0.75, 0.75, 3.25])
    assert len(calls) == 1


def test_piece_grad_and_vmap():
    c = jnp.array([1.0, 2.0, 0.25])
    x = jnp.array([-1.0, 0.0, 1.0])

    def total(coeffs):
        return jnp.sum(_Piece(tech=Chebtech2(coeffs=coeffs), interval=(-1.0, 1.0))(x))

    np.testing.assert_array_equal(jax.grad(total)(c), [3.0, 0.0, 1.0])
    piece = _Piece(tech=Chebtech2(coeffs=c), interval=(-1.0, 1.0))
    np.testing.assert_array_equal(jax.vmap(jax.grad(lambda z: piece(z)))(x), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(jax.jit(jax.vmap(lambda z: piece(z)))(x), [-0.75, 0.75, 3.25])


def test_piece_complex_points_and_coefficients():
    c = jnp.array([1.0, 2.0, 0.25]) * (1.0 + 2j)
    piece = _Piece(tech=Chebtech2(coeffs=c), interval=(-1.0, 1.0))
    x = jnp.array([-0.5 + 0.25j, 0.25j, 0.5 + 0.25j])
    expected = (1 + 2 * x + 0.25 * (2 * x * x - 1)) * (1.0 + 2j)
    np.testing.assert_array_equal(piece(x), expected)
