"""Full native test_hypot.m predicates, Chebfun7574c77.

Extreme large/small array inputs and piecewise public preference constructor,
with source sample count, arithmetic, numeric infinity norms and strict bounds.
NumPy RandomState6178 is a uniform-stream adapter, not a MATLAB draw identity
claim. Both previous unrelated probes remain supplemental below.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunHypot:
    def test_pythagorean_identity(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = f.hypot(g)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 30))
        np.testing.assert_allclose(np.asarray(h(xs)), 1.0, atol=1e-12)

    def test_scaling_robustness(self):
        # MATLAB's hypot example: no overflow for large components
        f = cj.chebfun(lambda x: 3e300 * (2 + jnp.sin(x)))
        g = cj.chebfun(lambda x: 4e300 * (2 + jnp.sin(x)))
        h = f.hypot(g)
        v = float(h(jnp.asarray(0.0)))
        assert np.isfinite(v)
        assert abs(v - 5e300 * 2) / 1e301 < 1e-10


import jax
import pytest

from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("clause", [1, 2])
def test_native_hypot_predicate(clause):
    eps = float(np.finfo(np.float64).eps)
    pref = ChebfunPref()
    xx = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
    if clause == 1:
        # Literal native arithmetic: scale the two columns, then multiply3/4.
        op = lambda x: jnp.stack((1e300 * x, 1e-300 * x), axis=-1)
        f = cj.chebfun(lambda x: 3 * op(x), pref=pref)
        g = cj.chebfun(lambda x: 4 * op(x), pref=pref)
        h = f.hypot(g)
        expected = jnp.hypot(3 * op(xx), 4 * op(xx))
        actual = h(xx)
        assert actual.shape == expected.shape == (100, 2)
        error = float(jnp.linalg.norm(actual - expected, ord=jnp.inf))
        normalized = error / float(h.vscale)
        bound = 10 * eps
    else:
        pref.splitting = 1
        op = lambda x: (
            jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x) * (np.pi / 16) + np.pi / 8
        )
        f = cj.chebfun(op, domain=(-1.0, -0.2, 0.1, 1.0), pref=pref)
        h = f.hypot(f)
        expected = jnp.hypot(op(xx), op(xx))
        actual = h(xx)
        assert actual.shape == expected.shape == (100,)
        error = float(jnp.linalg.norm(actual - expected, ord=jnp.inf))
        normalized = error
        bound = 100 * float(h.vscale) * eps
    print(
        {
            "clause": clause,
            "error": error,
            "tested_error": normalized,
            "bound": bound,
            "vscale": float(h.vscale),
            "shape": list(actual.shape),
        },
        flush=True,
    )
    assert normalized < bound


def test_native_hypot_numeric_other_and_explicit_pref():
    # Public wrapper contract: HYPOT accepts a numeric second operand and PREF.
    pref = ChebfunPref()
    f = cj.chebfun(lambda x: x, pref=pref)
    h = f.hypot(2.0, pref)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 101))
    error = float(jnp.linalg.norm(h(x) - jnp.hypot(x, 2.0), ord=jnp.inf))
    assert error < 100 * float(h.vscale) * float(np.finfo(np.float64).eps)
