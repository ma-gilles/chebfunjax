"""Source dispatcher controls supplementary to the forty original predicates.

Provenance
----------
MATLAB source : ultrapts.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import ultrapts


@pytest.mark.parametrize(
    "args,match",
    [
        ((3, -0.5), "sizeLAMBDA"),
        ((-1, 0.3), "ultrapts:n"),
        ((3, 0.3, "bad"), "inputs"),
        ((3, 0.3, (1, -1)), "inputs"),
        ((3, 0.3, (0, jnp.inf)), "inputs"),
        ((3, 0.3, (0, 1, 2)), "inputs"),
    ],
)
def test_source_errors(args, match):
    with pytest.raises(ValueError, match=match):
        ultrapts(*args)


def test_empty_and_singleton_angles():
    assert all(z.shape == (0,) for z in ultrapts(0, 0.3, angles=True))
    x, w, v, t = ultrapts(1, 0.3, angles=True)
    assert x[0] == 0 and w[0] > 0 and v[0] == 1 and t[0] == 1


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_special_cases(lam):
    x, w, v, t = ultrapts(12, lam, angles=True)
    assert jnp.max(jnp.abs(jnp.cos(t) - x)) < 2e-15
    assert jnp.max(jnp.abs(x + x[::-1])) < 2e-15
    assert jnp.max(jnp.abs(w - w[::-1])) < 2e-15
    assert jnp.all(v[::2] > 0) and jnp.all(v[1::2] < 0)


@pytest.mark.parametrize("method", ["REC", "gW"])
def test_explicit_methods(method):
    x, w, v = ultrapts(42, 0.3, method, bary=True)
    xr, wr, vr = ultrapts(42, 0.3, bary=True)
    assert jnp.max(jnp.abs(x - xr)) < 2e-14
    assert jnp.max(jnp.abs(w - wr)) < 2e-14
    assert jnp.max(jnp.abs(v - vr)) < 2e-14


def test_map_keeps_reference_angles_and_bary():
    x, w, v, t = ultrapts(42, 0.3, angles=True)
    y, z, u, s = ultrapts(42, 0.3, (0.0, 10.0), "rec", angles=True)
    assert jnp.max(jnp.abs(y - (x + 1) * 5)) < 2e-15
    assert jnp.max(jnp.abs(z - 5**0.6 * w)) < 2e-15
    assert jnp.array_equal(v, u) and jnp.array_equal(t, s)


@pytest.mark.parametrize(
    "lam,threshold", [(3.0, 100), (8.0, 500), (13.0, 1000), (20.0, 2000), (21.0, 3000)]
)
def test_source_default_thresholds(monkeypatch, lam, threshold):
    from chebfunjax.utils import ultraspherical_points as u

    calls = []

    def rec(n, lam):
        calls.append("rec")
        return jnp.zeros(n), jnp.ones(n), jnp.ones(n)

    def asy(n, lam):
        calls.append("asy")
        return jnp.zeros(n), jnp.ones(n), jnp.ones(n), jnp.zeros(n)

    monkeypatch.setattr(u, "_rec", rec)
    monkeypatch.setattr(u, "_asy", asy)
    ultrapts(threshold - 1, lam)
    ultrapts(threshold, lam)
    # Source method_set distinguishes an explicit 'default' from omission.
    ultrapts(threshold - 1, lam, "default")
    assert calls == ["rec", "asy", "asy"]


def test_asy_static_jit():
    from functools import partial

    import jax

    x, w, v = jax.jit(partial(ultrapts, 251, 0.8, bary=True))()
    xr, wr, vr = ultrapts(251, 0.8, bary=True)
    assert jnp.array_equal(x, xr)
    assert jnp.array_equal(w, wr)
    assert jnp.array_equal(v, vr)
