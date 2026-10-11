"""Source public REMEZ and TRIGREMEZ adapter controls, Chebfun7574c77."""

import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun


@pytest.mark.parametrize("rational", [False, True])
def test_remez_forwards_original_chebfun_and_source_domain(monkeypatch, rational):
    module = importlib.import_module("chebfunjax.utils.remez")
    f = cj.chebfun(lambda x: 1 + x, domain=(-2.0, 3.0))
    calls = []
    pair = (object(), object())

    def provider(original, m, **kwargs):
        calls.append((original, m, kwargs))
        if rational:
            return SimpleNamespace(as_chebfuns=lambda: pair)
        return SimpleNamespace(coeffs=jnp.asarray([1.0, 2.0]), domain=(-2.0, 3.0))

    monkeypatch.setattr(module, "minimax", provider)
    with pytest.warns(UserWarning, match="CHEBFUN:CHEBFUN:remez:.*deprecated"):
        result = cj.remez(f, 1, 1, silent=True) if rational else cj.remez(f, 1, silent=True)
    assert calls[0][0] is f and calls[0][1] == 1
    assert calls[0][2]["domain"] == (-2.0, 3.0)
    assert calls[0][2]["silent"] is True
    if rational:
        assert result is pair
        assert calls[0][2]["rational"] is True and calls[0][2]["denom"] == 1
    else:
        assert isinstance(result, Chebfun)
        np.testing.assert_array_equal(result.coeffs, [1.0, 2.0])


def test_trigremez_source_default_forwarding(monkeypatch):
    module = importlib.import_module("chebfunjax.utils.trigrational")
    f = cj.chebfun(lambda x: 1 + 0 * x, trig=True)
    seen = []
    sentinel = object()

    def rational(*args):
        seen.append(args)
        return sentinel

    monkeypatch.setattr(module, "_trigremez_rational", rational)
    assert cj.trigremez(f, 4, 1) is sentinel
    assert seen[0][3] == 100
    assert seen[0][4] == 1e-15 * (4 + 1 + 11)


def test_trigremez_actual_nonconvergence_warning():
    f = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), trig=True)
    with pytest.warns(
        UserWarning, match="CHEBFUN:CHEBFUN:trigremez:convergence:.*after 0 iterations"
    ):
        p, _, _ = cj.trigremez(f, 4, max_iter=0)
    assert isinstance(p, Chebfun)
    assert len(p) == 9


def test_remez_real_polynomial_output():
    f = cj.chebfun(lambda x: x * x)
    with pytest.warns(UserWarning, match="CHEBFUN:CHEBFUN:remez:.*deprecated"):
        p = cj.remez(f, 2)
    x = jnp.asarray([-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(p(x), x * x, rtol=0, atol=1e-14)


def test_trigremez_native_default_norm_forwarding(monkeypatch):
    module = importlib.import_module("chebfunjax.utils.trigrational")
    f = cj.chebfun(lambda x: 1 + x * 0, trig=True)
    seen = []
    original = Chebfun.norm

    def record(self, *args, **kwargs):
        seen.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chebfun, "norm", record)
    monkeypatch.setattr(module, "_trigremez_rational", lambda *args: args[-1])
    result = cj.trigremez(f, 4, 1)
    assert seen == [((), {})]
    assert result == float(original(f))
