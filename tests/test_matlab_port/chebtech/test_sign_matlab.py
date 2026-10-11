"""Literal8 native tech SIGN predicates, Chebfun7574c77.

Native8 use normest,50site grid,actual f evaluation and numeric matrixinf.
Supplemental source/API controls distinguish first-kind policy forwarding
from second-kind endpoint extrapolation. Retained probes follow below.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechSign:
    def test_sign_positive(self, Tech):
        # pass(type, 1): sign(sin(x) + 2) == 1.
        f = Tech.from_function(lambda x: jnp.sin(x) + 2.0)
        h = f.sign()
        assert float((h - 1.0).norm(jnp.inf)) < 10 * EPS

    def test_sign_negative(self, Tech):
        # pass(type, 2): sign(-(sin(x) + 2)) == -1.
        f = Tech.from_function(lambda x: -(jnp.sin(x) + 2.0))
        h = f.sign()
        assert float((h + 1.0).norm(jnp.inf)) < 10 * EPS

    def test_sign_complex(self, Tech):
        # pass(type, 3): sign(exp(1i pi x)) == exp(1i pi x).
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        h = f.sign()
        assert float((h - f).norm(jnp.inf)) < 1e2 * EPS

    def test_sign_complex_array(self, Tech):
        # pass(type, 4): complex array-valued sign == f./|f| pointwise.
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))

        def F(x):
            return jnp.stack(
                [
                    (2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    -(2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    (2 + jnp.sin(x)).astype(jnp.complex128),
                ],
                axis=-1,
            )

        f = Tech.from_function(F)
        ff = F(xx)
        gg = ff / jnp.abs(ff)
        h = f.sign()
        hh = h(xx)
        assert _ninf(hh - gg) < 10 * EPS


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("branch", ["positive", "negative", "complex", "array"])
def test_native_tech_sign(Tech, branch):
    if branch == "positive":
        f = Tech.from_function(lambda x: jnp.sin(x) + 2)
        e = float((f.sign() - 1).normest())
        bound = 10 * EPS
    elif branch == "negative":
        f = Tech.from_function(lambda x: -(jnp.sin(x) + 2))
        e = float((f.sign() + 1).normest())
        bound = 10 * EPS
    elif branch == "complex":
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        e = float((f.sign() - f).normest())
        bound = 100 * EPS
    else:
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 50))

        def op(x):
            v = (2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x)
            return jnp.stack((v, -v, 2 + jnp.sin(x)), axis=-1)

        f = Tech.from_function(op)
        ff = f(xx)
        expected = ff / jnp.abs(ff)
        actual = f.sign()(xx)
        assert actual.shape == expected.shape == (50, 3)
        e = float(jnp.linalg.norm(actual - expected, ord=jnp.inf))
        bound = 10 * EPS
    print({"native_type": Tech.__name__, "branch": branch, "error": e, "bound": bound}, flush=True)
    assert e < bound


def test_source_tech1_sign_extrapolate_forwarding(monkeypatch):
    calls = []
    original = Chebtech1.compose

    def observe(self, *args, **kwargs):
        calls.append(kwargs.get("extrapolate", False))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chebtech1, "compose", observe)
    f = Chebtech1.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
    f.sign()
    assert calls == [True]


def test_source_tech1_compose_extrapolate_api():
    f = Chebtech1.from_function(lambda x: (1 + x) * (1 + 1j))
    h = f.compose(lambda x: x / jnp.abs(x), extrapolate=True)
    phase = (1 + 1j) / jnp.sqrt(2.0)
    error = float(jnp.max(jnp.abs(h(jnp.asarray([-1.0, 0.0, 1.0])) - phase)))
    print({"supplement": "tech1_extrapolate_api", "endpoint_phase_error": error}, flush=True)
    assert error < 100 * EPS


def test_source_tech1_stored_zero_and_limits():
    import chebfunjax as cj
    from chebfunjax.chebpref import ChebfunPref

    pref = ChebfunPref()
    pref.tech = Chebtech1
    f = cj.chebfun(lambda x: (1 + 1j) * x, domain=(-1, 0, 1), pref=pref)
    assert all(isinstance(p.tech, Chebtech1) for p in f.funs)
    h = f.sign()
    assert h.domain.breakpoints == (-1.0, 0.0, 1.0)
    assert bool(jnp.all(h.point_values == jnp.sign(f.point_values)))
    assert complex(h(jnp.asarray(0.0))) == 0
    phase = (1 + 1j) / jnp.sqrt(2.0)
    e = max(
        float(jnp.max(jnp.abs(h.funs[0].tech(jnp.asarray([-1.0, 1.0])) + phase))),
        float(jnp.max(jnp.abs(h.funs[1].tech(jnp.asarray([-1.0, 1.0])) - phase))),
    )
    print(
        {"supplement": "tech1_stored_zero_limits", "stored_zero": True, "limit_error": e},
        flush=True,
    )
    assert e < 100 * EPS
