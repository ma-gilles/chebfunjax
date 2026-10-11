"""Port of MATLAB Chebfun tests/chebfun/test_complex.m (Fable 5).

All twelve source predicates are independent below; existing two controls remain
supplemental. NumPy RandomState(7681) is an explicit source-shaped uniform-stream
adapter, not a fresh MATLAB draw-equivalence claim.

Provenance
----------
MATLAB source : tests/chebfun/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = np.finfo(float).eps
_RNG = np.random.RandomState(7681)
_XR = jnp.asarray(2 * _RNG.rand(1000) - 1)
_SING_X = jnp.asarray(8.8 * _RNG.rand(100) - 1.9)
_UNBOUNDED_X = jnp.asarray((-3 * np.pi + 1e6) * _RNG.rand(100) - 1e6)

XS = jnp.asarray(np.linspace(-0.95, 0.95, 50))


class TestChebfunComplex:
    def test_basic(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        z = cj.complex_fun(f, g)
        exact = np.sin(np.asarray(XS)) + 1j * np.cos(np.asarray(XS))
        assert np.max(np.abs(np.asarray(z(XS)) - exact)) < 1e-14

    def test_complex_input_rejected(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * x))
        g = cj.chebfun(jnp.cos)
        with pytest.raises(ValueError):
            cj.complex_fun(f, g)


class TestChebfunComplexNative:
    @pytest.mark.parametrize("clause", range(1, 13), ids=lambda n: f"native_{n:02d}")
    def test_source_clause(self, clause):
        if clause in (1, 2, 3):
            empty = cj.chebfun()
            f = cj.chebfun(jnp.sin, domain=(-1.0, 1.0), splitting=True)
            args = (empty, empty) if clause == 1 else (f, empty) if clause == 2 else (empty, f)
            assert cj.complex_fun(*args).isempty()
        elif clause == 4:
            ref = cj.chebfun(jnp.cos, domain=(-1.0, 0.0, 1.0), splitting=True)
            imf = cj.chebfun(jnp.sin, domain=(-1.0, 0.0, 1.0), splitting=True)
            f = cj.complex_fun(ref, imf)
            assert np.max(np.abs(np.asarray(f(_XR) - jnp.exp(1j * _XR)))) < 1e2 * f.vscale * EPS
        elif clause in (5, 6, 7, 8):
            ref = cj.chebfun(
                lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x)], axis=-1),
                domain=(-1.0, 0.0, 1.0),
                splitting=True,
            )
            imf = cj.chebfun(
                lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1),
                domain=(-1.0, 0.0, 1.0),
                splitting=True,
            )
            alpha = -0.194758928283640
            if clause == 6:
                assert ref.isequal(cj.complex_fun(ref))
                return
            if clause == 5:
                f = cj.complex_fun(ref, imf)
                exact = jnp.stack([jnp.exp(1j * _XR), 1j * jnp.exp(1j * _XR)], axis=-1)
                multiplier = 1e2
            elif clause == 7:
                f = cj.complex_fun(ref, alpha)
                exact = jnp.stack([jnp.cos(_XR), -jnp.sin(_XR)], axis=-1) + 1j * alpha
                multiplier = 10
            else:
                f = cj.complex_fun(alpha, imf)
                exact = alpha + 1j * jnp.stack([jnp.sin(_XR), jnp.cos(_XR)], axis=-1)
                multiplier = 1e2
            values = np.asarray(f(_XR))
            assert values.shape == (1000, 2)
            assert (
                np.linalg.norm((values - np.asarray(exact)).ravel(), np.inf)
                < multiplier * f.vscale * EPS
            )
        elif clause in (9, 10):
            f = cj.chebfun(jnp.sin, domain=(-1.0, 1.0))
            g = cj.chebfun(lambda x: jnp.exp(1j * x), domain=(-1.0, 1.0))
            identifier = "AisNotReal" if clause == 9 else "BisNotReal"
            with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:complex:" + identifier):
                cj.complex_fun(g, f) if clause == 9 else cj.complex_fun(f, g)
        elif clause == 11:
            dom = (-2.0, 7.0)
            power = -1.64
            f = cj.chebfun(
                lambda x: jnp.cos(x) * (x - dom[0]) ** power,
                domain=dom,
                exps=(power, 0.0),
                splitting=True,
            )
            g = cj.chebfun(
                lambda x: jnp.sin(x) * (x - dom[0]) ** power,
                domain=dom,
                exps=(power, 0.0),
                splitting=True,
            )
            h = cj.complex_fun(f, g)
            exact = jnp.exp(1j * _SING_X) * (_SING_X - dom[0]) ** power
            err = np.linalg.norm(np.asarray(h(_SING_X) - exact), np.inf)
            bound = 1e2 * EPS * np.linalg.norm(np.asarray(exact), np.inf)
            print({"clause": 11, "error": err, "bound": bound}, flush=True)
            assert err < bound
        else:
            dom = (-np.inf, -3 * np.pi)
            f = cj.chebfun(lambda x: x * jnp.exp(x), domain=dom)
            g = cj.chebfun(lambda x: (1 - jnp.exp(x)) / x, domain=dom)
            h = cj.complex_fun(f, g)
            exact = (
                _UNBOUNDED_X * jnp.exp(_UNBOUNDED_X)
                + 1j * (1 - jnp.exp(_UNBOUNDED_X)) / _UNBOUNDED_X
            )
            err = np.linalg.norm(np.asarray(h(_UNBOUNDED_X) - exact), np.inf)
            bound = EPS * h.vscale
            print({"clause": 12, "error": err, "bound": bound}, flush=True)
            assert err < bound
