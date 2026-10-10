"""Native randnfun2 assertions plus deterministic source-constructor controls.

MATLAB source: tests/misc/test_randnfun2.m, randnfun2.m
Chebfun commit: 7574c77

Explicit JAX seeds are reproducible Python adapters. Their normal stream is
not claimed to match MATLAB's rng(0)/randn stream.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestRandnfun2:
    def test_all_12_native_assertions(self):
        # The native test resets rng(0) before each shape-invariance pair.
        # seed=0 restarts the corresponding deterministic JAX adapter stream.
        f = cj.randnfun2(.1, seed=0)
        # Native pass(1): abs(mean2(f.^2)-1) < .1.
        assert abs(float((f ** 2).mean2()) - 1) < .1
        # Native pass(2): abs(mean2(f)) < .1.
        assert abs(float(f.mean2())) < .1

        g = cj.randnfun2(.1, (2, 4, -1, 1), seed=0)
        # Native pass(3): shifted-domain sample equality, original 1e-12 bound.
        assert abs(float(f(jnp.asarray(.5), jnp.asarray(.5)))
                   - float(g(jnp.asarray(3.5), jnp.asarray(.5)))) < 1e-12

        h = cj.randnfun2(.2, (-2, 2, 0, 4), seed=0)
        # Native pass(4): doubled lambda/domain sample equality, 1e-12.
        assert abs(float(f(jnp.asarray(.5), jnp.asarray(.5)))
                   - float(h(jnp.asarray(1.), jnp.asarray(3.)))) < 1e-12

        # Native pass(5): norm(diff(f),'fro') < 1e-4.
        f = cj.randnfun2(1e6, seed=0)
        assert float(f.diff().norm("fro")) < 1e-4

        f = cj.randnfun2(.1, "trig", seed=0)
        # Native pass(6): abs(mean2(f.^2)-1) < .1.
        assert abs(float((f ** 2).mean2()) - 1) < .1
        # Native pass(7): abs(mean2(f)) < .1.
        assert abs(float(f.mean2())) < .1

        g = cj.randnfun2(.1, "trig", (2, 4, -1, 1), seed=0)
        # Native pass(8): shifted trig-domain sample equality, 1e-12.
        assert abs(float(f(jnp.asarray(.5), jnp.asarray(.5)))
                   - float(g(jnp.asarray(3.5), jnp.asarray(.5)))) < 1e-12

        h = cj.randnfun2("trig", .2, (-2, 2, 0, 4), seed=0)
        # Native pass(9): doubled trig lambda/domain equality, 1e-12.
        assert abs(float(f(jnp.asarray(.5), jnp.asarray(.5)))
                   - float(h(jnp.asarray(1.), jnp.asarray(3.)))) < 1e-12

        f = cj.randnfun2(6, "trig", seed=0)
        # Native pass(10): norm(diff(f),'fro') == 0 exactly.
        assert float(f.diff().norm("fro")) == 0

        # Native pass(11): norm(sqrt(10)*standard - big) == 0 exactly.
        f1 = jnp.sqrt(10.) * cj.randnfun2(.1, seed=0)
        f2 = cj.randnfun2(.1, "big", seed=0)
        assert float((f1 - f2).norm()) == 0

        # Native pass(12): norm(big - norm) == 0 exactly.
        f3 = cj.randnfun2(.1, "norm", seed=0)
        assert float((f2 - f3).norm()) == 0

    def test_centered_trig_coefficients_match_independent_cosine(self, monkeypatch):
        # A fixed coefficient is injected into the same JAX draw adapter used
        # by production. This is a phase/sign control, not an RNG fixture.
        import chebfunjax.utils._randnfun as draw_engine

        calls = []

        def fixed_normal(_key, rows, columns):
            assert (rows, columns) == (3, 3)
            draw = np.zeros((rows, columns))
            if not calls:
                draw[1, 2] = 1.0  # c[ky=0,kx=+1]
            calls.append(draw.copy())
            return jnp.asarray(draw)

        monkeypatch.setattr(draw_engine, "_normal_draw", fixed_normal)
        f = cj.randnfun2(2.0, (-1, 1, -1, 1), seed=0, trig=True)
        x = jnp.asarray([0.0, .25, .5, .75, 1.0])
        y = jnp.zeros_like(x)
        expected = jnp.cos(jnp.pi * x)
        assert len(calls) == 2
        assert jnp.max(jnp.abs(f(x, y) - expected)) < 1e-13

    def test_native_flexible_flag_and_domain_order(self):
        reference = cj.randnfun2(.1, (2, 4, -1, 1), seed=17, trig=True)
        flags_first = cj.randnfun2("trig", (2, 4, -1, 1), .1, seed=17)
        assert float((reference - flags_first).norm()) == 0

    def test_infinite_lambda_constant_and_big_zero(self):
        f = cj.randnfun2(float("inf"), seed=31)
        assert float(f.diff().norm("fro")) == 0
        g = cj.randnfun2("big", float("inf"), seed=31)
        assert float(g.norm("fro")) == 0

    def test_native_scalar_predicate_accepts_singleton_arrays(self):
        reference = cj.randnfun2(.1, seed=41)
        for singleton in (np.asarray([.1]), np.asarray([[.1]]),
                          jnp.asarray([.1])):
            candidate = cj.randnfun2(singleton, seed=41)
            assert float((candidate - reference).norm()) == 0

    def test_infinite_lambda_uses_one_real_draw(self, monkeypatch):
        import chebfunjax.utils._randnfun as draw_engine

        calls = []

        def fixed_normal(key, rows, columns):
            assert (rows, columns) == (1, 1)
            calls.append(key)
            return jnp.asarray([[2.5]])

        monkeypatch.setattr(draw_engine, "_normal_draw", fixed_normal)
        f = cj.randnfun2(float("inf"), "big", seed=47)
        assert len(calls) == 1
        assert float(f(0.0, 0.0)) == 0.0
