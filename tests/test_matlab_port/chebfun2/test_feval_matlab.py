"""Port of MATLAB Chebfun tests/chebfun2/test_feval.m (Fable 5).

Ported through pass(11) (scalar/vector/meshgrid evaluation on default
and stretched domains, complex-valued f).  pass(12+) evaluate along
chebfun paths f(c) — chebfunjax Chebfun2 cannot compose with a Chebfun.

Provenance
----------
MATLAB source : tests/chebfun2/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS
R = 0.126986816293506
S = 0.632359246225410


class TestChebfun2Feval:
    def test_identity_x(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f(jnp.asarray(np.pi / 6), jnp.asarray(np.pi / 12)))
                   - np.pi / 6) < TOL

    def test_identity_x_stretched(self):
        dom = (-1.0, 2.0, -np.pi / 2, np.pi)
        f = Chebfun2.from_function(lambda x, y: x, domain=dom)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0)))) < 1e-14
        assert abs(float(f(jnp.asarray(np.pi / 6), jnp.asarray(np.pi / 12)))
                   - np.pi / 6) < TOL

    def test_identity_y_stretched(self):
        dom = (-1.0, 2.0, -np.pi / 2, np.pi)
        f = Chebfun2.from_function(lambda x, y: y, domain=dom)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0)))) < 1e-14
        assert abs(float(f(jnp.asarray(np.pi / 6), jnp.asarray(np.pi / 12)))
                   - np.pi / 12) < TOL

    def test_scalar_vector_meshgrid(self):
        def fa(x, y):
            return jnp.cos(x) + jnp.sin(x * y)
        g = Chebfun2.from_function(fa)
        assert abs(float(fa(jnp.asarray(R), jnp.asarray(S))
                         - g(jnp.asarray(R), jnp.asarray(S)))) < TOL
        rng = np.random.default_rng(0)
        r = jnp.asarray(rng.uniform(0, 1, 10))
        s = jnp.asarray(rng.uniform(0, 1, 10))
        assert float(jnp.max(jnp.abs(fa(r, s) - g(r, s)))) < TOL
        rr, ss = jnp.meshgrid(r, s)
        assert float(jnp.max(jnp.abs(fa(rr, ss) - g(rr, ss)))) < TOL

    def test_strange_domain(self):
        dom = (-np.pi / 6, np.pi / 2, -np.pi / 12, float(np.sqrt(3)))

        def fa(x, y):
            return jnp.cos(x) + jnp.sin(x * y)
        g = Chebfun2.from_function(fa, domain=dom)
        assert abs(float(fa(jnp.asarray(R), jnp.asarray(S))
                         - g(jnp.asarray(R), jnp.asarray(S)))) < TOL
        rng = np.random.default_rng(1)
        r = jnp.asarray(rng.uniform(0, np.pi / 2, 10))
        s = jnp.asarray(rng.uniform(0, np.sqrt(3), 10))
        assert float(jnp.max(jnp.abs(fa(r, s) - g(r, s)))) < TOL
        rr, ss = jnp.meshgrid(r, s)
        assert float(jnp.max(jnp.abs(fa(rr, ss) - g(rr, ss)))) < 2 * TOL

    def test_complex_valued(self):
        f = Chebfun2.from_function(lambda x, y: x + 1j * y)
        v = complex(f(jnp.asarray(0.25), jnp.asarray(-0.5)))
        assert abs(v - (0.25 - 0.5j)) < TOL

    def test_eval_along_chebfun_path(self):
        pytest.skip("Chebfun2 cannot compose with a Chebfun path f(c(t))")
