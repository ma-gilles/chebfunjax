"""Port of MATLAB Chebfun tests/chebfun/test_unwrap.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_unwrap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunUnwrap:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().unwrap().isempty()

    def test_smooth_unchanged(self):
        f = cj.chebfun(jnp.exp)
        uf = f.unwrap()
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        err = jnp.abs(uf(xs) - f(xs))
        assert float(jnp.max(err)) < 100 * EPS * f.vscale

    def test_sawtooth_unwraps_to_line(self):
        # angle of e^{ix} on multiple periods wraps at +-pi; unwrap
        # restores the line x.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        brks = [0.0, np.pi, 3 * np.pi, 5 * np.pi, 6 * np.pi]

        def saw(x):
            return jnp.mod(x + np.pi, 2 * np.pi) - np.pi
        funs = [_Piece.from_function(
            lambda x, a=a, b=b: x - 2 * np.pi *
            jnp.round((0.5 * (a + b)) / (2 * np.pi)), a, b)
            for a, b in zip(brks[:-1], brks[1:])]
        f = Chebfun(funs=funs, domain=Domain(tuple(brks)))
        uf = f.unwrap()
        xs = jnp.asarray(np.linspace(0.1, 6 * np.pi - 0.1, 50))
        err = jnp.abs(uf(xs) - xs)
        assert float(jnp.max(err)) < 1e-8
