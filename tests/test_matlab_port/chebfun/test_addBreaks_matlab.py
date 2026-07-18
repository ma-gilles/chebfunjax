"""Port of MATLAB Chebfun tests/chebfun/test_addBreaks.m (Fable 5).

FIXED (Fable 5 audit): ``Chebfun.addBreaks(breaks)`` inserts breakpoints into
the domain without changing the represented function (scalar and array-valued).

Provenance
----------
MATLAB source : tests/chebfun/test_addBreaks.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)
_EXPECTED = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def _breakpoints(g):
    ivs = [np.asarray(fn.interval) for fn in g.funs]
    return [float(ivs[0][0])] + [float(iv[1]) for iv in ivs]


class TestChebfunAddbreaks:
    def test_scalar(self):
        # pass(1): addBreaks(sin, [-0.25 0.25]) inserts breaks, values unchanged.
        f = cj.chebfun(jnp.sin, domain=(-1, -0.5, 0, 0.5, 1))
        g = f.addBreaks([-0.25, 0.25])
        assert np.allclose(_breakpoints(g), _EXPECTED, atol=1e-14)
        assert float(jnp.max(jnp.abs(f(X) - g(X)))) < 10 * f.vscale * EPS

    def test_array_valued(self):
        # pass(2): array-valued [sin cos exp], same breakpoint insertion.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0, 0.5, 1),
        )
        g = f.addBreaks([-0.25, 0.25])
        assert g.n_columns == 3
        assert np.allclose(_breakpoints(g), _EXPECTED, atol=1e-14)
        assert float(jnp.max(jnp.abs(f(X) - g(X)))) < 10 * f.vscale * EPS
