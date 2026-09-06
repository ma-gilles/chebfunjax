"""Port of MATLAB Chebfun tests/chebfun/test_end.m (Fable 5).

MATLAB's ``end`` inside a chebfun subscript denotes the right endpoint of
the domain (continuous dimension) or the last column (discrete
dimension).  ``f(end)`` is ``f.end``; ``f(:, end)`` is
``f.extract_columns(-1)``; ``f(end, 2)`` / ``f(0, end)`` index the
evaluated row.

Provenance
----------
MATLAB source : tests/chebfun/test_end.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _normest(f):
    x = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
    return float(np.max(np.abs(np.asarray(f(x)))))


class TestChebfunEnd:
    def test_all_matlab_assertions(self):
        f = chebfun(lambda x: jnp.sin(np.pi * x))
        out = f.end
        assert isinstance(out, float) and abs(out) < EPS            # pass(1)

        three = lambda x: jnp.stack([jnp.sin(np.pi * x), jnp.cos(np.pi * x), x], axis=-1)  # noqa: E731
        f = chebfun(three)
        out = np.asarray(f.end)
        assert out.shape == (3,) and np.max(np.abs(out - [0, -1, 1])) < EPS  # pass(2)
        out = float(f.end[1])                                       # f(end, 2)
        assert abs(out + 1) < EPS                                   # pass(3)
        g = f.T
        out = np.asarray(g.end)
        assert out.shape == (3,) and np.max(np.abs(out - [0, -1, 1])) < EPS  # pass(4)

        out = float(f(jnp.asarray(0.0))[-1])                        # f(0, end)
        assert abs(out) < EPS                                       # pass(5)
        out = float(f(jnp.asarray(0.0))[-1])                        # pass(6)
        assert abs(out) < EPS

        out = np.asarray(f.end)                                     # f(end, :)
        assert out.shape == (3,) and np.max(np.abs(out - [0, -1, 1])) < EPS  # pass(7)
        out = f.extract_columns(2)                                  # f(:, end)
        x = chebfun(lambda t: t)
        assert _normest(out - x) < EPS                              # pass(8)
        g = f.T
        out = g.extract_columns(2).T                                # g(end, :)
        assert _normest(out - x.T) < EPS                            # pass(9)
        out = np.asarray(g.end)                                     # g(:, end)
        assert out.shape == (3,) and np.max(np.abs(out - [0, -1, 1])) < EPS  # pass(10)

        dom = (-2.0, 7.0)
        pow_ = -0.5
        # MATLAB's (x - 7)^(-0.5) is complex left of 7; JAX needs the cast.
        f = chebfun(lambda x: jnp.asarray(x - dom[1], jnp.complex128) ** pow_
                    * jnp.sin(100 * x),
                    domain=dom, exps=(0, pow_), splitting=True)
        out = f.end
        assert np.isscalar(out)                                     # pass(11)

        f = chebfun(lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x),
                    domain=(0.0, jnp.inf), splitting=True)
        out = f.end
        assert np.isscalar(out) and abs(out - 0.75) < 1e2 * EPS     # pass(12)
