"""Port of MATLAB Chebfun tests/chebfun/test_definePoint.m (Fable 5).

MATLAB's ``f(s) = v`` subsasgn is ``f.define_point(s, v)``.

Provenance
----------
MATLAB source : tests/chebfun/test_definePoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _dom(f):
    return list(f.domain.breakpoints)


def _pv(f):
    return np.asarray(f.point_values)


class TestChebfunDefinePoint:
    def test_all_matlab_assertions(self):
        f = chebfun(lambda x: x, domain=(-1.0, 0.0, 1.0))
        f = f.define_point(0.5, 1.0)
        assert _dom(f) == [-1, 0, .5, 1] and float(f(jnp.asarray(0.5))) == 1  # pass(1)
        f = f.define_point(-0.5, 2.0)
        assert _dom(f) == [-1, -.5, 0, .5, 1] and float(f(jnp.asarray(-0.5))) == 2  # pass(2)

        two = lambda x: jnp.stack([x, x], axis=-1)  # noqa: E731
        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point(0.5, [1.0, 2.0])
        assert _dom(f) == [-1, 0, .5, 1] and \
            np.array_equal(np.asarray(f(jnp.asarray(0.5))), [1, 2])  # pass(3)
        f = f.define_point(-0.5, [2.0, 3.0])
        assert _dom(f) == [-1, -.5, 0, .5, 1] and \
            np.array_equal(np.asarray(f(jnp.asarray(-0.5))), [2, 3])  # pass(4)

        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], [[1, 2], [3, 4]])
        assert _pv(f).shape == (5, 2) and np.array_equal(
            _pv(f), [[-1, -1], [1, 2], [0, 0], [3, 4], [1, 1]])       # pass(5)
        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], [[1, 2], [3, 4]])
        assert np.array_equal(
            _pv(f), [[-1, -1], [1, 2], [0, 0], [3, 4], [1, 1]])       # pass(6)

        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], [1, 2])
        assert _pv(f).shape == (5, 2) and np.array_equal(
            _pv(f), [[-1, -1], [1, 2], [0, 0], [1, 2], [1, 1]])       # pass(7)
        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], [1, 2])
        assert np.array_equal(
            _pv(f), [[-1, -1], [1, 2], [0, 0], [1, 2], [1, 1]])       # pass(8)

        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], 1)
        assert _pv(f).shape == (5, 2) and np.array_equal(
            _pv(f), [[-1, -1], [1, 1], [0, 0], [1, 1], [1, 1]])       # pass(9)
        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_point([-0.25, 0.5], 1)
        assert np.array_equal(
            _pv(f), [[-1, -1], [1, 1], [0, 0], [1, 1], [1, 1]])       # pass(10)

        # Singular pieces (exps).
        dom = (-2.0, -1.0, 0.0, 1.0)
        ops = [jnp.sin, lambda x: 1.0 / (1.0 + x) ** 0.5, lambda x: x + 1.0]
        f = chebfun(ops, domain=dom, exps=(0, 0, -0.5, 0, 0, 0))
        brk = [-0.7, -0.5]
        g = f.define_point(brk[0], 1.0)
        g = g.define_point(brk[1], 2.0)
        fpv = _pv(f)
        assert _dom(g) == sorted(set(list(dom) + brk))               # pass(11)
        assert float(g(jnp.asarray(brk[0]))) == 1
        assert float(g(jnp.asarray(brk[1]))) == 2
        assert np.array_equal(_pv(g), np.concatenate([fpv[:2], [1, 2], fpv[2:]]))

        # Unbounded domain.
        f = chebfun(lambda x: 1 - jnp.exp(-x ** 2), domain=(-jnp.inf, jnp.inf))
        brk = [-7.0, 2.0]
        g = f.define_point(brk[0], 3.0)
        g = g.define_point(brk[1], 4.0)
        fpv = _pv(f)
        assert _dom(g) == [-np.inf, -7, 2, np.inf]                  # pass(12)
        assert float(g(jnp.asarray(brk[0]))) == 3
        assert float(g(jnp.asarray(brk[1]))) == 4
        assert np.array_equal(_pv(g), [fpv[0], 3, 4, fpv[-1]])

        f = chebfun(0.0, domain=(-1.0, 1.0))
        f = f.define_point(1.0, 1.0)
        assert abs(float(f(jnp.asarray(1.0))) - 1) < EPS             # pass(13)
