"""Port of MATLAB Chebfun tests/chebfun/test_constructor_splitting.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_splitting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps
TOL = 1e9


def _check(f, F, n=100):
    bp = list(f.domain.breakpoints)
    xx = jnp.asarray(np.linspace(bp[0] + EPS, bp[-1] - EPS, n))
    err = float(np.max(np.abs(np.asarray(f(xx)) - np.asarray(F(xx)))))
    return err < TOL * float(np.max(np.atleast_1d(EPS * np.asarray(f.vscale))))


class TestChebfunConstructorSplitting:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(6178)

        F1 = jnp.sqrt
        f1 = chebfun(F1, domain=(0.0, 1.0), splitting=True, blowup=False)
        assert _check(f1, F1)                                       # pass(1)

        F2 = lambda x: jnp.sqrt(1 - x)  # noqa: E731
        f2 = chebfun(F2, domain=(0.0, 1.0), splitting=True, blowup=False)
        assert _check(f2, F2)                                       # pass(2)

        F3 = lambda x: jnp.sqrt(1 - x ** 2)  # noqa: E731
        f3 = chebfun(F3, domain=(-1.0, 1.0), splitting=True, blowup=False)
        assert _check(f3, F3)                                       # pass(3)

        F4 = lambda x: jnp.stack([jnp.sin(x), jnp.sign(x)], axis=-1)  # noqa: E731
        f4 = chebfun(F4, domain=(-1.0, 1.0), splitting=True, blowup=False)
        assert np.max(np.abs(np.asarray(list(f4.domain.breakpoints)) - [-1, 0, 1])) < 10 * EPS
        assert _check(f4, F4)                                       # pass(4)

        F5 = lambda x: jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x)  # noqa: E731
        f5 = chebfun(F5, domain=(-1.0, 1.0), splitting=True, blowup=False)
        assert np.max(np.abs(np.asarray(list(f5.domain.breakpoints))
                             - [-1, -0.2, 0.1, 1])) < 10 * EPS
        assert _check(f5, F5)                                       # pass(5)

        f = chebfun(lambda x: x > 0, domain=(-1.0, 1.0), splitting=True)
        x = chebfun("x", domain=(-1.0, 1.0))
        h = cj.heaviside(x)
        assert float((f - h).norm(2)) < 10 * EPS                    # pass(6)

        f = chebfun([lambda x: jnp.abs(x - 0.25), 0], domain=(0.0, 0.5, 1.0),
                    splitting=True)
        xx1 = jnp.asarray(np.linspace(0 + EPS, 0.5 - EPS, 20))
        err1 = float(np.max(np.abs(np.asarray(f(xx1)) - np.abs(np.asarray(xx1) - 0.25))))
        xx2 = jnp.asarray(np.linspace(0.5 + EPS, 1 - EPS, 20))
        err2 = float(np.max(np.abs(np.asarray(f(xx2)))))
        assert max(err1, err2) < 10 * f.vscale * EPS                # pass(7)

        dom = (0.0, jnp.inf)
        dom_check = (0.0, 100.0)
        x = np.sort((dom_check[1] - dom_check[0]) * rng.rand(100) + dom_check[0])
        op = lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x)  # noqa: E731
        f = chebfun(op, domain=dom, splitting=True)
        err = np.asarray(f(jnp.asarray(x))) - np.asarray(op(jnp.asarray(x)))
        assert np.max(np.abs(err)) < 1e6 * EPS * f.vscale           # pass(8)

        op = jnp.tan
        f = chebfun(op, domain=(-4.0, 4.0), splitting=True, blowup=1)
        dom = [-4, -np.pi / 2, np.pi / 2, 4]
        xs = [np.asarray((dom[k + 1] - dom[k]) * rng.rand(100) + dom[k]) for k in range(3)]
        err = np.concatenate([np.asarray(op(jnp.asarray(xk))) - np.asarray(f(jnp.asarray(xk)))
                              for xk in xs])
        assert np.max(np.abs(err)) < 1e5 * EPS * f.vscale           # pass(9)

        op = lambda x: (jnp.sin(100 * x) / jnp.exp(x ** 2) + 1) * x ** 2  # noqa: E731
        dom_test = (-200.0, 200.0)
        x = jnp.asarray((dom_test[1] - dom_test[0]) * rng.rand(100) + dom_test[0])
        f = chebfun(op, domain=(-jnp.inf, jnp.inf), exps=(2, 2), splitting=True)
        assert np.max(np.abs(np.asarray(f(x)) - np.asarray(op(x)))) < 1e5 * EPS * f.vscale  # pass(10)
