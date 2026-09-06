"""Port of MATLAB Chebfun tests/chebfun/test_trigratinterp.m (Fable 5).

``trigratinterp`` returns the approximant handle ``r`` together with the
numerator / denominator Fourier coefficient vectors; MATLAB's ``p`` and
``q`` are the trigonometric polynomials with those coefficients (built
here as trig chebfuns), so ``length(p) == 2*mu + 1``.

Provenance
----------
MATLAB source : tests/chebfun/test_trigratinterp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import trigpts
from chebfunjax.utils.ratapprox import trigratinterp

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


def _pq(f, m, n, NN=None, xi=None, tol=1e-14, domain=(-1.0, 1.0)):
    r, a, b, mu, nu, _poles, _res = trigratinterp(
        f, m, n, NN, xi, tol=tol, domain=domain)
    p = chebfun(jnp.asarray(a), coeffs=True, trig=True, domain=domain)
    q = chebfun(jnp.asarray(b), coeffs=True, trig=True, domain=domain)
    return p, q, r


def _tp(n):
    return np.asarray(trigpts(n)[0] if isinstance(trigpts(n), tuple) else trigpts(n))


def _pq_at(p, q, x):
    x = jnp.asarray(x)
    return np.asarray(p(x)) / np.asarray(q(x))


class TestChebfunTrigratinterp:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(6178)
        m, n = 1, 1
        N = m + n
        th = np.sort(-1 + 2 * rng.rand(2 * N + 1))
        fh = lambda t: np.tan(np.pi * t)  # noqa: E731
        fk = fh(th)
        p, q, r = _pq(jnp.asarray(fk), m, n, None, jnp.asarray(th), tol=0.0)
        assert len(p) == 2 * m + 1                                  # pass(1)
        assert len(q) == 2 * n + 1                                  # pass(2)
        assert np.max(np.abs(_pq_at(p, q, th) - fh(th))) < TOL      # pass(3)
        tt = 2 * rng.rand(100) - 1
        assert np.max(np.abs(_pq_at(p, q, tt) - fh(tt))) < 1e3 * TOL  # pass(4)

        N = 101
        p, q, r = _pq(lambda t: jnp.tan(np.pi * t), m, n, N, None, tol=0.0)
        assert len(p) == 2 * m + 1                                  # pass(5)
        assert len(q) == 2 * n + 1                                  # pass(6)
        assert np.max(np.abs(_pq_at(p, q, th) - fh(th))) < TOL      # pass(7)
        tt = 2 * rng.rand(100) - 1
        assert np.max(np.abs(_pq_at(p, q, tt) - fh(tt))) < 1e3 * TOL  # pass(8)

        m, n = 10, 15
        N = m + n
        t = chebfun("t")
        f = (np.pi * t).sin() / (np.pi * t).cos()
        th = _tp(2 * N + 1)
        fk = np.asarray(f(jnp.asarray(th)))
        p, q, r = _pq(jnp.asarray(fk), m, n)
        assert len(p) == 3                                          # pass(9)
        assert len(q) == 3                                          # pass(10)
        assert np.max(np.abs(_pq_at(p, q, th) - fh(th))) < 1e3 * TOL  # pass(11)
        tt = 2 * rng.rand(100) - 1
        assert np.max(np.abs(_pq_at(p, q, tt) - fh(tt))) < 1e3 * TOL  # pass(12)

        m, n = 5, 5
        N = m + n
        fh = lambda t: np.exp(-4 * np.sin(np.pi * t / 2) ** 2)  # noqa: E731
        f = chebfun(lambda t: jnp.exp(-4 * jnp.sin(np.pi * t / 2) ** 2), trig=True)
        th = -1 + 2 * rng.rand(2 * N + 1)
        p, q, r = _pq(f, m, n, None, jnp.asarray(th), tol=0.0)
        assert len(p) == 2 * m + 1                                  # pass(13)
        assert len(q) == 2 * n + 1                                  # pass(14)
        th = _tp(2 * N + 1)
        assert np.max(np.abs(_pq_at(p, q, th) - fh(th))) < 5e-10    # pass(15)

        m, n = 10, 15
        N = m + n
        fh = lambda t: np.cos(np.pi * t) / np.cos(2 * np.pi * t)  # noqa: E731
        th = _tp(2 * N + 1)
        fk = fh(th)
        p, q, r = _pq(jnp.asarray(fk), m, n)
        assert len(p) == 3                                          # pass(16)
        assert len(q) == 5                                          # pass(6, again)
        assert np.max(np.abs(_pq_at(p, q, th) - fh(th))) < 1e2 * TOL  # pass(17)
        tt = 2 * rng.rand(100) - 1
        assert np.max(np.abs(_pq_at(p, q, tt) - fh(tt))) < 1e3 * TOL  # pass(18)

        fh = lambda x: np.exp(np.sin(x))  # noqa: E731
        a, b = 2.0, 2.0 + 2 * np.pi
        f = chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(a, b), trig=True)
        xi = 2 + 2 * np.pi / 51 * np.arange(51)
        p, q, r = _pq(f, 6, 6, None, jnp.asarray(xi), domain=(a, b))
        xx = np.linspace(a, b, 10001)
        assert np.max(np.abs(fh(xx) - np.asarray(r(jnp.asarray(xx))))) < 10 * TOL  # pass(19)

        fi = np.array([2.0, 3.0, 1.0])
        xi = np.array([-1.0, 0.5, 0.8])
        p, q, r = _pq(jnp.asarray(fi), 1, 0, 3, jnp.asarray(xi))
        assert np.max(np.abs(_pq_at(p, q, xi) - fi)) < 10 * TOL     # pass(20)
        assert np.max(np.abs(np.asarray(r(jnp.asarray(xi))) - fi)) < 10 * TOL  # pass(21)
        assert len(p) == 3                                          # pass(22)
        assert len(q) == 1                                          # pass(23)
