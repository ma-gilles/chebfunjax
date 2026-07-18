"""Port of MATLAB Chebfun tests/diskfun/test_flipshiftrotate.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_flipshiftrotate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

# tol = 1000 * chebfun2eps (chebfun2eps default = eps = 2^-52)
TOL = 1000 * 2.220446049250313e-16


def _cart(fn):
    """diskfun(@(x,y) fn) with x = r cos(theta), y = r sin(theta)."""
    return Diskfun.from_function(
        lambda t, r: fn(r * jnp.cos(t), r * jnp.sin(t)))


def _norm_diff(a, b):
    th = np.linspace(-np.pi, np.pi, 45)
    rr = np.linspace(0.0, 1.0, 45)
    TH, RR = np.meshgrid(th, rr)
    da = np.asarray(a(jnp.asarray(TH), jnp.asarray(RR)))
    db = np.asarray(b(jnp.asarray(TH), jnp.asarray(RR)))
    return float(np.max(np.abs(da - db)))


class TestDiskfunFlipshiftrotate:
    def test_flips(self):
        f1 = lambda x, y: 10 * jnp.exp(-20 * (x - .5) ** 2 - 20 * (y + .5) ** 2)
        h1 = lambda x, y: 10 * jnp.exp(-20 * (-x - .5) ** 2 - 20 * (y + .5) ** 2)
        g1 = lambda x, y: 10 * jnp.exp(-20 * (x - .5) ** 2 - 20 * (-y + .5) ** 2)
        f = _cart(f1)
        h = _cart(h1)
        g = _cart(g1)

        def r1(x, y):
            return x * (f1(x, y) + h1(x, y) + g1(x, y)) + y

        w = _cart(r1)
        ud = _cart(lambda x, y: r1(x, -y))
        lr = _cart(lambda x, y: r1(-x, y))
        tp = _cart(lambda x, y: r1(y, x))

        assert _norm_diff(f.fliplr(), h) < TOL          # pass(1)
        assert _norm_diff(f.flipud(), g) < TOL          # pass(2)
        assert _norm_diff(w.flipud(), ud) < TOL         # pass(3)
        assert _norm_diff(w.fliplr(), lr) < TOL         # pass(4)
        assert _norm_diff(w.flipxy(), tp) < TOL         # pass(5)
        assert _norm_diff(w.flipdim(1), ud) < TOL       # pass(6)
        assert _norm_diff(w.flipdim(2), lr) < TOL       # pass(7)

    def test_rotate_circshift(self):
        def w(t, r):
            return jnp.exp(-20 * (r * jnp.cos(t) - .5) ** 2
                           - 20 * (r * jnp.sin(t)) ** 2)

        f = Diskfun.from_function(w)
        angles = [0.0, np.pi / 2, 5 * np.pi / 7,
                  -np.pi / 3, -np.pi / 4, -5 * np.pi]
        tg = np.linspace(-np.pi, np.pi, 30)
        rg = np.linspace(0.0, 1.0, 30)
        TG, RG = np.meshgrid(tg, rg)
        for th in angles:
            g = f.rotate(th)
            m = f.circshift(th)
            lhs = np.asarray(g(jnp.asarray(TG), jnp.asarray(RG)))
            rhs = np.asarray(w(jnp.asarray(TG - th), jnp.asarray(RG)))
            assert float(np.max(np.abs(lhs - rhs))) < TOL
            # rotate and circshift are the same operation
            assert _norm_diff(g, m) < TOL
