"""Port of MATLAB Chebfun tests/trigtech/test_minandmax.m (Opus 4.8[1m]).

minandmax(f) returns the global minimum and maximum (values and positions).

Provenance
----------
MATLAB source : tests/trigtech/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _spotcheck(fn, exact_min, exact_max):
    f = _tt(fn)
    (mn, mnp), (mx, mxp) = f.minandmax()
    y = np.array([np.asarray(mn), np.asarray(mx)])
    y_exact = np.array([exact_min, exact_max])
    positions = jnp.stack((jnp.asarray(mnp), jnp.asarray(mxp)))
    fx = np.asarray(fn(positions))
    vs = f.vscale
    return (np.max(np.abs(y - y_exact)) < 100 * vs * EPS
            and np.max(np.abs(fx - y_exact)) < 10 * vs * EPS)


class TestTrigtechMinandmax:
    def test_exp_neg_cos(self):
        assert _spotcheck(lambda x: jnp.exp(-jnp.cos(2 * jnp.pi * x)),
                          np.exp(-1), np.exp(1))

    def test_sin10(self):
        assert _spotcheck(lambda x: jnp.sin(10 * jnp.pi * x), -1, 1)

    def test_exp_sin100(self):
        assert _spotcheck(lambda x: jnp.exp(jnp.sin(jnp.pi * x) ** 100),
                          1, np.exp(1))

    def test_exp_neg_sin100(self):
        assert _spotcheck(lambda x: jnp.exp(-jnp.sin(jnp.pi * x) ** 100),
                          np.exp(-1), 1)

    def test_sign_approx(self):
        def fn(x):
            return 4 / jnp.pi * (
                jnp.sin(jnp.pi * x) + 1 / 3 * jnp.sin(3 * jnp.pi * x)
                + 1 / 5 * jnp.sin(5 * jnp.pi * x)
                + 1 / 7 * jnp.sin(7 * jnp.pi * x)
                + 1 / 9 * jnp.sin(9 * jnp.pi * x))
        assert _spotcheck(fn, -1.182328208857607, 1.182328208857607)

    def test_array_valued(self):
        def fun(x):
            return jnp.stack([
                jnp.exp(-jnp.cos(2 * jnp.pi * x)),
                jnp.sin(10 * jnp.pi * x),
                jnp.exp(-jnp.sin(jnp.pi * (x - 0.32)) ** 100)], axis=-1)
        f = _tt(fun)
        (mn, mnp), (mx, mxp) = f.minandmax()
        y = np.array([np.asarray(mn), np.asarray(mx)])
        y_exact = np.array([[np.exp(-1), -1, np.exp(-1)],
                            [np.exp(1), 1, 1]])
        scale_eps = np.max(np.asarray(f.vscale) * EPS)
        assert np.max(np.abs(y - y_exact)) < 100 * scale_eps
        for k in range(3):
            positions = jnp.stack((jnp.asarray(mnp[k]), jnp.asarray(mxp[k])))
            fx = np.asarray(fun(positions))
            # Native rejects only >: equality passes. Preserve its rejection
            # form also rather than changing the NaN comparison semantics.
            assert not (np.max(np.abs(fx[:, k] - y_exact[:, k])) > 10 * scale_eps)

    def test_complex_array_valued(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.exp(jnp.sin(2 * jnp.pi * x)), 1j * jnp.cos(20 * jnp.pi * x)],
            axis=-1))
        (mn, _), (mx, _) = f.minandmax()
        vals = np.stack([np.asarray(mn), np.asarray(mx)])
        f1 = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)))
        (mn1, _), (mx1, _) = f1.minandmax()
        f2 = _tt(lambda x: 1j * jnp.cos(20 * jnp.pi * x))
        (mn2, _), (mx2, _) = f2.minandmax()
        ref = np.stack([[complex(np.asarray(mn1)), complex(np.asarray(mn2))],
                        [complex(np.asarray(mx1)), complex(np.asarray(mx2))]])
        # Native matrix infinity norm is maximum absolute row sum.
        assert np.linalg.norm(np.abs(vals) - np.abs(ref), ord=np.inf) \
            < 1e2 * np.max(np.asarray(f.vscale) * EPS)
