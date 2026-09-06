"""Port of MATLAB Chebfun tests/misc/test_gpr.m (Fable 5).

MATLAB's chebfun-valued ``[f, fvar, fsamples] = gpr(...)`` is
``gpr_chebfun``; MATLAB's ``rng`` seeds map to ``numpy.random.seed`` (the
normal streams differ, so the noisy cases are statistical).

Provenance
----------
MATLAB source : tests/misc/test_gpr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.gpr import gpr_chebfun

jax.config.update("jax_enable_x64", True)


def _ev(f, x):
    return np.asarray(f(jnp.asarray(np.asarray(x, dtype=float))))


def _n(f):
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


class TestMiscGpr:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(23411)
        xx = np.linspace(-1, 1, 10) + 2e-2 * rng.randn(10)
        yy = np.exp(xx) * np.sin(2 * xx)
        f, _, fs = gpr_chebfun(xx, yy, n_samples=2, rng=np.random.default_rng(0))
        assert np.max(np.abs(_ev(f, xx) - yy)) < 1e-6                # pass(1)
        assert np.max(np.abs(_ev(fs.extract_columns(0), xx) - yy)) < 5e-4  # pass(2)
        assert np.max(np.abs(_ev(fs.extract_columns(1), xx) - yy)) < 5e-4  # pass(3)

        yn = yy + .1 * rng.randn(10)
        f, _, _ = gpr_chebfun(xx, yn, noise=.1)
        assert np.std(_ev(f, xx) - yn) < .25                        # pass(4)

        xs = 2e-100 * xx - 1e-100
        ys = rng.randn(10)
        f, _, _ = gpr_chebfun(xs, ys)
        assert np.max(np.abs(_ev(f, xs) - ys)) < 1e-6                # pass(5)
        yb = 1e100 * rng.randn(10)
        f, _, _ = gpr_chebfun(xs, yb)
        assert np.max(np.abs((_ev(f, xs) - yb) / yb)) < 1e-6         # pass(6)

        xx = np.linspace(-1, 1, 10)
        yy = 1e100 * np.sin(3 * xx) * np.exp(xx)
        f, _, _ = gpr_chebfun(xx, yy, sigma=1e100, length_scale=.1)
        assert np.max(np.abs((_ev(f, xx) - yy) / yy)) < 1e-10        # pass(7)
        f, _, _ = gpr_chebfun(xx, yy, sigma=1e100)
        assert np.max(np.abs((_ev(f, xx) - yy) / yy)) < 1e-10        # pass(8)
        f, _, _ = gpr_chebfun(xx, yy, length_scale=.1)
        assert np.max(np.abs((_ev(f, xx) - yy) / yy)) < 1e-10        # pass(9)
        f, _, _ = gpr_chebfun(xx, yy)
        assert np.max(np.abs((_ev(f, xx) - yy) / yy)) < 1e-3         # pass(10)

        x = np.arange(1, 6) ** 2
        y = np.sin(x)
        S = 1e50
        f, _, _ = gpr_chebfun(x, y)
        fbig, _, _ = gpr_chebfun(x, S * y)
        assert _n(f - fbig / S) < 5e-14                             # pass(11)
        fsmall, _, _ = gpr_chebfun(x, y / S)
        assert _n(f - S * fsmall) < 5e-14                           # pass(12)

        fbig, _, _ = gpr_chebfun(S * x, y)
        xx = np.linspace(1, 25, 50)
        assert np.linalg.norm(_ev(f, xx) - _ev(fbig, S * xx)) < 5e-14  # pass(13)
        fsmall, _, _ = gpr_chebfun(x / S, y)
        assert np.linalg.norm(_ev(f, xx) - _ev(fsmall, xx / S)) < 5e-14  # pass(14)

        N = 40
        xx = np.linspace(-1, 1, N)
        xx[1:-1] = xx[1:-1] + 1e-3 * rng.randn(N - 2)
        yy = np.exp(np.sin(np.pi * xx))
        f, _, fs = gpr_chebfun(xx, yy, domain=(-1.0, 1.0), trig=True, n_samples=2,
                               rng=np.random.default_rng(1))
        assert np.max(np.abs(_ev(f, xx) - yy)) < 1e-6                # pass(15)
        assert np.max(np.abs(_ev(fs.extract_columns(0), xx) - yy)) < 5e-4  # pass(16)
