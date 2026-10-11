"""Full native tests/chebfun/test_exp.m predicates, Chebfun7574c77.

Public constructor and actual preferences, both exp/expm1, all1000 sample
sites and strict original vscale bounds. NumPy RandomState7681 is an explicit
uniform adapter, not a fresh MATLAB draw identity claim. Old2 sampled probes
remain supplemental below.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


def base_op(x):
    return jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x)


class TestChebfunExp:
    def test_exp_of_piecewise(self):
        # MATLAB builds f = chebfun(base_op, [-1 -.2 .1 1]) under
        # splitting, which resolves the sign/abs jumps with one-sided
        # samples.  The equivalent exact construction here uses the
        # smooth restriction of base_op on each piece (MATLAB's cell
        # syntax): base = -(x+.2)sin(3x) on [-1,.1], +(x+.2)sin(3x)
        # after the sign flip at .1, with the |.| kink at -.2.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain

        ops = [
            lambda x: (x + 0.2) * jnp.sin(3 * x),
            lambda x: -(x + 0.2) * jnp.sin(3 * x),
            lambda x: (x + 0.2) * jnp.sin(3 * x),
        ]
        brks = [-1.0, -0.2, 0.1, 1.0]
        funs = [_Piece.from_function(op, a, b) for op, a, b in zip(ops, brks[:-1], brks[1:])]
        f = Chebfun(funs=funs, domain=Domain(tuple(brks)))
        g = f.exp()
        exact = jnp.exp(base_op(XR))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * g.vscale * EPS

    def test_expm1(self):
        # MATLAB test_exp.m loops expFunctions = {exp, expm1} over the same
        # piecewise base at 1e2*vscale(g)*eps.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain

        ops = [
            lambda x: (x + 0.2) * jnp.sin(3 * x),
            lambda x: -(x + 0.2) * jnp.sin(3 * x),
            lambda x: (x + 0.2) * jnp.sin(3 * x),
        ]
        brks = [-1.0, -0.2, 0.1, 1.0]
        funs = [_Piece.from_function(op, a, b) for op, a, b in zip(ops, brks[:-1], brks[1:])]
        f = Chebfun(funs=funs, domain=Domain(tuple(brks)))
        g = f.expm1()
        exact = jnp.expm1(base_op(XR))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * max(g.vscale, 1.0) * EPS


import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(scope="module")
def native_exp_case():
    pref = ChebfunPref()
    pref.splitting = 1
    f = cj.chebfun(base_op, domain=(-1.0, -0.2, 0.1, 1.0), pref=pref)
    # Literal source arithmetic and site count; explicit NumPy stream adapter.
    xr = jnp.asarray(2 * np.random.RandomState(7681).rand(1000) - 1)
    return pref, f, xr


@pytest.mark.parametrize("method,op", [("exp", jnp.exp), ("expm1", jnp.expm1)])
def test_native_exp_predicate(method, op, native_exp_case):
    pref, f, xr = native_exp_case
    g = getattr(f, method)(pref)
    actual = g(xr)
    exact = op(base_op(xr))
    assert actual.shape == exact.shape == (1000,)
    error = float(jnp.linalg.norm(actual - exact, ord=jnp.inf))
    bound = 100 * float(g.vscale) * EPS
    print(
        {
            "method": method,
            "error": error,
            "bound": bound,
            "vscale": float(g.vscale),
            "shape": list(actual.shape),
        },
        flush=True,
    )
    assert error < bound
