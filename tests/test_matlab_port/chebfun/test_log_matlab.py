"""Port of MATLAB Chebfun tests/chebfun/test_log.m (Fable 5).

All five native variants use public constructor/preferences and strict native
vscale bounds. NumPy RandomState7681 uniform sites are an explicit stream
adapter; no fresh MATLAB draw-equivalence claim. Older controls remain
supplemental below.

Provenance
----------
MATLAB source : tests/chebfun/test_log.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


def base_op(x):
    return 2 + jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x)


def _pw_base():
    ops = [
        lambda x: 2 + (x + 0.2) * jnp.sin(3 * x),
        lambda x: 2 - (x + 0.2) * jnp.sin(3 * x),
        lambda x: 2 + (x + 0.2) * jnp.sin(3 * x),
    ]
    brks = [-1.0, -0.2, 0.1, 1.0]
    funs = [_Piece.from_function(op, a, b) for op, a, b in zip(ops, brks[:-1], brks[1:])]
    return Chebfun(funs=funs, domain=Domain(tuple(brks)))


class TestChebfunLog:
    def test_log_of_piecewise(self):
        f = _pw_base()
        g = f.log()
        exact = jnp.log(base_op(XR))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * max(g.vscale, 1.0) * EPS

    @pytest.mark.parametrize(
        "meth,ref",
        [("log10", np.log10), ("log1p", np.log1p), ("log2", np.log2), ("reallog", np.log)],
    )
    def test_log_variants(self, meth, ref):
        # MATLAB test_log.m loops logFunctions = {log, log10, log1p, log2,
        # reallog} over the same piecewise base, at 1e2*vscale(g)*eps.
        f = _pw_base()
        g = getattr(f, meth)()
        exact = jnp.asarray(ref(np.asarray(base_op(XR))))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * max(g.vscale, 1.0) * EPS

    def test_reallog_complex_raises(self):
        # MATLAB reallog.m errors 'CHEBFUN:CHEBFUN:reallog:complex'.
        import chebfunjax as cj

        f = cj.chebfun(lambda x: 2.0 + 1j * x)
        with pytest.raises(ValueError):
            f.reallog()


@pytest.fixture(scope="module")
def native_log_case():
    pref = ChebfunPref()
    pref.splitting = 1
    f = chebfun(base_op, domain=(-1.0, -0.2, 0.1, 1.0), pref=pref)
    # Source seed7681 and1000 sites; adapter draws are explicit, not MATLAB refs.
    xr = jnp.asarray(2 * np.random.RandomState(7681).rand(1000) - 1)
    return pref, f, xr


class TestChebfunLogNative:
    @pytest.mark.parametrize(
        "method,op",
        [
            ("log", jnp.log),
            ("log10", jnp.log10),
            ("log1p", jnp.log1p),
            ("log2", jnp.log2),
            ("reallog", jnp.log),
        ],
    )
    def test_source_predicate(self, method, op, native_log_case):
        pref, f, xr = native_log_case
        g = getattr(f, method)(pref)
        values = np.asarray(g(xr))
        exact = np.asarray(op(base_op(xr)))
        assert values.shape == exact.shape == (1000,)
        err = np.linalg.norm(values - exact, np.inf)
        bound = 1e2 * g.vscale * EPS
        print({"method": method, "error": err, "bound": bound, "vscale": g.vscale}, flush=True)
        assert err < bound
