"""Port of MATLAB Chebfun tests/bndfun/test_constructor.m (Opus 4.8).

Self-validating where possible.  Several MATLAB assertions require features
chebfunjax does not have (array-valued funs, NaN/Inf detection, the
`extrapolate` preference, singular funs) and are marked xfail with a precise
reason.

Provenance
----------
MATLAB source : tests/bndfun/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


class TestBndfunConstructor:
    def test_scalar_interpolation_removable_singularity(self):
        # sin(x)/x is 1 at x = 0 (removable); the constructor must recover it.
        g = _bf(lambda x: jnp.sin(x) / x)
        assert abs(1 - float(g(jnp.float64(0.0)))) < 10 * g.vscale * EPS

    def test_array_valued_interpolation(self):
        # pass(2): array-valued [sin(x)/x  sin(x-3)/(x-3)] recovers the two
        # removable singularities (value 1 at x=0 and x=3 respectively).
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        g = _bf(
            lambda x: jnp.stack(
                [jnp.sin(x) / x, jnp.sin(x - 3) / (x - 3)], axis=-1
            )
        )
        gv = np.concatenate(
            [np.asarray(g(jnp.float64(0.0))), np.asarray(g(jnp.float64(3.0)))]
        )
        # MATLAB checks gv(1) (col 0 at x=0) and gv(4) (col 1 at x=3).
        assert float(np.max(np.abs(np.ones(2) - np.array([gv[0], gv[3]])))) < (
            10 * g.vscale * EPS
        )

    @pytest.mark.xfail(
        reason="chebfunjax does not detect/raise on NaN function values; the "
        "adaptive constructor never converges and returns an unhappy fun "
        "instead of raising 'Too many NaNs/Infs to handle.'"
    )
    def test_nan_raises(self):
        with pytest.raises(Exception):
            _bf(lambda x: x + jnp.nan, n=17)

    @pytest.mark.xfail(
        reason="chebfunjax does not detect/raise on Inf function values "
        "(see NaN case)."
    )
    def test_inf_raises(self):
        with pytest.raises(Exception):
            _bf(lambda x: x + jnp.inf, n=17)

    @pytest.mark.skip(
        reason="chebfunjax has no `extrapolate` preference and no array-valued "
        "Bndfun: there is no equivalent knob to assert the MATLAB behaviour "
        "(construct [F(x) F(x)] while avoiding endpoint |x|==1 evaluation)."
    )
    def test_extrapolate_avoids_endpoints(self):
        def F(x):
            if bool(np.any(np.abs(np.asarray(x)) == 1)):
                raise RuntimeError("Extrapolate should prevent endpoint evaluation.")
            return jnp.sin(x)

        _bf(lambda x: jnp.stack([F(x), F(x)], axis=-1))

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: "
        "(x-a)^-0.5 (x-b)^-1.6 sin(x) cannot be constructed."
    )
    def test_singular_function(self):
        powl, powr = -0.5, -1.6

        def op(x):
            return (x - DOM.a) ** powl * (x - DOM.b) ** powr * jnp.sin(x)

        f = _bf(op, n=17)
        xr = np.linspace(-2.0, 7.0, 100)
        vals_exact = op(xr)
        err = np.asarray(f(jnp.asarray(xr))) - vals_exact
        assert float(np.max(np.abs(err))) < 1e3 * EPS * float(
            np.max(np.abs(vals_exact))
        )
