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

    def test_nan_raises(self):
        # FIXED (Fable 5): the tech constructor now extrapolates NaN/Inf
        # samples (MATLAB @chebtech/populate.m). An all-NaN sample leaves no
        # good points, so extrapolate raises 'Too many NaNs/Infs to handle.'
        with pytest.raises(Exception):
            _bf(lambda x: x + jnp.nan, n=17)

    def test_inf_raises(self):
        # FIXED (Fable 5): all-Inf sample -> extrapolate raises (see NaN case).
        with pytest.raises(Exception):
            _bf(lambda x: x + jnp.inf, n=17)

    def test_extrapolate_avoids_endpoints(self):
        # MATLAB pass(15): with extrapolate=1 the operator is never evaluated at
        # the domain endpoints (the mapped grid's +/-1 == physical a, b).
        # FIXED (Fable 5): Bndfun.from_function forwards extrapolate= to the
        # tech, which samples interior points only and extrapolates endpoints.
        def F(x):
            xa = np.asarray(x)
            if bool(np.any((xa == DOM.a) | (xa == DOM.b))):
                raise RuntimeError("Extrapolate should prevent endpoint evaluation.")
            return jnp.sin(x)

        # Must not raise (array-valued, endpoints extrapolated):
        Bndfun.from_function(
            lambda x: jnp.stack([F(x), F(x)], axis=-1), DOM, extrapolate=True
        )

    def test_singular_function(self):
        # Blowup at BOTH endpoints: exponents (-0.5, -1.6) (MATLAB bndfun
        # with data.exponents=[-0.5 -1.6], pref.blowup).  The right base
        # (x-b) is negative, so the fractional power is complex -- build with
        # a complex base to match MATLAB's complex singfun.  Endpoints are
        # poles (MATLAB samples random interior points), so compare interior.
        powl, powr = -0.5, -1.6

        def op(x):
            return (
                (x - DOM.a).astype(jnp.complex128) ** powl
                * (x - DOM.b).astype(jnp.complex128) ** powr
                * jnp.sin(x)
            )

        f = Bndfun.from_function(op, DOM, exponents=(powl, powr))
        xr = np.linspace(-2.0, 7.0, 100)[1:-1]
        vals_exact = (
            (xr - DOM.a).astype(complex) ** powl
            * (xr - DOM.b).astype(complex) ** powr
            * np.sin(xr)
        )
        err = np.asarray(f(jnp.asarray(xr))) - vals_exact
        assert float(np.max(np.abs(err))) < 1e3 * EPS * float(
            np.max(np.abs(vals_exact))
        )
