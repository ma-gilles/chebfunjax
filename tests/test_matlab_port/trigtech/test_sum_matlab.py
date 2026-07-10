"""Port of MATLAB Chebfun tests/trigtech/test_sum.m (Opus 4.8).

Definite integral over [-1, 1] of a Fourier series equals 2*c_0.  Checked
against known closed-form integrals and the standard calculus identities
(linearity, integration-by-parts, fundamental theorem of calculus).

Provenance
----------
MATLAB source : tests/trigtech/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechSum:
    def test_integral_exp_sin(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(jnp.pi * x)) - 1)
        assert abs(float(f.sum()) - 0.532131755504017) < 10 * f.vscale * EPS

    def test_integral_rational(self):
        f = _tt(lambda x: 3.0 / (4 - jnp.cos(jnp.pi * x)))
        assert abs(float(f.sum()) - 1.549193338482967) < 10 * f.vscale * EPS

    def test_integral_high_frequency(self):
        f = _tt(lambda x: 1 + jnp.cos(1e4 * jnp.pi * x))
        exact = 2.0
        assert abs(float(f.sum()) - exact) / exact < 100 * f.vscale * EPS

    def test_integral_complex(self):
        f = _tt(lambda x: 1 + 1j * jnp.cos(40 * jnp.pi * x))
        exact = 2.0
        assert abs(complex(f.sum()) - exact) / exact < 10 * f.vscale * EPS

    def test_linearity(self):
        a, b = 2.0, -1j
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.cos(4 * jnp.sin(10 * jnp.pi * x)))
        tol_f = 10 * f.vscale * EPS
        tol_g = 10 * f.vscale * EPS
        lhs = complex((a * f + b * g).sum())
        rhs = a * complex(f.sum()) + b * complex(g.sum())
        assert abs(lhs - rhs) < max(tol_f, tol_g)

    def test_integration_by_parts(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        df = f.diff()
        g = _tt(lambda x: jnp.cos(4 * jnp.sin(10 * jnp.pi * x)))
        dg = g.diff()
        fg = f * g
        gdf = g * df
        fdg = f * dg
        tol_fg = 10 * fg.vscale * EPS
        tol_fdg = 10 * fdg.vscale * EPS
        tol_gdf = 10 * gdf.vscale * EPS
        lhs = complex(fdg.sum())
        rhs = complex(fg(jnp.array(1.0))) - complex(fg(jnp.array(-1.0))) - complex(gdf.sum())
        assert abs(lhs - rhs) < max(tol_fdg, tol_gdf, tol_fg)

    def test_ftc_f(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        df = f.diff()
        tol_f = 10 * f.vscale * EPS
        tol_df = 10 * df.vscale * EPS
        lhs = complex(df.sum())
        rhs = complex(f(jnp.array(1.0))) - complex(f(jnp.array(-1.0)))
        assert abs(lhs - rhs) < max(tol_df, tol_f)

    def test_ftc_g(self):
        g = _tt(lambda x: jnp.cos(4 * jnp.sin(10 * jnp.pi * x)))
        dg = g.diff()
        tol_g = 10 * g.vscale * EPS
        tol_dg = 10 * dg.vscale * EPS
        lhs = complex(dg.sum())
        rhs = complex(g(jnp.array(1.0))) - complex(g(jnp.array(-1.0)))
        assert abs(lhs - rhs) < max(tol_dg, tol_g)

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_valued_sum(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech and the sum DIM option")
    def test_sum_dim_option(self):
        raise AssertionError("sum(f, 2) DIM option not implemented")
