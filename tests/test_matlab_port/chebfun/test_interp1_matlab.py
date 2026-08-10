"""Port of MATLAB Chebfun tests/chebfun/test_interp1.m (Fable 5).

chebfunjax interp1 produces the global polynomial interpolant;
MATLAB's 'linear'/'pchip'/'spline' modes are separate methods (pchip/
spline exist on Chebfun; 'linear' has no counterpart).

Provenance
----------
MATLAB source : tests/chebfun/test_interp1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunInterp1:
    def test_polynomial_interpolant_hits_data(self):
        x = jnp.arange(11.0)
        y = jnp.sin(x)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).interp1(x, y)
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 1e4 * EPS

    def test_linear_mode(self):
        # MATLAB pass(1-3): 10 two-point pieces hitting the data exactly.
        x = jnp.arange(11.0)
        y = jnp.sin(x)
        f = cj.Chebfun.interp1(x, y, "linear")
        assert float(np.linalg.norm(np.asarray(f(x) - y))) < 10 * EPS
        assert len(f.funs) == 10
        assert len(f) == 20

    def test_linear_mode_array_valued(self):
        # MATLAB pass(4-6): array-valued data.
        x = jnp.arange(11.0)
        y = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = cj.Chebfun.interp1(x, y, "linear")
        assert float(np.linalg.norm(np.asarray(f(x) - y))) < 10 * EPS
        assert len(f.funs) == 10
        assert len(f) == 20

    def test_linear_mode_different_domain(self):
        # MATLAB pass(7-10): a domain wider than the data range adds
        # breakpoints at the domain endpoints.
        dom = (0.01, 10.01)
        x = jnp.arange(11.0)
        y = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = cj.Chebfun.interp1(x, y, "linear", dom)
        brk = np.asarray(f.domain.breakpoints)
        exact_brk = np.concatenate([[dom[0]], np.arange(1.0, 11.0),
                                    [dom[1]]])
        assert np.allclose(brk, exact_brk)
        xi = x[1:-1]
        err = np.linalg.norm(np.asarray(f(xi) - y[1:-1, :]))
        assert float(err) < 10 * EPS
        assert len(f.funs) == 11
        assert len(f) == 22

    def test_linear_mode_random_sites(self):
        # MATLAB pass(22-23): unsorted/random sites are sorted internally.
        rng = np.random.default_rng(6178)
        x = jnp.asarray(rng.uniform(size=11))
        y = jnp.sin(x)
        f = cj.Chebfun.interp1(x, y, "linear")
        assert float(np.linalg.norm(np.asarray(f(x) - y))) < 10 * EPS
        assert len(f.funs) == 10

    def test_unknown_method_raises(self):
        # MATLAB errors 'CHEBFUN:CHEBFUN:interp1:method'.
        x = jnp.arange(11.0)
        with pytest.raises(ValueError):
            cj.Chebfun.interp1(x, jnp.sin(x), "bogus")

    # FIXED (Fable 5, Big-Three array-valued epic): interp1 now takes
    # (n, m) y-data column-wise, so the polynomial-mode array case
    # ports (data-hit check at the same 1e4*eps scale as the scalar
    # polynomial test).
    def test_array_valued(self):
        x = jnp.arange(11.0)
        y = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).interp1(x, y)
        assert f.n_columns == 2
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 1e4 * EPS
