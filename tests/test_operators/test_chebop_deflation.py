"""Core mirror tests for Chebop deflation (Fable 5).

Exercises the deflation surface outside the MATLAB-port tree so the core
coverage gate sees it: the deflation factor (L2 + H1), the exact
product-rule Newton Jacobian, an end-to-end second-root find, the
multi-start fallback, and the guard / routing paths.

Provenance
----------
MATLAB source : @chebop/deflate.m, @chebmatrix/deflationFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.operators.chebop import (
    Chebop,
    _chebfun_to_values,
    _deflation_factor,
    _make_deflated_op,
    _normalize_deflation_roots,
    deflate,
)
from chebfunjax.utils.quadrature import chebpts

DOM = (0.0, 1.0)


def _mk(fn, n=16):
    return chebfun(fn, domain=DOM, n=n)


class TestDeflationFactor:
    def test_l2_single_root(self):
        u = _mk(lambda x: jnp.sin(3.0 * x))
        r = _mk(lambda x: 0.2 * x)
        p, alp = 3.0, 0.1
        got = _deflation_factor(u, [r], p, alp, "L2")
        nrm = float((u - r).norm(2))
        expected = 1.0 / nrm**p + alp
        assert math.isclose(got, expected, rel_tol=1e-11)

    def test_l2_multiple_roots(self):
        u = _mk(lambda x: jnp.cos(2.0 * x) + 0.5)
        r0 = _mk(lambda x: 0.1 + 0.0 * x)
        r1 = _mk(lambda x: 0.3 * x)
        p, alp = 2.0, 0.0
        got = _deflation_factor(u, [r0, r1], p, alp, "L2")
        prod = float((u - r0).norm(2)) ** 2 * float((u - r1).norm(2)) ** 2
        expected = 1.0 / prod ** (p / 2.0) + alp
        assert math.isclose(got, expected, rel_tol=1e-11)

    def test_h1_single_root(self):
        u = _mk(lambda x: jnp.sin(2.0 * x))
        r = _mk(lambda x: 0.4 * x)
        p, alp = 2.0, 0.2
        got = _deflation_factor(u, [r], p, alp, "H1")
        ur = u - r
        s = float(ur.norm(2)) ** 2 + float(ur.diff().norm(2)) ** 2
        expected = 1.0 / s ** (p / 2.0) + alp
        assert math.isclose(got, expected, rel_tol=1e-11)


class TestDeflatedOp:
    def test_op_output_is_factor_times_original(self):
        # The deflated operator returns M(u; r) * N(u) exactly.
        op = lambda x, u: u.diff(2) + u * u  # noqa: E731
        r = _mk(lambda x: 0.5 * x)
        dop = _make_deflated_op(op, [r], 2.0, 0.1, "L2")
        u = _mk(lambda x: jnp.sin(3.0 * x))
        xf = Chebfun.identity(Domain(DOM))
        M = _deflation_factor(u, [r], 2.0, 0.1, "L2")
        got = dop(xf, u)
        expected = op(xf, u) * M
        pts = jnp.linspace(0.0, 1.0, 11)
        assert np.allclose(np.asarray(got(pts)), np.asarray(expected(pts)),
                           rtol=1e-10, atol=1e-12)

    def test_normalize_roots_forms(self):
        r = _mk(lambda x: x)
        assert _normalize_deflation_roots(r) == [r]
        assert _normalize_deflation_roots([r, r]) == [r, r]
        # An object exposing `.blocks` (e.g. SystemSolution) is unwrapped.

        class _WithBlocks:
            blocks = [r]

        assert _normalize_deflation_roots(_WithBlocks()) == [r]


class TestExactJacobian:
    def test_jacobian_matches_finite_differences(self):
        # Exact product-rule Jacobian J_G = M J_N + N (x) dM/du of a deflated
        # operator equals a finite-difference Jacobian of the full deflated
        # residual.
        N = Chebop(lambda x, u: u.diff(2) + 2.0 * u.exp(), DOM, 0.0, 0.0)
        r0 = _mk(lambda x: 0.3 * jnp.sin(jnp.pi * x))
        Nd = deflate(N, r0, 1, 0)

        sz = 12
        disc = ChebColloc2Disc(sz, DOM)
        dom = Domain(DOM)
        rng = np.random.default_rng(0)
        uv = jnp.asarray(rng.standard_normal(sz) * 0.3, dtype=jnp.float64)
        ufun = Chebfun.from_values(uv, dom)
        x_fun = Chebfun.identity(dom)

        def defl_res(u_np):
            uf = Chebfun.from_values(jnp.asarray(u_np), dom)
            return np.asarray(_chebfun_to_values(Nd._apply_op(x_fun, uf), disc))

        Nu_v = jnp.asarray(defl_res(np.asarray(uv)))
        J_exact = np.asarray(
            Nd._jacobian_matrix_deflated(disc, x_fun, ufun, Nu_v))

        u0 = np.asarray(uv, dtype=float)
        h = 1e-6
        base = defl_res(u0)
        J_fd = np.zeros((sz, sz))
        for j in range(sz):
            up = u0.copy()
            up[j] += h
            J_fd[:, j] = (defl_res(up) - base) / h

        rel = np.max(np.abs(J_exact - J_fd)) / np.max(np.abs(J_fd))
        assert rel < 1e-5

    def test_jacobian_used_via_dispatch(self):
        # _jacobian_matrix routes deflated ops to the exact deflated Jacobian.
        N = Chebop(lambda x, u: u.diff(2) + u * u, DOM, 0.0, 0.0)
        r0 = _mk(lambda x: 0.2 * x)
        Nd = deflate(N, r0, 1, 0)
        disc = ChebColloc2Disc(10, DOM)
        dom = Domain(DOM)
        x_fun = Chebfun.identity(dom)
        ufun = _mk(lambda x: 0.1 * jnp.sin(jnp.pi * x), n=10)
        Nu = np.asarray(_chebfun_to_values(Nd._apply_op(x_fun, ufun), disc))
        J = Nd._jacobian_matrix(disc, x_fun, ufun, jnp.asarray(Nu))
        direct = Nd._jacobian_matrix_deflated(
            disc, x_fun, ufun, jnp.asarray(Nu))
        assert np.allclose(np.asarray(J), np.asarray(direct))


class TestEndToEnd:
    def test_bratu_second_root(self):
        # Deflation drives Newton to a distinct second solution of Bratu.
        N = Chebop(lambda x, u: u.diff(2) + 2.0 * u.exp(), DOM, 0.0, 0.0)
        r0 = N.solve(0.0)
        r1 = deflate(N, r0, 1, 0).solve(0.0)
        assert float(N(r0).norm()) < 1e-8
        assert float(N(r1).norm()) < 1e-8
        assert float((r0 - r1).norm()) > 1.0

    def test_method_matches_free_function(self):
        N = Chebop(lambda x, u: u.diff(2) + 2.0 * u.exp(), DOM, 0.0, 0.0)
        r0 = N.solve(0.0)
        a = deflate(N, r0, 1, 0).solve(0.0)
        b = N.deflate(r0, 1, 0).solve(0.0)
        pts = jnp.linspace(0.0, 1.0, 21)
        assert np.allclose(np.asarray(a(pts)), np.asarray(b(pts)),
                           rtol=1e-8, atol=1e-9)

    def test_multistart_fallback_exhausts(self):
        # Bratu has exactly two solutions below the critical parameter;
        # deflating BOTH leaves no third root, so the multi-start fallback
        # sweeps every candidate and warns rather than returning a stale root.
        # A small fixed size and iteration cap keep the exhaustive sweep cheap.
        N = Chebop(lambda x, u: u.diff(2) + 2.0 * u.exp(), DOM, 0.0, 0.0)
        r0 = N.solve(0.0)
        r1 = deflate(N, r0, 1, 0).solve(0.0)
        with pytest.warns(UserWarning, match="could not locate a new"):
            r2 = deflate(N, [r0, r1], 1, 0).solve(0.0, n=12, max_iter=3)
        # A Chebfun is still returned (best effort).
        assert isinstance(r2, Chebfun)


class TestGuardsAndRouting:
    def test_fresh_op_has_no_deflation(self):
        N = Chebop(lambda x, u: u.diff(2) + u, DOM, 0.0, 0.0)
        assert N._deflation is None

    def test_deflate_sets_state_and_makes_nonlinear(self):
        # Deflating even a linear operator yields a nonlinear one.
        N = Chebop(lambda x, u: u.diff(2), DOM, 0.0, 0.0)
        assert N._is_linear() is True
        r = _mk(lambda x: jnp.sin(jnp.pi * x))
        Nd = deflate(N, r, 1, 0)
        assert Nd._deflation is not None
        assert Nd._is_linear() is False

    def test_deflate_carries_bcs_and_init(self):
        N = Chebop(lambda x, u: u.diff(2) + u * u, DOM)
        N.lbc = 0.0
        N.rbc = 1.0
        init = _mk(lambda x: x)
        N.init = init
        r = _mk(lambda x: 0.5 * x)
        Nd = deflate(N, r, 2, 0.1)
        assert Nd._lbc_raw == 0.0
        assert Nd._rbc_raw == 1.0
        assert Nd.init is init
        assert Nd.domain == N.domain

    def test_default_init_variants(self):
        x_pts = 0.5 * chebpts(8, kind=2) + 0.5
        # Both Dirichlet endpoints -> straight line through them.
        N = Chebop(lambda x, u: u.diff(2), DOM, 0.0, 2.0)
        v = np.asarray(N._deflation_default_init_vals(x_pts))
        assert np.allclose(v, np.asarray(x_pts) * 2.0)
        # Single endpoint -> constant.
        N2 = Chebop(lambda x, u: u.diff(2), DOM)
        N2.lbc = 3.0
        v2 = np.asarray(N2._deflation_default_init_vals(x_pts))
        assert np.allclose(v2, 3.0)
        N3 = Chebop(lambda x, u: u.diff(2), DOM)
        N3.rbc = -1.0
        v3 = np.asarray(N3._deflation_default_init_vals(x_pts))
        assert np.allclose(v3, -1.0)
        # Callable / no scalar BC -> zeros.
        N4 = Chebop(lambda x, u: u.diff(2), DOM)
        N4.lbc = lambda u: u.diff()
        v4 = np.asarray(N4._deflation_default_init_vals(x_pts))
        assert np.allclose(v4, 0.0)

    def test_const_candidates_span_root_range(self):
        N = Chebop(lambda x, u: u.diff(2) + u * u, DOM, 0.0, 0.0)
        r0 = _mk(lambda x: 1.0 + 0.0 * x)
        r1 = _mk(lambda x: -2.0 + 0.0 * x)
        Nd = deflate(N, [r0, r1], 1, 0)
        cands = Nd._deflation_const_candidates()
        assert len(cands) == 9
        # Extends well beyond the [-2, 1] root value range on both sides.
        assert min(cands) < -2.0
        assert max(cands) > 1.0

    def test_deflate_rejects_system(self):
        N = Chebop(lambda x, u, v: [u.diff() + v, v.diff() + u], DOM)
        r = _mk(lambda x: x)
        with pytest.raises(ValueError, match="scalar"):
            deflate(N, r, 1, 0)

    def test_deflate_rejects_bad_type(self):
        N = Chebop(lambda x, u: u.diff(2) + u * u, DOM, 0.0, 0.0)
        r = _mk(lambda x: x)
        with pytest.raises(ValueError, match="norm type"):
            deflate(N, r, 1, 0, type="L7")

    def test_deflate_requires_op(self):
        N = Chebop(domain=DOM)
        r = _mk(lambda x: x)
        with pytest.raises(ValueError, match="operator is not set"):
            deflate(N, r, 1, 0)
