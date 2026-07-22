"""Core tests for Chebfun-level unbounded-domain support.

Mirrors (outside ``tests/test_matlab_port/``) the chebfun-factory routing of
infinite-endpoint domains to :class:`~chebfunjax.fun.unbndfun.Unbndfun` pieces,
and the Chebfun operations that must keep the unbounded mapping intact
(``n_columns``, ``extract_columns``, ``assign_columns``, ``mat2cell``,
``repmat``, ``real``/``imag``/``conj``, ``any``, ``isinf``, ``sum``, ``diff``).

These are independent of MATLAB golden references: results are checked against
closed-form values or against direct evaluation of the operand function.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _finite_pts(a, b, n=50, seed=0):
    """Random points in a finite check-window inside an unbounded domain."""
    lo = -1e2 if not math.isfinite(a) else a
    hi = 1e2 if not math.isfinite(b) else b
    rng = np.random.default_rng(seed)
    return jnp.asarray((hi - lo) * rng.uniform(size=n) + lo)


class TestFactoryRouting:
    def test_builds_unbndfun_piece(self):
        for dom in [(-jnp.inf, jnp.inf), (0.0, jnp.inf), (-jnp.inf, -3 * np.pi)]:
            f = cj.chebfun(lambda x: jnp.exp(-(x**2)), domain=dom)
            assert isinstance(f.funs[0], Unbndfun)
            assert f.domain == Domain((float(dom[0]), float(dom[1])))

    def test_evaluation_matches_operand(self):
        def op(x):
            return jnp.exp(-(x**2))
        f = cj.chebfun(op, domain=(-jnp.inf, jnp.inf))
        x = _finite_pts(-jnp.inf, jnp.inf)
        npt.assert_allclose(np.asarray(f(x)), np.asarray(op(x)), atol=1e-13)

    def test_trig_on_unbounded_rejected(self):
        try:
            cj.chebfun(jnp.sin, domain=(0.0, jnp.inf), trig=True)
        except ValueError:
            return
        raise AssertionError("trig=True on an unbounded domain must raise")


class TestUnboundedSum:
    def test_gaussian_integral(self):
        # ∫_{-inf}^{inf} exp(-x^2) dx = sqrt(pi).
        f = cj.chebfun(lambda x: jnp.exp(-(x**2)), domain=(-jnp.inf, jnp.inf))
        npt.assert_allclose(float(f.sum()), math.sqrt(math.pi), atol=1e-12)

    def test_exp_decay_integral(self):
        # ∫_0^{inf} exp(-x) dx = 1.
        f = cj.chebfun(lambda x: jnp.exp(-x), domain=(0.0, jnp.inf))
        npt.assert_allclose(float(f.sum()), 1.0, atol=1e-12)


class TestArrayValuedColumns:
    _DOM = (-jnp.inf, -3 * np.pi)

    def _f(self):
        return cj.chebfun(
            lambda x: jnp.stack(
                [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1),
            domain=self._DOM)

    def test_n_columns(self):
        assert self._f().n_columns == 3

    def test_extract_columns_keeps_mapping(self):
        f = self._f()
        g = f.extract_columns([2, 0])
        assert isinstance(g.funs[0], Unbndfun)
        assert g.funs[0].mapping_type == f.funs[0].mapping_type
        assert g.n_columns == 2
        x = _finite_pts(*self._DOM)
        want = jnp.stack([(1 - jnp.exp(x)) / x, jnp.exp(x)], axis=-1)
        npt.assert_allclose(np.asarray(g(x)), np.asarray(want), atol=1e-12)

    def test_extract_single_column_scalar(self):
        g = self._f().extract_columns(0)
        assert g.n_columns == 1 and isinstance(g.funs[0], Unbndfun)
        x = _finite_pts(*self._DOM)
        npt.assert_allclose(np.ravel(np.asarray(g(x))),
                            np.asarray(jnp.exp(x)), atol=1e-12)

    def test_mat2cell_keeps_mapping(self):
        C = self._f().mat2cell([1, 2])
        assert len(C) == 2
        assert C[0].n_columns == 1 and C[1].n_columns == 2
        assert all(isinstance(c.funs[0], Unbndfun) for c in C)

    def test_repmat_keeps_mapping(self):
        g = self._f().extract_columns(0).repmat(3)
        assert g.n_columns == 3 and isinstance(g.funs[0], Unbndfun)

    def test_assign_columns_keeps_mapping(self):
        f = self._f()
        repl = cj.chebfun(lambda x: jnp.exp(-(x**2)), domain=self._DOM)
        h = f.assign_columns(1, repl)
        assert isinstance(h.funs[0], Unbndfun) and h.n_columns == 3
        x = _finite_pts(*self._DOM)
        want = jnp.stack(
            [jnp.exp(x), jnp.exp(-(x**2)), (1 - jnp.exp(x)) / x], axis=-1)
        npt.assert_allclose(np.asarray(h(x)), np.asarray(want), atol=1e-11)


class TestComplexParts:
    _DOM = (-jnp.inf, jnp.inf)

    def _f(self):
        # A complex-valued smooth decaying function on (-inf, inf).
        return cj.chebfun(
            lambda x: (jnp.exp(-(x**2)) + 1j * x * jnp.exp(-(x**2))),
            domain=self._DOM)

    def test_real_imag_conj_keep_mapping(self):
        f = self._f()
        x = _finite_pts(*self._DOM)
        for part, ref in [
            (f.real(), lambda x: jnp.exp(-(x**2))),
            (f.imag(), lambda x: x * jnp.exp(-(x**2))),
        ]:
            assert isinstance(part.funs[0], Unbndfun)
            npt.assert_allclose(np.asarray(part(x)).real,
                                np.asarray(ref(x)), atol=1e-12)
        c = f.conj()
        assert isinstance(c.funs[0], Unbndfun)
        npt.assert_allclose(np.asarray(c(x)),
                            np.conj(np.asarray(f(x))), atol=1e-12)


class TestPredicates:
    def test_any_zero_is_false(self):
        z = cj.chebfun(lambda x: 0 * x, domain=(1.0, jnp.inf))
        assert not bool(z.any())

    def test_any_nonzero_is_true(self):
        f = cj.chebfun(lambda x: jnp.exp(-x), domain=(1.0, jnp.inf))
        assert bool(f.any())

    def test_isinf_false_for_smooth(self):
        f = cj.chebfun(lambda x: jnp.exp(x), domain=(-jnp.inf, -3 * np.pi))
        assert not bool(f.isinf())


class TestUnboundedCalculus:
    def test_diff_matches_analytic(self):
        # d/dx exp(-x) = -exp(-x) on [1, inf).
        f = cj.chebfun(lambda x: jnp.exp(-x), domain=(1.0, jnp.inf))
        df = f.diff()
        x = _finite_pts(1.0, jnp.inf, seed=3)
        npt.assert_allclose(np.asarray(df(x)),
                            np.asarray(-jnp.exp(-x)), atol=1e-10)

    def test_arithmetic_preserves_domain(self):
        # Chebfun<->Chebfun and scalar arithmetic must keep the Unbndfun piece
        # (a plain _Piece on an infinite interval evaluates to NaN).
        f = cj.chebfun(lambda x: jnp.exp(-x), domain=(0.0, jnp.inf))
        x = _finite_pts(0.0, jnp.inf, seed=4)
        ex = np.asarray(jnp.exp(-x))
        for h, ref in [
            (f + f, 2 * ex),          # binary op
            (f + 3.0, ex + 3.0),      # scalar add
            (-f, -ex),                # negation
            (2.0 * f, 2 * ex),        # scalar mul
            (f - f, np.zeros_like(ex)),
        ]:
            assert isinstance(h.funs[0], Unbndfun)
            npt.assert_allclose(np.asarray(h(x)), ref, atol=1e-12)


class TestWithTech:
    def test_piece_with_tech_rebuilds_piece(self):
        p = _Piece.from_function(jnp.sin, -1.0, 1.0)
        t2 = Chebtech2.from_function(jnp.cos)
        q = p.with_tech(t2)
        assert isinstance(q, _Piece)
        assert q.interval == p.interval
        npt.assert_allclose(float(q(jnp.float64(0.3))),
                            float(jnp.cos(jnp.float64(0.3))), atol=1e-12)

    def test_unbndfun_with_tech_keeps_mapping(self):
        d = Domain((0.0, jnp.inf))
        u = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        new = u.with_tech(u.onefun)
        assert isinstance(new, Unbndfun)
        assert new.domain == d and new.mapping_type == u.mapping_type
        assert new.tech is u.onefun

    def test_unbndfun_tech_is_onefun(self):
        d = Domain((-jnp.inf, jnp.inf))
        u = Unbndfun.from_function(lambda x: jnp.exp(-(x**2)), d)
        assert u.tech is u.onefun
