"""Port of MATLAB Chebfun tests/linop/test_linopAdjoint.m (Fable 5).

``linopAdjoint(L, bcType)`` is ``linop_adjoint(L, bc_type)``; the adjoint
constraint rows are applied with ``_apply_row`` (MATLAB
``Ls.constraint.functional * u``).

Provenance
----------
MATLAB source : tests/linop/test_linopAdjoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at, mult, zero_functional
from chebfunjax.operators.linop_adjoint import _apply_row, linop_adjoint

jax.config.update("jax_enable_x64", True)

TOL = 5e-13   # cheboppref bvpTol


def _n(f):
    if isinstance(f, (list, tuple)):
        return float(np.sqrt(sum(_n(c) ** 2 for c in f)))
    if hasattr(f, "blocks"):
        return _n([b for row in f.blocks for b in row])
    return float(np.real(np.asarray(f.norm(2))))


def _ip(u, v):
    """u' * v for Chebfuns / lists of Chebfuns."""
    if isinstance(u, (list, tuple)):
        return sum(_ip(a, b) for a, b in zip(u, v))
    return complex(np.asarray(u.conj().inner(v))) if hasattr(u, "inner") \
        else complex(np.asarray((u.conj() * v).sum()))


def _cols(M):
    return [b for row in M.blocks for b in row] if hasattr(M, "blocks") else M


def _cons(Ls, u):
    return [_apply_row(row, u if isinstance(u, list) else [u])
            for row, _v in Ls.constraint]


def _cvals(Ls):
    return np.array([v for _r, v in Ls.constraint], dtype=float)


class TestLinopAdjoint:
    def test_all_matlab_assertions(self):
        tol = TOL
        dom = (-1.0, 1.0)
        Id = I(dom)
        Dd = D(dom)
        x = chebfun(lambda t: t, domain=dom)
        c = (np.pi * x ** 2).sin()
        C = mult(c, dom)
        z = zero_functional(dom)
        El = eval_at(dom[0], dom)
        Er = eval_at(dom[1], dom)

        u = (x - dom[0]) * (x - dom[1]) * x.exp()
        nrm = _n(u)
        L = linop(-(Dd ** 2) + C).addbc(El, 0.0).addbc(Er, 0.0)
        Ls, op, bcL, bcR, bcM = linop_adjoint(L, "bvp")
        assert _n(-u.diff(2) + c * u - _cols(Ls * u)[0]) < nrm * tol       # pass(1)
        assert _n(-u.diff(2) + c * u - op(x, u)) < nrm * tol               # pass(2)
        assert np.linalg.norm(_cons(Ls, u)) < nrm * tol                    # pass(3)
        assert abs(float(bcL(u)(jnp.asarray(dom[0])))) < nrm * tol         # pass(4)
        assert abs(float(bcR(u)(jnp.asarray(dom[1])))) < nrm * tol         # pass(5)
        assert bcM is None                                                 # pass(6)
        assert np.linalg.norm(_cvals(Ls)) == 0                             # pass(7)
        assert abs(_ip(u, _cols(L * u)[0]) - _ip(_cols(Ls * u)[0], u)) < nrm * tol  # pass(8)

        L = linop(-(Dd ** 2) + C)
        Ls, op, bcL, bcR, bcM = linop_adjoint(L, "periodic")
        assert _n(-u.diff(2) + c * u - _cols(Ls * u)[0]) < nrm * tol       # pass(9)
        assert _n(-u.diff(2) + c * u - op(x, u)) < nrm * tol               # pass(10)
        assert len(Ls.constraint) == 0                                     # pass(11)
        assert bcL is None and bcR is None                                 # pass(12)-(13)
        assert bcM == "periodic"                                           # pass(14)
        assert _cvals(Ls).size == 0                                        # pass(15)
        assert abs(_ip(u, _cols(L * u)[0]) - _ip(_cols(Ls * u)[0], u)) < nrm * tol  # pass(16)

        u = x.exp()
        v = (x - dom[0]) * (x - dom[1]) * x.exp()
        nrm = _n(u) + _n(v)
        L = linop(Dd)
        Ls, op, bcL, bcR, bcM = linop_adjoint(L, "bvp")
        assert _n(-v.diff() - _cols(Ls * v)[0]) < nrm * tol                # pass(17)
        assert _n(-v.diff() - op(x, v)) < nrm * tol                        # pass(18)
        assert np.linalg.norm(_cons(Ls, v)) < nrm * tol                    # pass(19)
        assert abs(float(bcL(v)(jnp.asarray(dom[0])))) < nrm * tol         # pass(20)
        assert abs(float(bcR(v)(jnp.asarray(dom[1])))) < nrm * tol         # pass(21)
        assert bcM is None                                                 # pass(22)
        assert np.linalg.norm(_cvals(Ls)) == 0                             # pass(23)
        assert abs(_ip(v, _cols(L * u)[0]) - _ip(_cols(Ls * v)[0], u)) < nrm * tol  # pass(24)

        u1 = (x - dom[0]) * x.exp()
        u2 = (x - dom[0]) * x.sin()
        v1 = (x - dom[1]) * x.exp()
        v2 = (x - dom[1]) * x.sin()
        nrm = _n(u1) + _n(u2) + _n(v1) + _n(v2)
        L = linop([[Dd, Id], [Id, Dd]]).addbc([El, z], 0.0).addbc([z, El], 0.0)
        Ls, op, bcL, bcR, bcM = linop_adjoint(L, "bvp")
        exact = [-v1.diff() + v2, v1 - v2.diff()]
        got = _cols(Ls * [v1, v2])
        assert _n([exact[0] - got[0], exact[1] - got[1]]) < nrm * tol      # pass(25)
        o = op(x, v1, v2)
        assert _n([exact[0] - o[0], exact[1] - o[1]]) < nrm * tol          # pass(26)
        assert np.linalg.norm(_cons(Ls, [v1, v2])) < nrm * tol             # pass(27)
        assert bcL is None                                                 # pass(28)
        r = bcR(v1, v2)
        r = r if isinstance(r, list) else [r]
        assert np.linalg.norm([float(g(jnp.asarray(dom[1]))) for g in r]) < nrm * tol  # pass(29)
        assert bcM is None                                                 # pass(30)
        assert np.linalg.norm(_cvals(Ls)) == 0                             # pass(31)
        assert abs(_ip([v1, v2], _cols(L * [u1, u2])) - _ip(_cols(Ls * [v1, v2]), [u1, u2])) < nrm * tol  # pass(32)

        u1 = (x - dom[1]) * x.exp()
        u2 = (x - dom[1]) * x.sin()
        v1 = (x - dom[0]) * x.exp()
        v2 = (x - dom[0]) * x.sin()
        nrm = _n(u1) + _n(u2) + _n(v1) + _n(v2)
        L = linop([[Dd, -Dd], [Id, Dd]]).addbc([Er, z], 0.0).addbc([z, Er], 0.0)
        Ls, op, bcL, bcR, bcM = linop_adjoint(L, "bvp")
        exact = [-v1.diff() + v2, v1.diff() - v2.diff()]
        got = _cols(Ls * [v1, v2])
        assert _n([exact[0] - got[0], exact[1] - got[1]]) < nrm * tol      # pass(33)
        o = op(x, v1, v2)
        assert _n([exact[0] - o[0], exact[1] - o[1]]) < nrm * tol          # pass(34)
        assert np.linalg.norm(_cons(Ls, [v1, v2])) < nrm * tol             # pass(35)
        r = bcL(v1, v2)
        r = r if isinstance(r, list) else [r]
        assert np.linalg.norm([float(g(jnp.asarray(dom[0]))) for g in r]) < nrm * tol  # pass(36)
        assert bcR is None                                                 # pass(37)
        assert bcM is None                                                 # pass(38)
        assert np.linalg.norm(_cvals(Ls)) == 0                             # pass(39)
        assert abs(_ip([v1, v2], _cols(L * [u1, u2])) - _ip(_cols(Ls * [v1, v2]), [u1, u2])) < nrm * tol  # pass(40)
