"""Port of MATLAB Chebfun tests/chebfun/test_kronOp.m (Fable 5).

FIXED (Fable 5): kron(f, g, 'op') builds the rank-1 (or low-rank) integral
operator A = f*(g'*.), i.e. A*h = f*<g,h>.  It applies to a Chebfun and
realizes as a rank-k collocation matrix at chebcolloc1 / chebcolloc2 /
trigcolloc points (KronOp.matrix(n, kind)).  chebfunjax has no
``chebmatrix(A)`` wrapper for an operator block, so the discrete-form
assertions call KronOp.matrix(n, kind) directly -- the same rank-k matrix
MATLAB's matrix(chebmatrix(A), n, options) produces.

Provenance
----------
MATLAB source : tests/chebfun/test_kronOp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, kron
from chebfunjax.utils.quadrature import chebpts_ab, trigpts

TOL = 1000 * float(np.finfo(np.float64).eps)
_D = (0.0, 1.0)
_XX1 = chebpts_ab(200, 0.0, 1.0, kind=1)
_XX2 = chebpts_ab(200, 0.0, 1.0, kind=2)
_XXE = trigpts(200, (0.0, 1.0))[0]


def _cf(op):
    return Chebfun.from_function(op, Domain(_D))


class TestChebfunKronOp:
    def test_scalar_smooth(self):
        f, g, h = _cf(jnp.sin), _cf(jnp.tanh), _cf(jnp.cos)
        A = kron(f, g, "op")
        Ah = f * float((g * h).sum())
        assert float((A * h - Ah).norm()) < TOL          # pass(1)
        hv = np.asarray(h(_XX1))
        assert np.max(np.abs(np.asarray(Ah(_XX1))
                             - A.matrix(200, "chebcolloc1") @ hv)) < TOL
        hv = np.asarray(h(_XX2))
        assert np.max(np.abs(np.asarray(Ah(_XX2))
                             - A.matrix(200, "chebcolloc2") @ hv)) < TOL

    def test_periodic(self):
        f = _cf(lambda t: jnp.exp(jnp.sin(4 * np.pi * t)))
        g = _cf(lambda t: jnp.tanh(0.5 * jnp.cos(2 * np.pi * t)))
        h = _cf(lambda t: jnp.cos(2 * np.pi * t))
        A = kron(f, g, "op")
        Ah = f * float((g * h).sum())
        assert float((A * h - Ah).norm()) < TOL          # pass(4)
        for xx, kind in ((_XX1, "chebcolloc1"), (_XX2, "chebcolloc2"),
                         (_XXE, "trigcolloc")):
            hv = np.asarray(h(xx))
            err = np.max(np.abs(np.asarray(Ah(xx))
                                - A.matrix(200, kind) @ hv))
            assert err < TOL

    def test_array_valued(self):
        f = [_cf(jnp.exp), _cf(jnp.tanh)]
        g = [_cf(jnp.exp), _cf(lambda t: t / (1 + t ** 2))]
        u = _cf(lambda t: t)
        A = kron(f, g, "op")
        Au = _cf(lambda t: jnp.exp(t) + (1 - np.pi / 4) * jnp.tanh(t))
        assert float((A * u - Au).norm()) < TOL          # pass(8)
        for xx, kind in ((_XX1, "chebcolloc1"), (_XX2, "chebcolloc2")):
            uv = np.asarray(u(xx))
            err = np.max(np.abs(np.asarray(Au(xx))
                                - A.matrix(200, kind) @ uv))
            assert err < TOL

    def test_invalid_mode_raises(self):
        f, g = _cf(jnp.sin), _cf(jnp.tanh)
        with pytest.raises(ValueError, match="kron"):
            kron(f, g, "invalid")
