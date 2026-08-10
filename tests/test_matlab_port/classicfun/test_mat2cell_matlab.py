"""Port of MATLAB Chebfun tests/classicfun/test_mat2cell.m (Fable 5).

``Classicfun.mat2cell`` (inherited by ``Bndfun``) and ``Unbndfun.mat2cell``
now delegate to the onefun's ``mat2cell`` and re-wrap each block in the fun's
domain, so every MATLAB assertion is ported at MATLAB's tolerances.

MATLAB's ``normest(F{k} - g)`` estimates the sup-norm of the difference on a
sample grid; we use the exact same idea with a dense grid over the domain.

No gaps: all seven MATLAB passes are exercised.

Provenance
----------
MATLAB source : tests/classicfun/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)

# MATLAB: data.domain = [-2 7]; x = diff(domain)*rand(1000,1) + domain(1).
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))


def _normest(f, g, x):
    """MATLAB ``normest(f - g)``: sup-norm of the difference on a grid."""
    return float(jnp.max(jnp.abs(jnp.asarray(f(x)) - jnp.asarray(g(x)))))


class TestClassicfunMat2cellBndfun:
    """MATLAB passes 1-6 (BNDFUN)."""

    @pytest.fixture
    def pieces(self):
        f = Bndfun.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x), x], axis=-1), DOM)
        g = Bndfun.from_function(jnp.sin, DOM)
        h = Bndfun.from_function(
            lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1), DOM)
        ll = Bndfun.from_function(lambda x: x, DOM)
        return f, g, h, ll

    def _check_blocks(self, F, g, h, ll):
        # pass 1 / pass 4
        assert not F[0].isempty()
        assert _normest(F[0], g, X) < 1e2 * EPS * g.vscale
        # pass 2 / pass 5
        assert not F[1].isempty()
        assert _normest(F[1], h, X) < 10 * EPS * h.vscale
        # pass 3 / pass 6
        assert not F[2].isempty()
        assert _normest(F[2], ll, X) < 10 * EPS * ll.vscale

    def test_full_arguments(self, pieces):
        # pass(1:3): MATLAB mat2cell(f, 1, [1 2 1]).
        f, g, h, ll = pieces
        F = f.mat2cell([1, 2, 1])
        assert len(F) == 3
        self._check_blocks(F, g, h, ll)

    def test_two_arguments(self, pieces):
        # pass(4:6): MATLAB mat2cell(f, [1 2 1]) -- the M argument omitted.
        # chebfunjax has the single ``sizes`` argument, so both MATLAB forms
        # map onto the same call.
        f, g, h, ll = pieces
        F = f.mat2cell([1, 2, 1])
        assert len(F) == 3
        self._check_blocks(F, g, h, ll)

    def test_default_one_column_per_block(self, pieces):
        # MATLAB mat2cell(f) with C = ones(1, COL).
        f, g, _, ll = pieces
        F = f.mat2cell()
        assert len(F) == 4
        assert _normest(F[0], g, X) < 1e2 * EPS * g.vscale
        assert _normest(F[3], ll, X) < 10 * EPS * ll.vscale


class TestClassicfunMat2cellUnbndfun:
    """MATLAB pass 7 (UNBNDFUN on [-inf, -3*pi])."""

    def test_left_infinite_array_valued(self):
        # pass(7)
        dom = Domain((-jnp.inf, -3.0 * np.pi))
        # MATLAB: domCheck = [-1e6 -3*pi]; x = diff*rand(100,1) + domCheck(1).
        x = jnp.asarray(np.linspace(-1e6, -3.0 * np.pi, 100))

        def op(t):
            return jnp.stack(
                [jnp.exp(t), t * jnp.exp(t), (1 - jnp.exp(t)) / t], axis=-1)

        f = Unbndfun.from_function(op, dom)
        F = f.mat2cell([1, 2])
        assert len(F) == 2

        f1_exact = jnp.exp(x)
        f2_exact = jnp.stack(
            [x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1)

        err1 = jnp.asarray(F[0](x)) - f1_exact
        err2 = jnp.asarray(F[1](x)) - f2_exact
        err = jnp.concatenate([jnp.ravel(err1), jnp.ravel(err2)])
        assert float(jnp.max(jnp.abs(err))) < 1e2 * EPS * f.vscale
