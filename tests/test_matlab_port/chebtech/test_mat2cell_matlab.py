"""Port of MATLAB Chebfun tests/chebtech/test_mat2cell.m (Opus 4.8).

MATLAB ``mat2cell(f, 1, [1 2])`` splits an array-valued (quasimatrix)
chebtech by columns.  FIXED (Fable 5, Big-Three array-valued epic):
techs carry (n, m) coefficient matrices and ``mat2cell`` ports both
assertions at the same tolerances.

Provenance
----------
MATLAB source : tests/chebtech/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)


def _mk(Tech):
    f = Tech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
    g = Tech.from_function(lambda x: jnp.sin(x))
    h = Tech.from_function(
        lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1))
    return f, g, h


class TestChebtechMat2cell:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_first_column(self, Tech, kind):
        # pass(n, 1): sum(F{1} - g) < 10*vscale(g)*eps.
        f, g, h = _mk(Tech)
        F = f.mat2cell([1, 2])
        assert abs(complex((F[0] - g).sum())) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_remaining_columns(self, Tech, kind):
        # pass(n, 2): all(sum(F{2} - h)) < 10*max(vscale(h)*eps).
        f, g, h = _mk(Tech)
        F = f.mat2cell([1, 2])
        s = np.asarray((F[1] - h).sum())
        assert np.max(np.abs(s)) < 10 * h.vscale * EPS
