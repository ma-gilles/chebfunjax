"""Port of MATLAB Chebfun tests/chebtech/test_cell2mat.m (Opus 4.8).

MATLAB ``cell2mat([g h])`` horizontally concatenates scalar/array-valued
chebtechs into a single array-valued (quasimatrix) chebtech.  FIXED
(Fable 5, Big-Three array-valued epic): ``cell2mat`` ports at the same
tolerance.

Provenance
----------
MATLAB source : tests/chebtech/test_cell2mat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)


class TestChebtechCell2mat:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_horizontal_concat(self, Tech, kind):
        # pass(n, 1): all(sum(cell2mat([g h]) - f) < max(vscale(f)*eps)).
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = Tech.from_function(lambda x: jnp.sin(x))
        h = Tech.from_function(
            lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1))
        F = Tech.cell2mat([g, h])
        s = np.asarray((F - f).sum())
        assert np.max(np.abs(s)) < f.vscale * EPS
