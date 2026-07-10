"""Port of MATLAB Chebfun tests/chebtech/test_prolong.m (Opus 4.8).

The MATLAB test loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we
parametrize over ``[Chebtech1, Chebtech2]``.

Notes on gaps (see the report):
* Every sub-test except pass 1 operates on array-valued techs
  (``[F(x), -F(x)]``, and pass 7's ``v = [1 2 3]`` which MATLAB reads as three
  constant columns, cf. ``repmat([1 2 3], 5, 1)``), which chebfunjax does not
  implement.  Those are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_prolong.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

BOTH = [Chebtech1, Chebtech2]


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechProlong:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_prolong_to_length_one(self, Tech):
        # pass(n, 1): prolong(sin, 1) -> length 1, coeffs ~ 0 (sin is odd).
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = f.prolong(1)
        assert g.n == 1
        assert _ninf(g.coeffs) < 10 * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_skipped(self, Tech):
        # pass(n, 2)-(7): all operate on array-valued techs ([F(x), -F(x)] and
        # v = [1 2 3] read as three constant columns), which chebfunjax lacks.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix "
            "techs"
        )
