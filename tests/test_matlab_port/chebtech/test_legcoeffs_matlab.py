"""Port of MATLAB Chebfun tests/chebtech/test_legcoeffs.m (Opus 4.8).

Native scalar and matrix predicates exercise the public legcoeffs API.

Provenance
----------
MATLAB source : tests/chebtech/test_legcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.linalg.norm(jnp.asarray(a), ord=jnp.inf))


# (Tech, kind) pairs: chebtech1 uses 1st-kind points, chebtech2 uses 2nd-kind.
CASES = [(Chebtech1, 1), (Chebtech2, 2)]


class TestChebtechLegcoeffs:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_scalar_P3(self, Tech, kind):
        # pass(n, 1): legcoeffs of .5*(3x^2-1) == [0 0 1]', tol 10*eps.
        tol = 10 * EPS
        x = chebpts(3, kind)
        f = Tech.from_values(0.5 * (3 * x ** 2 - 1))
        lc = f.legcoeffs()
        assert _ninf(lc - jnp.array([0.0, 0.0, 1.0])) < tol

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_vector_P1P2P3(self, Tech, kind):
        # pass(n, 2): array-valued [P_0, P_1, P_2] -> legcoeffs == eye(3).
        # (MATLAB comment says [P_1,P_2,P_3] but the columns are 1, x,
        # .5*(3x^2-1), i.e. the degree 0/1/2 Legendre polynomials.)
        tol = 10 * EPS
        x = chebpts(3, kind)
        cols = [1.0 + 0.0 * x, x, 0.5 * (3 * x**2 - 1)]
        f = Tech.from_values(jnp.stack(cols, axis=-1))
        lc = f.legcoeffs()
        assert _ninf(lc - jnp.eye(3)) < tol
