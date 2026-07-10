"""Port of MATLAB Chebfun tests/chebtech/test_legcoeffs.m (Opus 4.8).

chebfunjax has NO ``legcoeffs`` method on Chebtech, but
``chebfunjax.utils.transforms.cheb2leg`` converts Chebyshev coefficients to
Legendre coefficients -- which is exactly the quantity MATLAB ``legcoeffs``
returns.  We therefore port the scalar assertion using
``cheb2leg(jnp.asarray(f.coeffs))`` (verified: P_3 = .5*(3x^2-1) maps to the
Legendre coefficient vector [0 0 1] to machine precision).

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
from chebfunjax.utils.transforms import cheb2leg

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


# (Tech, kind) pairs: chebtech1 uses 1st-kind points, chebtech2 uses 2nd-kind.
CASES = [(Chebtech1, 1), (Chebtech2, 2)]


class TestChebtechLegcoeffs:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_scalar_P3(self, Tech, kind):
        # pass(n, 1): legcoeffs of .5*(3x^2-1) == [0 0 1]', tol 10*eps.
        # cheb2leg(f.coeffs) is the Cheb->Leg coefficient conversion == legcoeffs.
        tol = 10 * EPS
        x = chebpts(3, kind)
        f = Tech.from_values(0.5 * (3 * x ** 2 - 1))
        lc = cheb2leg(jnp.asarray(f.coeffs))
        assert _ninf(lc - jnp.array([0.0, 0.0, 1.0])) < tol

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_vector_P1P2P3(self, Tech, kind):
        # pass(n, 2): array-valued [P_1, P_2, P_3] -> legcoeffs == eye(3).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )
