"""Port of MATLAB Chebfun tests/linop/test_coeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

import chebfunjax as cj
from chebfunjax.operators.blocks import D, I, mult, to_coeff, zeros_op

jax.config.update("jax_enable_x64", True)


class TestLinopCoeffs:
    def test_all_matlab_assertions(self):
        dom = (-2.0, -0.5, 1.0, 2.0)
        Id = I(dom)
        Dop = D(dom)
        zeros_op(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        u = (x ** 2).sin()
        U = mult(u)

        err = []

        # Quasimatrix [ 1 ]
        Ic = to_coeff(Id)
        err.append(float((Ic[0] - x ** 0).norm()))

        # Quasimatrix [ 1 0 ]
        Dc = to_coeff(Dop)
        err.append(float((Dc[0] - x ** 0).norm()))
        err.append(float((Dc[1] - 0).norm()))

        # Quasimatrix [ u ]
        Uc = to_coeff(U)
        err.append(float((Uc[0] - u).norm()))

        A = Dop * (Id + U * Dop)
        Ac = to_coeff(A)
        err.append(float((Ac[0] - u).norm()))
        err.append(float((Ac[1] - (u.diff() + 1)).norm()))
        err.append(float((Ac[2] - 0).norm()))

        assert all(e < 1e-15 for e in err), err
