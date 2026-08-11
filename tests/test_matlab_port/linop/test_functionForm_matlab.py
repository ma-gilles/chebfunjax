"""Port of MATLAB Chebfun tests/linop/test_functionForm.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_functionForm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

import chebfunjax as cj
from chebfunjax.operators.blocks import D, I, mult, to_function, zeros_op

jax.config.update("jax_enable_x64", True)


class TestLinopFunctionForm:
    def test_all_matlab_assertions(self):
        dom = (-2.0, 2.0)
        Id = I(dom)
        Dop = D(dom)
        zeros_op(dom)
        x = cj.chebfun(lambda t: t, domain=dom)
        u = (x ** 2).sin()
        U = mult(u)

        err = []

        eyeop = to_function(Id)
        err.append(float((eyeop(u) - u).norm()))

        diffop = to_function(Dop)
        two_x = diffop(x * x)
        two_x_again = Dop * (x * x)
        err.append(float((2 * x - two_x).norm()))

        multop = to_function(U)
        err.append(float((multop(x + 1) - u * (x + 1)).norm()))
        err.append(float((multop(x + 1) - U * (x + 1)).norm()))

        # ``D*(x.*x)`` and ``toFunction(D)(x.*x)`` are the same object.
        assert float((two_x - two_x_again).norm()) == 0.0
        assert all(e < 1e-14 for e in err)
