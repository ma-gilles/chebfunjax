"""Port of MATLAB Chebfun tests/unbndfun/test_mldivide.m (Opus 4.8).

``A\\B`` for array-valued unbndfuns solves the least-squares system whose
columns are the (array-valued) unbndfun columns.  chebfunjax's ``Unbndfun``
now supports array-valued techs and ``mldivide`` (delegating to the [-1, 1]
Chebtech2 QR, exactly MATLAB's ``A.onefun\\B.onefun``).

Provenance
----------
MATLAB source : tests/unbndfun/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf
DOM = (-INF, 3 * np.pi)
# MATLAB samples random INTERIOR points in [-1e6, 3*pi); the exp-growing
# columns peak at the finite endpoint x=3*pi, so a deterministic grid that
# stops just short of it mirrors the MATLAB sampling.
XR = np.linspace(-1e6, 3 * np.pi, 100)[:-1]
X = jnp.asarray(XR)


def _U(op):
    return Unbndfun.from_function(op, Domain(DOM))


class TestUnbndfunMldivide:
    def test_array_valued_solve(self):  # pass(1)
        opA = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
        )
        opB = lambda x: jnp.stack(
            [(2 * x + 1) * jnp.exp(x), jnp.exp(x), 2 * (1 - jnp.exp(x)) / x], axis=-1
        )
        A = _U(opA)
        B = _U(opB)
        X_sol = np.asarray(A.mldivide(B))  # 3x3 numeric solve
        res = A @ X_sol - B  # mtimes reconstruction residual
        err = float(np.max(np.abs(np.asarray(res(X)))))
        tol = 1e1 * max(EPS * float(A.vscale), EPS * float(B.vscale))
        assert err < tol
