"""Port of MATLAB Chebfun tests/chebop/test_cellOperator.m (Fable 5).

FIXED (Fable 5, Big-Three array-valued epic): the system defined via a
cell-returning operator ``op{n} = diff(u{n}, 2) + n*u{n}`` (n = 1, 2) is the
2x2 system [u'' + u, v'' + 2v] with lbc [u-1, v], rbc [u, v-1].  chebfunjax
expresses it with the multi-argument operator form (the ``u{1}/u{2}`` cell
indexing and ``N.numVars`` have no counterpart, but the underlying system is
identical).

Provenance
----------
MATLAB source : tests/chebop/test_cellOperator.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

XS = jnp.asarray(np.linspace(-0.99, 0.99, 40))


class TestChebopCelloperator:
    def test_cell_operator_system(self):
        N = Chebop(lambda r, u, v: [u.diff(2) + u, v.diff(2) + 2 * v], (-1.0, 1.0))
        N.lbc = lambda u, v: [u - 1, v]
        N.rbc = lambda u, v: [u, v - 1]
        sol = N.solve([0, 0], n=16)
        u, v = sol[0], sol[1]
        # pass: norm(N(u)) < 1e-10.
        res = N([u, v])
        assert float(jnp.max(jnp.abs(res[0](XS)))) < 1e-10
        assert float(jnp.max(jnp.abs(res[1](XS)))) < 1e-10
        # pass: norm(feval(N.lbc(u), -1)) < 1e-13 and rbc(1) < 1e-13.
        lbc = N.lbc(u, v)
        rbc = N.rbc(u, v)
        a, b = jnp.asarray(-1.0), jnp.asarray(1.0)
        assert abs(float(lbc[0](a))) < 1e-13 and abs(float(lbc[1](a))) < 1e-13
        assert abs(float(rbc[0](b))) < 1e-13 and abs(float(rbc[1](b))) < 1e-13
