"""Port of MATLAB Chebfun tests/misc/test_pswf.m (Fable 5).

MATLAB's chebfun-valued ``pswf`` is ``pswf(..., output="chebfun")``.

Provenance
----------
MATLAB source : tests/misc/test_pswf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import legpoly
from chebfunjax.utils.pswf import pswf

jax.config.update("jax_enable_x64", True)


def _gram(p):
    """Gram matrix of the columns, p' * p, by Clenshaw-Curtis quadrature
    exact for the polynomial products.  The columns are evaluated by
    barycentric interpolation of their stored Chebyshev-point values
    (backward stable, as MATLAB's feval is), not by a degree-1100
    Clenshaw sum."""
    from chebfunjax.utils.quadrature import chebpts, chebweights
    bp = [float(v) for v in p.domain.breakpoints]
    a, b = bp[0], bp[-1]
    V0 = np.asarray(p.funs[0].tech.values)
    if V0.ndim == 1:
        V0 = V0[:, None]
    n = V0.shape[0]
    xn = np.asarray(chebpts(n, kind=2))
    wb = (-1.0) ** np.arange(n)
    wb[0] *= 0.5
    wb[-1] *= 0.5
    nq = 2 * n + 3
    xq = np.asarray(chebpts(nq, kind=2))
    w = np.asarray(chebweights(nq)) * (b - a) / 2.0
    D = xq[:, None] - xn[None, :]
    exact = np.isclose(D, 0.0)
    D = np.where(exact, 1.0, D)
    Wt = wb[None, :] / D
    Vq = (Wt @ V0) / Wt.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    if hit.any():
        Vq[hit] = V0[exact[hit].argmax(axis=1)]
    return (Vq * w[:, None]).T @ Vq


class TestMiscPswf:
    def test_all_matlab_assertions(self):
        P, _ = pswf(np.arange(10), 4, output="chebfun")
        assert np.linalg.norm(_gram(P) - np.diag(2.0 / np.arange(1, 20, 2))) < 1e-10  # pass(1)
        assert len(np.asarray(P.extract_columns(9).roots())) == 9   # pass(2)

        P4pi05 = -0.335179227182412994123178747
        assert abs(float(pswf(4, np.pi, output="chebfun")[0](jnp.asarray(0.5))) - P4pi05) < 1e-10  # pass(3)
        val = 1.053221995207094811
        assert abs(float(pswf(0, 1, output="chebfun")[0](jnp.asarray(0.0))) - val) < 1e-10  # pass(4)
        val = 0.079588252262137702
        assert abs(float(pswf(100, 1, output="chebfun")[0](jnp.asarray(0.0))) - val) < 1e-10  # pass(5)
        val = -0.005959162025821166
        assert abs(float(pswf(33, 33, output="chebfun")[0](jnp.asarray(0.9))) - val) < 1e-10  # pass(6)
        val = -0.000000509171971545
        assert abs(float(pswf(1, 20, output="chebfun")[0](jnp.asarray(-1.0))) - val) < 1e-10  # pass(7)

        f, _ = pswf(np.arange(21), 100, output="chebfun")
        L = legpoly(np.arange(21))
        assert np.all(np.sign(np.asarray(f(jnp.asarray(0.001))))
                      == np.sign(np.asarray(L(jnp.asarray(0.001)))))    # pass(8)
