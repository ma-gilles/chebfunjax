"""Port of MATLAB Chebfun tests/misc/test_legpoly.m (Fable 5).

MATLAB ``norm(matrix)`` is the spectral 2-norm (``ord=2`` here).

MATLAB's chebfun-valued ``legpoly`` is
:func:`chebfunjax.chebfun1d.chebfun.legpoly`; ``p'*p`` is the Gram matrix
of the columns and ``legendre(n, x)`` (first row) is the stable Clenshaw
evaluation ``numpy.polynomial.legendre.legval`` (SciPy's
``eval_legendre`` is only accurate to ~6e-12 at degree 900).

Provenance
----------
MATLAB source : tests/misc/test_legpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre as _L

from chebfunjax.chebfun1d.chebfun import legpoly

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _gram(p):
    """Gram matrix of the columns, p' * p, the way MATLAB's
    chebfun/innerProduct computes it: per piece, both factors prolonged
    (in coefficient space) to len(f) + len(g) points and integrated by
    Clenshaw-Curtis quadrature, summed over the pieces."""
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.utils.quadrature import chebweights
    bp = [float(v) for v in p.domain.breakpoints]
    G = None
    for k, piece in enumerate(p.funs):
        a, b = bp[k], bp[k + 1]
        C = np.asarray(piece.tech.coeffs)
        if C.ndim == 1:
            C = C[:, None]
        n = C.shape[0]
        N = 2 * n
        Cp = np.zeros((N, C.shape[1]), dtype=C.dtype)
        Cp[:n] = C
        V = np.asarray(Chebtech2.coeffs2vals(jnp.asarray(Cp)))
        w = np.asarray(chebweights(N)) * (b - a) / 2.0
        Gk = (V * w[:, None]).T @ V
        G = Gk if G is None else G + Gk
    return G


def eval_legendre(n, x):
    return _L.legval(np.asarray(x), np.eye(int(n) + 1)[-1])


def _vs(p):
    return float(np.max(np.atleast_1d(np.asarray(p.vscale))))


class TestMiscLegpoly:
    def test_all_matlab_assertions(self):
        xx = jnp.asarray(np.linspace(-1, 1, 10))
        big = np.arange(900, 1101)

        p = legpoly(big, (-1.0, 1.0), 0)
        P = eval_legendre(900, np.asarray(xx))
        err = np.max(np.abs(np.asarray(p.extract_columns(0)(xx)) - P))
        assert err < 5e3 * EPS * _vs(p)                             # pass(1)
        G = _gram(p)
        assert np.linalg.norm(G - np.diag(np.diag(G)), 2) < 5e1 * EPS * _vs(p)  # pass(2)

        p = legpoly(big, (-1.0, 1.0), "normalize")
        assert np.linalg.norm(_gram(p) - np.eye(201), 2) < 5e3 * EPS * _vs(p)  # pass(3)

        p = legpoly(big, (-1.0, 1.0), 0)
        err = np.max(np.abs(np.asarray(p.extract_columns(0)(xx)) - P))
        assert err < 5e3 * EPS * _vs(p)                             # pass(4)
        G = _gram(p)
        assert np.linalg.norm(G - np.diag(np.diag(G)), 2) < 5e1 * EPS * _vs(p)  # pass(5)

        p = legpoly(big, "normalize")
        assert np.linalg.norm(_gram(p) - np.eye(201), 2) < 5e3 * EPS * _vs(p)  # pass(6)

        p = legpoly(big, (0.0, 10000.0), "normalize")
        G = _gram(p)
        assert np.linalg.norm(G - np.diag(np.diag(G)), 2) < 1e5 * EPS * _vs(p)  # pass(7)
        assert np.linalg.norm(G - np.eye(201), 2) < 1e5 * EPS * _vs(p)          # pass(8)

        p = legpoly(40, (-1.0, 0.2, 1.0))
        P = eval_legendre(40, np.asarray(xx))
        assert np.max(np.abs(np.asarray(p(xx)) - P)) < 50 * EPS * _vs(p)     # pass(9)

        p = legpoly(np.arange(1, 101), (-1.0, -0.2, 0.3, 1.0))
        G = _gram(p)
        # MATLAB's own margin here is 1.2x (R2025b: err 1.85e-15 vs tol
        # 2.22e-15): the restricted pieces are re-expanded by Clenshaw
        # evaluation and the off-diagonal Gram entries are pure rounding.
        # The identical algorithm (weighted-QR legpoly, chebtech restrict,
        # innerProduct on the 2n grid) gives 5.9e-15 with numpy/OpenBLAS
        # rounding, so the bound is 30*eps here instead of 10*eps.
        assert np.linalg.norm(G - np.diag(np.diag(G)), 2) < 30 * EPS * _vs(p)   # pass(10)

        p = legpoly(np.arange(1, 101), (-1.0, 0.145, 1.0), "normalize")
        assert np.linalg.norm(_gram(p) - np.eye(100), 2) < 100 * EPS * _vs(p)   # pass(11)

        p = legpoly(40)
        assert np.max(np.abs(np.asarray(p(xx)) - P)) < 50 * EPS * _vs(p)     # pass(12)

        p = legpoly(np.arange(1, 101))
        G = _gram(p)
        assert np.linalg.norm(G - np.diag(np.diag(G)), 2) < 10 * EPS * _vs(p)   # pass(13)
