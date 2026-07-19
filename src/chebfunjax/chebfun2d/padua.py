# uses-numpy: Padua-point index bookkeeping and the small dense DCT-style
# matrices are built with numpy in this non-JIT constructor helper; the
# returned arrays are converted to jax.numpy before entering the library.
"""Padua points and the Padua interpolation constructor for Chebfun2.

The Padua points are the (currently) only known optimal point set for total
degree bivariate polynomial interpolation on a square that admits a fast
transform to Chebyshev coefficients.  This module ports MATLAB Chebfun's
``paduapts.m`` and ``@chebfun2/paduaVals2coeffs.m`` and exposes them so a
Chebfun2 can be built directly from data sampled at the Padua points
(``chebfun2(f, dom, 'padua')`` in MATLAB).

Provenance
----------
MATLAB source : paduapts.m, @chebfun2/paduaVals2coeffs.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The Chebfun
    Developers.
References:
    Marco Caliari, Stefano De Marchi, Alvise Sommariva, Marco Vianello,
    "Padua2DM: fast interpolation and cubature at the Padua points in
    Matlab/Octave."
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.quadrature import chebpts_ab


def _padua_grid(n: int, dom=(-1.0, 1.0, -1.0, 1.0)):
    """Padua points of degree ``n`` on ``dom`` and the tensor-grid mask.

    Returns ``(xy, idx)`` where ``xy`` is an ``(m, 2)`` numpy array of Padua
    points (column-major ordering, consistent with Padua2DM) and ``idx`` is
    the boolean ``(n+1, n+2)``-or-``(n+2, n+1)`` mask that selects them from
    the full Chebyshev tensor grid.
    """
    xa, xb, ya, yb = dom
    if n == 0:
        return np.array([[xa, ya]], dtype=np.float64), np.array([[True]])

    # 1-D 2nd-kind Chebyshev grids, order flipped (descending) for
    # consistency with Padua2DM.
    xn1 = np.asarray(chebpts_ab(n + 1, xa, xb, 2), dtype=np.float64)[::-1]
    xn2 = np.asarray(chebpts_ab(n + 2, ya, yb, 2), dtype=np.float64)[::-1]

    # Full tensor grid: x1[r, c] = xn1[c], x2[r, c] = xn2[r], shape (n+2, n+1).
    x1, x2 = np.meshgrid(xn1, xn2)

    # Extract every other term (MATLAB column-major logical indexing).
    idx = np.ones((n + 1) * (n + 2), dtype=bool)
    idx[0::2] = False
    if n % 2 == 1:
        idx = idx.reshape(n + 2, n + 1, order="F")
    else:
        idx = idx.reshape(n + 1, n + 2, order="F").T

    sel = idx.flatten(order="F")
    xy = np.column_stack([x1.flatten(order="F")[sel],
                          x2.flatten(order="F")[sel]])
    return xy, idx


def paduapts(n: int, dom=(-1.0, 1.0, -1.0, 1.0)) -> jnp.ndarray:
    """Padua points of degree ``n`` on ``dom``.

    ``paduapts(n)`` returns an ``(m, 2)`` array of the ``x`` and ``y``
    coordinates of the degree-``n`` first-kind Padua points on
    ``[-1, 1] x [-1, 1]``; ``paduapts(n, [a, b, c, d])`` maps them to
    ``[a, b] x [c, d]``.  The ordering matches Padua2DM (and MATLAB
    ``paduapts``).

    Provenance
    ----------
    MATLAB source : paduapts.m
    Chebfun commit: 7574c77

    See Also
    --------
    chebfunjax.chebfun2d.chebfun2.Chebfun2.from_padua
    """
    xy, _ = _padua_grid(n, dom)
    return jnp.asarray(xy, dtype=jnp.float64)


def paduavals2coeffs(f, dom=(-1.0, 1.0, -1.0, 1.0)):
    """Bivariate Chebyshev coefficients of the Padua interpolant to ``f``.

    Given the values ``f`` of a function sampled at ``paduapts(n, dom)`` (in
    Padua2DM order), returns ``(C, V)`` where ``C`` is the ``(n+1, n+1)``
    bivariate Chebyshev coefficient matrix (row index = degree in ``y``,
    column index = degree in ``x``, upper-left triangular in total degree)
    and ``V`` is the matrix of values of the interpolant on the ``(n+1)``
    by ``(n+1)`` 2nd-kind Chebyshev tensor grid (row = ``y`` node, column =
    ``x`` node, in descending-node order).

    Provenance
    ----------
    MATLAB source : @chebfun2/paduaVals2coeffs.m
    Chebfun commit: 7574c77
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    m = f.size
    n = int(round(-1.5 + np.sqrt(0.25 + 2.0 * m)))

    # Padua points on the reference square (coefficients are domain-free).
    x, idx = _padua_grid(n, (-1.0, 1.0, -1.0, 1.0))

    # Interpolation weights.
    w = np.full(x.shape[0], 1.0 / (n * (n + 1)), dtype=np.float64)
    corner = np.all(np.abs(x) == 1.0, axis=1)
    interior = np.all(np.abs(x) != 1.0, axis=1)
    w[corner] *= 0.5
    w[interior] *= 2.0

    # Fill G on the tensor-grid mask (column-major, as in MATLAB).
    G = np.zeros(idx.shape, dtype=np.float64)
    Gf = G.flatten(order="F")
    sel = idx.flatten(order="F")
    Gf[sel] = 4.0 * w * f
    G = Gf.reshape(idx.shape, order="F")

    # Chebyshev-Vandermonde (DCT) matrices.
    a1 = np.arange(n + 1)
    a2 = np.arange(n + 2)
    Tn1 = np.cos(np.outer(a1, a1) * np.pi / n)
    Tn2 = np.cos(np.outer(a2, a2) * np.pi / (n + 1))
    C = Tn2 @ G @ Tn1

    # Modify a few entries and drop the last row.
    C[0, :] *= 0.5
    C[:, 0] *= 0.5
    C[0, -1] *= 0.5
    C = C[:-1, :]

    # Upper-left triangular part (total-degree truncation):
    # equivalent to C = fliplr(triu(fliplr(C))).
    C = np.fliplr(np.triu(np.fliplr(C)))

    # Values on the (n+1) x (n+1) tensor grid.
    V = Tn1 @ C @ Tn1
    return C, V
