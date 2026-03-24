"""Ultraspherical spectral discretization for ODEs on bounded intervals.

Implements the ultraspherical (Gegenbauer) spectral method of Olver & Townsend
(SIAM Review, 2013), which yields sparse, well-conditioned linear systems for
differential operators.  The method represents operators as banded matrices in
the C^{(lambda)} basis rather than the standard Chebyshev basis.

Key objects
-----------
- ``diffmat(n, k)``       -- banded differentiation matrix, C^{(0)} -> C^{(k)}
- ``convertmat(n, k1, k2)`` -- conversion matrix, C^{(k1)} -> C^{(k2)}
- ``multmat(n, a, lam)``  -- multiplication-by-f matrix in C^{(lam)}
- ``UltraS``              -- high-level class wrapping the above

The basis C^{(0)} is Chebyshev T; C^{(1)} is Chebyshev U; for k >= 1,
C^{(k)} is the Gegenbauer (ultraspherical) polynomial basis with parameter k.

Translated from MATLAB Chebfun class @ultraS (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Array = jax.Array


# ===========================================================================
# Low-level matrix builders (pure functions, no class needed)
# ===========================================================================


def diffmat(n: int, k: int = 1) -> jnp.ndarray:
    """Differentiation matrix: Chebyshev T coefficients -> C^{(k)} coefficients.

    Returns the n-by-n matrix D such that, if ``u_T`` is the vector of
    Chebyshev T coefficients of a polynomial ``u``, then ``D @ u_T`` is
    the vector of C^{(k)} coefficients of the k-th derivative of ``u``.

    The matrix is upper bidiagonal (bandwidth 1 above the diagonal) for
    k = 1 and becomes upper triangular with bandwidth 1 for higher k via
    the recurrence::

        D_k = 2*(k-1)*J * D_{k-1},   where J = diag(1, 0, n-2)

    For k = 0 the identity is returned.

    Parameters
    ----------
    n : int
        Number of Chebyshev coefficients (matrix size n x n).
    k : int, default 1
        Differentiation order.  Must be >= 0.

    Returns
    -------
    D : jnp.ndarray, shape (n, n)
        Banded differentiation matrix.

    Notes
    -----
    Developer notes from MATLAB Chebfun:

    For m = 1: D = diag(0:n-1, 1)   (superdiagonal of 0,1,...,n-1)
    For m = 2: D = diag(2*(1:n-2), 1) * D_{m=1}  (multiply by 2*1*J, 2*2*J, ...)
    General: D_m = (2*(m-1)*I_shift) * D_{m-1} where I_shift shifts one up.

    The formula at the MATLAB level:
      D_m=1 = spdiags((0:n-1)', 1, n, n)
      D_m>1 = spdiags(2*(m-1)*ones(n,1), 1, n, n) * D_{m-1}

    References
    ----------
    .. [1] S. Olver and A. Townsend, "A fast and well-conditioned spectral
       method", SIAM Review, 55(3), 462-489, 2013.

    Provenance
    ----------
    MATLAB source : @ultraS/diffmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    convertmat, multmat, UltraS
    """
    if k == 0:
        return jnp.eye(n, dtype=jnp.float64)
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    # k = 1: MATLAB: D = spdiags((0:n-1)', 1, n, n)
    # For superdiagonal k>0, MATLAB spdiags places v(i+k) at (i, i+k) (1-indexed).
    # So position (i, i+1) gets v(i+1) = i (0-indexed: D[i,i+1] = i+1 for i=0..n-2).
    # This gives superdiagonal [1, 2, ..., n-1].
    D = jnp.diag(jnp.arange(1, n, dtype=jnp.float64), 1)

    # k > 1: D_s = spdiags(2*s*ones(n,1), 1, n, n) * D_{s-1}
    # spdiags(c*ones(n,1), 1, n, n) places c on the superdiagonal.
    # MATLAB: for superdiagonal +1 with constant vector c*ones(n,1):
    #   position (i, i+1) gets c (0-indexed: D[i,i+1] = c for i=0..n-2)
    # But wait: spdiags(c*ones(n,1), 1, n, n) for constant vector:
    #   v(i+1) = c -> D[i-1, i] = c (1-indexed) -> D[i,i+1] = c (0-indexed)
    # So the matrix is c * (ones on superdiagonal) = c * J
    # where J[i,i+1] = 1 for i=0..n-2.
    J = jnp.diag(jnp.ones(n - 1, dtype=jnp.float64), 1)
    for s in range(1, k):
        D = (2.0 * s * J) @ D

    return D


def _spconvert(n: int, lam: float) -> jnp.ndarray:
    """Single-step conversion matrix: C^{(lam)} -> C^{(lam+1)}, size n x n.

    The conversion relation is::

        C_j^{(lam)} = (lam / (j + lam)) * (C_j^{(lam+1)} - C_{j-2}^{(lam+1)})

    So the matrix has:
      - diagonal 1 at position (0,0) and lam/(lam+1) at (1,1)
      - main diagonal: lam / (lam + j)   for j = 0, 1, ..., n-1
      - superdiagonal +2: -lam / (lam + j)  for j = 0, 1, ..., n-3

    For lam = 0 (Chebyshev T -> Chebyshev U):
      - (0,0) = 1,  (1,1) = 1/2
      - main diag j >= 2: 1/2
      - super+2 diag j >= 0: -1/2

    Parameters
    ----------
    n : int
        Matrix size.
    lam : float
        Source basis parameter (output is C^{(lam+1)}).

    Returns
    -------
    T : jnp.ndarray, shape (n, n)
        Sparse-structured conversion matrix.

    Provenance
    ----------
    MATLAB source : @ultraS/spconvert.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    if n == 1:
        return jnp.ones((1, 1), dtype=jnp.float64)

    j = jnp.arange(n, dtype=jnp.float64)

    if lam == 0:
        # T_n -> U_n conversion: T_n = 0.5*(U_n - U_{n-2}) for n>=2, T_0=U_0, T_1=0.5*U_1
        # Main diagonal: [1, 0.5, 0.5, ..., 0.5]
        main_diag = jnp.where(j == 0, 1.0, 0.5)
        # Super+2 diagonal: [-0.5, ..., -0.5] (length n-2).
        # MATLAB: dg = 0.5*ones(n-2); T = spdiags([1 0; .5 0; dg -dg], [0 2], n, n)
        # A[j, j+2] = -dg[j] = -0.5 for all j.  (constant, independent of j)
        super2_diag = -0.5 * jnp.ones(n - 2, dtype=jnp.float64)
    else:
        # C^{(lam)} -> C^{(lam+1)} conversion:
        # A[j, j] = lam/(lam+j) (main diagonal)
        # A[j, j+2] = -lam/(lam+j+2) (super+2 diagonal)
        #
        # MATLAB: dg = lam/(lam + (2:n-1))  [length n-2]
        #   dg[0] = lam/(lam+2), dg[1] = lam/(lam+3), ..., dg[n-3] = lam/(lam+n-1)
        # T = spdiags([main_col, super2_col], [0 2], n, n)
        # A[j, j+2] = -dg[j] = -lam/(lam+j+2) for j=0..n-3
        main_diag = lam / (lam + j)
        # super2_diag[j] goes at position (j, j+2) -> value = -lam/(lam+j+2)
        super2_diag = -lam / (lam + j[:n - 2] + 2.0)

    T = jnp.diag(main_diag) + jnp.diag(super2_diag, 2)
    return T


def convertmat(n: int, k1: int, k2: int) -> jnp.ndarray:
    """Conversion matrix: C^{(k1)} -> C^{(k2+1)}, size n x n.

    Returns the n-by-n matrix that maps coefficients in the C^{(k1)}
    ultraspherical basis to coefficients in the C^{(k2+1)} basis.  If
    k2 < k1 the identity is returned.

    The matrix is formed as the product of single-step conversions::

        S = S_{k2} @ S_{k2-1} @ ... @ S_{k1}

    where each S_s is the n x n single-step matrix from C^{(s)} to C^{(s+1)}.

    Parameters
    ----------
    n : int
        Matrix size (number of coefficients).
    k1 : int
        Source basis parameter (C^{(k1)}).
    k2 : int
        Target basis parameter is C^{(k2+1)}.  Pass k2 = k1 - 1 to get
        the identity; pass k2 = k1 to get one step; etc.

    Returns
    -------
    S : jnp.ndarray, shape (n, n)
        Banded conversion matrix.

    Notes
    -----
    The common calling convention (mirroring the MATLAB code) is::

        S = convertmat(n, 0, k-1)   # Cheb T -> C^{(k)}
        S = convertmat(n, k-1, k-1) # C^{(k-1)} -> C^{(k)}  (one step)

    Examples
    --------
    >>> S = convertmat(6, 0, 0)   # Cheb T -> C^{(1)} = Cheb U
    >>> S.shape
    (6, 6)
    >>> import jax.numpy as jnp
    >>> jnp.allclose(S[0,0], 1.0)
    Array(True, dtype=bool)

    Provenance
    ----------
    MATLAB source : @ultraS/convertmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffmat, multmat
    """
    S = jnp.eye(n, dtype=jnp.float64)
    for s in range(k1, k2 + 1):
        S = _spconvert(n, float(s)) @ S
    return S


def _sphankel(r: jnp.ndarray) -> jnp.ndarray:
    """Sparse (dense) Hankel matrix from a first-column vector.

    A Hankel matrix is constant along anti-diagonals.  This is equivalent
    to an upside-down Toeplitz matrix.

    Parameters
    ----------
    r : jnp.ndarray, shape (m,)
        First anti-diagonal (bottom-left to top-right).  The Hankel matrix
        H has H[i, j] = r[m - 1 - (i + j)] when i + j < m.

    Returns
    -------
    H : jnp.ndarray, shape (m, m)
        Hankel matrix.

    Provenance
    ----------
    MATLAB source : @ultraS/sphankel.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    m = r.shape[0]
    if m == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    # Build Toeplitz from reversed r, then flip columns -> Hankel
    r_flip = r[::-1]
    # Toeplitz: T[i,j] = r_flip[|i-j|]
    # We build it via outer difference of indices
    i_idx = jnp.arange(m)
    j_idx = jnp.arange(m)
    idx = jnp.abs(i_idx[:, None] - j_idx[None, :])
    T = r_flip[idx]
    # Flip left-right to get Hankel
    H = jnp.fliplr(jnp.triu(T))
    return H


def multmat(n: int, a: jnp.ndarray, lam: int) -> jnp.ndarray:
    """Multiplication matrix in the C^{(lam)} ultraspherical basis.

    Returns the n-by-n matrix M such that, if ``u_lam`` is the vector of
    C^{(lam)} coefficients of ``u``, then ``M @ u_lam`` gives the C^{(lam)}
    coefficients of ``f * u``, where ``f`` is represented by its Chebyshev T
    coefficients ``a``.

    Three cases are handled:

    - lam = 0: Multiplication in Chebyshev T.  M is Toeplitz + Hankel + rank-1.
    - lam = 1: Multiplication in Chebyshev U.  M is Toeplitz + Hankel.
    - lam >= 2: Constructed via a three-term recurrence.

    Parameters
    ----------
    n : int
        Matrix size (number of modes).
    a : jnp.ndarray, shape (p,)
        Chebyshev T coefficients of the multiplier ``f``.  Will be
        zero-padded or truncated to length ``n``.
    lam : int
        Basis parameter.  Must be >= 0.

    Returns
    -------
    M : jnp.ndarray, shape (n, n)
        Multiplication matrix in C^{(lam)}.

    Notes
    -----
    Developer notes from MATLAB Chebfun:

    For lam = 0 (Chebyshev T):
      Uses T_j * T_k = (T_{j+k} + T_{|j-k|}) / 2
      -> M = Toeplitz(2*a[0], a) + Hankel(a[1:])  (scaled by 1/2)

    For lam = 1 (Chebyshev U):
      Uses U_j * U_k = (U_{j+k} - U_{|j-k|-2}) / ...
      -> M = Toeplitz - Hankel  (similar structure)

    For lam >= 2:
      Converts a from C^{(0)} to C^{(lam)}, then applies three-term
      recurrence for multiplication in the Gegenbauer basis.

    References
    ----------
    .. [1] S. Olver and A. Townsend, "A fast and well-conditioned spectral
       method", SIAM Review, 55(3), 462-489, 2013.

    Provenance
    ----------
    MATLAB source : @ultraS/multmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffmat, convertmat
    """
    # Prolong or truncate coefficients to length n
    if a.shape[0] < n:
        a = jnp.concatenate([a, jnp.zeros(n - a.shape[0], dtype=jnp.float64)])
    else:
        a = a[:n]

    # Scalar case
    if n == 1 or jnp.all(a[1:] == 0):
        return a[0] * jnp.eye(n, dtype=jnp.float64)

    if lam == 0:
        # Multiplication in Chebyshev T.
        # M = Toeplitz([2a0; a1; ...; a_{n-1}], [2a0, a1, ..., a_{n-1}]) / 2
        #   + Hankel([a1, ..., a_{n-1}]) / 2    (lower-right block)
        a2 = a / 2.0  # scaled coefficients
        # Build Toeplitz: T[i,j] = a2[|i-j|]  with T[0,0] = 2*a2[0] = a[0]
        i_idx = jnp.arange(n)
        j_idx = jnp.arange(n)
        top_row = jnp.concatenate([jnp.array([2.0 * a2[0]]), a2[1:]])
        col_idx = jnp.abs(i_idx[:, None] - j_idx[None, :])
        M = top_row[col_idx]

        # Add Hankel on the sub-block (rows 1..n-1, cols 0..n-2)
        if n >= 2:
            H = _sphankel(a2[1:])  # (n-1) x (n-1)
            # Place H into rows [1:n], cols [0:n-1]
            M = M.at[1:, :n - 1].add(H)

    elif lam == 1:
        # Multiplication in Chebyshev U.
        # M = Toeplitz([2a0; a1; ...; a_{n-1}], [2a0, a1, ..., a_{n-1}]) / 2
        #   - Hankel([a2, ..., a_{n-1}]) / 2   on sub-block rows [0:n-2], cols [0:n-2]
        a2 = a / 2.0
        i_idx = jnp.arange(n)
        j_idx = jnp.arange(n)
        top_row = jnp.concatenate([jnp.array([2.0 * a2[0]]), a2[1:]])
        col_idx = jnp.abs(i_idx[:, None] - j_idx[None, :])
        M = top_row[col_idx]

        # Subtract Hankel on sub-block (rows 0..n-3, cols 0..n-3)
        if n >= 3:
            H = _sphankel(a2[2:])  # (n-2) x (n-2)
            M = M.at[:n - 2, :n - 2].add(-H)

    else:
        # lam >= 2: three-term recurrence in C^{(lam)}.
        # First convert a from C^{(0)} to C^{(lam)}.
        a = convertmat(n, 0, lam - 1) @ a

        m = 2 * n
        eye = jnp.eye(m, dtype=jnp.float64)

        # Multiplication-by-x matrix in C^{(lam)}:
        #   Mx_{j,j-1} = j / (2*(lam + j - 1))
        #   Mx_{j,j}   = 0
        #   Mx_{j,j+1} = (lam + 2*j + 2*lam - lam) ... see formula below
        # d1[j] = (1 + [2*lam, 2*lam+1, ..., 2*lam+m-2]) / [1, 2*(lam+1), ..., 2*(lam+m-1)]
        # d2[j] = j / (2*(lam + j - 1))   for j=1..m, with d2[0]=0 implied

        d1_num = jnp.concatenate([
            jnp.array([1.0]),
            2.0 * lam + jnp.arange(m - 1, dtype=jnp.float64)
        ])
        d1_den = jnp.concatenate([
            jnp.array([1.0]),
            2.0 * (lam + 1.0 + jnp.arange(m - 1, dtype=jnp.float64))
        ])
        d1 = d1_num / d1_den  # superdiagonal of Mx

        j_idx = jnp.arange(1, m + 1, dtype=jnp.float64)
        d2 = j_idx / (2.0 * (lam - 1.0 + j_idx))  # subdiagonal of Mx

        # Mx = diag(d2[1:m], -1) + diag(d1[0:m], +1), truncated to m x m
        Mx = jnp.diag(d2[:m - 1], -1) + jnp.diag(d1[:m - 1], 1)
        Mx = Mx[:m, :m]

        M1 = 2.0 * lam * Mx

        # Three-term recurrence: M_f = sum_j a[j] * M_j
        # M_0 = I, M_1 = 2*lam*Mx
        # M_{j+1} = 2*(j+lam)/(j+1) * Mx @ M_j - (j+2*lam-1)/(j+1) * M_{j-1}
        M = a[0] * eye
        M = M + a[1] * M1

        M0 = eye
        Mcur = M1
        nnz_a = jnp.sum(jnp.abs(a) > jnp.finfo(float).eps).item()
        n_terms = min(int(nnz_a), a.shape[0] - 2)

        for nn in range(n_terms):
            M2 = (2.0 * (nn + lam) / (nn + 1.0)) * Mx @ Mcur \
                 - ((nn + 2.0 * lam - 1.0) / (nn + 1.0)) * M0
            M = M + a[nn + 2] * M2
            M0 = Mcur
            Mcur = M2

        # Truncate to n x n
        M = M[:n, :n]

    return M


# ===========================================================================
# UltraS class
# ===========================================================================


class UltraS(eqx.Module):
    """Ultraspherical spectral discretization on a bounded interval.

    Represents differential operators as banded matrices in the ultraspherical
    (Gegenbauer) polynomial basis C^{(k)}, where k tracks the differentiation
    order.  This is the key to obtaining well-conditioned linear systems for
    ODEs.

    The basis sequence is::

        C^{(0)} = Chebyshev T polynomials  (input representation)
        C^{(1)} = Chebyshev U polynomials
        C^{(2)}, C^{(3)}, ...              (Gegenbauer polynomials)

    A k-th order differential operator maps C^{(0)} -> C^{(k)}.

    Attributes
    ----------
    n : int
        Number of discretization points / Chebyshev coefficients.
    domain : Domain
        The interval on which the operator is defined.

    Methods
    -------
    diffmat(k)
        k-th order differentiation matrix, C^{(0)} -> C^{(k)}.
    conversion(k)
        Conversion matrix S_k, C^{(k-1)} -> C^{(k)} (one step).
    convertmat(k1, k2)
        Multi-step conversion, C^{(k1)} -> C^{(k2+1)}.
    multmat(a, lam)
        Multiplication-by-f matrix in C^{(lam)} basis.
    points()
        Chebyshev-2 collocation points for this discretization.

    Examples
    --------
    Solve u'' + u = 0 with u(-1) = sin(-1), u(1) = sin(1):

    >>> import jax.numpy as jnp
    >>> from chebfunjax.discretization.ultras import UltraS
    >>> from chebfunjax.domain import Domain
    >>> n = 32
    >>> disc = UltraS(n=n, domain=Domain((-1.0, 1.0)))
    >>> D2 = disc.diffmat(2)
    >>> S = disc.conversion(1) @ disc.conversion(2)   # C^{(0)} -> C^{(2)}
    >>> # Interior rows: (D2 + S) u = 0
    >>> # Boundary rows: u(-1) = sin(-1), u(1) = sin(1)
    >>> L = jnp.vstack([D2 + S])   # simplified; see test for full system

    Notes
    -----
    The ultraspherical method is described in detail in:

    .. [1] S. Olver and A. Townsend, "A fast and well-conditioned spectral
       method", SIAM Review, 55(3), 462-489, 2013.

    Provenance
    ----------
    MATLAB source : @ultraS/ultraS.m, @ultraS/diffmat.m,
        @ultraS/convertmat.m, @ultraS/multmat.m, @ultraS/spconvert.m,
        @ultraS/sphankel.m
    Chebfun commit: 7574c77
    Original authors: Sheehan Olver, Alex Townsend.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    diffmat, convertmat, multmat
    """

    n: int = eqx.field(static=True)
    domain: Domain

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n: int, domain: Union[Domain, tuple] = (-1.0, 1.0)):
        """Create an ultraspherical discretization of size n on domain.

        Parameters
        ----------
        n : int
            Number of discretization points.
        domain : Domain or (a, b), default (-1, 1)
            The interval.  If a tuple is passed it is converted to a Domain.
        """
        object.__setattr__(self, 'n', n)
        if not isinstance(domain, Domain):
            domain = Domain(tuple(float(x) for x in domain))
        object.__setattr__(self, 'domain', domain)

    # ------------------------------------------------------------------
    # Core matrix builders
    # ------------------------------------------------------------------

    def diffmat(self, k: int = 1) -> jnp.ndarray:
        """k-th order differentiation matrix, C^{(0)} -> C^{(k)}, size n x n.

        Maps Chebyshev T coefficients to C^{(k)} coefficients of the k-th
        derivative.  The matrix is banded (bandwidth k above the main
        diagonal).

        Includes the domain scaling: if the interval is [a, b], the
        matrix is multiplied by ``(2 / (b - a))^k``.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        D : jnp.ndarray, shape (n, n)
            Differentiation matrix in the ultraspherical sense.

        Examples
        --------
        >>> disc = UltraS(n=8, domain=Domain((-1.0, 1.0)))
        >>> D1 = disc.diffmat(1)
        >>> D1.shape
        (8, 8)
        >>> float(D1[0, 1])   # first C^{(1)} coefficient of derivative of T_1
        1.0

        Provenance
        ----------
        MATLAB source : @ultraS/diffmat.m, @ultraS/diff.m
        Chebfun commit: 7574c77
        """
        D = diffmat(self.n, k)
        # Scale by (2/(b-a))^k for non-standard domains
        a, b = self.domain.a, self.domain.b
        scl = (2.0 / (b - a)) ** k
        return scl * D

    def conversion(self, k: int) -> jnp.ndarray:
        """Single-step conversion matrix S_k: C^{(k-1)} -> C^{(k)}, size n x n.

        This is the elementary building block for ultraspherical operators.
        Stacking conversions gives the full change-of-basis from C^{(0)}
        (Chebyshev T) to C^{(m)}::

            S = disc.conversion(m) @ ... @ disc.conversion(2) @ disc.conversion(1)
              = disc.convertmat(0, m-1)

        Parameters
        ----------
        k : int
            Target basis order (>= 1).  Returns a matrix mapping C^{(k-1)} -> C^{(k)}.

        Returns
        -------
        S : jnp.ndarray, shape (n, n)
            One-step conversion matrix.

        Provenance
        ----------
        MATLAB source : @ultraS/spconvert.m, @ultraS/convert.m
        Chebfun commit: 7574c77
        """
        return _spconvert(self.n, float(k - 1))

    def convertmat(self, k1: int, k2: int) -> jnp.ndarray:
        """Multi-step conversion matrix: C^{(k1)} -> C^{(k2+1)}, size n x n.

        Equivalent to chaining ``k2 - k1 + 1`` single-step conversions.
        If k2 < k1 the identity is returned.

        Parameters
        ----------
        k1 : int
            Source basis order.
        k2 : int
            Target is C^{(k2+1)}.

        Returns
        -------
        S : jnp.ndarray, shape (n, n)
            Conversion matrix.

        Provenance
        ----------
        MATLAB source : @ultraS/convertmat.m
        Chebfun commit: 7574c77
        """
        return convertmat(self.n, k1, k2)

    def multmat(self, a: jnp.ndarray, lam: int) -> jnp.ndarray:
        """Multiplication matrix by f in C^{(lam)} basis, size n x n.

        Given the Chebyshev T coefficients ``a`` of a function ``f``,
        returns the matrix M such that M @ u gives the C^{(lam)}
        coefficients of ``f * u``, where ``u`` is in C^{(lam)}.

        Parameters
        ----------
        a : jnp.ndarray, shape (p,)
            Chebyshev T coefficients of the multiplier f.
        lam : int
            Basis parameter (>= 0).

        Returns
        -------
        M : jnp.ndarray, shape (n, n)
            Multiplication matrix.

        Provenance
        ----------
        MATLAB source : @ultraS/multmat.m
        Chebfun commit: 7574c77
        """
        return multmat(self.n, a, lam)

    def points(self) -> jnp.ndarray:
        """Chebyshev 2nd-kind collocation points on this domain.

        Returns
        -------
        x : jnp.ndarray, shape (n,)
            Chebyshev points of the 2nd kind mapped to [a, b].

        Examples
        --------
        >>> disc = UltraS(n=5, domain=Domain((-1.0, 1.0)))
        >>> x = disc.points()
        >>> x.shape
        (5,)
        >>> float(x[0])   # leftmost point = -1
        -1.0

        Provenance
        ----------
        MATLAB source : @ultraS/ultraS.m  (uses chebtech2 grid)
        Chebfun commit: 7574c77
        """
        a, b = self.domain.a, self.domain.b
        x_std = chebpts(self.n, kind=2)  # on [-1, 1]
        # Map to [a, b]: x = a + (b - a) * (x_std + 1) / 2
        return a + (b - a) * 0.5 * (x_std + 1.0)

    # ------------------------------------------------------------------
    # High-level ODE system builder
    # ------------------------------------------------------------------

    def build_ode_system(
        self, coeffs_list: list[jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Build the ultraspherical ODE system matrix and conversion matrix.

        Given coefficient functions [c_0, c_1, ..., c_m] representing the
        differential operator ``L u = c_m u^{(m)} + ... + c_1 u' + c_0 u``,
        returns the banded system matrix L_mat and the conversion matrix S
        from C^{(0)} to C^{(m)}.

        The operator is assembled as::

            L = sum_{j=0}^{m} S_{j->m} * M_{c_j}^{(j)} * D^j

        where S_{j->m} converts from C^{(j)} to C^{(m)}.

        Parameters
        ----------
        coeffs_list : list of jnp.ndarray
            Chebyshev T coefficient vectors [c_0, c_1, ..., c_m], where
            c_j are the coefficients of the multiplier for D^j.
            Length m+1 defines the order m.

        Returns
        -------
        L_mat : jnp.ndarray, shape (n, n)
            Assembled operator matrix.
        S : jnp.ndarray, shape (n, n)
            Conversion matrix from C^{(0)} to C^{(m)} (used to convert the
            right-hand side).

        Provenance
        ----------
        MATLAB source : @ultraS/quasi2diffmat.m
        Chebfun commit: 7574c77
        """
        # flip to match MATLAB convention: c = fliplr(disc.coeffs)
        # coeffs_list[0] = c_0 (constant term), coeffs_list[-1] = c_m (highest)
        m = len(coeffs_list) - 1  # ODE order
        n = self.n

        L_mat = jnp.zeros((n, n), dtype=jnp.float64)
        for j, cj in enumerate(coeffs_list):
            Dj = self.diffmat(j)          # D^j: C^{(0)} -> C^{(j)}
            Mj = self.multmat(cj, j)       # mult by c_j in C^{(j)}
            # Convert from C^{(j)} to C^{(m)}
            Sj2m = convertmat(n, j, m - 1)  # C^{(j)} -> C^{(m)}
            L_mat = L_mat + Sj2m @ Mj @ Dj

        # Conversion matrix: C^{(0)} -> C^{(m)}
        S = convertmat(n, 0, m - 1)

        return L_mat, S

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        a, b = self.domain.a, self.domain.b
        return f"UltraS(n={self.n}, domain=[{a}, {b}])"
