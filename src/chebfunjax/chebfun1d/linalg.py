# uses-numpy: continuous QR uses numpy for Householder reflections (not JIT-safe)
"""Quasimatrix linear algebra: QR, SVD for multi-column Chebfuns.

A *quasimatrix* is a collection of Chebfun columns sharing a common domain.
It supports QR factorization (giving orthonormal Chebfun columns and an upper-
triangular matrix R) and SVD (via QR + discrete SVD).

The QR algorithm is the continuous Householder method of Trefethen (2010),
implemented in terms of continuous L2 inner products.

Translated from MATLAB Chebfun @chebfun/qr.m, @chebfun/svd.m, and
abstractQR.m (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
[1] L.N. Trefethen, "Householder triangularization of a quasimatrix",
    IMA J Numer Anal (2010) 30(4): 887–897.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.polynomials import legpoly

# Machine epsilon
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Quasimatrix — multi-column Chebfun
# ============================================================================


class Quasimatrix:
    """A quasimatrix: a finite collection of Chebfun columns on a shared domain.

    A quasimatrix is the continuous analogue of a matrix: its columns are
    functions (Chebfun objects) rather than vectors of numbers.  Operations
    such as QR and SVD extend naturally to this setting via continuous L2
    inner products.

    This class is intentionally lightweight: it holds a Python list of
    single-piece :class:`~chebfunjax.chebfun1d.chebfun.Chebfun` objects and
    a shared :class:`~chebfunjax.domain.Domain`.  All columns must share the
    same single-interval domain.

    Parameters
    ----------
    cols : list[Chebfun]
        The columns of the quasimatrix.  All must have a single-piece domain
        that matches ``domain``.
    domain : Domain
        The shared domain (single interval [a, b]).

    Notes
    -----
    JAX contract: NOT JIT-safe (construction is adaptive; QR/SVD use Python
    loops with data-dependent termination).

    Provenance
    ----------
    MATLAB source : @chebfun/qr.m, abstractQR.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun.qr, Chebfun.svd
    """

    def __init__(self, cols: list[Chebfun], domain: Domain) -> None:
        if len(cols) == 0:
            raise ValueError(
                "Quasimatrix must have at least one column."
            )
        if domain.n_intervals != 1:
            raise ValueError(
                f"Quasimatrix only supports single-interval domains, "
                f"got domain with {domain.n_intervals} intervals."
            )
        for i, col in enumerate(cols):
            if col.domain != domain:
                raise ValueError(
                    f"Column {i} has domain {col.domain} which does not match "
                    f"the quasimatrix domain {domain}."
                )
        self.cols = list(cols)
        self.domain = domain

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cols(self) -> int:
        """Number of columns."""
        return len(self.cols)

    @property
    def shape(self) -> tuple[str, int]:
        """(inf, n_cols) analogous to numpy shape."""
        return ("inf", self.n_cols)

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Chebfun:
        """Return the idx-th column as a Chebfun."""
        return self.cols[idx]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x) -> jnp.ndarray:
        """Evaluate all columns at point(s) x.

        Parameters
        ----------
        x : scalar or array_like, shape (m,)
            Evaluation point(s) in the domain.

        Returns
        -------
        jnp.ndarray, shape (m, n_cols) or (n_cols,) for scalar x
        """
        import jax.numpy as jnp
        results = [col(x) for col in self.cols]
        # Each result is shape () or (m,)
        out = jnp.stack(results, axis=-1)
        return out

    # ------------------------------------------------------------------
    # Inner product (continuous L2)
    # ------------------------------------------------------------------

    def _col_inner(self, i: int, j: int) -> float:
        """<cols[i], cols[j]> on the domain."""
        return float(self.cols[i].inner(self.cols[j]))

    # ------------------------------------------------------------------
    # Arithmetic: scale a column by a scalar
    # ------------------------------------------------------------------

    def _scale_col(self, i: int, s: float) -> None:
        """Scale column i in-place by scalar s."""
        self.cols[i] = self.cols[i] * s

    # ------------------------------------------------------------------
    # Factory: build from list of callables
    # ------------------------------------------------------------------

    @classmethod
    def from_functions(
        cls,
        funcs: Sequence,
        domain: Domain | None = None,
        n: int | None = None,
    ) -> "Quasimatrix":
        """Build a quasimatrix from a list of callables.

        Parameters
        ----------
        funcs : list of callables
            Each callable is a vectorized function f(x) on the domain.
        domain : Domain or None
            Domain for all columns. Default is [-1, 1].
        n : int or None
            Fixed degree (None = adaptive).

        Returns
        -------
        Quasimatrix
        """
        if domain is None:
            domain = Domain((-1.0, 1.0))
        cols = [Chebfun.from_function(f, domain=domain, n=n) for f in funcs]
        return cls(cols=cols, domain=domain)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.n_cols
        a, b = self.domain.a, self.domain.b
        col_word = "column" if n == 1 else "columns"
        return f"Quasimatrix [{a}, {b}] with {n} {col_word}"


# ============================================================================
# Core: continuous inner product helper for a domain
# ============================================================================


def _chebfun_inner(f: Chebfun, g: Chebfun) -> float:
    """L2 inner product int_a^b f(x)*g(x) dx.

    Parameters
    ----------
    f, g : Chebfun
        Single-piece Chebfuns on the same domain.

    Returns
    -------
    float
    """
    return float(f.inner(g))


def _chebfun_norm_est(f: Chebfun) -> float:
    """Cheap norm estimate: sqrt of inner product."""
    return float(f.norm())


# ============================================================================
# Legendre starting basis for Householder QR
# ============================================================================


def _legendre_basis(n_cols: int, domain: Domain) -> list[Chebfun]:
    """Build L2-orthonormal Legendre polynomials on the domain.

    Constructs the first ``n_cols`` Legendre polynomials P_0, ..., P_{n-1},
    L2-normalised so that <P_i, P_j> = delta_{ij}, on the given interval
    [a, b].

    The standard normalisation on [-1, 1] is
        ||P_k||_2^2 = 2 / (2k + 1),
    so on [a, b] (length L = b - a) it becomes L / (2k + 1).
    We therefore scale each P_k by sqrt((2k + 1) / L).

    Parameters
    ----------
    n_cols : int
        Number of Legendre basis functions.
    domain : Domain
        Single-interval domain [a, b].

    Returns
    -------
    list[Chebfun], length n_cols
        Orthonormal Legendre Chebfuns on [a, b].
    """
    a, b = domain.a, domain.b
    L = b - a  # length of interval

    basis = []
    for k in range(n_cols):
        # Chebyshev (first-kind) coefficients of Legendre P_k on [-1, 1]
        cheb_ref = legpoly(k, normalize=False)  # shape (k+1,)
        # Build the Chebtech2 on the reference interval
        tech = Chebtech2.from_coeffs(cheb_ref)
        # Wrap in a _Piece on [a, b] — NO rescaling of coefficients needed:
        # the reference Chebtech2 on [-1, 1] gives the same polynomial values
        # when evaluated via the affine map from [a, b] to [-1, 1].
        piece = _Piece(tech=tech, interval=(float(a), float(b)))
        col = Chebfun(funs=[piece], domain=domain)

        # L2 normalise: ||P_k||_2 on [a, b] = sqrt(L / (2k + 1))
        norm_sq = L / (2.0 * k + 1.0)
        scale = 1.0 / np.sqrt(norm_sq)
        col = col * scale
        basis.append(col)

    return basis


# ============================================================================
# abstractQR: continuous Householder QR
# ============================================================================


def abstract_qr(
    cols: list[Chebfun],
    basis: list[Chebfun],
    inner: callable = _chebfun_inner,
    norm_est: callable = _chebfun_norm_est,
    tol: float = _EPS,
) -> tuple[list[Chebfun], jnp.ndarray]:
    """Continuous Householder QR factorisation (abstractQR).

    Factorises the quasimatrix A (a list of Chebfun columns) as A = Q * R
    where Q has L2-orthonormal columns and R is upper-triangular.

    This is a direct translation of MATLAB's ``abstractQR.m``, implementing
    the algorithm of Trefethen [1] using a continuous L2 inner product.

    Parameters
    ----------
    cols : list[Chebfun], length n
        The columns of the quasimatrix to factorise.  Modified in-place
        during the algorithm (pass copies if you need the original).
    basis : list[Chebfun], length n
        Initial L2-orthonormal basis (e.g. Legendre polynomials).
        Modified in-place.
    inner : callable, optional
        Binary function ``inner(f, g) -> float`` computing the L2 inner
        product.  Default: continuous integral on the domain.
    norm_est : callable, optional
        Unary function ``norm_est(f) -> float`` providing a cheap upper
        bound on the L2 norm of a Chebfun.  Default: exact L2 norm.
    tol : float, optional
        Tolerance for determining near-zero norms.  Default: machine epsilon.

    Returns
    -------
    Q : list[Chebfun], length n
        L2-orthonormal Chebfun columns.
    R : jnp.ndarray, shape (n, n)
        Upper-triangular factor.

    References
    ----------
    [1] L.N. Trefethen, "Householder triangularization of a quasimatrix",
        IMA J Numer Anal (2010) 30(4): 887–897.

    Provenance
    ----------
    MATLAB source : abstractQR.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun.qr, Chebfun.svd
    """
    n = len(cols)
    # Work on mutable copies
    A = [col for col in cols]   # working columns of A
    E = [e for e in basis]      # working Legendre basis
    V = [None] * n              # Householder vectors

    R = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        # Scale: max of ||E[:,k]|| and ||A[:,k]|| (both ~1 for well-scaled A)
        scl = max(norm_est(E[k]), norm_est(A[k]))

        # Inner product of E[:,k] and A[:,k]
        ex = inner(E[k], A[k])
        aex = abs(ex)

        # Adjust sign of E[:,k] to match sign of A[:,k] projected onto E[:,k]
        if aex < tol * scl:
            s = 1.0
        else:
            s = -np.sign(ex / aex)
        E[k] = E[k] * s

        # Diagonal entry: norm of A[:,k]
        r = np.sqrt(max(inner(A[k], A[k]), 0.0))
        R[k, k] = r

        # Householder vector: v = r*E[:,k] - A[:,k]
        v = E[k] * r - A[k]

        # Re-orthogonalise v against previous E[:,i]
        for i in range(k):
            ev = inner(E[i], v)
            v = v - E[i] * ev

        # Normalise v
        nv = np.sqrt(max(inner(v, v), 0.0))
        if nv < tol * scl:
            v = E[k]
        else:
            v = v * (1.0 / nv)

        V[k] = v

        # Apply Householder reflection to remaining columns
        for j in range(k + 1, n):
            av = inner(v, A[j])
            A[j] = A[j] - v * (2.0 * av)

            # Off-diagonal entry
            rr = inner(E[k], A[j])
            R[k, j] = rr

            # Project out E[:,k] component from A[:,j]
            A[j] = A[j] - E[k] * rr

    # Form Q from the Householder vectors
    Q = [e for e in E]  # start from final E basis
    for k in range(n - 1, -1, -1):
        for j in range(k, n):
            vq = inner(V[k], Q[j])
            Q[j] = Q[j] - V[k] * (2.0 * vq)

    return Q, jnp.array(R, dtype=jnp.float64)


# ============================================================================
# Public: qr and svd on Chebfun quasimatrices
# ============================================================================


def qr_quasimatrix(
    qm: Quasimatrix,
) -> tuple[Quasimatrix, jnp.ndarray]:
    """QR factorization of a quasimatrix.

    Factorises ``qm`` (an n-column quasimatrix on [a, b]) as A = Q * R where
    Q has L2-orthonormal columns (as a Quasimatrix) and R is an n x n upper-
    triangular matrix.

    Parameters
    ----------
    qm : Quasimatrix
        The quasimatrix to factorise.  Must have a single-interval domain.

    Returns
    -------
    Q : Quasimatrix
        n-column quasimatrix with L2-orthonormal columns.
    R : jnp.ndarray, shape (n, n)
        Upper-triangular factor.  A = Q @ R in the quasimatrix sense:
        A[:,j] = sum_i Q[:,i] * R[i,j].

    Notes
    -----
    NOT JIT-safe (Python loops, adaptive construction).

    Algorithm: Continuous Householder method [1].  The starting orthonormal
    basis E is formed from L2-normalised Legendre polynomials on the domain.

    References
    ----------
    [1] L.N. Trefethen, "Householder triangularization of a quasimatrix",
        IMA J Numer Anal (2010) 30(4): 887–897.

    Provenance
    ----------
    MATLAB source : @chebfun/qr.m, abstractQR.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    svd_quasimatrix, Chebfun.qr, Chebfun.svd
    """
    n = qm.n_cols
    domain = qm.domain

    if n == 1:
        # Trivial: single column, just normalise
        f = qm.cols[0]
        r = float(f.norm())
        if r > 0.0:
            Q_col = f * (1.0 / r)
        else:
            # Zero function: return 1/sqrt(b-a)
            a, b = domain.a, domain.b
            Q_col = Chebfun.from_function(
                lambda x: jnp.full_like(x, 1.0 / np.sqrt(b - a)),
                domain=domain,
            )
        R = jnp.array([[r]], dtype=jnp.float64)
        return Quasimatrix(cols=[Q_col], domain=domain), R

    # Build Legendre basis
    basis = _legendre_basis(n, domain)

    # Run abstractQR
    Q_cols, R = abstract_qr(
        cols=list(qm.cols),
        basis=basis,
        inner=_chebfun_inner,
        norm_est=_chebfun_norm_est,
    )

    # Ensure diagonal of R is non-negative (A = QR = (Q*S)*(S*R))
    # This matches MATLAB's sign convention
    diag_signs = np.array(
        [np.sign(float(R[i, i])) for i in range(n)], dtype=np.float64
    )
    diag_signs[diag_signs == 0.0] = 1.0

    R_np = np.array(R)
    for i in range(n):
        R_np[i, :] *= diag_signs[i]
        # flip Q column sign correspondingly
        Q_cols[i] = Q_cols[i] * diag_signs[i]

    R_out = jnp.array(R_np, dtype=jnp.float64)
    Q_out = Quasimatrix(cols=Q_cols, domain=domain)
    return Q_out, R_out


def svd_quasimatrix(
    qm: Quasimatrix,
) -> tuple[Quasimatrix, jnp.ndarray, jnp.ndarray]:
    """SVD of a quasimatrix: A = U * S * V^T.

    Computes the singular value decomposition of an n-column quasimatrix A via:
    1. QR factorisation: A = Q * R  (continuous Householder)
    2. Discrete SVD of R: R = U_r * S * V^T
    3. U = Q * U_r  (quasimatrix times matrix)

    Parameters
    ----------
    qm : Quasimatrix
        The quasimatrix to decompose.

    Returns
    -------
    U : Quasimatrix
        n-column quasimatrix with L2-orthonormal columns (left singular
        functions).
    S : jnp.ndarray, shape (n,)
        Singular values in non-increasing order.
    V : jnp.ndarray, shape (n, n)
        Right singular vectors (columns are orthonormal).

    Notes
    -----
    NOT JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun/svd.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    qr_quasimatrix, Chebfun.svd
    """
    # Step 1: QR
    Q, R = qr_quasimatrix(qm)

    # Step 2: Discrete SVD of R
    R_np = np.array(R)
    U_r, S_np, Vt_np = np.linalg.svd(R_np, full_matrices=False)

    # Step 3: U = Q * U_r  (quasimatrix times matrix)
    # U[:,i] = sum_j Q[:,j] * U_r[j,i]
    n = qm.n_cols
    U_cols = []
    for i in range(n):
        col = Q.cols[0] * float(U_r[0, i])
        for j in range(1, n):
            col = col + Q.cols[j] * float(U_r[j, i])
        U_cols.append(col)

    U = Quasimatrix(cols=U_cols, domain=qm.domain)
    S = jnp.array(S_np, dtype=jnp.float64)
    V = jnp.array(Vt_np.T, dtype=jnp.float64)  # shape (n, n), columns = right sing. vecs
    return U, S, V


# ============================================================================
# Convenience: attach qr / svd to Chebfun as a "quasimatrix" factory
# ============================================================================

def chebfun_qr(cols: list[Chebfun]) -> tuple[Quasimatrix, jnp.ndarray]:
    """QR factorization of a list of Chebfun columns.

    Convenience wrapper: builds a Quasimatrix from ``cols`` and calls
    :func:`qr_quasimatrix`.

    Parameters
    ----------
    cols : list[Chebfun]
        Columns of the quasimatrix.  All must share the same single-interval
        domain.

    Returns
    -------
    Q : Quasimatrix
    R : jnp.ndarray, shape (n, n)

    See Also
    --------
    qr_quasimatrix, Chebfun.qr
    """
    if not cols:
        raise ValueError("cols must be a non-empty list of Chebfun objects.")
    domain = cols[0].domain
    qm = Quasimatrix(cols=cols, domain=domain)
    return qr_quasimatrix(qm)


def chebfun_svd(cols: list[Chebfun]) -> tuple[Quasimatrix, jnp.ndarray, jnp.ndarray]:
    """SVD of a list of Chebfun columns.

    Convenience wrapper: builds a Quasimatrix from ``cols`` and calls
    :func:`svd_quasimatrix`.

    Parameters
    ----------
    cols : list[Chebfun]
        Columns of the quasimatrix.  All must share the same single-interval
        domain.

    Returns
    -------
    U : Quasimatrix
    S : jnp.ndarray, shape (n,)
    V : jnp.ndarray, shape (n, n)

    See Also
    --------
    svd_quasimatrix, Chebfun.svd
    """
    if not cols:
        raise ValueError("cols must be a non-empty list of Chebfun objects.")
    domain = cols[0].domain
    qm = Quasimatrix(cols=cols, domain=domain)
    return svd_quasimatrix(qm)
