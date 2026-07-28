# uses-numpy: adaptive Tucker construction uses numpy for pivot selection (not JIT-safe)
"""Chebfun3 — Tucker-format approximation of 3D functions.

Represents a trivariate smooth function f(x, y, z) on a cuboid
[xa, xb] x [ya, yb] x [za, zb] as a Tucker decomposition:

    f(x, y, z) ≈ Σ_ijk  core[i, j, k] * X_i(x) * Y_j(y) * Z_k(z)

where X_i, Y_j, Z_k are univariate Chebyshev functions (Chebtech2 on
the reference interval [-1, 1]) and ``core`` is a 3D tensor.

Construction uses the Chebfun3f algorithm (three-phase Tucker construction
via alternating ACA on mode-1, 2, 3 unfoldings):

    Phase 1: Find fiber indices via alternating ACA on a coarse tensor grid.
    Phase 2: Refine fiber samples until Chebyshev coefficients decay.
    Phase 3: QR + DEIM to build factor matrices; compute Tucker core.

References
----------
[1] S. Dolgov, D. Kressner, C. Stroessner, "Functional Tucker approximation
    using Chebyshev interpolation", SIAM J. Sci. Comput., 43 (2021),
    A2190–A2210.
[2] B. Hashemi and L. N. Trefethen, "Chebfun in three dimensions",
    SIAM J. Sci. Comput., 39 (2017), C341–C363.

Translated from MATLAB Chebfun classes @chebfun3 and @chebfun3/chebfun3f.m
(commit 7574c77).
Original: Copyright 2023 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.transforms import vals2coeffs

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Affine mapping helpers
# ============================================================================


def _ref_to_phys(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map reference coordinate(s) t in [-1, 1] to physical [a, b]."""
    return 0.5 * (b - a) * t + 0.5 * (a + b)


def _phys_to_ref(x: jax.Array, a: float, b: float) -> jax.Array:
    """Map physical coordinate(s) x in [a, b] to reference [-1, 1]."""
    return (2.0 * x - (a + b)) / (b - a)


def _chebpts_phys_np(n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev-2 points on physical interval [a, b] (NumPy output)."""
    # Chebyshev-2 points on [-1,1]: -cos(k*pi/(n-1)), k=0,...,n-1
    if n == 1:
        return np.array([0.5 * (a + b)])
    k = np.arange(n)
    t = -np.cos(k * np.pi / (n - 1))
    return _ref_to_phys(t, a, b)


# ============================================================================
# Phase 1 helpers: ACA on a 2D matrix (mode unfolding)
# ============================================================================


def _cheb_gram(n: int) -> np.ndarray:
    """Gram matrix G[i,j] = int_{-1}^{1} T_i(x) T_j(x) dx.

    T_i T_j = (T_{i+j} + T_{|i-j|})/2 and int T_k = 2/(1-k^2) for even
    k (0 for odd k).
    """
    def intT(k):
        return 2.0 / (1.0 - k * k) if k % 2 == 0 else 0.0
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = 0.5 * (intT(i + j) + intT(abs(i - j)))
    return G


def _aca(
    A: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Cross Approximation with full pivoting on a 2D matrix.

    Computes a low-rank approximation A ≈ A[:,col_idx] * A[row_idx,:]
    by iteratively picking the entry of maximum absolute value in the
    residual and performing a rank-1 update.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Input matrix (mode unfolding of a tensor).
    tol : float
        Stop when max|residual| < tol.
    max_iter : int
        Maximum number of rank-1 steps.

    Returns
    -------
    Ac : np.ndarray, shape (m, r)
        Selected columns of original A (skeleton columns).
    Ar : np.ndarray, shape (n, r)
        Selected rows of original A, transposed (skeleton rows).
    At : np.ndarray, shape (r, r)
        Intersection matrix A[row_idx, col_idx].
    row_idx : np.ndarray, shape (r,), dtype int
        Selected row indices.
    col_idx : np.ndarray, shape (r,), dtype int
        Selected column indices.

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3f.m  (ACA subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford
        and The Chebfun Developers.
    """
    A_orig = A.copy()
    A = A.copy()
    row_idx = []
    col_idx = []

    for _ in range(max_iter):
        flat = int(np.argmax(np.abs(A)))
        err = np.abs(A.flat[flat])
        if err < tol:
            break
        i = flat // A.shape[1]
        j = flat % A.shape[1]
        row_idx.append(i)
        col_idx.append(j)
        # Rank-1 update
        piv = A[i, j]
        A = A - np.outer(A[:, j], A[i, :]) / piv

    if len(row_idx) == 0:
        # Zero matrix
        row_idx = [0]
        col_idx = [0]

    row_idx = np.array(row_idx, dtype=int)
    col_idx = np.array(col_idx, dtype=int)

    Ac = A_orig[:, col_idx]              # shape (m, r)
    Ar = A_orig[row_idx, :].T            # shape (n, r)
    At = A_orig[np.ix_(row_idx, col_idx)]  # shape (r, r)

    return Ac, Ar, At, row_idx, col_idx


# ============================================================================
# Phase 3 helper: DEIM interpolation points
# ============================================================================


def _deim(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Empirical Interpolation Method.

    Given a basis matrix U of shape (n, r), select r interpolation
    indices such that the submatrix U[indices, :] is well-conditioned.

    Parameters
    ----------
    U : np.ndarray, shape (n, r)
        Orthonormal (or near-orthonormal) basis matrix, typically from QR.

    Returns
    -------
    indices : np.ndarray, shape (r,), dtype int
        DEIM interpolation point indices.
    UI : np.ndarray, shape (r, r)
        Submatrix U[indices, :].

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3f.m  (DEIM subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Chaturantabut & Sorensen, "Nonlinear model reduction via
        discrete empirical interpolation", SIAM J. Sci. Comput., 2010.
    """
    r = U.shape[1]
    indices = []
    # First index: max abs in first column
    idx = int(np.argmax(np.abs(U[:, 0])))
    indices.append(idx)

    for l in range(1, r):  # noqa: E741
        # Solve U[indices, :l] c = U[indices, l] then residual = U[:,l] - U[:,:l]*c
        UI_prev = U[np.array(indices), :l]  # shape (l, l)
        rhs = U[np.array(indices), l]       # shape (l,)
        c = np.linalg.lstsq(UI_prev, rhs, rcond=None)[0]  # shape (l,)
        residual = U[:, l] - U[:, :l] @ c
        idx = int(np.argmax(np.abs(residual)))
        indices.append(idx)

    indices = np.array(indices, dtype=int)
    UI = U[indices, :]  # shape (r, r)
    return indices, UI


# ============================================================================
# Tucker core computation helpers
# ============================================================================


def _invtprod(
    X: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Compute X times_1 inv(U) times_2 inv(V) times_3 inv(W).

    For a 3-tensor X of shape (r1, r2, r3) and square matrices U (r1 x r1),
    V (r2 x r2), W (r3 x r3), compute the Tucker-mode products with inverses.

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3f.m  (invtprod subfunction)
    Chebfun commit: 7574c77
    """
    r1, r2, r3 = X.shape
    # Mode-1: inv(U) applied along axis 0
    X = np.linalg.solve(U, X.reshape(r1, r2 * r3)).reshape(r1, r2, r3)
    # Mode-2: inv(V) applied along axis 1
    X = X.transpose(1, 0, 2)  # (r2, r1, r3)
    X = np.linalg.solve(V, X.reshape(r2, r1 * r3)).reshape(r2, r1, r3)
    X = X.transpose(1, 0, 2)  # (r1, r2, r3)
    # Mode-3: inv(W) applied along axis 2
    X = X.transpose(2, 0, 1)  # (r3, r1, r2)
    X = np.linalg.solve(W, X.reshape(r3, r1 * r2)).reshape(r3, r1, r2)
    X = X.transpose(1, 2, 0)  # (r1, r2, r3)
    return X


def _tprod(
    X: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Tucker mode-product: X times_1 U times_2 V times_3 W.

    For a 3-tensor X of shape (r1, r2, r3) and matrices U (m1, r1),
    V (m2, r2), W (m3, r3), returns tensor of shape (m1, m2, m3).
    """
    r1, r2, r3 = X.shape
    m1 = U.shape[0]
    m2 = V.shape[0]
    m3 = W.shape[0]
    # Mode-1: (m1, r2, r3)
    Y = (U @ X.reshape(r1, r2 * r3)).reshape(m1, r2, r3)
    # Mode-2: (m1, m2, r3)
    Y = Y.transpose(1, 0, 2)  # (r2, m1, r3)
    Y = (V @ Y.reshape(r2, m1 * r3)).reshape(m2, m1, r3)
    Y = Y.transpose(1, 0, 2)  # (m1, m2, r3)
    # Mode-3: (m1, m2, m3)
    Y = Y.transpose(2, 0, 1)  # (r3, m1, m2)
    Y = (W @ Y.reshape(r3, m1 * m2)).reshape(m3, m1, m2)
    Y = Y.transpose(1, 2, 0)  # (m1, m2, m3)
    return Y


# ============================================================================
# Happiness check (1D resolution check on a matrix of fibers)
# ============================================================================


def _is_happy_matrix(
    M: np.ndarray,
    tol: float,
) -> bool:
    """Check if ALL columns of M (fiber values) are resolved to tolerance tol.

    For each column, compute its Chebyshev coefficients, then check the
    column-wise sum of **absolute Chebyshev coefficients** (matching the
    MATLAB Chebfun3 happinessCheck3 logic: ``sum(abs(coeffs), 2)``).
    This avoids creating non-smooth functions via |values| that would
    not have rapidly decaying Chebyshev series.

    Parameters
    ----------
    M : np.ndarray, shape (n, r)
        Each column is a 1D fiber sampled at n Chebyshev-2 points.
    tol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if the sum of absolute Chebyshev coefficients satisfies
        standard_chop (i.e., all fibers are resolved).

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3f.m  (happinessCheck3 subfunction)
    Chebfun commit: 7574c77
    """
    n, r = M.shape
    if n < 4:
        return False

    # Compute Chebyshev coefficients for each column
    all_coeffs = np.zeros((n, r))
    for j in range(r):
        v = jnp.asarray(M[:, j], dtype=jnp.float64)
        c = vals2coeffs(v)
        all_coeffs[:, j] = np.array(c)

    # Sum absolute values of coefficients across fibers
    # (matches MATLAB: UChebtech.coeffs = sum(abs(UChebtech.coeffs), 2))
    sum_abs_coeffs = np.sum(np.abs(all_coeffs), axis=1)  # shape (n,)

    # vscale is the max of the first column of M (matching MATLAB's vsclU)
    vscale = float(np.max(np.abs(M[:, 0])))
    if vscale == 0.0:
        return True

    # Use a relative tolerance referenced to the vscale
    rel_tol = max(tol / vscale, _EPS)
    cutoff = standard_chop(jnp.asarray(sum_abs_coeffs, dtype=jnp.float64), rel_tol)
    return int(cutoff) < n


# ============================================================================
# Evaluate a tensor on index sets
# ============================================================================


def _eval_tensor(
    f: Callable,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    I: np.ndarray,  # noqa: E741
    J: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Evaluate f at the tensor product of selected physical points.

    Parameters
    ----------
    f : callable
        f(xx, yy, zz) -> ndarray; must accept 3D arrays from np.meshgrid.
    x_pts, y_pts, z_pts : np.ndarray
        1D arrays of all available points in each direction.
    I, J, K : np.ndarray
        1D int arrays of index subsets in x, y, z respectively.

    Returns
    -------
    T : np.ndarray, shape (len(I), len(J), len(K))
        Tensor of function values T[i, j, k] = f(x_pts[I[i]], y_pts[J[j]], z_pts[K[k]]).
    """
    xi = x_pts[I]
    yj = y_pts[J]
    zk = z_pts[K]
    # Build ndgrid-style meshgrid (indexing='ij')
    xx, yy, zz = np.meshgrid(xi, yj, zk, indexing='ij')
    xx_j = jnp.asarray(xx, dtype=jnp.float64)
    yy_j = jnp.asarray(yy, dtype=jnp.float64)
    zz_j = jnp.asarray(zz, dtype=jnp.float64)
    T = np.array(f(xx_j, yy_j, zz_j), dtype=np.float64)
    return T


# ============================================================================
# Grid refinement
# ============================================================================


def _grid_interp2(vals: jax.Array, dom2) -> Callable:
    """Spectral interpolant of values on a 2D tensor Chebyshev grid.

    ``vals[i, j]`` are function values at the (a_i, b_j) tensor product of
    2nd-kind Chebyshev points on ``dom2 = (a0, a1, b0, b1)``.  Returns an
    evaluator ``ev(a, b)`` (broadcastable arrays) computing the bivariate
    Chebyshev interpolant, used to rebuild reduced grids as Chebfun2s
    (MATLAB does this via the chebfun2 values-matrix constructor).
    """
    coeffs = vals2coeffs(vals2coeffs(vals).T).T  # (na, nb) Chebyshev coeffs
    na, nb = coeffs.shape
    a0, a1, b0, b1 = float(dom2[0]), float(dom2[1]), float(dom2[2]), \
        float(dom2[3])

    def ev(a, b):
        ta = jnp.clip((2.0 * jnp.asarray(a, dtype=jnp.float64) - a0 - a1)
                      / (a1 - a0), -1.0, 1.0)
        tb = jnp.clip((2.0 * jnp.asarray(b, dtype=jnp.float64) - b0 - b1)
                      / (b1 - b0), -1.0, 1.0)
        Ta = jnp.cos(jnp.arange(na) * jnp.arccos(ta)[..., None])  # (..., na)
        Tb = jnp.cos(jnp.arange(nb) * jnp.arccos(tb)[..., None])  # (..., nb)
        return jnp.einsum("...a,ab,...b->...", Ta, coeffs, Tb)

    return ev


def _reffun(n: int) -> int:
    """Next Chebyshev-2 grid size via the chebfun3f refinement rule.

    next = floor(sqrt(2)^(floor(2*log2(n)) + 1)) + 1
    """
    if n < 2:
        return 9
    logn = np.floor(2.0 * np.log2(n))
    return int(np.floor(np.sqrt(2) ** (logn + 1))) + 1


# ============================================================================
# Main class
# ============================================================================


class Chebfun3(eqx.Module):
    """Tucker-format approximation of a trivariate function on a cuboid.

    Represents f(x, y, z) on [xa, xb] x [ya, yb] x [za, zb] as:

        f(x, y, z) ≈ Σ_ijk  core[i, j, k] * X_i(x) * Y_j(y) * Z_k(z)

    where X_i are column functions (of x), Y_j are row functions (of y),
    Z_k are tube functions (of z), and ``core`` is a 3D Tucker core tensor.

    Attributes
    ----------
    cols : list of Chebtech2
        Column factor functions X_i(x), each a Chebtech2 on [-1, 1].
        Physical domain is [xa, xb] (mapped from ``domain``).
    rows : list of Chebtech2
        Row factor functions Y_j(y), each a Chebtech2 on [-1, 1].
        Physical domain is [ya, yb].
    tubes : list of Chebtech2
        Tube factor functions Z_k(z), each a Chebtech2 on [-1, 1].
        Physical domain is [za, zb].
    core : jax.Array, shape (rx, ry, rz)
        Tucker core tensor.
    domain : tuple (xa, xb, ya, yb, za, zb)
        Physical domain. Static field.

    Notes
    -----
    Construction is NOT JIT-safe (Python adaptive loop).
    Evaluation IS JIT-safe, grad-safe, and vmap-safe.

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3.m, @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: S. Dolgov, D. Kressner, C. Stroessner, "Functional Tucker
        approximation using Chebyshev interpolation", SIAM J. Sci. Comput.,
        43 (2021), A2190–A2210.

    See Also
    --------
    chebfun3, Chebtech2
    """

    @classmethod
    def empty(cls) -> "Chebfun3":
        """The empty Chebfun3 (MATLAB chebfun3()): no data; isempty() is
        True and operations on it are undefined.

        Provenance
        ----------
        MATLAB source : @chebfun3/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Chebfun3 (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @chebfun3/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    cols: list    # list of Chebtech2 in x
    rows: list    # list of Chebtech2 in y
    tubes: list   # list of Chebtech2 in z
    core: jax.Array  # Tucker core tensor, shape (rx, ry, rz)
    domain: tuple = eqx.field(static=True)  # (xa, xb, ya, yb, za, zb)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        domain: tuple[float, float, float, float, float, float] = (
            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        ),
        tol: float = _EPS,
        max_rank: int = 128,
        min_samples: int = 9,
        _restarts: int = 0,
        _r_init: tuple = (6, 6, 6),
        _n_init: tuple | None = None,
    ) -> "Chebfun3":
        """Construct a Chebfun3 from a callable f(x, y, z).

        Uses the Chebfun3f algorithm (three-phase Tucker construction):

        Phase 1 — Identify fiber indices on a coarse grid via alternating
            ACA on mode-1, 2, 3 unfoldings of the evaluation tensor.

        Phase 2 — Resolve the fiber samples adaptively by increasing the
            1D grid size until Chebyshev coefficients fall below ``tol``.

        Phase 3 — Compute QR factorizations of the fiber matrices, apply
            DEIM to find interpolation points, build the Tucker core and
            convert columns to Chebtech2 objects.

        Parameters
        ----------
        f : callable
            f(xx, yy, zz) accepting 3D ndgrid-style arrays.  Must be
            fully vectorised: ``xx[i,j,k]``, ``yy[i,j,k]``, ``zz[i,j,k]``
            correspond to x, y, z coordinates.
        domain : 6-tuple of floats, optional
            (xa, xb, ya, yb, za, zb).  Default is (-1, 1, -1, 1, -1, 1).
        tol : float, optional
            Target relative tolerance.  Default is machine epsilon (~2.2e-16).
        max_rank : int, optional
            Maximum rank in each mode.  Default 128.
        min_samples : int, optional
            Minimum number of grid points per direction in Phase 1.
            Default 9.

        Returns
        -------
        Chebfun3
            A Tucker-format approximation.

        Raises
        ------
        ValueError
            If f returns Inf or NaN on the initial grid.

        Warns
        -----
        RuntimeWarning
            If construction did not converge.

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun3/chebfun3f.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2023 by The University of Oxford
            and The Chebfun Developers.
        """
        xa, xb = float(domain[0]), float(domain[1])
        ya, yb = float(domain[2]), float(domain[3])
        za, zb = float(domain[4]), float(domain[5])
        dom = (xa, xb, ya, yb, za, zb)

        # Complex-valued functions: the Tucker constructor real-casts,
        # so build re/im separately and recombine exactly (Fable 5 audit
        # -- the imaginary part was previously silently dropped, same
        # bug as Chebfun2).
        xp_ = jnp.asarray([0.5 * (xa + xb) + 0.25 * (xb - xa)])
        yp_ = jnp.asarray([0.5 * (ya + yb) + 0.25 * (yb - ya)])
        zp_ = jnp.asarray([0.5 * (za + zb) + 0.25 * (zb - za)])
        if jnp.iscomplexobj(jnp.asarray(f(xp_, yp_, zp_))):
            kw = dict(domain=domain, tol=tol, max_rank=max_rank,
                      min_samples=min_samples)
            fre = cls.from_function(
                lambda x, y, z: jnp.real(f(x, y, z)), **kw)
            fim = cls.from_function(
                lambda x, y, z: jnp.imag(f(x, y, z)), **kw)
            return fre + fim * 1j

        # ----------------------------------------------------------------
        # Helper: sample f on full tensor grid (n1 x n2 x n3)
        # ----------------------------------------------------------------
        def _full_tensor(n1: int, n2: int, n3: int) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]:
            """Return (x_pts, y_pts, z_pts, vals) on Cheb-2 grid."""
            xp = _chebpts_phys_np(n1, xa, xb)
            yp = _chebpts_phys_np(n2, ya, yb)
            zp = _chebpts_phys_np(n3, za, zb)
            T = _eval_tensor(f, xp, yp, zp,
                             np.arange(n1), np.arange(n2), np.arange(n3))
            return xp, yp, zp, T

        # ----------------------------------------------------------------
        # Helper: compute getTol-style absolute tolerance from a matrix M
        # ----------------------------------------------------------------
        def _get_abs_tol(M: np.ndarray, dom_diff: float, old_tol: float) -> float:
            """Compute absolute tolerance matching MATLAB's getTol."""
            n = M.shape[0]
            rel_tol = 2.0 * n ** (4.0 / 5.0) * tol
            vscale = float(np.max(np.abs(M))) if M.size > 0 else 0.0
            if n > 1:
                k = np.arange(n)
                pts = -np.cos(k * np.pi / (n - 1))
                diffs = np.diff(M, axis=0)
                dpts = np.diff(pts)[:, None]
                grad_norms = float(np.max(np.abs(diffs / dpts)))
            else:
                grad_norms = 0.0
            abs_t = max(dom_diff * grad_norms, vscale) * rel_tol
            abs_t = max(abs_t, old_tol, tol)
            return abs_t

        # ================================================================
        # PHASE 1: Alternating ACA to find fiber indices
        # ================================================================

        if _n_init is not None:
            n = [int(v) for v in _n_init]
        else:
            n = [max(min_samples, 9)] * 3
        # Initial ranks for random initialization
        r = [int(v) for v in _r_init]
        abs_tol_running = tol

        xp = _chebpts_phys_np(n[0], xa, xb)
        yp = _chebpts_phys_np(n[1], ya, yb)
        zp = _chebpts_phys_np(n[2], za, zb)

        # Initialize random fiber indices (spread across interval)
        rng = np.random.default_rng(16051821 + _restarts)

        def _init_indices(ri: int, ni: int) -> np.ndarray:
            """Draw ri indices spread uniformly in [0, ni)."""
            box = max(1, ni // ri)
            idx = []
            for q in range(ri):
                lo = q * box
                hi = min(lo + box, ni) - 1
                idx.append(int(rng.integers(lo, max(lo + 1, hi + 1))))
            return np.array(sorted(set(idx[:ri])), dtype=int)

        J = _init_indices(r[1], n[1])
        K = _init_indices(r[2], n[2])

        # Store the fiber indices found in each ACA round
        I_idx = np.arange(n[0], dtype=int)
        J_idx = np.arange(n[1], dtype=int)
        K_idx = np.arange(n[2], dtype=int)

        happy_phase1 = False
        phase1_iters = 0

        while not happy_phase1 and phase1_iters < 20:
            phase1_iters += 1
            need_refine = False

            for _inner in range(2):
                # ---- ACA on mode-1 unfolding ----
                T1 = _eval_tensor(f, xp, yp, zp,
                                  np.arange(n[0]), J, K)
                vscale_T1 = float(np.max(np.abs(T1))) if T1.size > 0 else 0.0
                if not np.isfinite(vscale_T1):
                    raise ValueError(
                        "Chebfun3.from_function: function returned Inf or NaN "
                        f"on the grid over domain ({xa},{xb})x({ya},{yb})x({za},{zb})."
                    )

                M1 = T1.reshape(n[0], len(J) * len(K))
                abs_tol_running = _get_abs_tol(M1, xb - xa, abs_tol_running)
                _, _, _, I_idx, col1 = _aca(M1, abs_tol_running, max_rank)
                r[0] = len(I_idx)
                # Save which (j,k) pairs the selected columns correspond to
                J_from1 = J[col1 // len(K)]
                K_from1 = K[col1 % len(K)]

                # ---- ACA on mode-2 unfolding ----
                T2 = _eval_tensor(f, xp, yp, zp,
                                  I_idx, np.arange(n[1]), K)
                # mode-2 unfolding: permute to (n2, n1*n3)
                M2 = T2.transpose(1, 0, 2).reshape(n[1], len(I_idx) * len(K))
                abs_tol_running = _get_abs_tol(M2, yb - ya, abs_tol_running)
                _, _, _, J_idx, col2 = _aca(M2, abs_tol_running, max_rank)
                r[1] = len(J_idx)
                I_from2 = I_idx[col2 // len(K)]
                K_from2 = K[col2 % len(K)]

                # ---- ACA on mode-3 unfolding ----
                T3 = _eval_tensor(f, xp, yp, zp,
                                  I_idx, J_idx, np.arange(n[2]))
                # mode-3 unfolding: permute to (n3, n1*n2)
                M3 = T3.transpose(2, 0, 1).reshape(n[2], len(I_idx) * len(J_idx))
                abs_tol_running = _get_abs_tol(M3, zb - za, abs_tol_running)
                _, _, _, K_idx, col3 = _aca(M3, abs_tol_running, max_rank)
                r[2] = len(K_idx)
                I_from3 = I_idx[col3 // len(J_idx)]
                J_from3 = J_idx[col3 % len(J_idx)]

                # Update J, K for next ACA-1
                J = J_idx
                K = K_idx

                # Check if ranks are small enough relative to grid
                factor = 2.0 * np.sqrt(2.0)
                ref0 = r[0] * factor > n[0]
                ref1 = r[1] * factor > n[1]
                ref2 = r[2] * factor > n[2]
                if ref0 or ref1 or ref2:
                    if ref0:
                        n[0] = _reffun(n[0])
                    if ref1:
                        n[1] = _reffun(n[1])
                    if ref2:
                        n[2] = _reffun(n[2])
                    xp = _chebpts_phys_np(n[0], xa, xb)
                    yp = _chebpts_phys_np(n[1], ya, yb)
                    zp = _chebpts_phys_np(n[2], za, zb)
                    J = _init_indices(max(r[1], 3), n[1])
                    K = _init_indices(max(r[2], 3), n[2])
                    need_refine = True
                    break
                elif min(r) < 2:
                    break

            if not need_refine:
                happy_phase1 = True

        # Handle rank-zero function
        if r[0] == 0 or r[1] == 0 or r[2] == 0:
            zero = Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))
            return cls(
                cols=[zero],
                rows=[zero],
                tubes=[zero],
                core=jnp.zeros((1, 1, 1), dtype=jnp.float64),
                domain=dom,
            )

        # Physical pivot locations for Phase 2 (fiber sampling)
        # Uf: mode-1 fibers at selected (J, K) pairs from ACA-1
        # We keep the pivot columns from ACA-1, ACA-2, ACA-3
        # The number of fibers = r[i] from each mode
        r1, r2, r3 = r

        # Build fiber matrices at the coarse-grid resolution using the pivot
        # fiber indices found in Phase 1.
        # Uf[i, j] = f(x_pts[i], y_pts[J_from1[j]], z_pts[K_from1[j]])
        # stored as n1 x r1 matrix
        Uf_coarse = np.zeros((n[0], r1))
        for j in range(r1):
            T_col = _eval_tensor(f, xp, yp, zp,
                                 np.arange(n[0]),
                                 np.array([J_from1[j]]),
                                 np.array([K_from1[j]]))
            Uf_coarse[:, j] = T_col[:, 0, 0]

        Vf_coarse = np.zeros((n[1], r2))
        for j in range(r2):
            T_col = _eval_tensor(f, xp, yp, zp,
                                 np.array([I_from2[j]]),
                                 np.arange(n[1]),
                                 np.array([K_from2[j]]))
            Vf_coarse[:, j] = T_col[0, :, 0]

        Wf_coarse = np.zeros((n[2], r3))
        for j in range(r3):
            T_col = _eval_tensor(f, xp, yp, zp,
                                 np.array([I_from3[j]]),
                                 np.array([J_from3[j]]),
                                 np.arange(n[2]))
            Wf_coarse[:, j] = T_col[0, 0, :]

        # ================================================================
        # PHASE 2: Refine fiber grids until Chebyshev coefficients decay
        # ================================================================

        m = list(n)
        xp_f = _chebpts_phys_np(m[0], xa, xb)
        yp_f = _chebpts_phys_np(m[1], ya, yb)
        zp_f = _chebpts_phys_np(m[2], za, zb)

        Uf = Uf_coarse.copy()
        Vf = Vf_coarse.copy()
        Wf = Wf_coarse.copy()

        # Refine if initial grid is already not enough.
        # Phase 2 uses Chebtech2 doubling: 2*m - 1 (preserves nesting).
        res_u = _is_happy_matrix(Uf, abs_tol_running)
        res_v = _is_happy_matrix(Vf, abs_tol_running)
        res_w = _is_happy_matrix(Wf, abs_tol_running)

        if not res_u:
            m[0] = 2 * m[0] - 1
        if not res_v:
            m[1] = 2 * m[1] - 1
        if not res_w:
            m[2] = 2 * m[2] - 1

        # Physical pivot locations (for Phase 2 fiber re-sampling)
        xp[I_idx] if len(I_idx) > 0 else xp[[0]]  # not used for x-fibers
        y_piv_u = yp[J_from1]  # y-locations used when sampling x-fibers
        z_piv_u = zp[K_from1]  # z-locations used when sampling x-fibers

        x_piv_v = xp[I_from2]  # x-locations used when sampling y-fibers
        z_piv_v = zp[K_from2]  # z-locations used when sampling y-fibers

        x_piv_w = xp[I_from3]  # x-locations used when sampling z-fibers
        y_piv_w = yp[J_from3]  # y-locations used when sampling z-fibers

        failure = False
        max_samples = 2**14 + 1

        while not (res_u and res_v and res_w) and not failure:
            xp_f = _chebpts_phys_np(m[0], xa, xb)
            yp_f = _chebpts_phys_np(m[1], ya, yb)
            zp_f = _chebpts_phys_np(m[2], za, zb)

            if not res_u:
                if m[0] > max_samples:
                    warnings.warn(
                        "Chebfun3.from_function: x-fibers not resolved "
                        f"with {m[0]} points. Stopping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failure = True
                    break
                Uf_new = np.zeros((m[0], r1))
                for j in range(r1):
                    xx_j = jnp.full(m[0], float(y_piv_u[j]), dtype=jnp.float64)
                    zz_j = jnp.full(m[0], float(z_piv_u[j]), dtype=jnp.float64)
                    xx_phys = jnp.asarray(xp_f, dtype=jnp.float64)
                    vals = np.array(
                        f(xx_phys, xx_j, zz_j), dtype=np.float64
                    )
                    Uf_new[:, j] = vals
                Uf = Uf_new

            if not res_v:
                if m[1] > max_samples:
                    warnings.warn(
                        "Chebfun3.from_function: y-fibers not resolved "
                        f"with {m[1]} points. Stopping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failure = True
                    break
                Vf_new = np.zeros((m[1], r2))
                for j in range(r2):
                    xx_j = jnp.full(m[1], float(x_piv_v[j]), dtype=jnp.float64)
                    zz_j = jnp.full(m[1], float(z_piv_v[j]), dtype=jnp.float64)
                    yy_phys = jnp.asarray(yp_f, dtype=jnp.float64)
                    vals = np.array(
                        f(xx_j, yy_phys, zz_j), dtype=np.float64
                    )
                    Vf_new[:, j] = vals
                Vf = Vf_new

            if not res_w:
                if m[2] > max_samples:
                    warnings.warn(
                        "Chebfun3.from_function: z-fibers not resolved "
                        f"with {m[2]} points. Stopping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failure = True
                    break
                Wf_new = np.zeros((m[2], r3))
                for j in range(r3):
                    xx_j = jnp.full(m[2], float(x_piv_w[j]), dtype=jnp.float64)
                    yy_j = jnp.full(m[2], float(y_piv_w[j]), dtype=jnp.float64)
                    zz_phys = jnp.asarray(zp_f, dtype=jnp.float64)
                    vals = np.array(
                        f(xx_j, yy_j, zz_phys), dtype=np.float64
                    )
                    Wf_new[:, j] = vals
                Wf = Wf_new

            abs_tol_running = _get_abs_tol(Uf, xb - xa, abs_tol_running)
            abs_tol_running = _get_abs_tol(Vf, yb - ya, abs_tol_running)
            abs_tol_running = _get_abs_tol(Wf, zb - za, abs_tol_running)

            res_u = _is_happy_matrix(Uf, abs_tol_running)
            res_v = _is_happy_matrix(Vf, abs_tol_running)
            res_w = _is_happy_matrix(Wf, abs_tol_running)

            if not res_u:
                m[0] = 2 * m[0] - 1  # Chebtech2 doubling: 2*m - 1
            if not res_v:
                m[1] = 2 * m[1] - 1
            if not res_w:
                m[2] = 2 * m[2] - 1

        # ================================================================
        # PHASE 3: QR + DEIM, evaluate core, build Chebtech2 objects
        # ================================================================

        # QR decompositions of fiber matrices
        QU, RU = np.linalg.qr(Uf, mode='reduced')  # (m0, r1)
        QV, RV = np.linalg.qr(Vf, mode='reduced')  # (m1, r2)
        QW, RW = np.linalg.qr(Wf, mode='reduced')  # (m2, r3)

        # DEIM interpolation points
        I_deim, QUI = _deim(QU)  # I_deim: shape (r1,); QUI: (r1, r1)
        J_deim, QVJ = _deim(QV)  # J_deim: shape (r2,); QVJ: (r2, r2)
        K_deim, QWK = _deim(QW)  # K_deim: shape (r3,); QWK: (r3, r3)

        # Evaluate f at the DEIM interpolation points
        xp_deim = xp_f[I_deim]
        yp_deim = yp_f[J_deim]
        zp_deim = zp_f[K_deim]

        # Evaluate f on the (r1 x r2 x r3) grid of DEIM points
        xx, yy, zz = np.meshgrid(xp_deim, yp_deim, zp_deim, indexing='ij')
        xx_j = jnp.asarray(xx, dtype=jnp.float64)
        yy_j = jnp.asarray(yy, dtype=jnp.float64)
        zz_j = jnp.asarray(zz, dtype=jnp.float64)
        T_deim = np.array(f(xx_j, yy_j, zz_j), dtype=np.float64)

        # Tucker core: core = T_deim x_1 inv(QUI) x_2 inv(QVJ) x_3 inv(QWK)
        core_np = _invtprod(T_deim, QUI, QVJ, QWK)

        # Scaling: ensure factor matrices have decaying coefficients
        # col_scaling[i] = max over (j,k) of |core[i,j,k]|
        eps_small = np.finfo(np.float64).tiny
        col_scaling = np.maximum(np.max(np.abs(core_np), axis=(1, 2)), eps_small)
        row_scaling = np.maximum(np.max(np.abs(core_np), axis=(0, 2)), eps_small)
        tube_scaling = np.maximum(np.max(np.abs(core_np), axis=(0, 1)), eps_small)

        # Scale factor matrices
        QU_scaled = QU * col_scaling[np.newaxis, :]    # (m0, r1)
        QV_scaled = QV * row_scaling[np.newaxis, :]    # (m1, r2)
        QW_scaled = QW * tube_scaling[np.newaxis, :]   # (m2, r3)

        # Rescale core accordingly
        core_scaled = _invtprod(
            core_np,
            np.diag(col_scaling),
            np.diag(row_scaling),
            np.diag(tube_scaling),
        )

        # ----------------------------------------------------------------
        # Build Chebtech2 objects for each fiber
        # ----------------------------------------------------------------
        cols_list = []
        for i in range(r1):
            v = jnp.asarray(QU_scaled[:, i], dtype=jnp.float64)
            c = vals2coeffs(v)
            vscale = float(jnp.max(jnp.abs(v)))
            if vscale > 0:
                rel_tol = max(tol, _EPS)
                cutoff = standard_chop(c, rel_tol)
                c = c[:cutoff]
            cols_list.append(Chebtech2.from_coeffs(c))

        rows_list = []
        for i in range(r2):
            v = jnp.asarray(QV_scaled[:, i], dtype=jnp.float64)
            c = vals2coeffs(v)
            vscale = float(jnp.max(jnp.abs(v)))
            if vscale > 0:
                rel_tol = max(tol, _EPS)
                cutoff = standard_chop(c, rel_tol)
                c = c[:cutoff]
            rows_list.append(Chebtech2.from_coeffs(c))

        tubes_list = []
        for i in range(r3):
            v = jnp.asarray(QW_scaled[:, i], dtype=jnp.float64)
            c = vals2coeffs(v)
            vscale = float(jnp.max(jnp.abs(v)))
            if vscale > 0:
                rel_tol = max(tol, _EPS)
                cutoff = standard_chop(c, rel_tol)
                c = c[:cutoff]
            tubes_list.append(Chebtech2.from_coeffs(c))

        core_jax = jnp.asarray(core_scaled, dtype=jnp.float64)

        result = cls(
            cols=cols_list,
            rows=rows_list,
            tubes=tubes_list,
            core=core_jax,
            domain=dom,
        )

        # ------------------------------------------------------------
        # Sample test + restart (MATLAB chebfun3f outer "while ~happy"
        # loop).  Without this, functions of the form g(y)*h(z) (rank 1
        # in one variable) can lock the alternating ACA into a rank
        # underestimate: the discoverable rank in each mode is capped
        # by the number of candidate fibers in the others.  MATLAB
        # detects the inaccuracy at off-grid Halton points and restarts
        # with doubled candidate ranks.  (Fable 5 audit.)
        # ------------------------------------------------------------
        def _halton_1d(count: int, base: int) -> np.ndarray:
            out = np.zeros(count)
            for idx in range(count):
                frac, denom, q = 0.0, 1.0, idx + 1
                while q > 0:
                    denom /= base
                    q, rem = divmod(q, base)
                    frac += rem * denom
                out[idx] = frac
            return out

        n_test = 30
        x_t = xa + (xb - xa) * _halton_1d(n_test, 2)
        y_t = ya + (yb - ya) * _halton_1d(n_test, 3)
        z_t = za + (zb - za) * _halton_1d(n_test, 5)
        v_op = np.asarray(f(jnp.asarray(x_t), jnp.asarray(y_t),
                            jnp.asarray(z_t)))
        # Evaluate the Tucker representation in plain numpy: routing
        # this through the jitted __call__ makes XLA trace/compile a
        # graph unrolled over all 3*rank coefficient arrays, which for
        # high-rank results costs minutes and tens of GB (observed:
        # 295 s / 22 GB for rank 33 vs 0.02 s here).
        from numpy.polynomial import chebyshev as _Cnp
        sx = 2.0 * (x_t - xa) / (xb - xa) - 1.0
        sy = 2.0 * (y_t - ya) / (yb - ya) - 1.0
        sz = 2.0 * (z_t - za) / (zb - za) - 1.0
        Cv = np.stack([_Cnp.chebval(sx, np.asarray(c.coeffs))
                       for c in cols_list])
        Rv = np.stack([_Cnp.chebval(sy, np.asarray(c.coeffs))
                       for c in rows_list])
        Tv = np.stack([_Cnp.chebval(sz, np.asarray(c.coeffs))
                       for c in tubes_list])
        v_fun = np.einsum("ijk,ip,jp,kp->p",
                          np.asarray(core_scaled), Cv, Rv, Tv)
        sample_err = float(np.max(np.abs(v_op - v_fun)))
        if sample_err <= 10.0 * abs_tol_running:
            return result
        max_restarts = 10
        if _restarts + 1 >= max_restarts:
            warnings.warn(
                "Chebfun3.from_function: max number of restarts "
                f"reached (sample-test error {sample_err:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return result

        # Increase n (MATLAB restart formula)
        n_new = tuple(
            int(np.floor(np.sqrt(2.0) **
                         (np.floor(2 * np.log2(ni)) + 1))) + 1
            for ni in n
        )
        # Ensure r is large enough for (1, r, r)-type functions
        r_new = list(r)
        if r[0] > 1 or r[1] > 1 or r[2] > 1:
            if r[0] < 3:
                r_new[1] = max(6, 2 * r[1])
                r_new[2] = max(6, 2 * r[2])
            elif r[1] < 2:
                r_new[0] = max(6, 2 * r[0])
                r_new[2] = max(6, 2 * r[2])
            elif r[2] < 2:
                r_new[0] = max(6, 2 * r[0])
                r_new[1] = max(6, 2 * r[1])
        r_new = tuple(max(v, 3) for v in r_new)
        return cls.from_function(
            f, domain=domain, tol=tol, max_rank=max_rank,
            min_samples=min_samples, _restarts=_restarts + 1,
            _r_init=r_new, _n_init=n_new,
        )

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        """Evaluate f(x, y, z) at point(s).

        Computes:
            Σ_ijk  core[i, j, k] * X_i(tx) * Y_j(ty) * Z_k(tz)

        where tx, ty, tz are the reference-interval images of x, y, z.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb].  Must broadcast with x.
        z : jax.Array, scalar or shape (m,)
            z-coordinates in [za, zb].  Must broadcast with x and y.

        Returns
        -------
        jax.Array, same shape as broadcast(x, y, z)
            Approximated function values.

        Notes
        -----
        JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebfun3/feval.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        xa, xb, ya, yb, za, zb = self.domain
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        z = jnp.asarray(z, dtype=jnp.float64)

        # Map to reference interval [-1, 1]
        tx = _phys_to_ref(x, xa, xb)
        ty = _phys_to_ref(y, ya, yb)
        tz = _phys_to_ref(z, za, zb)

        # Evaluate each fiber
        # xi[i] = X_i(tx),  shape: broadcast shape of tx
        # yj[j] = Y_j(ty),  zk[k] = Z_k(tz)
        # Result = Σ_ijk core[i,j,k] * xi[i] * yj[j] * zk[k]

        # Broadcast x, y, z to a common shape
        bcast_shape = jnp.broadcast_shapes(
            jnp.shape(tx), jnp.shape(ty), jnp.shape(tz)
        )
        tx = jnp.broadcast_to(tx, bcast_shape)
        ty = jnp.broadcast_to(ty, bcast_shape)
        tz = jnp.broadcast_to(tz, bcast_shape)

        # Evaluate each factor list once and contract with the core.
        # (The previous triple nested loop emitted r1*r2*r3 graph
        # nodes with redundant fiber evaluations -- at rank 33 that is
        # ~36k Clenshaw subgraphs, minutes of XLA compile time, and
        # >20 GB of compile memory.  This form is r1+r2+r3 evaluations
        # plus one einsum.)
        xi = jnp.stack([c(tx) for c in self.cols])      # (r1, ...)
        yj = jnp.stack([r(ty) for r in self.rows])      # (r2, ...)
        zk = jnp.stack([t(tz) for t in self.tubes])     # (r3, ...)
        return jnp.einsum("ijk,i...,j...,k...->...",
                          self.core, xi, yj, zk)

    # ------------------------------------------------------------------
    # Triple integral
    # ------------------------------------------------------------------

    def mean(self, dim: int = 1):
        """Average over one variable (MATLAB mean(f, dim)).

        Provenance
        ----------
        MATLAB source : @chebfun3/mean.m
        Chebfun commit: 7574c77
        """
        d = self.domain
        i = dim - 1
        return self.sum(dim) * (1.0 / (d[2 * i + 1] - d[2 * i]))

    def mean2(self, dims: tuple = (1, 2)):
        """Average over two variables (MATLAB mean2).

        Provenance
        ----------
        MATLAB source : @chebfun3/mean2.m
        Chebfun commit: 7574c77
        """
        d = self.domain
        vol = 1.0
        for dd in dims:
            vol *= d[2 * (dd - 1) + 1] - d[2 * (dd - 1)]
        return self.sum2(dims) * (1.0 / vol)

    def hosvd(self):
        r"""Higher-order SVD (MATLAB hosvd): returns
        ``(sv, g)`` where ``sv`` is a list of the three mode-k singular
        value vectors (continuous L2 sense) and ``g`` is an equivalent
        Chebfun3 whose factors are L2-orthonormal and whose core is
        all-orthogonal.

        Provenance
        ----------
        MATLAB source : @chebfun3/hosvd.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        d = self.domain
        scales = [(d[1] - d[0]) / 2.0, (d[3] - d[2]) / 2.0,
                  (d[5] - d[4]) / 2.0]
        factor_lists = [self.cols, self.rows, self.tubes]

        # Coefficient matrices (padded) and L2 Gram factors per mode
        Cmats, Rs = [], []
        for mode in range(3):
            fl = factor_lists[mode]
            nmax = max(len(_np.asarray(t.coeffs)) for t in fl)
            C = _np.zeros((nmax, len(fl)))
            for i, t in enumerate(fl):
                ci = _np.asarray(t.coeffs)
                C[: len(ci), i] = ci
            # Gram of T_j on [-1,1]: <T_i,T_j> = int T_i T_j dx
            W = _cheb_gram(nmax) * scales[mode]
            G = C.T @ W @ C
            # Cholesky with symmetrization safeguard
            G = 0.5 * (G + G.T)
            jitter = 1e-15 * max(_np.max(_np.abs(G)), 1.0)
            R = _np.linalg.cholesky(
                G + jitter * _np.eye(G.shape[0])).T
            Cmats.append(C)
            Rs.append(R)

        core = _np.asarray(self.core)
        # core' = core x1 R1 x2 R2 x3 R3
        core1 = _np.einsum("pi,ijk->pjk", Rs[0], core)
        core1 = _np.einsum("qj,pjk->pqk", Rs[1], core1)
        core1 = _np.einsum("rk,pqk->pqr", Rs[2], core1)

        sv, Us = [], []
        for mode in range(3):
            M = _np.moveaxis(core1, mode, 0).reshape(
                core1.shape[mode], -1)
            U, S, _ = _np.linalg.svd(M, full_matrices=False)
            sv.append(jnp.asarray(S, dtype=jnp.float64))
            Us.append(U)
        # all-orthogonal core
        core2 = _np.einsum("pi,ijk->pjk", Us[0].T, core1)
        core2 = _np.einsum("qj,pjk->pqk", Us[1].T, core2)
        core2 = _np.einsum("rk,pqk->pqr", Us[2].T, core2)

        # new factor functions: A @ inv(R) @ U in coefficient space
        new_factors = []
        for mode in range(3):
            B = Cmats[mode] @ _np.linalg.solve(Rs[mode], Us[mode])
            new_factors.append([
                Chebtech2.from_coeffs(
                    jnp.asarray(B[:, i], dtype=jnp.float64))
                for i in range(B.shape[1])])
        g = Chebfun3(
            cols=new_factors[0], rows=new_factors[1],
            tubes=new_factors[2],
            core=jnp.asarray(core2, dtype=jnp.float64),
            domain=d)
        return sv, g

    def _compress(self, tol: float | None = None) -> "Chebfun3":
        """Recompress the Tucker representation via truncated HOSVD.

        Drops the trailing mode-k directions whose singular values fall
        below ``tol`` relative to the dominant one, restoring the minimal
        rank after exact-but-rank-inflating operations (block-diagonal
        plus).  Returns ``self`` unchanged when nothing truncates.

        Provenance
        ----------
        MATLAB source : @chebfun3/plus.m (the constructor recompression
        its active path performs), via @chebfun3/hosvd.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        # hosvd is real-valued (float64 factors + real core); compressing
        # a complex representation through it would silently discard the
        # imaginary part.  Keep complex sums in their exact block-diagonal
        # form instead.
        if jnp.iscomplexobj(self.core) or any(
                jnp.iscomplexobj(c.coeffs)
                for c in list(self.cols) + list(self.rows)
                + list(self.tubes)):
            return self

        sv, g = self.hosvd()
        if tol is None:
            tol = 10.0 * _EPS
        smax = max(float(_np.max(_np.asarray(s))) for s in sv)
        if smax == 0.0:
            return self
        keep = []
        for s in sv:
            sn = _np.asarray(s)
            keep.append(max(1, int(_np.sum(sn > tol * smax))))
        if tuple(keep) == tuple(self.core.shape):
            return self
        k0, k1, k2 = keep
        return Chebfun3(
            cols=list(g.cols[:k0]), rows=list(g.rows[:k1]),
            tubes=list(g.tubes[:k2]),
            core=g.core[:k0, :k1, :k2],
            domain=self.domain)

    @eqx.filter_jit
    def sum3(self) -> jax.Array:
        """Definite triple integral over the domain.

        Computes:
            ∫∫∫ f(x,y,z) dx dy dz

        using the Tucker structure:
            Σ_ijk  core[i,j,k] * (∫ X_i dx) * (∫ Y_j dy) * (∫ Z_k dz)

        Each 1D integral uses Chebyshev moments (exact for polynomials).
        Physical-domain integrals are obtained by scaling with half-widths.

        Returns
        -------
        jax.Array, scalar
            The triple integral.

        Notes
        -----
        JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun3/sum3.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        xa, xb, ya, yb, za, zb = self.domain
        # Scale factors: ∫_a^b f(x) dx = (b-a)/2 * ∫_{-1}^{1} f(t) dt
        sx = 0.5 * (xb - xa)
        sy = 0.5 * (yb - ya)
        sz = 0.5 * (zb - za)

        # Integral of each fiber over [-1, 1]
        ix = jnp.array([col.sum() for col in self.cols], dtype=jnp.float64)
        iy = jnp.array([row.sum() for row in self.rows], dtype=jnp.float64)
        iz = jnp.array([tube.sum() for tube in self.tubes], dtype=jnp.float64)

        # sum3 = Σ_ijk core[i,j,k] * ix[i] * iy[j] * iz[k]
        # = core x_1 ix x_2 iy x_3 iz  (Tucker triple contraction)
        result = jnp.einsum('ijk,i,j,k->', self.core, ix, iy, iz)
        return result * sx * sy * sz

    def diff(self, dim: int = 1, k: int = 1) -> "Chebfun3":
        """k-th partial derivative in the Cartesian direction dim.

        The Tucker structure makes this a per-factor operation: d/dx
        differentiates only the column fibers (with the chain-rule
        scaling for the physical interval), leaving the core and the
        other factors untouched. dims 1, 2, 3 = x, y, z.

        Provenance
        ----------
        MATLAB source : @chebfun3/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        xa, xb, ya, yb, za, zb = self.domain
        if dim == 1:
            scale = (2.0 / (xb - xa)) ** k
            new_cols = [
                type(c).from_coeffs(c.diff(k).coeffs * scale)
                for c in self.cols
            ]
            return Chebfun3(cols=new_cols, rows=list(self.rows),
                            tubes=list(self.tubes), core=self.core,
                            domain=self.domain)
        if dim == 2:
            scale = (2.0 / (yb - ya)) ** k
            new_rows = [
                type(r).from_coeffs(r.diff(k).coeffs * scale)
                for r in self.rows
            ]
            return Chebfun3(cols=list(self.cols), rows=new_rows,
                            tubes=list(self.tubes), core=self.core,
                            domain=self.domain)
        scale = (2.0 / (zb - za)) ** k
        new_tubes = [
            type(t).from_coeffs(t.diff(k).coeffs * scale)
            for t in self.tubes
        ]
        return Chebfun3(cols=list(self.cols), rows=list(self.rows),
                        tubes=new_tubes, core=self.core,
                        domain=self.domain)

    def grad(self) -> tuple["Chebfun3", "Chebfun3", "Chebfun3"]:
        """Cartesian gradient (f_x, f_y, f_z).

        Provenance
        ----------
        MATLAB source : @chebfun3/grad.m
        Chebfun commit: 7574c77
        """
        return self.diff(1), self.diff(2), self.diff(3)

    def laplacian(self) -> "Chebfun3":
        """Laplacian f_xx + f_yy + f_zz (MATLAB laplacian).

        Provenance
        ----------
        MATLAB source : @chebfun3/laplacian.m
        Chebfun commit: 7574c77
        """
        return self.diff(1, 2) + self.diff(2, 2) + self.diff(3, 2)

    def lap(self) -> "Chebfun3":
        """Alias of :meth:`laplacian` (MATLAB lap).

        Provenance
        ----------
        MATLAB source : @chebfun3/lap.m
        Chebfun commit: 7574c77
        """
        return self.laplacian()

    def cumsum(self, dim: int = 1) -> "Chebfun3":
        """Indefinite integral along one variable, returning a Chebfun3
        that vanishes at the lower edge of that variable (MATLAB cumsum).

        The Tucker structure makes this a per-factor operation: integrating
        in x replaces every column fiber by its antiderivative (with the
        physical-interval half-width scaling), leaving the core and the
        other factors untouched.  ``dim`` = 1, 2, 3 -> x, y, z.

        Provenance
        ----------
        MATLAB source : @chebfun3/cumsum.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb, za, zb = self.domain
        if dim == 1:
            half = 0.5 * (xb - xa)
            new_cols = [type(c).from_coeffs(c.cumsum().coeffs * half)
                        for c in self.cols]
            return Chebfun3(cols=new_cols, rows=list(self.rows),
                            tubes=list(self.tubes), core=self.core,
                            domain=self.domain)
        if dim == 2:
            half = 0.5 * (yb - ya)
            new_rows = [type(r).from_coeffs(r.cumsum().coeffs * half)
                        for r in self.rows]
            return Chebfun3(cols=list(self.cols), rows=new_rows,
                            tubes=list(self.tubes), core=self.core,
                            domain=self.domain)
        if dim == 3:
            half = 0.5 * (zb - za)
            new_tubes = [type(t).from_coeffs(t.cumsum().coeffs * half)
                         for t in self.tubes]
            return Chebfun3(cols=list(self.cols), rows=list(self.rows),
                            tubes=new_tubes, core=self.core,
                            domain=self.domain)
        raise ValueError("Integration direction must be x, y, or z.")

    def cumsum2(self, dims: tuple[int, int] = (1, 2)) -> "Chebfun3":
        """Double indefinite integral along two variables (MATLAB cumsum2).

        Provenance
        ----------
        MATLAB source : @chebfun3/cumsum2.m
        Chebfun commit: 7574c77
        """
        d1, d2 = dims
        return self.cumsum(d1).cumsum(d2)

    def cumsum3(self) -> "Chebfun3":
        """Triple indefinite integral in x, then y, then z (MATLAB cumsum3).

        Provenance
        ----------
        MATLAB source : @chebfun3/cumsum3.m
        Chebfun commit: 7574c77
        """
        return self.cumsum(1).cumsum(2).cumsum(3)

    def sum(self, dim: int = 1):
        """Integrate over one variable, returning a Chebfun2.

        Contracts the core against the integrals of the chosen factor
        fibers; the result is a function of the remaining two variables
        (e.g. ``sum(dim=1)`` integrates over x and returns f(y, z)).

        Provenance
        ----------
        MATLAB source : @chebfun3/sum.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d import chebfun2

        xa, xb, ya, yb, za, zb = self.domain
        if dim == 1:
            w = jnp.array([c.sum() for c in self.cols], dtype=jnp.float64)
            M = jnp.einsum('ijk,i->jk', self.core, w) * (0.5 * (xb - xa))
            f1, f2 = self.rows, self.tubes
            dom2 = (ya, yb, za, zb)
        elif dim == 2:
            w = jnp.array([r.sum() for r in self.rows], dtype=jnp.float64)
            M = jnp.einsum('ijk,j->ik', self.core, w) * (0.5 * (yb - ya))
            f1, f2 = self.cols, self.tubes
            dom2 = (xa, xb, za, zb)
        else:
            w = jnp.array([t.sum() for t in self.tubes], dtype=jnp.float64)
            M = jnp.einsum('ijk,k->ij', self.core, w) * (0.5 * (zb - za))
            f1, f2 = self.cols, self.rows
            dom2 = (xa, xb, ya, yb)

        a1, b1, a2, b2 = dom2

        def _fn(u, v):
            tu = 2.0 * (u - a1) / (b1 - a1) - 1.0
            tv = 2.0 * (v - a2) / (b2 - a2) - 1.0
            U = jnp.stack([g(tu) for g in f1])  # (r1, ...)
            V = jnp.stack([h(tv) for h in f2])  # (r2, ...)
            return jnp.einsum('ij,i...,j...->...', M, U, V)

        return chebfun2(_fn, domain=dom2)

    def sum2(self, dims: tuple[int, int] = (1, 2)):
        """Integrate over two variables, returning a 1D Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun3/sum2.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import chebfun

        xa, xb, ya, yb, za, zb = self.domain
        dims = tuple(sorted(dims))
        wx = jnp.array([c.sum() for c in self.cols], dtype=jnp.float64)
        wy = jnp.array([r.sum() for r in self.rows], dtype=jnp.float64)
        wz = jnp.array([t.sum() for t in self.tubes], dtype=jnp.float64)
        if dims == (1, 2):
            v = jnp.einsum('ijk,i,j->k', self.core, wx, wy) \
                * (0.25 * (xb - xa) * (yb - ya))
            fibers, (a, b) = self.tubes, (za, zb)
        elif dims == (1, 3):
            v = jnp.einsum('ijk,i,k->j', self.core, wx, wz) \
                * (0.25 * (xb - xa) * (zb - za))
            fibers, (a, b) = self.rows, (ya, yb)
        else:
            v = jnp.einsum('ijk,j,k->i', self.core, wy, wz) \
                * (0.25 * (yb - ya) * (zb - za))
            fibers, (a, b) = self.cols, (xa, xb)

        def _fn(t):
            tr = 2.0 * (t - a) / (b - a) - 1.0
            F = jnp.stack([g(tr) for g in fibers])
            return jnp.einsum('i,i...->...', v, F)

        return chebfun(_fn, domain=(a, b))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> tuple[int, int, int]:
        """Tucker rank (rx, ry, rz) of the approximation."""
        return (len(self.cols), len(self.rows), len(self.tubes))

    def fix_the_rank(self, fixed_rank) -> "Chebfun3":
        """Truncate (or zero-pad) the Tucker rank to ``fixed_rank``.

        ``fixed_rank`` is a triple ``(t1, t2, t3)``; each mode's factors and
        the corresponding slice of the core tensor are truncated to the first
        ``t`` (dominant, ACA-pivot order) or padded with zero factors.  This
        is the ``chebfun3(op, 'rank', [t1 t2 t3])`` truncation.

        NOTE: the truncated approximation depends on the factor ordering of
        the underlying ACA construction; for the exact MATLAB result the
        chebfun3 constructor's pivoting must also match.

        Provenance
        ----------
        MATLAB source : @chebfun3/constructor.m (fixTheRank)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        t = [int(v) for v in fixed_rank]
        if len(t) != 3 or any(v < 0 for v in t):
            raise ValueError(
                "fix_the_rank: fixed_rank must be three nonnegative integers.")
        r = [len(self.cols), len(self.rows), len(self.tubes)]
        cols = list(self.cols)
        rows = list(self.rows)
        tubes = list(self.tubes)
        core = _np.asarray(self.core)

        def _zero_fun(sample):
            return Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))

        # Mode 1 (cols)
        if r[0] > t[0]:
            cols = cols[:t[0]]
            core = core[:t[0], :, :]
        elif r[0] < t[0]:
            cols = cols + [_zero_fun(cols[0]) for _ in range(t[0] - r[0])]
            pad = _np.zeros((t[0], core.shape[1], core.shape[2]), dtype=core.dtype)
            pad[:r[0], :, :] = core
            core = pad
        # Mode 2 (rows)
        if r[1] > t[1]:
            rows = rows[:t[1]]
            core = core[:, :t[1], :]
        elif r[1] < t[1]:
            rows = rows + [_zero_fun(rows[0]) for _ in range(t[1] - r[1])]
            pad = _np.zeros((core.shape[0], t[1], core.shape[2]), dtype=core.dtype)
            pad[:, :r[1], :] = core
            core = pad
        # Mode 3 (tubes)
        if r[2] > t[2]:
            tubes = tubes[:t[2]]
            core = core[:, :, :t[2]]
        elif r[2] < t[2]:
            tubes = tubes + [_zero_fun(tubes[0]) for _ in range(t[2] - r[2])]
            pad = _np.zeros((core.shape[0], core.shape[1], t[2]), dtype=core.dtype)
            pad[:, :, :r[2]] = core
            core = pad

        return Chebfun3(cols=cols, rows=rows, tubes=tubes,
                        core=jnp.asarray(core, dtype=self.core.dtype),
                        domain=self.domain)

    def isreal(self) -> bool:
        """True if f is real-valued (MATLAB isreal).

        Checks the stored Tucker representation: a Chebfun3 is real iff its
        core and all factor coefficients are real (a purely structural test,
        matching MATLAB, so ``real(1i*x+y-z)`` reports real).

        Provenance
        ----------
        MATLAB source : @chebfun3/isreal.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return True
        if jnp.iscomplexobj(self.core):
            if float(jnp.max(jnp.abs(jnp.imag(self.core)))) > 0.0:
                return False
        for factors in (self.cols, self.rows, self.tubes):
            for t in factors:
                c = jnp.asarray(t.coeffs)
                if jnp.iscomplexobj(c) and \
                        float(jnp.max(jnp.abs(jnp.imag(c)))) > 0.0:
                    return False
        return True

    def iszero(self) -> bool:
        """True if f is the zero function (MATLAB iszero).

        Provenance
        ----------
        MATLAB source : @chebfun3/iszero.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return True
        return float(jnp.max(jnp.abs(self.core))) == 0.0

    def isequal(self, other: "Chebfun3") -> bool:
        """True if f and g are the same function (MATLAB isequal).

        Compares on the common domain by the L-infinity difference over a
        dense lattice (down to a scaled machine-epsilon threshold), which is
        robust to the non-unique Tucker representation of a given function.

        Provenance
        ----------
        MATLAB source : @chebfun3/isequal.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        if not isinstance(other, Chebfun3):
            return False
        if tuple(self.domain) != tuple(other.domain):
            return False
        d = self.domain
        grids = [jnp.asarray(_np.linspace(d[2 * i], d[2 * i + 1], 13))
                 for i in range(3)]
        xx, yy, zz = jnp.meshgrid(*grids, indexing="ij")
        a = self(xx, yy, zz)
        b = other(xx, yy, zz)
        scale = max(float(jnp.max(jnp.abs(a))),
                    float(jnp.max(jnp.abs(b))), 1.0)
        return float(jnp.max(jnp.abs(a - b))) <= 1e4 * _EPS * scale

    # ------------------------------------------------------------------
    # Arithmetic (MATLAB @chebfun3 plus/minus/times/rdivide/power)
    # Added by Claude Fable 5: Chebfun3 previously had NO arithmetic.
    # ------------------------------------------------------------------

    def _const_like(self, c) -> "Chebfun3":
        """Rank-(1,1,1) constant Chebfun3 on this function's domain."""
        one = Chebtech2(coeffs=jnp.ones(1, dtype=jnp.float64), ishappy=True)
        core = jnp.asarray(c).reshape(1, 1, 1)
        return Chebfun3(cols=[one], rows=[one], tubes=[one], core=core,
                        domain=self.domain)

    def _check_same_domain(self, other: "Chebfun3") -> None:
        if tuple(self.domain) != tuple(other.domain):
            raise ValueError(
                "Chebfun3 arithmetic requires matching domains: "
                f"{self.domain} vs {other.domain}")

    def __neg__(self) -> "Chebfun3":
        return Chebfun3(cols=list(self.cols), rows=list(self.rows),
                        tubes=list(self.tubes), core=-self.core,
                        domain=self.domain)

    def __add__(self, other) -> "Chebfun3":
        """f + g: exact block-diagonal Tucker embedding, then compression.

        The embedding is exact; the truncated-HOSVD compression then
        restores the minimal Tucker rank (so ``rank(f+f)`` stays
        ``rank(f)`` instead of doubling), which is what MATLAB's
        constructor-based ``@chebfun3/plus.m`` achieves.  A previous
        version re-approximated the sum through the full adaptive
        constructor instead -- semantically the MATLAB path, but it made
        EVERY addition cost a 3D adaptive construction and hung the abs/
        compose chains for ~30 minutes on CI.

        Provenance
        ----------
        MATLAB source : @chebfun3/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun3):
            self._check_same_domain(other)
            r1 = self.core.shape
            r2 = other.core.shape
            dt = jnp.result_type(self.core.dtype, other.core.dtype)
            core = jnp.zeros((r1[0] + r2[0], r1[1] + r2[1],
                              r1[2] + r2[2]), dtype=dt)
            core = core.at[:r1[0], :r1[1], :r1[2]].set(self.core)
            core = core.at[r1[0]:, r1[1]:, r1[2]:].set(other.core)
            out = Chebfun3(
                cols=list(self.cols) + list(other.cols),
                rows=list(self.rows) + list(other.rows),
                tubes=list(self.tubes) + list(other.tubes),
                core=core,
                domain=self.domain)
            return out._compress()
        if isinstance(other, (int, float, complex)):
            return self + self._const_like(other)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other) -> "Chebfun3":
        if isinstance(other, Chebfun3):
            return self + (-other)
        if isinstance(other, (int, float, complex)):
            return self + self._const_like(-other)
        return NotImplemented

    def __rsub__(self, other) -> "Chebfun3":
        return (-self) + other

    def __mul__(self, other) -> "Chebfun3":
        """Scalar multiply scales the core (exact); f.*g re-approximates
        the pointwise product with the constructor, as MATLAB
        @chebfun3/times.m does.

        Provenance
        ----------
        MATLAB source : @chebfun3/times.m, mtimes.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            return Chebfun3(cols=list(self.cols), rows=list(self.rows),
                            tubes=list(self.tubes),
                            core=self.core * other, domain=self.domain)
        if isinstance(other, Chebfun3):
            self._check_same_domain(other)
            return Chebfun3.from_function(
                lambda x, y, z: self(x, y, z) * other(x, y, z),
                domain=self.domain)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Chebfun3":
        if isinstance(other, (int, float, complex)):
            return self * (1.0 / other)
        if isinstance(other, Chebfun3):
            self._check_same_domain(other)
            return Chebfun3.from_function(
                lambda x, y, z: self(x, y, z) / other(x, y, z),
                domain=self.domain)
        return NotImplemented

    def __rtruediv__(self, other) -> "Chebfun3":
        return Chebfun3.from_function(
            lambda x, y, z: other / self(x, y, z), domain=self.domain)

    def permute(self, order) -> "Chebfun3":
        """Permute the variables (MATLAB permute): permute(f, [2 1 3])
        gives g(x,y,z) = f(y,x,z), with the domain permuted to match.

        Provenance
        ----------
        MATLAB source : @chebfun3/permute.m
        Chebfun commit: 7574c77
        """
        order = [int(o) - 1 if min(order) == 1 else int(o)
                 for o in order]
        d = self.domain
        ivals = [(d[0], d[1]), (d[2], d[3]), (d[4], d[5])]
        new_dom = ivals[order[0]] + ivals[order[1]] + ivals[order[2]]

        def g(x, y, z):
            args = [None, None, None]
            args[order[0]], args[order[1]], args[order[2]] = x, y, z
            return self(*args)

        return Chebfun3.from_function(g, domain=new_dom)

    def restrict(self, dom):
        """Restrict to a subdomain (MATLAB restrict / {}-indexing).

        ``dom`` is ``(xa,xb, ya,yb, za,zb)``.  Degenerate intervals
        collapse dimensions: 3 points -> float, 2 points -> Chebfun,
        1 point -> Chebfun2, none -> Chebfun3.

        Provenance
        ----------
        MATLAB source : @chebfun3/restrict.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        v = [float(t) for t in dom]
        pts = [v[0] == v[1], v[2] == v[3], v[4] == v[5]]
        fixed = [v[0], v[2], v[4]]
        free = [i for i in range(3) if not pts[i]]
        if len(free) == 0:
            return float(self(jnp.asarray(v[0]), jnp.asarray(v[2]),
                              jnp.asarray(v[4])))
        if len(free) == 1:
            i = free[0]

            def f1(t, i=i):
                args = [jnp.full_like(t, fixed[k]) for k in range(3)]
                args[i] = t
                return self(*args)

            return Chebfun.from_function(
                f1, Domain((v[2 * i], v[2 * i + 1])))
        if len(free) == 2:
            i, j = free

            def f2(s, t, i=i, j=j):
                args = [jnp.full_like(s, fixed[k]) for k in range(3)]
                args[i], args[j] = s, t
                return self(*args)

            return Chebfun2.from_function(
                f2, domain=(v[2 * i], v[2 * i + 1],
                            v[2 * j], v[2 * j + 1]))
        return Chebfun3.from_function(
            lambda x, y, z: self(x, y, z), domain=tuple(v))

    def squeeze(self):
        """Collapse constant dimensions (MATLAB squeeze):
        returns a Chebfun (one active variable), Chebfun2 (two), or
        self (all three active).

        Provenance
        ----------
        MATLAB source : @chebfun3/squeeze.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        d = self.domain
        grids = [jnp.asarray(_np.linspace(d[2 * i], d[2 * i + 1], 17))
                 for i in range(3)]
        xx, yy, zz = jnp.meshgrid(*grids, indexing="ij")
        vals = self(xx, yy, zz)
        vs = max(float(jnp.max(jnp.abs(vals))), 1.0)
        tol = 1e5 * float(_np.finfo(float).eps) * vs
        active = []
        for i in range(3):
            # variation ALONG axis i: range of values as x_i sweeps,
            # for every fixed combination of the other coordinates
            rng = vals.max(axis=i) - vals.min(axis=i)
            if float(jnp.max(rng)) > tol:
                active.append(i)
        mids = [0.5 * (d[2 * i] + d[2 * i + 1]) for i in range(3)]
        if len(active) == 3:
            return self
        if len(active) == 2:
            i, j = active

            def f2(s, t, i=i, j=j):
                args = [jnp.full_like(s, mids[k]) for k in range(3)]
                args[i], args[j] = s, t
                return self(*args)

            return Chebfun2.from_function(
                f2, domain=(d[2 * i], d[2 * i + 1],
                            d[2 * j], d[2 * j + 1]))
        i = active[0] if active else 0

        def f1(t, i=i):
            args = [jnp.full_like(t, mids[k]) for k in range(3)]
            args[i] = t
            return self(*args)

        return Chebfun.from_function(
            f1, Domain((d[2 * i], d[2 * i + 1])))

    def mean3(self) -> jax.Array:
        """Mean value over the box (MATLAB mean3; Fable 5)."""
        xa, xb, ya, yb, za, zb = self.domain
        vol = (xb - xa) * (yb - ya) * (zb - za)
        return self.sum3() / vol

    def std3(self) -> jax.Array:
        """Standard deviation over the box (MATLAB std3)."""
        mu = float(self.mean3())
        var = (self - mu) * (self - mu)
        return jnp.sqrt(var.mean3())

    def norm(self) -> jax.Array:
        """L2 norm sqrt(int |f|^2) (MATLAB norm; Fable 5)."""
        f2 = self * self if not any(
            jnp.iscomplexobj(c.coeffs) for c in self.cols) else None
        if f2 is None:
            f2 = Chebfun3.from_function(
                lambda x, y, z: jnp.abs(self(x, y, z)) ** 2,
                domain=self.domain)
        return jnp.sqrt(jnp.abs(f2.sum3()))

    def minandmax3(self, ngrid: int = 41, n_starts: int = 48):
        """Global extrema via dense-grid seed + multi-start Newton polish
        (MATLAB minandmax3).

        A single projected-gradient descent from the best grid point is not
        robust for oscillatory functions: the coarse seed grid rarely lands
        in the basin of the global extremum, so the polish settles into a
        nearby local one.  We therefore polish from the ``n_starts`` most
        extremal grid points and keep the best result -- this recovers the
        true global extremum (e.g. Wagon's function, min3 = -3.32833834566,
        which single-start seeding misses by ~1%).
        """
        import numpy as _np
        xa, xb, ya, yb, za, zb = self.domain
        g1 = _np.linspace(xa, xb, ngrid)
        g2 = _np.linspace(ya, yb, ngrid)
        g3 = _np.linspace(za, zb, ngrid)
        XX, YY, ZZ = _np.meshgrid(g1, g2, g3, indexing="ij")
        xr, yr, zr = XX.ravel(), YY.ravel(), ZZ.ravel()
        V = _np.asarray(self(jnp.asarray(xr), jnp.asarray(yr),
                             jnp.asarray(zr)))
        grads = [self.diff(dim=d) for d in (1, 2, 3)]
        lo = _np.array([xa, ya, za])
        hi = _np.array([xb, yb, zb])
        step0 = _np.min(hi - lo) / (ngrid - 1)

        def _polish(p0, sgn):
            p = _np.array(p0, dtype=float)
            step = step0
            for _ in range(80):     # projected gradient ascent/descent
                g = _np.array([float(gr(jnp.asarray(p[0]),
                                        jnp.asarray(p[1]),
                                        jnp.asarray(p[2])))
                               for gr in grads])
                pn = _np.clip(p + sgn * step * g /
                              max(_np.linalg.norm(g), 1e-30), lo, hi)
                vo = float(self(jnp.asarray(p[0]), jnp.asarray(p[1]),
                                jnp.asarray(p[2])))
                vn = float(self(jnp.asarray(pn[0]), jnp.asarray(pn[1]),
                                jnp.asarray(pn[2])))
                if sgn * (vn - vo) > 0:
                    p = pn
                else:
                    step *= 0.5
                    if step < 1e-13:
                        break
            return float(self(jnp.asarray(p[0]), jnp.asarray(p[1]),
                              jnp.asarray(p[2]))), p

        out_vals, out_locs = [], []
        k = int(max(1, min(n_starts, V.size)))
        for which in ("min", "max"):
            sgn = -1.0 if which == "min" else 1.0
            # Top-k most extremal grid seeds (argsort ascending; min takes
            # the smallest, max the largest).
            order = _np.argsort(V)
            seeds = order[:k] if which == "min" else order[-k:]
            best_val = _np.inf if which == "min" else -_np.inf
            best_p = None
            for i in seeds:
                val, p = _polish((xr[i], yr[i], zr[i]), sgn)
                if (which == "min" and val < best_val) or \
                   (which == "max" and val > best_val):
                    best_val, best_p = val, p
            out_vals.append(best_val)
            out_locs.append(best_p.tolist())
        return jnp.asarray(out_vals), jnp.asarray(out_locs)

    def max3(self):
        vals, locs = self.minandmax3()
        return vals[1], locs[1]

    def min3(self):
        vals, locs = self.minandmax3()
        return vals[0], locs[0]

    def sample(self, m: int, n: int, p: int) -> jax.Array:
        """Values on an m-by-n-by-p tensor Chebyshev grid.

        Returns ``V[i, j, k] = f(x_i, y_j, z_k)`` with 2nd-kind Chebyshev
        points in each direction (natural x, y, z index order).

        Provenance
        ----------
        MATLAB source : @chebfun3/sample.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.quadrature import chebpts_ab

        d = self.domain
        x = chebpts_ab(m, d[0], d[1], kind=2)
        y = chebpts_ab(n, d[2], d[3], kind=2)
        z = chebpts_ab(p, d[4], d[5], kind=2)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        return self(X, Y, Z)

    def _extremum(self, g, dim: int, reducer):
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2

        if g is not None:
            raise ValueError(
                "Unable to maximize/minimize two Chebfun3 objects.")
        if dim == 0:
            raise ValueError(
                "Dimension argument must be a positive integer scalar "
                "within indexing range.")
        if dim not in (1, 2, 3):
            # MATLAB returns f itself for out-of-range dims (like max()).
            return self
        d = self.domain
        n = 129  # MATLAB's sampling resolution
        vals = self.sample(n, n, n)
        v = reducer(vals, dim - 1)
        rem = [i for i in (0, 1, 2) if i != dim - 1]
        dom2 = (d[2 * rem[0]], d[2 * rem[0] + 1],
                d[2 * rem[1]], d[2 * rem[1] + 1])
        ev = _grid_interp2(v, dom2)
        return Chebfun2.from_function(ev, domain=dom2)

    def max(self, g=None, dim: int = 1) -> "Chebfun2":
        """Maximum along one variable, as a Chebfun2 (MATLAB max).

        ``max(f)`` / ``dim=1`` maximizes over x, returning a Chebfun2 in
        (y, z); ``dim=2`` over y -> (x, z); ``dim=3`` over z -> (x, y).
        For a ``dim`` outside 1-3, f itself is returned, as in MATLAB.

        Provenance
        ----------
        MATLAB source : @chebfun3/max.m
        Chebfun commit: 7574c77
        """
        return self._extremum(g, dim, lambda v, ax: jnp.max(v, axis=ax))

    def min(self, g=None, dim: int = 1) -> "Chebfun2":
        """Minimum along one variable, as a Chebfun2 (MATLAB min).

        Provenance
        ----------
        MATLAB source : @chebfun3/min.m
        Chebfun commit: 7574c77
        """
        return self._extremum(g, dim, lambda v, ax: jnp.min(v, axis=ax))

    def _extremum2(self, g, dims, reducer):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.domain import Domain

        if g is not None:
            raise ValueError(
                "Unable to maximize/minimize two Chebfun3 objects.")
        s = tuple(sorted(int(t) for t in dims))
        if s not in ((1, 2), (1, 3), (2, 3)):
            if 0 in s:
                raise ValueError(
                    "Dimension arguments must be two positive integer "
                    "scalars within indexing range.")
            return self
        d = self.domain
        n = 129  # MATLAB's sampling resolution
        vals = self.sample(n, n, n)
        v = reducer(vals, (s[0] - 1, s[1] - 1))
        rem = ({0, 1, 2} - {s[0] - 1, s[1] - 1}).pop()
        return Chebfun.from_values(
            v, Domain((d[2 * rem], d[2 * rem + 1])))

    def max2(self, g=None, dims=(1, 2)):
        """Maximum along two variables, as a 1D Chebfun (MATLAB max2).

        Provenance
        ----------
        MATLAB source : @chebfun3/max2.m
        Chebfun commit: 7574c77
        """
        return self._extremum2(g, dims, lambda v, ax: jnp.max(v, axis=ax))

    def min2(self, g=None, dims=(1, 2)):
        """Minimum along two variables, as a 1D Chebfun (MATLAB min2).

        Provenance
        ----------
        MATLAB source : @chebfun3/min2.m
        Chebfun commit: 7574c77
        """
        return self._extremum2(g, dims, lambda v, ax: jnp.min(v, axis=ax))

    def compose(self, op) -> "Chebfun3":
        """Re-approximate op(f(x, y, z)) (MATLAB compose; Fable 5)."""
        return Chebfun3.from_function(
            lambda x, y, z: op(self(x, y, z)), domain=self.domain)

    def exp(self):
        return self.compose(jnp.exp)

    def sin(self):
        return self.compose(jnp.sin)

    def cos(self):
        return self.compose(jnp.cos)

    def sqrt(self):
        return self.compose(jnp.sqrt)

    def log(self):
        return self.compose(jnp.log)

    def tanh(self):
        return self.compose(jnp.tanh)

    def abs(self):
        return self.compose(jnp.abs)

    def real(self) -> "Chebfun3":
        """Real part, re-approximated adaptively (a complex Chebfun3's
        real part is not directly available from its Tucker factors).

        Provenance
        ----------
        MATLAB source : @chebfun3/real.m
        Chebfun commit: 7574c77
        """
        return Chebfun3.from_function(
            lambda x, y, z: jnp.real(self(x, y, z)), domain=self.domain)

    def imag(self) -> "Chebfun3":
        """Imaginary part (MATLAB imag).

        Provenance
        ----------
        MATLAB source : @chebfun3/imag.m
        Chebfun commit: 7574c77
        """
        return Chebfun3.from_function(
            lambda x, y, z: jnp.imag(self(x, y, z)), domain=self.domain)

    def conj(self) -> "Chebfun3":
        """Complex conjugate (MATLAB conj).

        Provenance
        ----------
        MATLAB source : @chebfun3/conj.m
        Chebfun commit: 7574c77
        """
        return Chebfun3.from_function(
            lambda x, y, z: jnp.conj(self(x, y, z)), domain=self.domain)

    @classmethod
    def complex(cls, re: "Chebfun3", im: "Chebfun3") -> "Chebfun3":
        """Complex Chebfun3 from real and imaginary parts (MATLAB complex).

        Provenance
        ----------
        MATLAB source : @chebfun3/complex.m
        Chebfun commit: 7574c77
        """
        re._check_same_domain(im)
        return Chebfun3.from_function(
            lambda x, y, z: re(x, y, z) + 1j * im(x, y, z),
            domain=re.domain)

    def std(self, flag=None, dim: int = 1):
        """Standard deviation along one variable, returning a Chebfun2.

        ``std(f) = sqrt(mean((f - mean(f))**2))`` taken over the x-variable
        by default (``dim`` = 1, 2, 3 -> x, y, z).  ``flag`` is accepted and
        ignored to mirror MATLAB's ``std(f, flag, dim)`` syntax.

        Provenance
        ----------
        MATLAB source : @chebfun3/std.m
        Chebfun commit: 7574c77
        """
        d = self.domain
        i = dim - 1
        m2 = self.mean(dim)  # Chebfun2 over the two remaining variables

        if dim == 1:
            def _mfun(x, y, z):
                return m2(y, z)
        elif dim == 2:
            def _mfun(x, y, z):
                return m2(x, z)
        elif dim == 3:
            def _mfun(x, y, z):
                return m2(x, y)
        else:
            raise ValueError("dim must be 1, 2, or 3.")

        m3 = Chebfun3.from_function(_mfun, domain=d)
        width = d[2 * i + 1] - d[2 * i]
        var = ((self - m3) ** 2).sum(dim) * (1.0 / width)
        return var.sqrt()

    def std2(self, flag=None, dims: tuple[int, int] = (1, 2)):
        """Standard deviation along two variables, returning a Chebfun.

        ``std2(f) = sqrt(mean2((f - mean2(f))**2))`` over the (x, y)
        variables by default.  ``flag`` is accepted and ignored to mirror
        MATLAB's ``std2(f, flag, dims)`` syntax.

        Provenance
        ----------
        MATLAB source : @chebfun3/std2.m
        Chebfun commit: 7574c77
        """
        d = self.domain
        s = tuple(sorted(dims))
        m1 = self.mean2(s)  # Chebfun over the one remaining variable

        if s == (1, 2):
            def _mfun(x, y, z):
                return m1(z)
        elif s == (1, 3):
            def _mfun(x, y, z):
                return m1(y)
        elif s == (2, 3):
            def _mfun(x, y, z):
                return m1(x)
        else:
            raise ValueError("dims must be two distinct values in 1, 2, 3.")

        m3 = Chebfun3.from_function(_mfun, domain=d)
        wa = d[2 * (s[0] - 1) + 1] - d[2 * (s[0] - 1)]
        wb = d[2 * (s[1] - 1) + 1] - d[2 * (s[1] - 1)]
        var = ((self - m3) ** 2).sum2(s) * (1.0 / (wa * wb))
        return var.sqrt()

    def integral(self, curve=None, domain=None):
        """Line integral of f along a parametric curve, or the triple
        definite integral when no curve is given (MATLAB integral).

        With ``curve`` supplied the value is
        ``int_C f ds = int_{t0}^{t1} f(g(t)) |g'(t)| dt`` where the curve
        ``g(t) = (x(t), y(t), z(t))`` is given either as a callable
        returning the three coordinates (with ``domain=(t0, t1)``) or as an
        array-valued Chebfun with three columns.  Without a curve it returns
        :meth:`sum3`.

        Provenance
        ----------
        MATLAB source : @chebfun3/integral.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain

        if curve is None:
            return self.sum3()

        # Build a scalar Chebfun for each coordinate over the parameter
        # interval, so we can differentiate the curve and integrate the
        # resulting arc-length-weighted integrand with the 1D machinery.
        if hasattr(curve, "funs") or isinstance(curve, Chebfun):
            cdom = curve.domain
            t0, t1 = float(cdom[0]), float(cdom[-1])

            def _comp(k):
                return Chebfun.from_function(
                    lambda t, k=k: jnp.asarray(curve(t))[..., k],
                    Domain((t0, t1)))
        else:
            if domain is None:
                raise ValueError(
                    "A callable curve requires domain=(t0, t1).")
            t0, t1 = float(domain[0]), float(domain[1])

            def _comp(k):
                return Chebfun.from_function(
                    lambda t, k=k: jnp.asarray(curve(t))[k],
                    Domain((t0, t1)))

        cx, cy, cz = _comp(0), _comp(1), _comp(2)
        dx, dy, dz = cx.diff(), cy.diff(), cz.diff()

        def _integrand(t):
            xt, yt, zt = cx(t), cy(t), cz(t)
            speed = jnp.sqrt(dx(t) ** 2 + dy(t) ** 2 + dz(t) ** 2)
            return self(xt, yt, zt) * speed

        return Chebfun.from_function(
            _integrand, Domain((t0, t1))).sum()

    def integral2(self, surface=None, domain=None):
        """Surface integral of f over a parametric surface, or the double
        definite integral when no surface is given (MATLAB integral2).

        With ``surface`` supplied the value is
        ``int int_S f dS = int int_DOM f(S(u,v)) |S_u x S_v| du dv`` where
        the surface ``S(u, v) = (x(u,v), y(u,v), z(u,v))`` is given either
        as a callable returning the three coordinates (with
        ``domain=(ua, ub, va, vb)``) or as a Chebfun2v with three
        components.  Without a surface it returns :meth:`sum2`.

        Provenance
        ----------
        MATLAB source : @chebfun3/integral2.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d import chebfun2

        if surface is None:
            return self.sum2()

        if hasattr(surface, "components"):
            comps = surface.components
            s1, s2, s3 = comps[0], comps[1], comps[2]
            sdom = s1.approx.domain if hasattr(s1, "approx") else s1.domain

            def _S(u, v):
                return s1(u, v), s2(u, v), s3(u, v)
        else:
            if domain is None:
                raise ValueError(
                    "A callable surface requires "
                    "domain=(ua, ub, va, vb).")
            sdom = tuple(float(t) for t in domain)
            _S = surface

        def _stacked(uv):
            comp = _S(uv[0], uv[1])
            return jnp.stack([comp[0], comp[1], comp[2]])

        jac = jax.jacfwd(_stacked)

        def _one(u, v):
            jm = jac(jnp.stack([u, v]))          # (3, 2): columns S_u, S_v
            cr = jnp.cross(jm[:, 0], jm[:, 1])
            area = jnp.sqrt(jnp.sum(cr ** 2))
            x, y, z = _S(u, v)
            return self(x, y, z) * area

        def _integrand(u, v):
            u = jnp.asarray(u, dtype=jnp.float64)
            v = jnp.asarray(v, dtype=jnp.float64)
            return jnp.vectorize(_one)(u, v)

        return chebfun2(_integrand, domain=sdom).sum2()

    def __pow__(self, p) -> "Chebfun3":
        return Chebfun3.from_function(
            lambda x, y, z: self(x, y, z) ** p, domain=self.domain)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Plot boundary face slices of this Chebfun3 (calls :func:`chebfunjax.plotting.plot_chebfun3`)."""
        from chebfunjax.plotting import plot_chebfun3
        return plot_chebfun3(self, **kwargs)

    def surf(self, **kwargs):
        """Cross-section surfaces of this Chebfun3 (calls :func:`chebfunjax.plotting.surf_chebfun3`)."""
        from chebfunjax.plotting import surf_chebfun3
        return surf_chebfun3(self, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact summary line, similar to MATLAB Chebfun3 display.

        Examples
        --------
        >>> f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        >>> repr(f)
        'Chebfun3(rank=(2, 2, 2), domain=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))'

        Provenance
        ----------
        MATLAB source : @chebfun3/disp.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb, za, zb = self.domain
        rx, ry, rz = self.rank
        return (
            f"Chebfun3(rank=({rx}, {ry}, {rz}), "
            f"domain=(({xa}, {xb}), ({ya}, {yb}), ({za}, {zb})))"
        )


# ============================================================================
# Factory function
# ============================================================================


def chebfun3(
    f: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    domain: tuple[float, float, float, float, float, float] = (
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    ),
    tol: float = _EPS,
    max_rank: int = 128,
    min_samples: int = 9,
    rank=None,
) -> Chebfun3:
    """Construct a Chebfun3 approximation of a trivariate function.

    Convenience factory wrapping ``Chebfun3.from_function``.

    Parameters
    ----------
    f : callable
        f(xx, yy, zz) accepting ndgrid-style 3D arrays.
    domain : 6-tuple of floats, optional
        (xa, xb, ya, yb, za, zb).  Default is (-1, 1, -1, 1, -1, 1).
    tol : float, optional
        Target relative tolerance.  Default is machine epsilon (~2.2e-16).
    max_rank : int, optional
        Maximum rank in each mode.  Default 128.
    min_samples : int, optional
        Minimum grid points per direction in Phase 1.  Default 9.

    Returns
    -------
    Chebfun3
        A Tucker-format approximation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun3d.chebfun3 import chebfun3
    >>> f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
    >>> abs(float(f(0.0, 0.0, 0.0)) - 1.0) < 1e-12
    True

    Provenance
    ----------
    MATLAB source : @chebfun3/chebfun3.m (constructor entry point)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun3, Chebfun3.from_function
    """
    g = Chebfun3.from_function(
        f,
        domain=domain,
        tol=tol,
        max_rank=max_rank,
        min_samples=min_samples,
    )
    # chebfun3(f, 'rank', [t1 t2 t3]): construct fully, then fix the rank.
    if rank is not None:
        g = g.fix_the_rank(rank)
    return g


from chebfunjax.utils.misc import make_empty_aware  # noqa: E402

make_empty_aware(Chebfun3, ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', '__pow__', '__neg__', 'sum3', 'mean3', 'std3', 'norm', 'permute', 'squeeze', 'restrict', 'minandmax3', 'max3', 'min3', 'compose', 'exp', 'sin', 'cos', 'sqrt', 'log', 'tanh', 'abs', 'hosvd'])
