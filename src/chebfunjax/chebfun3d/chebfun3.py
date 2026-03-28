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

        n = [max(min_samples, 9), max(min_samples, 9), max(min_samples, 9)]
        # Initial ranks for random initialization
        r = [6, 6, 6]
        abs_tol_running = tol

        xp = _chebpts_phys_np(n[0], xa, xb)
        yp = _chebpts_phys_np(n[1], ya, yb)
        zp = _chebpts_phys_np(n[2], za, zb)

        # Initialize random fiber indices (spread across interval)
        rng = np.random.default_rng(16051821)

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

        return cls(
            cols=cols_list,
            rows=rows_list,
            tubes=tubes_list,
            core=core_jax,
            domain=dom,
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

        result = jnp.zeros(bcast_shape, dtype=jnp.float64)
        r1 = len(self.cols)
        r2 = len(self.rows)
        r3 = len(self.tubes)

        for i in range(r1):
            xi_val = self.cols[i](tx)
            for j in range(r2):
                yj_val = self.rows[j](ty)
                for k in range(r3):
                    zk_val = self.tubes[k](tz)
                    result = result + self.core[i, j, k] * xi_val * yj_val * zk_val

        return result

    # ------------------------------------------------------------------
    # Triple integral
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> tuple[int, int, int]:
        """Tucker rank (rx, ry, rz) of the approximation."""
        return (len(self.cols), len(self.rows), len(self.tubes))

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
    return Chebfun3.from_function(
        f,
        domain=domain,
        tol=tol,
        max_rank=max_rank,
        min_samples=min_samples,
    )
