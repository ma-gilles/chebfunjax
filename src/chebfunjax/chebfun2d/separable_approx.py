# uses-numpy: adaptive 2D construction uses numpy for pivot selection (not JIT-safe)
"""SeparableApprox — low-rank approximation of 2D functions.

Represents a bivariate smooth function f(x, y) on a rectangle [xa, xb] x [ya, yb]
as a sum of rank-1 outer products:

    f(x, y) ≈ Σ_j  d_j * c_j(y) * r_j(x)

where c_j are column slices (functions of y), r_j are row slices (functions of x),
and d_j are scalar pivot values forming a diagonal matrix D.

The approximation is computed via Gaussian elimination with complete pivoting
(the Chebfun2 algorithm), which is the continuous analogue of GE on a matrix.

Translated from MATLAB Chebfun classes @separableApprox and @chebfun2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
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
from chebfunjax.utils.quadrature import chebpts

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Affine mapping helpers
# ============================================================================


def _ref_to_phys(t: jax.Array, a: float, b: float) -> jax.Array:
    """Map reference coordinate t in [-1, 1] to physical coordinate in [a, b]."""
    return 0.5 * (b - a) * t + 0.5 * (a + b)


def _phys_to_ref(x: jax.Array, a: float, b: float) -> jax.Array:
    """Map physical coordinate x in [a, b] to reference coordinate in [-1, 1]."""
    return (2.0 * x - (a + b)) / (b - a)


def _chebpts_phys(n: int, a: float, b: float) -> jax.Array:
    """Chebyshev-2 points on physical interval [a, b]."""
    t = chebpts(n, kind=2)
    return _ref_to_phys(t, a, b)


# ============================================================================
# Phase 1 helper: Adaptive Cross Approximation with complete pivoting (ACA)
# ============================================================================


def _complete_aca(
    A: np.ndarray,
    abs_tol: float,
    factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Gaussian elimination with complete pivoting on a matrix A.

    This is the discrete (matrix-level) version of the Chebfun2 algorithm,
    used in Phase 1 to find pivot locations and estimate the numerical rank.

    Parameters
    ----------
    A : np.ndarray, shape (ny, nx)
        Matrix of function values on a Chebyshev tensor grid.
        Rows correspond to y-points, columns to x-points.
    abs_tol : float
        Absolute tolerance; stop when max|residual| < abs_tol.
    factor : int
        Maximum rank is bounded by min(nx, ny) / factor.

    Returns
    -------
    pivot_vals : np.ndarray, shape (r,)
        Pivot values (d_j).
    pivot_pos : np.ndarray, shape (r, 2)
        Pivot positions as (row_idx, col_idx) pairs.
    row_vals : np.ndarray, shape (r, nx)
        Selected rows of A (after GE updates).
    col_vals : np.ndarray, shape (ny, r)
        Selected columns of A (after GE updates).
    ifail : bool
        True if GE did not converge (rank exceeded limit).

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m  (completeACA subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Townsend & Trefethen, "An extension of Chebfun to two
        dimensions", SISC, 35(6), C495–C518, 2013.
    """
    A = A.copy()
    ny, nx = A.shape
    width = min(ny, nx)

    pivot_vals = []
    pivot_pos = []
    row_vals_list = []
    col_vals_list = []
    ifail = True

    # Find initial maximum entry.
    # NumPy stores arrays in row-major (C) order, so flat index k corresponds
    # to A[k // nx, k % nx] for an (ny, nx) array.
    flat_idx = int(np.argmax(np.abs(A)))
    row = flat_idx // nx
    col = flat_idx % nx

    # Bias toward diagonal for square matrices (improves nonneg-definite detection)
    if ny == nx:
        diag_vals = np.abs(np.diag(A))
        diag_max = np.max(diag_vals)
        inf_norm = np.abs(A.flat[flat_idx])
        if diag_max - inf_norm > -abs_tol:
            diag_idx = int(np.argmax(diag_vals))
            row = diag_idx
            col = diag_idx

    scl = np.abs(A[row, col])

    if scl == 0.0:
        # Zero function
        return (
            np.array([0.0]),
            np.array([[0, 0]], dtype=int),
            np.zeros((1, nx)),
            np.zeros((ny, 1)),
            False,
        )

    z_rows = 0

    while True:
        inf_norm = np.max(np.abs(A))
        if inf_norm <= abs_tol:
            ifail = False
            break
        if z_rows >= width / factor:
            ifail = True
            break
        if z_rows >= min(ny, nx):
            ifail = True
            break

        # Extract current row/col
        r = A[row, :].copy()
        c = A[:, col].copy()
        piv = A[row, col]

        row_vals_list.append(r)
        col_vals_list.append(c)
        pivot_vals.append(piv)
        pivot_pos.append([row, col])

        # One step of GE
        A = A - np.outer(c, r) / piv

        z_rows += 1

        # Find next pivot (NumPy row-major: flat index k -> A[k//nx, k%nx])
        flat_idx = int(np.argmax(np.abs(A)))
        row = flat_idx // nx
        col = flat_idx % nx

        # Bias toward diagonal for square matrices
        if ny == nx:
            diag_vals = np.abs(np.diag(A))
            diag_max = np.max(diag_vals)
            inf_norm_cur = np.max(np.abs(A))
            if diag_max - inf_norm_cur > -abs_tol:
                diag_idx = int(np.argmax(diag_vals))
                row = diag_idx
                col = diag_idx

    if len(pivot_vals) == 0:
        return (
            np.array([0.0]),
            np.array([[0, 0]], dtype=int),
            np.zeros((1, nx)),
            np.zeros((ny, 1)),
            False,
        )

    pivot_vals = np.array(pivot_vals)
    pivot_pos = np.array(pivot_pos, dtype=int)
    # Stack: row_vals shape (r, nx), col_vals shape (ny, r)
    row_vals = np.stack(row_vals_list, axis=0)
    col_vals = np.stack(col_vals_list, axis=1)

    return pivot_vals, pivot_pos, row_vals, col_vals, ifail


# ============================================================================
# Phase 2 helper: GE on skeleton (update col/row slices at new resolution)
# ============================================================================


def _ge_on_skeleton(
    col_vals: np.ndarray,
    row_vals: np.ndarray,
    pivot_vals: np.ndarray,
    pivot_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-apply Gaussian elimination on skeleton column/row slices.

    After re-sampling at higher resolution, the col/row values at the new
    grid still need to be updated by the GE steps for earlier pivots, so
    that each slice represents the *residual* function after removing
    previous terms.

    Parameters
    ----------
    col_vals : np.ndarray, shape (ny_new, r)
        Column slices at new y-resolution, stacked horizontally.
    row_vals : np.ndarray, shape (r, nx_new)
        Row slices at new x-resolution, stacked vertically.
    pivot_vals : np.ndarray, shape (r,)
        Pivot values from Phase 1.
    pivot_pos : np.ndarray, shape (r, 2)
        Pivot positions (row_idx into col_vals, col_idx into row_vals).

    Returns
    -------
    col_vals_updated, row_vals_updated
        Same shapes, GE-updated.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m  (Phase 2 GE loop, lines ~173-179)
    Chebfun commit: 7574c77
    """
    col_vals = col_vals.copy()
    row_vals = row_vals.copy()
    r = len(pivot_vals)

    for k in range(r - 1):
        piv = pivot_vals[k]
        row_at_pivot_y = pivot_pos[k + 1 :, 0]  # row indices for later cols
        col_at_pivot_x = pivot_pos[k + 1 :, 1]  # col indices for later rows

        # Update later columns: col[:, k+1:] -= col[:, k] * (row[k, PP[k+1:, 1]] / piv)
        scale = row_vals[k, col_at_pivot_x] / piv
        col_vals[:, k + 1 :] = col_vals[:, k + 1 :] - np.outer(col_vals[:, k], scale)

        # Update later rows: row[k+1:, :] -= col[PP[k+1:, 0], k] * (row[k, :] / piv)
        scale_r = col_vals[row_at_pivot_y, k] / piv
        row_vals[k + 1 :, :] = row_vals[k + 1 :, :] - np.outer(scale_r, row_vals[k, :])

    return col_vals, row_vals


# ============================================================================
# Happiness check for a 1D array of values on [-1, 1]
# ============================================================================


def _is_happy(values: np.ndarray, tol: float) -> bool:
    """Check if a set of values (treated as a Chebtech2) is resolved to tol.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Values at Chebyshev-2 points of the 2nd kind.
    tol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if the Chebyshev coefficients of the values satisfy standard_chop.
    """
    from chebfunjax.utils.transforms import vals2coeffs

    v = jnp.asarray(values, dtype=jnp.float64)
    c = vals2coeffs(v)
    # standard_chop uses relative tolerance; convert abs_tol to relative
    vscale = float(jnp.max(jnp.abs(v)))
    if vscale == 0.0:
        return True
    rel_tol = tol / vscale
    rel_tol = max(rel_tol, _EPS)
    cutoff = standard_chop(c, rel_tol)
    return int(cutoff) < c.shape[0]


# ============================================================================
# Grid refinement (matches MATLAB gridRefine for chebtech2)
# ============================================================================


def _grid_refine(n: int) -> int:
    """Return the next grid size for adaptive refinement (Chebtech2 strategy).

    Doubles the 'interior' size: next = 2^(floor(log2(n-1)) + 1) + 1.
    This preserves nesting: chebpts of the coarser grid are a subset of the
    finer grid.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m  (gridRefine subfunction)
    Chebfun commit: 7574c77
    """
    if n <= 1:
        return 3
    m = int(n) - 1
    p = int(np.floor(np.log2(m)))
    return 2 ** (p + 1) + 1


# ============================================================================
# Main class
# ============================================================================


class SeparableApprox(eqx.Module):
    """Low-rank approximation of a bivariate function on a rectangle.

    Represents f(x, y) ≈ Σ_j d_j * c_j(y) * r_j(x), where

    - c_j are column slices (1D Chebtech2 on [ya, yb]),
    - r_j are row slices (1D Chebtech2 on [xa, xb]),
    - d_j are scalar pivot values.

    Construction uses Gaussian elimination with complete pivoting on a
    Chebyshev tensor grid (Phase 1 to find pivot locations, Phase 2 to
    resolve column/row slices adaptively).

    Attributes
    ----------
    cols : list of Chebtech2
        Column slices c_j(y), each a Chebtech2 on the reference interval [-1, 1].
        Physical domain is [ya, yb] (stored in domain).
    rows : list of Chebtech2
        Row slices r_j(x), each a Chebtech2 on the reference interval [-1, 1].
        Physical domain is [xa, xb] (stored in domain).
    pivots : jax.Array, shape (r,)
        Diagonal values d_j of the D matrix.
    domain : tuple (xa, xb, ya, yb)
        Physical domain of the approximation. Static field.

    Notes
    -----
    Construction is NOT JIT-safe (Python loops with data-dependent
    termination). Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @separableApprox/separableApprox.m,
        @chebfun2/constructor.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend & L. N. Trefethen, "An extension of Chebfun to
        two dimensions", SISC, 35(6), C495–C518, 2013.

    See Also
    --------
    Chebtech2
    """

    cols: list  # list of Chebtech2 (column slices, functions of y)
    rows: list  # list of Chebtech2 (row slices, functions of x)
    pivots: jax.Array  # shape (r,)
    domain: tuple = eqx.field(static=True)  # (xa, xb, ya, yb)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array, jax.Array], jax.Array],
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        tol: float = _EPS,
        max_rank: int = 513,
        min_samples: int = 9,
        max_samples: int = 2**14 + 1,
    ) -> "SeparableApprox":
        """Construct a SeparableApprox from a callable f(x, y).

        Uses Gaussian elimination with complete pivoting (the Chebfun2
        algorithm) to find a numerically low-rank representation.

        Phase 1: Sample f on a coarse Chebyshev tensor grid, find pivot
        locations via GE with complete pivoting.

        Phase 2: Resolve the column and row slices at those pivot locations
        by increasing the 1D grid size until Chebyshev coefficients decay
        below tol.

        Parameters
        ----------
        f : callable
            A function f(x, y) accepting arrays x (shape (m,)) and y (shape (n,))
            via meshgrid broadcasting, or scalar pairs. Must be vectorised:
            ``f(xx, yy)`` where xx and yy are 2D arrays from meshgrid.
        domain : tuple of 4 floats, optional
            (xa, xb, ya, yb). Default is (-1, 1, -1, 1).
        tol : float, optional
            Target relative tolerance. Default is machine epsilon (~2.2e-16).
        max_rank : int, optional
            Maximum allowed rank. Default 512.
        min_samples : int, optional
            Minimum number of Chebyshev points per dimension in Phase 1.
            Default 9 (= 2^3 + 1, matching Chebfun2 defaults).
        max_samples : int, optional
            Maximum number of Chebyshev points per dimension. Default 2^14 + 1.

        Returns
        -------
        SeparableApprox
            A low-rank approximation.

        Raises
        ------
        ValueError
            If f is infinite or NaN on the grid.
        RuntimeWarning
            If construction did not converge (rank or sample limit reached).

        Notes
        -----
        This method is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2/constructor.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Townsend & Trefethen, SISC 2013.
        """
        xa, xb, ya, yb = float(domain[0]), float(domain[1]), float(domain[2]), float(domain[3])
        factor = 4  # Ratio between grid size and number of pivots

        # Minimum grid size must be one plus a power of 2 for nesting
        _grid_refine(min_samples - 1) if min_samples > 1 else min_samples
        # Ensure it's at least 9 (2^3 + 1)
        min_grid_x = max(9, min_samples)
        min_grid_y = max(9, min_samples)

        # --- Helper: sample f on a tensor grid ---
        def _sample_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Sample f at Chebyshev points; return x_pts, y_pts, vals."""
            x_pts = np.array(_chebpts_phys(nx, xa, xb))
            y_pts = np.array(_chebpts_phys(ny, ya, yb))
            xx, yy = np.meshgrid(x_pts, y_pts)  # shape (ny, nx)
            xx_j = jnp.asarray(xx, dtype=jnp.float64)
            yy_j = jnp.asarray(yy, dtype=jnp.float64)
            vals = np.array(f(xx_j, yy_j), dtype=np.float64)
            return x_pts, y_pts, vals

        # --- Helper: evaluate f on a 1D grid at fixed x-pivot ---
        def _sample_col(x_pivot: float, ny: int) -> np.ndarray:
            """Sample f(x_pivot, y) on ny Chebyshev y-points."""
            y_pts = np.array(_chebpts_phys(ny, ya, yb))
            xx_j = jnp.full(ny, x_pivot, dtype=jnp.float64)
            yy_j = jnp.asarray(y_pts, dtype=jnp.float64)
            vals = np.array(f(xx_j, yy_j), dtype=np.float64)
            return vals

        def _sample_row(y_pivot: float, nx: int) -> np.ndarray:
            """Sample f(x, y_pivot) on nx Chebyshev x-points."""
            x_pts = np.array(_chebpts_phys(nx, xa, xb))
            xx_j = jnp.asarray(x_pts, dtype=jnp.float64)
            yy_j = jnp.full(nx, y_pivot, dtype=jnp.float64)
            vals = np.array(f(xx_j, yy_j), dtype=np.float64)
            return vals

        # ================================================================
        # PHASE 1: Find pivot locations via complete ACA
        # ================================================================

        grid_x = min_grid_x
        grid_y = min_grid_y
        is_happy = False
        failure = False

        x_pts, y_pts, vals = _sample_grid(grid_x, grid_y)

        # Check for inf/nan
        vscale = float(np.max(np.abs(vals)))
        if not np.isfinite(vscale):
            raise ValueError(
                "SeparableApprox.from_function: function returned Inf or NaN "
                f"on the initial grid over domain ({xa}, {xb}) x ({ya}, {yb})."
            )

        # 2D tolerance (matches MATLAB's getTol logic)
        abs_tol = max(tol * vscale, 1e4 * tol)

        pivot_vals, pivot_pos, row_vals_mat, col_vals_mat, ifail = _complete_aca(
            vals, abs_tol, factor
        )

        # If rank-1 case, ensure row_vals is a proper 2D array
        if row_vals_mat.ndim == 1:
            row_vals_mat = row_vals_mat[np.newaxis, :]

        strike = 1
        while (
            ifail
            and grid_x <= factor * (max_rank - 1) + 1
            and grid_y <= factor * (max_rank - 1) + 1
            and strike < 3
        ):
            grid_x = _grid_refine(grid_x)
            grid_y = _grid_refine(grid_y)
            x_pts, y_pts, vals = _sample_grid(grid_x, grid_y)
            vscale = float(np.max(np.abs(vals)))
            abs_tol = max(tol * vscale, 1e4 * tol)
            pivot_vals, pivot_pos, row_vals_mat, col_vals_mat, ifail = _complete_aca(
                vals, abs_tol, factor
            )
            if row_vals_mat.ndim == 1:
                row_vals_mat = row_vals_mat[np.newaxis, :]
            if abs(pivot_vals[0]) < 1e4 * vscale * tol:
                strike += 1

        if grid_x > factor * (max_rank - 1) + 1 or grid_y > factor * (max_rank - 1) + 1:
            warnings.warn(
                "SeparableApprox.from_function: function appears to be high rank. "
                "Returning best approximation found.",
                RuntimeWarning,
                stacklevel=2,
            )
            failure = True

        # Handle zero function
        if len(pivot_vals) == 1 and pivot_vals[0] == 0.0:
            zero_col = Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))
            zero_row = Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))
            return cls(
                cols=[zero_col],
                rows=[zero_row],
                # d_j=1/inf -> 0 effectively; cols/rows are zero
                pivots=jnp.array([1.0], dtype=jnp.float64),
                domain=(xa, xb, ya, yb),
            )

        r = len(pivot_vals)

        # Physical pivot locations
        piv_x_phys = np.array([x_pts[pivot_pos[j, 1]] for j in range(r)])
        piv_y_phys = np.array([y_pts[pivot_pos[j, 0]] for j in range(r)])

        # ================================================================
        # PHASE 2: Resolve column and row slices adaptively
        # ================================================================

        ny = grid_y
        nx = grid_x

        resolved_cols = _is_happy(np.sum(col_vals_mat, axis=1), abs_tol)
        resolved_rows = _is_happy(np.sum(row_vals_mat, axis=0), abs_tol)
        is_happy = resolved_cols and resolved_rows

        while not is_happy and not failure:
            if not resolved_cols:
                ny = _grid_refine(ny)
                if ny > max_samples:
                    warnings.warn(
                        "SeparableApprox.from_function: column slices not resolved "
                        f"with {ny} points. Stopping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failure = True
                    break
                # Resample columns at new y-resolution
                col_vals_new = np.zeros((ny, r))
                for j in range(r):
                    col_vals_new[:, j] = _sample_col(float(piv_x_phys[j]), ny)
            else:
                col_vals_new = np.zeros((ny, r))
                for j in range(r):
                    col_vals_new[:, j] = _sample_col(float(piv_x_phys[j]), ny)

            if not resolved_rows:
                nx = _grid_refine(nx)
                if nx > max_samples:
                    warnings.warn(
                        "SeparableApprox.from_function: row slices not resolved "
                        f"with {nx} points. Stopping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failure = True
                    break
                row_vals_new = np.zeros((r, nx))
                for j in range(r):
                    row_vals_new[j, :] = _sample_row(float(piv_y_phys[j]), nx)
            else:
                row_vals_new = np.zeros((r, nx))
                for j in range(r):
                    row_vals_new[j, :] = _sample_row(float(piv_y_phys[j]), nx)

            # Apply GE updates on the skeleton
            pivot_pos_local = np.zeros((r, 2), dtype=int)
            y_pts_new = np.array(_chebpts_phys(ny, ya, yb))
            x_pts_new = np.array(_chebpts_phys(nx, xa, xb))
            for j in range(r):
                # Find nearest grid point to the pivot location
                pivot_pos_local[j, 0] = int(np.argmin(np.abs(y_pts_new - piv_y_phys[j])))
                pivot_pos_local[j, 1] = int(np.argmin(np.abs(x_pts_new - piv_x_phys[j])))

            col_vals_mat, row_vals_mat = _ge_on_skeleton(
                col_vals_new, row_vals_new, pivot_vals, pivot_pos_local
            )

            if r == 1:
                row_vals_mat = row_vals_mat.reshape(1, -1)

            resolved_cols = _is_happy(np.sum(col_vals_mat, axis=1), abs_tol)
            resolved_rows = _is_happy(np.sum(row_vals_mat, axis=0), abs_tol)
            is_happy = resolved_cols and resolved_rows

        # ================================================================
        # Build Chebtech2 objects for each column and row slice
        # ================================================================

        from chebfunjax.utils.transforms import vals2coeffs

        cols_list = []
        rows_list = []

        for j in range(r):
            # Column slice c_j(y): values on reference [-1,1] = values on [ya,yb]
            col_v = jnp.asarray(col_vals_mat[:, j], dtype=jnp.float64)
            col_c = vals2coeffs(col_v)
            # Chop coefficients
            col_vscale = float(jnp.max(jnp.abs(col_v)))
            if col_vscale > 0:
                col_rel_tol = max(tol * col_vscale / col_vscale, tol)
                col_cutoff = standard_chop(col_c, col_rel_tol)
                col_c = col_c[:col_cutoff]
            cols_list.append(Chebtech2.from_coeffs(col_c))

            # Row slice r_j(x): values on reference [-1,1] = values on [xa,xb]
            row_v = jnp.asarray(row_vals_mat[j, :], dtype=jnp.float64)
            row_c = vals2coeffs(row_v)
            row_vscale = float(jnp.max(jnp.abs(row_v)))
            if row_vscale > 0:
                row_rel_tol = max(tol * row_vscale / row_vscale, tol)
                row_cutoff = standard_chop(row_c, row_rel_tol)
                row_c = row_c[:row_cutoff]
            rows_list.append(Chebtech2.from_coeffs(row_c))

        # Store d_j = 1/piv_j so that f(x,y) = Σ_j d_j * c_j(y) * r_j(x)
        # (This matches the CDR formula: C * diag(1/pivotValues) * R.')
        d_arr = jnp.asarray(1.0 / pivot_vals, dtype=jnp.float64)

        return cls(
            cols=cols_list,
            rows=rows_list,
            pivots=d_arr,
            domain=(xa, xb, ya, yb),
        )

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Evaluate f(x, y) using the low-rank representation.

        Computes  Σ_j d_j * c_j(y) * r_j(x)

        where each c_j and r_j is evaluated via Clenshaw's algorithm after
        mapping (x, y) from the physical domain [xa, xb] x [ya, yb] to
        the reference interval [-1, 1].

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb]. Must broadcast with x.

        Returns
        -------
        jax.Array, same shape as broadcast(x, y)
            Approximated function values.

        Notes
        -----
        This method is JIT-safe, vmap-safe, and grad-safe (gradients with
        respect to x and y are available via automatic differentiation).

        Provenance
        ----------
        MATLAB source : @separableApprox/feval.m, @chebfun2/feval.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb = self.domain
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        # Map to reference interval
        tx = _phys_to_ref(x, xa, xb)
        ty = _phys_to_ref(y, ya, yb)

        # Accumulate: result = Σ_j  d_j * c_j(ty) * r_j(tx)
        result = jnp.zeros_like(jnp.broadcast_arrays(x, y)[0], dtype=jnp.float64)
        for j in range(len(self.cols)):
            cj_val = self.cols[j](ty)
            rj_val = self.rows[j](tx)
            result = result + self.pivots[j] * cj_val * rj_val

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Number of terms in the low-rank decomposition."""
        return len(self.cols)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, dim: int = 1, k: int = 1) -> "SeparableApprox":
        """Partial derivative of the approximation.

        Parameters
        ----------
        dim : int, default 1
            Dimension: 1 = y-direction, 2 = x-direction.
        k : int, default 1
            Differentiation order.

        Returns
        -------
        SeparableApprox

        Notes
        -----
        Differentiates each col or row slice independently with the
        chain-rule factor (2/(b-a))^k for the affine map.

        Provenance
        ----------
        MATLAB source : @separableApprox/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if dim not in (1, 2):
            raise ValueError(
                f"SeparableApprox.diff: dim must be 1 (y) or 2 (x), got dim={dim}."
            )
        if k < 0:
            raise ValueError(
                f"SeparableApprox.diff: k must be >= 0, got k={k}."
            )
        if k == 0:
            return self

        xa, xb, ya, yb = self.domain

        def _diff_physical(tech: Chebtech2, a: float, b: float, order: int) -> Chebtech2:
            scale = jnp.float64((2.0 / (b - a)) ** order)
            diff_tech = tech.diff(order)
            return Chebtech2.from_coeffs(diff_tech.coeffs * scale)

        if dim == 1:
            new_cols = [_diff_physical(c, ya, yb, k) for c in self.cols]
            new_rows = list(self.rows)
        else:
            new_cols = list(self.cols)
            new_rows = [_diff_physical(r, xa, xb, k) for r in self.rows]

        return SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=self.pivots,
            domain=self.domain,
        )

    def sum(self, dim: int | None = None):
        """Integrate the approximation over one or both dimensions.

        Parameters
        ----------
        dim : int or None, optional
            - None: double integral (scalar).
            - 1: integrate over y, return SeparableApprox (function of x).
            - 2: integrate over x, return SeparableApprox (function of y).

        Returns
        -------
        SeparableApprox or jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @separableApprox/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if dim is None:
            return self.sum2()
        if dim not in (1, 2):
            raise ValueError(
                f"SeparableApprox.sum: dim must be None, 1, or 2, got dim={dim}."
            )

        xa, xb, ya, yb = self.domain
        r = self.rank

        if dim == 1:
            col_integrals = jnp.array(
                [float(c.sum() * jnp.float64((yb - ya) / 2.0)) for c in self.cols],
                dtype=jnp.float64,
            )
            new_pivots = self.pivots * col_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]
            new_rows = list(self.rows)
        else:
            row_integrals = jnp.array(
                [float(rw.sum() * jnp.float64((xb - xa) / 2.0)) for rw in self.rows],
                dtype=jnp.float64,
            )
            new_pivots = self.pivots * row_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = list(self.cols)
            new_rows = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]

        return SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=new_pivots,
            domain=self.domain,
        )

    def sum2(self) -> jax.Array:
        """Double integral over the domain.

        Returns
        -------
        jax.Array, scalar

        Provenance
        ----------
        MATLAB source : @separableApprox/sum2.m, @separableApprox/integral2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        xa, xb, ya, yb = self.domain
        r = self.rank
        total = jnp.float64(0.0)
        for j in range(r):
            col_int = self.cols[j].sum() * jnp.float64((yb - ya) / 2.0)
            row_int = self.rows[j].sum() * jnp.float64((xb - xa) / 2.0)
            total = total + self.pivots[j] * col_int * row_int
        return total

    def norm(self, p=2) -> jax.Array:
        """Frobenius (L2) norm of the approximation.

        Parameters
        ----------
        p : int or str, default 2
            Only ``2`` and ``'fro'`` are supported.

        Returns
        -------
        jax.Array, scalar

        Provenance
        ----------
        MATLAB source : @separableApprox/norm.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if p not in (2, "fro", 2.0):
            raise NotImplementedError(
                f"SeparableApprox.norm: only p=2/'fro' is implemented, got p={p!r}."
            )
        xa, xb, ya, yb = self.domain
        r = self.rank
        col_scale = jnp.float64((yb - ya) / 2.0)
        row_scale = jnp.float64((xb - xa) / 2.0)

        norm_sq = jnp.float64(0.0)
        for j in range(r):
            for k in range(r):
                col_ip = self.cols[j].inner(self.cols[k]) * col_scale
                row_ip = self.rows[j].inner(self.rows[k]) * row_scale
                norm_sq = norm_sq + self.pivots[j] * self.pivots[k] * col_ip * row_ip

        return jnp.sqrt(jnp.abs(norm_sq))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display like Chebfun2.

        Examples
        --------
        >>> f = SeparableApprox.from_function(lambda x, y: jnp.cos(x + y))
        >>> repr(f)
        'SeparableApprox(rank=2, domain=(-1.0, 1.0, -1.0, 1.0))'
        """
        xa, xb, ya, yb = self.domain
        return (
            f"SeparableApprox(rank={self.rank}, "
            f"domain=({xa}, {xb}, {ya}, {yb}))"
        )
