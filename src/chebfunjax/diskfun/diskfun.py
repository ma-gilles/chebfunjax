# uses-numpy: adaptive 2D construction uses numpy for pivot selection (not JIT-safe)
"""Diskfun — low-rank approximation of functions on the unit disk.

Represents a real-valued function f(theta, r) on the unit disk
(theta in [-pi, pi], r in [0, 1]) as a sum of rank-1 outer products:

    f(theta, r) ≈ Σ_j  (1/d_j) * c_j(r) * row_j(theta)

where:
  - c_j are column slices (Chebtech2 in r, on [-1, 1] mapped to [0, 1]),
  - row_j are row slices (Trigtech in theta, periodic on [-pi, pi]),
  - d_j are scalar pivot values.

The construction uses the BMC-II (block mirror-centrosymmetric) structure
of functions on the disk in doubled-up polar coordinates.  In this
coordinate system, a function f(theta, r) is extended to a periodic
function on [-pi, pi] x [-1, 1] via

    F(theta, r) = f(theta, |r|)  (even in r)

which enables a Fourier × Chebyshev spectral representation.

Algorithm: GE with 2x2 block pivoting on the doubled-up function matrix.
Described in:
  A. Townsend, H. Wilber, and G. Wright, "Computing with functions on
  spherical and polar geometries II: The disk", SIAM J. Sci. Comput.,
  39(5), C238–C262, 2017.

Translated from MATLAB Chebfun class @diskfun (commit 7574c77).
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
from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs
from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import vals2coeffs

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Grid helpers (matching MATLAB getPoints for diskfun)
# ============================================================================


def _disk_col_pts(m: int) -> np.ndarray:
    """Chebyshev-2 points for the r direction (including origin).

    Returns m+1 points on [0, 1]: the upper half of the 2m+1 doubled-up
    Chebyshev grid on [-1, 1].  The first point is r=0 (the origin/pole).

    Matches MATLAB: y = chebpts(2*m+1, [-1, 1]); y = y((2*m)/2+1:end)
    which selects indices (m+1) through (2m+1), i.e. [0, ..., 1].

    Provenance
    ----------
    MATLAB source : @diskfun/constructor.m  (getPoints subfunction)
    Chebfun commit: 7574c77
    """
    pts_full = np.array(chebpts(2 * m + 1, kind=2))  # 2m+1 points on [-1, 1]
    # MATLAB: y = y((2*m)/2+1:end)  — picks from index m (0-based) onwards
    return pts_full[m:]  # shape (m+1,) from 0 to 1


def _disk_row_pts(n: int) -> np.ndarray:
    """Equispaced trigonometric points for theta on [-pi, pi).

    Returns 2n equispaced points: trigpts(2n) scaled to [-pi, pi).

    Matches MATLAB: x = trigpts(2*n, [-pi, pi])

    Provenance
    ----------
    MATLAB source : @diskfun/constructor.m  (getPoints subfunction)
    Chebfun commit: 7574c77
    """
    return np.linspace(-np.pi, np.pi, 2 * n, endpoint=False, dtype=np.float64)


# ============================================================================
# Tolerance helper
# ============================================================================


def _get_tol(F: np.ndarray, hx: float, hy: float, pseudo_level: float) -> tuple[float, float]:
    """Compute construction tolerance for diskfun/spherefun.

    Provenance
    ----------
    MATLAB source : @diskfun/constructor.m  (getTol subfunction)
    Chebfun commit: 7574c77
    """
    m, n = F.shape
    grid = max(m, n)
    dfdx = np.diff(F[: m - 1, :], axis=1) / hx
    dfdy = np.diff(F[:, : n - 1], axis=0) / hy
    jac_norm = np.max(np.maximum(np.abs(dfdx.ravel()), np.abs(dfdy.ravel())))
    vscale = float(np.max(np.abs(F)))
    dom_scale = np.pi  # max of |dom|; domain is [-pi,pi] x [0,1]
    tol = (grid ** (2.0 / 3.0)) * dom_scale * max(jac_norm, vscale) * pseudo_level
    return tol, vscale


# ============================================================================
# Phase 1: GE with 2x2 block pivoting on the doubled-up function matrix
# ============================================================================


def _phase_one_disk(
    F: np.ndarray,
    tol: float,
    alpha: float,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """GE with 2x2 block pivoting to find pivot locations and rank.

    Operates on the doubled-up function matrix of size (m, 2n):
      - F[:, :n]   = f(theta_j, r_i)          — the original block
      - F[:, n:2n] = f(theta_j + pi, r_i)      — the pi-shifted block

    Splits into Fp = 0.5*(B + C) and Fm = 0.5*(B - C) where
      C = F[:, :n] and B = F[:, n:2n].

    Parameters
    ----------
    F : np.ndarray, shape (m, 2n)
        Doubled-up function values; rows are r-points, cols are theta-points.
        Row 0 is r=0 (the origin).
    tol : float
        Construction tolerance.
    alpha : float
        Coupling parameter: do 2x2 update when max/min pivot <= alpha.
    factor : float
        Rank bound = min(m, n) / factor.  If 0, no bound.

    Returns
    -------
    pivot_indices : np.ndarray, shape (rk, 2)  [0-based]
        (row_idx, col_idx) pairs in the reduced (m-1) x n space.
        Row indices are 0-based into the Fp/Fm matrices (without the origin row).
        These will be adjusted to account for the origin row in the full grid.
    pivot_array : np.ndarray, shape (rk, 2)
        (evp, evm) pivot pairs for each GE step.
    remove_poles : bool
        True if the origin row required a pole removal step.
    is_happy : bool
        True if GE converged below tol.

    Provenance
    ----------
    MATLAB source : @diskfun/constructor.m  (PhaseOne subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Townsend, Wilber, Wright, SISC 39(5) 2017.
    """
    m, n2 = F.shape
    n = n2 // 2
    minsize = min(m, n)
    width = minsize / factor if factor > 0 else np.inf

    # Split into plus/minus blocks
    C = F[:, :n]  # original block
    B = F[:, n:]  # pi-shifted block
    Fp = 0.5 * (B + C)
    Fm = 0.5 * (B - C)

    # Check pole (origin row r=0): is Fp[0, :] approximately constant?
    pole_val = float(np.mean(Fp[0, :]))
    remove_poles = abs(pole_val) > tol

    pivot_indices = []
    pivot_array = []
    rank_count = 0
    pole_col = 0

    if remove_poles:
        # Remove the pole: zero out origin row using column of max inf-norm
        pole_col = int(np.argmax(np.max(np.abs(Fp), axis=0)))
        row_val = float(np.max(np.abs(Fp[:, pole_col])))
        row_pole = row_val * np.ones((1, n))
        col_pole = Fp[:, pole_col].copy()
        Fp = Fp - np.outer(col_pole, row_pole[0] / row_val)
        rank_count += 1

    # Remove origin row before rank determination
    Fp = Fp[1:, :]
    Fm = Fm[1:, :]

    maxp_val = float(np.max(np.abs(Fp))) if Fp.size > 0 else 0.0
    maxm_val = float(np.max(np.abs(Fm))) if Fm.size > 0 else 0.0

    # Zero function
    if maxp_val == 0.0 and maxm_val == 0.0 and not remove_poles:
        pivot_indices = np.array([[0, 0]], dtype=int)
        pivot_array = np.array([[0.0, 0.0]])
        return pivot_indices, pivot_array, remove_poles, True

    idxp = int(np.argmax(np.abs(Fp))) if maxp_val > 0 else 0
    idxm = int(np.argmax(np.abs(Fm))) if maxm_val > 0 else 0

    while (max(maxp_val, maxm_val) > tol) and (rank_count < width) and (rank_count < minsize):
        # Choose pivot: whichever of Fp, Fm has larger max
        if maxp_val >= maxm_val:
            idx = idxp
        else:
            idx = idxm

        # Convert flat index to (row, col). numpy argmax flattens in
        # C (row-major) order — j = idx // ncols, k = idx % ncols. The
        # previous column-major (MATLAB-order) conversion put the "pivot"
        # at a scrambled location, so GE eliminated around tiny or zero
        # values: divide-by-zero NaNs and rank frozen at 4 regardless of
        # grid (exp(x) built as rank 3 with 3% value error vs MATLAB 13).
        j, k = divmod(idx, Fp.shape[1])

        evp = float(Fp[j, k])
        evm = float(Fm[j, k])
        absevp = abs(evp)
        absevm = abs(evm)

        pivot_indices.append([j, k])

        # Choose 2x2 or 1x1 update
        if max(absevp, absevm) <= alpha * min(absevp, absevm):
            # Rank-2 update (both pivots used)
            cp = Fp[:, k].copy()
            rp = Fp[j, :].copy()
            cm = Fm[:, k].copy()
            rm = Fm[j, :].copy()
            Fp = Fp - np.outer(cp, rp) / evp
            Fm = Fm - np.outer(cm, rm) / evm
            pivot_array.append([evp, evm])
            rank_count += 2
        else:
            if absevp > absevm:
                # Only plus pivot
                cp = Fp[:, k].copy()
                rp = Fp[j, :].copy()
                Fp = Fp - np.outer(cp, rp) / evp
                evm = 0.0
                rank_count += 1
            else:
                # Only minus pivot
                cm = Fm[:, k].copy()
                rm = Fm[j, :].copy()
                Fm = Fm - np.outer(cm, rm) / evm
                evp = 0.0
                rank_count += 1
            pivot_array.append([evp, evm])

        maxp_val = float(np.max(np.abs(Fp))) if Fp.size > 0 else 0.0
        maxm_val = float(np.max(np.abs(Fm))) if Fm.size > 0 else 0.0
        idxp = int(np.argmax(np.abs(Fp))) if maxp_val > 0 else 0
        idxm = int(np.argmax(np.abs(Fm))) if maxm_val > 0 else 0

    is_happy = max(maxp_val, maxm_val) <= tol

    if len(pivot_indices) == 0:
        pivot_indices = np.array([[0, 0]], dtype=int)
        pivot_array = np.array([[0.0, 0.0]])
    else:
        pivot_indices = np.array(pivot_indices, dtype=int)
        pivot_array = np.array(pivot_array)

    # Adjust pivot indices: add 1 to row indices to account for removed origin
    pivot_indices[:, 0] += 1

    # Prepend pole pivot if needed
    if remove_poles:
        pivot_indices = np.vstack([[0, pole_col], pivot_indices])
        pivot_array = np.vstack([[pole_val, 0.0], pivot_array])

    return pivot_indices, pivot_array, remove_poles, is_happy


# ============================================================================
# Phase 2: Resolve column and row slices adaptively
# ============================================================================


def _phase_two_disk(
    f: Callable,
    pivot_indices: np.ndarray,
    pivot_array: np.ndarray,
    n: int,
    m: int,
    vscale: float,
    max_sample: int,
    remove_poles: bool,
    tol: float,
) -> tuple[list, list, np.ndarray, list, list]:
    """Resolve column (Chebtech2) and row (Trigtech) slices adaptively.

    Evaluates f along the skeleton slices at increasing resolution until the
    Chebyshev/Fourier coefficients are resolved.  Uses the same 2x2 GE
    elimination as Phase 1 on the skeleton.

    Parameters
    ----------
    f : callable
        f(theta, r) -> float, vectorised over theta and r arrays.
    pivot_indices : np.ndarray, shape (rk, 2)
        0-based (row_idx_in_col_pts, col_idx_in_row_pts) pivot locations.
    pivot_array : np.ndarray, shape (rk, 2)
        (evp, evm) pairs from Phase 1.
    n : int
        Initial grid size (for row direction).
    m : int
        Initial grid size (for column direction).
    vscale : float
        Value scale estimate.
    max_sample : int
        Maximum allowed grid size.
    remove_poles : bool
        Whether to handle the origin pole.
    tol : float
        Construction tolerance.

    Returns
    -------
    cols_list : list of Chebtech2
        Column slices (in r on [0, 1]), one per total rank.
    rows_list : list of Trigtech
        Row slices (in theta on [-pi, pi]), one per total rank.
    pivots : np.ndarray, shape (total_rank,)
        Pivot values (1/d_j).
    idx_plus : list of int
        0-based indices into the full list that are "plus" terms.
    idx_minus : list of int
        0-based indices into the full list that are "minus" terms.

    Provenance
    ----------
    MATLAB source : @diskfun/constructor.m  (PhaseTwo subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    rk = pivot_indices.shape[0]
    id_rows = pivot_indices[:, 0]  # row indices into the r-grid
    id_cols = pivot_indices[:, 1]  # col indices into the theta-grid

    # Physical pivot locations (using initial grid)
    r_pts_init = _disk_col_pts(m)
    th_pts_init = _disk_row_pts(n)
    row_pivots = r_pts_init[id_rows]
    col_pivots = th_pts_init[id_cols]

    happy_cols = False
    happy_rows = False
    failure = False

    # Track how pivot grid indices scale when we refine
    id_rows_cur = id_rows.copy()
    id_cols_cur = id_cols.copy()

    m_cur = m
    n_cur = n

    cols_plus = None
    cols_minus = None
    rows_plus = None
    rows_minus = None
    idx_plus_raw = []
    idx_minus_raw = []
    pivots_raw = np.array([])

    # Count non-zero pivot components
    n_pos = int(np.sum(np.abs(pivot_array[:, 0]) > 0))
    n_neg = int(np.sum(np.abs(pivot_array[:, 1]) > 0))
    total_rank = n_pos + n_neg

    if total_rank == 0:
        # Zero function
        zero_col = Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))
        zero_row = Trigtech.from_coeffs(jnp.zeros(1, dtype=jnp.complex128))
        return [zero_col], [zero_row], np.array([1.0]), [0], []

    while not (happy_cols and happy_rows) and not failure:
        r_pts = _disk_col_pts(m_cur)
        th_pts = _disk_row_pts(n_cur)

        # Sample columns at col_pivots (theta values), over all r_pts
        # newCols: evaluate at theta = col_pivots + pi (shifted) and theta = col_pivots
        # Shape: (m_cur+1, rk)
        new_cols_shifted = np.zeros((m_cur + 1, rk))
        new_cols_unshifted = np.zeros((m_cur + 1, rk))
        for jj in range(rk):
            th_val = col_pivots[jj]
            new_cols_shifted[:, jj] = np.array(
                f(
                    jnp.full(m_cur + 1, th_val + np.pi, dtype=jnp.float64),
                    jnp.asarray(r_pts, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )
            new_cols_unshifted[:, jj] = np.array(
                f(
                    jnp.full(m_cur + 1, th_val, dtype=jnp.float64),
                    jnp.asarray(r_pts, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )

        new_cols_plus = 0.5 * (new_cols_shifted + new_cols_unshifted)
        new_cols_minus = 0.5 * (new_cols_shifted - new_cols_unshifted)

        # Sample rows at row_pivots (r values), over all theta_pts (doubled up: 2*n)
        # Shape: (rk, 2*n_cur)
        new_rows = np.zeros((rk, 2 * n_cur))
        for ii in range(rk):
            r_val = row_pivots[ii]
            new_rows[ii, :] = np.array(
                f(
                    jnp.asarray(th_pts, dtype=jnp.float64),
                    jnp.full(2 * n_cur, r_val, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )

        # Split rows into plus/minus: first n_cur = theta, second n_cur = theta+pi
        new_rows_plus = 0.5 * (new_rows[:, :n_cur] + new_rows[:, n_cur:])
        new_rows_minus = 0.5 * (-new_rows[:, :n_cur] + new_rows[:, n_cur:])

        # Initialize storage
        cols_plus_cur = np.zeros((m_cur + 1, n_pos))
        cols_minus_cur = np.zeros((m_cur + 1, n_neg))
        rows_plus_cur = np.zeros((n_pos, n_cur))
        rows_minus_cur = np.zeros((n_neg, n_cur))

        plus_count = 0
        minus_count = 0
        pivot_count = 0

        idx_plus_raw = []
        idx_minus_raw = []
        pivots_raw = np.zeros(total_rank)

        # Handle pole removal: fix the first row of new_rows_plus
        if remove_poles:
            new_rows_plus[0, :] = pivot_array[0, 0]

        # GE skeleton elimination
        for ii in range(rk):
            evp = float(pivot_array[ii, 0])
            evm = float(pivot_array[ii, 1])

            if evp != 0.0 and evm != 0.0:
                cp = new_cols_plus[:, ii].copy()
                rp = new_rows_plus[ii, :].copy()
                cm = new_cols_minus[:, ii].copy()
                rm = new_rows_minus[ii, :].copy()

                cols_plus_cur[:, plus_count] = cp
                rows_plus_cur[plus_count, :] = rp
                cols_minus_cur[:, minus_count] = cm
                rows_minus_cur[minus_count, :] = rm

                new_cols_plus -= np.outer(cp, rp[id_cols_cur]) / evp
                new_rows_plus -= np.outer(cp[id_rows_cur] / evp, rp)
                new_cols_minus -= np.outer(cm, rm[id_cols_cur]) / evm
                new_rows_minus -= np.outer(cm[id_rows_cur] / evm, rm)

                if abs(evp) >= abs(evm):
                    idx_plus_raw.append(pivot_count)
                    idx_minus_raw.append(pivot_count + 1)
                    pivots_raw[pivot_count] = evp
                    pivots_raw[pivot_count + 1] = evm
                else:
                    idx_minus_raw.append(pivot_count)
                    idx_plus_raw.append(pivot_count + 1)
                    pivots_raw[pivot_count] = evm
                    pivots_raw[pivot_count + 1] = evp

                plus_count += 1
                minus_count += 1
                pivot_count += 2

            elif evp != 0.0:
                cp = new_cols_plus[:, ii].copy()
                rp = new_rows_plus[ii, :].copy()
                cols_plus_cur[:, plus_count] = cp
                rows_plus_cur[plus_count, :] = rp

                new_cols_plus -= np.outer(cp, rp[id_cols_cur]) / evp
                new_rows_plus -= np.outer(cp[id_rows_cur] / evp, rp)

                idx_plus_raw.append(pivot_count)
                pivots_raw[pivot_count] = evp
                plus_count += 1
                pivot_count += 1

            elif evm != 0.0:
                cm = new_cols_minus[:, ii].copy()
                rm = new_rows_minus[ii, :].copy()
                cols_minus_cur[:, minus_count] = cm
                rows_minus_cur[minus_count, :] = rm

                new_cols_minus -= np.outer(cm, rm[id_cols_cur]) / evm
                new_rows_minus -= np.outer(cm[id_rows_cur] / evm, rm)

                idx_minus_raw.append(pivot_count)
                pivots_raw[pivot_count] = evm
                minus_count += 1
                pivot_count += 1

        # Enforce boundary conditions
        if remove_poles:
            if plus_count > 1:
                cols_plus_cur[0, 1:plus_count] = 0.0
        elif plus_count > 0:
            cols_plus_cur[0, :plus_count] = 0.0

        if minus_count > 0:
            cols_minus_cur[0, :minus_count] = 0.0

        # Trim to used
        cols_plus_cur = cols_plus_cur[:, :plus_count]
        cols_minus_cur = cols_minus_cur[:, :minus_count]
        rows_plus_cur = rows_plus_cur[:plus_count, :]
        rows_minus_cur = rows_minus_cur[:minus_count, :]
        pivots_raw = pivots_raw[:pivot_count]

        cols_plus = cols_plus_cur
        cols_minus = cols_minus_cur
        rows_plus = rows_plus_cur
        rows_minus = rows_minus_cur

        # Happiness check for columns (Chebtech2-style on r)
        temp1 = np.sum(
            np.hstack([cols_plus, cols_minus])
            if (cols_plus.size > 0 and cols_minus.size > 0)
            else (cols_plus if cols_plus.size > 0 else cols_minus),
            axis=1,
        )
        temp2 = np.sum(
            np.hstack([cols_plus, -cols_minus])
            if (cols_plus.size > 0 and cols_minus.size > 0)
            else (cols_plus if cols_plus.size > 0 else -cols_minus),
            axis=1,
        )
        # Doubled-up column values on [-1, 1]: [flipud(temp2); temp1[1:]]
        col_vals_doubled = np.concatenate([temp2[::-1], temp1[1:]])
        happy_cols = _is_happy_cheb(col_vals_doubled, tol)

        # Happiness check for rows (Trigtech-style in theta)
        rp_sum = np.sum(
            np.vstack([rows_plus, rows_minus])
            if (rows_plus.size > 0 and rows_minus.size > 0)
            else (rows_plus if rows_plus.size > 0 else rows_minus),
            axis=0,
        )
        rm_sum = np.sum(
            np.vstack([rows_plus, -rows_minus])
            if (rows_plus.size > 0 and rows_minus.size > 0)
            else (rows_plus if rows_plus.size > 0 else -rows_minus),
            axis=0,
        )
        row_vals_doubled = np.concatenate([rp_sum, rm_sum])
        happy_rows = _is_happy_trig(row_vals_doubled, tol)

        # Adaptively refine
        if not happy_cols:
            m_new = 2 * m_cur
            if m_new + 1 > max_sample:
                warnings.warn(
                    "Diskfun.from_function: column slices not resolved.",
                    RuntimeWarning,
                    stacklevel=5,
                )
                failure = True
                break
            # Update id_rows for finer grid:  ii = 1:2:m+1 -> 0-based: 0,2,...,m
            id_rows_cur = 2 * id_rows_cur  # maps old index i to 2*i in doubled grid
            m_cur = m_new

        if not happy_rows:
            n_new = 2 * n_cur
            if n_new > max_sample:
                warnings.warn(
                    "Diskfun.from_function: row slices not resolved.",
                    RuntimeWarning,
                    stacklevel=5,
                )
                failure = True
                break
            # Update id_cols for finer grid (0-based).
            # MATLAB (1-based): id_cols = 2*id_cols - 1
            # 0-based equivalent: id_cols = 2 * id_cols
            id_cols_cur = 2 * id_cols_cur
            n_cur = n_new

    # Build full doubled-up columns and rows
    # plus terms: [flipud(col); col[1:]]   (even extension)
    # minus terms: [-flipud(col); col[1:]] (odd extension)
    total = len(idx_plus_raw) + len(idx_minus_raw)
    n_col_full = 2 * (cols_plus.shape[0] if cols_plus.size > 0 else cols_minus.shape[0]) - 1
    n_row_full = 2 * (rows_plus.shape[1] if rows_plus.size > 0 else rows_minus.shape[1])

    cols_full = np.zeros((n_col_full, total))
    rows_full = np.zeros((n_row_full, total))

    if cols_plus.size > 0:
        for kk, gidx in enumerate(idx_plus_raw):
            c = cols_plus[:, kk]
            cols_full[:, gidx] = np.concatenate([c[::-1], c[1:]])
    if cols_minus.size > 0:
        for kk, gidx in enumerate(idx_minus_raw):
            c = cols_minus[:, kk]
            cols_full[:, gidx] = np.concatenate([-c[::-1], c[1:]])
    if rows_plus.size > 0:
        for kk, gidx in enumerate(idx_plus_raw):
            r = rows_plus[kk, :]
            rows_full[:, gidx] = np.concatenate([r, r])
    if rows_minus.size > 0:
        for kk, gidx in enumerate(idx_minus_raw):
            r = rows_minus[kk, :]
            rows_full[:, gidx] = np.concatenate([-r, r])

    # Build Chebtech2 (for r on [-1,1] — the doubled domain) and Trigtech objects
    cols_list = []
    rows_list = []

    for j in range(total):
        # Column: values on doubled [-1, 1] Chebyshev grid
        cv = jnp.asarray(cols_full[:, j], dtype=jnp.float64)
        cc = vals2coeffs(cv)
        cv_scale = float(jnp.max(jnp.abs(cv)))
        if cv_scale > 0:
            cutoff = standard_chop(cc, max(_EPS, tol / cv_scale))
            cc = cc[:cutoff]
        cols_list.append(Chebtech2.from_coeffs(cc))

        # Row: values on doubled theta grid (length 2*n_cur) — periodic
        rv = jnp.asarray(rows_full[:, j], dtype=jnp.float64)
        rc = trig_vals2coeffs(rv.astype(jnp.complex128))
        rv_scale = float(jnp.max(jnp.abs(rv)))
        if rv_scale > 0:
            from chebfunjax.tech.trigtech import _chop_cutoff_to_ncoeffs, _trig_abs_coeffs_for_chop

            chop_in = _trig_abs_coeffs_for_chop(rc)
            chop_rel = max(_EPS, tol / rv_scale)
            cutoff_exp = standard_chop(chop_in.astype(jnp.float64), chop_rel)
            n_keep = _chop_cutoff_to_ncoeffs(int(cutoff_exp), rc.shape[0])
            from chebfunjax.tech.trigtech import _trig_prolong_coeffs

            rc = _trig_prolong_coeffs(rc, n_keep)
        rows_list.append(Trigtech.from_coeffs(rc, is_real=True))

    return cols_list, rows_list, pivots_raw, idx_plus_raw, idx_minus_raw


# ============================================================================
# Happiness checks
# ============================================================================


def _is_happy_cheb(values: np.ndarray, tol: float) -> bool:
    """Check if Chebyshev-2 values are resolved."""
    v = jnp.asarray(values, dtype=jnp.float64)
    c = vals2coeffs(v)
    vscale = float(jnp.max(jnp.abs(v)))
    if vscale == 0.0:
        return True
    rel_tol = max(tol / vscale, _EPS)
    cutoff = standard_chop(c, rel_tol)
    return int(cutoff) < c.shape[0]


def _is_happy_trig(values: np.ndarray, tol: float) -> bool:
    """Check if trigonometric values are resolved."""
    v = jnp.asarray(values, dtype=jnp.float64)
    c = trig_vals2coeffs(v.astype(jnp.complex128))
    vscale = float(jnp.max(jnp.abs(v)))
    if vscale == 0.0:
        return True
    from chebfunjax.tech.trigtech import _trig_abs_coeffs_for_chop

    chop_in = _trig_abs_coeffs_for_chop(c)
    rel_tol = max(tol / vscale, _EPS)
    cutoff = standard_chop(chop_in.astype(jnp.float64), rel_tol)
    return int(cutoff) < chop_in.shape[0]


# ============================================================================
# Main class
# ============================================================================


class Diskfun(eqx.Module):
    """Low-rank approximation of a function on the unit disk.

    Represents f(theta, r) ≈ Σ_j (1/d_j) * c_j(r) * row_j(theta), where

    - c_j are column slices (Chebtech2 in r, values on [-1, 1] via even extension),
    - row_j are row slices (Trigtech in theta on [-pi, pi]),
    - d_j are scalar pivot values,
    - idx_plus, idx_minus track the "plus" and "minus" terms in the BMC-II decomposition.

    The plus/minus split comes from the doubled-up polar coordinates:
      Fp(theta, r) = 0.5 * [f(theta + pi, r) + f(theta, r)]  — even in theta shift
      Fm(theta, r) = 0.5 * [f(theta + pi, r) - f(theta, r)]  — odd in theta shift

    Attributes
    ----------
    cols : list of Chebtech2
        Column slices c_j(r).  Coefficients on [-1, 1] (even extension of [0, 1]).
    rows : list of Trigtech
        Row slices row_j(theta).  Periodic on [-pi, pi].
    pivots : jax.Array, shape (r,)
        Pivot values d_j. f(x,y) ≈ Σ_j (1/d_j) * c_j * row_j.
    idx_plus : tuple of int
        Indices (0-based) into cols/rows/pivots for the "plus" terms.
    idx_minus : tuple of int
        Indices (0-based) into cols/rows/pivots for the "minus" terms.

    Notes
    -----
    Construction is NOT JIT-safe (Python loops with data-dependent termination).
    Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @diskfun/diskfun.m, @diskfun/constructor.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend, H. Wilber, and G. Wright, "Computing with
        functions on spherical and polar geometries II: The disk",
        SIAM J. Sci. Comput., 39(5), C238–C262, 2017.

    See Also
    --------
    Spherefun, SeparableApprox
    """

    @classmethod
    def empty(cls) -> "Diskfun":
        """The empty Diskfun (MATLAB diskfun()): no data; isempty() is
        True and operations on it are undefined.

        Provenance
        ----------
        MATLAB source : @diskfun/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Diskfun (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @diskfun/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    cols: list  # list of Chebtech2 (column slices, functions of r)
    rows: list  # list of Trigtech (row slices, functions of theta)
    pivots: jax.Array  # shape (r,), pivot values d_j
    idx_plus: tuple = eqx.field(static=True)  # 0-based indices of plus terms
    idx_minus: tuple = eqx.field(static=True)  # 0-based indices of minus terms

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable,
        tol: float = _EPS,
        max_rank: int = 512,
        max_sample: int = 2**14,
    ) -> "Diskfun":
        """Construct a Diskfun from a callable.

        The function ``f`` should accept (theta, r) where theta is the
        angle in [-pi, pi] and r is the radius in [0, 1].  Both arguments
        are JAX arrays and the function should be vectorised.

        Internally, ``f`` is extended to the doubled-up polar domain
        [-pi, pi] x [0, 1] and represented using the BMC-II GE algorithm.

        Parameters
        ----------
        f : callable
            f(theta, r) -> array_like.  Must accept 1D JAX arrays for
            both arguments and return an array of the same shape.
            theta is in [-pi, pi], r is in [0, 1].
        tol : float, optional
            Target relative tolerance. Default is machine epsilon.
        max_rank : int, optional
            Maximum allowed rank. Default 512.
        max_sample : int, optional
            Maximum grid size per dimension. Default 2^14.

        Returns
        -------
        Diskfun

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @diskfun/constructor.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Townsend, Wilber, Wright, SISC 39(5) 2017.
        """
        alpha = 100.0  # coupling parameter
        min_sample = 4
        factor = 8.0  # rank bound = min(m, n) / factor
        pseudo_level = _EPS

        is_happy = False
        failure = False
        grid = min_sample
        # "0 + noise" strike counter (MATLAB @diskfun/constructor): a tiny
        # dominant pivot must persist for three successive grids before the
        # function is judged to be numerically zero.  See the strike logic
        # below.
        strike = 1

        # Sample on tensor grid and run Phase 1
        while not is_happy and not failure and strike < 3:
            # Double the grid BEFORE sampling, matching MATLAB
            # @diskfun/constructor.  Fix by Claude Opus 4.8: running phase
            # one at the coarse min_sample grid on the first pass lets a
            # sum of terms with different angular orders (e.g.
            # r^2 cos(2t) + r^6 cos(6t)) alias to rank 1 and falsely
            # report happy=True, producing a wrong low-rank diskfun. This
            # is the same coarse-grid false-convergence fixed in the
            # Spherefun constructor. Starting at 2*min_sample exposes the
            # true rank.
            grid = 2 * grid
            r_pts = _disk_col_pts(grid)  # shape (grid+1,)
            th_pts = _disk_row_pts(grid)  # shape (2*grid,)

            # Build doubled-up sample matrix: F[i, j] = f(th_pts[j], r_pts[i])
            # Shape: (grid+1, 2*grid)
            th_j = jnp.asarray(th_pts, dtype=jnp.float64)
            r_i = jnp.asarray(r_pts, dtype=jnp.float64)
            th2d, r2d = jnp.meshgrid(th_j, r_i)  # shapes (grid+1, 2*grid)
            F = np.array(f(th2d, r2d), dtype=np.float64)

            vscale = float(np.max(np.abs(F)))
            if not np.isfinite(vscale):
                raise ValueError(
                    "Diskfun.from_function: function returned Inf or NaN on the initial grid."
                )

            tol_abs, vscale = _get_tol(F, 2.0 * np.pi / (2 * grid), 1.0 / grid, pseudo_level)
            # (no 1e4*eps floor: MATLAB's getTol is used as computed)

            pivot_indices, pivot_array, remove_poles, happy_rank = _phase_one_disk(
                F, tol_abs, alpha, factor
            )

            if grid > factor * (max_rank - 1):
                warnings.warn(
                    "Diskfun.from_function: function appears to be high rank. "
                    "Returning best approximation found.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                failure = True
                break

            # If the function is 0 + noise the dominant pivot is tiny.
            # MATLAB @diskfun/constructor treats this as convergence only
            # after THREE successive strikes -- a single spurious tiny pivot
            # from evaluation noise at a coarse grid must not stop the loop.
            # The previous port declared happiness on the first occurrence,
            # which collapsed genuinely high-rank functions to rank 1 when
            # they were reconstructed from a noisy evaluation (e.g. a
            # diskfun re-approximated from its own values: rounding noise at
            # r = 0 spuriously triggered pole removal, seeding pivot 0 with a
            # ~1e-13 value that tripped this test).  Accumulating strikes
            # instead lets the grid refine until the true rank appears.
            if max(abs(pivot_array[0, 0]), abs(pivot_array[0, 1])) < 1e4 * tol:
                strike += 1

            if happy_rank:
                is_happy = True
            # else: grid is doubled at the top of the next iteration
            # (moved there by Opus 4.8 to avoid coarse-grid
            # false-convergence, matching the Spherefun fix).

        # Phase 2: resolve slices
        cols_list, rows_list, pivots_arr, idx_plus, idx_minus = _phase_two_disk(
            f,
            pivot_indices,
            pivot_array,
            grid,
            grid,
            vscale,
            max_sample,
            remove_poles,
            tol_abs,
        )

        return cls(
            cols=cols_list,
            rows=rows_list,
            pivots=jnp.asarray(pivots_arr, dtype=jnp.float64),
            idx_plus=tuple(idx_plus),
            idx_minus=tuple(idx_minus),
        )

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, theta: jax.Array, r: jax.Array) -> jax.Array:
        """Evaluate the Diskfun at polar coordinates (theta, r).

        Parameters
        ----------
        theta : jax.Array
            Angle(s) in [-pi, pi].
        r : jax.Array
            Radius/radii in [0, 1]. Must broadcast with theta.

        Returns
        -------
        jax.Array
            Function values at (theta, r), same shape as broadcast(theta, r).

        Notes
        -----
        JIT-safe, vmap-safe, grad-safe.

        Provenance
        ----------
        MATLAB source : @diskfun/feval.m
        Chebfun commit: 7574c77
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        r = jnp.asarray(r, dtype=jnp.float64)

        # Columns are Chebtech2 on [-1, 1] (the doubled r domain).
        # The physical radius r is in [0, 1], which is already in the upper half of [-1, 1].
        # So we evaluate the column directly at r (no further mapping needed):
        # r in [0, 1] is passed to Chebtech2.__call__ which expects a value in [-1, 1].
        r_ref = r  # r in [0, 1] subset of [-1, 1]

        # Rows are Trigtech on [-1, 1] corresponding to theta in [-pi, pi].
        # Map theta from [-pi, pi] to [-1, 1] for Trigtech evaluation.
        th_ref = theta / jnp.pi

        result = jnp.zeros_like(jnp.broadcast_arrays(theta, r)[0], dtype=jnp.float64)
        for j in range(len(self.cols)):
            cj_val = self.cols[j](r_ref)
            rj_val = self.rows[j](th_ref)
            result = result + (1.0 / self.pivots[j]) * cj_val * rj_val

        return result

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def sum2(self) -> jax.Array:
        """MATLAB-parity alias for :meth:`sum` (integral over the disk).

        Provenance
        ----------
        MATLAB source : @diskfun/sum2.m
        Chebfun commit: 7574c77
        """
        return self.sum()

    def sum(self, dim: int | None = None):
        """Definite integration of the Diskfun.

        With no ``dim`` (the chebfunjax default), returns the scalar
        integral over the whole unit disk, ``∫∫ f(theta, r) r dr dtheta``
        (identical to :meth:`sum2`).

        With ``dim=1`` integrates only over the radial direction (with the
        disk measure ``r dr`` on ``r`` in ``[0, 1]``) and returns a
        1D :class:`~chebfunjax.chebfun1d.Chebfun` in ``theta`` (a periodic
        trig chebfun on ``[-pi, pi]``).  With ``dim=2`` integrates only over
        the angular direction ``theta`` in ``[-pi, pi]`` and returns a
        1D chebfun in ``r`` on ``[0, 1]``.  This mirrors MATLAB
        ``sum(F, DIM)`` (where ``sum(F)`` defaults to ``sum(F, 1)``); the
        chebfunjax no-argument form keeps its historical "integrate over the
        whole disk" meaning, so pass an explicit ``dim`` for the partial
        integrals.

        Parameters
        ----------
        dim : {None, 1, 2}, optional
            ``None`` -> scalar integral over the disk; ``1`` -> integrate
            over ``r``, returns a chebfun in ``theta``; ``2`` -> integrate
            over ``theta``, returns a chebfun in ``r``.

        Returns
        -------
        jax.Array (scalar) if ``dim is None``, else Chebfun.

        Provenance
        ----------
        MATLAB source : @diskfun/sum.m, @diskfun/sum2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if dim is not None:
            if dim not in (1, 2):
                raise ValueError("Diskfun.sum: dim must be 1, 2, or None.")
            return self._partial_sum(dim)
        if len(self.idx_plus) == 0:
            return jnp.array(0.0, dtype=jnp.float64)

        result = jnp.array(0.0, dtype=jnp.float64)

        for j in self.idx_plus:
            # Integrate row over [-pi, pi]: ∫_{-pi}^{pi} row_j(theta) d(theta)
            # row_j is stored on [-1, 1] via th_ref = theta/pi; d(theta) = pi * d(th_ref)
            # So integral = pi * ∫_{-1}^{1} row_j(t) dt = pi * 2 * c_0
            row_coeffs = self.rows[j].coeffs
            n_row = row_coeffs.shape[0]
            c0_idx = n_row // 2
            int_row = jnp.pi * 2.0 * jnp.real(row_coeffs[c0_idx])

            # Integrate col * r over [0, 1]:
            # c_j is stored on [-1, 1] (doubled domain), but we only need [0, 1] part.
            # The col is an even function: c_j(r) = c_j(-r).
            # ∫_0^1 c_j(r) * r dr = ∫_0^1 c_j(r) * r dr
            # Map r in [0,1] to t in [-1, 1]: r = (t+1)/2, dr = dt/2, and r_ref = t
            # But the col is defined on [-1,1] and we want ∫_0^1 c(2r-1) * r dr
            # Let t = 2r-1, r = (t+1)/2, dr = dt/2
            # = ∫_{-1}^{1} c(t) * (t+1)/2 * dt/2  = (1/4) ∫_{-1}^{1} c(t) * (t+1) dt
            # = (1/4) [∫c(t)dt + ∫c(t)*t dt]
            # For Chebyshev coefficients: ∫_{-1}^{1} c(t) dt = sum_k a_k * ∫T_k dt
            # ∫_{-1}^1 T_0 dt = 2; ∫_{-1}^1 T_k dt = 2*((-1)^k+1)/(1-k^2) for k>0,k even; 0 odd
            # ∫_{-1}^1 T_k(t)*t dt = ∫_{-1}^1 T_k * T_1 dt (since t=T_1)
            #   = 0 for k>2, = 1 for k=2 (normalization), = 1/2 for k=0...
            # Actually: ∫_{-1}^1 T_j(t)*t dt = 0 for j≠1 and... let's use a direct approach.
            int_col = _integrate_cheb_times_r(self.cols[j].coeffs)

            result = result + (1.0 / self.pivots[j]) * int_col * int_row

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Total number of terms in the low-rank decomposition."""
        return len(self.cols)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Plot this Diskfun on the unit disk (calls :func:`chebfunjax.plotting.plot_disk`)."""
        from chebfunjax.plotting import plot_disk
        return plot_disk(self, **kwargs)

    def surf(self, **kwargs):
        """3D surface plot on the disk (calls :func:`chebfunjax.plotting.surf_disk`)."""
        from chebfunjax.plotting import surf_disk
        return surf_disk(self, **kwargs)

    def contour(self, **kwargs):
        """Contour plot on the disk (calls :func:`chebfunjax.plotting.contour_disk`)."""
        from chebfunjax.plotting import contour_disk
        return contour_disk(self, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic + composition via constructor re-approximation
    # (MATLAB @diskfun semantics; added by Claude Fable 5 -- Diskfun
    # previously had NO arithmetic).
    # ------------------------------------------------------------------

    def norm(self) -> jax.Array:
        """L2 norm over the disk: sqrt(int |f|^2 dA) (Fable 5)."""
        f2 = Diskfun.from_function(lambda t, r: self(t, r) ** 2)
        return jnp.sqrt(jnp.abs(f2.sum()))

    def mean(self) -> jax.Array:
        """Mean value over the disk (integral / pi)."""
        import numpy as _np
        return self.sum() / _np.pi

    def _reapprox(self, op1) -> "Diskfun":
        return Diskfun.from_function(lambda t, r: op1(self(t, r)))

    def _binary(self, other, op2) -> "Diskfun":
        if isinstance(other, Diskfun):
            return Diskfun.from_function(
                lambda t, r: op2(self(t, r), other(t, r)))
        return Diskfun.from_function(
            lambda t, r: op2(self(t, r), other))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __neg__(self):
        return self._reapprox(lambda v: -v)

    def __pow__(self, p):
        return self._reapprox(lambda v: v ** p)

    def compose(self, op):
        """Re-approximate op(f) (MATLAB compose)."""
        return self._reapprox(op)

    def exp(self):
        return self._reapprox(jnp.exp)

    def sin(self):
        return self._reapprox(jnp.sin)

    def cos(self):
        return self._reapprox(jnp.cos)

    def sqrt(self):
        return self._reapprox(jnp.sqrt)

    def __abs__(self):
        return self._reapprox(jnp.abs)

    def abs(self) -> "Diskfun":
        """Absolute value ``|f|`` of the Diskfun.

        Re-approximates ``abs(f)`` on the disk.  Like MATLAB
        ``@diskfun/abs``, this is only well behaved when ``f`` does not pass
        through (or come numerically close to) zero on the interior, where
        ``|f|`` would develop a non-smooth ridge.

        Provenance
        ----------
        MATLAB source : @diskfun/abs.m (separableApprox/abs)
        Chebfun commit: 7574c77
        """
        return self._reapprox(jnp.abs)

    # ------------------------------------------------------------------
    # Reflections and rotation (angular coordinate maps)
    # ------------------------------------------------------------------

    def _remap_theta(self, phi_ref) -> "Diskfun":
        """Return ``g(theta, r) = f(phi(theta), r)`` for an affine angular
        coordinate map (radius unchanged).

        In the chebfunjax low-rank representation
        ``f(theta, r) = sum_j (1/d_j) cols_j(r) rows_j(theta)`` a pure
        angular remap only affects the (1D, periodic) row slices, so each
        row Trigtech is rebuilt from the mapped reference coordinate while
        ``cols``, ``pivots`` and the plus/minus split are reused verbatim.
        Rebuilding the 1D rows (rather than reconstructing the whole 2D
        Diskfun) is exact for these band-limited slices and sidesteps the
        constructor's coarse-grid rank estimation.

        ``phi_ref`` maps the *reference* angular coordinate ``th_ref =
        theta/pi`` (used by :meth:`__call__`); the row Trigtechs are
        periodic, so it may map outside ``[-1, 1]``.
        """
        from chebfunjax.tech.trigtech import Trigtech

        new_rows = []
        for rj in self.rows:
            n_row = int(rj.coeffs.shape[0])
            new_rows.append(
                Trigtech.from_function(
                    lambda t, _rj=rj: _rj(phi_ref(t)), n=n_row))
        return Diskfun(
            cols=self.cols,
            rows=new_rows,
            pivots=self.pivots,
            idx_plus=self.idx_plus,
            idx_minus=self.idx_minus,
        )

    def fliplr(self) -> "Diskfun":
        """Reflect over the y-axis: ``G(x, y) = F(-x, y)``.

        In polar coordinates this maps ``theta -> pi - theta``.

        Provenance
        ----------
        MATLAB source : @diskfun/fliplr.m
        Chebfun commit: 7574c77
        """
        # theta -> pi - theta  ==>  th_ref -> 1 - th_ref
        return self._remap_theta(lambda t: 1.0 - t)

    def flipud(self) -> "Diskfun":
        """Reflect over the x-axis: ``G(x, y) = F(x, -y)``.

        In polar coordinates this maps ``theta -> -theta``.

        Provenance
        ----------
        MATLAB source : @diskfun/flipud.m
        Chebfun commit: 7574c77
        """
        # theta -> -theta  ==>  th_ref -> -th_ref
        return self._remap_theta(lambda t: -t)

    def flipxy(self) -> "Diskfun":
        """Swap the axes: ``G(x, y) = F(y, x)``.

        In polar coordinates this maps ``theta -> pi/2 - theta``.

        Provenance
        ----------
        MATLAB source : @diskfun/flipxy.m
        Chebfun commit: 7574c77
        """
        # theta -> pi/2 - theta  ==>  th_ref -> 1/2 - th_ref
        return self._remap_theta(lambda t: 0.5 - t)

    def flipdim(self, dim: int) -> "Diskfun":
        """Reflect in a chosen direction: ``dim=1`` -> :meth:`flipud`
        (``F(x, -y)``), ``dim=2`` -> :meth:`fliplr` (``F(-x, y)``).

        Provenance
        ----------
        MATLAB source : @diskfun/flipdim.m
        Chebfun commit: 7574c77
        """
        if dim == 1:
            return self.flipud()
        if dim == 2:
            return self.fliplr()
        raise ValueError("Diskfun.flipdim: dim must be 1 or 2.")

    def rotate(self, alpha: float = 0.0) -> "Diskfun":
        """Rotate the Diskfun by ``alpha`` radians about the origin.

        Positive ``alpha`` rotates counter-clockwise: the rotated function
        satisfies ``g(theta, r) = f(theta - alpha, r)``.

        Provenance
        ----------
        MATLAB source : @diskfun/rotate.m, @diskfun/circshift.m
        Chebfun commit: 7574c77
        """
        # theta -> theta - alpha  ==>  th_ref -> th_ref - alpha/pi
        a_ref = float(alpha) / float(np.pi)
        return self._remap_theta(lambda t: t - a_ref)

    def circshift(self, alpha: float = 0.0) -> "Diskfun":
        """Alias of :meth:`rotate` (MATLAB @diskfun/circshift)."""
        return self.rotate(alpha)

    # ------------------------------------------------------------------
    # Partial integration and global optimization
    # ------------------------------------------------------------------

    def _partial_sum(self, dim: int):
        """Partial definite integral (helper for :meth:`sum` with a dim)."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
        from chebfunjax.domain import Domain

        if self.isempty():
            dom = (-np.pi, np.pi) if dim == 1 else (0.0, 1.0)
            return Chebfun(funs=[], domain=Domain(dom))

        nterms = len(self.cols)
        inv_p = [1.0 / self.pivots[j] for j in range(nterms)]

        if dim == 1:
            # Integrate over r with the disk measure -> function of theta.
            wcol = [inv_p[j]
                    * _integrate_cheb_times_r(self.cols[j].coeffs)
                    for j in range(nterms)]

            def g_theta(theta):
                th_ref = jnp.asarray(theta, dtype=jnp.float64) / jnp.pi
                out = jnp.zeros_like(jnp.asarray(theta, dtype=jnp.float64),
                                     dtype=jnp.complex128)
                for j in range(nterms):
                    out = out + wcol[j] * self.rows[j](th_ref)
                return jnp.real(out)

            return chebfun(g_theta, domain=(-np.pi, np.pi), trig=True)

        # dim == 2: integrate over theta -> function of r on [0, 1].
        wrow = []
        for j in range(nterms):
            rc = self.rows[j].coeffs
            c0 = rc.shape[0] // 2
            wrow.append(inv_p[j] * jnp.pi * 2.0 * jnp.real(rc[c0]))

        def g_r(r):
            rr = jnp.asarray(r, dtype=jnp.float64)
            out = jnp.zeros_like(rr)
            for j in range(nterms):
                out = out + wrow[j] * self.cols[j](rr)
            return out

        return chebfun(g_r, domain=(0.0, 1.0))

    def minandmax2(self, n_theta: int = 240, n_r: int = 120):
        """Global minimum and maximum of the Diskfun over the closed disk.

        A tensor grid in ``(theta, r)`` locates candidate extrema, which are
        then polished with a bound-constrained quasi-Newton step
        (``r`` in ``[0, 1]``, ``theta`` free) using exact JAX gradients of
        the evaluation.  Returns ``(Y, X)`` where ``Y = [min_value,
        max_value]`` and ``X`` are the corresponding ``(theta, r)`` points,
        mirroring MATLAB ``[Y, X] = minandmax2(F)``.

        Provenance
        ----------
        MATLAB source : @diskfun/minandmax2.m
        Chebfun commit: 7574c77
        """
        import jax
        from scipy.optimize import minimize

        f = self

        def scal(p):
            return f(p[0], p[1])

        vg = jax.jit(jax.value_and_grad(scal))

        th = np.linspace(-np.pi, np.pi, n_theta)
        rr = np.linspace(0.0, 1.0, n_r)
        TH, RR = np.meshgrid(th, rr)
        F = np.asarray(f(jnp.asarray(TH), jnp.asarray(RR)), dtype=np.float64)

        def polish(idx_flat, sign):
            # sign=+1 minimizes f, sign=-1 maximizes f (minimizes -f).
            i0 = np.unravel_index(idx_flat, F.shape)
            p0 = np.array([TH[i0], RR[i0]], dtype=np.float64)

            def fun(p):
                v, g = vg(jnp.asarray(p, dtype=jnp.float64))
                return sign * float(v), sign * np.asarray(g, dtype=np.float64)

            res = minimize(fun, p0, jac=True, method="L-BFGS-B",
                           bounds=[(None, None), (0.0, 1.0)],
                           options={"ftol": 1e-15, "gtol": 1e-14,
                                    "maxiter": 500})
            return sign * float(res.fun), res.x

        vmin, xmin = polish(int(np.argmin(F)), +1)
        vmax, xmax = polish(int(np.argmax(F)), -1)
        Y = jnp.asarray([vmin, vmax], dtype=jnp.float64)
        X = jnp.asarray([xmin, xmax], dtype=jnp.float64)
        return Y, X

    def max2(self) -> float:
        """Global maximum of the Diskfun over the disk (MATLAB max2)."""
        return float(self.minandmax2()[0][1])

    def min2(self) -> float:
        """Global minimum of the Diskfun over the disk (MATLAB min2)."""
        return float(self.minandmax2()[0][0])

    def diffx(self) -> "Diskfun":
        r"""Cartesian partial derivative :math:`\\partial f / \\partial x`.

        In polar coordinates ``(theta, r)``,

        .. math::
            \\partial_x = \\cos\\theta \\, \\partial_r
                          - \\frac{\\sin\\theta}{r} \\, \\partial_\\theta .

        Computed spectrally (see :func:`_diskfun_reconstruct`).
        Implemented and verified by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @diskfun/diff.m (dim = 1)
        Chebfun commit: 7574c77
        """
        return _diskfun_reconstruct(self, "x")

    def diffy(self) -> "Diskfun":
        r"""Cartesian partial derivative :math:`\\partial f / \\partial y`.

        .. math::
            \\partial_y = \\sin\\theta \\, \\partial_r
                          + \\frac{\\cos\\theta}{r} \\, \\partial_\\theta .

        Implemented and verified by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @diskfun/diff.m (dim = 2)
        Chebfun commit: 7574c77
        """
        return _diskfun_reconstruct(self, "y")

    def diff(self, dim: int = 1, k: int = 1) -> "Diskfun":
        """Cartesian derivative: ``dim=1`` -> x, ``dim=2`` -> y, applied k times."""
        if dim not in (1, 2):
            raise ValueError("dim must be 1 (x) or 2 (y)")
        f = self
        for _ in range(int(k)):
            f = f.diffx() if dim == 1 else f.diffy()
        return f

    def laplacian(self) -> "Diskfun":
        r"""Laplacian :math:`\\nabla^2 f = f_{xx} + f_{yy}` on the disk.

        In polar coordinates,

        .. math::
            \\nabla^2 f = f_{rr} + \\frac{1}{r} f_r
                          + \\frac{1}{r^2} f_{\\theta\\theta} .

        Computed spectrally and reconstructed through a smooth
        Fourier(theta) x Chebyshev(r) modal fit (so the 1/r, 1/r^2
        factors are only ever evaluated at interior nodes, never the
        origin).  Verified against exact harmonic polynomials
        (``Re(z^n) = r^n cos(n theta)`` has zero Laplacian) and against
        MATLAB @diskfun.  Implemented by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @diskfun/laplacian.m
        Chebfun commit: 7574c77
        """
        return _diskfun_reconstruct(self, "laplacian")

    lap = laplacian

    @staticmethod
    def poisson(f, bc=None, m: int = 40) -> "Diskfun":
        r"""Solve the Poisson equation :math:`\\nabla^2 u = f` on the disk.

        Homogeneous Dirichlet boundary condition ``u = 0`` on ``r = 1``.
        Solved spectrally: the right-hand side is expanded in Fourier
        modes in theta, and each mode's radial function is found by a
        Chebyshev collocation solve of

        .. math::
            u_m'' + \\frac{1}{r} u_m' - \\frac{m^2}{r^2} u_m = f_m ,

        with ``u_m(1) = 0`` and the pole regularity condition at the
        origin (``u_m(0) = 0`` for ``m != 0``, ``u_m'(0) = 0`` for
        ``m = 0``).  Implemented and verified by Claude Opus 4.8 against
        manufactured solutions and MATLAB ``diskfun.poisson``.

        Parameters
        ----------
        f : Diskfun or callable
            Right-hand side ``f(theta, r)``.
        m : int, default 40
            Radial Chebyshev resolution.

        Returns
        -------
        Diskfun

        Provenance
        ----------
        MATLAB source : @diskfun/poisson.m (result-equivalent).
        Chebfun commit: 7574c77
        """
        if isinstance(f, Diskfun):
            def fval(t, r):
                return f(t, r)
        else:
            fval = f
        return _diskfun_poisson(fval, int(m), 0.0, bc)

    @staticmethod
    def harmonic(L: int, m: int, bc: str = "dirichlet") -> "Diskfun":
        r"""Cylindrical (disk) harmonic: the L2-normalized eigenfunction
        of the Laplacian on the unit disk (MATLAB diskfun.harmonic),

        .. math::
            Y = c\, J_{|L|}(j_{|L|,m}\, r)
            \begin{cases}\cos(L\theta) & L \ge 0\\
            \sin(|L|\theta) & L < 0\end{cases}

        where ``j`` is the m-th positive root of :math:`J_{|L|}`
        (Dirichlet) or of :math:`J'_{|L|}` (Neumann), and ``c``
        normalizes the disk L2 norm to 1.

        Provenance
        ----------
        MATLAB source : @diskfun/harmonic.m
        Chebfun commit: 7574c77
        """
        from scipy.special import jn_zeros, jnp_zeros, jv
        Labs = abs(int(L))
        if bc.lower().startswith("n"):
            j = float(jnp_zeros(Labs, m)[m - 1])
            nrm = np.sqrt(2.0) / (
                np.sqrt((1.0 - Labs * Labs / (j * j))
                        * (1 + (Labs == 0)) * np.pi)
                * abs(jv(Labs, j)))
        else:
            j = float(jn_zeros(Labs, m)[m - 1])
            nrm = np.sqrt(2.0) / (
                np.sqrt((1 + (Labs == 0)) * np.pi)
                * abs(jv(Labs + 1, j)))

        def f(t, r):
            rad = jnp.asarray(
                jv(Labs, np.asarray(r, dtype=float) * j),
                dtype=jnp.float64)
            ang = jnp.cos(Labs * t) if L >= 0 else jnp.sin(Labs * t)
            return nrm * rad * ang

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return Diskfun.from_function(f)

    @staticmethod
    def helmholtz(f, K: float, bc=None, m: int = 40,
                  n: int | None = None) -> "Diskfun":
        r"""Solve the Helmholtz equation
        :math:`\nabla^2 u + K^2 u = f` on the disk with Dirichlet
        boundary data ``u(1, theta) = bc(theta)`` (MATLAB
        diskfun.helmholtz).

        Parameters
        ----------
        f : Diskfun or callable
            Right-hand side ``f(theta, r)``.
        K : float
            Helmholtz wavenumber (K = 0 reduces to Poisson).
        bc : callable, float, or None
            Dirichlet boundary data as a function of theta
            (None means homogeneous).
        m : int, default 40
            Radial Chebyshev resolution.
        n : int, optional
            Angular resolution (inferred if omitted; accepted for
            MATLAB signature compatibility).

        Provenance
        ----------
        MATLAB source : @diskfun/helmholtz.m (result-equivalent).
        Chebfun commit: 7574c77
        """
        if isinstance(f, Diskfun):
            def fval(t, r):
                return f(t, r)
        else:
            fval = f
        return _diskfun_poisson(fval, int(m), float(K), bc)

    def __repr__(self) -> str:
        """Compact display.

        Provenance
        ----------
        MATLAB source : @diskfun/display.m
        Chebfun commit: 7574c77
        """
        return (
            f"Diskfun(rank={self.rank}, n_plus={len(self.idx_plus)}, n_minus={len(self.idx_minus)})"
        )


# ============================================================================
# Integration helpers
# ============================================================================


def _integrate_cheb_times_r(coeffs: jax.Array) -> jax.Array:
    """Compute ∫_0^1 p(r) * r dr where p is a Chebtech2 on [-1, 1].

    The Chebtech2 represents a function on the doubled-up r-domain [-1, 1].
    The physical radius is r in [0, 1], so we integrate over [0, 1] only.

    Matches MATLAB @diskfun/sum2.m:
        measure = chebfun(@(r) r, [-1, 1]);
        cols = cols .* (measure * ones(1, ...));
        intCols = sum(cols, [0 1]);   % integrate over [0, 1]

    We compute ∫_0^1 c(r) * r dr directly using the Chebyshev coefficient formula:

        ∫_a^b T_k(r) r dr  (integral of T_k times r over [0, 1])

    Using the antiderivative of T_k(r) * r over [0, 1]:
        ∫ T_k(r) r dr can be computed analytically.

    Since the column is an even function (c(r) = c(-r) due to the doubling),
    only even-indexed Chebyshev coefficients are nonzero.  For even k:

        ∫_0^1 T_k(r) r dr

    We use the recurrence-based formula with Gauss-Chebyshev quadrature
    adapted for the half-interval [0, 1].

    For simplicity and correctness, we use the following exact formula:
    Since T_k'(r) = k * U_{k-1}(r) (derivative), and integrating by parts:
        ∫_0^1 T_k(r) r dr = [r T_k(r) / k]_0^1 - ∫_0^1 T_k(r)/k dr  ... messy

    Instead, use the simple exact Chebyshev integral weights over [0, 1]:
        W_k = ∫_0^1 T_k(r) dr  (no r factor), then separately handle r*T_k.

    Expanding directly:
        ∫_0^1 T_k(r) r dr
    uses the substitution r = cos(t), t in [0, pi/2]:
        = ∫_{pi/2}^0 cos(kt) cos(t) (-sin(t)) dt
        = ∫_0^{pi/2} cos(kt) cos(t) sin(t) dt
        = (1/2) ∫_0^{pi/2} cos(kt) sin(2t) dt

    This is computed using the product-to-sum formula:
        cos(kt) sin(2t) = (1/2)[sin((k+2)t) + sin((2-k)t)] for k != 2
        cos(kt) sin(2t) = (1/2)[sin(4t) + sin(0)] = (1/2)sin(4t) for k = 2

    ∫_0^{pi/2} sin(nt) dt = (1 - cos(n*pi/2)) / n  for n != 0
                           = 0                       for n = 0

    This gives exact weights W_k for the integral ∫_0^1 T_k(r) r dr.

    Provenance
    ----------
    Derived from standard Chebyshev integral formulas.
    MATLAB reference: @diskfun/sum2.m (sum over r with Jacobian r).
    Chebfun commit: 7574c77
    """
    n = coeffs.shape[0]
    ks = jnp.arange(n, dtype=jnp.float64)

    # Compute W_k = ∫_0^1 T_k(r) r dr using:
    # W_k = (1/2) * ∫_0^{pi/2} cos(kt) sin(2t) dt
    #      = (1/4) * [∫_0^{pi/2} sin((k+2)t) dt + ∫_0^{pi/2} sin((2-k)t) dt]
    # where sin((2-k)t) for k>2 means sin of negative argument.
    #
    # ∫_0^{pi/2} sin(mt) dt = (1 - cos(m*pi/2)) / m  for m != 0
    #                        = 0                       for m = 0

    def int_sin(m):
        """∫_0^{pi/2} sin(m*t) dt = (1 - cos(m*pi/2)) / m for m != 0, else 0."""
        return jnp.where(
            m == 0,
            0.0,
            (1.0 - jnp.cos(m * jnp.pi / 2.0)) / m,
        )

    m_plus = ks + 2.0  # k + 2
    m_minus = 2.0 - ks  # 2 - k (can be negative for k > 2)

    # For sin((2-k)t) with k > 2: sin(-(k-2)t) = -sin((k-2)t)
    # We compute int_sin(|2-k|) * sign(2-k)
    # Actually int_sin(m) for m < 0: sin(m*t) = -sin(-m*t)
    # => ∫_0^{pi/2} sin(m*t) dt = -∫_0^{pi/2} sin(-m*t) dt = -(1 - cos(-m*pi/2)) / (-m)
    # = (1 - cos(m*pi/2)) / m  (since cos is even)
    # So int_sin(m) = (1 - cos(m*pi/2)) / m also works for m < 0.

    W = 0.25 * (int_sin(m_plus) + int_sin(m_minus))

    # ∫_0^1 c(r) r dr = sum_k a_k * W_k
    return jnp.dot(coeffs.astype(jnp.float64), W)


def _diskfun_reconstruct(f: "Diskfun", kind: str) -> "Diskfun":
    """Spectral Cartesian derivative / Laplacian of a Diskfun (Opus 4.8).

    ``kind`` is ``"x"``, ``"y"`` or ``"laplacian"``.  The relevant
    combination of radial/angular derivatives is evaluated at interior
    Chebyshev(r) x uniform(theta) nodes (never at r = 0, so the 1/r and
    1/r^2 factors stay finite), then fit to a smooth Fourier(theta) x
    Chebyshev(r) modal expansion and reconstructed with
    ``Diskfun.from_function``.

    Verified against exact harmonic polynomials and MATLAB @diskfun.
    """
    cols = f.cols
    rows = f.rows
    piv = np.asarray(f.pivots)
    cols_d = [c.diff() for c in cols]
    cols_dd = [c.diff().diff() for c in cols]
    rows_d = [r.diff() for r in rows]
    rows_dd = [r.diff().diff() for r in rows]
    inv_pi = 1.0 / np.pi

    ncol = max(c.coeffs.shape[0] for c in cols)
    nrow = max(r.coeffs.shape[0] for r in rows)
    nr = ncol + 6
    nth = nrow + 8

    # interior radial nodes: s = 2r-1 at Chebyshev-Gauss points of
    # order nr, so the radial fit below is a well-conditioned square
    # solve (essentially exact) instead of an ill-conditioned lstsq.
    # (The previous |cos| nodes gave the fit ~1e-11 noise, which sent
    # the downstream constructor's GE down a noise-pivot path and
    # produced degenerate results -- the actual root cause of the
    # diffx/diffy/laplacian corruption found in the Fable 5 audit.)
    k = np.arange(nr)
    s_gauss = np.cos(np.pi * (k + 0.5) / nr)
    r_nodes = (s_gauss + 1.0) / 2.0    # in (0, 1), never 0
    # theta samples on [0, 2*pi) so the standard FFT mode convention
    # applies directly (no phase offset); the reconstruction below uses
    # the actual theta, which is periodic, so any eval range is fine.
    th_nodes = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    TH, RR = np.meshgrid(th_nodes, r_nodes, indexing="ij")
    thj = jnp.asarray(TH.ravel())
    rj = jnp.asarray(RR.ravel())
    tr = thj / np.pi

    frr = np.zeros(TH.size)
    fr = np.zeros(TH.size)
    ftt = np.zeros(TH.size)
    fr_only = np.zeros(TH.size)   # d/dr (for x/y)
    fth = np.zeros(TH.size)       # d/dtheta (for x/y)
    for j in range(len(cols)):
        w = float(1.0 / piv[j])
        rowv = np.asarray(jnp.real(rows[j](tr)))
        rowdv = np.asarray(jnp.real(rows_d[j](tr))) * inv_pi
        rowddv = np.asarray(jnp.real(rows_dd[j](tr))) * inv_pi * inv_pi
        colv = np.asarray(jnp.real(cols[j](rj)))
        coldv = np.asarray(jnp.real(cols_d[j](rj)))
        colddv = np.asarray(jnp.real(cols_dd[j](rj)))
        frr += w * colddv * rowv
        fr += w * coldv * rowv
        ftt += w * colv * rowddv
        fr_only += w * coldv * rowv
        fth += w * colv * rowdv

    Rr = np.asarray(RR.ravel())
    if kind == "laplacian":
        V = frr + fr / Rr + ftt / Rr ** 2
    elif kind == "x":
        V = np.cos(TH.ravel()) * fr_only \
            - np.sin(TH.ravel()) / Rr * fth
    elif kind == "y":
        V = np.sin(TH.ravel()) * fr_only \
            + np.cos(TH.ravel()) / Rr * fth
    else:
        raise ValueError(f"unknown kind {kind!r}")
    V = V.reshape(TH.shape)

    # angular FFT -> true complex Fourier coefficients c_m of
    # e^{i m theta} (theta sampled on [0, 2*pi), so no phase offset).
    Fhat = np.fft.rfft(V, axis=0) / nth
    mmax = Fhat.shape[0]
    # radial fit: Chebyshev in s = 2r - 1
    deg = nr - 1
    s_nodes = 2.0 * r_nodes - 1.0
    vand = np.polynomial.chebyshev.chebvander(s_nodes, deg)
    coefs = np.linalg.solve(vand, Fhat.T)  # (deg+1, mmax), exact


    def ev(theta, r):
        theta = np.asarray(theta, dtype=float)
        r = np.asarray(r, dtype=float)
        shape = np.broadcast(theta, r).shape
        theta = np.broadcast_to(theta, shape).copy()
        r = np.broadcast_to(r, shape).copy()
        # Diskfun.from_function samples the DOUBLED disk (r < 0); the
        # radial Chebyshev fit is only valid on r in [0, 1], so map
        # (theta, -r) -> (theta + pi, r) (the BMC identity).  Without
        # this the fit EXTRAPOLATED for r < 0 and the constructor
        # ingested garbage on half the grid -- the root cause of the
        # diffx/diffy/laplacian mode corruption found in the Fable 5
        # audit.
        neg = r < 0
        r = np.abs(r)
        theta = np.where(neg, theta + np.pi, theta)
        s = np.clip(2.0 * r - 1.0, -1.0, 1.0).ravel()
        vd = np.polynomial.chebyshev.chebvander(s, deg)
        modes = vd @ coefs  # (npts, mmax) complex
        out = np.zeros(s.shape)
        th_flat = theta.ravel()
        for m in range(mmax):
            fac = 1.0 if m == 0 else 2.0
            out = out + fac * np.real(modes[:, m] * np.exp(1j * m * th_flat))
        return jnp.asarray(out.reshape(shape), dtype=jnp.float64)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return Diskfun.from_function(lambda t, r: ev(t, r))


def _cheb_diff_matrix(n: int) -> tuple:
    """Chebyshev differentiation matrix on Gauss-Lobatto nodes of [-1,1].

    Returns (D, x) with x descending from 1 to -1.  (Trefethen, ATAP.)
    """
    if n == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def _diskfun_poisson(f, n: int, K: float = 0.0, bc=None) -> "Diskfun":
    """Fast spectral Poisson/Helmholtz solver on the disk.

    See :meth:`Diskfun.poisson` / :meth:`Diskfun.helmholtz`.  Per
    angular Fourier mode, solve the radial ODE by Chebyshev collocation
    with u(1) = bc-mode and pole regularity.  (Poisson by Opus 4.8;
    the K^2 term and non-homogeneous Dirichlet data added in the
    Fable 5 audit.)
    """
    D, x = _cheb_diff_matrix(n)
    r = (x + 1.0) / 2.0            # map [-1,1] -> [0,1]
    Dr = 2.0 * D
    Drr = Dr @ Dr
    nth = 2 * n + 8
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    TH, RR = np.meshgrid(th, r, indexing="ij")
    F = np.asarray(f(jnp.asarray(TH.ravel()), jnp.asarray(RR.ravel()))
                   ).reshape(TH.shape)
    Fhat = np.fft.rfft(F, axis=0) / nth
    mmax = Fhat.shape[0]
    # Dirichlet boundary data, mode by mode
    if bc is None:
        bchat = np.zeros(mmax, dtype=complex)
    elif callable(bc):
        bvals = np.asarray(bc(jnp.asarray(th)), dtype=float)
        bchat = np.fft.rfft(bvals) / nth
    else:
        bchat = np.zeros(mmax, dtype=complex)
        bchat[0] = float(bc)
    Uhat = np.zeros_like(Fhat)
    r_safe = r.copy()
    r_safe[np.abs(r_safe) < 1e-14] = 1e-14
    inv_r = np.diag(1.0 / r_safe)
    inv_r2 = np.diag(1.0 / r_safe ** 2)
    for mm in range(mmax):
        L = Drr + inv_r @ Dr - mm * mm * inv_r2 \
            + (K * K) * np.eye(len(r))
        rhs = Fhat[mm].astype(complex).copy()
        A = L.astype(complex).copy()
        # u(1) = bc_m : r = 1 is x = 1 -> index 0 (x descending)
        A[0, :] = 0.0
        A[0, 0] = 1.0
        rhs[0] = bchat[mm]
        # regularity at r = 0 (index n): u=0 for m!=0, u'=0 for m=0
        if mm == 0:
            A[n, :] = Dr[n, :]
            rhs[n] = 0.0
        else:
            A[n, :] = 0.0
            A[n, n] = 1.0
            rhs[n] = 0.0
        Uhat[mm] = np.linalg.solve(A, rhs)

    deg = n
    vand = np.polynomial.chebyshev.chebvander(x, deg)
    coefs = np.linalg.lstsq(vand, Uhat.T, rcond=None)[0]

    def ev(theta, rr):
        theta = np.asarray(theta)
        rr = np.asarray(rr)
        shape = np.broadcast(theta, rr).shape
        s = (2.0 * rr - 1.0).ravel()
        vd = np.polynomial.chebyshev.chebvander(s, deg)
        modes = vd @ coefs
        out = np.zeros(s.shape)
        thf = np.broadcast_to(theta, shape).ravel()
        for mm in range(mmax):
            fac = 1.0 if mm == 0 else 2.0
            out = out + fac * np.real(modes[:, mm] * np.exp(1j * mm * thf))
        return jnp.asarray(out.reshape(shape), dtype=jnp.float64)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return Diskfun.from_function(lambda t, r: ev(t, r))


from chebfunjax.utils.misc import make_empty_aware  # noqa: E402

make_empty_aware(Diskfun, ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', '__pow__', '__neg__', 'sum', 'sum2', 'mean', 'norm', 'laplacian', 'diffx', 'diffy', 'compose', 'exp', 'sin', 'cos', 'sqrt'])
