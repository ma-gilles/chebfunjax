# uses-numpy: adaptive 2D construction uses numpy for pivot selection (not JIT-safe)
"""Spherefun — low-rank approximation of functions on the unit sphere.

Represents a real-valued function f(lam, theta) on the unit sphere
(lam in [-pi, pi] longitude, theta in [0, pi] colatitude) as a sum of
rank-1 outer products:

    f(lam, theta) ≈ Σ_j (1/d_j) * c_j(theta) * row_j(lam)

where:
  - c_j are column slices (Trigtech in theta, on the doubled domain [-pi, pi]),
  - row_j are row slices (Trigtech in lam, periodic on [-pi, pi]),
  - d_j are scalar pivot values.

The construction uses the BMC-I (block mirror-centrosymmetric) structure
of functions on the sphere in doubled-up colatitude coordinates.  The
function is extended to the doubled domain theta in [-pi, pi] via the
even extension: F(lam, theta) = f(lam, |theta|).

Algorithm: GE with 2x2 block pivoting on the doubled-up function matrix.
Described in:
  A. Townsend, H. Wilber, and G. Wright, "Computing with functions on
  spherical and polar geometries I: The sphere", SIAM J. Sci. Comput.,
  38(4), C403–C425, 2016.

Translated from MATLAB Chebfun class @spherefun (commit 7574c77).
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

from chebfunjax.tech.trigtech import (
    Trigtech,
    _chop_cutoff_to_ncoeffs,
    _trig_abs_coeffs_for_chop,
    _trig_prolong_coeffs,
    trig_vals2coeffs,
)
from chebfunjax.utils.misc import standard_chop

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Grid helpers (matching MATLAB getPoints for spherefun, colatitude domain)
# ============================================================================


def _sphere_col_pts(m: int) -> np.ndarray:
    """Colatitude points theta for the column direction, on [0, pi].

    Returns m+1 points: 0, pi/m, 2pi/m, ..., pi (linspace(0, pi, m+1)).

    Matches MATLAB: y = linspace(0, pi, m+1).'  for the colatitude case.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m  (getPoints subfunction)
    Chebfun commit: 7574c77
    """
    return np.linspace(0.0, np.pi, m + 1, dtype=np.float64)


def _sphere_row_pts(n: int) -> np.ndarray:
    """Longitude points lam for the row direction, on [-pi, pi).

    Returns 2n equispaced points: trigpts(2n) scaled to [-pi, pi).

    Matches MATLAB: x = trigpts(2*n, [-pi, pi])

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m  (getPoints subfunction)
    Chebfun commit: 7574c77
    """
    return np.linspace(-np.pi, np.pi, 2 * n, endpoint=False, dtype=np.float64)


# ============================================================================
# Tolerance helper
# ============================================================================


def _get_tol_sphere(
    F: np.ndarray, hx: float, hy: float, pseudo_level: float
) -> tuple[float, float]:
    """Compute construction tolerance for spherefun.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m  (getTol subfunction)
    Chebfun commit: 7574c77
    """
    m, n = F.shape
    grid = max(m, n)
    dfdx = np.diff(F[: m - 1, :], axis=1) / hx
    dfdy = np.diff(F[:, : n - 1], axis=0) / hy
    jac_norm = float(np.max(np.maximum(np.abs(dfdx.ravel()), np.abs(dfdy.ravel()))))
    vscale = float(np.max(np.abs(F)))
    dom_scale = np.pi  # max |dom| for [-pi,pi] x [0,pi]
    tol = (grid ** (2.0 / 3.0)) * dom_scale * max(jac_norm, vscale) * pseudo_level
    return tol, vscale


# ============================================================================
# Phase 1: GE with 2x2 block pivoting on the doubled-up function matrix
# ============================================================================


def _phase_one_sphere(
    F: np.ndarray,
    tol: float,
    alpha: float,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """GE with 2x2 block pivoting on the doubled-up sphere function matrix.

    Operates on F of shape (m, 2n), where:
      - F[:, :n]  = f(lam_j, theta_i)           — the original block
      - F[:, n:2n]= f(lam_j + pi, theta_i)       — the pi-shifted block

    Splits into Fp and Fm and removes the pole rows (theta=0 and theta=pi)
    before rank determination.

    Parameters
    ----------
    F : np.ndarray, shape (m, 2n)
        Doubled-up function values; rows are theta-points (0 to pi),
        cols are lam-points (2n equispaced on [-pi, pi)).
    tol : float
        Construction tolerance.
    alpha : float
        Coupling parameter.
    factor : float
        Rank bound = min(2m-2, n) / factor.

    Returns
    -------
    pivot_indices : np.ndarray, shape (rk, 2)  [0-based]
        (row_idx, col_idx) pairs into the reduced grid (without poles).
        Adjusted to include the pole rows in the full F indexing.
    pivot_array : np.ndarray, shape (rk, 2)
        (evp, evm) pivot pairs.
    remove_poles : bool
        True if poles needed removal.
    is_happy : bool
        True if converged below tol.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m  (PhaseOne subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Townsend, Wilber, Wright, SISC 38(4) 2016.
    """
    m, n2 = F.shape
    n = n2 // 2

    # The effective rank bound accounts for doubling in theta
    minsize = min(2 * m - 2, n)
    width = minsize / factor if factor > 0 else np.inf

    C = F[:, :n]
    B = F[:, n:]
    Fp = 0.5 * (B + C)
    Fm = 0.5 * (B - C)

    # Check poles at theta=0 (row 0) and theta=pi (row m-1)
    pole1 = float(np.mean(Fp[0, :]))
    pole2 = float(np.mean(Fp[m - 1, :]))
    remove_poles = (abs(pole1) > tol) or (abs(pole2) > tol)

    pivot_indices = []
    pivot_array = []
    rank_count = 0
    pole_col = 0
    pole_val = 0.0

    if remove_poles:
        pole_col = int(np.argmax(np.max(np.abs(Fp), axis=0)))
        pole_val = float(np.max(np.abs(Fp[:, pole_col])))
        row_pole = pole_val * np.ones((1, n))
        col_pole = Fp[:, pole_col].copy()
        Fp = Fp - np.outer(col_pole, row_pole[0] / pole_val)
        rank_count += 1

    # Remove pole rows (first and last)
    Fp = Fp[1 : m - 1, :]
    Fm = Fm[1 : m - 1, :]
    mr = m - 2  # reduced row dimension (interior points only)

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
        if maxp_val >= maxm_val:
            idx = idxp
        else:
            idx = idxm

        j = idx % mr
        k = idx // mr

        evp = float(Fp[j, k])
        evm = float(Fm[j, k])
        absevp = abs(evp)
        absevm = abs(evm)

        pivot_indices.append([j, k])

        if max(absevp, absevm) <= alpha * min(absevp, absevm):
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
                cp = Fp[:, k].copy()
                rp = Fp[j, :].copy()
                Fp = Fp - np.outer(cp, rp) / evp
                evm = 0.0
                rank_count += 1
            else:
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

    # Adjust row indices: add 1 to account for removed north pole row
    pivot_indices[:, 0] += 1

    # Prepend pole pivot if needed
    if remove_poles:
        pivot_indices = np.vstack([[0, pole_col], pivot_indices])
        pivot_array = np.vstack([[pole_val, 0.0], pivot_array])

    return pivot_indices, pivot_array, remove_poles, is_happy


# ============================================================================
# Phase 2: Resolve column and row slices adaptively
# ============================================================================


def _phase_two_sphere(
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
    """Resolve column (Trigtech in theta) and row (Trigtech in lam) slices.

    Parameters
    ----------
    f : callable
        f(lam, theta) -> array_like, vectorised.
    pivot_indices : np.ndarray, shape (rk, 2)
        0-based (row_idx_in_col_pts, col_idx_in_row_pts).
    pivot_array : np.ndarray, shape (rk, 2)
        (evp, evm) pairs from Phase 1.
    n : int
        Initial grid size (row/longitude direction).
    m : int
        Initial grid size (column/colatitude direction).
    vscale : float
        Value scale estimate.
    max_sample : int
        Maximum allowed grid size.
    remove_poles : bool
        Whether pole removal was needed.
    tol : float
        Construction tolerance.

    Returns
    -------
    cols_list : list of Trigtech
        Column slices (in theta, doubled domain [-pi, pi]).
    rows_list : list of Trigtech
        Row slices (in lam, [-pi, pi]).
    pivots : np.ndarray, shape (total_rank,)
    idx_plus : list of int
    idx_minus : list of int

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m  (PhaseTwo subfunction)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    rk = pivot_indices.shape[0]
    id_rows = pivot_indices[:, 0]
    id_cols = pivot_indices[:, 1]

    # Physical pivot locations (using initial grid)
    th_pts_init = _sphere_col_pts(m)  # shape (m+1,) on [0, pi]
    lam_pts_init = _sphere_row_pts(n)  # shape (2*n,) on [-pi, pi)

    row_pivots = th_pts_init[id_rows]
    col_pivots = lam_pts_init[id_cols]

    # Count non-zero pivot components
    n_pos = int(np.sum(np.abs(pivot_array[:, 0]) > 0))
    n_neg = int(np.sum(np.abs(pivot_array[:, 1]) > 0))
    total_rank = n_pos + n_neg

    if total_rank == 0:
        zero_col = Trigtech.from_coeffs(jnp.zeros(1, dtype=jnp.complex128))
        zero_row = Trigtech.from_coeffs(jnp.zeros(1, dtype=jnp.complex128))
        return [zero_col], [zero_row], np.array([1.0]), [0], []

    happy_cols = False
    happy_rows = False
    failure = False

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
    pivots_raw = np.zeros(total_rank)

    while not (happy_cols and happy_rows) and not failure:
        th_pts = _sphere_col_pts(m_cur)  # shape (m_cur+1,) on [0, pi]
        lam_pts = _sphere_row_pts(n_cur)  # shape (2*n_cur,) on [-pi, pi)

        # Sample columns: for each col_pivot (lam value), sample f over theta [0, pi]
        # Evaluate at lam = col_pivot + pi (shifted) and lam = col_pivot
        new_cols_shifted = np.zeros((m_cur + 1, rk))
        new_cols_unshifted = np.zeros((m_cur + 1, rk))
        for jj in range(rk):
            lam_val = col_pivots[jj]
            new_cols_shifted[:, jj] = np.array(
                f(
                    jnp.full(m_cur + 1, lam_val + np.pi, dtype=jnp.float64),
                    jnp.asarray(th_pts, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )
            new_cols_unshifted[:, jj] = np.array(
                f(
                    jnp.full(m_cur + 1, lam_val, dtype=jnp.float64),
                    jnp.asarray(th_pts, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )

        new_cols_plus = 0.5 * (new_cols_shifted + new_cols_unshifted)
        new_cols_minus = 0.5 * (new_cols_shifted - new_cols_unshifted)

        # Sample rows: for each row_pivot (theta value), sample f over 2*n_cur lam points
        new_rows = np.zeros((rk, 2 * n_cur))
        for ii in range(rk):
            th_val = row_pivots[ii]
            new_rows[ii, :] = np.array(
                f(
                    jnp.asarray(lam_pts, dtype=jnp.float64),
                    jnp.full(2 * n_cur, th_val, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )

        # Split rows: first n_cur = lam (second half = lam+pi)
        new_rows_plus = 0.5 * (new_rows[:, n_cur:] + new_rows[:, :n_cur])
        new_rows_minus = 0.5 * (new_rows[:, n_cur:] - new_rows[:, :n_cur])

        # Enforce pole rows zeroed out in Fp
        if remove_poles:
            new_rows_plus[0, :] = pivot_array[0, 0]

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

        # Enforce zero at poles
        if remove_poles:
            if plus_count > 1:
                cols_plus_cur[0, 1:plus_count] = 0.0
                cols_plus_cur[-1, 1:plus_count] = 0.0
        elif plus_count > 0:
            cols_plus_cur[0, :plus_count] = 0.0
            cols_plus_cur[-1, :plus_count] = 0.0

        if minus_count > 0:
            cols_minus_cur[0, :minus_count] = 0.0
            cols_minus_cur[-1, :minus_count] = 0.0

        cols_plus = cols_plus_cur[:, :plus_count]
        cols_minus = cols_minus_cur[:, :minus_count]
        rows_plus = rows_plus_cur[:plus_count, :]
        rows_minus = rows_minus_cur[:minus_count, :]
        pivots_raw = pivots_raw[:pivot_count]

        # Happiness check for columns (Trigtech-style in theta)
        cp_arr = cols_plus if cols_plus.size > 0 else np.zeros((m_cur + 1, 0))
        cm_arr = cols_minus if cols_minus.size > 0 else np.zeros((m_cur + 1, 0))

        def _safe_sum_cols(a, b):
            if a.size == 0 and b.size == 0:
                return np.zeros(m_cur + 1)
            if a.size == 0:
                return np.sum(b, axis=1)
            if b.size == 0:
                return np.sum(a, axis=1)
            return np.sum(np.hstack([a, b]), axis=1)

        temp1 = _safe_sum_cols(cp_arr, cm_arr)
        temp2 = _safe_sum_cols(cp_arr, -cm_arr)
        # Doubled-up col values: [flipud(temp2); temp1[1:m_cur]]  (shape 2*m_cur)
        col_vals_doubled = np.concatenate([temp2[::-1], temp1[1:m_cur]])
        happy_cols = _is_happy_trig(col_vals_doubled, tol)

        # Happiness check for rows (Trigtech)
        def _safe_sum_rows(a, b):
            if a.size == 0 and b.size == 0:
                return np.zeros(n_cur)
            if a.size == 0:
                return np.sum(b, axis=0)
            if b.size == 0:
                return np.sum(a, axis=0)
            return np.sum(np.vstack([a, b]), axis=0)

        rp_sum = _safe_sum_rows(rows_plus, rows_minus)
        rm_sum = _safe_sum_rows(rows_plus, -rows_minus)
        row_vals_doubled = np.concatenate([rp_sum, rm_sum])
        happy_rows = _is_happy_trig(row_vals_doubled, tol)

        if not happy_cols:
            m_new = 2 * m_cur
            if m_new + 1 > max_sample:
                warnings.warn(
                    "Spherefun.from_function: column slices not resolved.",
                    RuntimeWarning,
                    stacklevel=5,
                )
                failure = True
                break
            # MATLAB: ii = [1:2:m-1 m+2:2:2*m]; id_rows = ii(id_rows)
            # This maps old index i to: for i in 1..m-1 -> 2*i-1 (1-based odd)
            # In 0-based: old i -> 2*i (even indices in the new grid)
            id_rows_cur = 2 * id_rows_cur
            m_cur = m_new

        if not happy_rows:
            n_new = 2 * n_cur
            if n_new > max_sample:
                warnings.warn(
                    "Spherefun.from_function: row slices not resolved.",
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
    total = len(idx_plus_raw) + len(idx_minus_raw)
    cp_h = (
        cols_plus.shape[0]
        if cols_plus is not None and cols_plus.size > 0
        else (cols_minus.shape[0] if cols_minus is not None else 1)
    )
    # Doubled column: 2*(m+1) - 2 = 2*m points (no repeated endpoints)
    n_col_full = 2 * cp_h - 2
    n_row_full = 2 * (
        rows_plus.shape[1]
        if rows_plus is not None and rows_plus.size > 0
        else (rows_minus.shape[1] if rows_minus is not None else 1)
    )

    cols_full = np.zeros((n_col_full, total))
    rows_full = np.zeros((n_row_full, total))

    if cols_plus is not None and cols_plus.size > 0:
        for kk, gidx in enumerate(idx_plus_raw):
            c = cols_plus[:, kk]
            # [flipud(c); c[1:end-1]] — even extension, no repeated poles
            cols_full[:, gidx] = np.concatenate([c[::-1], c[1:-1]])
    if cols_minus is not None and cols_minus.size > 0:
        for kk, gidx in enumerate(idx_minus_raw):
            c = cols_minus[:, kk]
            # [-flipud(c); c[1:end-1]] — odd extension
            cols_full[:, gidx] = np.concatenate([-c[::-1], c[1:-1]])
    if rows_plus is not None and rows_plus.size > 0:
        for kk, gidx in enumerate(idx_plus_raw):
            r = rows_plus[kk, :]
            rows_full[:, gidx] = np.concatenate([r, r])
    if rows_minus is not None and rows_minus.size > 0:
        for kk, gidx in enumerate(idx_minus_raw):
            r = rows_minus[kk, :]
            rows_full[:, gidx] = np.concatenate([-r, r])

    # Build Trigtech objects for cols and rows
    cols_list = []
    rows_list = []

    for j in range(total):
        # Column: trigtech on doubled theta domain (length 2*m points)
        cv = jnp.asarray(cols_full[:, j], dtype=jnp.float64)
        cc = trig_vals2coeffs(cv.astype(jnp.complex128))
        cv_scale = float(jnp.max(jnp.abs(cv)))
        if cv_scale > 0:
            chop_in = _trig_abs_coeffs_for_chop(cc)
            chop_rel = max(_EPS, tol / cv_scale)
            cutoff_exp = standard_chop(chop_in.astype(jnp.float64), chop_rel)
            n_keep = _chop_cutoff_to_ncoeffs(int(cutoff_exp), cc.shape[0])
            cc = _trig_prolong_coeffs(cc, n_keep)
        cols_list.append(Trigtech.from_coeffs(cc, is_real=True))

        # Row: trigtech on lam domain (length 2*n points)
        rv = jnp.asarray(rows_full[:, j], dtype=jnp.float64)
        rc = trig_vals2coeffs(rv.astype(jnp.complex128))
        rv_scale = float(jnp.max(jnp.abs(rv)))
        if rv_scale > 0:
            chop_in = _trig_abs_coeffs_for_chop(rc)
            chop_rel = max(_EPS, tol / rv_scale)
            cutoff_exp = standard_chop(chop_in.astype(jnp.float64), chop_rel)
            n_keep = _chop_cutoff_to_ncoeffs(int(cutoff_exp), rc.shape[0])
            rc = _trig_prolong_coeffs(rc, n_keep)
        rows_list.append(Trigtech.from_coeffs(rc, is_real=True))

    return cols_list, rows_list, pivots_raw, idx_plus_raw, idx_minus_raw


# ============================================================================
# Happiness check helper
# ============================================================================


def _is_happy_trig(values: np.ndarray, tol: float) -> bool:
    """Check if trigonometric values are resolved."""
    v = jnp.asarray(values, dtype=jnp.float64)
    c = trig_vals2coeffs(v.astype(jnp.complex128))
    vscale = float(jnp.max(jnp.abs(v)))
    if vscale == 0.0:
        return True
    chop_in = _trig_abs_coeffs_for_chop(c)
    rel_tol = max(tol / vscale, _EPS)
    cutoff = standard_chop(chop_in.astype(jnp.float64), rel_tol)
    return int(cutoff) < chop_in.shape[0]


# ============================================================================
# Main class
# ============================================================================


class Spherefun(eqx.Module):
    """Low-rank approximation of a function on the unit sphere.

    Represents f(lam, theta) ≈ Σ_j (1/d_j) * c_j(theta) * row_j(lam), where

    - c_j are column slices (Trigtech in theta, doubled domain [-pi, pi]),
    - row_j are row slices (Trigtech in lam, periodic on [-pi, pi]),
    - d_j are scalar pivot values,
    - idx_plus, idx_minus track the "plus" and "minus" terms in the BMC-I decomposition.

    Here theta is the colatitude (0 to pi: north pole to south pole) and
    lam is the longitude (-pi to pi).

    The plus/minus split uses the pi-shift in longitude:
      Fp(lam, theta) = 0.5 * [f(lam + pi, theta) + f(lam, theta)]
      Fm(lam, theta) = 0.5 * [f(lam + pi, theta) - f(lam, theta)]

    Attributes
    ----------
    cols : list of Trigtech
        Column slices c_j(theta).  On the doubled domain [-pi, pi].
    rows : list of Trigtech
        Row slices row_j(lam).  Periodic on [-pi, pi].
    pivots : jax.Array, shape (r,)
        Pivot values d_j.
    idx_plus : tuple of int
        0-based indices into cols/rows/pivots for the "plus" terms.
    idx_minus : tuple of int
        0-based indices into cols/rows/pivots for the "minus" terms.

    Notes
    -----
    Construction is NOT JIT-safe.  Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @spherefun/spherefun.m, @spherefun/constructor.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend, H. Wilber, and G. Wright, "Computing with
        functions on spherical and polar geometries I: The sphere",
        SIAM J. Sci. Comput., 38(4), C403–C425, 2016.

    See Also
    --------
    Diskfun, SeparableApprox
    """

    cols: list  # list of Trigtech (column slices, functions of theta)
    rows: list  # list of Trigtech (row slices, functions of lam)
    pivots: jax.Array  # shape (r,)
    idx_plus: tuple = eqx.field(static=True)
    idx_minus: tuple = eqx.field(static=True)

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
    ) -> "Spherefun":
        """Construct a Spherefun from a callable.

        The function ``f`` should accept (lam, theta) where lam is the
        longitude in [-pi, pi] and theta is the colatitude in [0, pi].
        Both arguments are JAX arrays and the function should be vectorised.

        Parameters
        ----------
        f : callable
            f(lam, theta) -> array_like.  Vectorised over 1D arrays.
            lam in [-pi, pi], theta in [0, pi].
        tol : float, optional
            Target relative tolerance. Default is machine epsilon.
        max_rank : int, optional
            Maximum allowed rank. Default 512.
        max_sample : int, optional
            Maximum grid size per dimension. Default 2^14.

        Returns
        -------
        Spherefun

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @spherefun/constructor.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Townsend, Wilber, Wright, SISC 38(4) 2016.
        """
        alpha = 100.0
        min_sample = 4
        factor = 8.0
        pseudo_level = _EPS

        is_happy = False
        failure = False
        grid = min_sample

        while not is_happy and not failure:
            th_pts = _sphere_col_pts(grid)  # shape (grid+1,)
            lam_pts = _sphere_row_pts(grid)  # shape (2*grid,)

            # Build doubled-up sample matrix: F[i, j] = f(lam_pts[j], th_pts[i])
            # Shape: (grid+1, 2*grid)
            lam_j = jnp.asarray(lam_pts, dtype=jnp.float64)
            th_i = jnp.asarray(th_pts, dtype=jnp.float64)
            lam2d, th2d = jnp.meshgrid(lam_j, th_i)  # shapes (grid+1, 2*grid)
            F = np.array(f(lam2d, th2d), dtype=np.float64)

            vscale = float(np.max(np.abs(F)))
            if not np.isfinite(vscale):
                raise ValueError(
                    "Spherefun.from_function: function returned Inf or NaN on the initial grid."
                )

            tol_abs, vscale = _get_tol_sphere(F, np.pi / grid, np.pi / grid, pseudo_level)
            tol_abs = max(tol_abs, 1e4 * tol)

            pivot_indices, pivot_array, remove_poles, happy_rank = _phase_one_sphere(
                F, tol_abs, alpha, factor
            )

            if grid > factor * (max_rank - 1):
                warnings.warn(
                    "Spherefun.from_function: function appears to be high rank. "
                    "Returning best approximation found.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                failure = True
                break

            if max(abs(pivot_array[0, 0]), abs(pivot_array[0, 1])) < 1e4 * tol:
                is_happy = True
                break

            if happy_rank:
                is_happy = True
            else:
                grid = 2 * grid

        # Phase 2: resolve slices
        cols_list, rows_list, pivots_arr, idx_plus, idx_minus = _phase_two_sphere(
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
    def __call__(self, lam: jax.Array, theta: jax.Array) -> jax.Array:
        """Evaluate the Spherefun at spherical coordinates (lam, theta).

        Parameters
        ----------
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        theta : jax.Array
            Colatitude(s) in [0, pi]. Must broadcast with lam.

        Returns
        -------
        jax.Array
            Function values at (lam, theta), same shape as broadcast(lam, theta).

        Notes
        -----
        JIT-safe, vmap-safe, grad-safe.

        Provenance
        ----------
        MATLAB source : @spherefun/feval.m
        Chebfun commit: 7574c77
        """
        lam = jnp.asarray(lam, dtype=jnp.float64)
        theta = jnp.asarray(theta, dtype=jnp.float64)

        # Map lam from [-pi, pi] to [-1, 1] for Trigtech evaluation
        lam_ref = lam / jnp.pi
        # Map theta from [0, pi] to [-1, 1] for Trigtech evaluation
        # The column Trigtech is on the doubled domain [-pi, pi], mapped to [-1, 1]
        # The col stores the doubled-up function; for theta in [0, pi], we use
        # the mapping theta -> theta/pi in [0, 1] -> th_ref = theta/pi - 1/2...
        # Actually the col Trigtech is on [-1, 1] corresponding to the full doubled
        # domain [-pi, pi].  For theta in [0, pi]: th_ref = theta/pi - 1 maps to [-1, 0]
        # but the Trigtech period is 2 (from -1 to 1) and the physical domain is [-pi, pi].
        # Physical theta in [-pi, pi] maps to t in [-1, 1]: t = theta/pi.
        # For colatitude theta in [0, pi], t = theta/pi in [0, 1].
        th_ref = theta / jnp.pi

        result = jnp.zeros_like(jnp.broadcast_arrays(lam, theta)[0], dtype=jnp.float64)
        for j in range(len(self.cols)):
            cj_val = self.cols[j](th_ref)
            rj_val = self.rows[j](lam_ref)
            result = result + (1.0 / self.pivots[j]) * cj_val * rj_val

        return result

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def sum(self) -> jax.Array:
        """Definite integral of the Spherefun over the unit sphere.

        Computes ∫∫ f(lam, theta) sin(theta) d(theta) d(lam)
        over the full sphere (lam in [-pi, pi], theta in [0, pi]).

        Only the "plus" terms contribute (minus terms integrate to zero).

        The integral of each plus term factorises as:
            (1/d_j) * (∫_0^pi c_j(theta) sin(theta) d(theta)) * (∫_{-pi}^{pi} row_j(lam) d(lam))

        The latitude integral uses the cosine series trick (fast Fourier method):
            ∫_0^pi col(theta) sin(theta) d(theta) = Σ_{k even} a_k * 2/(1 - k^2)
        where a_k are the cosine coefficients of col.

        Returns
        -------
        jax.Array, scalar
            Definite integral ∫∫_S f sin(theta) d(theta) d(lam).

        Notes
        -----
        For the full sphere of radius 1: ∫∫ 1 * sin(theta) d(theta) d(lam) = 4π.

        Provenance
        ----------
        MATLAB source : @spherefun/sum2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if len(self.idx_plus) == 0:
            return jnp.array(0.0, dtype=jnp.float64)

        result = jnp.array(0.0, dtype=jnp.float64)

        for j in self.idx_plus:
            # Integrate row over [-pi, pi]:
            # row_j is a Trigtech on [-1, 1] with th_ref = lam/pi
            # d(lam) = pi * d(lam_ref)
            # ∫_{-pi}^{pi} row_j(lam) d(lam) = pi * 2 * c_0
            row_coeffs = self.rows[j].coeffs
            n_row = row_coeffs.shape[0]
            c0_idx_row = n_row // 2
            int_row = jnp.pi * 2.0 * jnp.real(row_coeffs[c0_idx_row])

            # Integrate col * sin(theta) over [0, pi]:
            # col is a Trigtech on [-1, 1] with th_ref = theta/pi (for theta in [0, pi])
            # We use the cosine series identity (MATLAB sum2.m fast code):
            # col(theta) = Σ_k a_k cos(k * theta)   (even in theta, so cosine series)
            # ∫_0^pi col(theta) sin(theta) d(theta) = Σ_k a_k 2/(1 - k^2) for even k
            int_col = _integrate_trigtech_times_sin(self.cols[j])

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

    def __repr__(self) -> str:
        """Compact display.

        Provenance
        ----------
        MATLAB source : @spherefun/display.m
        Chebfun commit: 7574c77
        """
        return (
            f"Spherefun(rank={self.rank}, "
            f"n_plus={len(self.idx_plus)}, n_minus={len(self.idx_minus)})"
        )


# ============================================================================
# Integration helper
# ============================================================================


def _integrate_trigtech_times_sin(col: Trigtech) -> jax.Array:
    """Compute ∫_0^pi col(theta) sin(theta) d(theta).

    The column Trigtech is defined on [-1, 1] with argument t = theta/pi.
    The function col(theta) is an even function of theta (for t in [-1,1]).
    It has the cosine expansion: col(theta) = Σ_{k>=0} a_k cos(k*theta).

    Using the identity:
        ∫_0^pi cos(k*theta) sin(theta) d(theta) = 2/(1 - k^2)  if k even (k>=0)
                                                  = 0             if k odd

    For k=1 the formula is singular: ∫_0^pi cos(theta) sin(theta) d(theta) = 0.
    For k=0: ∫_0^pi sin(theta) d(theta) = 2.

    Fast computation: extract cosine coefficients from the Trigtech (which
    uses complex Fourier coefficients in descending order), then multiply
    by the integration factors.

    Follows MATLAB @spherefun/sum2.m (fast code path).

    Parameters
    ----------
    col : Trigtech
        Column slice; defined on [-1, 1] with t = theta/pi.
        Assumed to be an even function (cosine series in theta).

    Returns
    -------
    jax.Array, scalar
        ∫_0^pi col(theta) sin(theta) d(theta)

    Provenance
    ----------
    MATLAB source : @spherefun/sum2.m
    Chebfun commit: 7574c77
    """
    coeffs = col.coeffs  # complex, descending wavenumber order, length N
    n = coeffs.shape[0]
    c0_idx = n // 2

    # Extract one-sided cosine coefficients (even part):
    # For an even function: c_k = c_{-k} (Hermitian symmetry).
    # The real cosine coefficients are a_k = 2 * Re(c_k) for k > 0, a_0 = Re(c_0).
    # But we use the MATLAB approach: extract trigcoeffs (one-sided cosine, a).
    # trigcoeffs(cols) gives [a0; a1; a2; ...] in MATLAB (cosine coefficients).
    #
    # In our Trigtech, c0_idx = n//2 holds c_0.
    # For a real even function: coeffs[c0_idx - k] = conj(coeffs[c0_idx + k]).
    # Re(c_k) for k >= 0 are the one-sided cosine coefficients (up to factor 2 for k>0).
    #
    # MATLAB formula: k = (0:m-1)'; intFactor = 2/(1 - k(1:2:end)^2)
    # They work with the one-sided (length m) cosine coefficient vector.
    # m = size(a, 1) = (N+1)/2 or so.
    #
    # Here we construct the one-sided coefficients from c0_idx onwards.
    # One-sided: a[k] = Re(coeffs[c0_idx + k]) for k = 0, 1, ..., c0_idx
    # (for even N we have fewer modes)

    half = c0_idx  # number of positive modes
    # a[0] = Re(c_0), a[k] = Re(c_k) for k = 1..half (but the Fourier series is in pi*k*t)
    # Since t = theta/pi, f(theta) = Σ_k c_k exp(i*pi*k*t) = Σ_k c_k exp(i*k*theta)
    # So the wavenumbers are integers (after accounting for t = theta/pi).
    # For even function: a_k (cosine coeff) = 2*Re(c_k) for k>0, Re(c_0) for k=0.
    a_re = jnp.real(coeffs[c0_idx:])  # length half+1: a_0, a_1, ..., a_half
    # Actual cosine amplitudes: a[0] = a_re[0], a[k] = 2*a_re[k] for k>=1
    # But MATLAB uses trigcoeffs which gives the one-sided form directly.
    # Let's match MATLAB: a = [Re(c_0); 2*Re(c_1); 2*Re(c_2); ...]
    k_vals = jnp.arange(half + 1, dtype=jnp.float64)  # 0, 1, ..., half
    factor_k = jnp.where(k_vals == 0, 1.0, 2.0)
    a = factor_k * a_re  # shape (half+1,)

    # Integration factors: 2/(1-k^2) for k even, 0 for k odd
    # a has m = half+1 entries; MATLAB uses k(1:2:end) which are k=0,2,4,...
    # intFactor = 2/(1 - k^2) for k = 0, 2, 4, ...
    # k=0: 2/1 = 2; k=2: 2/(1-4)=-2/3; k=4: 2/(1-16)=-2/15; etc.
    k_even = k_vals[::2]  # 0, 2, 4, ...
    k_even_sq = k_even**2
    int_factor = 2.0 / (1.0 - k_even_sq)
    a_even = a[::2]  # cosine coeffs at even wavenumbers

    # Integral = Σ_{k even} a_k * int_factor_k
    # = Σ a_even * int_factor (element-wise)
    int_col = jnp.dot(a_even, int_factor)

    return int_col
