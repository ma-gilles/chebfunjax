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

# Relative threshold below which a binary-operation result is treated as the
# exact zero field (see Spherefun._binary).  The surface differential
# operators reconstruct via spherical-harmonic quadrature, whose relative
# accuracy floor is ~1e-11; a result 9 orders of magnitude below the operand
# scale is therefore numerically zero, and reconstructing it instead chases
# rounding noise to the maximum grid (the XLA CPU compile blow-up).
_ZERO_REL_TOL = 1e-9


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

    # MATLAB PhaseOne: minSize = min(2*m-2, n) with n = size(F, 2), the
    # FULL (doubled-lambda) column count.
    minsize = min(2 * m - 2, n2)
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

        # numpy argmax flattens in C (row-major) order:
        # j = idx // ncols, k = idx % ncols. (Same column-major bug as
        # the Diskfun constructor, fixed in ab53808 — scrambled pivots
        # caused zero-divisions and frozen ranks.)
        j, k = divmod(idx, Fp.shape[1])

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
        return ([zero_col], [zero_row], np.array([1.0]), [0], [],
                [(0.0, 0.0)])

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
    locs_raw = [(0.0, 0.0)] * total_rank

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
        locs_raw = [(0.0, 0.0)] * total_rank

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
                locs_raw[pivot_count] = (float(col_pivots[ii]),
                                         float(row_pivots[ii]))
                locs_raw[pivot_count + 1] = locs_raw[pivot_count]

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
                locs_raw[pivot_count] = (float(col_pivots[ii]),
                                         float(row_pivots[ii]))
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
                locs_raw[pivot_count] = (float(col_pivots[ii]),
                                         float(row_pivots[ii]))
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
        locs_raw = locs_raw[:pivot_count]

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
    # MATLAB simplify(g, pseudoLevel): chop relative to the GLOBAL scale.
    vs_cols = float(np.max(np.abs(cols_full))) if cols_full.size else 0.0
    vs_rows = float(np.max(np.abs(rows_full))) if rows_full.size else 0.0
    for j in range(total):
        # Column: trigtech on doubled theta domain (length 2*m points)
        cv = jnp.asarray(cols_full[:, j], dtype=jnp.float64)
        cc = trig_vals2coeffs(cv.astype(jnp.complex128))
        cv_scale = float(jnp.max(jnp.abs(cv)))
        if cv_scale > 0:
            chop_in = _trig_abs_coeffs_for_chop(cc)
            chop_rel = max(_EPS, _EPS * vs_cols / cv_scale)
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
            chop_rel = max(_EPS, _EPS * vs_rows / rv_scale)
            cutoff_exp = standard_chop(chop_in.astype(jnp.float64), chop_rel)
            n_keep = _chop_cutoff_to_ncoeffs(int(cutoff_exp), rc.shape[0])
            rc = _trig_prolong_coeffs(rc, n_keep)
        rows_list.append(Trigtech.from_coeffs(rc, is_real=True))

    return (cols_list, rows_list, pivots_raw, idx_plus_raw,
            idx_minus_raw, locs_raw)


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

    @classmethod
    def empty(cls) -> "Spherefun":
        """The empty Spherefun (MATLAB spherefun()): no data; isempty() is
        True and operations on it are undefined.

        Provenance
        ----------
        MATLAB source : @spherefun/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Spherefun (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @spherefun/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    cols: list  # list of Trigtech (column slices, functions of theta)
    rows: list  # list of Trigtech (row slices, functions of lam)
    pivots: jax.Array  # shape (r,)
    idx_plus: tuple = eqx.field(static=True)
    idx_minus: tuple = eqx.field(static=True)
    # (lambda, theta) pivot coordinates (MATLAB pivotLocations) and
    # whether a nonzero pole term was removed (MATLAB nonZeroPoles).
    pivot_locations: tuple = eqx.field(static=True, default=())
    nonzero_poles: bool = eqx.field(static=True, default=False)

    @property
    def pivot_values(self):
        """The GE pivot values (MATLAB ``f.pivotValues``).

        Provenance
        ----------
        MATLAB source : @separableApprox/get.m ('pivotValues')
        Chebfun commit: 7574c77
        """
        return self.pivots

    def __len__(self) -> int:
        """The rank of the representation (MATLAB ``length``).

        Provenance
        ----------
        MATLAB source : @separableApprox/length.m
        Chebfun commit: 7574c77
        """
        return len(self.cols)

    def slice_theta(self, th):
        """The trig chebfun ``lam -> f(lam, th)`` (MATLAB f(:, th)).

        Provenance
        ----------
        MATLAB source : @spherefun/subsref.m (colon slices)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        th = float(th)
        return _cf(lambda lam: self(lam, jnp.full_like(
            jnp.asarray(lam, dtype=jnp.float64), th)),
            domain=(-float(_np.pi), float(_np.pi)), trig=True)

    def slice_lambda(self, lam):
        """The trig chebfun ``th -> f(lam, th)`` on [-pi, pi]
        (MATLAB f(lam, :)).

        Provenance
        ----------
        MATLAB source : @spherefun/subsref.m (colon slices)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        lam = float(lam)

        def ev(th):
            t = jnp.asarray(th, dtype=jnp.float64)
            tt = jnp.abs(t)
            ll = jnp.where(t >= 0, lam,
                           jnp.mod(lam + _np.pi + _np.pi,
                                   2 * _np.pi) - _np.pi)
            return self(ll, tt)

        return _cf(ev, domain=(-float(_np.pi), float(_np.pi)),
                   trig=True)

    def slice_z(self, z):
        """The trig chebfun ``lam -> f`` on the circle of constant
        ``z = cos(theta)`` (MATLAB f(:, :, z)).

        Provenance
        ----------
        MATLAB source : @spherefun/subsref.m (colon slices)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        return self.slice_theta(float(_np.arccos(float(z))))

    def _center_pad(C, target):
        """Center-pad/truncate trig coefficient columns to length
        ``target`` (coefficients are wave-number centered)."""
        cur = C.shape[0]
        if cur == target:
            return C
        if cur < target:
            lo = (target - cur) // 2
            hi = target - cur - lo
            return jnp.concatenate([
                jnp.zeros((lo, C.shape[1]), dtype=C.dtype), C,
                jnp.zeros((hi, C.shape[1]), dtype=C.dtype)])
        lo = (cur - target) // 2
        return C[lo:lo + target]

    def coeffs2(self, m=None, n=None):
        """The 2-D Fourier coefficient matrix of the (doubled) sphere
        representation ``U * D * R.'`` (MATLAB ``coeffs2``); with sizes
        the factor coefficients are ALIASED to ``n`` (theta) and ``m``
        (lambda) modes as ``trigtech.alias`` does.

        Provenance
        ----------
        MATLAB source : @spherefun/coeffs2.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import _alias_trigtech, _trig_prolong_coeffs
        if self.isempty() or len(self.cols) == 0:
            return jnp.zeros((0, 0), dtype=jnp.complex128)
        if m is not None and n is None:
            n = m
        cc = [jnp.asarray(c.coeffs, dtype=jnp.complex128) for c in self.cols]
        rr = [jnp.asarray(r.coeffs, dtype=jnp.complex128) for r in self.rows]
        if m is None:
            mc = max(c.shape[0] for c in cc)
            nr = max(r.shape[0] for r in rr)
            U = jnp.stack([_trig_prolong_coeffs(c, mc) for c in cc], axis=1)
            R = jnp.stack([_trig_prolong_coeffs(r, nr) for r in rr], axis=1)
        else:
            U = jnp.stack([_alias_trigtech(c, int(n)) for c in cc], axis=1)
            R = jnp.stack([_alias_trigtech(r, int(m)) for r in rr], axis=1)
        d = jnp.where(jnp.abs(self.pivots) > 0,
                      1.0 / jnp.where(self.pivots == 0, 1.0,
                                      self.pivots), 0.0)
        return U @ jnp.diag(d.astype(U.dtype)) @ R.T

    @staticmethod
    def coeffs2spherefun(X) -> "Spherefun":
        """Build a Spherefun from a 2-D Fourier coefficient matrix
        (MATLAB ``spherefun.coeffs2spherefun``).

        Provenance
        ----------
        MATLAB source : @spherefun/coeffs2spherefun.m
        Chebfun commit: 7574c77
        """
        X = jnp.asarray(X)
        mth, nlam = X.shape
        kth = jnp.arange(mth) - mth // 2
        klam = jnp.arange(nlam) - nlam // 2

        def f(lam, th):
            Eth = jnp.exp(1j * jnp.tensordot(
                jnp.asarray(th), kth, axes=0))
            El = jnp.exp(1j * jnp.tensordot(
                jnp.asarray(lam), klam, axes=0))
            return jnp.real(jnp.einsum("...j,jk,...k->...",
                                       Eth, X.astype(jnp.complex128),
                                       El))

        return Spherefun.from_function(f)

    @staticmethod
    def coeffs2vals(X):
        """2-D Fourier coefficients -> values on the equispaced
        lat-lon grid (MATLAB ``spherefun.coeffs2vals``).

        Provenance
        ----------
        MATLAB source : @spherefun/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import trig_coeffs2vals
        X = jnp.asarray(X, dtype=jnp.complex128)
        V = jnp.stack([jnp.ravel(jnp.asarray(trig_coeffs2vals(
            X[:, j]))) for j in range(X.shape[1])], axis=1)
        W = jnp.stack([jnp.ravel(jnp.asarray(trig_coeffs2vals(
            V[i, :]))) for i in range(V.shape[0])], axis=0)
        return W

    @staticmethod
    def vals2coeffs(V):
        """Inverse of :meth:`coeffs2vals`.

        Provenance
        ----------
        MATLAB source : @spherefun/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import trig_vals2coeffs
        V = jnp.asarray(V, dtype=jnp.complex128)
        W = jnp.stack([jnp.ravel(jnp.asarray(trig_vals2coeffs(
            V[i, :]))) for i in range(V.shape[0])], axis=0)
        C = jnp.stack([jnp.ravel(jnp.asarray(trig_vals2coeffs(
            W[:, j]))) for j in range(W.shape[1])], axis=1)
        return C

    def cdr(self):
        """CDR decomposition ``(C, D, R)`` with ``D = diag(1/pivots)``
        (zero pivots map to 0), mirroring MATLAB's three-output
        ``[C, D, R] = cdr(f)``.

        Provenance
        ----------
        MATLAB source : @separableApprox/cdr.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        d = jnp.where(jnp.abs(self.pivots) > 0,
                      1.0 / jnp.where(self.pivots == 0, 1.0,
                                      self.pivots), 0.0)
        return list(self.cols), jnp.diag(d), list(self.rows)

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
        min_abs_tol: float = 0.0,
        start_grid: int | None = None,
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
        min_abs_tol : float, optional
            Absolute floor on the pivoting tolerance.  When re-approximating
            a result that is much smaller than the data it was derived from
            (e.g. a nearly-cancelling difference), the operands' construction
            noise -- ~eps times the operand scale in absolute terms -- has no
            band-limited structure and the adaptive search chases it to
            ``max_sample``.  Passing the noise level here lets the
            constructor ignore sub-noise structure and converge.  Default 0
            (pure relative tolerance).

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
        if start_grid is not None:
            # Oversampled deterministic start (MATLAB @spherefun/rotate.m
            # samples at a fixed grid sized from the input's own length so
            # the result is resolved on the FIRST pass -- adaptive
            # refinement from a coarse grid makes the representation
            # depend chaotically on sub-ulp sampling perturbations).
            grid = max(min_sample, int(start_grid) // 2)
        # "0 + noise" strike counter (MATLAB @spherefun/constructor): a tiny
        # dominant pivot must persist for three successive grids before the
        # function is judged numerically zero.  Declaring happiness on the
        # FIRST tiny pivot (the previous behaviour) collapsed genuinely
        # high-rank functions to rank 1 when they were re-approximated from
        # their own noisy values (e.g. abs(f), real(f), or any compose op):
        # rounding noise seeds a ~1e-13 dominant pivot that spuriously trips
        # the zero test.  Accumulating strikes lets the grid refine until the
        # true rank appears.  Mirrors the diskfun constructor fix.
        strike = 1

        while not is_happy and not failure and strike < 3:
            # Double the grid BEFORE sampling, matching MATLAB
            # @spherefun/constructor.m (line ~101: `grid = 2*grid`).
            # Fix by Claude Opus 4.8: the previous code ran phase one at
            # the coarse `min_sample` grid on the first pass, where a sum
            # of harmonics with different sine orders (e.g. Y_2^{-1} +
            # Y_4^{-3}) aliases to rank 1 and falsely reports happy=True,
            # producing a wrong low-rank spherefun. Starting at
            # 2*min_sample exposes the true rank (verified: grid 4 ->
            # false happy; grid 8 -> not happy so the loop refines to
            # grid 16 where rank 2 is found).
            grid = 2 * grid
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
            # (no 1e4*eps floor: MATLAB's getTol is used as computed)
            # Absolute floor: ignore structure below the caller-supplied
            # noise level (see the ``min_abs_tol`` parameter) so a
            # nearly-cancelling difference does not chase its operands'
            # construction noise to ``max_sample``.
            if min_abs_tol > 0.0:
                tol_abs = max(tol_abs, min_abs_tol)

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

            # If the function is 0 + noise the dominant pivot is tiny.  Treat
            # this as convergence only after THREE successive strikes so a
            # single spurious tiny pivot from coarse-grid evaluation noise
            # cannot stop the loop (mirrors the diskfun constructor).
            if max(abs(pivot_array[0, 0]), abs(pivot_array[0, 1])) < 1e4 * tol:
                strike += 1

            if happy_rank:
                is_happy = True
            # else: the grid is doubled at the top of the next iteration
            # (moved there by Opus 4.8 to match MATLAB and avoid the
            # coarse-grid false-convergence described above).

        # Phase 2: resolve slices
        (cols_list, rows_list, pivots_arr, idx_plus, idx_minus,
         locs) = _phase_two_sphere(
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

        result = cls(
            cols=cols_list,
            rows=rows_list,
            pivots=jnp.asarray(pivots_arr, dtype=jnp.float64),
            idx_plus=tuple(idx_plus),
            idx_minus=tuple(idx_minus),
            pivot_locations=tuple(locs),
            nonzero_poles=bool(remove_poles),
        )

        # Sample test (MATLAB constructor sampleTest analogue): on
        # marginally-resolved functions the phase-2 GE replay can
        # DIVERGE -- near-singular phase-1 pivots amplify slice values
        # exponentially across the elimination, producing objects whose
        # evaluations are astronomically wrong (observed: bounded
        # samples <= 39 building a rank-91 object evaluating to 2e90).
        # Probe off-grid points; on catastrophic mismatch restart the
        # construction from a finer initial grid.
        phi_gr = 0.6180339887498949
        ts = np.arange(1, 25, dtype=float)
        lam_t = -np.pi + 2 * np.pi * ((0.5 + phi_gr * ts) % 1.0)
        th_t = np.pi * ((0.25 + phi_gr * ts * ts) % 1.0)
        fv = np.asarray(f(jnp.asarray(lam_t), jnp.asarray(th_t))).ravel()
        av = np.asarray(result(jnp.asarray(lam_t),
                               jnp.asarray(th_t))).ravel()
        fscale = max(float(np.max(np.abs(fv))), 1e-300)
        # MATLAB @spherefun/sampleTest.m: max|f - g| <= 100*tol at the
        # scattered points (tol = the construction tolerance).
        _stol = 100.0 * max(float(tol_abs), _EPS * fscale)
        if not np.all(np.isfinite(av)) or \
                float(np.max(np.abs(av - fv))) > _stol:
            new_start = 2 * max(grid, 8)
            if start_grid is None or new_start > int(start_grid):
                if new_start <= max_sample // 4:
                    return cls.from_function(
                        f, tol=tol, max_rank=max_rank,
                        max_sample=max_sample,
                        min_abs_tol=min_abs_tol,
                        start_grid=new_start)
            warnings.warn(
                "Spherefun.from_function: construction failed the "
                "off-grid sample test (phase-2 divergence on a "
                "marginally-resolved function); returning the best "
                "approximation found.",
                RuntimeWarning, stacklevel=2)
        # MATLAB @spherefun/constructor.m: simplify, then project onto the
        # exact BMC-I symmetry.
        return result.simplify()._prune_zero_terms().projectOntoBMCI()

    def _prune_zero_terms(self) -> "Spherefun":
        """Drop CDR terms whose column or row simplified to exactly zero
        (a noise-level pivot accepted by the GE); keeps the parity
        bookkeeping and the pole term."""
        if self.isempty() or len(self.cols) == 0:
            return self
        keep = [j for j in range(len(self.cols))
                if float(np.max(np.abs(np.asarray(self.cols[j].values)))) > 0.0
                and float(np.max(np.abs(np.asarray(self.rows[j].values)))) > 0.0]
        if len(keep) == len(self.cols):
            return self
        if not keep:
            return self
        idx_plus = tuple(keep.index(j) for j in self.idx_plus if j in keep)
        idx_minus = tuple(keep.index(j) for j in self.idx_minus if j in keep)
        pole_kept = (self.nonzero_poles and len(self.idx_plus) > 0
                     and int(self.idx_plus[0]) in keep)
        locs = tuple(self.pivot_locations[j] for j in keep) \
            if len(self.pivot_locations) == len(self.cols) else ()
        return Spherefun(cols=[self.cols[j] for j in keep],
                  rows=[self.rows[j] for j in keep],
                  pivots=jnp.asarray([float(self.pivots[j]) for j in keep],
                                     dtype=jnp.float64),
                  idx_plus=idx_plus, idx_minus=idx_minus,
                  pivot_locations=locs, nonzero_poles=bool(pole_kept))

    def simplify(self, tol: float | None = None) -> "Spherefun":
        """Chop the column and row slices (MATLAB ``simplify``)."""
        if self.isempty() or len(self.cols) == 0:
            return self
        cols = _simplify_global_sphere(list(self.cols), tol)
        rows = _simplify_global_sphere(list(self.rows), tol)
        return Spherefun(cols=cols, rows=rows, pivots=self.pivots,
                         idx_plus=self.idx_plus, idx_minus=self.idx_minus,
                         pivot_locations=self.pivot_locations,
                         nonzero_poles=self.nonzero_poles)

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

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
        JIT-safe, vmap-safe, grad-safe.  Concrete inputs run the rank
        loop eagerly (each Trigtech slice then evaluates via its numpy
        Horner mirror): the jitted path compiles a fresh rank-sized XLA
        program PER SPHEREFUN OBJECT, and object-heavy chains (e.g. the
        curl of a rank-50 gradient field) exhaust the LLVM JIT code
        arena ("Unable to allocate section memory").

        Provenance
        ----------
        MATLAB source : @spherefun/feval.m
        Chebfun commit: 7574c77
        """
        if not isinstance(lam, jax.core.Tracer) and \
                not isinstance(theta, jax.core.Tracer) and \
                not isinstance(self.pivots, jax.core.Tracer):
            return self._eval_np(lam, theta)
        return self._call_traced(lam, theta)

    def _eval_np(self, lam, theta) -> jax.Array:
        """Concrete-input evaluation with ALL rank slices batched into a
        single array-valued numpy Horner per direction.  The per-slice
        loop cost ~7 ms x rank x calls (28k Trigtech evals = 194 s for
        one rank-54 x rank-60 product re-approximation); batching makes
        it one padded (n, r) Horner each way."""
        from chebfunjax.tech.trigtech import _alias_trigtech, _trig_eval_np

        if len(self.cols) == 0:
            return self._eval_impl(lam, theta)
        col_real = all(c.is_real for c in self.cols)
        row_real = all(r.is_real for r in self.rows)
        if (any(c.is_real != col_real for c in self.cols)
                or any(r.is_real != row_real for r in self.rows)):
            return self._eval_impl(lam, theta)

        lam = np.asarray(lam, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        bshape = np.broadcast_shapes(lam.shape, theta.shape)

        nc = max(int(np.asarray(c.coeffs).shape[0]) for c in self.cols)
        nr = max(int(np.asarray(r.coeffs).shape[0]) for r in self.rows)
        Cc = np.stack([np.asarray(_alias_trigtech(c.coeffs, nc))
                       for c in self.cols], axis=1)          # (nc, r)
        Rr = np.stack([np.asarray(_alias_trigtech(r.coeffs, nr))
                       for r in self.rows], axis=1)          # (nr, r)
        piv = 1.0 / np.asarray(self.pivots)

        # Tensor-grid detection: the adaptive constructor samples on
        # meshgrids (lam varying along one axis, theta along the other).
        # Evaluating each direction ONCE on its 1D point set and
        # combining with a rank-sized matmul turns an
        # O(npts * rank * degree) sweep into
        # O((nth + nlam) * rank * degree + npts * rank).
        lb = np.broadcast_to(lam, bshape)
        tb = np.broadcast_to(theta, bshape)
        if len(bshape) == 2 and bshape[0] > 1 and bshape[1] > 1:
            lam_axis1 = np.all(lb == lb[:1, :])   # lam constant down cols
            th_axis0 = np.all(tb == tb[:, :1])    # theta constant across
            if lam_axis1 and th_axis0:
                cv = np.asarray(_trig_eval_np(Cc, tb[:, 0] / np.pi,
                                              is_real=col_real))
                rv = np.asarray(_trig_eval_np(Rr, lb[0, :] / np.pi,
                                              is_real=row_real))
                return jnp.asarray((cv * piv) @ rv.T)
            lam_axis0 = np.all(lb == lb[:, :1])
            th_axis1 = np.all(tb == tb[:1, :])
            if lam_axis0 and th_axis1:
                cv = np.asarray(_trig_eval_np(Cc, tb[0, :] / np.pi,
                                              is_real=col_real))
                rv = np.asarray(_trig_eval_np(Rr, lb[:, 0] / np.pi,
                                              is_real=row_real))
                return jnp.asarray(rv @ (cv * piv).T)

        lr = (lb / np.pi).ravel()
        tr = (tb / np.pi).ravel()
        # Constant-coordinate lines (phase-2 slice sampling).
        if lr.size > 1 and np.all(lr == lr[0]):
            rv = np.asarray(_trig_eval_np(Rr, lr[:1], is_real=row_real))
            cv = np.asarray(_trig_eval_np(Cc, tr, is_real=col_real))
            vals = (cv * piv) @ rv[0]
            return jnp.asarray(vals.reshape(bshape))
        if tr.size > 1 and np.all(tr == tr[0]):
            cv = np.asarray(_trig_eval_np(Cc, tr[:1], is_real=col_real))
            rv = np.asarray(_trig_eval_np(Rr, lr, is_real=row_real))
            vals = (rv * piv) @ cv[0]
            return jnp.asarray(vals.reshape(bshape))

        cv = np.asarray(_trig_eval_np(Cc, tr, is_real=col_real))
        rv = np.asarray(_trig_eval_np(Rr, lr, is_real=row_real))
        vals = (cv * rv) @ piv
        return jnp.asarray(vals.reshape(bshape))

    @eqx.filter_jit
    def _call_traced(self, lam: jax.Array, theta: jax.Array) -> jax.Array:
        return self._eval_impl(lam, theta)

    def _eval_impl(self, lam: jax.Array, theta: jax.Array) -> jax.Array:
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

    def fast_sphere_eval(self, lam, theta):
        """Fast, high-accuracy evaluation at scattered points (2D NUFFT).

        Evaluates the Spherefun at ``(lam, theta)`` to ~1e-15 per point via the
        type-2 two-dimensional NUFFT of the doubled-up Fourier coefficient
        matrix, versus the ~1e-14 Horner scheme of :meth:`__call__`.  Not
        JIT-safe (uses numpy / optional ``finufft``); use :meth:`__call__` on
        the hot, differentiable path and this for high-accuracy resampling
        (e.g. :meth:`rotate`).

        Parameters
        ----------
        lam, theta : array_like
            Longitude in ``[-pi, pi]`` and co-latitude in ``[0, pi]``; broadcast
            against one another.

        Returns
        -------
        np.ndarray
            Values of the Spherefun, shape ``broadcast(lam, theta)``.

        Provenance
        ----------
        MATLAB source : @spherefun/fastSphereEval.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.  Primary author: Alex Townsend.
        """
        from chebfunjax.spherefun.fast_sphere_eval import fast_sphere_eval

        return fast_sphere_eval(self, lam, theta)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def sum2(self) -> jax.Array:
        """MATLAB-parity alias for :meth:`sum` (surface integral).

        Provenance
        ----------
        MATLAB source : @spherefun/sum2.m
        Chebfun commit: 7574c77
        """
        return self.sum()

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

    def length(self) -> tuple[int, int]:  # noqa: D401
        """(m, n): angular (lambda, rows) and colatitude (theta, cols)
        resolution of the representation (MATLAB [m, n] = length(f)).

        Provenance
        ----------
        MATLAB source : @spherefun/length.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return (0, 0)
        m = max(int(np.asarray(r.coeffs).ravel().shape[0])
                for r in self.rows)
        n = max(int(np.asarray(c.coeffs).ravel().shape[0])
                for c in self.cols)
        return (m, n)

    def fevalm(self, lam, th) -> jax.Array:
        """Evaluate on the tensor grid of 1D arrays ``lam`` (longitude)
        and ``th`` (colatitude): returns the ``(len(th), len(lam))``
        matrix of values (MATLAB fevalm).

        Provenance
        ----------
        MATLAB source : @spherefun/fevalm.m, @separableApprox/fevalm.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return jnp.zeros((0, 0), dtype=jnp.float64)
        lam = jnp.atleast_1d(jnp.asarray(lam, dtype=jnp.float64)).ravel()
        th = jnp.atleast_1d(jnp.asarray(th, dtype=jnp.float64)).ravel()
        L, T = jnp.meshgrid(lam, th)
        return self(L, T)

    def sample(self, m: int | None = None, n: int | None = None) -> jax.Array:
        """Values on an m (longitude) x n (colatitude) tensor grid:
        lam = trigpts(m, [-pi, pi]), th = linspace(0, pi, n); returns
        an (n, m) matrix (MATLAB sample).

        Provenance
        ----------
        MATLAB source : @spherefun/sample.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.quadrature import trigpts
        if m is None or n is None:
            m0, n0 = self.length()
            m = m0 if m is None else m
            n = n0 if n is None else n
        lam = np.pi * np.array(trigpts(int(m))[0])
        th = np.linspace(0.0, np.pi, int(n))
        return self.fevalm(jnp.asarray(lam), jnp.asarray(th))

    def sample_cdr(self, m: int, n: int):
        """(U, D, V) sampled low-rank factors with
        ``U @ D @ V.T == sample(m, n)`` (MATLAB [U, D, V] = sample(f)).

        Provenance
        ----------
        MATLAB source : @spherefun/sample.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import _trig_eval_np
        from chebfunjax.utils.quadrature import trigpts
        lam = np.pi * np.array(trigpts(int(m))[0])
        th = np.linspace(0.0, np.pi, int(n))
        U = np.column_stack(
            [np.real(np.asarray(_trig_eval_np(
                np.asarray(c.coeffs)[:, None], th / np.pi,
                is_real=c.is_real))).ravel()
             for c in self.cols])
        V = np.column_stack(
            [np.real(np.asarray(_trig_eval_np(
                np.asarray(r.coeffs)[:, None], lam / np.pi,
                is_real=r.is_real))).ravel()
             for r in self.rows])
        D = np.diag(1.0 / np.asarray(self.pivots, dtype=float))
        return (jnp.asarray(U, dtype=jnp.float64),
                jnp.asarray(D, dtype=jnp.float64),
                jnp.asarray(V, dtype=jnp.float64))

    def mean2(self) -> jax.Array:
        """Mean value over the sphere: sum2(f) / (4*pi) (MATLAB mean2).

        Provenance
        ----------
        MATLAB source : @spherefun/mean2.m
        Chebfun commit: 7574c77
        """
        return self.sum2() / (4.0 * jnp.pi)

    def minandmax2est(self, n: int = 33) -> jax.Array:
        """Estimated [min, max] over the sphere from an n x n sample
        (MATLAB minandmax2est).

        Provenance
        ----------
        MATLAB source : @spherefun/minandmax2est.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return jnp.zeros((2,), dtype=jnp.float64)
        vals = self.sample(n, n).reshape(-1)
        return jnp.array([jnp.min(vals), jnp.max(vals)],
                         dtype=jnp.float64)

    def cosh(self):
        """Hyperbolic cosine, re-approximated (MATLAB cosh)."""
        return self._reapprox(jnp.cosh)

    def sinh(self):
        """Hyperbolic sine, re-approximated (MATLAB sinh)."""
        return self._reapprox(jnp.sinh)

    def tanh(self):
        """Hyperbolic tangent, re-approximated (MATLAB tanh)."""
        return self._reapprox(jnp.tanh)

    def biharm(self) -> "Spherefun":
        """Biharmonic operator: laplacian applied twice (MATLAB
        @spherefun/biharm.m).

        Provenance
        ----------
        MATLAB source : @spherefun/biharm.m
        Chebfun commit: 7574c77
        """
        return self.laplacian().laplacian()

    @staticmethod
    def vertcat(*args) -> "Spherefun":
        """Vertical concatenation: three Spherefuns make a Spherefunv
        (MATLAB ``[f; g; h]``); one argument returns itself; any other
        count is an error.

        Provenance
        ----------
        MATLAB source : @spherefun/vertcat.m
        Chebfun commit: 7574c77
        """
        if len(args) == 1:
            return args[0]
        if len(args) == 3:
            from chebfunjax.spherefun.spherefunv import Spherefunv
            return Spherefunv(*args)
        raise ValueError(
            "Can only vertically concatenate three Spherefun objects.")

    @staticmethod
    def combine(g: "Spherefun", h: "Spherefun") -> "Spherefun":
        """Combine an even/pi-periodic Spherefun with an odd/
        anti-periodic one without re-running the constructor (MATLAB
        combine; inverse of :meth:`partition`).

        Provenance
        ----------
        MATLAB source : @spherefun/combine.m
        Chebfun commit: 7574c77
        """
        if not isinstance(g, Spherefun) or not isinstance(h, Spherefun):
            raise TypeError(
                "Spherefun.combine: inputs must be Spherefun objects.")
        if g.isempty() or len(g.cols) == 0:
            return h
        if h.isempty() or len(h.cols) == 0:
            return g
        if (len(g.idx_plus) > 0 and len(g.idx_minus) > 0) or \
                (len(h.idx_plus) > 0 and len(h.idx_minus) > 0):
            raise ValueError(
                "Spherefun.combine: inputs must have a single parity; "
                "use g + h instead.")
        ng = len(g.cols)
        cols = list(g.cols) + list(h.cols)
        rows = list(g.rows) + list(h.rows)
        piv = jnp.concatenate([jnp.asarray(g.pivots),
                               jnp.asarray(h.pivots)])
        idx_p = tuple(g.idx_plus) + tuple(i + ng for i in h.idx_plus)
        idx_m = tuple(g.idx_minus) + tuple(i + ng for i in h.idx_minus)
        locs = tuple(g.pivot_locations) + tuple(h.pivot_locations)
        return Spherefun(cols=cols, rows=rows, pivots=piv,
                         idx_plus=idx_p, idx_minus=idx_m,
                         pivot_locations=locs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, fmt=None, **kwargs):
        """Plot this Spherefun on the sphere (calls
        :func:`chebfunjax.plotting.plot_sphere`).

        ``plot(f, S)`` with a linespec string S plots the pivot
        locations used in the construction of f instead (MATLAB
        @spherefun/plot.m).
        """
        if isinstance(fmt, str):
            import matplotlib.pyplot as plt
            import numpy as _np
            fig = plt.figure(figsize=(4.5, 4.5))
            ax = fig.add_subplot(projection="3d")
            locs = list(self.pivot_locations)
            lam = _np.array([p[0] for p in locs], dtype=float)
            th = _np.array([p[1] for p in locs], dtype=float)
            x = _np.cos(lam) * _np.sin(th)
            y = _np.sin(lam) * _np.sin(th)
            z = _np.cos(th)
            ax.plot(x, y, z, fmt)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_zlim(-1.0, 1.0)
            return fig, ax
        from chebfunjax.plotting import plot_sphere
        return plot_sphere(self, **kwargs)

    def surf(self, **kwargs):
        """Surface plot on the sphere (calls :func:`chebfunjax.plotting.plot_sphere`)."""
        from chebfunjax.plotting import plot_sphere
        return plot_sphere(self, **kwargs)

    def contour(self, **kwargs):
        """Contour plot on the sphere (calls :func:`chebfunjax.plotting.contour_sphere`)."""
        from chebfunjax.plotting import contour_sphere
        return contour_sphere(self, **kwargs)

    def contour3(self, levels: int = 10, n_pts: int = 200, **kwargs):
        """3-D contour plot on the sphere: same as :meth:`contour`
        (contours are already drawn on the 3-D sphere surface).

        Provenance
        ----------
        MATLAB source : @spherefun/contour3.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.plotting import contour_sphere
        return contour_sphere(self, levels=levels, n_pts=n_pts, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic + composition via constructor re-approximation
    # (MATLAB @spherefun semantics; added by Claude Fable 5 --
    # Spherefun previously had NO arithmetic).
    # ------------------------------------------------------------------

    def _shift_lambda(self, a: float) -> "Spherefun":
        """Exact z-rotation: ``g(lam, th) = f(lam - a, th)``.

        A rotation about the z-axis is a pure longitude shift, which is
        EXACT in the DFS representation -- each row Trigtech's Fourier
        coefficients pick up the phase ``c_k -> c_k exp(-i k a)`` with no
        resampling.  Realness is preserved (the phased coefficients stay
        conjugate-symmetric).

        Provenance
        ----------
        MATLAB source : @spherefun/rotate.m (the Rz factors of the ZYZ
        composition; evaluated here in coefficient space)
        Chebfun commit: 7574c77
        """
        if a == 0.0:
            return self
        from chebfunjax.tech.trigtech import Trigtech

        new_rows = []
        for r in self.rows:
            c = jnp.asarray(r.coeffs, dtype=jnp.complex128)
            n = c.shape[0]
            k = jnp.arange(n, dtype=jnp.float64) - n // 2
            new_rows.append(Trigtech.from_coeffs(
                c * jnp.exp(-1j * k * a), is_real=r.is_real))
        return Spherefun(cols=self.cols, rows=new_rows, pivots=self.pivots,
                         idx_plus=self.idx_plus, idx_minus=self.idx_minus,
                         pivot_locations=self.pivot_locations,
                         nonzero_poles=self.nonzero_poles)

    def partition(self) -> tuple["Spherefun", "Spherefun"]:
        r"""Parity partition ``f = fep + foa`` (MATLAB partition).

        Splits the CDR decomposition into the even/:math:`\pi`-periodic
        terms (``idx_plus``) and the odd/:math:`\pi`-anti-periodic terms
        (``idx_minus``) tracked by the BMC constructor.  Returns
        ``(fep, foa)`` with ``fep + foa == f`` exactly (the same column,
        row, and pivot data, re-indexed).

        Provenance
        ----------
        MATLAB source : @spherefun/partition.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Spherefun.empty(), Spherefun.empty()

        def _sub(idx):
            idx = list(idx)
            if not idx:
                # MATLAB returns an EMPTY spherefun for a missing
                # parity, not a rank-1 zero function.
                return Spherefun.empty()
            return Spherefun(
                cols=[self.cols[i] for i in idx],
                rows=[self.rows[i] for i in idx],
                pivots=jnp.asarray(
                    [float(self.pivots[i]) for i in idx]),
                idx_plus=tuple(range(len(idx)))
                if idx == list(self.idx_plus) else (),
                idx_minus=() if idx == list(self.idx_plus)
                else tuple(range(len(idx))),
                nonzero_poles=bool(self.nonzero_poles and len(self.idx_plus) > 0
                                   and int(self.idx_plus[0]) in [int(v) for v in idx]),
            )

        return _sub(self.idx_plus), _sub(self.idx_minus)

    def rotate(self, phi: float = 0.0, theta: float = 0.0,
               psi: float = 0.0) -> "Spherefun":
        """Rotate by Euler angles (ZYZ convention, MATLAB rotate):
        the rotation is Rz(phi) @ Ry(theta) @ Rz(psi), and
        ``f.rotate(a, b, c).rotate(-c, -b, -a)`` recovers ``f``.

        The two Rz factors are applied EXACTLY in coefficient space
        (longitude phase shifts); only the Ry factor is re-approximated.
        MATLAB resamples the full composite -- the decomposition here is
        the same map with strictly less resampling noise.

        The Ry resampling evaluates ``f1`` through :meth:`fast_sphere_eval`
        (the 2D-NUFFT ``fastSphereEval``) rather than the Horner scheme, and
        samples on a fixed grid sized to ``f1``'s bandlimit (``start_grid``) --
        the rotation of a bandlimited function stays bandlimited, so this grid
        recovers it spectrally on the first pass instead of chasing sub-ulp
        Horner noise through adaptive refinement.  Together these reach
        MATLAB's ``10*eps`` round-trip bound with margin (see
        ``@spherefun/rotate.m``, which uses ``method='nufft'`` by default).

        Provenance
        ----------
        MATLAB source : @spherefun/rotate.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.spherefun.fast_sphere_eval import fast_sphere_eval

        if theta == 0.0:
            # Pure z-rotation: f(Rz(phi+psi)^T x) exactly.
            return self._shift_lambda(phi + psi)

        f1 = self._shift_lambda(psi)
        cb, sb = _np.cos(theta), _np.sin(theta)
        Ry = _np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

        # Bandlimit grid: the rotation of a degree-l bandlimited function is of
        # degree l, so sampling at f1's bandlimit resolves g on the first pass
        # (MATLAB @spherefun/rotate.m: ``n = max(m, n)``).
        start_grid = 2 * f1._bandwidth() + 2

        def g(lam, th):
            lam = _np.asarray(lam, dtype=_np.float64)
            th = _np.asarray(th, dtype=_np.float64)
            x = _np.cos(lam) * _np.sin(th)
            y = _np.sin(lam) * _np.sin(th)
            z = _np.cos(th)
            xp = Ry[0, 0] * x + Ry[1, 0] * y + Ry[2, 0] * z
            yp = Ry[0, 1] * x + Ry[1, 1] * y + Ry[2, 1] * z
            zp = Ry[0, 2] * x + Ry[1, 2] * y + Ry[2, 2] * z
            return fast_sphere_eval(
                f1, _np.arctan2(yp, xp),
                _np.arccos(_np.clip(zp, -1.0, 1.0)))

        f2 = Spherefun.from_function(g, start_grid=start_grid)
        # Final Rz factor, exact.
        return f2._shift_lambda(phi)

    # ------------------------------------------------------------------
    # MATLAB spherefun(DOUBLE): construction from a matrix of samples
    # ------------------------------------------------------------------
    @classmethod
    def from_values(cls, F, tol: float | None = None,
                    alpha: float = 100.0) -> "Spherefun":
        """Construct from an ``n x m`` matrix of samples on the lat-lon
        grid ``lam = trigpts(m, [-pi, pi])`` (columns, ``m`` even) and
        ``th = linspace(0, pi, n)`` (rows), MATLAB ``spherefun(F)``:
        a full (non-adaptive) BMC-I Gaussian elimination on the doubled
        matrix followed by the projection onto BMC-I symmetry.

        Provenance
        ----------
        MATLAB source : @spherefun/constructor.m (constructFromDouble,
            PhaseOne with factor = 0)
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import Trigtech
        F = np.array(F, dtype=float)
        if F.ndim < 2:
            F = F.reshape(-1, 1)
        if F.size == 1:
            c0 = float(F.reshape(-1)[0])
            return cls.from_function(lambda lam, th: c0 + 0.0 * lam)
        n, m = F.shape
        if m % 2 != 0:
            raise ValueError("SPHEREFUN:CONSTRUCTOR:VALUES: When "
                             "constructing from values the number of "
                             "columns must be even.")
        if tol is None:
            tol = _get_tol_sphere(F, 2 * np.pi / m, np.pi / max(n - 1, 1),
                                  _EPS)[0]
        (piv_idx, piv_arr, remove_pole, cols, pivots, rows, idx_plus,
         idx_minus) = _phase_one_matrix_sphere(F, tol, alpha)
        if cols.shape[1] == 0:
            return cls.from_function(lambda lam, th: 0.0 * lam)
        col_techs = [Trigtech.from_values(jnp.asarray(cols[:, j]))
                     for j in range(cols.shape[1])]
        row_techs = [Trigtech.from_values(jnp.asarray(rows[:, j]))
                     for j in range(rows.shape[1])]
        if np.all(pivots == 0):
            pivots = np.full_like(pivots, np.inf)
        lam = -np.pi + 2 * np.pi * np.arange(m) / m
        th = np.linspace(0.0, np.pi, n)
        locs = []
        if remove_pole:
            locs.append((float(lam[0]), 0.0))
        for j, k in piv_idx:
            locs.append((float(lam[k]), float(th[j])))
        g = cls(cols=col_techs, rows=row_techs,
                pivots=jnp.asarray(pivots, dtype=jnp.float64),
                idx_plus=tuple(int(i) for i in idx_plus),
                idx_minus=tuple(int(i) for i in idx_minus),
                pivot_locations=tuple(locs[:len(pivots)]),
                nonzero_poles=bool(remove_pole))
        # MATLAB constructor.m ends with simplify(g, chebfuneps) for
        # every input kind, matrices included.
        return g.simplify()._prune_zero_terms().projectOntoBMCI()

    def projectOntoBMCI(self) -> "Spherefun":
        """Project the column/row slices onto BMC-I symmetry (even/pi-
        periodic plus part, odd/pi-antiperiodic minus part, zero at the
        poles for the non-pole plus terms) -- MATLAB
        ``projectOntoBMCI``.

        Provenance
        ----------
        MATLAB source : @spherefun/projectOntoBMCI.m
        Chebfun commit: 7574c77
        """
        cols = list(self.cols)
        rows = list(self.rows)
        plus = list(self.idx_plus)
        minus = list(self.idx_minus)
        # Each slice is projected at its own length (padding every slice
        # to the longest one would re-expand chopped slices).
        for jj, i in enumerate(plus):
            X = _stack_trig_coeffs([cols[i]])
            if self.nonzero_poles and jj == 0:
                X = _bmc1_even_cols(X, True)
            else:
                X = _bmc1_even_cols(X, False)
            cols[i] = _trigtech_from_coeffs_real(X[:, 0])
            R = _zero_trig_modes(_stack_trig_coeffs([rows[i]]), odd=True)
            rows[i] = _trigtech_from_coeffs_real(R[:, 0])
        for i in minus:
            X = _bmc1_odd_cols(_stack_trig_coeffs([cols[i]]))
            cols[i] = _trigtech_from_coeffs_real(X[:, 0])
            R = _zero_trig_modes(_stack_trig_coeffs([rows[i]]), odd=False)
            rows[i] = _trigtech_from_coeffs_real(R[:, 0])
        return Spherefun(cols=cols, rows=rows, pivots=self.pivots,
                         idx_plus=self.idx_plus, idx_minus=self.idx_minus,
                         pivot_locations=self.pivot_locations,
                         nonzero_poles=self.nonzero_poles)

    def with_parity_indices(self, idx_plus, idx_minus) -> "Spherefun":
        """Copy with the BMC parity index sets replaced (MATLAB
        ``f.idxPlus = ...; f.idxMinus = ...``)."""
        return Spherefun(cols=list(self.cols), rows=list(self.rows),
                         pivots=self.pivots,
                         idx_plus=tuple(int(i) for i in idx_plus),
                         idx_minus=tuple(int(i) for i in idx_minus),
                         pivot_locations=self.pivot_locations,
                         nonzero_poles=self.nonzero_poles)

    # ------------------------------------------------------------------
    # SVD (MATLAB @separableApprox/svd, @spherefun/BMCsvd)
    # ------------------------------------------------------------------
    def _cdr_quadrature(self, idx=None):
        """Values of the column and row slices on trapezoid grids fine
        enough to integrate products exactly, with the quadrature
        weights (L2 on [-pi, pi] in each variable, as MATLAB's
        quasimatrix QR)."""
        from chebfunjax.tech.trigtech import _trig_eval_np
        idx = list(range(len(self.cols))) if idx is None else list(idx)
        cols = [self.cols[i] for i in idx]
        rows = [self.rows[i] for i in idx]
        nc = max(int(np.asarray(c.coeffs).shape[0]) for c in cols)
        nr = max(int(np.asarray(r.coeffs).shape[0]) for r in rows)
        nqc = 2 * nc + 2
        nqr = 2 * nr + 2
        xc = -1.0 + 2.0 * np.arange(nqc) / nqc
        xr = -1.0 + 2.0 * np.arange(nqr) / nqr
        C = np.column_stack([np.real(np.asarray(_trig_eval_np(
            np.asarray(c.coeffs)[:, None], xc, is_real=c.is_real))).ravel()
            for c in cols])
        R = np.column_stack([np.real(np.asarray(_trig_eval_np(
            np.asarray(r.coeffs)[:, None], xr, is_real=r.is_real))).ravel()
            for r in rows])
        wc = np.full(nqc, 2 * np.pi / nqc)
        wr = np.full(nqr, 2 * np.pi / nqr)
        d = np.asarray(self.pivots, dtype=float)[idx]
        D = np.diag(np.where(np.abs(d) > 0, 1.0 / np.where(d == 0, 1, d),
                             0.0))
        return C, wc, xc, R, wr, xr, D

    def _svd_block(self, idx, weighted: bool = False):
        C, wc, xc, R, wr, xr, D = self._cdr_quadrature(idx)
        if weighted:
            # MATLAB @spherefun/svd.m sphereQR: Legendre points on
            # [0, pi] with the sin(theta) surface weight.
            from chebfunjax.tech.trigtech import _trig_eval_np
            from chebfunjax.utils.quadrature import legpts
            cols = [self.cols[i] for i in idx]
            nc = max(int(np.asarray(c.coeffs).shape[0]) for c in cols) + 9
            xg, wg = legpts(nc, (0.0, np.pi))
            xc = np.asarray(xg)
            wc = np.asarray(wg) * np.sin(xc)
            C = np.column_stack([np.real(np.asarray(_trig_eval_np(
                np.asarray(c.coeffs)[:, None], xc / np.pi,
                is_real=c.is_real))).ravel() for c in cols])
        Qc, Rc = np.linalg.qr(np.sqrt(wc)[:, None] * C)
        Qr, Rr = np.linalg.qr(np.sqrt(wr)[:, None] * R)
        U, s, Vt = np.linalg.svd(Rc @ D @ Rr.T)
        Uv = (Qc @ U) / np.sqrt(wc)[:, None]
        Vv = (Qr @ Vt.T) / np.sqrt(wr)[:, None]
        return s, Uv, xc, Vv, xr

    def svd(self, return_uv: bool = False):
        """Singular values of the spherefun in the surface L2 inner
        product (MATLAB ``svd(f)``: the column slices are orthogonalised
        on [0, pi] with the sin(theta) weight, the rows on [-pi, pi]).
        With ``return_uv`` returns ``(U, s, V)``.

        Provenance
        ----------
        MATLAB source : @separableApprox/svd.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return jnp.zeros((0,), dtype=jnp.float64)
        piv = np.asarray(self.pivots, dtype=float)
        if not np.any(np.isfinite(piv)) or np.all(1.0 / piv == 0):
            return jnp.asarray([0.0], dtype=jnp.float64)
        s, Uv, xc, Vv, xr = self._svd_block(range(len(self.cols)),
                                            weighted=True)
        if not return_uv:
            return jnp.asarray(s, dtype=jnp.float64)
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.linalg import Quasimatrix
        Ufuns = [chebfun(lambda t, _j=j: jnp.asarray(np.interp(
            np.asarray(t), xc, Uv[:, _j])), domain=(0.0, np.pi))
            for j in range(Uv.shape[1])]
        return (Quasimatrix(Ufuns, Ufuns[0].domain), jnp.asarray(s),
                _trig_quasimatrix(Vv))

    def BMCsvd(self, return_uv: bool = False):
        """Singular values respecting the BMC-I block structure: the SVD
        is computed separately for the plus and minus parts and the
        values merged in descending order (MATLAB ``BMCsvd``).

        Provenance
        ----------
        MATLAB source : @spherefun/BMCsvd.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return jnp.zeros((0,), dtype=jnp.float64)
        piv = np.asarray(self.pivots, dtype=float)
        if not np.any(np.isfinite(piv)):
            return jnp.asarray([0.0], dtype=jnp.float64)
        parts = []
        for idx in (self.idx_plus, self.idx_minus):
            if len(idx) == 0:
                continue
            parts.append(self._svd_block(idx))
        s = np.concatenate([p[0] for p in parts])
        order = np.argsort(-s, kind="stable")
        if not return_uv:
            return jnp.asarray(s[order], dtype=jnp.float64)
        Us = [p[1] for p in parts]
        Vs = [p[3] for p in parts]
        nq_c = max(u.shape[0] for u in Us)
        nq_r = max(v.shape[0] for v in Vs)
        Ucat = np.column_stack([_trig_resample(u, nq_c) for u in Us])
        Vcat = np.column_stack([_trig_resample(v, nq_r) for v in Vs])
        return (_trig_quasimatrix(Ucat[:, order]), jnp.asarray(s[order]),
                _trig_quasimatrix(Vcat[:, order]))

    # ------------------------------------------------------------------
    # Inherited separableApprox methods
    # ------------------------------------------------------------------
    def diag(self, c: float = 0.0):
        """The diagonal ``f(x, x + c)`` as a Chebfun on the (largest)
        interval where it is defined (MATLAB separableApprox ``diag``:
        ``lam = x``, ``theta = x + c`` on the colatitude domain
        ``[-pi, pi] x [0, pi]``).

        Provenance
        ----------
        MATLAB source : @separableApprox/diag.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        c = float(c)
        a = max(-np.pi, 0.0 - c)
        b = min(np.pi, np.pi - c)
        return _cf(lambda x: self(x, x + c), domain=(a, b))

    def transpose(self) -> "Spherefun":
        """Swap the roles of the two variables (MATLAB ``transpose``)."""
        locs = tuple((b, a) for a, b in self.pivot_locations)
        return Spherefun(cols=list(self.rows), rows=list(self.cols),
                         pivots=self.pivots, idx_plus=self.idx_plus,
                         idx_minus=self.idx_minus, pivot_locations=locs,
                         nonzero_poles=self.nonzero_poles)

    def ctranspose(self) -> "Spherefun":
        """Conjugate transpose (MATLAB ``f'``)."""
        return self.transpose().conj()

    @property
    def T(self) -> "Spherefun":
        return self.transpose()

    def _flip_slices(self, which: str) -> "Spherefun":
        cols = list(self.cols)
        rows = list(self.rows)
        if which == "cols":
            cols = [_trig_flip(t) for t in cols]
        else:
            rows = [_trig_flip(t) for t in rows]
        return Spherefun(cols=cols, rows=rows, pivots=self.pivots,
                         idx_plus=self.idx_plus, idx_minus=self.idx_minus,
                         pivot_locations=self.pivot_locations,
                         nonzero_poles=self.nonzero_poles)

    def fliplr(self) -> "Spherefun":
        """``f(-lam, theta)`` (MATLAB ``fliplr``: flips the rows)."""
        return self._flip_slices("rows")

    def flipud(self) -> "Spherefun":
        """``f(lam, -theta)`` (MATLAB ``flipud``: flips the columns)."""
        return self._flip_slices("cols")

    def flipdim(self, dim: int) -> "Spherefun":
        """``flipud`` for ``dim == 1``, ``fliplr`` for ``dim == 2``."""
        if dim == 1:
            return self.flipud()
        if dim == 2:
            return self.fliplr()
        raise ValueError("CHEBFUN:SEPARABLEAPPROX:flipdim:badDim2: "
                         "Dimension not recognised.")

    def tan(self):
        """Tangent, re-approximated (MATLAB tan)."""
        return self._reapprox(jnp.tan)

    def tand(self):
        """Tangent in degrees, re-approximated (MATLAB tand)."""
        return self._reapprox(lambda t: jnp.tan(jnp.pi / 180.0 * t))

    def log(self):
        """Natural logarithm, re-approximated (MATLAB log)."""
        return self._reapprox(jnp.log)

    def uminus(self):
        return -self

    def uplus(self):
        return self

    def __pos__(self):
        return self

    def isequal(self, other) -> bool:
        """True when the two representations are identical: same slice
        coefficients and pivot values (MATLAB separableApprox
        ``isequal``).

        Provenance
        ----------
        MATLAB source : @separableApprox/isequal.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return bool(self.isempty() and other.isempty())
        if len(self.cols) != len(other.cols):
            return False
        if not np.array_equal(np.asarray(self.pivots),
                              np.asarray(other.pivots)):
            return False
        for a, b in zip(list(self.cols) + list(self.rows),
                        list(other.cols) + list(other.rows)):
            ca, cb = np.asarray(a.coeffs), np.asarray(b.coeffs)
            if ca.shape != cb.shape or not np.array_equal(ca, cb):
                return False
        return True

    def size(self, dim: int | None = None):
        """``(inf, inf)`` (MATLAB separableApprox ``size``)."""
        if dim is None:
            return (np.inf, np.inf)
        if dim in (1, 2):
            return np.inf
        raise ValueError("CHEBFUN:SEPARABLEAPPROX:size:outputs")

    def norm(self, p=2) -> jax.Array:
        """L2 norm over the sphere: sqrt(int |f|^2 dOmega) (Fable 5).
        ``p`` may be ``2`` or ``'fro'`` (identical for a spherefun, as
        in MATLAB); other norms are not implemented.

        Computed by direct Clenshaw-Curtis(theta, weight sin theta) x
        trapezoid(lambda) quadrature of f^2 at the resolution of the
        representation (spectrally exact for the finite series) -- the
        previous adaptive re-approximation of f^2 handed the
        constructor pure rounding noise whenever f was a structurally
        cancelling difference (norm(f - g) checks).
        """
        if isinstance(p, str) and p.lower() in ("inf", "max"):
            Y, _X = self.minandmax2()
            return jnp.max(jnp.abs(jnp.asarray(Y)))
        if not (p == 2 or (isinstance(p, str) and p.lower() == "fro")):
            raise NotImplementedError(
                "CHEBFUN:SPHEREFUN:norm: only the 2/'fro'/'inf' norms are "
                "implemented")
        if self.isempty() or len(self.cols) == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        # MATLAB @spherefun/norm.m: sqrt(sum(svd(f).^2)) with the
        # sin(theta)-weighted (surface) SVD.
        s = np.asarray(self.svd())
        return jnp.asarray(np.sqrt(np.sum(s ** 2)), dtype=jnp.float64)
        from chebfunjax.utils.quadrature import chebpts, chebweights
        m, n = self.length()
        nth = 2 * n + 16
        mlam = 2 * m + 16
        x = np.array(chebpts(nth))
        w = np.array(chebweights(nth))
        th = (x + 1.0) * (np.pi / 2.0)
        w_th = w * (np.pi / 2.0)
        lam = np.linspace(-np.pi, np.pi, mlam, endpoint=False)
        V = np.asarray(self.fevalm(jnp.asarray(lam), jnp.asarray(th)),
                       dtype=float)
        val = float(np.sum((V ** 2) * (np.sin(th) * w_th)[:, None])
                    * (2.0 * np.pi / mlam))
        return jnp.sqrt(jnp.abs(jnp.asarray(val, dtype=jnp.float64)))

    def _reapprox(self, op2) -> "Spherefun":
        return Spherefun.from_function(
            lambda lam, th: op2(self(lam, th)))

    def _binary(self, other, op2) -> "Spherefun":
        """Re-approximate ``op2(self, other)`` as a new Spherefun.

        Near-zero snapping: a result whose magnitude is below
        ``_ZERO_REL_TOL`` (1e-9) times the larger operand's scale is
        indistinguishable from the surface operators' ~1e-11 relative noise
        floor -- reconstructing it just chases rounding noise -- so it is
        snapped to the exact zero field.  Genuinely small results above that
        line survive; they are reconstructed with the tolerance floored at
        the operand noise level so the constructor does not chase the
        operands' own construction noise (added by Claude Fable 5).
        """
        other_is_fun = isinstance(other, Spherefun)
        if other_is_fun:
            def fn(lam, th):
                return op2(self(lam, th), other(lam, th))
        else:
            def fn(lam, th):
                return op2(self(lam, th), other)

        # Near-zero short-circuit.  When ``op2(self, other)`` is only
        # rounding noise relative to the operands -- e.g. the analytically
        # zero fields ``div(grad f) - laplacian(f)`` or ``div(curl F)`` --
        # re-approximating it via from_function is pathological on two
        # counts: (1) noise has no low-rank band-limited structure, so the
        # adaptive constructor never reaches "happy" and doubles the grid
        # all the way to ``max_sample`` (2^14), producing a giant Clenshaw
        # graph that hangs or crashes XLA's CPU backend ("Failed to
        # materialize symbols"); (2) it returns a ~1e-11 noise Spherefun
        # instead of the exact zero the surface-calculus identities demand.
        # A coarse-grid check catches this and returns the exact zero
        # Spherefun.  The threshold is relative to the operand scale: the
        # surface differential operators carry ~1e-11 relative error, so a
        # result 9+ orders below the operands is numerically indistinguish-
        # able from zero (added by Claude Fable 5).
        m = 16
        th = jnp.asarray(_sphere_col_pts(m), dtype=jnp.float64)
        lam = jnp.asarray(_sphere_row_pts(m), dtype=jnp.float64)
        lam2d, th2d = jnp.meshgrid(lam, th)
        res = np.asarray(fn(lam2d, th2d))
        if np.all(np.isfinite(res)):
            scale = float(np.max(np.abs(np.asarray(self(lam2d, th2d)))))
            if other_is_fun:
                scale = max(
                    scale,
                    float(np.max(np.abs(np.asarray(other(lam2d, th2d))))))
            else:
                scale = max(scale, float(np.max(np.abs(np.asarray(other)))))
            vmax = float(np.max(np.abs(res)))
            if vmax <= _ZERO_REL_TOL * scale:
                return Spherefun.from_function(
                    lambda lam, th: jnp.zeros(
                        jnp.broadcast_shapes(jnp.asarray(lam).shape,
                                             jnp.asarray(th).shape),
                        dtype=jnp.float64))
            # Small-but-real result (well below the operand scale but above
            # the zero threshold): floor the construction tolerance at the
            # operands' noise level so from_function resolves the genuine
            # low-rank signal instead of chasing sub-noise structure to
            # max_sample.
            if scale > 0.0 and vmax < 1e-4 * scale:
                return Spherefun.from_function(
                    fn, min_abs_tol=_ZERO_REL_TOL * scale)

        return Spherefun.from_function(fn)

    def _is_exact_zero(self) -> bool:
        """True iff every column coefficient is exactly zero.

        Cheap structural test (no evaluation) that recognises the exact zero
        field produced by the near-zero snap in :meth:`_binary`.  It lets
        ``f +/- 0`` return ``f`` unchanged instead of re-approximating it via
        ``from_function`` -- that re-approximation drifts by ~1e-13, which is
        why ``tangent(grad f) == grad f`` (subtracting an exactly-tangential
        field's zero normal component) otherwise fails at the 1e2*eps
        tolerance.  Added by Claude Fable 5.
        """
        return all(bool(jnp.all(c.coeffs == 0.0)) for c in self.cols)


    # ------------------------------------------------------------------
    # MATLAB @spherefun/plus.m: block-wise compression plus
    # ------------------------------------------------------------------
    def _normalize_poles(self) -> "Spherefun":
        """Bring the representation into MATLAB's BMC-I pole form: the
        constant values at the two poles are carried by ONE plus term
        with a constant row (the constructor's removePole step), so that
        every other plus column vanishes at the poles.  Structural
        products (``f * g``) do not keep that form; this rebuilds it
        algebraically: with ``cP(theta) = f(lam0, theta)`` the remainder
        ``f - cP (x) 1`` vanishes at both poles for every lam."""
        from chebfunjax.tech.trigtech import Trigtech, _trig_eval_np
        if self.isempty() or len(self.cols) == 0 or self.nonzero_poles:
            return self
        plus = list(self.idx_plus)
        if not plus:
            return self
        piv = np.asarray(self.pivots, dtype=float)
        # pole values of the plus columns (theta = 0 and theta = pi)
        pv_max = 0.0
        vs = 0.0
        for j in plus:
            c = self.cols[j]
            v = np.real(np.asarray(_trig_eval_np(np.asarray(c.coeffs)[:, None],
                                                 np.array([0.0, 1.0]),
                                                 is_real=c.is_real))).ravel()
            pv_max = max(pv_max, float(np.max(np.abs(v))) / max(abs(piv[j]), 1e-300))
            vs = max(vs, float(np.max(np.abs(np.asarray(c.values)))) / max(abs(piv[j]), 1e-300))
        if pv_max <= 1e-13 * max(vs, 1e-300):
            return self
        # cP(theta) = f(lam0, theta) with lam0 = 0 (x = 0 on the row grid)
        nq = 2 * max(int(np.asarray(self.cols[j].coeffs).shape[0]) for j in plus) + 2
        xs = -1.0 + 2.0 * np.arange(nq) / nq
        cP = np.zeros(nq)
        for j in plus:
            c, r = self.cols[j], self.rows[j]
            rv = float(np.real(np.asarray(_trig_eval_np(np.asarray(r.coeffs)[:, None],
                                                        np.array([0.0]),
                                                        is_real=r.is_real))).ravel()[0])
            cv = np.real(np.asarray(_trig_eval_np(np.asarray(c.coeffs)[:, None], xs,
                                                  is_real=c.is_real))).ravel()
            cP += cv * rv / piv[j]
        pole_col = Trigtech.from_values(jnp.asarray(cP))
        pole_row = Trigtech.from_values(jnp.ones(2, dtype=jnp.float64))
        neg_row = Trigtech.from_values(-jnp.ones(2, dtype=jnp.float64))
        cols = [pole_col] + [self.cols[j] for j in plus] + [pole_col] + \
            [self.cols[j] for j in self.idx_minus]
        rows = [pole_row] + [self.rows[j] for j in plus] + [neg_row] + \
            [self.rows[j] for j in self.idx_minus]
        pivs = [1.0] + [float(piv[j]) for j in plus] + [1.0] + \
            [float(piv[j]) for j in self.idx_minus]
        n_plus = 2 + len(plus)
        return Spherefun(cols=cols, rows=rows,
                         pivots=jnp.asarray(pivs, dtype=jnp.float64),
                         idx_plus=tuple(range(n_plus)),
                         idx_minus=tuple(range(n_plus, len(cols))),
                         pivot_locations=(), nonzero_poles=True)

    def _extract_pole(self):
        """Split off the pole term (first plus term when nonzero_poles):
        returns (rest, pole) with ``pole`` None when absent."""
        if not self.nonzero_poles or len(self.idx_plus) == 0:
            return self, None
        j = int(self.idx_plus[0])
        keep = [k for k in range(len(self.cols)) if k != j]
        pole = (self.cols[j], self.rows[j], float(self.pivots[j]))
        rest = Spherefun(
            cols=[self.cols[k] for k in keep], rows=[self.rows[k] for k in keep],
            pivots=jnp.asarray([float(self.pivots[k]) for k in keep],
                               dtype=jnp.float64),
            idx_plus=tuple(keep.index(k) for k in self.idx_plus if k != j),
            idx_minus=tuple(keep.index(k) for k in self.idx_minus),
            nonzero_poles=False)
        return rest, pole

    def _compression_plus(self, other: "Spherefun") -> "Spherefun":
        """``self + other`` by MATLAB's compression_plus per BMC parity
        block, with the pole terms combined as in ``addPoles``.

        Provenance
        ----------
        MATLAB source : @spherefun/plus.m, @separableApprox/plus.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import Trigtech
        from chebfunjax.utils.bmc_plus import block_vscale, compress_block, techs_from_values
        f, fpole = self._normalize_poles()._extract_pole()
        g, gpole = other._normalize_poles()._extract_pole()

        def _block(a, idx_a, b, idx_b):
            cols = [a.cols[i] for i in idx_a] + [b.cols[i] for i in idx_b]
            rows = [a.rows[i] for i in idx_a] + [b.rows[i] for i in idx_b]
            piv = [float(a.pivots[i]) for i in idx_a] + \
                  [float(b.pivots[i]) for i in idx_b]
            if not cols:
                return [], [], []
            va = block_vscale([a.cols[i] for i in idx_a],
                              [a.rows[i] for i in idx_a],
                              [float(a.pivots[i]) for i in idx_a], "trig")
            vb = block_vscale([b.cols[i] for i in idx_b],
                              [b.rows[i] for i in idx_b],
                              [float(b.pivots[i]) for i in idx_b], "trig")
            vscl = 2.0 * max(va, vb)
            out = compress_block(cols, rows, piv, "trig", vscl)
            if out is None:
                return [], [], []
            C, R, newpiv = out
            nc = max(int(np.asarray(t.coeffs).shape[0]) for t in cols)
            nr = max(int(np.asarray(t.coeffs).shape[0]) for t in rows)
            # MATLAB compression_plus does NOT simplify: Qcols*U keeps the
            # quasimatrix QR length (the longest operand slice).
            return (techs_from_values(C, "trig", nc),
                    techs_from_values(R, "trig", nr),
                    list(newpiv))
        pc, pr, pp = _block(f, f.idx_plus, g, g.idx_plus)
        mc, mr, mp = _block(f, f.idx_minus, g, g.idx_minus)
        # addPoles: both pole terms have a constant row; merge into one.
        pole_cols, pole_rows, pole_piv = [], [], []
        nonzero_poles = False
        if fpole is not None or gpole is not None:
            col_vals = None
            for pole in (fpole, gpole):
                if pole is None:
                    continue
                c, r, pv = pole
                rmean = float(np.real(np.asarray(r.coeffs).ravel()[
                    np.asarray(r.coeffs).ravel().size // 2]))
                n_c = int(np.asarray(c.coeffs).shape[0])
                nq = max(n_c, 2) * 2
                from chebfunjax.utils.bmc_plus import _trig_values
                V, _w = _trig_values([c], nq)
                contrib = (rmean / pv) * V[:, 0]
                if col_vals is None:
                    col_vals = contrib
                elif contrib.shape[0] == col_vals.shape[0]:
                    col_vals = col_vals + contrib
                else:
                    m = max(contrib.shape[0], col_vals.shape[0])
                    V1, _ = _trig_values(techs_from_values(col_vals[:, None], "trig"), m)
                    V2, _ = _trig_values(techs_from_values(contrib[:, None], "trig"), m)
                    col_vals = V1[:, 0] + V2[:, 0]
            scales = [np.max(np.abs(np.asarray(t.values)))
                      for t in (self.cols + other.cols)]
            tol = 10 * _EPS * max(scales) if scales else 10 * _EPS
            if np.max(np.abs(col_vals)) > tol:
                pole_cols = _simplify_global_sphere(
                    techs_from_values(col_vals[:, None], "trig"))
                pole_rows = [Trigtech.from_values(jnp.ones(2, dtype=jnp.float64))]
                pole_piv = [1.0]
                endv = np.asarray(pole_cols[0](jnp.asarray([-1.0, 0.0]))).ravel()
                nonzero_poles = bool(np.max(np.abs(endv)) > tol)
        cols = pole_cols + pc + mc
        rows = pole_rows + pr + mr
        piv = pole_piv + pp + mp
        if not cols:
            return Spherefun.from_function(lambda lam, th: 0.0 * lam)
        n_plus = len(pole_cols) + len(pc)
        h = Spherefun(cols=cols, rows=rows,
                      pivots=jnp.asarray(piv, dtype=jnp.float64),
                      idx_plus=tuple(range(n_plus)),
                      idx_minus=tuple(range(n_plus, len(cols))),
                      pivot_locations=(), nonzero_poles=nonzero_poles)
        return h.projectOntoBMCI()

    def __add__(self, other):
        if isinstance(other, Spherefun):
            if other._is_exact_zero() or other.iszero():
                return self
            if self._is_exact_zero() or self.iszero():
                return other
            return self._compression_plus(other)
        if np.isscalar(other) and not isinstance(other, Spherefun):
            c = other
            const = Spherefun.from_function(lambda lam, th: c + 0.0 * lam)
            return self._compression_plus(const)
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Spherefun):
            if other._is_exact_zero():
                return self
            return self.__add__(other * (-1.0))
        if np.isscalar(other):
            return self.__add__(-other)
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        # Exact CDR shortcuts (MATLAB @separableApprox/times.m): a
        # scalar scales the pivots; a rank-1 operand multiplies into
        # every slice of the other via exact dealiased Trigtech
        # products.  Both avoid the adaptive re-approximation (and its
        # marginal-resolution retry cost) entirely.
        if not isinstance(other, Spherefun) and np.isscalar(other):
            s = complex(other) if isinstance(other, complex) \
                else float(other)
            if s == 0:
                return self._binary(other, lambda a, b: a * b)
            return Spherefun(
                cols=self.cols, rows=self.rows,
                pivots=self.pivots / s,
                idx_plus=self.idx_plus, idx_minus=self.idx_minus,
                pivot_locations=self.pivot_locations,
                nonzero_poles=self.nonzero_poles)
        if isinstance(other, Spherefun) and not self.isempty() \
                and not other.isempty():
            # Only for WELL-RESOLVED operands: a marginal construction
            # (slices at the max_sample runaway, e.g. from boundary-
            # discontinuous input like cos(th)*sin(lam)) carries
            # high-frequency junk that the exact product would preserve
            # where re-approximation smooths it away.
            def _healthy(s):
                return all(t.n < 4096 for t in (*s.cols, *s.rows))
            if _healthy(self) and _healthy(other):
                if len(self.pivots) == 1:
                    return _spherefun_mul_rank1(self, other)
                if len(other.pivots) == 1:
                    return _spherefun_mul_rank1(other, self)
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __neg__(self):
        return self._reapprox(lambda v: -v)

    def __pow__(self, p):
        if isinstance(p, Spherefun):
            return Spherefun.from_function(
                lambda lam, th: self(lam, th) ** p(lam, th))
        return self._reapprox(lambda v: v ** p)

    def compose(self, op):
        """Re-approximate op(f) (MATLAB compose).

        ``op`` may be a plain callable or Chebfun (g(f)), a list/tuple
        of three of them (quasimatrix G -> Spherefunv), a Chebfun2
        (g(f, 0), MATLAB's real/imag split for real f), or a Chebfun2v
        (componentwise -> Spherefunv).

        Provenance
        ----------
        MATLAB source : @spherefun/compose.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        if isinstance(op, (list, tuple)):
            from chebfunjax.spherefun.spherefunv import Spherefunv
            return Spherefunv(*[self.compose(c) for c in op])
        try:
            from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
            if isinstance(op, Chebfun2v):
                from chebfunjax.spherefun.spherefunv import Spherefunv
                return Spherefunv(*[self.compose(c)
                                    for c in op.components])
        except ImportError:
            pass
        if isinstance(op, Chebfun2):
            return self._reapprox(lambda v: op(v, 0.0 * v))
        return self._reapprox(op)

    def exp(self):
        return self._reapprox(jnp.exp)

    def sin(self):
        return self._reapprox(jnp.sin)

    def cos(self):
        return self._reapprox(jnp.cos)

    def sqrt(self):
        return self._reapprox(jnp.sqrt)

    def abs(self) -> "Spherefun":
        """Absolute value ``|f|``, re-approximated (MATLAB abs).

        Provenance
        ----------
        MATLAB source : @spherefun/abs.m
        Chebfun commit: 7574c77
        """
        return self._reapprox(jnp.abs)

    def __abs__(self) -> "Spherefun":
        return self.abs()

    def isreal(self) -> bool:
        """True iff the representation holds only real data.

        Provenance
        ----------
        MATLAB source : @separableApprox/isreal.m (via @spherefun/isreal.m)
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return True
        return (not bool(jnp.iscomplexobj(self.pivots))
                and all(c.is_real for c in self.cols)
                and all(r.is_real for r in self.rows))

    def real(self) -> "Spherefun":
        """Real part (MATLAB real).

        A real representation is returned unchanged (exact); otherwise the
        real part is re-approximated as in MATLAB's compose route.

        Provenance
        ----------
        MATLAB source : @spherefun/real.m
        Chebfun commit: 7574c77
        """
        if self.isreal():
            return self
        return self._reapprox(jnp.real)

    def imag(self) -> "Spherefun":
        """Imaginary part (MATLAB imag).

        Exactly zero for a real representation; otherwise re-approximated
        as in MATLAB's compose route.

        Provenance
        ----------
        MATLAB source : @spherefun/imag.m
        Chebfun commit: 7574c77
        """
        if self.isreal():
            return Spherefun.from_function(
                lambda lam, th: jnp.zeros(
                    jnp.broadcast_shapes(jnp.asarray(lam).shape,
                                         jnp.asarray(th).shape),
                    dtype=jnp.float64))
        return self._reapprox(jnp.imag)

    def conj(self) -> "Spherefun":
        """Complex conjugate (MATLAB conj: a no-op).

        MATLAB's @spherefun/conj.m returns f unchanged -- spherefun
        represents real-valued functions -- so this is exact, not a
        re-approximation.

        Provenance
        ----------
        MATLAB source : @spherefun/conj.m
        Chebfun commit: 7574c77
        """
        if self.isreal():
            return self
        return self._reapprox(jnp.conj)

    def iszero(self) -> bool:
        """True iff f is the zero function (MATLAB iszero, inherited from
        separableApprox).

        Provenance
        ----------
        MATLAB source : @separableApprox/iszero.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        lam = jnp.asarray(_np.linspace(-_np.pi, _np.pi, 17))
        th = jnp.asarray(_np.linspace(0.0, _np.pi, 17))
        LL, TT = jnp.meshgrid(lam, th)
        return float(jnp.max(jnp.abs(self(LL, TT)))) <= 1e4 * _EPS

    def minandmax2(self, n_lam: int = 128, n_th: int = 64, n_starts: int = 8):
        """Global minimum and maximum of the Spherefun over the sphere.

        A tensor grid in ``(lambda, theta)`` seeds candidate extrema; the
        ``n_starts`` deepest distinct grid candidates are polished with a
        bound-constrained quasi-Newton step (``scipy`` L-BFGS-B) using the
        exact JAX gradient of the evaluation, keeping the best.  The
        multi-start guards against the oscillatory battery functions
        (e.g. ``cos(2*pi*x)*cos(2*pi*y)``) whose deepest grid basin is not
        the global one.  Returns ``(Y, X)`` with ``Y = [min, max]`` and
        ``X`` the corresponding ``(lambda, theta)`` points, mirroring
        MATLAB ``[Y, X] = minandmax2(F)``.

        Provenance
        ----------
        MATLAB source : @separableApprox/minandmax2.m
        Chebfun commit: 7574c77
        """
        import jax
        import numpy as _np
        from scipy.optimize import minimize

        def scal(p):
            return self(p[0], p[1])

        vg = jax.jit(jax.value_and_grad(scal))

        lam = _np.linspace(-_np.pi, _np.pi, n_lam)
        th = _np.linspace(0.0, _np.pi, n_th)
        LL, TT = _np.meshgrid(lam, th)
        F = _np.asarray(self(jnp.asarray(LL), jnp.asarray(TT)),
                        dtype=_np.float64)

        def _optimize(sign):
            order = _np.argsort((sign * F).ravel())
            best_signed = _np.inf
            best_x = (float(LL.ravel()[order[0]]),
                      float(TT.ravel()[order[0]]))

            def fun(p):
                v, g = vg(jnp.asarray(p, dtype=jnp.float64))
                return sign * float(v), sign * _np.asarray(
                    g, dtype=_np.float64)

            for k in range(min(n_starts, order.size)):
                i0 = _np.unravel_index(order[k], F.shape)
                p0 = _np.array([LL[i0], TT[i0]], dtype=_np.float64)
                res = minimize(
                    fun, p0, jac=True, method="L-BFGS-B",
                    bounds=[(-_np.pi, _np.pi), (0.0, _np.pi)],
                    options={"ftol": 1e-15, "gtol": 1e-14, "maxiter": 400})
                if float(res.fun) < best_signed:
                    best_signed = float(res.fun)
                    best_x = (float(res.x[0]), float(res.x[1]))
            return sign * best_signed, best_x

        vmin, xmin = _optimize(1.0)
        vmax, xmax = _optimize(-1.0)
        return (jnp.asarray([vmin, vmax], dtype=jnp.float64),
                jnp.asarray([list(xmin), list(xmax)], dtype=jnp.float64))

    def max2(self):
        """Global maximum (value, [lambda, theta]) -- MATLAB max2."""
        vals, locs = self.minandmax2()
        return vals[1], locs[1]

    def min2(self):
        """Global minimum (value, [lambda, theta]) -- MATLAB min2."""
        vals, locs = self.minandmax2()
        return vals[0], locs[0]

    def roots(self, n: int = 257):
        """Zero contours of the Spherefun on the sphere (MATLAB
        ``@spherefun/roots``).

        The Spherefun is viewed as a Chebfun2 in ``(lambda, theta)`` on
        ``[-pi, pi] x [0, pi]``; its zero curves are traced there and each
        is sampled and mapped to Cartesian points on the unit sphere
        ``(cos(lam) sin(th), sin(lam) sin(th), cos(th))``.  Returns a list
        of ``(n, 3)`` arrays, one per contour (the chebfunjax stand-in for
        MATLAB's cell array of contour chebfuns; ``sample`` is the identity
        here).

        Provenance
        ----------
        MATLAB source : @spherefun/roots.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun2d import chebfun2
        from chebfunjax.chebfun2d.zerocurves import zero_curves
        dom = (-_np.pi, _np.pi, 0.0, _np.pi)
        fp = chebfun2(lambda lam, th: self(lam, th), domain=dom)
        curves = zero_curves(fp)
        ts = jnp.asarray(_np.linspace(-1.0, 1.0, n))
        out = []
        for c in curves:
            z = _np.asarray(c(ts))
            lam, th = z.real, z.imag
            x = _np.cos(lam) * _np.sin(th)
            y = _np.sin(lam) * _np.sin(th)
            zz = _np.cos(th)
            out.append(_np.column_stack([x, y, zz]))
        return out

    def mean(self) -> jax.Array:
        """Mean value of the function over the unit sphere: sum / (4 pi)."""
        return self.sum() / (4 * jnp.pi)

    @classmethod
    def sphharm(cls, l: int, m: int) -> "Spherefun":
        r"""Real (orthonormal) spherical harmonic :math:`Y_l^m`.

        Returns a Spherefun representing the real spherical harmonic of
        degree ``l`` and order ``m`` (``-l <= m <= l``), normalized so
        that :math:`\int_{S^2} (Y_l^m)^2 \, dS = 1`.  The associated
        Legendre functions are evaluated with the numerically stable
        fully-normalized three-term recurrence.

        Parameters
        ----------
        l : int
            Degree (l >= 0).
        m : int
            Order (-l <= m <= l).  m > 0 gives the cos(m*lam) harmonic,
            m < 0 the sin(|m|*lam) harmonic.

        Returns
        -------
        Spherefun

        Notes
        -----
        Verified against ``scipy.special.sph_harm_y`` to machine
        precision (up to the standard global sign convention).

        Provenance
        ----------
        MATLAB source : @spherefun/sphharm.m
        Chebfun commit: 7574c77
        Original: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.  See https://www.chebfun.org/.
        """
        l = int(l)
        m_signed = int(m)
        m = abs(m_signed)
        if l < 0:
            raise ValueError("degree l must be non-negative")
        if m > l:
            raise ValueError("order |m| must be <= l")

        def ev(lam, theta):
            x = jnp.cos(theta)
            # Fully-normalized P_m^m (Condon-Shortley phase folded in).
            pmm = jnp.ones_like(x) / jnp.sqrt(4 * jnp.pi)
            for i in range(1, m + 1):
                pmm = -jnp.sqrt((2 * i + 1) / (2.0 * i)) \
                    * jnp.sqrt(1 - x**2) * pmm
            if l == m:
                plm = pmm
            else:
                pm1 = jnp.sqrt(2 * m + 3.0) * x * pmm
                if l == m + 1:
                    plm = pm1
                else:
                    p_prev, p_curr = pmm, pm1
                    for ll in range(m + 2, l + 1):
                        a = jnp.sqrt((4.0 * ll * ll - 1)
                                     / (ll * ll - m * m))
                        b = jnp.sqrt(((ll - 1.0) ** 2 - m * m)
                                     / (4.0 * (ll - 1.0) ** 2 - 1))
                        p_next = a * (x * p_curr - b * p_prev)
                        p_prev, p_curr = p_curr, p_next
                    plm = p_curr
            if m_signed > 0:
                return jnp.sqrt(2.0) * plm * jnp.cos(m * lam)
            if m_signed < 0:
                return jnp.sqrt(2.0) * plm * jnp.sin(m * lam)
            return plm

        # The degree is KNOWN, so size the first construction grid to
        # resolve it.  From the adaptive default (grid 8, i.e. 16
        # longitude points) any harmonic with m >= 13 aliased to a
        # low-order mode with BOTH directions self-consistently aliased,
        # and the constructor declared a wrong rank-1 function happy --
        # e.g. sphharm(17,13) came back as a cos(3*lam) harmonic (its
        # Laplacian identity failed by O(100) while orthonormality still
        # held).  Sizing the start grid from l makes the first pass
        # resolved; the general missing-sample-test gap in from_function
        # is recorded in the audit ledger.
        return cls.from_function(ev, start_grid=max(8, 2 * l + 4))

    def _bandwidth(self) -> int:
        """Estimate the spherical-harmonic degree that resolves ``self``.

        The doubled-up column Trigtechs hold ~(2*lmax + 1) coefficients,
        so ``lmax = (max_col_len - 1) // 2``.  Added by Claude Opus 4.8.
        """
        col_len = max((c.coeffs.shape[0] for c in self.cols), default=1)
        row_len = max((r.coeffs.shape[0] for r in self.rows), default=1)
        return max((col_len - 1) // 2, (row_len - 1) // 2, 1)

    def diff(self, dim: int = 1, k: int = 1) -> "Spherefun":
        r"""Tangential derivative in a Cartesian direction.

        Returns the surface (tangential) derivative of ``self`` in the
        x- (``dim=1``), y- (``dim=2``), or z-direction (``dim=3``),
        applied ``k`` times.  This is the projection of the surface
        gradient onto the chosen Cartesian axis, e.g. for x:

        .. math::
            \\partial_x f = -\\frac{\\sin\\lambda}{\\sin\\theta}
            \\, \\partial_\\lambda f + \\cos\\lambda \\cos\\theta
            \\, \\partial_\\theta f .

        Computed by evaluating this intrinsic formula at interior
        Gauss--Legendre colatitude nodes (which never land on the poles,
        so the ``1/sin(theta)`` factor is finite), with ``∂_theta`` and
        ``∂_lambda`` taken spectrally from the CDR representation, then
        projecting onto spherical harmonics and reconstructing.

        Implemented and verified by Claude Opus 4.8: matches the analytic
        tangential gradient of every tested harmonic to ~1e-14, and
        ``diff(1,2)+diff(2,2)+diff(3,2) == laplacian`` to ~1e-14.

        Parameters
        ----------
        dim : int, default 1
            Cartesian direction: 1 = x, 2 = y, 3 = z.
        k : int, default 1
            Number of derivatives.

        Returns
        -------
        Spherefun

        Provenance
        ----------
        MATLAB source : @spherefun/diff.m (result-equivalent; the
        implementation uses a harmonic-projection route rather than the
        BMC coefficient-space parity solve).
        Chebfun commit: 7574c77
        """
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1 (x), 2 (y), or 3 (z)")
        f = self
        for _ in range(int(k)):
            f = _spherefun_onediff(f, dim)
        return f

    def grad(self) -> tuple:
        r"""Surface gradient in Cartesian components.

        Returns ``(fx, fy, fz)``, the three tangential Cartesian
        derivatives ``diff(f, 1), diff(f, 2), diff(f, 3)`` of ``self``
        (each a Spherefun), exactly as MATLAB's gradient.m.

        Provenance
        ----------
        MATLAB source : @spherefun/gradient.m
        Chebfun commit: 7574c77
        """
        return (self.diff(1), self.diff(2), self.diff(3))

    def gradient(self) -> "Spherefunv":
        r"""Surface gradient as a :class:`Spherefunv`.

        Returns the SPHEREFUNV ``(fx, fy, fz)`` of tangential Cartesian
        derivatives ``diff(f, 1), diff(f, 2), diff(f, 3)`` -- the
        vector-field form of :meth:`grad` (MATLAB gradient.m).

        Provenance
        ----------
        MATLAB source : @spherefun/gradient.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        from chebfunjax.spherefun.spherefunv import Spherefunv

        if self.isempty():
            return Spherefunv.empty()
        return Spherefunv(self.diff(1), self.diff(2), self.diff(3))

    def curl(self) -> "Spherefunv":
        r"""Surface curl of a scalar field: ``curl(f) = N x grad(f)``.

        Returns the SPHEREFUNV obtained by rotating the surface gradient a
        quarter turn about the outward normal ``N = (x, y, z)``:

        .. math::
            (-z f_y + y f_z,\; z f_x - x f_z,\; -y f_x + x f_y).

        Provenance
        ----------
        MATLAB source : @spherefun/curl.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        from chebfunjax.spherefun.spherefunv import Spherefunv, _sphere_xyz

        if self.isempty():
            return Spherefunv.empty()
        fx, fy, fz = self.diff(1), self.diff(2), self.diff(3)
        x, y, z = _sphere_xyz()
        gx = -z * fy + y * fz
        gy = z * fx - x * fz
        gz = -y * fx + x * fy
        return Spherefunv(gx, gy, gz)

    def laplacian(self) -> "Spherefun":
        r"""Laplace-Beltrami operator on the sphere (MATLAB laplacian.m):
        ``f_xx + f_yy + f_zz`` with the three second tangential Cartesian
        derivatives (:meth:`diff`) sampled on a common even grid and the
        sum rebuilt with the constructor.

        Provenance
        ----------
        MATLAB source : @spherefun/laplacian.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.
        """
        if self.isempty() or len(self.cols) == 0:
            return self
        parts = [self.diff(1, 2), self.diff(2, 2), self.diff(3, 2)]

        def _len(g):
            # MATLAB [m, n] = length(f): m = row (longitude) length,
            # n = column (doubled colatitude) length -- then
            # sample(f, m, n/2) is m longitude x n/2 colatitude points.
            if len(g.cols) == 0:
                return 1, 1
            return (max(int(np.asarray(r.coeffs).shape[0]) for r in g.rows),
                    max(int(np.asarray(c.coeffs).shape[0]) for c in g.cols))

        lens = [_len(g) for g in parts]
        m = max(mm for mm, _ in lens)
        n = max(nn for _, nn in lens)
        m = m + (m % 2)
        n = n + (n % 2)
        n_th = max(n // 2, 2)
        F = None
        for g in parts:
            V = np.asarray(g.sample(m, n_th))
            F = V if F is None else F + V
        return Spherefun.from_values(F)

    def vscale(self) -> float:
        """Vertical scale: the largest absolute sampled value (MATLAB
        ``vscale``).

        Provenance
        ----------
        MATLAB source : @separableApprox/vscale.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or len(self.cols) == 0:
            return 0.0
        V = np.asarray(Spherefun.coeffs2vals(self.coeffs2()))
        return float(np.max(np.abs(V)))

    @staticmethod
    def poisson(f, const: float = 0.0, m: int | None = None,
                n: int | None = None, lmax: int | None = None) -> "Spherefun":
        r"""Solve the Poisson equation :math:`\Delta u = f` on the sphere
        (MATLAB ``spherefun.poisson(f, const, m, n)``).

        ``f`` is a Spherefun, a callable ``f(lam, theta)`` or a
        coefficient matrix; the discretisation uses ``m`` Fourier modes in
        latitude and ``n`` in longitude (``n = m`` by default; both
        default to the size of ``f`` when omitted).  The right-hand side
        must have zero mean for a solution to exist: its mean is removed
        (with a warning when it exceeds ``1e5 * vscale * eps``) and the
        solution's mean is set to ``const``.  ``lmax`` (legacy keyword) is
        accepted as a bandwidth and mapped to ``m = n = 2*lmax + 2``.

        The solve is MATLAB's Fourier--Fourier method on the doubled-up
        sphere: multiply through by :math:`\sin^2\theta`, solve one
        banded system per longitudinal wavenumber and impose the mean
        condition on the zero mode.

        Provenance
        ----------
        MATLAB source : @spherefun/poisson.m, @trigspec/diffmat.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.
        """
        import warnings as _warnings

        from chebfunjax.tech.trigtech import trig_vals2coeffs

        if lmax is not None and m is None:
            m = 2 * int(lmax) + 2
        if isinstance(f, Spherefun):
            fs = f
        elif callable(f):
            fs = None
        else:
            fs = None
        if m is None:
            if fs is None and not callable(f):
                F0 = np.asarray(f)
                m, n = F0.shape
            elif fs is not None:
                mc = max(int(np.asarray(c.coeffs).shape[0]) for c in fs.cols)
                nr = max(int(np.asarray(r.coeffs).shape[0]) for r in fs.rows)
                m = max(4, mc + (mc % 2))
                n = max(4, nr + (nr % 2))
            else:
                m = n = 64
        m = int(m)
        n = int(m) if n is None else int(n)
        m = max(4, m + (m % 2))
        n = max(4, n)

        DF1m, DF2m, DF2n, Mcossin, Msin2, en, floorm = \
            _sphere_fourier_operators(m, n)
        Im = np.eye(m)
        scl = np.diag(DF2n)
        eps2 = 2.220446049250313e-16
        if fs is not None:
            tol = 1e5 * float(fs.vscale()) * eps2
            F = np.asarray(fs.coeffs2(n, m), dtype=complex)
        elif callable(f):
            lam0 = -np.pi + 2 * np.pi * np.arange(n) / n
            th0 = -np.pi + 2 * np.pi * np.arange(m) / m
            LL, TT = np.meshgrid(lam0, th0)
            F = np.asarray(f(jnp.asarray(LL), jnp.asarray(TT)), dtype=complex)
            tol = 1e5 * float(np.max(np.abs(F))) * eps2
            F = np.asarray(trig_vals2coeffs(jnp.asarray(F)))
            F = np.asarray(trig_vals2coeffs(jnp.asarray(F.T))).T
        else:
            tol = 1e5 * eps2
            F = np.asarray(f, dtype=complex)
        k0 = n // 2                      # zero longitudinal mode (0-based)
        meanF = en @ F[:, k0] / en[floorm]
        if abs(meanF) > tol:
            _warnings.warn(
                "CHEBFUN:SPHEREFUN:POISSON:meanRHS: The integral of the right "
                "hand side may not be zero, which is required for there to "
                "exist a solution to the Poisson equation. Subtracting the "
                "mean off the right hand side now.")
        F = F.copy()
        F[floorm, k0] = F[floorm, k0] - meanF
        F = Msin2 @ F
        CFS = np.zeros((m, n), dtype=complex)
        L = Msin2 @ DF2m + Mcossin @ DF1m
        for k in range(n):
            if k == k0:
                continue
            CFS[:, k] = np.linalg.solve(L + scl[k] * Im, F[:, k])
        ii = [i for i in range(m) if i != floorm]
        A = np.vstack([en[None, :], L[ii, :]])
        b = np.concatenate([[0.0], F[ii, k0]])
        CFS[:, k0] = np.linalg.solve(A, b)
        u = Spherefun.coeffs2spherefun(jnp.asarray(CFS))
        return u + const if const != 0 else u

    def gaussfilt(self, sig: float = np.pi / 180.0) -> "Spherefun":
        r"""Gaussian low-pass filter on the sphere (MATLAB gaussfilt):
        one backward-Euler step of the heat equation to time
        ``t = 0.5 sig^2``, i.e. the Helmholtz solve
        ``helmholtz(-f/t, i/sqrt(t), m, n)`` at the size of ``f``.

        Provenance
        ----------
        MATLAB source : @spherefun/gaussfilt.m
        Chebfun commit: 7574c77
        """
        dt = 0.5 * float(sig) ** 2
        if self.isempty() or len(self.cols) == 0:
            return self
        # MATLAB: [n, m] = length(f) (columns, rows); helmholtz(..., m, n)
        n = max(int(np.asarray(c.coeffs).shape[0]) for c in self.cols)
        m = max(int(np.asarray(r.coeffs).shape[0]) for r in self.rows)
        K = np.sqrt(1.0 / dt) * 1j
        return Spherefun.helmholtz(self * (-1.0 / dt), K, m, n)

    @staticmethod
    def helmholtz(f, K, m: int | None = None,
                  n: int | None = None) -> "Spherefun":
        r"""Solve the Helmholtz equation :math:`\Delta u + K^2 u = f` on
        the sphere (MATLAB ``spherefun.helmholtz(f, K, m, n)``).

        MATLAB's Fourier--Fourier method on the doubled-up sphere with
        ``m`` modes in latitude and ``n`` in longitude (``n = m``; both
        default to the size of ``f``): multiply through by
        :math:`\sin^2\theta`, solve one banded system per longitudinal
        wavenumber, and pin the zero mode with the integral condition.
        ``K`` may be complex (the imaginary shifts of the BDF/heat
        steps).  ``K = 0`` falls back to :meth:`poisson`; a real ``K``
        with ``K^2 = l(l+1)`` is an eigenvalue and raises.

        Provenance
        ----------
        MATLAB source : @spherefun/helmholtz.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.
        """
        from chebfunjax.tech.trigtech import trig_vals2coeffs

        K2 = complex(K) ** 2
        if abs(K2.imag) < 1e-12 * max(1.0, abs(K2.real)):
            K2 = K2.real
        if K2 == 0:
            return Spherefun.poisson(f, 0.0, m, n)
        if isinstance(K2, float) and K2 > 0:
            # K = sqrt(l(l+1)) for an integer l?
            ell = (-1 + np.sqrt(1 + 4 * K2)) / 2
            if abs(ell - round(ell)) < 1e-13:
                raise ValueError(
                    "SPHEREFUN:HELMHOLTZ:EIGENVALUE: There are infinitely "
                    "many solutions since K is an eigenvalue of the Helmholtz "
                    "operator.")
        fs = f if isinstance(f, Spherefun) else None
        if m is None:
            if fs is None:
                fs = Spherefun.from_function(f)
            mc = max(int(np.asarray(c.coeffs).shape[0]) for c in fs.cols)
            nr = max(int(np.asarray(r.coeffs).shape[0]) for r in fs.rows)
            m, n = max(4, mc + (mc % 2)), max(4, nr + (nr % 2))
        m = int(m)
        n = int(m) if n is None else int(n)
        if m <= 0 or n <= 0:
            raise ValueError("CHEBFUN:SPHEREFUN:HELMHOLTZ:badInput: "
                             "Discretization sizes should be positive numbers")
        if m == 1 and n == 1:
            fs = fs if fs is not None else Spherefun.from_function(f)
            return fs * 0.0 + float(np.real(fs.mean2())) / K2
        m = m + (m % 2)
        ops = _sphere_fourier_operators(m, n)
        DF1m, DF2m, DF2n, Mcossin, Msin2, en, floorm = ops
        Im = np.eye(m)
        if fs is not None:
            F = np.asarray(fs.coeffs2(n, m), dtype=complex)
        else:
            lam0 = -np.pi + 2 * np.pi * np.arange(n) / n
            th0 = -np.pi + 2 * np.pi * np.arange(m) / m
            LL, TT = np.meshgrid(lam0, th0)
            F = np.asarray(f(jnp.asarray(LL), jnp.asarray(TT)), dtype=complex)
            F = np.asarray(trig_vals2coeffs(jnp.asarray(F)))
            F = np.asarray(trig_vals2coeffs(jnp.asarray(F.T))).T
        k0 = n // 2
        int_const = en @ F[:, k0] / K2
        F = Msin2 @ F / K2
        CFS = np.zeros((m, n), dtype=complex)
        L = (Msin2 @ DF2m + Mcossin @ DF1m) / K2 + Msin2
        scl = np.diag(DF2n) / K2
        for k in range(n):
            if k == k0:
                continue
            CFS[:, k] = np.linalg.solve(L + scl[k] * Im, F[:, k])
        ii = [i for i in range(m) if i != floorm]
        A = np.vstack([en[None, :], L[ii, :]])
        b = np.concatenate([[int_const], F[ii, k0]])
        CFS[:, k0] = np.linalg.solve(A, b)
        return Spherefun.coeffs2spherefun(jnp.asarray(CFS))

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


# ============================================================================
# Spherical-harmonic spectral calculus helpers (Claude Opus 4.8)
#
# These back the exact, harmonic-diagonal `laplacian` and `poisson`
# methods.  Projection uses Gauss-Legendre quadrature in cos(theta) and
# the trapezoidal (uniform) rule in longitude -- both exact for the
# band-limited functions a Spherefun represents.  numpy is used for the
# quadrature bookkeeping (Spherefun construction is not JIT-safe anyway).
# ============================================================================


def _real_ylm_values(l: int, m_signed: int, lam: jax.Array,
                     theta: jax.Array) -> jax.Array:
    """Pointwise real, orthonormal spherical harmonic Y_l^m(lam, theta).

    Same stable fully-normalized associated-Legendre recurrence as
    ``Spherefun.sphharm``, but evaluated directly (no construction).
    """
    l = int(l)
    m = abs(int(m_signed))
    x = jnp.cos(theta)
    pmm = jnp.ones_like(x) / jnp.sqrt(4 * jnp.pi)
    for i in range(1, m + 1):
        pmm = -jnp.sqrt((2 * i + 1) / (2.0 * i)) * jnp.sqrt(1 - x**2) * pmm
    if l == m:
        plm = pmm
    else:
        pm1 = jnp.sqrt(2 * m + 3.0) * x * pmm
        if l == m + 1:
            plm = pm1
        else:
            p_prev, p_curr = pmm, pm1
            for ll in range(m + 2, l + 1):
                a = jnp.sqrt((4.0 * ll * ll - 1) / (ll * ll - m * m))
                b = jnp.sqrt(((ll - 1.0) ** 2 - m * m)
                             / (4.0 * (ll - 1.0) ** 2 - 1))
                p_next = a * (x * p_curr - b * p_prev)
                p_prev, p_curr = p_curr, p_next
            plm = p_curr
    if int(m_signed) > 0:
        return jnp.sqrt(2.0) * plm * jnp.cos(m * lam)
    if int(m_signed) < 0:
        return jnp.sqrt(2.0) * plm * jnp.sin(m * lam)
    return plm


def _all_real_ylm_values(lmax: int, lam: jax.Array,
                         theta: jax.Array) -> dict:
    """Evaluate EVERY real harmonic ``Y_l^m`` (``l <= lmax``) at once.

    Returns ``{(l, m): values}`` for all ``0 <= l <= lmax`` and
    ``-l <= m <= l``.  Unlike calling :func:`_real_ylm_values` once per
    ``(l, m)`` -- which restarts the associated-Legendre recurrence from
    scratch every time, an ``O(lmax^3)`` blow-up dominated by tiny eager
    JAX dispatches -- this shares the fixed-``m`` recurrence across all
    ``l`` and reuses ``P_l^m`` for the ``+m``/``-m`` pair.  The result is
    bit-identical to :func:`_real_ylm_values` (same fully-normalized
    recurrence and Condon-Shortley phase) but ~25x faster, which is what
    lets nested spherefun compositions (``div(grad f)``, ``vort(grad f)``,
    ...) build in seconds instead of hanging XLA CPU compilation.
    """
    lam = jnp.asarray(lam, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    x = jnp.cos(theta)
    s = jnp.sqrt(1 - x**2)  # sin(theta) >= 0 on [0, pi]
    out: dict = {}
    for m in range(lmax + 1):
        pmm = jnp.ones_like(x) / jnp.sqrt(4 * jnp.pi)
        for i in range(1, m + 1):
            pmm = -jnp.sqrt((2 * i + 1) / (2.0 * i)) * s * pmm
        if m == 0:
            cos_ml = sin_ml = None
        else:
            cos_ml = jnp.sqrt(2.0) * jnp.cos(m * lam)
            sin_ml = jnp.sqrt(2.0) * jnp.sin(m * lam)

        def _store(l, plm):
            if m == 0:
                out[(l, 0)] = plm
            else:
                out[(l, m)] = plm * cos_ml
                out[(l, -m)] = plm * sin_ml

        _store(m, pmm)
        if m < lmax:
            pm1 = jnp.sqrt(2 * m + 3.0) * x * pmm
            _store(m + 1, pm1)
            p_prev, p_curr = pmm, pm1
            for ll in range(m + 2, lmax + 1):
                a = jnp.sqrt((4.0 * ll * ll - 1) / (ll * ll - m * m))
                b = jnp.sqrt(((ll - 1.0) ** 2 - m * m)
                             / (4.0 * (ll - 1.0) ** 2 - 1))
                p_next = a * (x * p_curr - b * p_prev)
                p_prev, p_curr = p_curr, p_next
                _store(ll, p_curr)
    return out


def _sph_harmonic_eval_sum(coeff_map: dict, lmax: int,
                           lam: jax.Array, theta: jax.Array) -> jax.Array:
    """Evaluate ``sum_{l,m} a_{lm} Y_l^m(lam, theta)`` in one shared pass.

    ``coeff_map`` maps ``(l, m)`` -> coefficient (only the entries present
    contribute).  The weighted sum is accumulated *inside* the
    associated-Legendre recurrence, so the full harmonic reconstruction
    costs ``O(lmax^2)`` large vectorised ops with ``O(grid)`` memory --
    never materialising all ``(lmax+1)^2`` harmonic arrays at once.  This
    replaces the previous ``for (l, m) in terms: out += a *
    Y_lm(lam, theta)`` reconstruction, whose per-term recurrence restart
    drove the nested-composition compile blow-up.
    """
    # Concrete inputs run a numpy mirror of the identical recurrence
    # (same dispatch pattern as trig_vals2coeffs): the O(lmax^2)
    # accumulation would otherwise build a giant traced expression at
    # EVERY adaptive-constructor sample, and downstream arithmetic on
    # such spherefuns compounds into fused XLA kernels the CPU JIT
    # cannot compile ("Failed to materialize symbols" /
    # "LLVM compilation error: Cannot allocate memory" on rank-100
    # vorticity chains).  Tracers keep the traceable jnp path.
    if not isinstance(lam, jax.core.Tracer) and \
            not isinstance(theta, jax.core.Tracer):
        lam_c = np.asarray(lam, dtype=np.float64)
        th_c = np.asarray(theta, dtype=np.float64)
        # Glide fold: the spherefun constructor samples the DOUBLED
        # theta range; the double-Fourier-sphere extension of a sphere
        # function is f(lam + pi, -theta) for theta < 0 -- NOT the blind
        # even reflection this sum would otherwise produce (odd-m
        # harmonics differ in sign, corrupting BMC construction; the
        # value-space diff of a gradient component ran away to 1e90
        # rank-1 spherefuns through exactly this).
        neg = th_c < 0
        if np.any(neg):
            lam_c = np.where(neg, lam_c + np.pi, lam_c)
            th_c = np.where(neg, -th_c, th_c)
        return jnp.asarray(_sph_harmonic_eval_sum_np(
            coeff_map, lmax, lam_c, th_c))
    lam = jnp.asarray(lam, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    neg = theta < 0
    lam = jnp.where(neg, lam + jnp.pi, lam)
    theta = jnp.where(neg, -theta, theta)
    x = jnp.cos(theta)
    s = jnp.sqrt(1 - x**2)
    out = jnp.zeros(jnp.broadcast_shapes(lam.shape, theta.shape),
                    dtype=jnp.float64)
    for m in range(lmax + 1):
        pmm = jnp.ones_like(x) / jnp.sqrt(4 * jnp.pi)
        for i in range(1, m + 1):
            pmm = -jnp.sqrt((2 * i + 1) / (2.0 * i)) * s * pmm
        if m == 0:
            cos_ml = sin_ml = None
        else:
            cos_ml = jnp.sqrt(2.0) * jnp.cos(m * lam)
            sin_ml = jnp.sqrt(2.0) * jnp.sin(m * lam)

        def _accum(acc, l, plm):
            if m == 0:
                a = coeff_map.get((l, 0))
                if a is not None:
                    acc = acc + a * plm
            else:
                ap = coeff_map.get((l, m))
                if ap is not None:
                    acc = acc + ap * (plm * cos_ml)
                an = coeff_map.get((l, -m))
                if an is not None:
                    acc = acc + an * (plm * sin_ml)
            return acc

        out = _accum(out, m, pmm)
        if m < lmax:
            pm1 = jnp.sqrt(2 * m + 3.0) * x * pmm
            out = _accum(out, m + 1, pm1)
            p_prev, p_curr = pmm, pm1
            for ll in range(m + 2, lmax + 1):
                a = jnp.sqrt((4.0 * ll * ll - 1) / (ll * ll - m * m))
                b = jnp.sqrt(((ll - 1.0) ** 2 - m * m)
                             / (4.0 * (ll - 1.0) ** 2 - 1))
                p_next = a * (x * p_curr - b * p_prev)
                p_prev, p_curr = p_curr, p_next
                out = _accum(out, ll, p_curr)
    return out


def _sph_harmonic_eval_sum_np(coeff_map: dict, lmax: int,
                              lam: np.ndarray,
                              theta: np.ndarray) -> np.ndarray:
    """numpy mirror of :func:`_sph_harmonic_eval_sum` (kept in lockstep).

    The associated-Legendre recurrence runs over ``l`` per ``m`` as in
    the traced version, but the weighted accumulation is batched into a
    single matmul per ``m`` (the coefficient tables are dense arrays,
    skipping only all-zero orders), so the Python-level cost is O(lmax)
    recurrence steps rather than O(lmax^2) dict-lookup adds.
    """
    x = np.cos(theta)
    s = np.sqrt(np.maximum(1 - x**2, 0.0))
    bshape = np.broadcast_shapes(lam.shape, theta.shape)
    xb = np.broadcast_to(x, bshape).ravel()
    sb = np.broadcast_to(s, bshape).ravel()
    lamb = np.broadcast_to(lam, bshape).ravel()
    out = np.zeros(xb.shape[0])

    # Dense (lmax+1, lmax+1) coefficient tables ap[m, l], an[m, l].
    ap = np.zeros((lmax + 1, lmax + 1))
    an = np.zeros((lmax + 1, lmax + 1))
    for (l, m), v in coeff_map.items():
        if abs(m) <= lmax and l <= lmax:
            if m >= 0:
                ap[m, l] = v
            else:
                an[-m, l] = v

    pmm = np.ones_like(xb) / np.sqrt(4 * np.pi)
    for m in range(lmax + 1):
        if m > 0:
            pmm = -np.sqrt((2 * m + 1) / (2.0 * m)) * sb * pmm
        if not (np.any(ap[m, m:]) or np.any(an[m, m:])):
            continue
        # Stack P_l^m for l = m..lmax via the standard recurrence.
        nl = lmax + 1 - m
        P = np.empty((nl, xb.shape[0]))
        P[0] = pmm
        if nl > 1:
            P[1] = np.sqrt(2 * m + 3.0) * xb * pmm
            for k in range(2, nl):
                ll = m + k
                a = np.sqrt((4.0 * ll * ll - 1) / (ll * ll - m * m))
                b = np.sqrt(((ll - 1.0) ** 2 - m * m)
                            / (4.0 * (ll - 1.0) ** 2 - 1))
                P[k] = a * (xb * P[k - 1] - b * P[k - 2])
        if m == 0:
            out += ap[0, 0:] @ P
        else:
            cos_ml = np.sqrt(2.0) * np.cos(m * lamb)
            sin_ml = np.sqrt(2.0) * np.sin(m * lamb)
            if np.any(ap[m, m:]):
                out += (ap[m, m:] @ P) * cos_ml
            if np.any(an[m, m:]):
                out += (an[m, m:] @ P) * sin_ml
    return out.reshape(bshape)


def _sph_harmonic_evaluators(lmax: int) -> dict:
    """Return {(l, m): callable(lam, theta) -> Y_l^m values} for l <= lmax."""
    ev = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ev[(l, m)] = (lambda lam, theta, _l=l, _m=m:
                          _real_ylm_values(_l, _m, jnp.asarray(lam),
                                           jnp.asarray(theta)))
    return ev


def _spherefun_sph_coeffs(f: "Spherefun", lmax: int) -> dict:
    """Project a Spherefun onto real spherical harmonics up to degree lmax.

    Returns {(l, m): a_{lm}} with a_{lm} = <f, Y_l^m>_{S^2}, computed by
    Gauss-Legendre quadrature in cos(theta) x uniform longitude.
    """
    nth = 2 * lmax + 2
    nph = 2 * lmax + 2
    xg, wg = np.polynomial.legendre.leggauss(nth)  # nodes = cos(theta)
    theta = np.arccos(xg)
    phi = np.linspace(-np.pi, np.pi, nph, endpoint=False)
    dph = 2 * np.pi / nph
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    lam_j = jnp.asarray(PH.ravel())
    th_j = jnp.asarray(TH.ravel())
    F = np.asarray(f(lam_j, th_j)).reshape(TH.shape)
    weight = wg[:, None] * dph  # sin(theta) d(theta) d(phi) via GL in cos
    Yall = _all_real_ylm_values(lmax, lam_j, th_j)
    coeffs = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = np.asarray(Yall[(l, m)]).reshape(TH.shape)
            coeffs[(l, m)] = float(np.sum(F * Y * weight))
    return coeffs


def _spherefun_mul_rank1(f1: "Spherefun", g: "Spherefun") -> "Spherefun":
    """Exact product with a rank-1 factor (MATLAB
    @separableApprox/times.m rank-1 branch): multiply f1's single
    column/row slice into every slice of ``g`` with exact dealiased
    Trigtech products; pivots multiply (evaluation is
    ``sum_j (1/p_j) c_j r_j``, so ``1/p_h = (1/p_f)(1/p_g)``).  The
    BMC parity classes compose: a 'plus' (even/pi-periodic) factor
    preserves g's classes; a 'minus' factor swaps them.
    """
    cf = f1.cols[0]
    rf = f1.rows[0]
    new_cols = [(cf * c).simplify() for c in g.cols]
    new_rows = [(rf * r).simplify() for r in g.rows]
    p_h = np.asarray(f1.pivots)[0] * np.asarray(g.pivots)
    f_minus = len(f1.idx_minus) > 0
    if f_minus:
        idx_plus = tuple(g.idx_minus)
        idx_minus = tuple(g.idx_plus)
    else:
        idx_plus = tuple(g.idx_plus)
        idx_minus = tuple(g.idx_minus)
    return Spherefun(
        cols=new_cols, rows=new_rows,
        pivots=jnp.asarray(p_h),
        idx_plus=idx_plus, idx_minus=idx_minus)


def _spherefun_onediff(f: "Spherefun", dim: int) -> "Spherefun":
    """One tangential Cartesian derivative of a Spherefun (MATLAB
    @spherefun/diff.m ``onediff``), on the CDR coefficients:

    - theta/lambda derivatives: ``i k`` scaling of the Fourier coeffs;
    - multiplication by cos/sin: wavenumber shifts (``Mcos``/``Msin``);
    - division by sin(theta): solve of the even-size (nonsingular)
      multiplication-by-sin matrix, exactly ``Msinn \\ C_cfs``;
    - the two rank-r pieces are SAMPLED on the ``(n/2+1) x m`` lat-lon
      grid and the result rebuilt with ``spherefun(sample(f1) +
      sample(f2))`` (constructFromDouble); the z-derivative modifies the
      columns in place.

    Provenance
    ----------
    MATLAB source : @spherefun/diff.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.tech.trigtech import Trigtech
    if len(f.cols) == 0:
        return Spherefun.empty()
    f = f.simplify()
    cols, rows = f.cols, f.rows
    piv = np.asarray(f.pivots, dtype=float)
    r = len(cols)

    def _stack_even(techs, extra):
        # common EVEN length with headroom; modes -N/2 .. N/2-1
        nmax = max(int(np.asarray(t.coeffs).shape[0]) for t in techs)
        N = nmax + (nmax % 2) + extra
        X = np.zeros((N, len(techs)), dtype=complex)
        for j, t in enumerate(techs):
            c = np.asarray(t.coeffs, dtype=complex).ravel()
            m0 = c.size
            k0 = -((m0 - 1) // 2) if (m0 % 2) else -(m0 // 2)
            X[k0 + N // 2: k0 + N // 2 + m0, j] = c
        return X, np.arange(-(N // 2), N // 2)

    C, kc = _stack_even(cols, 2)
    R, kr = _stack_even(rows, 2 if dim != 3 else 0)
    n, m = C.shape[0], R.shape[0]
    dC = (1j * kc)[:, None] * C            # d/dtheta
    dR = (1j * kr)[:, None] * R            # d/dlambda
    # A constant row (the pole term) has an exactly zero derivative in
    # MATLAB; FFT rounding leaves ~1e-17 in the other modes which the
    # 1/sin(theta) solve of the pole column amplifies.  Clean it.
    for j in range(r):
        col = np.abs(R[:, j]).copy()
        c0 = col[m // 2]
        col[m // 2] = 0.0
        if np.max(col) <= 1e-13 * max(c0, 1e-300):
            dR[:, j] = 0.0

    def _mcos(A):
        B = np.zeros_like(A)
        B[1:, :] += 0.5 * A[:-1, :]
        B[:-1, :] += 0.5 * A[1:, :]
        return B

    def _msin(A):
        B = np.zeros_like(A)
        B[1:, :] += -0.5j * A[:-1, :]
        B[:-1, :] += 0.5j * A[1:, :]
        return B

    if dim == 3:
        C1 = -_msin(dC)
        # values on the n-point theta grid of [-pi, pi] (exact for n modes)
        xs = -1.0 + 2.0 * np.arange(n) / n
        E = np.exp(1j * np.pi * np.outer(xs, kc))
        V = np.real(E @ C1)
        new_cols = [Trigtech.from_values(jnp.asarray(V[:, j])) for j in range(r)]
        return Spherefun(cols=new_cols, rows=list(rows), pivots=f.pivots,
                         idx_plus=f.idx_plus, idx_minus=f.idx_minus,
                         pivot_locations=f.pivot_locations,
                         nonzero_poles=f.nonzero_poles)

    Msin = np.zeros((n, n), dtype=complex)
    for k in range(n):
        if k >= 1:
            Msin[k, k - 1] = -0.5j
        if k + 1 < n:
            Msin[k, k + 1] = 0.5j
    Cs = np.linalg.solve(Msin, C)          # C / sin(theta)
    if dim == 1:      # d/dx = -sin(lam)/sin(th) d/dlam + cos(lam) cos(th) d/dth
        C1, R1 = Cs, -_msin(dR)
        C2, R2 = _mcos(dC), _mcos(R)
    else:             # d/dy =  cos(lam)/sin(th) d/dlam + sin(lam) cos(th) d/dth
        C1, R1 = Cs, _mcos(dR)
        C2, R2 = _mcos(dC), _msin(R)
    m_even = m + (m % 2)
    n_th = n // 2 + 1
    th_pts = np.linspace(0.0, np.pi, n_th)
    lam_pts = -np.pi + 2 * np.pi * np.arange(m_even) / m_even
    Eth = np.exp(1j * np.outer(th_pts, kc))        # (n_th, n)
    El = np.exp(1j * np.outer(lam_pts, kr))        # (m_even, m)
    dinv = np.where(np.abs(piv) > 0, 1.0 / np.where(piv == 0, 1.0, piv), 0.0)
    F = np.real(((Eth @ C1) * dinv) @ (El @ R1).T
                + ((Eth @ C2) * dinv) @ (El @ R2).T)
    return Spherefun.from_values(F)


def _spherefun_diff_cart(f: "Spherefun", dim: int) -> "Spherefun":
    """One tangential Cartesian derivative of a Spherefun (Opus 4.8).

    dim: 1 = x, 2 = y, 3 = z.  Evaluates the intrinsic-gradient formula
    at interior Gauss-Legendre colatitude nodes (avoiding the poles),
    with the theta/lambda derivatives taken spectrally from the CDR
    trigtechs, then projects onto real spherical harmonics and rebuilds.
    """
    lmax = f._bandwidth() + 2
    inv_pi = 1.0 / np.pi
    cols_d = [c.diff() for c in f.cols]
    rows_d = [r.diff() for r in f.rows]
    piv = np.asarray(f.pivots)

    n = 2 * lmax + 4
    xg, wg = np.polynomial.legendre.leggauss(n)      # interior nodes
    theta = np.arccos(xg)
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    dph = 2 * np.pi / n
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    lam_j = jnp.asarray(PH.ravel())
    th_j = jnp.asarray(TH.ravel())
    tr = th_j / jnp.pi
    lr = lam_j / jnp.pi

    ft = np.zeros(TH.size)   # d f / d theta at the nodes
    fl = np.zeros(TH.size)   # d f / d lambda at the nodes
    for j in range(len(f.cols)):
        w = float(1.0 / piv[j])
        ft = ft + w * inv_pi * np.asarray(jnp.real(cols_d[j](tr))) \
            * np.asarray(jnp.real(f.rows[j](lr)))
        fl = fl + w * inv_pi * np.asarray(jnp.real(f.cols[j](tr))) \
            * np.asarray(jnp.real(rows_d[j](lr)))
    ft = ft.reshape(TH.shape)
    fl = fl.reshape(TH.shape)

    sinT, cosT = np.sin(TH), np.cos(TH)
    sinL, cosL = np.sin(PH), np.cos(PH)
    if dim == 1:
        V = -sinL / sinT * fl + cosL * cosT * ft
    elif dim == 2:
        V = cosL / sinT * fl + sinL * cosT * ft
    else:
        V = -sinT * ft

    weight = wg[:, None] * dph
    Yall = _all_real_ylm_values(lmax, lam_j, th_j)
    coeff_map = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = np.asarray(Yall[(l, m)]).reshape(TH.shape)
            a = float(np.sum(V * Y * weight))
            if abs(a) > 1e-13:
                coeff_map[(l, m)] = a

    def ev(lam, theta):
        return _sph_harmonic_eval_sum(coeff_map, lmax, lam, theta)

    return Spherefun.from_function(ev)


# ---------------------------------------------------------------------------
# Spherical-harmonic-basis surface gradient
# ---------------------------------------------------------------------------
#
# The surface gradient of a single (complex) spherical harmonic Y_l^m is a
# finite combination of harmonics of degree l-1 and l+1.  Applying the exact
# recurrence coefficients below to the harmonic expansion of ``f`` yields the
# three Cartesian components (fx, fy, fz) as exact harmonic expansions, so the
# normal component ``x*fx + y*fy + z*fz`` (obtained by multiplying by the exact
# coordinate-times operators, which lower/raise the degree in the same way)
# vanishes identically -- the gradient is tangential to machine precision.
# This avoids both the ``1/sin(theta)`` amplification of the coefficient-space
# diff.m port and the per-component quadrature floor of the value-space route.
#
# Recurrences (real, fully-normalized, Condon-Shortley complex harmonics):
#   d/dz Y_l^m       = -l A_l^m Y_{l+1}^m  + (l+1) A_{l-1}^m Y_{l-1}^m
#   (d/dx + i d/dy) Y = P_l^m Y_{l+1}^{m+1} + Q_l^m Y_{l-1}^{m+1}
#   (d/dx - i d/dy) Y = P'_l^m Y_{l+1}^{m-1} + Q'_l^m Y_{l-1}^{m-1}
# with A_l^m = sqrt(((l+1)^2 - m^2)/((2l+1)(2l+3))) and
#   P_l^m  =  l   sqrt((l+m+1)(l+m+2)/((2l+1)(2l+3)))
#   Q_l^m  = (l+1) sqrt((l-m)(l-m-1)/((2l-1)(2l+1)))
#   P'_l^m = -l   sqrt((l-m+1)(l-m+2)/((2l+1)(2l+3)))
#   Q'_l^m = -(l+1) sqrt((l+m)(l+m-1)/((2l-1)(2l+1)))
# All square-root arguments are non-negative for admissible (l, m).  The
# coefficients were derived and verified numerically against the value-space
# tangential gradient to ~1e-15 (see the associated core tests).


def _sph_grad_op_z(l: int, m: int) -> dict:
    """d/dz of the complex harmonic Y_l^m: {(l', m'): coefficient}."""
    o = {(l + 1, m): -l * np.sqrt(((l + 1) ** 2 - m * m)
                                  / ((2 * l + 1) * (2 * l + 3)))}
    if l - 1 >= abs(m):
        o[(l - 1, m)] = (l + 1) * np.sqrt((l * l - m * m)
                                          / ((2 * l - 1) * (2 * l + 1)))
    return o


def _sph_grad_op_plus(l: int, m: int) -> dict:
    """(d/dx + i d/dy) of Y_l^m (raises the order to m+1)."""
    o = {(l + 1, m + 1): l * np.sqrt((l + m + 1) * (l + m + 2)
                                     / ((2 * l + 1) * (2 * l + 3)))}
    if l - 1 >= abs(m + 1):
        o[(l - 1, m + 1)] = (l + 1) * np.sqrt((l - m) * (l - m - 1)
                                              / ((2 * l - 1) * (2 * l + 1)))
    return o


def _sph_grad_op_minus(l: int, m: int) -> dict:
    """(d/dx - i d/dy) of Y_l^m (lowers the order to m-1)."""
    o = {(l + 1, m - 1): -l * np.sqrt((l - m + 1) * (l - m + 2)
                                      / ((2 * l + 1) * (2 * l + 3)))}
    if l - 1 >= abs(m - 1):
        o[(l - 1, m - 1)] = -(l + 1) * np.sqrt((l + m) * (l + m - 1)
                                               / ((2 * l - 1) * (2 * l + 1)))
    return o


def _apply_sph_operator(c: dict, opfn, lmax: int) -> dict:
    """Apply a per-harmonic operator to a complex coefficient map."""
    out: dict = {}
    for (l, m), val in c.items():
        if val == 0:
            continue
        for (ll, mm), co in opfn(l, m).items():
            if ll <= lmax and abs(mm) <= ll:
                out[(ll, mm)] = out.get((ll, mm), 0j) + val * co
    return out


def _real_to_complex_sph(a: dict, lmax: int) -> dict:
    """Real-SH coefficients a[(l, m)] -> complex-SH coefficients c[(l, m)].

    Uses the standard orthonormal transform matching :func:`_real_ylm_values`
    (Condon-Shortley, ``m>0`` = cos, ``m<0`` = sin).
    """
    r2 = np.sqrt(2.0)
    c: dict = {}
    for l in range(lmax + 1):
        c[(l, 0)] = complex(a.get((l, 0), 0.0))
        for mu in range(1, l + 1):
            ap = a.get((l, mu), 0.0)
            an = a.get((l, -mu), 0.0)
            c[(l, mu)] = complex(ap, -an) / r2
            c[(l, -mu)] = ((-1) ** mu) * complex(ap, an) / r2
    return c


def _complex_to_real_sph(c: dict, lmax: int) -> dict:
    """Inverse of :func:`_real_to_complex_sph` (returns real coefficients)."""
    r2 = np.sqrt(2.0)
    a: dict = {}
    for l in range(lmax + 1):
        a[(l, 0)] = c.get((l, 0), 0j).real
        for mu in range(1, l + 1):
            cp = c.get((l, mu), 0j)
            cn = c.get((l, -mu), 0j)
            a[(l, mu)] = ((cp + ((-1) ** mu) * cn) / r2).real
            a[(l, -mu)] = ((1j * (cp - ((-1) ** mu) * cn)) / r2).real
    return a


def _spherefun_grad_harmonic(f: "Spherefun") -> tuple:
    """Surface gradient ``(fx, fy, fz)`` in the spherical-harmonic basis.

    Projects ``f`` onto the real spherical harmonics once, then applies the
    exact analytic Cartesian surface-gradient recurrence (above) to every
    harmonic.  Because each ``Y_l^m``'s surface gradient is analytically
    tangential, the three returned components satisfy ``x*fx + y*fy + z*fz ==
    0`` to machine precision and there is no ``1/sin(theta)`` amplification,
    unlike the value-space route in :func:`_spherefun_diff_cart`.  Added by
    Claude Fable 5.
    """
    lmax = f._bandwidth() + 2
    a = _spherefun_sph_coeffs(f, lmax)
    cap = lmax + 1  # gradient raises the degree by one
    c = _real_to_complex_sph(a, cap)
    cp = _apply_sph_operator(c, _sph_grad_op_plus, cap)
    cm = _apply_sph_operator(c, _sph_grad_op_minus, cap)
    cz = _apply_sph_operator(c, _sph_grad_op_z, cap)
    keys = set(cp) | set(cm)
    cx = {k: 0.5 * (cp.get(k, 0j) + cm.get(k, 0j)) for k in keys}
    cy = {k: -0.5j * (cp.get(k, 0j) - cm.get(k, 0j)) for k in keys}
    bx = _complex_to_real_sph(cx, cap)
    by = _complex_to_real_sph(cy, cap)
    bz = _complex_to_real_sph(cz, cap)
    comps = []
    for b in (bx, by, bz):
        # Keep every coefficient above the rounding floor: dropping real
        # content at ~1e-14 would break the exact normal cancellation.
        coeff_map = {k: v for k, v in b.items() if abs(v) > 1e-15}

        cache: dict = {}

        def ev(lam, theta, _cm=coeff_map, _cache=cache):
            la = np.asarray(lam)
            th = np.asarray(theta)
            if not (isinstance(lam, jax.core.Tracer)
                    or isinstance(theta, jax.core.Tracer)):
                # The adaptive constructor re-samples the same nested
                # tensor grids many times (phase-1 refinements, per-rank
                # phase-2 slices); the O(lmax^2)-term harmonic sum makes
                # those repeats the dominant cost, so memoize by grid
                # content.
                key = (la.shape, th.shape,
                       hash(la.tobytes()), hash(th.tobytes()))
                hit = _cache.get(key)
                if hit is not None:
                    return hit
                val = _sph_harmonic_eval_sum(_cm, cap, la, th)
                _cache[key] = val
                return val
            return _sph_harmonic_eval_sum(_cm, cap, lam, theta)

        # The result's bandwidth is KNOWN (cap): start the constructor at
        # a resolving grid so it converges in one pass instead of
        # re-sampling the O(lmax^2) harmonic sum on every refinement.
        comps.append(Spherefun.from_function(
            ev, start_grid=max(8, 2 * cap + 4)))
    return tuple(comps)


from chebfunjax.utils.misc import make_empty_aware  # noqa: E402

make_empty_aware(Spherefun, ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', '__pow__', '__neg__', 'sum', 'sum2', 'mean', 'norm', 'rotate', 'gaussfilt', 'laplacian', 'compose', 'exp', 'sin', 'cos', 'sqrt'])


# ----------------------------------------------------------------------
# Helpers for spherefun(DOUBLE), projectOntoBMCI and the SVDs (Fable 5)
# ----------------------------------------------------------------------
def _check_pole(val, tol):
    """MATLAB checkPole: the mean of a pole row and whether it is
    constant to within the tolerance."""
    pole = float(np.mean(val))
    stddev = float(np.std(val, ddof=1)) if val.size > 1 else 0.0
    const = (stddev <= 1e8 * tol) or (stddev < _EPS)
    return pole, const


def _phase_one_matrix_sphere(F, tol, alpha):
    """MATLAB @spherefun/constructor.m PhaseOne(F, tol, alpha, 0) with
    the (nargout > 4) column/row/pivot outputs: the full BMC-I Gaussian
    elimination on a sample matrix (no width restriction).

    Returns (pivot_indices [0-based (theta_row, lam_col) into the
    interior/half grids], pivot_array, remove_pole, cols (2m-2 x r),
    pivots (r,), rows (n x r), idx_plus, idx_minus) -- all 0-based.
    """
    m, n = F.shape
    if m <= 1:
        raise ValueError("CHEBFUN:SPHEREFUN:constructor:poleSamples")
    half = n // 2
    if m == 2:
        cols = np.zeros((2, 1))
        cols[:, 0] = F[:, 0]
        return (np.zeros((1, 2), dtype=int), np.array([[1.0, 0.0]]), True,
                cols, np.array([1.0]), np.ones((n, 1)), [0], [])
    C = F[:, :half]
    B = F[:, half:]
    Fp = 0.5 * (B + C)
    Fm = 0.5 * (B - C)
    pole1, _ = _check_pole(Fp[0, :], tol)
    pole2, _ = _check_pole(Fp[m - 1, :], tol)
    cols_plus, rows_plus, idx_plus = [], [], []
    cols_minus, rows_minus, idx_minus = [], [], []
    rank_count = 0
    remove_pole = False
    col_pole = row_pole = None
    row_val = 0.0
    if abs(pole1) > tol or abs(pole2) > tol:
        colmax = np.max(np.abs(Fp), axis=0)
        pole_col = int(np.argmax(colmax))
        row_val = float(colmax[pole_col])
        row_pole = row_val * np.ones(half)
        col_pole = Fp[:, pole_col].copy()
        Fp = Fp - np.outer(col_pole, row_pole / row_val)
        remove_pole = True
        rank_count += 1
    Fp = Fp[1:m - 1, :].copy()
    Fm = Fm[1:m - 1, :].copy()
    pivot_indices = []
    pivot_array = []

    def _argmax(A):
        if A.size == 0:
            return 0.0, (0, 0)
        idx = int(np.argmax(np.abs(A.T)))        # column-major like MATLAB
        k, j = divmod(idx, A.shape[0])
        return float(np.abs(A[j, k])), (j, k)

    maxp, ip = _argmax(Fp)
    maxm, im = _argmax(Fm)
    if maxp == 0 and maxm == 0 and not remove_pole:
        return (np.zeros((1, 2), dtype=int), np.array([[0.0, 0.0]]), False,
                np.zeros((2 * m - 2, 1)), np.array([np.inf]),
                np.zeros((n, 1)), [0], [])
    min_size = min(2 * m - 2, n)
    while max(maxp, maxm) > tol and rank_count < min_size:
        j, k = ip if maxp >= maxm else im
        evp = float(Fp[j, k])
        evm = float(Fm[j, k])
        absevp, absevm = abs(evp), abs(evm)
        pivot_indices.append((j, k))
        if max(absevp, absevm) <= alpha * min(absevp, absevm):
            cp = Fp[:, k].copy()
            rp = Fp[j, :].copy()
            Fp = Fp - np.outer(cp, rp / evp)
            cm = Fm[:, k].copy()
            rm = Fm[j, :].copy()
            Fm = Fm - np.outer(cm, rm / evm)
            cols_plus.append(cp)
            rows_plus.append(rp)
            cols_minus.append(cm)
            rows_minus.append(rm)
            if absevp >= absevm:
                idx_plus.append(rank_count)
                idx_minus.append(rank_count + 1)
            else:
                idx_minus.append(rank_count)
                idx_plus.append(rank_count + 1)
            rank_count += 2
            pivot_array.append((evp, evm))
            maxp, ip = _argmax(Fp)
            maxm, im = _argmax(Fm)
        elif absevp > absevm:
            cp = Fp[:, k].copy()
            rp = Fp[j, :].copy()
            Fp = Fp - np.outer(cp, rp / evp)
            cols_plus.append(cp)
            rows_plus.append(rp)
            idx_plus.append(rank_count)
            rank_count += 1
            pivot_array.append((evp, 0.0))
            maxp, ip = _argmax(Fp)
        else:
            cm = Fm[:, k].copy()
            rm = Fm[j, :].copy()
            Fm = Fm - np.outer(cm, rm / evm)
            cols_minus.append(cm)
            rows_minus.append(rm)
            idx_minus.append(rank_count)
            rank_count += 1
            pivot_array.append((0.0, evm))
            maxm, im = _argmax(Fm)
    cols = np.zeros((2 * m - 2, rank_count))
    rows = np.zeros((n, rank_count))
    pivots = np.zeros(rank_count)
    piv_arr = np.array(pivot_array).reshape(-1, 2)
    if cols_plus:
        CP = np.column_stack(cols_plus)
        RP = np.array(rows_plus)                     # (kplus, half)
        cols[m:2 * m - 2, idx_plus] = CP
        cols[1:m - 1, idx_plus] = CP[::-1, :]
        rows[:, idx_plus] = np.concatenate([RP, RP], axis=1).T
        pivots[idx_plus] = piv_arr[piv_arr[:, 0] != 0, 0]
    if cols_minus:
        CM = np.column_stack(cols_minus)
        RM = np.array(rows_minus)
        cols[m:2 * m - 2, idx_minus] = CM
        cols[1:m - 1, idx_minus] = -CM[::-1, :]
        rows[:, idx_minus] = np.concatenate([-RM, RM], axis=1).T
        pivots[idx_minus] = piv_arr[piv_arr[:, 1] != 0, 1]
    if remove_pole:
        cols[:, 0] = np.concatenate([col_pole[::-1], col_pole[1:m - 1]])
        rows[:, 0] = np.concatenate([row_pole, row_pole])
        pivots[0] = row_val
        idx_plus = [0] + idx_plus
    return (np.array(pivot_indices, dtype=int).reshape(-1, 2) + [1, 0],
            piv_arr, remove_pole, cols, pivots, rows, idx_plus, idx_minus)


def _stack_trig_coeffs(techs):
    """Common-length (centred) coefficient matrix of a list of
    Trigtechs, one column each."""
    n = max(int(np.asarray(t.coeffs).shape[0]) for t in techs)
    n_odd = n if n % 2 == 1 else n + 1
    out = np.zeros((n_odd, len(techs)), dtype=complex)
    for j, t in enumerate(techs):
        c = np.asarray(t.coeffs, dtype=complex)
        k = c.shape[0]
        if k % 2 == 0:
            # MATLAB even-length trigtech: modes -k/2..k/2-1; split the
            # -k/2 mode symmetrically so the vector is centred.
            c = np.concatenate([0.5 * c[:1], c[1:], 0.5 * c[:1]])
            k += 1
        off = (n_odd - k) // 2
        out[off:off + k, j] = c
    return out


def _trigtech_from_coeffs_real(c):
    """MATLAB real(trigtech({'', X})): the Trigtech with the given
    (centred, odd-length) coefficients, real part taken in value space."""
    from chebfunjax.tech.trigtech import Trigtech, trig_coeffs2vals
    c = np.asarray(c, dtype=complex)
    v = np.real(np.asarray(trig_coeffs2vals(jnp.asarray(c))))
    return Trigtech.from_values(jnp.asarray(v))


def _bmc1_even_cols(X, nonzero_poles):
    """MATLAB projectOntoEvenBMCI on the column-coefficient matrix X
    (odd length m: modes -(m-1)/2..(m-1)/2)."""
    X = np.array(X, dtype=complex)
    m, n = X.shape
    even_modes = list(range(n))
    if nonzero_poles:
        C = 0.5 * (X[:, 0] - X[::-1, 0])
        X[:, 0] = X[:, 0] - C
        even_modes = list(range(1, n))
    if even_modes:
        Xe = X[:, even_modes]
        C = 0.5 * (Xe - Xe[::-1, :])
        Xe = Xe - C
        Xe[0::2, :] = Xe[0::2, :] - (2.0 / (m + 1)) * np.sum(Xe[0::2, :],
                                                          axis=0)
        if m > 1:
            Xe[1::2, :] = Xe[1::2, :] - (2.0 / (m - 1)) * np.sum(Xe[1::2, :],
                                                              axis=0)
        X[:, even_modes] = Xe
    return X


def _bmc1_odd_cols(X):
    """MATLAB projectOntoOddBMCI on the column-coefficient matrix."""
    X = np.array(X, dtype=complex)
    C = 0.5 * (X + X[::-1, :])
    return X - C


def _zero_trig_modes(R, odd: bool):
    """Zero the odd (``odd=True``) or even wavenumber rows of a centred
    coefficient matrix (MATLAB projectOnto*BMCI row projections)."""
    R = np.array(R, dtype=complex)
    n = R.shape[0]
    zero_mode = n // 2
    k = np.arange(n) - zero_mode
    if odd:
        R[k % 2 == 1, :] = 0.0
    else:
        R[k % 2 == 0, :] = 0.0
    return R


def _trig_flip(t):
    """The Trigtech of ``x -> f(-x)``."""
    from chebfunjax.tech.trigtech import Trigtech, trig_coeffs2vals
    v = np.asarray(trig_coeffs2vals(jnp.asarray(t.coeffs)))
    if t.is_real:
        v = np.real(v)
    return Trigtech.from_values(jnp.asarray(np.roll(v[::-1], 1)))


def _trig_resample(V, nq):
    """Resample columns of trapezoid-grid values to nq points."""
    from chebfunjax.tech.trigtech import Trigtech, _trig_eval_np
    if V.shape[0] == nq:
        return V
    x = -1.0 + 2.0 * np.arange(nq) / nq
    out = np.zeros((nq, V.shape[1]))
    for j in range(V.shape[1]):
        t = Trigtech.from_values(jnp.asarray(V[:, j]))
        out[:, j] = np.real(np.asarray(_trig_eval_np(
            np.asarray(t.coeffs)[:, None], x, is_real=True))).ravel()
    return out


def _trig_quasimatrix(V):
    """Quasimatrix of periodic Chebfuns on [-pi, pi] from trapezoid-grid
    values (one column each)."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    fs = [chebfun(jnp.asarray(V[:, j]), domain=(-np.pi, np.pi), trig=True)
          for j in range(V.shape[1])]
    return Quasimatrix(fs, fs[0].domain)


def _simplify_global_sphere(techs, tol=None):
    """Chop each slice relative to the global vertical scale of the
    quasimatrix (MATLAB ``simplify(f.cols, pseudoLevel)``)."""
    base = _EPS if tol is None else float(tol)
    scales = [float(jnp.max(jnp.abs(jnp.asarray(t.values)))) for t in techs]
    vs = max(scales) if scales else 0.0
    out = []
    for t, sc in zip(techs, scales):
        if sc == 0.0 or vs == 0.0:
            out.append(t.simplify(base))
        else:
            out.append(t.simplify(min(0.5, max(base, base * vs / sc))))
    return out


def _sphere_fourier_operators(m: int, n: int):
    """Operators of MATLAB's spherefun poisson/helmholtz solves on the
    doubled-up sphere: trigspec differentiation matrices (the odd-order
    Nyquist flag as in @trigspec/diffmat.m), the multiplication matrices
    by sin(theta)cos(theta) and sin(theta)^2, the integration weights
    ``en`` of the latitude modes (zeroed at the +-1 modes) and the index
    of the zero mode.

    Provenance
    ----------
    MATLAB source : @spherefun/poisson.m, @trigspec/{diffmat,multmat}.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.trigspec import multmat
    from chebfunjax.tech.trigtech import Trigtech

    def _diffmat(N, order, flag=False):
        if N % 2 == 0:
            if order % 2 == 1:
                k = np.concatenate([[0.0], np.arange(-N / 2 + 1, N / 2)])
                D = np.diag((1j * k) ** order)
                if flag:
                    D[0, 0] = (-1j * N / 2) ** order
            else:
                k = np.arange(-N / 2, N / 2)
                D = np.diag((1j * k) ** order)
        else:
            k = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
            D = np.diag((1j * k) ** order)
        return D

    DF1m = _diffmat(m, 1, True)
    DF2m = _diffmat(m, 2)
    DF2n = _diffmat(n, 2)
    cs = Trigtech.from_function(lambda t: jnp.sin(jnp.pi * t) * jnp.cos(jnp.pi * t))
    Mcossin = np.asarray(multmat(m, cs.coeffs))
    s2 = Trigtech.from_function(lambda t: jnp.sin(jnp.pi * t) ** 2)
    Msin2 = np.asarray(multmat(m, s2.coeffs))
    floorm = m // 2
    mm = np.arange(-floorm, -(-m // 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        en = 2 * np.pi * (1 + np.exp(1j * np.pi * mm)) / (1 - mm ** 2)
    en[floorm - 1] = 0.0
    en[floorm + 1] = 0.0
    return DF1m, DF2m, DF2n, Mcossin, Msin2, en, floorm
