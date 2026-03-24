# uses-numpy: greedy AAA algorithm is not JIT-safe (iterative point selection, SVD)
"""AAA rational approximation.

Translated from MATLAB Chebfun (commit 7574c77): aaa.m.

Original algorithm:
    Y. Nakatsukasa, O. Sete, and L. N. Trefethen,
    "The AAA algorithm for rational approximation",
    SIAM J. Sci. Comp. 40 (2018), A1494–A1522.

Original authors: Copyright 2023 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.

Design notes
------------
- The greedy support-point selection loop is NOT JIT-safe: it uses
  Python-level data-dependent control flow with dynamic array sizes.
- The returned callable ``r(zz)`` IS JIT-safe: it is a thin wrapper
  around ``_reval``, which has static shapes given the support points.
- dtype is always complex128 internally (generalises to complex input) but
  for real inputs the imaginary part will be negligible and the caller
  should cast to float64 if needed.
"""

from __future__ import annotations

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg as spla

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aaa(
    F: jnp.ndarray | Callable,
    Z: jnp.ndarray,
    *,
    tol: float = 1e-13,
    mmax: int = 100,
    cleanup: bool = True,
    cleanup_tol: float | None = None,
) -> tuple[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """AAA rational approximation.

    Computes a near-best rational approximant to ``F`` on the sample set ``Z``
    using the Adaptive Antoulas–Anderson (AAA) algorithm.  The approximant is
    represented in barycentric form:

    .. math::

        r(z) = \\frac{\\sum_j w_j f_j / (z - z_j)}{\\sum_j w_j / (z - z_j)}

    The main loop is a greedy algorithm: at each step it picks the sample
    point with the largest residual, adds it as a new support point, and
    solves a small least-squares problem for the barycentric weights.  The
    loop terminates when the sup-norm error falls below ``tol * ||F||_inf``.

    .. note::
        The construction loop is **not JIT-safe** (greedy point selection).
        The returned callable ``r`` **is JIT-safe** — pass it into ``jax.jit``
        after construction.

    Parameters
    ----------
    F : array_like or callable
        Function values at ``Z``, or a callable to evaluate.  If callable,
        ``F(Z)`` is called once.  Must have the same length as ``Z`` if given
        as an array.
    Z : array_like, shape (M,)
        Sample points (real or complex).
    tol : float, optional
        Relative tolerance for convergence (default 1e-13).
    mmax : int, optional
        Maximum number of support points / barycentric terms (default 100).
        The approximant will have degree at most ``mmax - 1``.
    cleanup : bool, optional
        If ``True`` (default), apply Froissart-doublet removal: poles whose
        residue is negligible relative to nearby sample-set distances are
        removed and the weights are recomputed.
    cleanup_tol : float or None, optional
        Threshold for the cleanup step.  Defaults to ``tol``
        (or ``1e-13`` when ``tol == 0``).

    Returns
    -------
    r : callable
        Rational approximant as a function handle.  ``r(zz)`` evaluates
        the approximant at points ``zz``; it is JIT-safe.
    pol : jnp.ndarray, complex
        Poles of the rational approximant (from generalised eigenvalue problem).
    res : jnp.ndarray, complex
        Residues at the poles (via least-squares, accurate).
    zer : jnp.ndarray, complex
        Zeros of the rational approximant.
    zj : jnp.ndarray, complex
        Support (interpolation) points selected by the greedy loop.
    fj : jnp.ndarray, complex
        Function values at the support points.
    wj : jnp.ndarray, complex
        Barycentric weights.

    Examples
    --------
    Approximate |x| on [-1, 1]:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.aaa import aaa
    >>> Z = jnp.linspace(-1, 1, 1000)
    >>> r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z)
    >>> err = jnp.max(jnp.abs(r(Z) - jnp.abs(Z)))  # should be < 1e-13

    References
    ----------
    .. [1] Y. Nakatsukasa, O. Sete, and L. N. Trefethen,
       "The AAA algorithm for rational approximation",
       SIAM J. Sci. Comp. 40 (2018), A1494–A1522.

    Provenance
    ----------
    MATLAB source : aaa.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford and The
        Chebfun Developers.
    """
    # ---- Input handling ----
    Z = jnp.asarray(Z, dtype=jnp.complex128).ravel()
    M = Z.shape[0]

    if callable(F):
        F_vals = jnp.asarray(F(Z), dtype=jnp.complex128).ravel()
    else:
        F_vals = jnp.asarray(F, dtype=jnp.complex128).ravel()
        if F_vals.shape[0] != M:
            raise ValueError(
                f"F and Z must have the same length, got {F_vals.shape[0]} and {M}."
            )

    # Remove Inf/NaN entries and duplicate Z values
    keep = jnp.isfinite(F_vals)
    F_vals = F_vals[keep]
    Z = Z[keep]
    # Unique Z (stable order)
    Z_np = np.array(Z)
    F_np = np.array(F_vals)
    _, uni = np.unique(Z_np, return_index=True)
    uni = np.sort(uni)  # keep stable order
    Z_np = Z_np[uni]
    F_np = F_np[uni]
    M = len(Z_np)

    if cleanup_tol is None:
        cleanup_tol = tol if tol > 0 else 1e-13

    abstol = tol * np.linalg.norm(F_np, np.inf)

    # ---- AAA greedy iteration ----
    # Working in numpy for the loop (dynamic arrays, data-dependent branching)
    J = list(range(M))          # indices of remaining (non-support) points
    zj = np.zeros(0, dtype=complex)
    fj = np.zeros(0, dtype=complex)
    C = np.zeros((M, 0), dtype=complex)   # Cauchy matrix columns
    A = np.zeros((M, 0), dtype=complex)   # Loewner matrix columns
    R = np.full(M, np.mean(F_np), dtype=complex)  # current rational approx
    errvec = []

    wj = np.array([], dtype=complex)

    for m in range(1, mmax + 1):
        # --- Select next support point: largest |F(J) - R(J)| ---
        J_arr = np.array(J)
        resids = np.abs(F_np[J_arr] - R[J_arr])
        jj = int(np.argmax(resids))  # local index in J
        idx = J_arr[jj]              # global index

        # Update support points and Cauchy / Loewner matrices.
        # The Cauchy column has a pole (inf) at the support point itself;
        # the corresponding row is excluded from the SVD via the J index set.
        # inf / inf = NaN in the R update, which is corrected below.
        zj = np.append(zj, Z_np[idx])
        fj = np.append(fj, F_np[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            new_col = 1.0 / (Z_np - Z_np[idx])          # Cauchy column (inf at idx)
            loewner_col = (F_np - fj[-1]) * new_col      # Loewner column (NaN at idx)
        C = np.column_stack([C, new_col]) if C.shape[1] > 0 else new_col[:, None]
        A = np.column_stack([A, loewner_col]) if A.shape[1] > 0 else loewner_col[:, None]
        J.pop(jj)                                        # remove from free set

        # --- Compute barycentric weights via SVD of Loewner submatrix ---
        J_arr = np.array(J)
        n_free = len(J_arr)

        if n_free >= m:
            # Tall-skinny: reduced SVD of A[J, :]
            A_sub = A[J_arr, :]
            # Column scaling to improve conditioning (Fei Xue)
            col_norms = np.linalg.norm(A_sub, axis=0)
            doscale = False
            # Quick conditioning estimate: compare largest and smallest sing val
            try:
                _, s, V = np.linalg.svd(A_sub, full_matrices=False)
                eps_machine = np.finfo(float).eps
                if s[0] / (s[-1] + 1e-300) > 1.0 / (3.0 * eps_machine):
                    doscale = True
            except np.linalg.LinAlgError:
                doscale = True

            if doscale:
                col_norms_safe = np.where(col_norms > 0, col_norms, 1.0)
                A_scaled = A_sub / col_norms_safe[None, :]
                _, s, V = np.linalg.svd(A_scaled, full_matrices=False)
                idx_min = np.argmin(s)
                # Handle multiple minimum singular values
                tol_sv = s[idx_min] * (1 + 1e-10)
                mm = np.where(s <= tol_sv)[0]
                nm = len(mm)
                wj = V[mm, :].T @ (np.ones(nm) / np.sqrt(nm))
                wj = wj / col_norms_safe  # un-scale
                wj = wj / np.linalg.norm(wj)
            else:
                idx_min = np.argmin(s)
                tol_sv = s[idx_min] * (1 + 1e-10)
                mm = np.where(s <= tol_sv)[0]
                nm = len(mm)
                wj = V[mm, :].T @ (np.ones(nm) / np.sqrt(nm))

        elif n_free >= 1:
            # More columns than rows: compute null space
            A_sub = A[J_arr, :]
            V = spla.null_space(A_sub)
            nm = V.shape[1]
            wj = V @ (np.ones(nm) / np.sqrt(nm))
        else:
            # No free rows (all points are support points)
            wj = np.ones(m, dtype=complex) / np.sqrt(m)

        # --- Evaluate rational approximant at all sample points ---
        # At support points, C has an inf column; D becomes inf, R = NaN.
        # We fix R at support points below.
        i0 = np.where(wj != 0)[0]
        with np.errstate(invalid="ignore"):
            if len(i0) > 0:
                N_vec = C[:, i0] @ (wj[i0] * fj[i0])
                D_vec = C[:, i0] @ wj[i0]
            else:
                N_vec = np.zeros(M, dtype=complex)
                D_vec = np.ones(M, dtype=complex)
            R = N_vec / D_vec
        # At support points, D = inf; interpolate by setting R = F there
        Dinf = ~np.isfinite(D_vec)
        R[Dinf] = F_np[Dinf]

        # --- Check convergence ---
        maxerr = np.linalg.norm(F_np - R, np.inf)
        errvec.append(maxerr)
        if maxerr <= abstol:
            break

    # ---- Remove zero-weight support points ----
    nonzero = wj != 0
    zj = zj[nonzero]
    fj = fj[nonzero]
    wj = wj[nonzero]

    # ---- Convert to JAX arrays ----
    zj_jnp = jnp.array(zj)
    fj_jnp = jnp.array(fj)
    wj_jnp = jnp.array(wj)

    # ---- Cleanup: remove Froissart doublets ----
    if cleanup:
        zj_jnp, fj_jnp, wj_jnp = _cleanup(
            zj_jnp, fj_jnp, wj_jnp,
            jnp.array(Z_np), jnp.array(F_np),
            cleanup_tol,
        )

    # ---- Build poles, zeros from generalised eigenvalue problem ----
    pol, zer = _prz_poles_zeros(zj_jnp, fj_jnp, wj_jnp)

    # ---- Accurate residues via least-squares ----
    res = _compute_residues(
        pol, zer, jnp.array(Z_np), jnp.array(F_np)
    )

    # ---- Build callable ----
    r = _make_callable(zj_jnp, fj_jnp, wj_jnp)

    return r, pol, res, zer, zj_jnp, fj_jnp, wj_jnp


# ---------------------------------------------------------------------------
# Barycentric evaluation (JIT-safe)
# ---------------------------------------------------------------------------

def _make_callable(
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
) -> Callable:
    """Return a JIT-safe callable for the rational approximant."""

    @jax.jit
    def r(zz: jnp.ndarray) -> jnp.ndarray:
        return _reval(zz, zj, fj, wj)

    return r


@jax.jit
def _reval(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a barycentric rational approximant.

    Parameters
    ----------
    zz : jnp.ndarray
        Evaluation points (any shape; will be flattened and reshaped).
    zj, fj, wj : jnp.ndarray, shape (m,)
        Support points, values, and barycentric weights.

    Returns
    -------
    r : jnp.ndarray
        Values of the rational approximant at ``zz`` (same shape as ``zz``).

    Notes
    -----
    This function is JIT-safe and differentiable w.r.t. ``zz``.
    It handles the 0/0 case (evaluation at support points) by replacing
    NaNs with the correct interpolated value.

    Provenance
    ----------
    MATLAB source : reval (sub-function of aaa.m)
    Chebfun commit: 7574c77
    """
    orig_shape = zz.shape
    zv = zz.ravel()

    # Cauchy matrix: CC[i, j] = 1 / (zv[i] - zj[j])
    CC = 1.0 / (zv[:, None] - zj[None, :])  # (M, m)

    N = CC @ (wj * fj)   # numerator
    D = CC @ wj           # denominator

    r = N / D

    # Fix NaNs at support points (0/0 case).
    # An NaN occurs when zv[i] == zj[k] for some k.
    diff = zv[:, None] - zj[None, :]   # (M, m)
    exact_match = diff == 0.0          # (M, m)
    has_match = jnp.any(exact_match, axis=1)   # (M,)
    match_idx = jnp.argmax(exact_match, axis=1)  # (M,) — index of match
    matched_val = fj[match_idx]

    r = jnp.where(has_match, matched_val, r)

    return r.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Poles, zeros (via generalised eigenvalue problem) — runs via numpy/scipy
# ---------------------------------------------------------------------------

def _prz_poles_zeros(
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute poles and zeros via the generalised eigenvalue formulation.

    The poles are eigenvalues of the pencil (E, B) where:

    .. code-block:: text

        E = [[0,   wj^T     ],
             [1,  diag(zj)  ]]   (m+1) x (m+1)

        B = diag([0, 1, ..., 1])

    Zeros replace ``wj`` with ``wj * fj`` in the top row.

    Parameters
    ----------
    zj, fj, wj : jnp.ndarray, shape (m,)

    Returns
    -------
    pol : jnp.ndarray, complex, shape (p,)
        Finite poles.
    zer : jnp.ndarray, complex, shape (q,)
        Finite zeros.

    Provenance
    ----------
    MATLAB source : prz (sub-function of aaa.m)
    Chebfun commit: 7574c77
    """
    m = zj.shape[0]

    # Work in numpy for the eigenvalue computation
    zj_np = np.array(zj, dtype=complex)
    fj_np = np.array(fj, dtype=complex)
    wj_np = np.array(wj, dtype=complex)

    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0

    # --- Poles ---
    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = wj_np
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zj_np)
    evals_pol = spla.eig(E, B, right=False)
    pol_np = evals_pol[np.isfinite(evals_pol)]

    # --- Zeros ---
    E[0, 1:] = wj_np * fj_np
    evals_zer = spla.eig(E, B, right=False)
    zer_np = evals_zer[np.isfinite(evals_zer)]

    return jnp.array(pol_np), jnp.array(zer_np)


def _compute_residues(
    pol: jnp.ndarray,
    zer: jnp.ndarray,
    Z: jnp.ndarray,
    F: jnp.ndarray,
) -> jnp.ndarray:
    """Compute accurate residues at the poles via least-squares.

    Fits the partial-fraction expansion
    ``r(z) = sum_{k=0}^{deg} c_k z^k + sum_j res_j / (z - pol_j)``
    in a least-squares sense to the data (Z, F), where
    ``deg = max(0, len(zer) - len(pol))``.

    Parameters
    ----------
    pol : jnp.ndarray, complex, shape (p,)
    zer : jnp.ndarray, complex, shape (q,)
    Z : jnp.ndarray, complex, shape (M,)
        Sample points.
    F : jnp.ndarray, complex, shape (M,)
        Function values.

    Returns
    -------
    res : jnp.ndarray, complex, shape (p,)

    Provenance
    ----------
    MATLAB source : aaa.m (lines 299–305)
    Chebfun commit: 7574c77
    """
    pol_np = np.array(pol, dtype=complex)
    zer_np = np.array(zer, dtype=complex)
    Z_np = np.array(Z, dtype=complex)
    F_np = np.array(F, dtype=complex)

    n_pol = pol_np.shape[0]
    n_zer = zer_np.shape[0]
    deg = max(0, n_zer - n_pol)
    Z_np.shape[0]

    # Build Vandermonde + partial-fraction matrix
    # Columns: z^0, z^1, ..., z^deg, 1/(z-pol_0), ..., 1/(z-pol_{p-1})
    Acols = [Z_np[:, None] ** k for k in range(deg + 1)]
    Acols += [1.0 / (Z_np[:, None] - p) for p in pol_np]
    if Acols:
        A_ls = np.column_stack(Acols)
    else:
        return jnp.array([], dtype=jnp.complex128)

    c, _, _, _ = np.linalg.lstsq(A_ls, F_np, rcond=None)
    res_np = c[deg + 1:]   # drop polynomial part
    return jnp.array(res_np)


# ---------------------------------------------------------------------------
# Cleanup: Froissart doublet removal
# ---------------------------------------------------------------------------

def _cleanup(
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    Z: jnp.ndarray,
    F: jnp.ndarray,
    cleanup_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Remove spurious pole-zero pairs (Froissart doublets).

    A pole is deemed spurious if its residue is small compared to
    ``cleanup_tol * geom_mean(|F|) * min_dist(pol, Z)``.

    For each spurious pole, the nearest support point is removed and the
    barycentric weights are recomputed from the remaining support points.

    Parameters
    ----------
    zj, fj, wj : jnp.ndarray
        Support points, values, weights.
    Z : jnp.ndarray
        All sample points.
    F : jnp.ndarray
        Function values at sample points.
    cleanup_tol : float

    Returns
    -------
    zj, fj, wj : jnp.ndarray
        Pruned support data.

    Provenance
    ----------
    MATLAB source : cleanup (sub-function of aaa.m)
    Chebfun commit: 7574c77
    """
    pol, zer = _prz_poles_zeros(zj, fj, wj)

    # Accurate residues
    res = _compute_residues(pol, zer, Z, F)

    pol_np = np.array(pol, dtype=complex)
    res_np = np.array(res, dtype=complex)
    Z_np = np.array(Z, dtype=complex)
    F_np = np.array(F, dtype=complex)
    zj_np = np.array(zj, dtype=complex)
    fj_np = np.array(fj, dtype=complex)
    wj_np = np.array(wj, dtype=complex)

    # Geometric mean of |F| (ignoring zeros)
    absF = np.abs(F_np[F_np != 0])
    if len(absF) > 0:
        geom_mean = np.exp(np.mean(np.log(absF)))
    else:
        geom_mean = 0.0

    # Minimum distance from each pole to Z
    if len(pol_np) == 0:
        return jnp.array(zj_np), jnp.array(fj_np), jnp.array(wj_np)

    Zdist = np.array([np.min(np.abs(p - Z_np)) for p in pol_np])

    # Identify spurious poles
    spurious_mask = np.abs(res_np) / (Zdist + 1e-300) < cleanup_tol * geom_mean
    ii = np.where(spurious_mask)[0]
    ni = len(ii)

    if ni == 0:
        return jnp.array(zj_np), jnp.array(fj_np), jnp.array(wj_np)

    if ni == 1:
        warnings.warn("AAA cleanup: 1 Froissart doublet removed.", stacklevel=3)
    else:
        warnings.warn(
            f"AAA cleanup: {ni} Froissart doublets removed.", stacklevel=3
        )

    # Remove closest support point for each spurious pole
    remove_idx = set()
    for j in ii:
        dists = np.abs(zj_np - pol_np[j])
        remove_idx.add(int(np.argmin(dists)))

    keep = np.array([k for k in range(len(zj_np)) if k not in remove_idx])
    if len(keep) == 0:
        return (
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )

    zj_np = zj_np[keep]
    fj_np = fj_np[keep]
    m = len(zj_np)

    # Remove support points from sample set
    mask = np.ones(len(Z_np), dtype=bool)
    for z in zj_np:
        mask &= (Z_np != z)
    Z_sub = Z_np[mask]
    F_sub = F_np[mask]
    M_sub = len(Z_sub)

    if M_sub == 0 or m == 0:
        return (
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )

    # Rebuild Loewner matrix and recompute weights
    C = 1.0 / (Z_sub[:, None] - zj_np[None, :])  # (M_sub, m)
    SF = np.diag(F_sub)
    Sf = np.diag(fj_np)
    A_mat = SF @ C - C @ Sf

    _, _, V = np.linalg.svd(A_mat, full_matrices=False)
    wj_np = V[m - 1, :]  # last row of V^T = last right singular vector

    return jnp.array(zj_np), jnp.array(fj_np), jnp.array(wj_np)
