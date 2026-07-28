# uses-numpy: greedy AAA algorithm is not JIT-safe (iterative point selection, SVD)
"""AAA rational approximation and trigonometric AAA.

Translated from MATLAB Chebfun (commit 7574c77): aaa.m, aaatrig.m.

Original algorithm:
    Y. Nakatsukasa, O. Sete, and L. N. Trefethen,
    "The AAA algorithm for rational approximation",
    SIAM J. Sci. Comp. 40 (2018), A1494–A1522.

Trigonometric extension:
    P. J. Baddoo, "The AAAtrig algorithm for rational approximation
    of periodic functions", SIAM J. Sci. Comp. (2021).

Original authors: Copyright 2023 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.

Design notes
------------
- The greedy support-point selection loop is NOT JIT-safe: it uses
  Python-level data-dependent control flow with dynamic array sizes.
- The returned callable ``r(zz)`` IS JIT-safe: it is a thin wrapper
  around ``_reval`` / ``_revaltrig``, which have static shapes given the
  support points.
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
    degree: int | None = None,
    lawson: "int | float | None" = None,
    damping: float = 1.0,
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
    degree : int or None, optional
        Maximal rational degree ``N`` (like ``mmax = N + 1``).  Unlike
        ``mmax``, specifying ``degree`` turns the AAA-Lawson iteration on by
        default (adaptive number of steps), driving the approximant toward
        the minimax (near-best) rational of that degree.
    lawson : int, float, or None, optional
        Number of AAA-Lawson iteratively reweighted least-squares (IRLS)
        steps used to push the approximant toward minimax.  ``lawson=0``
        disables the iteration (plain AAA).  ``None`` (default) selects the
        MATLAB default: adaptive Lawson when ``degree`` is given, otherwise
        off.  A finite value takes exactly that many steps.
    damping : float, optional
        Lawson damping ratio applied at each IRLS step (default 1.0 =
        standard); values < 1 can be more robust.
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

    # --- Degree / Lawson bookkeeping (MATLAB aaa.m parseInputs) ---
    # ``degree`` acts like ``mmax = degree + 1`` but turns Lawson on by
    # default.  ``nlawson`` defaults to Inf (adaptive); if no degree was
    # given and Lawson was not requested, it collapses to 0 (plain AAA).
    degree_flag = degree is not None
    if degree_flag:
        mmax = int(degree) + 1
    nlawson = float("inf") if lawson is None else float(lawson)
    if (not degree_flag) and (nlawson == float("inf")):
        nlawson = 0.0
    dampratio = float(damping)

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
                # numpy's svd returns Vh; the null vector needs the
                # CONJUGATE transpose (plain .T silently breaks every
                # complex-valued approximation)
                wj = V[mm, :].conj().T @ (np.ones(nm) / np.sqrt(nm))
                wj = wj / col_norms_safe  # un-scale
                wj = wj / np.linalg.norm(wj)
            else:
                idx_min = np.argmin(s)
                tol_sv = s[idx_min] * (1 + 1e-10)
                mm = np.where(s <= tol_sv)[0]
                nm = len(mm)
                wj = V[mm, :].conj().T @ (np.ones(nm) / np.sqrt(nm))

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

    maxerrAAA = maxerr   # error at the end of the AAA greedy phase

    # ---- AAA-Lawson iteration (barycentric IRLS toward minimax) ----
    if nlawson > 0:
        zj, fj, wj = _aaa_lawson(
            zj, fj, wj, Z_np, F_np, nlawson, dampratio, maxerrAAA,
        )

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
    # MATLAB only runs the doublet cleanup when no Lawson steps were taken;
    # the Lawson iteration itself suppresses spurious poles.
    if cleanup and nlawson == 0:
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
# AAA-Lawson iteration (barycentric IRLS)
# ---------------------------------------------------------------------------

def _aaa_lawson(zj, fj, wj, Z, F, nlawson, dampratio, maxerrAAA):
    """Barycentric iteratively reweighted least-squares (Lawson) refinement.

    Refines the AAA barycentric weights toward the minimax approximant by
    IRLS on the Loewner/Cauchy system.  ``nlawson`` is either a finite
    number of steps or ``inf`` (adaptive: at least 20 steps, then continue
    while the max error keeps decreasing, up to 1000).  Returns the updated
    ``(zj, fj, wj)``; support points are unchanged, only the weights and
    the values ``fj`` are recomputed from the IRLS solution.

    Provenance
    ----------
    MATLAB source : aaa.m (Lawson iteration block)
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford and The
        Chebfun Developers.
    Algorithm: Nakatsukasa & Trefethen, "An algorithm for real and complex
        rational minimax approximation", SIAM J. Sci. Comput. 42 (2020).
    """
    Z = np.asarray(Z, dtype=complex)
    F = np.asarray(F, dtype=complex)
    M = len(Z)
    nj = len(zj)

    wj0 = wj.copy()
    fj0 = fj.copy()

    # Cauchy/Loewner matrix A (M x 2*nj): columns [1/(Z-zj), F/(Z-zj)] per zj.
    A = np.zeros((M, 2 * nj), dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(nj):
            col = 1.0 / (Z - zj[j])
            A[:, 2 * j] = col
            A[:, 2 * j + 1] = F * col
    # Support-point rows are special: the barycentric value there is exact.
    for j in range(nj):
        i = np.where(Z == zj[j])[0]
        A[i, :] = 0.0
        A[i, 2 * j] = 1.0
        A[i, 2 * j + 1] = F[i]

    wt_new = np.ones(M)
    maxerrold = maxerrAAA
    maxerr = maxerrAAA
    c = None
    stepno = 0
    inf = float("inf")
    while (((nlawson < inf) and (stepno < nlawson))
           or ((nlawson == inf) and (stepno < 20))
           or ((nlawson == inf) and (maxerr / maxerrold < 0.999)
               and (stepno < 1000))):
        stepno += 1
        wt = wt_new
        # W = diag(sqrt(wt)); smallest right singular vector of W*A.
        WA = np.sqrt(wt)[:, None] * A
        _, _, Vh = np.linalg.svd(WA, full_matrices=False)
        c = Vh[-1, :].conj()

        c_num = c[0::2]     # odd (1-based) entries -> numerator coeffs
        c_den = c[1::2]     # even (1-based) entries -> denominator coeffs
        with np.errstate(divide="ignore", invalid="ignore"):
            cauchy = 1.0 / (Z[:, None] - zj[None, :])   # (M, nj)
            denom = cauchy @ c_den
            num = -(cauchy @ c_num)
            Rvals = num / denom
        for j in range(nj):
            i = np.where(Z == zj[j])[0]
            Rvals[i] = -c_num[j] / c_den[j]

        abserr = np.abs(F - Rvals)
        maxerrold = maxerr
        maxerr = float(np.max(abserr))
        relerr = abserr / maxerr
        wt_new = wt * ((1.0 - dampratio) + dampratio * relerr)
        wt_new = wt_new / np.linalg.norm(wt_new, np.inf)

    if c is not None:
        wj = c[1::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            fj = -c[0::2] / wj
        # If adaptive Lawson failed to improve, restore the AAA weights.
        if (maxerr > maxerrAAA) and (nlawson == inf):
            wj = wj0
            fj = fj0

    return zj, fj, wj


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


def _prz_residues(
    pol: np.ndarray,
    zj: np.ndarray,
    fj: np.ndarray,
    wj: np.ndarray,
) -> np.ndarray:
    """Analytic residues of a barycentric rational at its poles.

    Uses the closed-form residue of a quotient of analytic functions,
    ``res = N(pol) / D'(pol)`` with ``N(t) = sum_j (fj*wj)/(t - zj)`` and
    ``D'(t) = -sum_j wj/(t - zj)^2``.  This is exactly what MATLAB Chebfun's
    ``prz`` computes and hands to ``cleanup`` for Froissart-doublet detection.

    These residues depend only on the (well-determined) poles and the
    barycentric data, so they are deterministic across LAPACK builds — unlike
    the least-squares residues from :func:`_compute_residues`, whose values at
    near-structural poles are set by a rank-deficient solve and vary by many
    orders of magnitude between platforms.

    Provenance
    ----------
    MATLAB source : prz (sub-function of aaa.m, lines 715-717)
    Chebfun commit: 7574c77
    """
    pol = np.asarray(pol, dtype=complex)
    zj = np.asarray(zj, dtype=complex)
    fj = np.asarray(fj, dtype=complex)
    wj = np.asarray(wj, dtype=complex)
    if pol.size == 0:
        return np.zeros(0, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        CC = 1.0 / (pol[:, None] - zj[None, :])   # (p, m)
        N = CC @ (fj * wj)
        Ddiff = -(CC ** 2) @ wj
        res = N / Ddiff
    return res


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

    finite = np.all(np.isfinite(A_ls), axis=1) & np.isfinite(F_np)
    c, _, _, _ = np.linalg.lstsq(A_ls[finite], F_np[finite],
                                 rcond=None)
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
    pol, _zer = _prz_poles_zeros(zj, fj, wj)

    pol_np = np.array(pol, dtype=complex)
    Z_np = np.array(Z, dtype=complex)
    F_np = np.array(F, dtype=complex)
    zj_np = np.array(zj, dtype=complex)
    fj_np = np.array(fj, dtype=complex)
    wj_np = np.array(wj, dtype=complex)

    # Residues for spurious detection: MATLAB's cleanup uses the analytic prz
    # residues (aaa.m lines 274, 717), NOT the least-squares residues.  The
    # least-squares residues at near-structural poles are set by a
    # rank-deficient solve and vary by many orders of magnitude across LAPACK
    # builds, which made cleanup non-deterministic (see _prz_residues).
    res_np = _prz_residues(pol_np, zj_np, fj_np, wj_np)

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

    # Identify spurious poles.  Divide directly like MATLAB: for Zdist == 0 the
    # ratio is Inf/NaN, which compares False (not spurious).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(res_np) / Zdist
    spurious_mask = ratio < cleanup_tol * geom_mean
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

    # For each spurious pole, remove the closest *remaining* support point.
    # MATLAB removes them sequentially (aaa.m lines 513-519): the support set
    # shrinks as we go, so ni distinct points are removed even if two spurious
    # poles share a nearest support point.
    for j in ii:
        azp = np.abs(zj_np - pol_np[j])
        jj = int(np.argmin(azp))   # find(azp == min(azp), 1): first minimiser
        zj_np = np.delete(zj_np, jj)
        fj_np = np.delete(fj_np, jj)

    if len(zj_np) == 0:
        return (
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )

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
    # numpy returns Vh: the right singular vector is the CONJUGATE of
    # its last row (plain row breaks complex-valued data)
    wj_np = V[m - 1, :].conj()

    return jnp.array(zj_np), jnp.array(fj_np), jnp.array(wj_np)


# ---------------------------------------------------------------------------
# Trigonometric AAA  (aaatrig)
# ---------------------------------------------------------------------------


def aaatrig(
    F: jnp.ndarray | Callable,
    Z: jnp.ndarray,
    *,
    tol: float = 1e-13,
    mmax: int = 100,
    form: str = "odd",
    cleanup: bool = True,
    cleanup_tol: float | None = None,
) -> tuple[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, list]:
    """Trigonometric AAA rational approximation.

    Computes a near-best trigonometric rational approximant to ``F`` on the
    sample set ``Z`` using the AAAtrig algorithm.  The approximant is
    periodic with period 2*pi and represented in trigonometric barycentric
    form:

    .. math::

        r(z) = \\frac{\\sum_j w_j f_j \\, \\text{cst}((z-z_j)/2)}
                     {\\sum_j w_j \\, \\text{cst}((z-z_j)/2)}

    where cst = csc for ``form='odd'`` (default) and cst = cot for
    ``form='even'``.

    .. note::
        The construction loop is **not JIT-safe** (greedy point selection).
        The returned callable ``r`` **is JIT-safe**.

    Parameters
    ----------
    F : array_like or callable
        Function values at ``Z``, or a callable.
    Z : array_like, shape (M,)
        Sample points (real, typically in [0, 2*pi]).
    tol : float, optional
        Relative tolerance (default 1e-13).
    mmax : int, optional
        Maximum number of support points (default 100).
    form : {'odd', 'even'}, optional
        Trigonometric basis type.  'odd' uses csc; 'even' uses cot.
    cleanup : bool, optional
        Remove Froissart doublets (default True).
    cleanup_tol : float or None, optional
        Cleanup threshold (defaults to ``tol``).

    Returns
    -------
    r : callable
        Trigonometric rational approximant (JIT-safe).
    pol : jnp.ndarray, complex
        Poles.
    res : jnp.ndarray, complex
        Residues.
    zer : jnp.ndarray, complex
        Zeros.
    zj : jnp.ndarray, complex
        Support points.
    fj : jnp.ndarray, complex
        Function values at support points.
    wj : jnp.ndarray, complex
        Barycentric weights.
    errvec : list of float
        Error at each greedy step.

    References
    ----------
    .. [1] P. J. Baddoo, "The AAAtrig algorithm for rational approximation
       of periodic functions", SIAM J. Sci. Comp. (2021).
    .. [2] Y. Nakatsukasa, O. Sete, and L. N. Trefethen,
       "The AAA algorithm for rational approximation",
       SIAM J. Sci. Comp. 40 (2018), A1494–A1522.

    Provenance
    ----------
    MATLAB source : aaatrig.m
    Chebfun commit: 7574c77
    Original authors: Peter Baddoo, Yuji Nakatsukasa, Lloyd N. Trefethen.
        Copyright 2017-2021 by The University of Oxford and The Chebfun
        Developers.

    See Also
    --------
    aaa
    """
    Z_in = jnp.asarray(Z, dtype=jnp.complex128).ravel()
    M = Z_in.shape[0]

    if callable(F):
        F_vals = jnp.asarray(F(Z_in), dtype=jnp.complex128).ravel()
    else:
        F_vals = jnp.asarray(F, dtype=jnp.complex128).ravel()
        if F_vals.shape[0] != M:
            raise ValueError(
                f"F and Z must have the same length, got {F_vals.shape[0]} and {M}."
            )

    # Remove Inf/NaN
    keep = jnp.isfinite(F_vals)
    F_vals = F_vals[keep]
    Z_in = Z_in[keep]

    Z_np = np.array(Z_in, dtype=complex)
    F_np = np.array(F_vals, dtype=complex)

    # Project to [0, 2*pi)
    Z_np = Z_np - 2 * np.pi * np.floor(np.real(Z_np / (2 * np.pi)))

    # Remove duplicates
    _, uni = np.unique(Z_np, return_index=True)
    uni = np.sort(uni)
    Z_np = Z_np[uni]
    F_np = F_np[uni]
    M = len(Z_np)

    if cleanup_tol is None:
        cleanup_tol = tol if tol > 0 else 1e-13

    reltol = tol * np.linalg.norm(F_np, np.inf)

    # Basis function
    if form == "even":
        def cst(z):
            return 1.0 / np.tan(z)
    else:
        def cst(z):
            return 1.0 / np.sin(z)

    # AAA greedy iteration
    J = list(range(M))
    zj = np.zeros(0, dtype=complex)
    fj = np.zeros(0, dtype=complex)
    C = np.zeros((M, 0), dtype=complex)
    errvec = []
    R = np.full(M, np.mean(F_np), dtype=complex)
    wj = np.array([], dtype=complex)

    for m in range(1, mmax + 1):
        J_arr = np.array(J)
        resids = np.abs(F_np[J_arr] - R[J_arr])
        jj = int(np.argmax(resids))
        idx = J_arr[jj]

        zj = np.append(zj, Z_np[idx])
        fj = np.append(fj, F_np[idx])

        with np.errstate(divide="ignore", invalid="ignore"):
            new_col = cst((Z_np - Z_np[idx]) / 2.0)
        C = np.column_stack([C, new_col]) if C.shape[1] > 0 else new_col[:, None]
        J.pop(jj)

        # Loewner matrix and SVD
        J_arr = np.array(J)
        if len(J_arr) > 0:
            SF = np.diag(F_np[J_arr])
            Sf = np.diag(fj)
            A_sub = SF @ C[J_arr, :] - C[J_arr, :] @ Sf
            _, _, V = np.linalg.svd(A_sub, full_matrices=False)
            wj = V[m - 1, :].conj() if V.shape[0] >= m \
                else V[-1, :].conj()
        else:
            wj = np.ones(m, dtype=complex) / np.sqrt(m)

        # Evaluate approximant
        with np.errstate(invalid="ignore"):
            N_vec = C @ (wj * fj)
            D_vec = C @ wj
            R = F_np.copy()
            J_cur = np.array(J)
            if len(J_cur) > 0:
                R[J_cur] = N_vec[J_cur] / D_vec[J_cur]

        maxerr = np.linalg.norm(F_np - R, np.inf)
        errvec.append(maxerr)
        if maxerr <= reltol:
            break

    # Remove zero-weight support points
    nonzero = wj != 0
    zj = zj[nonzero]
    fj = fj[nonzero]
    wj = wj[nonzero]

    zj_jnp = jnp.array(zj)
    fj_jnp = jnp.array(fj)
    wj_jnp = jnp.array(wj)

    # Poles and zeros
    pol, res, zer = _prztrig_np(zj, fj, wj, form)

    # Cleanup
    if cleanup:
        zj_jnp, fj_jnp, wj_jnp = _cleanup_trig(
            zj_jnp, fj_jnp, wj_jnp,
            jnp.array(Z_np), jnp.array(F_np),
            cleanup_tol, form,
        )
        pol, res, zer = _prztrig_np(
            np.array(zj_jnp), np.array(fj_jnp), np.array(wj_jnp), form
        )

    pol_jnp = jnp.array(pol)
    res_jnp = jnp.array(res)
    zer_jnp = jnp.array(zer)

    r = _make_trig_callable(zj_jnp, fj_jnp, wj_jnp, form)

    return r, pol_jnp, res_jnp, zer_jnp, zj_jnp, fj_jnp, wj_jnp, errvec


def _make_trig_callable(
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    form: str,
) -> Callable:
    """Return a JIT-safe callable for the trigonometric rational approximant."""
    import jax

    @jax.jit
    def r(zz: jnp.ndarray) -> jnp.ndarray:
        return _revaltrig(zz, zj, fj, wj, form)

    return r


@jax.jit
def _revaltrig_csc(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate odd trigonometric barycentric rational function (csc basis)."""
    orig_shape = zz.shape
    zv = zz.ravel()

    diff_half = (zv[:, None] - zj[None, :]) / 2.0
    CC = 1.0 / jnp.sin(diff_half)

    N = CC @ (wj * fj)
    D = CC @ wj
    r = N / D

    # Fix NaNs at support points
    diff = zv[:, None] - zj[None, :]
    exact = diff == 0.0
    has_match = jnp.any(exact, axis=1)
    match_idx = jnp.argmax(exact, axis=1)
    r = jnp.where(has_match, fj[match_idx], r)

    return r.reshape(orig_shape)


@jax.jit
def _revaltrig_cot(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate even trigonometric barycentric rational function (cot basis)."""
    orig_shape = zz.shape
    zv = zz.ravel()

    diff_half = (zv[:, None] - zj[None, :]) / 2.0
    CC = 1.0 / jnp.tan(diff_half)

    N = CC @ (wj * fj)
    D = CC @ wj
    r = N / D

    diff = zv[:, None] - zj[None, :]
    exact = diff == 0.0
    has_match = jnp.any(exact, axis=1)
    match_idx = jnp.argmax(exact, axis=1)
    r = jnp.where(has_match, fj[match_idx], r)

    return r.reshape(orig_shape)


def _revaltrig(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    form: str,
) -> jnp.ndarray:
    """Dispatch to csc or cot evaluator."""
    if form == "even":
        return _revaltrig_cot(zz, zj, fj, wj)
    else:
        return _revaltrig_csc(zz, zj, fj, wj)


def _prztrig_np(
    zj: np.ndarray,
    fj: np.ndarray,
    wj: np.ndarray,
    form: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute poles, residues, zeros of a trigonometric barycentric rational."""
    m = len(wj)

    if form == "odd":
        # Coordinate transformation: z -> exp(i*z)
        zjp = np.exp(1j * zj)
        wjp = wj * np.exp(1j * zj / 2)

        B = np.eye(m + 1, dtype=complex)
        B[0, 0] = 0.0

        # Poles
        Ep = np.zeros((m + 1, m + 1), dtype=complex)
        Ep[0, 1:] = wjp
        Ep[1:, 0] = 1.0
        Ep[1:, 1:] = np.diag(zjp)

        with np.errstate(divide="ignore", invalid="ignore"):
            polp_eig = spla.eig(Ep, B, right=False)
        polp = polp_eig[np.isfinite(polp_eig)]
        pol = -1j * np.log(polp + 0j)

        # Zeros
        Ep[0, 1:] = fj * wjp
        with np.errstate(divide="ignore", invalid="ignore"):
            zerp_eig = spla.eig(Ep, B, right=False)
        zerp = zerp_eig[np.isfinite(zerp_eig)]
        zer = -1j * np.log(zerp + 0j)

        # Handle poles/zeros at +/-i*Inf
        zer[np.abs(zerp) < 1e-10] = 1j * np.inf
        zer[np.abs(zerp) > 1e10] = -1j * np.inf
        pol[np.abs(polp) < 1e-10] = 1j * np.inf
        pol[np.abs(polp) > 1e10] = -1j * np.inf

    else:
        # form == "even": cot basis
        d_mask = zj != np.pi
        zjp = np.tan(zj[d_mask] / 2)
        wjp = wj[d_mask] * (1 + zjp ** 2)
        cd = np.sum(zjp * wj[d_mask])

        B = np.eye(m + 1, dtype=complex)
        B[0, 0] = 0.0

        if np.all(d_mask):
            Ep = np.zeros((m + 1, m + 1), dtype=complex)
            Ep[0, 0] = cd
            Ep[0, 1:] = wjp
            Ep[1:, 0] = 1.0
            Ep[1:, 1:] = np.diag(zjp)
        else:
            Ep = np.zeros((m + 1, m + 1), dtype=complex)
            Ep[0, 0] = -wj[~d_mask][0]
            Ep[0, 1] = cd
            Ep[0, 2:] = wjp
            Ep[1, 0] = 1.0
            Ep[2:, 1] = 1.0
            Ep[2:, 2:] = np.diag(zjp)
            Ep[1, 1] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            polp_eig = spla.eig(Ep, B, right=False)
        polp = polp_eig[np.isfinite(polp_eig)]
        pol = 2 * np.arctan(polp)

        cn = np.sum(fj[d_mask] * zjp * wj[d_mask])
        if np.all(d_mask):
            Ez = Ep.copy()
            Ez[0, 0] = cn
            Ez[0, 1:] = fj[d_mask] * wjp
        else:
            Ez = Ep.copy()
            Ez[0, 0] = -fj[~d_mask][0] * wj[~d_mask][0]
            Ez[0, 1] = cn
            Ez[0, 2:] = fj[d_mask] * wjp

        with np.errstate(divide="ignore", invalid="ignore"):
            zerp_eig = spla.eig(Ez, B, right=False)
        zerp = zerp_eig[np.isfinite(zerp_eig)]
        zer = 2 * np.arctan(zerp)

    # Project to [0, 2*pi) and compute residues
    pol = pol - 2 * np.pi * np.floor(np.real(pol / (2 * np.pi)))
    zer = zer - 2 * np.pi * np.floor(np.real(zer / (2 * np.pi)))

    if len(pol) > 0:
        if form == "odd":
            def N_fn(t):
                return (1.0 / np.sin((t[:, None] - zj[None, :]) / 2)) @ (fj * wj)

            def Ddiff_fn(t):
                d = (t[:, None] - zj[None, :]) / 2
                return -0.5 * (1.0 / np.sin(d)) * (1.0 / np.tan(d)) @ wj
        else:
            def N_fn(t):
                return (1.0 / np.tan((t[:, None] - zj[None, :]) / 2)) @ (fj * wj)

            def Ddiff_fn(t):
                d = (t[:, None] - zj[None, :]) / 2
                return -0.5 / np.sin(d) ** 2 @ wj

        pol_real = np.real(pol)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                res = N_fn(pol_real[:, None].T.ravel()) / Ddiff_fn(pol_real[:, None].T.ravel())
        except Exception:
            res = np.zeros_like(pol)
    else:
        res = np.array([], dtype=complex)

    return pol, res, zer


def _cleanup_trig(
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    Z: jnp.ndarray,
    F: jnp.ndarray,
    cleanup_tol: float,
    form: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Remove Froissart doublets from a trigonometric AAA approximant."""
    pol, res, zer = _prztrig_np(
        np.array(zj), np.array(fj), np.array(wj), form
    )
    pol_np = np.array(pol, dtype=complex)
    res_np = np.array(res, dtype=complex)
    Z_np = np.array(Z, dtype=complex)
    F_np = np.array(F, dtype=complex)
    zj_np = np.array(zj, dtype=complex)
    fj_np = np.array(fj, dtype=complex)
    wj_np = np.array(wj, dtype=complex)

    if len(pol_np) == 0:
        return zj, fj, wj

    # Geometric mean of |F|
    absF = np.abs(F_np[F_np != 0])
    geom_mean = np.exp(np.mean(np.log(absF))) if len(absF) > 0 else 0.0

    Zdist = np.array([np.min(np.abs(p - Z_np)) for p in pol_np])
    spurious = np.abs(res_np) / (Zdist + 1e-300) < cleanup_tol * geom_mean
    ii = np.where(spurious)[0]

    if len(ii) == 0:
        return zj, fj, wj

    import warnings
    warnings.warn(f"AAAtrig cleanup: {len(ii)} Froissart doublets removed.", stacklevel=4)

    remove_idx = set()
    for j in ii:
        # Find closest support point modulo 2*pi
        np_diff = np.floor(np.real((zj_np - pol_np[j]) / np.pi)).astype(int)
        azp = np.abs(zj_np - (pol_np[j] + np_diff * 2 * np.pi))
        remove_idx.add(int(np.argmin(azp)))

    keep = [k for k in range(len(zj_np)) if k not in remove_idx]
    if len(keep) == 0:
        return (
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )

    zj_np = zj_np[keep]
    fj_np = fj_np[keep]
    m = len(zj_np)

    # Remove support points from Z
    mask = np.ones(len(Z_np), dtype=bool)
    for z in zj_np:
        mask &= (Z_np != z)
    Z_sub = Z_np[mask]
    F_sub = F_np[mask]

    if form == "even":
        def cst_np(z):
            return 1.0 / np.tan(z)
    else:
        def cst_np(z):
            return 1.0 / np.sin(z)

    C = cst_np((Z_sub[:, None] - zj_np[None, :]) / 2.0)
    SF = np.diag(F_sub)
    Sf = np.diag(fj_np)
    A_mat = SF @ C - C @ Sf

    _, _, V = np.linalg.svd(A_mat, full_matrices=False)
    wj_np = V[m - 1, :].conj()   # Vh row -> conjugate (complex data)

    return jnp.array(zj_np), jnp.array(fj_np), jnp.array(wj_np)
