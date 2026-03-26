# uses-numpy: Remez exchange uses numpy/scipy for iterative optimization (not JIT-safe)
# uses-numpy: adaptive Remez exchange loop is not JIT-safe (data-dependent
#             control flow, dynamic array sizes, scipy linear-algebra calls)
"""Best polynomial (and rational) approximation via the Remez exchange algorithm.

Translated from MATLAB Chebfun (commit 7574c77): minimax.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Design notes
------------
- The main Remez exchange loop is **not JIT-safe**: it uses Python-level
  data-dependent control flow and creates arrays of varying length.
- All inner linear-algebra (barycentric weights, QR, eigenvalues) is
  carried out in NumPy/SciPy for precision and speed.
- The returned coefficient array *is* a JAX array (float64).
- The polynomial case (``rational=False``) is highly reliable and matches
  MATLAB accuracy to near machine precision.
- The rational case (``rational=True``) uses the barycentric-Remez method
  of Beckermann, Filip, Nakatsukasa and Trefethen (2018); it is not yet
  implemented.

References
----------
.. [1] R. Pachon and L. N. Trefethen, "Barycentric-Remez algorithms for best
   polynomial approximation in the chebfun system", BIT Numerical Mathematics,
   49:721-742, 2009.
.. [2] B. Beckermann, S. Filip, Y. Nakatsukasa and L. N. Trefethen,
   "Rational minimax approximation via adaptive barycentric representations",
   arXiv:1705.10132.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.interpolation import bary, bary_weights
from chebfunjax.utils.quadrature import chebpts_ab
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

__all__ = ["minimax", "trigremez", "MinimaxResult", "TrigremezResult"]

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class MinimaxResult:
    """Result of a minimax approximation computation.

    Attributes
    ----------
    coeffs : jnp.ndarray, shape (n+1,)
        Chebyshev coefficients of the best polynomial approximant ``p``,
        expressed on the approximation ``domain``.  The polynomial evaluates
        as ``sum_{k=0}^{n} coeffs[k] * T_k(x_hat)`` where
        ``x_hat = 2*(x - a)/(b - a) - 1`` maps ``x`` from ``[a, b]``
        to ``[-1, 1]``.
    xk : jnp.ndarray, shape (n+2,)
        Equioscillation reference points (the final exchange set).
    err : float
        Supremum norm of the error ``f - p`` on the domain.
    delta : float
        Normalised equioscillation deviation ``(err - |h|) / normf``.
        This is near zero for a converged best approximation.
    iter : int
        Number of Remez iterations performed.
    domain : tuple[float, float]
        Approximation domain ``(a, b)``.
    """

    coeffs: jnp.ndarray
    xk: jnp.ndarray
    err: float
    delta: float
    iter: int
    domain: tuple[float, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def minimax(
    f: Callable,
    n: int,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    tol: float | None = None,
    max_iter: int = 30,
    init_xk: np.ndarray | None = None,
    breakpoints: Sequence[float] | None = None,
    rational: bool = False,
) -> MinimaxResult:
    """Best polynomial approximation of degree ``n`` via the Remez algorithm.

    Computes the minimax (best Chebyshev / supremum-norm) polynomial
    approximation of degree ``n`` to the real-valued function ``f`` on
    ``domain``.  The implementation follows the Remez exchange algorithm
    with the full-exchange rule (Pachon & Trefethen 2009).

    Parameters
    ----------
    f : callable
        Real-valued function.  Must accept a 1-D ``jnp.ndarray`` and return
        a 1-D array-like of the same shape.  Evaluated many times inside the
        loop.
    n : int
        Degree of the best polynomial approximant (number of free
        coefficients is ``n + 1``).
    domain : tuple[float, float], optional
        Approximation interval ``(a, b)``.  Default ``(-1.0, 1.0)``.
    tol : float or None, optional
        Relative equioscillation tolerance for convergence.  The algorithm
        stops when ``|err - |h|| / err < tol``.
        Default: ``1e-14 * (n**2 + 10)`` (matches MATLAB polynomial case).
    max_iter : int, optional
        Maximum number of Remez iterations.  Default 30.
    init_xk : array_like or None, optional
        Initial reference set (length ``n + 2``).  If ``None``, Chebyshev
        points of the 2nd kind are used.
    breakpoints : sequence of float or None, optional
        Additional breakpoints (e.g., kink locations) for the error function.
        These are added to the sub-interval partition in ``_find_extrema``
        to improve root-finding accuracy near non-smooth points of ``f``.
        If ``None``, no extra breakpoints are added.  For piecewise-smooth
        functions (e.g., ``abs(x)`` with a kink at 0), passing the kink
        location here reproduces MATLAB Chebfun's behavior, which detects
        breakpoints automatically via ``splitting=on``.
    rational : bool, optional
        Not yet implemented.  Must be ``False`` (default).

    Returns
    -------
    result : MinimaxResult
        Dataclass with fields:

        - ``coeffs`` — Chebyshev coefficients of the best polynomial
          (length ``n+1``).
        - ``xk`` — equioscillation reference points (length ``n+2``).
        - ``err`` — max-norm error ``max|f - p|`` on the domain.
        - ``delta`` — normalised equioscillation deviation (near 0 when
          converged).
        - ``iter`` — number of iterations performed.
        - ``domain`` — the approximation domain ``(a, b)``.

    Raises
    ------
    NotImplementedError
        If ``rational=True``.
    ValueError
        If ``n < 0`` or the domain is invalid.

    Examples
    --------
    Approximate ``|x|`` with a degree-10 polynomial on ``[-1, 1]``:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.minimax import minimax
    >>> res = minimax(jnp.abs, 10)
    >>> round(float(res.err), 4)
    0.0278
    >>> len(res.coeffs)
    11

    Notes
    -----
    Developer notes from MATLAB Chebfun:

    At each iteration:

    1. Compute barycentric weights for the current reference ``xk``
       (length ``n+2``).
    2. Solve for the levelled reference error ``h`` and the polynomial
       values at ``xk`` via the barycentric formula:
       ``h = (w \xb7 fk) / (w \xb7 sigma)``, where ``sigma = [+1, -1, +1, ...]``.
    3. Interpolate ``fk - h * sigma`` at ``xk`` using barycentric
       interpolation; sample at ``n+1`` Chebyshev-2 pts to get Chebyshev
       coefficients.
    4. Refine the reference via the full-exchange rule: find all extrema of
       ``f - p`` above level ``|h|`` and select ``n+2`` consecutive extrema
       containing the maximum.
    5. Repeat until ``|err - |h|| / err < tol``.

    The best approximant over all iterations (minimum ``err - |h|``) is
    returned.

    Provenance
    ----------
    MATLAB source : minimax.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Pachon & Trefethen, BIT Numerical Mathematics, 49, 2009.
        [2] Beckermann, Filip, Nakatsukasa, Trefethen, arXiv:1705.10132.

    See Also
    --------
    aaa, chebpts, bary, bary_weights
    """
    if rational:
        raise NotImplementedError(
            "minimax: rational=True is not yet implemented. "
            "Use rational=False for the polynomial Remez algorithm."
        )

    a, b = float(domain[0]), float(domain[1])
    if a >= b:
        raise ValueError(
            f"minimax: domain must satisfy a < b, got ({a}, {b})."
        )
    if n < 0:
        raise ValueError(f"minimax: degree n must be >= 0, got {n}.")

    n_ref = n + 2  # size of reference set

    # ---- Default tolerance (matches MATLAB polynomial case) ----
    if tol is None:
        tol = 1e-14 * (n ** 2 + 10)

    # ---- Extra breakpoints (kink locations) ----
    extra_bkpts: list[float] = []
    if breakpoints is not None:
        for bp in breakpoints:
            bpf = float(bp)
            if a < bpf < b:
                extra_bkpts.append(bpf)
        extra_bkpts.sort()

    # ---- Estimate function norm ----
    # Sample on a dense Chebyshev-2 grid to estimate max|f|.
    n_dense = max(4 * n_ref, 512)
    dense_pts = np.array(chebpts_ab(n_dense, a, b), dtype=np.float64)
    fvals_dense = np.asarray(f(jnp.array(dense_pts)), dtype=np.float64).ravel()
    normf = float(np.max(np.abs(fvals_dense)))
    if normf == 0.0:
        normf = float(np.finfo(np.float64).eps)

    # ---- Initialise reference set xk ----
    if init_xk is not None:
        xk = np.asarray(init_xk, dtype=np.float64).ravel()
        if len(xk) != n_ref:
            raise ValueError(
                f"minimax: init_xk must have length n+2={n_ref}, "
                f"got {len(xk)}."
            )
        xk = np.sort(xk)
    else:
        # Chebyshev-2 pts on [a, b] (ascending order from chebpts_ab)
        xk = np.array(chebpts_ab(n_ref, a, b), dtype=np.float64)

    xo = xk.copy()

    # ---- Iteration state ----
    iter_count = 0
    delta_min = np.inf
    diffx = 1.0

    # Initialise h so the while condition triggers at least one iteration
    err = normf
    h = 2.0 * err + 1.0

    # Best-so-far storage
    p_coeffs_min: np.ndarray | None = None
    err_min = np.inf
    xk_min = xk.copy()

    # ---- Main Remez loop ----
    while (
        abs(abs(h) - abs(err)) / abs(err) > tol
        and iter_count < max_iter
        and diffx > 0
    ):
        # Machine-precision convergence guard
        if abs(abs(h) - abs(err)) / normf < 1e-14:
            break

        # ---- Compute trial polynomial ----
        fk = np.asarray(f(jnp.array(xk)), dtype=np.float64).ravel()
        w_jax = bary_weights(jnp.array(xk, dtype=jnp.float64))
        w = np.array(w_jax, dtype=np.float64)

        p_coeffs, h = _compute_trial_polynomial(fk, xk, w, n, n_ref, a, b)

        # Perturb exactly-zero levelled error
        if h == 0.0:
            h = 1e-19

        # ---- Full-exchange: update reference set ----
        xk_new, err_new, _flag = _exchange(
            xk, h, 2, f, p_coeffs, n_ref, a, b, extra_bkpts
        )

        # If overshoot, fall back to one-point exchange
        if err_new / normf > 1e5:
            xk_new, err_new, _flag = _exchange(
                xo, h, 1, f, p_coeffs, n_ref, a, b, extra_bkpts
            )

        xk = xk_new
        err = err_new
        diffx = float(np.max(np.abs(xo - xk))) if len(xo) == len(xk) else 1.0
        delta = err - abs(h)

        # Store best (minimum delta) result
        if delta < delta_min:
            p_coeffs_min = p_coeffs.copy()
            err_min = err
            xk_min = xk.copy()
            delta_min = delta

        xo = xk.copy()
        iter_count += 1

    # Use best result over all iterations
    if p_coeffs_min is not None:
        p_coeffs_final = p_coeffs_min
        err_final = err_min
        xk_final = xk_min
    else:
        # Loop never iterated -- compute from initial xk
        fk = np.asarray(f(jnp.array(xk)), dtype=np.float64).ravel()
        w_jax = bary_weights(jnp.array(xk, dtype=jnp.float64))
        w = np.array(w_jax, dtype=np.float64)
        p_coeffs_final, h = _compute_trial_polynomial(fk, xk, w, n, n_ref, a, b)
        p_vals_final = _eval_poly_bary(xk, p_coeffs_final, a, b)
        err_final = float(np.max(np.abs(fk - p_vals_final)))
        delta_min = err_final - abs(h)
        xk_final = xk.copy()

    # Warn if not converged
    if (
        abs(abs(h) - abs(err)) / abs(err) > tol
        and abs(abs(h) - abs(err)) / normf >= 1e-14
    ):
        warnings.warn(
            f"minimax: algorithm did not converge after {iter_count} "
            f"iterations to tolerance {tol:.3e}. "
            f"Best delta/normf = {delta_min / normf:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return MinimaxResult(
        coeffs=jnp.array(p_coeffs_final, dtype=jnp.float64),
        xk=jnp.array(xk_final, dtype=jnp.float64),
        err=float(err_final),
        delta=float(delta_min) / normf,
        iter=iter_count,
        domain=(a, b),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_trial_polynomial(
    fk: np.ndarray,
    xk: np.ndarray,
    w: np.ndarray,
    m: int,
    n_ref: int,
    a: float,
    b: float,
) -> tuple[np.ndarray, float]:
    """Compute trial polynomial and levelled reference error.

    Given ``fk = f(xk)`` and barycentric weights ``w`` for the current
    reference ``xk``, solves the Chebyshev approximation system:

    ``h = (w . fk) / (w . sigma)``

    and constructs the polynomial of degree ``m`` that interpolates
    ``fk - h * sigma`` at ``xk``.

    Parameters
    ----------
    fk : np.ndarray, shape (n_ref,)
        Function values at ``xk``.
    xk : np.ndarray, shape (n_ref,)
        Current reference points (ascending order).
    w : np.ndarray, shape (n_ref,)
        Barycentric weights for ``xk``.
    m : int
        Polynomial degree.
    n_ref : int
        Length of reference (= m + 2).
    a, b : float
        Domain endpoints.

    Returns
    -------
    coeffs : np.ndarray, shape (m+1,)
        Chebyshev coefficients of the trial polynomial on [a, b].
        Stored in ascending degree order (c[0] = T_0 coefficient, etc.),
        matching the convention of ``vals2coeffs``.
    h : float
        Levelled reference error.

    Provenance
    ----------
    MATLAB source : computeTrialFunctionPolynomial (sub-function of minimax.m)
    Chebfun commit: 7574c77
    """
    # Alternating-sign vector sigma = [+1, -1, +1, -1, ...]
    sigma = np.ones(n_ref, dtype=np.float64)
    sigma[1::2] = -1.0

    # Levelled reference error
    h = float(w.dot(fk) / w.dot(sigma))

    # Values to be interpolated at the reference
    pk = fk - h * sigma

    # Evaluate the barycentric interpolant at m+1 Chebyshev-2 pts on [a, b].
    # chebpts_ab returns ascending order.
    cheb_pts_asc = np.array(chebpts_ab(m + 1, a, b), dtype=np.float64)

    p_vals_at_cheb = np.array(
        bary(
            jnp.array(cheb_pts_asc, dtype=jnp.float64),
            jnp.array(pk, dtype=jnp.float64),
            jnp.array(xk, dtype=jnp.float64),
            jnp.array(w, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )

    # vals2coeffs accepts values in ascending x-order (matching chebpts_ab).
    coeffs = np.array(
        vals2coeffs(jnp.array(p_vals_at_cheb, dtype=jnp.float64)),
        dtype=np.float64,
    )
    return coeffs, h


def _eval_poly_bary(
    x: np.ndarray,
    coeffs: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Evaluate a polynomial (given by Chebyshev coefficients) at points x.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points in [a, b].
    coeffs : np.ndarray, shape (m+1,)
        Chebyshev coefficients (ascending degree order).
    a, b : float
        Domain endpoints.

    Returns
    -------
    vals : np.ndarray, shape (len(x),)
        Polynomial values at ``x``.

    Notes
    -----
    Uses coeffs2vals to convert to values at Chebyshev-2 pts, then
    barycentric interpolation to evaluate at arbitrary x.
    """
    m_plus1 = len(coeffs)
    # coeffs2vals returns values in ascending x-order (matching chebpts_ab).
    vals_asc = np.array(
        coeffs2vals(jnp.array(coeffs, dtype=jnp.float64)),
        dtype=np.float64,
    )
    # chebpts_ab returns ascending order.
    cheb_pts_asc = np.array(chebpts_ab(m_plus1, a, b), dtype=np.float64)
    w = np.array(
        bary_weights(jnp.array(cheb_pts_asc, dtype=jnp.float64)),
        dtype=np.float64,
    )
    result = np.array(
        bary(
            jnp.array(x, dtype=jnp.float64),
            jnp.array(vals_asc, dtype=jnp.float64),
            jnp.array(cheb_pts_asc, dtype=jnp.float64),
            jnp.array(w, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    return result


def _find_extrema(
    f: Callable,
    p_coeffs: np.ndarray,
    xk: np.ndarray,
    a: float,
    b: float,
    extra_bkpts: list[float] | None = None,
) -> np.ndarray:
    """Find extrema of the error function ``f - p`` on the domain.

    Sub-divides the domain at the current reference points ``xk``
    (plus any extra breakpoints for non-smooth functions) and finds
    the roots of ``(f - p)'`` in each sub-interval via a Chebyshev-U
    colleague matrix eigenvalue problem (Remez exchange step).

    Parameters
    ----------
    f : callable
        Target function.
    p_coeffs : np.ndarray, shape (m+1,)
        Chebyshev coefficients of the polynomial on [a, b].
    xk : np.ndarray, shape (n+2,)
        Current reference points (used as sub-interval breakpoints).
    a, b : float
        Domain endpoints.
    extra_bkpts : list of float or None, optional
        Additional breakpoints (e.g., kink locations of ``f``).

    Returns
    -------
    rts : np.ndarray
        Sorted, unique candidate extrema (including the endpoints a and b).

    Provenance
    ----------
    MATLAB source : findExtrema, rootsdiff (sub-functions of minimax.m)
    Chebfun commit: 7574c77
    """
    # Sub-interval breakpoints: domain endpoints + reference points + extra kinks
    all_bkpts = [a, b]
    if extra_bkpts:
        all_bkpts.extend(extra_bkpts)
    doms = np.unique(np.concatenate([np.array(all_bkpts), xk]))
    doms = np.sort(doms)

    all_roots: list[float] = []

    for i in range(len(doms) - 1):
        ai, bi = doms[i], doms[i + 1]
        if ai >= bi - 1e-15 * (b - a):
            continue

        # Adaptively sample error on this sub-interval
        nn = 8   # start with 2^3 Chebyshev-2 pts
        max_nn = 64

        cU = np.array([], dtype=np.float64)
        while nn <= max_nn:
            pts_asc = np.array(chebpts_ab(nn + 1, ai, bi), dtype=np.float64)

            fvals_sub = np.asarray(
                f(jnp.array(pts_asc)), dtype=np.float64
            ).ravel()
            p_vals_sub = _eval_poly_bary(pts_asc, p_coeffs, a, b)
            err_vals = fvals_sub - p_vals_sub  # ascending order

            # vals2coeffs accepts ascending x-order (matching chebpts_ab).
            c_err = np.array(
                vals2coeffs(jnp.array(err_vals, dtype=jnp.float64)),
                dtype=np.float64,
            )

            # Chebyshev-U derivative coefficients:
            # d/dx [T_k(x)] = k * U_{k-1}(x)
            # so cU[k] = c_err[k+1] * (k+1) for k = 0, ..., n-2
            cU = c_err[1:] * np.arange(1, len(c_err), dtype=np.float64)

            if len(cU) == 0:
                break

            norm_cU = np.linalg.norm(cU)
            if norm_cU == 0.0:
                break

            # Check if coefficients have decayed
            if abs(cU[-1]) / (norm_cU + 1e-300) < 1e-3:
                break
            nn *= 2

        sub_roots = _roots_chebyshevU(cU, ai, bi)
        all_roots.extend(sub_roots.tolist())

    # Combine with domain endpoints and deduplicate
    all_roots_arr = np.unique(
        np.array([a, b] + all_roots, dtype=np.float64)
    )
    return np.sort(all_roots_arr)


def _roots_chebyshevU(
    cU: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Real roots of a Chebyshev-U series in [a, b].

    Finds the roots of ``sum_{k=0}^{n-1} cU[k] U_k(x)`` (``x`` on [-1,1])
    via the companion matrix eigenvalue problem, then maps back to [a, b].

    ``cU`` is in ascending-degree order (cU[0] = U_0 coefficient,
    cU[n-1] = U_{n-1} coefficient).  Internally the array is reversed to
    leading-coefficient-first form, matching the MATLAB ``rootsdiff``
    implementation.

    Parameters
    ----------
    cU : np.ndarray, shape (n,)
        Chebyshev-U coefficients in ascending-degree order.
    a, b : float
        Domain interval for the output roots.

    Returns
    -------
    roots : np.ndarray
        Real roots strictly inside (a, b), sorted ascending.
        Roots exactly at the interval endpoints are excluded to avoid
        spurious duplicates when sub-intervals share boundaries.

    Provenance
    ----------
    MATLAB source : rootsdiff (sub-function of minimax.m)
    Chebfun commit: 7574c77
    """
    if len(cU) == 0:
        return np.array([], dtype=np.float64)

    # Truncate trailing negligible coefficients (ascending order)
    norm_cU = np.linalg.norm(cU)
    if norm_cU == 0.0:
        return np.array([], dtype=np.float64)

    tol_sig = 1e-14
    sig_idx = np.where(np.abs(cU) / norm_cU > tol_sig)[0]
    if len(sig_idx) == 0:
        return np.array([], dtype=np.float64)

    # Keep only up to last significant coefficient, then flip to leading-first
    # (matching MATLAB: `cU = flipud(cU(1:len))` where len is 1-indexed)
    cU_flipped = cU[sig_idx[-1] :: -1]   # highest-degree coeff first

    n = len(cU_flipped)
    if n <= 1:
        return np.array([], dtype=np.float64)

    if n == 2:
        # Linear U series: cU_flip[0]*U_1(x) + cU_flip[1]*U_0(x) = 0
        # 2*cU_flip[0]*x + cU_flip[1] = 0  -> x = -cU_flip[1]/(2*cU_flip[0])
        ei = np.array([-cU_flipped[1] / (2.0 * cU_flipped[0])], dtype=np.float64)
    else:
        # Chebyshev-U companion matrix for polynomial of degree n-1.
        # The matrix is (n-1) x (n-1) with off-diagonals = 1/2 and a
        # modified first row.  This is identical to the Chebyshev-T companion
        # matrix (the three-term recurrence for U is the same structure).
        #
        # MATLAB rootsdiff:
        #   oh = ones(len-2,1)/2;
        #   C = diag(oh,1) + diag(oh,-1);
        #   cU = -cU(2:end)/cU(1)/2; cU(2) = cU(2)+.5;
        #   C(1,:) = cU.';
        # Here cU is already flipped (leading coeff first), so cU(1) = highest.
        length = n - 1
        oh = np.ones(length - 1, dtype=np.float64) * 0.5
        C = np.diag(oh, 1) + np.diag(oh, -1)

        # Normalised first row (MATLAB: cU = -cU(2:end)/cU(1)/2)
        cU_row = -cU_flipped[1:] / cU_flipped[0] / 2.0
        cU_row[1] = cU_row[1] + 0.5    # correct for the off-diagonal entry
        C[0, :] = cU_row

        try:
            ei = np.linalg.eigvals(C)
        except np.linalg.LinAlgError:
            return np.array([], dtype=np.float64)

        # Keep real roots strictly inside (-1, 1).
        # We exclude eigenvalues at or near +-1 (i.e., the sub-interval
        # endpoints) because those correspond to boundary extrema that are
        # already added separately as domain endpoints.  Including them would
        # create near-duplicate points that confuse the exchange step.
        ei_real = np.real(ei[np.abs(np.imag(ei)) < 1e-5])
        ei = ei_real[np.abs(ei_real) < 1.0 - 1e-10]

    if len(ei) == 0:
        return np.array([], dtype=np.float64)

    # Map from [-1, 1] to [a, b]
    roots = (a + b) / 2.0 + ei * (b - a) / 2.0
    roots = np.clip(roots, a, b)
    return np.sort(roots)


def _exchange(
    xk: np.ndarray,
    h: float,
    method: int,
    f: Callable,
    p_coeffs: np.ndarray,
    n_pts: int,
    a: float,
    b: float,
    extra_bkpts: list[float] | None = None,
) -> tuple[np.ndarray, float, int]:
    """One Remez exchange step.

    Finds all extrema of the error ``f - p`` on the domain, then selects
    ``n_pts`` consecutive extrema with alternating sign that include the
    maximum error.

    Parameters
    ----------
    xk : np.ndarray, shape (n_pts,)
        Current reference points.
    h : float
        Current levelled reference error.
    method : {1, 2}
        1 = one-point exchange (keep only maximum); 2 = full exchange.
    f : callable
        Target function.
    p_coeffs : np.ndarray, shape (m+1,)
        Chebyshev coefficients of the trial polynomial.
    n_pts : int
        Required size of the new reference set (``n + 2``).
    a, b : float
        Domain endpoints.
    extra_bkpts : list of float or None, optional
        Additional breakpoints for ``_find_extrema``.

    Returns
    -------
    xk_new : np.ndarray
        Updated reference points.
    norme : float
        Max-norm of the error ``f - p`` on the set of extrema.
    flag : int
        1 if ``len(xk_new) == n_pts``; 0 otherwise.

    Provenance
    ----------
    MATLAB source : exchange (sub-function of minimax.m)
    Chebfun commit: 7574c77
    """
    # ---- Find all extrema of f - p ----
    rr = _find_extrema(f, p_coeffs, xk, a, b, extra_bkpts)

    # Evaluate error at extrema
    f_rr = np.asarray(f(jnp.array(rr)), dtype=np.float64).ravel()
    p_rr = _eval_poly_bary(rr, p_coeffs, a, b)
    err_rr = f_rr - p_rr

    # ---- Select candidates ----
    if method == 1:
        # One-point exchange: maximum error only
        pos = np.array([int(np.argmax(np.abs(err_rr)))])
    else:
        # Full exchange: all extrema above |h|
        pos = np.where(np.abs(err_rr) >= np.abs(h))[0]
        if len(pos) == 0:
            pos = np.array([int(np.argmax(np.abs(err_rr)))])

    # ---- Merge candidates with current reference ----
    # Build the alternating sign vector for the current reference
    v = np.ones(n_pts, dtype=np.float64)
    v[1::2] = -1.0

    r_merge = np.concatenate([rr[pos], xk])
    er_merge = np.concatenate([err_rr[pos], v * h])

    # Sort by position
    sort_idx = np.argsort(r_merge, kind="stable")
    r_merge = r_merge[sort_idx]
    er_merge = er_merge[sort_idx]

    # Remove duplicates (keep the one with larger absolute error)
    unique_mask = np.concatenate([[True], np.diff(r_merge) != 0])
    r_merge = r_merge[unique_mask]
    er_merge = er_merge[unique_mask]

    # ---- Build alternating sequence ----
    # Keep adjacent points with alternating sign; prefer largest absolute value
    s = [r_merge[0]]
    es = [er_merge[0]]
    for i in range(1, len(r_merge)):
        if np.sign(er_merge[i]) == np.sign(es[-1]):
            # Same sign -- replace if larger
            if abs(er_merge[i]) > abs(es[-1]):
                s[-1] = r_merge[i]
                es[-1] = er_merge[i]
        else:
            # Alternating -- keep
            s.append(r_merge[i])
            es.append(er_merge[i])

    s = np.array(s, dtype=np.float64)
    es = np.array(es, dtype=np.float64)

    # ---- Select n_pts consecutive alternating points ----
    norme = float(np.max(np.abs(es)))
    index = int(np.argmax(np.abs(es)))
    d = max(index - n_pts + 1, 0)

    if n_pts <= len(s):
        xk_new = s[d : d + n_pts]
        flag = 1
    else:
        xk_new = s
        flag = 0

    return xk_new, norme, flag


# ===========================================================================
# Trigonometric minimax — trigremez
# ===========================================================================


@dataclass
class TrigremezResult:
    """Result of a trigonometric best-approximation computation.

    Attributes
    ----------
    coeffs : np.ndarray, shape (2*m+1,)
        Fourier (trigonometric) coefficients of the best trigonometric
        polynomial approximant of degree *m*.  Stored in ascending-frequency
        order: ``[c_{-m}, ..., c_0, ..., c_m]``.
    xk : np.ndarray
        Equioscillation reference points on the period.
    err : float
        Supremum norm of the error ``f - p`` on the domain.
    delta : float
        Normalised equioscillation deviation; near zero when converged.
    iter : int
        Number of Remez iterations performed.
    domain : tuple[float, float]
        Approximation period ``(a, b)``.
    """

    coeffs: np.ndarray
    xk: np.ndarray
    err: float
    delta: float
    iter: int
    domain: tuple[float, float]


def trigremez(
    f: Callable,
    m: int,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    tol: float | None = None,
    max_iter: int = 40,
) -> TrigremezResult:
    r"""Best trigonometric polynomial approximation via the trig Remez algorithm.

    Computes the best degree-*m* trigonometric polynomial approximant to the
    real-valued periodic function *f* on *domain* in the infinity norm.  The
    approximant equioscillates at ``2m + 2`` or more points.

    Parameters
    ----------
    f : callable
        Real-valued periodic function.  Must accept a 1-D ``np.ndarray`` and
        return a 1-D array-like of the same shape.  The function is assumed to
        be *2*(b-a)*-periodic and continuous on ``[a, b]``.
    m : int
        Degree of the best trigonometric polynomial (number of Fourier modes
        is ``2*m + 1``).
    domain : (float, float), optional
        One full period ``[a, b]``.  Default ``(-1.0, 1.0)``.
    tol : float or None, optional
        Relative equioscillation tolerance.  Default ``1e-13``.
    max_iter : int, optional
        Maximum Remez iterations.  Default 40.

    Returns
    -------
    result : TrigremezResult

    Notes
    -----
    The algorithm mirrors the polynomial Remez exchange loop in
    :func:`minimax`, but uses a trigonometric rather than Chebyshev basis.
    The initial reference is the ``2m+2`` equispaced points on ``[a, b)``.
    The trial polynomial is computed via barycentric trigonometric
    interpolation; extrema are found by sampling on a dense grid.

    Provenance
    ----------
    MATLAB source : @chebfun/trigremez.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        M. Javed, DPhil thesis, Oxford, 2017.

    See Also
    --------
    minimax

    Examples
    --------
    Best degree-5 trig polynomial for ``|sin(pi*x)|`` on ``[-1, 1]``:

    >>> import numpy as np
    >>> from chebfunjax.utils.minimax import trigremez
    >>> result = trigremez(lambda x: np.abs(np.sin(np.pi * x)), 5)
    >>> result.err < 0.04
    True
    """
    if tol is None:
        tol = 1e-13

    a, b = float(domain[0]), float(domain[1])
    period = b - a
    n_ref = 2 * m + 2  # number of equioscillation points for trig poly of degree m

    # ---- Dense grid for norms & extrema ----
    n_dense = max(8 * n_ref, 2048)
    x_dense = np.linspace(a, b, n_dense, endpoint=False)
    fvals_dense = np.asarray(f(x_dense), dtype=np.float64).ravel()
    normf = float(np.max(np.abs(fvals_dense)))
    if normf == 0.0:
        normf = float(np.finfo(np.float64).eps)

    # ---- Initial reference: equispaced ----
    xk = np.linspace(a, b, n_ref, endpoint=False)

    def _eval_trig_interp(xk_pts, fk, x_eval):
        """Barycentric trigonometric interpolation at xk_pts with values fk,
        evaluated at x_eval. Uses the standard trig barycentric formula."""
        N = len(xk_pts)
        # Trig barycentric weights: w_j = (-1)^j / 2
        w = np.ones(N) * 0.5
        w[1::2] = -0.5

        x_eval = np.asarray(x_eval, dtype=float)
        num = np.zeros_like(x_eval)
        den = np.zeros_like(x_eval)
        for j in range(N):
            # Kernel: cot(pi*(x - xk_j)/T) * w_j  (or the trig barycentric kernel)
            diff = np.pi * (x_eval - xk_pts[j]) / period
            # Avoid singularity at the nodes
            cot_val = np.cos(diff) / np.where(np.abs(np.sin(diff)) < 1e-14,
                                               1e-14 * np.sign(np.sin(diff) + 1e-300),
                                               np.sin(diff))
            num += w[j] * fk[j] * cot_val
            den += w[j] * cot_val

        # Handle exact nodes
        result = np.where(np.abs(den) < 1e-12 * normf, 0.0, num / den)
        for j in range(N):
            mask = np.abs(x_eval - xk_pts[j]) < 1e-14 * period
            result = np.where(mask, fk[j], result)
        return result

    err = normf
    h = 2.0 * err + 1.0
    iter_count = 0
    delta_min = np.inf
    best_xk = xk.copy()
    best_fk = None
    best_err = np.inf

    while (
        abs(abs(h) - abs(err)) > tol * abs(err)
        and iter_count < max_iter
    ):
        fk = np.asarray(f(xk), dtype=np.float64).ravel()

        # Levelled error h (alternating signs)
        sigma = np.ones(n_ref)
        sigma[1::2] = -1.0
        w_bary = np.ones(n_ref) * 0.5
        w_bary[1::2] = -0.5
        h = float(np.dot(w_bary, fk) / np.dot(w_bary, sigma))

        # Trig interpolant of fk - h*sigma at xk
        gk = fk - h * sigma

        # Evaluate error on dense grid
        p_dense = _eval_trig_interp(xk, gk, x_dense)
        err_dense = fvals_dense - p_dense
        err = float(np.max(np.abs(err_dense)))
        delta = err - abs(h)

        if delta < delta_min:
            delta_min = delta
            best_xk = xk.copy()
            best_fk = gk.copy()
            best_err = err

        # Find extrema of error on dense grid
        # (sign changes in derivative → local extrema)
        d_err = np.diff(err_dense)
        sign_changes = np.where(np.diff(np.sign(d_err)) != 0)[0] + 1
        extrema = np.sort(np.unique(
            np.concatenate([[0, n_dense - 1], sign_changes])
        ))
        extrema_x = x_dense[extrema]
        extrema_e = err_dense[extrema]

        # Select n_ref alternating extrema with highest error
        # Build alternating sequence
        s = [extrema_x[0]]
        es = [extrema_e[0]]
        for i in range(1, len(extrema_x)):
            if np.sign(extrema_e[i]) == np.sign(es[-1]):
                if abs(extrema_e[i]) > abs(es[-1]):
                    s[-1] = extrema_x[i]
                    es[-1] = extrema_e[i]
            else:
                s.append(extrema_x[i])
                es.append(extrema_e[i])

        s = np.array(s)
        es = np.array(es)

        if len(s) >= n_ref:
            idx_max = int(np.argmax(np.abs(es)))
            d_idx = max(idx_max - n_ref + 1, 0)
            d_idx = min(d_idx, max(0, len(s) - n_ref))
            xk = s[d_idx : d_idx + n_ref]
        else:
            # Not enough alternations; keep best from dense
            top_idx = np.argsort(np.abs(err_dense))[-n_ref:]
            xk = np.sort(x_dense[top_idx])

        iter_count += 1

    # ---- Extract Fourier coefficients of best approximant ----
    if best_fk is None:
        best_fk = np.asarray(f(best_xk), dtype=np.float64).ravel()

    # Re-interpolate on a uniform grid and FFT
    n_out = max(4 * m + 10, 64)
    x_out = np.linspace(a, b, n_out, endpoint=False)
    p_vals = _eval_trig_interp(best_xk, best_fk, x_out)
    C_fft = np.fft.fft(p_vals) / n_out
    # Centred Fourier coefficients for frequencies -m, ..., m
    coeffs = np.array(
        [C_fft[-k if k > 0 else 0] if k != 0 else C_fft[0] for k in range(-m, m + 1)],
        dtype=complex,
    )
    # Adjust for negative frequencies
    for i, k in enumerate(range(-m, m + 1)):
        if k < 0:
            coeffs[i] = C_fft[n_out + k] if n_out + k < n_out else 0.0
        elif k == 0:
            coeffs[i] = C_fft[0]
        else:
            coeffs[i] = C_fft[k]

    if np.allclose(np.imag(coeffs), 0, atol=1e-12):
        coeffs = np.real(coeffs)

    return TrigremezResult(
        coeffs=coeffs,
        xk=best_xk,
        err=float(best_err),
        delta=float(delta_min) / normf,
        iter=iter_count,
        domain=(a, b),
    )
