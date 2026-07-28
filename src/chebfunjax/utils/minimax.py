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
from scipy.linalg import eig

from chebfunjax.utils.interpolation import bary, bary_weights
from chebfunjax.utils.quadrature import chebpts_ab
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

__all__ = [
    "minimax",
    "trigremez",
    "MinimaxResult",
    "MinimaxRationalResult",
    "TrigremezResult",
]

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
    max_iter: int | None = None,
    init_xk: np.ndarray | None = None,
    breakpoints: Sequence[float] | None = None,
    rational: bool = False,
    denom: int | None = None,
) -> MinimaxResult | MinimaxRationalResult:
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
        **Deprecated and ignored.**  This was a non-MATLAB extension that
        forced kink locations to be sub-interval boundaries; because boundary
        extrema were then excluded from the reference, it produced
        non-equioscillating, sub-optimal approximants (e.g. ``|x-0.5|`` with
        ``n=1`` reported an error of 0.25 versus the exact 0.375).  MATLAB's
        ``minimax`` has no such option and the plain algorithm already resolves
        kinks correctly via adaptive sampling, so any value passed here now
        emits a :class:`DeprecationWarning` and has no effect.
    rational : bool, optional
        If ``True``, compute the best *rational* approximant of type
        ``(n, denom)`` instead of a polynomial.  The numerator degree is the
        positional argument ``n`` and the denominator degree is ``denom``
        (defaulting to ``n``, i.e. the diagonal type ``(n, n)``).  A
        :class:`MinimaxRationalResult` is returned in this case.  Uses the
        adaptive barycentric-Remez algorithm of Filip, Nakatsukasa,
        Beckermann & Trefethen (2018) with an AAA-Lawson initial reference.
    denom : int or None, optional
        Denominator degree for the rational case.  Ignored unless
        ``rational=True``.  Defaults to ``n`` (diagonal type).

    Returns
    -------
    result : MinimaxResult or MinimaxRationalResult
        A :class:`MinimaxRationalResult` when ``rational=True``, else a
        :class:`MinimaxResult` with fields:

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
        denom_deg = n if denom is None else int(denom)
        if n < 0 or denom_deg < 0:
            raise ValueError(
                f"minimax: rational degrees must be >= 0, got ({n}, {denom_deg})."
            )
        if denom_deg == 0:
            # Type (m, 0) is a polynomial; fall through to the polynomial path
            # and wrap the result as a (trivial-denominator) rational result.
            poly_kw = dict(
                domain=domain, tol=tol, init_xk=init_xk,
                breakpoints=breakpoints, rational=False,
            )
            if max_iter is not None:
                poly_kw["max_iter"] = max_iter
            poly = minimax(f, n, **poly_kw)
            a_, b_ = poly.domain

            def _poly_r(x, _c=np.array(poly.coeffs), _a=a_, _b=b_):
                return _eval_poly_bary(
                    np.asarray(x, dtype=np.float64), _c, _a, _b
                )

            return MinimaxRationalResult(
                r=_poly_r, err=float(poly.err), xk=poly.xk,
                delta=float(poly.delta), iter=int(poly.iter), m=n, n=0,
                support=jnp.array([]), wN=jnp.array([]), wD=jnp.array([]),
                poles=jnp.array([]), zeros=jnp.array([]),
                domain=(a_, b_), success=True,
            )
        return _minimax_rational(
            f, n, denom_deg, domain=domain, tol=tol, max_iter=max_iter,
            init_xk=init_xk,
        )

    a, b = float(domain[0]), float(domain[1])
    if a >= b:
        raise ValueError(
            f"minimax: domain must satisfy a < b, got ({a}, {b})."
        )
    if n < 0:
        raise ValueError(f"minimax: degree n must be >= 0, got {n}.")

    n_ref = n + 2  # size of reference set

    # ---- Default tolerance / iteration cap (matches MATLAB polynomial case) ----
    if tol is None:
        tol = 1e-14 * (n ** 2 + 10)
    if max_iter is None:
        max_iter = 30

    # ---- Extra breakpoints (deprecated no-op) ----
    # The ``breakpoints`` argument is a non-MATLAB extension.  MATLAB's minimax
    # has no such option: it detects kinks automatically through the chebfun
    # ``splitting`` representation.  Passing kink locations here forced them to
    # be sub-interval boundaries in ``_find_extrema``; because the colleague
    # matrix excludes roots at sub-interval endpoints, the extremum AT the kink
    # (where the error typically peaks) was dropped, so the reference stalled at
    # a non-equioscillating set and the reported error was far below the true
    # sup error (e.g. |x-0.5|, n=1 reported 0.25 vs the exact 0.375).  The plain
    # algorithm already resolves kinks correctly via adaptive sampling, so the
    # argument is now ignored.
    extra_bkpts: list[float] = []
    if breakpoints is not None:
        warnings.warn(
            "minimax: the 'breakpoints' argument is deprecated and now ignored. "
            "MATLAB minimax has no such option; the algorithm resolves kinks "
            "automatically. Passing breakpoints previously produced "
            "non-equioscillating, sub-optimal approximants.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        # Break the exact symmetry of the reference.  For an even f on a
        # bit-exactly symmetric reference the Remez exchange stalls: the error
        # of the interpolant is even with a central feature, so its extrema
        # collapse to only n+1 alternating points and the trial solve
        # h = (w . f)/(w . sigma) degenerates (w . f cancels to ~0).  quadfix's
        # sine chebpts (1c3fd5e) are bit-exactly antisymmetric, so |x| (and any
        # even f) hits this; the cosine form used before carried ~1e-16
        # asymmetry that incidentally avoided it.  A tiny *monotone* jitter of
        # the interior points restores the generic (asymmetric) reference the
        # algorithm expects; it washes out as the reference converges to the
        # true equioscillation set (endpoints untouched).
        if n_ref > 2:
            jitter = 1e3 * np.finfo(np.float64).eps * (b - a)
            xk[1:-1] = xk[1:-1] + jitter * np.arange(1, n_ref - 1, dtype=np.float64)

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
        xk_new, err_new, flag = _exchange(
            xk, h, 2, f, p_coeffs, n_ref, a, b, extra_bkpts
        )

        # If overshoot, fall back to one-point exchange
        if err_new / normf > 1e5:
            xk_new, err_new, flag = _exchange(
                xo, h, 1, f, p_coeffs, n_ref, a, b, extra_bkpts
            )

        if flag == 0:
            # The exchange could not assemble a full (n+2)-point alternating
            # reference.  Adopting the undersized set would crash the next
            # trial-polynomial solve (w vs sigma size mismatch), so stop and
            # return the best-so-far from earlier (converged) iterations
            # rather than the current, possibly degenerate, one.
            break

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
# Rational minimax — barycentric-Remez (Filip, Nakatsukasa, Beckermann,
# Trefethen 2018).  Public entry point is ``minimax(..., rational=True)``.
# ===========================================================================


@dataclass
class MinimaxRationalResult:
    """Result of a rational minimax (best type-(m, n) rational) approximation.

    Attributes
    ----------
    r : Callable
        Numerically stable barycentric evaluator of the best rational
        approximant ``p/q``.  Accepts and returns ``np.ndarray``.  This is the
        MATLAB ``rh`` (function-handle) output and is the recommended way to
        evaluate the approximant.
    err : float
        Supremum-norm error ``max|f - r|`` on the domain.
    xk : jnp.ndarray
        Final equioscillation reference (length ``m + n + 2``).
    delta : float
        Normalised equioscillation deviation ``(err - |h|)/normf`` (near zero
        when converged).
    iter : int
        Number of Remez iterations performed.
    m, n : int
        Numerator / denominator degrees actually used (after any
        even/odd symmetry reduction).
    support : jnp.ndarray
        Barycentric support points of the approximant.
    wN, wD : jnp.ndarray
        Barycentric numerator / denominator weights (``r = N/D`` with
        ``N(x)=sum wN_i/(x-support_i)``, ``D(x)=-sum wD_i/(x-support_i)``).
    poles, zeros : jnp.ndarray
        Poles and zeros of the approximant (complex).
    domain : tuple[float, float]
        Approximation domain ``(a, b)``.
    success : bool
        Whether a valid (sign-consistent) trial interpolant was produced.
    """

    r: Callable
    err: float
    xk: jnp.ndarray
    delta: float
    iter: int
    m: int
    n: int
    support: jnp.ndarray
    wN: jnp.ndarray
    wD: jnp.ndarray
    poles: jnp.ndarray
    zeros: jnp.ndarray
    domain: tuple[float, float]
    success: bool


def _chebpts1p(nn: int, a: float, b: float) -> np.ndarray:
    """``nn + 1`` Chebyshev-2 points on ``[a, b]`` in ascending order."""
    return np.array(chebpts_ab(nn + 1, a, b), dtype=np.float64)


def _orthspace(z: np.ndarray, dim: int, q: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal projection space for a nondiagonal type ``(m, n)``.

    Provenance
    ----------
    MATLAB source : orthspace (sub-function of minimax.m), Chebfun commit 7574c77
    """
    z = np.asarray(z, dtype=np.float64)
    L = len(z)
    if dim == 0:
        return np.eye(L)
    if q is None:
        q = np.ones(L)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 1)
    Q = q / np.linalg.norm(q)
    for _ in range(2, dim + 1):
        Qtmp = z[:, None] * Q[:, -1:]
        Qtmp = Qtmp - Q @ (Q.T @ Qtmp)
        Qtmp = Qtmp - Q @ (Q.T @ Qtmp)  # CGS2
        Qtmp = Qtmp / np.linalg.norm(Qtmp)
        Q = np.hstack([Q, Qtmp])
    Qfull, _ = np.linalg.qr(Q, mode="complete")
    return Qfull[:, dim:]


def _leja(x: np.ndarray, start_index: int, n_pts: int) -> np.ndarray:
    """Pick ``n_pts`` points from ``x`` in a Leja sequence from ``x[start]``.

    Provenance
    ----------
    MATLAB source : leja (sub-function of minimax.m), Chebfun commit 7574c77
    """
    x = np.asarray(x, dtype=np.float64)
    nx = len(x)
    xx = np.zeros(n_pts, dtype=np.float64)
    xx[0] = x[start_index]
    for j in range(1, n_pts):
        with np.errstate(divide="ignore"):  # log(0) at repeated points -> -inf
            p = np.array(
                [np.sum(np.log(np.abs(x[i] - xx[:j]))) for i in range(nx)],
                dtype=np.float64,
            )
        xx[j] = x[int(np.argmax(p))]
    return xx


def _roots_diff_rat(
    vals: np.ndarray,
    dom: tuple[float, float],
    err_handle: Callable,
) -> np.ndarray:
    """Roots of ``d/dx`` of the error via a Chebyshev-U colleague matrix.

    ``vals`` are the error samples at ``chebpts2`` on the sub-interval; the
    routine resamples adaptively until the Chebyshev series has decayed.

    Provenance
    ----------
    MATLAB source : rootsdiff (sub-function of minimax.m), Chebfun commit 7574c77
    """
    a, b = dom
    nn = len(vals) - 1
    tol = 1e-3
    c = np.array([1.0])
    while True:
        c = np.array(
            vals2coeffs(jnp.array(vals, dtype=jnp.float64)), dtype=np.float64
        )
        cU = c[1:] * np.arange(1, len(c), dtype=np.float64)
        nz = np.where(np.abs(cU) / (np.linalg.norm(cU) + 1e-300) > 1e-14)[0]
        if len(nz) == 0:
            return np.array([], dtype=np.float64)
        cU = cU[: nz[-1] + 1][::-1]  # significant part, highest degree first
        if len(cU) <= 1:
            return np.array([], dtype=np.float64)
        decayed = c[0] == 0.0 or abs(c[-1] / c[0]) <= tol
        if not decayed and nn <= 2 ** 6:
            nn *= 2
            xx = _chebpts1p(nn, a, b)
            vals = np.asarray(err_handle(xx), dtype=np.float64).ravel()
            continue
        break
    if len(cU) == 2:
        ei = np.array([-cU[1] / (2.0 * cU[0])], dtype=np.float64)
        ei = ei[np.abs(ei) <= 1.0 + 1e-7]
    else:
        length = len(cU)
        oh = np.ones(length - 2, dtype=np.float64) * 0.5
        C = np.diag(oh, 1) + np.diag(oh, -1)
        row = -cU[1:] / cU[0] / 2.0
        row[1] = row[1] + 0.5
        C[0, :] = row
        ei = np.linalg.eigvals(C)
        ei = np.real(ei[(np.abs(np.imag(ei)) < 1e-5) & (np.abs(ei) <= 1.0 + 1e-7)])
    if len(ei) == 0:
        return np.array([], dtype=np.float64)
    return (a + b) / 2.0 + ei * (b - a) / 2.0


def _find_extrema_rat(
    f: Callable,
    rh: Callable,
    xk: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Local extrema of ``f - rh`` using the current reference as breakpoints.

    Provenance
    ----------
    MATLAB source : findExtrema (sub-function of minimax.m), Chebfun commit 7574c77
    """
    def err_handle(x):
        return np.asarray(f(jnp.array(x)), dtype=np.float64).ravel() - np.asarray(
            rh(x), dtype=np.float64
        ).ravel()

    doms = np.unique(np.concatenate([np.array([a, b]), np.asarray(xk)]))
    doms = np.sort(doms)
    nn = 2 ** 3
    mid = (doms[:-1] + doms[1:]) / 2.0
    rad = (doms[1:] - doms[:-1]) / 2.0
    cosv = np.cos(np.pi * np.arange(nn, -1, -1) / nn)  # ascending on [-1, 1]
    xx = np.outer(np.ones(nn + 1), mid) + np.outer(cosv, rad)  # (nn+1, nseg)
    valerr = np.asarray(f(jnp.array(xx)), dtype=np.float64) - np.asarray(
        rh(xx), dtype=np.float64
    )
    rts: list[float] = []
    for k in range(len(doms) - 1):
        rnow = _roots_diff_rat(valerr[:, k], (doms[k], doms[k + 1]), err_handle)
        rts.extend(np.atleast_1d(rnow).tolist())
    out = np.unique(np.concatenate([np.array([a, b]), np.array(rts, dtype=np.float64)]))
    return np.sort(out)


def _exchange_rat(
    xk: np.ndarray,
    h: float,
    method: int,
    f: Callable,
    rh: Callable,
    n_pts: int,
    a: float,
    b: float,
) -> tuple[np.ndarray, float, int]:
    """One Remez exchange step for the rational case.

    Provenance
    ----------
    MATLAB source : exchange (sub-function of minimax.m), Chebfun commit 7574c77
    """
    def err_handle(x):
        return np.asarray(f(jnp.array(x)), dtype=np.float64).ravel() - np.asarray(
            rh(x), dtype=np.float64
        ).ravel()

    rr = _find_extrema_rat(f, rh, xk, a, b)
    er_rr = err_handle(rr)
    if method == 1:
        pos = np.array([int(np.argmax(np.abs(er_rr)))])
    else:
        pos = np.where(np.abs(er_rr) >= np.abs(h))[0]
        if len(pos) == 0:
            pos = np.array([int(np.argmax(np.abs(er_rr)))])

    v = np.ones(n_pts, dtype=np.float64)
    v[1::2] = -1.0
    r = np.concatenate([rr[pos], np.asarray(xk, dtype=np.float64)])
    er = np.concatenate([er_rr[pos], v * h])
    idx = np.argsort(r, kind="stable")
    r = r[idx]
    er = er[idx]
    keep = np.concatenate([[True], np.diff(r) != 0])
    r = r[keep]
    er = er[keep]

    s = [r[0]]
    es = [er[0]]
    for i in range(1, len(r)):
        if np.sign(er[i]) == np.sign(es[-1]):
            if abs(er[i]) > abs(es[-1]):
                s[-1] = r[i]
                es[-1] = er[i]
        else:
            s.append(r[i])
            es.append(er[i])
    s = np.array(s, dtype=np.float64)
    es = np.array(es, dtype=np.float64)

    norme = float(np.max(np.abs(es)))
    index = int(np.argmax(np.abs(es)))
    d = max(index - n_pts + 1, 0)
    if n_pts <= len(s):
        return s[d : d + n_pts], norme, 1
    return s, norme, 0


def _make_reval(
    xsupport: np.ndarray, wN: np.ndarray, wD: np.ndarray
) -> Callable:
    """Barycentric evaluator ``r = N/D`` with support ``xsupport``.

    Provenance
    ----------
    MATLAB source : reval (sub-function of minimax.m), Chebfun commit 7574c77
    """
    xs = np.asarray(xsupport, dtype=np.float64)
    wN = np.asarray(wN, dtype=np.float64).ravel()
    wD = np.asarray(wD, dtype=np.float64).ravel()

    def rh(zz):
        zz = np.asarray(zz, dtype=np.float64)
        shape = zz.shape
        zv = zz.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            CC = 1.0 / (zv[:, None] - xs[None, :])
            N = CC @ wN
            D = -(CC @ wD)  # note the sign flip in D
            r = N / D
        bad = np.where(~np.isfinite(r))[0]
        for j in bad:
            match = np.where(zv[j] == xs)[0]
            if len(match) > 0:
                r[j] = -wN[match[0]] / wD[match[0]]
        return r.reshape(shape)

    return rh


def _sorted_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Householder QR with descending row-norm sorting (as in MATLAB)."""
    nrm = np.linalg.norm(A, axis=1)
    ix = np.argsort(-nrm)
    Q, R = np.linalg.qr(A[ix, :])
    ixx = np.empty_like(ix)
    ixx[ix] = np.arange(len(ix))
    return Q[ixx, :], R


def _compute_trial_rational(
    f: Callable,
    xk: np.ndarray,
    m: int,
    n: int,
    hpre: float,
    a: float,
    b: float,
) -> tuple:
    """Barycentric trial rational via the symmetric eigenproblem.

    Solves ``(F + h*sigma) N = D`` in barycentric form; the eigenvector of a
    symmetric matrix that yields a denominator with no sign changes gives the
    trial approximant and the levelled reference error ``h``.

    Provenance
    ----------
    MATLAB source : computeTrialFunctionRational (sub-function of minimax.m)
    Chebfun commit: 7574c77
    """
    xk = np.asarray(xk, dtype=np.float64)
    L = len(xk)
    # Shape invariant: the barycentric trial solver needs EXACTLY m+n+2
    # reference points, so that len(xsupport) == m+1 and every downstream QR,
    # eigenproblem and matmul (Q.T@..@Q, C@bet, vstack slicing) is conformable.
    # A shorter reference -- which can arise on a platform where the exchange
    # step's alternating-extrema count rounds differently -- would otherwise
    # crash with "matmul: Input operand ... core dimension" instead of failing
    # gracefully.  Report failure and let the caller keep its best iterate.
    if L != m + n + 2:
        return None, 1e-19, False, None, None, None
    fk = np.asarray(f(jnp.array(xk)), dtype=np.float64).ravel()

    xsupport = xk[1::2].copy()
    xsuppind = np.arange(1, L, 2)
    xother = xk[0::2].copy()
    xotherind = np.arange(0, L, 2)

    Qmn = None
    Qmnall = None
    if m != n:
        xadd = _leja(xother, 0, len(xother))
        need = max(m, n) + 1 - len(xsupport)
        xsupport = np.concatenate([xsupport, xadd[:need]])
        supp_set = set(xsupport.tolist())
        xother_l, xotherind_l, xsuppind_l = [], [], []
        for ii in range(L):
            if xk[ii] not in supp_set:
                xother_l.append(xk[ii])
                xotherind_l.append(ii)
            else:
                xsuppind_l.append(ii)
        xother = np.array(xother_l, dtype=np.float64)
        xotherind = np.array(xotherind_l, dtype=int)
        xsuppind = np.array(xsuppind_l, dtype=int)

    xsupport = np.sort(xsupport)
    if m != n:
        Qmn = _orthspace(xsupport, abs(m - n), np.ones(len(xsupport)))
        Qmnall, _ = np.linalg.qr(Qmn, mode="complete")

    with np.errstate(divide="ignore"):
        C = 1.0 / (xk[:, None] - xsupport[None, :])

    # Build Cstar = sqrt(|Delta|)*C in the log domain (avoids over/underflow).
    Xkdiff = np.abs(xk[:, None] - xk[None, :])
    np.fill_diagonal(Xkdiff, 1.0)
    SX = np.sum(np.log(Xkdiff), axis=0)

    Xtdiff = np.abs(xother[:, None] - xsupport[None, :])
    ST = np.sum(np.log(Xtdiff.T), axis=0)
    VV = np.exp(ST - 0.5 * SX[xotherind])[:, None] * np.ones((1, len(xsupport)))
    Div = xother[:, None] - xsupport[None, :]
    C1 = VV / Div

    Xtdiff2 = np.abs(xsupport[:, None] - xsupport[None, :])
    np.fill_diagonal(Xtdiff2, 1.0)
    ST2 = np.sum(np.log(Xtdiff2.T), axis=0)
    C2 = np.diag(np.exp(ST2 - 0.5 * SX[xsuppind]))

    Cstar = np.zeros((L, len(xsupport)), dtype=np.float64)
    Cstar[xsuppind, :] = C2
    Cstar[xotherind, :] = C1

    if m == n:
        Q, R = _sorted_qr(Cstar)
    elif m > n:
        Q, R = _sorted_qr(Cstar @ Qmn)
        Qall, Rall = _sorted_qr(Cstar @ Qmnall)
    else:
        Q, R = _sorted_qr(Cstar)
        Qpart, Rpart = _sorted_qr(Cstar @ Qmn)

    S = (-1.0) ** np.arange(L)
    QSQ = Q.T @ (S[:, None] * (fk[:, None] * Q))
    QSQ = (QSQ + QSQ.T) / 2.0
    d_eig, VR = np.linalg.eigh(-QSQ)
    beta = np.linalg.solve(R, VR)

    if m == n:
        alpha = np.linalg.solve(R, -(Q.T @ (fk[:, None] * Q)) @ VR)
    elif m > n:
        alpha = Qmnall @ np.linalg.solve(Rall, (Qall.T @ (-fk[:, None] * Q)) @ VR)
    else:
        alpha = np.linalg.solve(Rpart, (Qpart.T @ (-fk[:, None] * Q)) @ VR)
    vt = np.vstack([alpha, beta])

    bet = vt[m + 1 :, :] if m <= n else Qmn @ vt[m + 1 :, :]
    Dvals = C @ bet

    def node(z):
        return np.prod(z - xsupport)

    nodevec = np.array([node(x) for x in xother], dtype=np.float64)
    checksign = np.zeros((L, VR.shape[1]), dtype=np.float64)
    mn = max(m, n)
    signfac = ((-1.0) ** np.arange(mn, mn + len(xsupport)))[:, None]
    checksign[: len(xsupport), :] = signfac * bet
    checksign[len(xsupport) :, :] = np.sign(nodevec[:, None] * Dvals[xotherind, :])
    pos = np.where(
        (np.abs(np.sum(np.sign(checksign), axis=0)) == m + n + 2)
        & (np.sum(np.abs(Dvals), axis=0) > 1e-7)
    )[0]

    if len(pos) == 0:
        return _compute_trial_rational_fallback(f, xk, m, n, hpre, a, b)
    if len(pos) > 1:
        pos = np.array([pos[int(np.argmin(np.abs(np.abs(hpre) - np.abs(d_eig[pos]))))]])
    p = pos[0]

    h = float(-d_eig[p])
    wD = vt[m + 1 :, p] if m <= n else (Qmn @ vt[m + 1 :, :])[:, p]
    wN = vt[: m + 1, p] if m >= n else (Qmn @ vt[: m + 1, :])[:, p]
    return _make_reval(xsupport, wN, wD), h, True, xsupport, wN, wD


def _compute_trial_rational_fallback(
    f: Callable,
    xk: np.ndarray,
    m: int,
    n: int,
    hpre: float,
    a: float,
    b: float,
) -> tuple:
    """Fallback trial rational using midpoint support points.

    Provenance
    ----------
    MATLAB source : computeTrialFunctionRational fallback branch (minimax.m)
    Chebfun commit: 7574c77
    """
    xk = np.asarray(xk, dtype=np.float64)
    L = len(xk)
    fk = np.asarray(f(jnp.array(xk)), dtype=np.float64).ravel()

    xsupport = (xk[0:-1:2] + xk[1::2]) / 2.0
    xadd = (xk[1:-1:2] + xk[2::2]) / 2.0
    if a not in xk:
        xadd = np.concatenate([[(a + xk[0]) / 2.0], xadd])
    if b not in xk:
        xadd = np.concatenate([[(b + xk[-1]) / 2.0], xadd])
    num = abs(max(m, n) + 1 - len(xsupport))
    xadd_leja = _leja(xadd, 0, num) if (num > 0 and len(xadd) > 0) else np.array([])
    if m != n:
        need = max(m, n) + 1 - len(xsupport)
        xsupport = np.concatenate([xsupport, xadd_leja[:need]])
    xsupport = np.sort(xsupport)

    Qmn = Qmnall = None
    if m != n:
        Qmn = _orthspace(xsupport, abs(m - n), np.ones(len(xsupport)))
        Qmnall, _ = np.linalg.qr(Qmn, mode="complete")

    with np.errstate(divide="ignore"):
        C = 1.0 / (xk[:, None] - xsupport[None, :])
    Delta = np.zeros(L, dtype=np.float64)
    for ii in range(L):
        others = np.delete(xk, ii)
        Delta[ii] = -np.exp(
            2 * np.sum(np.log(np.abs(np.prod(xk[ii] - xsupport))))
            - np.sum(np.log(np.abs(xk[ii] - others)))
        )
    sq = np.sqrt(np.abs(Delta))[:, None]

    if m == n:
        Q, R = np.linalg.qr(sq * C)
    elif m > n:
        Q, R = np.linalg.qr((sq * C) @ Qmn)
        Qall, Rall = np.linalg.qr((sq * C) @ Qmnall)
    else:
        Q, R = np.linalg.qr(sq * C)
        Qpart, Rpart = np.linalg.qr((sq * C) @ Qmn)

    S = (-1.0) ** np.arange(L)
    QSQ = Q.T @ (S[:, None] * (fk[:, None] * Q))
    QSQ = (QSQ + QSQ.T) / 2.0
    d_eig, VR = np.linalg.eigh(-QSQ)
    beta = np.linalg.solve(R, VR)
    if m == n:
        alpha = np.linalg.solve(R, -(Q.T @ (fk[:, None] * Q)) @ VR)
    elif m > n:
        alpha = Qmnall @ np.linalg.solve(Rall, (Qall.T @ (-fk[:, None] * Q)) @ VR)
    else:
        alpha = np.linalg.solve(Rpart, (Qpart.T @ (-fk[:, None] * Q)) @ VR)
    vt = np.vstack([alpha, beta])

    if m <= n:
        Dvals = C[:, : n + 1] @ vt[m + 1 :, :]
    else:
        Dvals = C @ (Qmn @ vt[m + 1 :, :])

    def node(z):
        return np.prod(z - xsupport)

    nodevec = np.array([node(x) for x in xk], dtype=np.float64)
    pos = np.where(
        (np.abs(np.sum(np.sign(nodevec[:, None] * Dvals), axis=0)) == m + n + 2)
        & (np.sum(np.abs(Dvals), axis=0) > 1e-4)
    )[0]
    if len(pos) == 0:
        return None, 1e-19, False, None, None, None
    if len(pos) > 1:
        pos = np.array([pos[int(np.argmin(np.abs(np.abs(hpre) - np.abs(d_eig[pos]))))]])
    p = pos[0]
    h = float(-d_eig[p])
    wD = vt[m + 1 :, p] if m <= n else (Qmn @ vt[m + 1 :, :])[:, p]
    wN = vt[: m + 1, p] if m >= n else (Qmn @ vt[: m + 1, :])[:, p]
    return _make_reval(xsupport, wN, wD), h, True, xsupport, wN, wD


def _aaamn_lawson(
    f: Callable,
    Z: np.ndarray,
    m: int,
    n: int,
    lawson_iter: int | None = None,
    tol_lawson: float = 1e-5,
    tol: float = 1e-15,
) -> tuple[Callable, np.ndarray]:
    """AAA + Lawson near-best rational approximation (for reference init).

    Provenance
    ----------
    MATLAB source : aaamn_lawson (sub-function of minimax.m), Chebfun commit 7574c77
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    F = np.asarray(f(jnp.array(Z)), dtype=np.float64).ravel()
    M = len(Z)
    mmax, nmax = m + 1, n + 1
    if lawson_iter is None:
        lawson_iter = max(5, min(20, mmax, nmax))

    J = list(range(M))
    z: list[float] = []
    fvals: list[float] = []
    C = np.zeros((M, 0), dtype=np.float64)
    R = np.full(M, np.mean(F))
    w = None
    Q = None
    mn = 1
    for mn in range(1, max(mmax, nmax) + 1):
        j = int(np.argmax(np.abs(F - R)))
        z.append(Z[j])
        fvals.append(F[j])
        if j in J:
            J.remove(j)
        with np.errstate(divide="ignore"):
            col = 1.0 / (Z - Z[j])
        col[j] = 0.0
        C = np.hstack([C, col[:, None]])
        fa = np.array(fvals)
        A = F[:, None] * C - C * fa[None, :]
        zc = np.array(z)
        if mn > min(nmax, mmax):
            q = fa if mmax < nmax else np.ones(len(z))
            Q = _orthspace(zc, mn - min(mmax, nmax), q)
            _, _, Vh = np.linalg.svd(A[J, :] @ Q, full_matrices=False)
            w = Q @ Vh[-1, :]
        else:
            _, _, Vh = np.linalg.svd(A[J, :], full_matrices=False)
            w = Vh[mn - 1, :]
        N = C @ (w * fa)
        D = C @ w
        R = F.copy()
        R[J] = N[J] / D[J]
        if np.max(np.abs(F - R)) < tol * np.max(np.abs(F)):
            break

    zc = np.array(z)
    fa = np.array(fvals)
    wf = w * fa
    wei = np.ones(len(J))
    nrmbest = np.inf
    SFC = F[:, None] * C
    if mn > min(nmax, mmax):
        if mn > nmax:
            Amat = np.hstack([SFC @ Q, -C])
        else:
            Q = _orthspace(zc, mn - min(mmax, nmax), np.ones(len(z)))
            Amat = np.hstack([SFC, -C @ Q])
    else:
        Amat = np.hstack([SFC, -C])
    rate = 1.0
    nrmincreased = 0
    best = dict(z=zc, w=w, wf=wf, f=fa)
    for _ in range(lawson_iter):
        weiold = wei
        wei = wei * np.power(np.abs(F[J] - R[J]), rate)
        wei = wei / np.sum(wei)
        if np.linalg.norm(weiold - wei) / np.linalg.norm(wei) < tol_lawson:
            break
        Dw = np.sqrt(wei)[:, None]
        _, _, Vh = np.linalg.svd(Dw * Amat[J, :], full_matrices=False)
        V = Vh.T
        if mn > min(nmax, mmax):
            if mn > nmax:
                w = Q @ V[:nmax, -1]
                wf = V[nmax:, -1]
            else:
                w = V[:nmax, -1]
                wf = Q @ V[nmax:, -1]
        else:
            w = V[:mn, -1]
            wf = V[mn : 2 * mn, -1]
        fa = wf / w
        N = C @ wf
        D = C @ w
        R = F.copy()
        R[J] = N[J] / D[J]
        err = np.max(np.abs(F - R))
        if err < nrmbest:
            nrmbest = err
            best = dict(z=zc, w=w.copy(), wf=wf.copy(), f=fa.copy())
        else:
            nrmincreased += 1
        if nrmincreased >= 3:
            rate = max(rate / 2, 0.01)
            nrmincreased = 0

    z_b, w_b, wf_b, f_b = best["z"], best["w"], best["wf"], best["f"]

    def rr(zz):
        zz = np.asarray(zz, dtype=np.float64)
        shape = zz.shape
        zv = zz.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            CC = 1.0 / (zv[:, None] - z_b[None, :])
            r = (CC @ wf_b) / (CC @ w_b)
        bad = np.where(~np.isfinite(r))[0]
        for jj in bad:
            match = np.where(zv[jj] == z_b)[0]
            if len(match) > 0:
                r[jj] = f_b[match[0]]
        return r.reshape(shape)

    return rr, z_b


def _find_reference_rat(
    f: Callable, rh: Callable, m: int, n: int, z: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Turn AAA-Lawson support points into an ``m+n+2`` extremal reference.

    Provenance
    ----------
    MATLAB source : findReference (sub-function of minimax.m), Chebfun commit 7574c77
    """
    xk = _find_extrema_rat(f, rh, np.sort(z), a, b)
    target = m + n + 2
    if len(xk) > target:
        ix = np.argsort(-np.diff(xk))
        xk = np.sort(np.concatenate([[xk[0]], xk[1 + ix[: m + n + 1]]]))
    elif len(xk) < target:
        add = target - len(xk)
        ix = np.argsort(-np.diff(xk))
        ix = ix[:add]
        xk = np.sort(np.concatenate([xk, (xk[ix] + xk[ix + 1]) / 2.0]))
    return xk


def _aaa_lawson_init(f: Callable, m: int, n: int, a: float, b: float) -> np.ndarray:
    """AAA-Lawson-based initial reference (the robust default init).

    Provenance
    ----------
    MATLAB source : AAALawsonInit (sub-function of minimax.m), Chebfun commit 7574c77
    """
    NN = int(max(10 * max(m, n), round(1e5 / max(m, n))))
    Z = np.linspace(a, b, NN)
    rh, z = _aaamn_lawson(f, Z, m, n)
    xk = _find_reference_rat(f, rh, m, n, z, a, b)
    for _ in range(2):
        num = max(2, int(round(NN / len(xk))))
        parts = [np.linspace(xk[ii], xk[ii + 1], num) for ii in range(len(xk) - 1)]
        Z = np.unique(np.concatenate(parts))
        rh, z = _aaamn_lawson(f, Z, m, n)
        xk = _find_reference_rat(f, rh, m, n, z, a, b)
    return xk


def _adjust_degrees_for_symmetries(
    f: Callable, m: int, n: int, a: float, b: float
) -> tuple[int, int, int]:
    """Reduce type degrees when ``f`` is even or odd.

    Provenance
    ----------
    MATLAB source : adjustDegreesForSymmetries (sub-function of minimax.m)
    Chebfun commit: 7574c77
    """
    x = _chebpts1p(128, a, b)
    fv = np.asarray(f(jnp.array(x)), dtype=np.float64).ravel()
    c = np.array(vals2coeffs(jnp.array(fv, dtype=jnp.float64)), dtype=np.float64)
    c = c.copy()
    c[0] = 2 * c[0]
    vscale = np.max(np.abs(fv))
    if vscale == 0:
        vscale = 1.0
    eps = float(np.finfo(np.float64).eps)
    sym = 0
    if np.max(np.abs(c[1::2])) / vscale < eps:  # even
        sym = 1
        if m % 2 == 1:
            m = max(m - 1, 0)
        if n % 2 == 1:
            n = max(n - 1, 0)
    elif np.max(np.abs(c[0::2])) / vscale < eps:  # odd
        sym = 2
        if m % 2 == 0:
            m = m - 1
        if n % 2 == 1:
            n = n - 1
    return m, n, sym


def _pzeros(
    zj: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    m: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zeros and poles of a barycentric approximant via generalized eig.

    Provenance
    ----------
    MATLAB source : pzeros (sub-function of minimax.m), Chebfun commit 7574c77
    """
    def _ge(coeffs):
        L = len(coeffs)
        B = np.eye(L + 1)
        B[0, 0] = 0.0
        E = np.zeros((L + 1, L + 1), dtype=np.float64)
        E[0, 1:] = coeffs
        E[1:, 0] = 1.0
        E[1:, 1:] = np.diag(zj)
        ev = eig(E, B, right=False)
        return ev[np.isfinite(ev)]

    zer = np.array([]) if (m == 0 and len(alpha) == 0) else _ge(alpha)
    pol = np.array([]) if (n == 0 or len(beta) == 0) else _ge(beta)
    return zer, pol


def _minimax_rational(
    f: Callable,
    m: int,
    n: int,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    tol: float | None = None,
    max_iter: int | None = None,
    init_xk: np.ndarray | None = None,
) -> MinimaxRationalResult:
    """Best type-(m, n) rational approximation via barycentric-Remez.

    See :func:`minimax` for the public interface and parameter description.

    Provenance
    ----------
    MATLAB source : minimaxKernel (rational branch of minimax.m)
    Chebfun commit: 7574c77
    """
    a, b = float(domain[0]), float(domain[1])
    m, n, _sym = _adjust_degrees_for_symmetries(f, m, n, a, b)

    if m == -1:
        # Odd f with numerator degree 0: best approximant is the zero function.
        xd = _chebpts1p(512, a, b)
        err = float(np.max(np.abs(np.asarray(f(jnp.array(xd)), dtype=np.float64))))
        return MinimaxRationalResult(
            r=lambda x: np.zeros_like(np.asarray(x, dtype=np.float64)),
            err=err, xk=jnp.array([a, b], dtype=jnp.float64), delta=0.0, iter=0,
            m=m, n=n, support=jnp.array([]), wN=jnp.array([]), wD=jnp.array([]),
            poles=jnp.array([]), zeros=jnp.array([]), domain=(a, b), success=True,
        )

    N = m + n
    if tol is None:
        tol = 1e-4
    if max_iter is None:
        max_iter = 10 + round(max(m, n) / 2)

    xd = _chebpts1p(max(512, 8 * (N + 2)), a, b)
    scale = float(np.max(np.abs(np.asarray(f(jnp.array(xd)), dtype=np.float64))))
    if scale == 0:
        return MinimaxRationalResult(
            r=lambda x: np.zeros_like(np.asarray(x, dtype=np.float64)),
            err=0.0, xk=jnp.array([a, b], dtype=jnp.float64), delta=0.0, iter=0,
            m=m, n=n, support=jnp.array([]), wN=jnp.array([]), wD=jnp.array([]),
            poles=jnp.array([]), zeros=jnp.array([]), domain=(a, b), success=True,
        )

    # Scale-normalise: minimax is homogeneous, minimax(c f) = c minimax(f).
    forig = f

    def fs(x, _f=forig, _s=scale):
        # Shape-preserving: 2-D inputs are used in _find_extrema_rat.
        return np.asarray(_f(jnp.array(x)), dtype=np.float64) / _s

    normf = 1.0
    if init_xk is not None:
        xk = np.asarray(init_xk, dtype=np.float64).ravel()
    else:
        xk = _aaa_lawson_init(fs, m, n, a, b)

    xk = np.asarray(xk, dtype=np.float64)
    xo = xk.copy()
    iter_count = 0
    deltamin = np.inf
    diffx = 1.0
    err = normf
    h = 2 * err + 1
    interp_success = True
    best = None
    support = wN = wD = None

    while (
        abs(abs(h) - abs(err)) / abs(err) > tol
        and iter_count < max_iter
        and diffx > 0
        and interp_success
    ):
        hpre = h
        if abs(abs(h) - abs(err)) / normf < 1e-14:
            break
        err = np.inf
        rh, h, interp_success, support, wN, wD = _compute_trial_rational(
            fs, xk, m, n, hpre, a, b
        )
        if not interp_success:
            break
        if h == 0:
            h = 1e-19
        xk_new, err, flag = _exchange_rat(xk, h, 2, fs, rh, N + 2, a, b)
        # Record the best iterate using the CURRENT (valid, length-N+2)
        # reference the trial was computed on -- support/wN/wD are consistent
        # with this xk.
        delta = err - abs(h)
        if delta < deltamin:
            deltamin = delta
            best = dict(support=support, wN=wN, wD=wD, err=err, xk=xk.copy(), h=h)
        # Shape-consistency guard: _compute_trial_rational requires exactly
        # N+2 = m+n+2 reference points.  ``flag == 0`` (or a short set) means the
        # exchange could not assemble a full alternating reference -- which can
        # happen on a platform where the extrema/sign count rounds differently
        # than it does here.  Stop with the best full-length iterate instead of
        # feeding a short reference back in (that crashed downstream on a matmul
        # core-dimension mismatch).
        if flag == 0 or len(xk_new) != N + 2:
            break
        xk = xk_new
        diffx = float(np.max(np.abs(xo - xk))) if len(xo) == len(xk) else 1.0
        xo = xk.copy()
        iter_count += 1

    if best is None:
        # No successful iteration; report failure with a best-effort evaluator.
        rh0 = _make_reval(support, wN, wD) if support is not None else (
            lambda x: np.zeros_like(np.asarray(x, dtype=np.float64))
        )
        return MinimaxRationalResult(
            r=lambda x, _r=rh0, _s=scale: _s * np.asarray(_r(x), dtype=np.float64),
            err=float(err) * scale if np.isfinite(err) else np.inf,
            xk=jnp.array(xk, dtype=jnp.float64),
            delta=float(deltamin) / normf, iter=iter_count, m=m, n=n,
            support=jnp.array(support if support is not None else []),
            wN=jnp.array(wN if wN is not None else []),
            wD=jnp.array(wD if wD is not None else []),
            poles=jnp.array([]), zeros=jnp.array([]), domain=(a, b),
            success=False,
        )

    support = np.asarray(best["support"], dtype=np.float64)
    wN = np.asarray(best["wN"], dtype=np.float64)
    wD = np.asarray(best["wD"], dtype=np.float64)
    # Undo the scale normalisation on the numerator weights only (D unchanged).
    rh_scaled = _make_reval(support, wN * scale, wD)
    zer, pol = _pzeros(support, wN, wD, m, n)

    return MinimaxRationalResult(
        r=rh_scaled,
        err=float(best["err"]) * scale,
        xk=jnp.array(best["xk"], dtype=jnp.float64),
        delta=float(deltamin) / normf,
        iter=iter_count,
        m=m,
        n=n,
        support=jnp.array(support, dtype=jnp.float64),
        wN=jnp.array(wN * scale, dtype=jnp.float64),
        wD=jnp.array(wD, dtype=jnp.float64),
        poles=jnp.array(pol),
        zeros=jnp.array(zer),
        domain=(a, b),
        success=True,
    )


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
