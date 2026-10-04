"""Miscellaneous utility functions for Chebyshev approximation.

Translated from MATLAB Chebfun (commit 7574c77): standardChop.m, gridsample.m,
abstractQR.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Machine epsilon for float64 — used as the default tolerance in standard_chop.
# In MATLAB Chebfun this comes from chebfunpref().chebfuneps (= eps ≈ 2.2e-16).
# ---------------------------------------------------------------------------
_EPS = jnp.finfo(jnp.float64).eps


def standard_chop(coeffs: jnp.ndarray, tol: float | None = None) -> int:
    """Chopping rule for truncating a Chebyshev coefficient series.

    Determines an appropriate cutoff point for a sequence of Chebyshev (or
    Fourier) coefficients.  The algorithm scans for a *plateau* in the
    monotonically non-increasing *envelope* of the absolute values and then
    fine-tunes the cutoff so that the retained coefficients carry all the
    information above roughly ``tol`` relative accuracy.

    This is THE core convergence check used throughout Chebfun: adaptive
    construction (``chebtech``, ``trigtech``), BVP solvers, simplification,
    and Chebfun2.

    Parameters
    ----------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev (or Fourier) coefficients, ordered ``c_0, c_1, ..., c_{n-1}``.
    tol : float or None, optional
        Target relative accuracy in ``(0, 1)``.  Default is machine epsilon
        (``~2.2e-16``).

    Returns
    -------
    cutoff : int
        A positive integer (1-based length).

        * If ``cutoff == len(coeffs)`` the series is "not happy" — no
          satisfactory chopping point was found.
        * If ``cutoff < len(coeffs)`` the series is "happy" and only
          ``coeffs[:cutoff]`` should be retained.

    Notes
    -----
    The algorithm has three steps:

    1. Build a monotonically non-increasing *envelope* of ``|coeffs|``,
       normalised to start at 1.
    2. Scan the envelope for a *plateau*: a stretch
       ``envelope[j], ..., envelope[j2]`` with ``j2 = round(1.25*j + 5)``
       that is "flat enough" relative to its height and ``tol``.
       "Flat enough" means ``envelope[j2]/envelope[j] > r`` where
       ``r = 3*(1 - log(envelope[j]) / log(tol))``, ranging from ``r = 0``
       (when ``envelope[j] ~ tol``) to ``r = 1`` (when ``envelope[j] ~ tol^{2/3}``).
    3. Fine-tune the cutoff by finding the minimum of
       ``log10(envelope) + linear_bias`` within the plateau region, where the
       linear bias steers the cutoff toward shorter series.

    ``COEFFS`` will never be chopped unless it has length >= 17 and falls
    below ``tol^{1/3}``.  It will always be chopped if there is a long enough
    segment below ``tol``.  The final ``coeffs[cutoff-1]`` will never be
    smaller than ``tol^{7/6}`` (all relative to ``max(|coeffs|)``).

    These parameters are the result of extensive experimentation; they are
    **not** derived from first principles, and no claim of optimality is made.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = 10.0 ** (-jnp.arange(1, 51, dtype=jnp.float64))
    >>> standard_chop(coeffs)
    18

    Provenance
    ----------
    MATLAB source : standardChop.m
    Chebfun commit: 7574c77
    Original authors: Jared Aurentz and Nick Trefethen, July 2015
    Algorithm:
        J. L. Aurentz and L. N. Trefethen, "Chopping a Chebyshev series",
        http://arxiv.org/abs/1512.01803, December 2015.

    See Also
    --------
    gridsample, abstract_qr
    """
    tol = float(_EPS) if tol is None else float(tol)
    if tol >= 1:
        return 1
    coeffs = jnp.ravel(jnp.atleast_1d(jnp.asarray(coeffs)))
    if coeffs.size < 17:
        return int(coeffs.size)
    return int(_standard_chop_kernel(coeffs, jnp.asarray(tol, dtype=jnp.float64)))


@jax.jit
def _standard_chop_kernel(coeffs: jnp.ndarray, tol: jnp.ndarray) -> jnp.ndarray:
    """JAX implementation of standardChop's three source-indexed steps."""
    n = coeffs.size
    magnitudes = jnp.asarray(jnp.abs(coeffs), dtype=jnp.float64)
    envelope = jax.lax.associative_scan(jnp.maximum, magnitudes, reverse=True)
    scale = envelope[0]
    envelope = envelope / jnp.where(scale == 0, 1.0, scale)

    # MATLAB uses half-away rounding. For positive source indices this exact
    # integer expression is round(1.25*j + 5), including j=6,14,22,... ties.
    j = jnp.arange(2, n + 1)
    windows = (5 * j + 22) // 4
    e1 = envelope[j - 1]
    e2 = envelope[jnp.minimum(windows - 1, n - 1)]
    r = 3.0 * (1.0 - jnp.log(e1) / jnp.log(tol))
    plateau = (windows <= n) & ((e1 == 0) | (e2 / e1 > r))
    first = jnp.min(jnp.where(plateau, j, n + 1))
    found = first <= n
    plateau_point = first - 1
    window = jnp.minimum((5 * first + 22) // 4, n)

    threshold = tol ** (7.0 / 6.0)
    j3 = jnp.sum(envelope >= threshold)
    shortened = j3 < window
    window = jnp.where(shortened, j3 + 1, window)
    envelope = envelope.at[window - 1].set(
        jnp.where(shortened, threshold, envelope[window - 1])
    )
    indices = jnp.arange(n)
    bias = indices * ((-1.0 / 3.0) * jnp.log10(tol)) / (window - 1)
    cc = jnp.where(indices < window, jnp.log10(envelope) + bias, jnp.inf)
    cutoff = jnp.maximum(jnp.argmin(cc), 1)
    cutoff = jnp.where(envelope[jnp.minimum(plateau_point - 1, n - 1)] == 0,
                       plateau_point, cutoff)
    return jnp.where(scale == 0, 1, jnp.where(found, cutoff, n))


def gridsample(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    domain: tuple[float, float] | None = None,
    kind: str = "cheb",
) -> jnp.ndarray:
    """Sample a function on a Chebyshev or trigonometric grid.

    Parameters
    ----------
    f : callable
        Function mapping an array of points to an array of values.
    n : int
        Number of grid points.
    domain : (float, float) or None, optional
        Interval ``[a, b]``.  Default is ``[-1, 1]``.
    kind : {'cheb', 'trig'}, default 'cheb'
        ``'cheb'`` for Chebyshev points (2nd kind), ``'trig'`` for
        equispaced trigonometric points.

    Returns
    -------
    v : jnp.ndarray, shape (n,)
        Function values at the grid points.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> v = gridsample(jnp.sin, 5)
    >>> v.shape
    (5,)

    Provenance
    ----------
    MATLAB source : gridsample.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Aurentz and Trefethen, "Block operators and spectral
        discretizations".

    See Also
    --------
    standard_chop, abstract_qr
    """
    from chebfunjax.utils.quadrature import chebpts, chebpts_ab

    if kind == "cheb":
        if domain is None:
            x = chebpts(n, kind=2)
        else:
            a, b = domain
            x = chebpts_ab(n, a, b, kind=2)
    elif kind == "trig":
        if domain is None:
            a, b = -1.0, 1.0
        else:
            a, b = domain
        x = jnp.linspace(a, b, n, endpoint=False, dtype=jnp.float64)
    else:
        raise ValueError(
            f"kind must be 'cheb' or 'trig', got {kind!r}. "
            f"Use 'cheb' for Chebyshev points or 'trig' for trigonometric points."
        )

    return f(x)


def abstract_qr(
    A: jnp.ndarray,
    E: jnp.ndarray,
    inner_product: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    my_norm: Callable[[jnp.ndarray], float] | None = None,
    tol: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Abstract Householder QR factorisation with a user-supplied inner product.

    Computes a weighted QR factorisation of ``A``, where the orthogonality of
    the columns of ``Q`` is measured by ``inner_product`` instead of the
    standard Euclidean inner product.  ``E`` is a matrix of the same shape as
    ``A`` that provides an orthonormal basis for the column space (typically a
    Legendre–Vandermonde matrix when the columns of ``A`` are function values
    on a Chebyshev grid).

    Parameters
    ----------
    A : jnp.ndarray, shape (m, p)
        Input matrix (or matrix of function samples).
    E : jnp.ndarray, shape (m, p)
        Basis matrix (e.g., Legendre–Chebyshev–Vandermonde matrix).
    inner_product : callable (u, v) -> scalar
        Inner product ``<u, v>`` (conjugate-linear in the first argument).
    my_norm : callable (u) -> float, optional
        Norm estimate for thresholding.  Default: ``jnp.linalg.norm``.
    tol : float, optional
        Tolerance for deciding when a column is numerically zero.
        Default: machine epsilon (~2.2e-16).

    Returns
    -------
    Q : jnp.ndarray, shape (m, p)
        Orthogonal factor (columns are orthonormal w.r.t. ``inner_product``).
    R : jnp.ndarray, shape (p, p)
        Upper-triangular factor.

    Notes
    -----
    The algorithm is the abstract Householder triangularisation described in:

        L. N. Trefethen, "Householder triangularization of a quasimatrix",
        IMA J. Numer. Anal., 30(4):887–897, 2010.

    This function uses Python loops over columns and is therefore **not
    JIT-safe**.  It is intended for construction-time use (e.g., computing
    a QR of an array-valued chebfun for ``qr``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> E = jnp.eye(3, 2)
    >>> Q, R = abstract_qr(A, E, lambda u, v: jnp.dot(u, v))
    >>> Q.shape, R.shape
    ((3, 2), (2, 2))

    Provenance
    ----------
    MATLAB source : abstractQR.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        L. N. Trefethen, "Householder triangularization of a quasimatrix",
        IMA J. Numer. Anal., 30(4):887–897, 2010.

    See Also
    --------
    standard_chop, gridsample
    """
    if my_norm is None:
        my_norm = jnp.linalg.norm
    if tol is None:
        tol = float(_EPS)

    a_in = jnp.asarray(A)
    e_in = jnp.asarray(E)
    dtype = jnp.result_type(a_in.dtype, e_in.dtype, jnp.float64)
    a_work = jnp.asarray(a_in, dtype=dtype)
    e_work = jnp.asarray(e_in, dtype=dtype)
    num_cols = a_work.shape[1]
    r_mat = jnp.zeros((num_cols, num_cols), dtype=dtype)
    v_mat = a_work

    for k in range(num_cols):
        a_col = a_work[:, k]
        e_col = e_work[:, k]
        scale = jnp.maximum(my_norm(e_col), my_norm(a_col))

        ex = inner_product(e_col, a_col)
        abs_ex = jnp.abs(ex)
        if bool(abs_ex < tol * scale):
            sign = jnp.asarray(1.0, dtype=dtype)
        else:
            sign = -ex / abs_ex
        e_work = e_work.at[:, k].set(e_col * sign)
        e_col = e_work[:, k]

        r_kk_sq = inner_product(a_col, a_col)
        r_kk = jnp.sqrt(jnp.real(r_kk_sq))
        r_mat = r_mat.at[k, k].set(r_kk)

        v = r_kk * e_col - a_col
        for i in range(k):
            e_prev = e_work[:, i]
            v = v - e_prev * inner_product(e_prev, v)

        v_norm_sq = inner_product(v, v)
        v_norm = jnp.sqrt(jnp.real(v_norm_sq))
        if bool(v_norm < tol * scale):
            v = e_col
        else:
            v = v / v_norm
        v_mat = v_mat.at[:, k].set(v)

        for j in range(k + 1, num_cols):
            a_j = a_work[:, j]
            a_j = a_j - 2.0 * v * inner_product(v, a_j)
            r_kj = inner_product(e_col, a_j)
            r_mat = r_mat.at[k, j].set(r_kj)
            a_work = a_work.at[:, j].set(a_j - e_col * r_kj)

    q_mat = e_work
    for k in range(num_cols - 1, -1, -1):
        v = v_mat[:, k]
        for j in range(k, num_cols):
            q_j = q_mat[:, j]
            q_mat = q_mat.at[:, j].set(
                q_j - 2.0 * v * inner_product(v, q_j)
            )

    return q_mat, r_mat

def isSubset(A, B, tol: float = 0.0) -> bool:
    """True if hyper-rectangle A is contained in B up to tol
    (MATLAB isSubset).  A and B are length-2N vectors
    [a1 b1 a2 b2 ...] as used for chebfun/chebfun2/chebfun3 domains.

    Provenance
    ----------
    MATLAB source : isSubset.m
    Chebfun commit: 7574c77
    """
    a = jnp.ravel(jnp.asarray(A, dtype=jnp.float64))
    b = jnp.ravel(jnp.asarray(B, dtype=jnp.float64))
    if a.size == 0:
        return True
    if a.size != b.size:
        raise ValueError("Domains must have the same number of entries.")
    if a.size % 2:
        raise ValueError("Domains must contain endpoint pairs.")
    for i in range(0, a.size, 2):
        if bool(a[i] < b[i] - tol) or bool(b[i + 1] + tol < a[i + 1]):
            return False
    return True

def make_empty_aware(cls, names):
    """Wrap the named methods of cls so that calling them on the
    empty object (cls.empty()) returns the empty object instead of
    crashing -- MATLAB propagates emptiness through its command set
    (see the per-class test_emptyObjects.m files).

    Provenance
    ----------
    MATLAB source : emptiness handling across @chebfun2/@chebfun3/
        @spherefun/@diskfun methods
    Chebfun commit: 7574c77
    """
    import functools

    for _nm in names:
        _orig = getattr(cls, _nm, None)
        if _orig is None:
            continue

        def _wrap(orig):
            @functools.wraps(orig)
            def wrapper(self, *a, **k):
                if getattr(self, "_is_empty_object", False):
                    return cls.empty()
                return orig(self, *a, **k)
            return wrapper

        setattr(cls, _nm, _wrap(_orig))


def op_arity(fn, default: int) -> int:
    """Number of REQUIRED positional parameters of ``fn``.

    Parameters with defaults are excluded: ``lambda u, _e=eps:`` is the
    standard Python idiom for capturing a loop variable, and counting
    ``_e`` makes a one-unknown operator look like ``op(x, u)`` -- the
    solver then passes the independent variable (or a probe object) in
    as the unknown and the captured constant receives the unknown.
    Every arity decision on user callables must go through here
    (ode-nonlin/BlowupFK and AllenCahn were both broken by raw
    ``len(signature(op).parameters)`` counts).
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return default
    required = [q for q in params
                if q.default is q.empty
                and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
    return len(required) if required else default
