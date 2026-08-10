"""Miscellaneous utility functions for Chebyshev approximation.

Translated from MATLAB Chebfun (commit 7574c77): standardChop.m, gridsample.m,
abstractQR.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

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
    # --- Default tolerance ---
    if tol is None:
        tol = float(_EPS)
    else:
        tol = float(tol)

    # If tol >= 1, everything is within tolerance; keep only 1 coefficient.
    if tol >= 1:
        return 1

    # Ensure coeffs is a 1-D real array. The envelope is built from
    # magnitudes anyway (MATLAB standardChop), so complex coefficients are
    # reduced via abs() rather than a real-cast that drops imaginary parts.
    # All array work is numpy: standard_chop is called *outside* JIT
    # (adaptive construction is a Python loop) and eager jnp primitives
    # compile one kernel per length -- a large compile tax across
    # rootfinding and constructors.
    import numpy as _np

    coeffs = _np.atleast_1d(_np.asarray(coeffs))
    if _np.iscomplexobj(coeffs):
        coeffs = _np.abs(coeffs)
    coeffs = coeffs.astype(_np.float64).ravel()

    n = coeffs.shape[0]
    cutoff = int(n)

    # Require at least 17 coefficients before attempting to chop.
    if n < 17:
        return cutoff

    # ------------------------------------------------------------------
    # Step 1: Build the envelope — a monotonically non-increasing sequence
    #         normalised to begin at 1.
    # ------------------------------------------------------------------
    b = _np.abs(coeffs)

    # Reverse cumulative maximum: m[j] = max(|c_j|, |c_{j+1}|, ..., |c_{n-1}|).
    # Equivalent to MATLAB's cummax(..., 'reverse').
    m = _np.flip(_np.maximum.accumulate(_np.flip(b)))

    m0 = float(m[0])
    if m0 == 0.0:
        return 1

    envelope_np = m / m0  # normalised, envelope[0] == 1

    # ------------------------------------------------------------------
    # Step 2: Scan for a plateau.
    # ------------------------------------------------------------------
    log_tol = _np.log(tol)
    plateau_point = None

    for j in range(2, n + 1):  # 1-based j matching the MATLAB code
        j2 = int(round(1.25 * j + 5))
        if j2 > n:
            # No plateau found.
            return cutoff

        e1 = float(envelope_np[j - 1])
        e2 = float(envelope_np[j2 - 1])
        r = 3.0 * (1.0 - _np.log(e1) / log_tol) if e1 > 0 else 0.0
        plateau = (e1 == 0.0) or (e2 / e1 > r)
        if plateau:
            plateau_point = j - 1  # 1-based index
            break

    if plateau_point is None:
        return cutoff

    # ------------------------------------------------------------------
    # Step 3: Fine-tune the cutoff.
    # ------------------------------------------------------------------
    if envelope_np[plateau_point - 1] == 0.0:
        cutoff = plateau_point
    else:
        j3 = int(_np.sum(envelope_np >= tol ** (7.0 / 6.0)))
        if j3 < j2:
            j2 = j3 + 1
            envelope_np[j2 - 1] = tol ** (7.0 / 6.0)
        cc = _np.log10(envelope_np[:j2])
        cc = cc + _np.linspace(0.0, (-1.0 / 3.0) * _np.log10(tol), j2)
        d = int(_np.argmin(cc))
        cutoff = max(d, 1)  # d is 0-based index; cutoff is 1-based length

    return cutoff


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
        my_norm = lambda u: float(jnp.linalg.norm(u))  # noqa: E731
    if tol is None:
        tol = float(_EPS)

    num_cols = A.shape[1]

    # Work with mutable numpy arrays to match the MATLAB loop structure.
    import numpy as _np

    A_work = _np.array(A, dtype=_np.float64 if jnp.isrealobj(A) else _np.complex128)
    E_work = _np.array(E, dtype=A_work.dtype)
    R = _np.zeros((num_cols, num_cols), dtype=A_work.dtype)
    V = _np.copy(A_work)  # Will store Householder vectors

    for k in range(num_cols):
        # Scale for deciding if a column is numerically zero.
        scl = max(my_norm(E_work[:, k]), my_norm(A_work[:, k]))

        # Inner product of E(:,k) and A(:,k)
        ex = inner_product(jnp.asarray(E_work[:, k]), jnp.asarray(A_work[:, k]))
        ex = complex(ex) if jnp.iscomplexobj(jnp.asarray(ex)) else float(ex)
        aex = abs(ex)

        # Adjust sign of E(:,k)
        if aex < tol * scl:
            s = 1.0
        else:
            s = -ex / aex  # = -sign(ex/|ex|)
        E_work[:, k] = E_work[:, k] * s

        # Compute the norm of A(:,k) via the inner product
        r_kk = _np.sqrt(
            float(_np.real(inner_product(jnp.asarray(A_work[:, k]), jnp.asarray(A_work[:, k]))))
        )
        R[k, k] = r_kk

        # Compute the Householder reflection vector
        v = r_kk * E_work[:, k] - A_work[:, k]

        # Orthogonalise against previous basis vectors
        for i in range(k):
            ev = inner_product(jnp.asarray(E_work[:, i]), jnp.asarray(v))
            ev = complex(ev) if jnp.iscomplexobj(jnp.asarray(ev)) else float(ev)
            v = v - E_work[:, i] * ev

        # Normalise
        nv = _np.sqrt(
            float(_np.real(inner_product(jnp.asarray(v), jnp.asarray(v))))
        )
        if nv < tol * scl:
            v = E_work[:, k].copy()
        else:
            v = v / nv

        # Store Householder vector
        V[:, k] = v

        # Apply Householder reflection to remaining columns
        for j in range(k + 1, num_cols):
            av = inner_product(jnp.asarray(v), jnp.asarray(A_work[:, j]))
            av = complex(av) if jnp.iscomplexobj(jnp.asarray(av)) else float(av)
            A_work[:, j] = A_work[:, j] - 2.0 * v * av

            rr = inner_product(jnp.asarray(E_work[:, k]), jnp.asarray(A_work[:, j]))
            rr = complex(rr) if jnp.iscomplexobj(jnp.asarray(rr)) else float(rr)
            R[k, j] = rr

            A_work[:, j] = A_work[:, j] - E_work[:, k] * rr

    # Form Q from V (backward accumulation of Householder reflections)
    Q = _np.copy(E_work)
    for k in range(num_cols - 1, -1, -1):
        for j in range(k, num_cols):
            vq = inner_product(jnp.asarray(V[:, k]), jnp.asarray(Q[:, j]))
            vq = complex(vq) if jnp.iscomplexobj(jnp.asarray(vq)) else float(vq)
            Q[:, j] = Q[:, j] - 2.0 * V[:, k] * vq

    # Preserve complexness: a real ``A`` still yields float64 output, but a
    # complex quasimatrix must not have its imaginary part discarded.
    out_dtype = jnp.complex128 if _np.iscomplexobj(A_work) else jnp.float64
    return jnp.asarray(Q, dtype=out_dtype), jnp.asarray(R, dtype=out_dtype)


def isSubset(A, B, tol: float = 0.0) -> bool:
    """True if hyper-rectangle A is contained in B up to tol
    (MATLAB isSubset).  A and B are length-2N vectors
    [a1 b1 a2 b2 ...] as used for chebfun/chebfun2/chebfun3 domains.

    Provenance
    ----------
    MATLAB source : isSubset.m
    Chebfun commit: 7574c77
    """
    import numpy as _np
    A = _np.asarray(A, dtype=float).ravel()
    B = _np.asarray(B, dtype=float).ravel()
    if len(A) != len(B) or len(A) % 2 != 0:
        raise ValueError("A and B must be matching length-2N vectors")
    for i in range(0, len(A), 2):
        if A[i] < B[i] - tol or A[i + 1] > B[i + 1] + tol:
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
