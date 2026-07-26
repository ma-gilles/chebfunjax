# uses-numpy: scattered-point NUFFT evaluation (dense contraction / finufft; not JIT-safe)
"""Fast, high-accuracy evaluation of a :class:`Spherefun` at scattered points.

This module ports MATLAB Chebfun's ``@spherefun/fastSphereEval.m``: given a
spherefun ``f`` and arbitrary evaluation points ``(lambda, theta)`` (longitude
and co-latitude), it evaluates ``f`` to ~1e-15 accuracy per point via the
type-2 two-dimensional non-uniform FFT of the doubled-up Fourier--Fourier
coefficient matrix.

Why this exists
---------------
A spherefun is stored in CDR (columns / rows Trigtechs and pivots) form.  The
default evaluation (:meth:`Spherefun.__call__`) sums ``rank`` separable
products, each evaluated by the real Horner scheme in ``@trigtech/horner.m``.
For a high-rank function this accumulates ~1e-14 error per point.  Assembling
the full 2D Fourier coefficient matrix ``G`` and evaluating the trigonometric
series directly (a type-2 2D NUFFT) is backward stable and reaches ~1e-15 per
point.  That sub-ulp gain is what lets :meth:`Spherefun.rotate` reach MATLAB's
``10*eps`` round-trip bound (see ``tests/test_matlab_port/spherefun/
test_rotate_matlab.py``).

Algorithm
---------
The spherefun evaluates to a doubled-up Fourier series

    f(lambda, theta) = sum_{k,l} G[k, l] * exp(1i * k * theta) * exp(1i * l * lambda)

where ``k`` (co-latitude wavenumbers) and ``l`` (longitude wavenumbers) are the
centred integer modes of the column and row Trigtechs, and

    G = C * diag(1/pivots) * R^T                             (coefficient space)

with ``C[k, j]`` the column-Trigtech coefficient of term ``j`` at wavenumber
``k`` and ``R[l, j]`` the row-Trigtech coefficient.  The sum is a type-2 2D
NUFFT (see :func:`chebfunjax.utils.nufft.nufft2`), evaluated exactly by a dense
contraction, or via ``finufft`` when installed.

References
----------
.. [1] D. Ruiz-Antoln and A. Townsend, "A nonuniform fast Fourier transform
   based on low rank approximation", SIAM J. Sci. Comput., 40, A529-A547, 2018.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fast_sphere_eval"]


def _centered_wavenumbers(n: int) -> np.ndarray:
    """Centred integer wavenumbers of a length-``n`` Trigtech.

    The Trigtech stores ``f(x) = sum_k c_k exp(i*pi*k*x)`` with the constant
    mode ``c_0`` at index ``n // 2`` (see ``@trigtech/horner.m`` and
    :func:`chebfunjax.spherefun.spherefun._shift_lambda`), so the wavenumbers
    are ``arange(n) - n // 2``.
    """
    return np.arange(n) - (n // 2)


def _assemble_coeff_matrix(f):
    """Assemble the doubled-up 2D Fourier coefficient matrix ``G``.

    Returns
    -------
    G : np.ndarray, shape (nk, nl), complex128
        ``G[a, b]`` is the coefficient of ``exp(1i*(kmin+a)*theta) *
        exp(1i*(lmin+b)*lambda)``.
    kmin, lmin : int
        The smallest co-latitude / longitude wavenumbers (row/col offsets).
    """
    cols = f.cols
    rows = f.rows
    pivots = np.asarray(f.pivots)

    col_lens = [c.coeffs.shape[0] for c in cols]
    row_lens = [r.coeffs.shape[0] for r in rows]

    kmin = min(int(_centered_wavenumbers(nc).min()) for nc in col_lens)
    kmax = max(int(_centered_wavenumbers(nc).max()) for nc in col_lens)
    lmin = min(int(_centered_wavenumbers(nr).min()) for nr in row_lens)
    lmax = max(int(_centered_wavenumbers(nr).max()) for nr in row_lens)

    nk = kmax - kmin + 1
    nl = lmax - lmin + 1
    G = np.zeros((nk, nl), dtype=np.complex128)

    for j in range(len(cols)):
        cc = np.asarray(cols[j].coeffs, dtype=np.complex128)
        rr = np.asarray(rows[j].coeffs, dtype=np.complex128)
        ki = _centered_wavenumbers(cc.shape[0]) - kmin
        li = _centered_wavenumbers(rr.shape[0]) - lmin
        G[np.ix_(ki, li)] += (1.0 / pivots[j]) * np.outer(cc, rr)

    return G, kmin, lmin


def _nufft2d2(G, kmin, lmin, lam, theta):
    r"""Type-2 2D NUFFT: sum_{a,b} G[a,b] exp(1i*(kmin+a)*theta + 1i*(lmin+b)*lambda).

    Uses ``finufft`` when installed (O(nk*nl*log + N)); otherwise an exact
    dense contraction (O(N*nk*nl)), matching :func:`chebfunjax.utils.nufft.nufft2`.
    """
    nk, nl = G.shape
    theta = np.asarray(theta, dtype=np.float64).ravel()
    lam = np.asarray(lam, dtype=np.float64).ravel()

    try:
        import finufft as _finufft

        # finufft 2d type-2: c[j] = sum_{p,q} f[p,q] exp(isign*1i*(p*s[j]+q*t[j]))
        # with p in -nk//2..nk//2-1 (mode 0 of the array = -floor(nk/2)).
        # Our array index a=0 is wavenumber kmin; finufft's is -floor(nk/2).
        # Feed s = theta, t = lambda, isign=+1, and correct the offset between
        # kmin and -floor(nk/2).
        s = np.mod(theta + np.pi, 2.0 * np.pi) - np.pi
        t = np.mod(lam + np.pi, 2.0 * np.pi) - np.pi
        vals = _finufft.nufft2d2(
            s, t, np.asfortranarray(G), isign=1, eps=1e-15
        )
        off_k = kmin - (-(nk // 2))
        off_l = lmin - (-(nl // 2))
        vals = vals * np.exp(1j * (off_k * theta + off_l * lam))
        return vals
    except ImportError:
        pass

    # Exact dense contraction.
    kw = np.arange(kmin, kmin + nk)
    lw = np.arange(lmin, lmin + nl)
    a_theta = np.exp(1j * np.outer(theta, kw))  # (N, nk)
    b_lam = np.exp(1j * np.outer(lam, lw))       # (N, nl)
    return np.sum((a_theta @ G) * b_lam, axis=1)


def fast_sphere_eval(f, lam, theta):
    r"""Fast, high-accuracy evaluation of a spherefun at scattered points.

    Evaluates ``f`` at the points ``(lam, theta)`` -- longitude ``lam`` with
    ``-pi <= lam <= pi`` and co-latitude ``theta`` with ``0 <= theta <= pi`` --
    via the type-2 two-dimensional NUFFT of the doubled-up Fourier--Fourier
    coefficient matrix.  Reaches ~1e-15 per point, versus ~1e-14 for the
    Horner-scheme :meth:`Spherefun.__call__`.

    Parameters
    ----------
    f : Spherefun
        The spherefun to evaluate.
    lam, theta : array_like
        Evaluation points; broadcast against one another.  ``lam`` is longitude
        in ``[-pi, pi]``, ``theta`` co-latitude in ``[0, pi]``.

    Returns
    -------
    np.ndarray
        Values of ``f`` at ``(lam, theta)``, shape ``broadcast(lam, theta)``.
        Real for a real-valued ``f`` (all chebfunjax spherefuns are real);
        complex otherwise.

    Notes
    -----
    NOT JIT-safe (uses numpy / optional ``finufft``).  For a real spherefun the
    imaginary part is discarded, matching ``real(fastSphereEval(...))`` in
    ``@spherefun/rotate.m``.

    Provenance
    ----------
    MATLAB source : @spherefun/fastSphereEval.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.  Primary author: Alex Townsend.
    Algorithm: Ruiz-Antoln & Townsend, SISC 40(1), 2018.
    """
    lam_a = np.asarray(lam, dtype=np.float64)
    theta_a = np.asarray(theta, dtype=np.float64)
    out_shape = np.broadcast(lam_a, theta_a).shape

    if getattr(f, "isempty", None) is not None and f.isempty():
        return np.zeros(out_shape, dtype=np.float64)

    lam_f = np.broadcast_to(lam_a, out_shape).ravel()
    theta_f = np.broadcast_to(theta_a, out_shape).ravel()

    G, kmin, lmin = _assemble_coeff_matrix(f)
    vals = _nufft2d2(G, kmin, lmin, lam_f, theta_f)

    is_real = bool(f.isreal()) if hasattr(f, "isreal") else True
    if is_real:
        vals = vals.real
    return vals.reshape(out_shape)
