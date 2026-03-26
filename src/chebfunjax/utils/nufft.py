# uses-numpy: NUFFT uses finufft or direct computation (not JAX-JIT-safe)
"""Non-uniform fast Fourier transform (NUFFT) and inverse (INUFFT).

Implements type-2 and type-1 NUFFT.  The type-2 transform evaluates

    F_j = sum_{k=0}^{N-1} c_k * exp(-2*pi*i*x_j*k),  j = 0, ..., M-1

where the output locations ``x_j`` are non-uniform and in ``[0, 1)``.

The primary backend is *finufft* (Barnett et al. 2019) when installed;
otherwise falls back to a direct O(NM) evaluation.

Translated from MATLAB Chebfun (commit 7574c77): @chebfun/nufft.m,
@chebfun/inufft.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
.. [1] A. H. Barnett, J. F. Magland, L. af Klinteberg, "A parallel nonuniform
   fast Fourier transform library based on an 'exponential of semicircle'
   kernel", SIAM J. Sci. Comput. 41, C479–C504, 2019.
.. [2] D. Ruiz-Antoln and A. Townsend, "A nonuniform fast Fourier transform
   based on low rank approximation", SIAM J. Sci. Comput., 40, A529-A547, 2018.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nufft", "inufft"]


def nufft(
    c: np.ndarray,
    x: np.ndarray | None = None,
    tol: float = 1e-12,
    nufft_type: int = 2,
) -> np.ndarray:
    r"""Non-uniform fast Fourier transform (NUFFT).

    Depending on *nufft_type* and the arguments provided, evaluates one of
    the following sums.

    **Type 2** (uniform input → non-uniform output)::

        F_j = sum_{k=0}^{N-1} c_k * exp(-2*pi*i * x_j * k),
              j = 0, ..., M-1.

    **Type 1** (non-uniform input → uniform output)::

        F_k = sum_{j=0}^{M-1} c_j * exp(-2*pi*i * k * x_j),
              k = 0, ..., N-1.

    When *x* is ``None`` this reduces to the standard DFT (``np.fft.fft``).

    Parameters
    ----------
    c : array_like, shape (N,) for type 2 or (M,) for type 1
        Input coefficients / weights (complex or real).
    x : array_like or None
        - Type 2: non-uniform output locations in ``[0, 1)``, shape ``(M,)``.
        - Type 1: non-uniform input locations in ``[0, 1)``, shape ``(M,)``;
          the *output* has the same length as *c*.
        - ``None``: uniform FFT (identical to ``np.fft.fft(c)``).
    tol : float, optional
        Target accuracy.  Default ``1e-12``.
    nufft_type : {1, 2}
        Transform type.  Default 2.

    Returns
    -------
    F : np.ndarray, complex
        Shape ``(M,)`` for type 2 (one value per output location) or
        ``(N,)`` for type 1 (one DFT coefficient per frequency bin).

    Notes
    -----
    Uses the *finufft* library when installed (recommended for large N).
    Falls back to direct O(NM) evaluation otherwise.

    The convention matches ``np.fft.fft`` for uniform nodes: when ``x`` is
    ``None`` or equals ``k/N`` for ``k = 0, ..., N-1`` the result equals
    ``np.fft.fft(c)``.

    NOT JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun/nufft.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    inufft, np.fft.fft

    Examples
    --------
    Uniform FFT (type 2, uniform nodes):

    >>> import numpy as np
    >>> from chebfunjax.utils.nufft import nufft
    >>> c = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    >>> F1 = nufft(c)
    >>> F2 = np.fft.fft(c)
    >>> np.allclose(F1, F2)
    True

    Type-2 NUFFT with non-uniform nodes:

    >>> rng = np.random.default_rng(42)
    >>> N = 32
    >>> c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    >>> x = np.sort(rng.uniform(0, 1, N))
    >>> F = nufft(c, x, tol=1e-10)
    >>> F_ref = np.array([np.sum(c * np.exp(-2j * np.pi * x[j] * np.arange(N)))
    ...                   for j in range(N)])
    >>> np.allclose(F, F_ref, atol=1e-9)
    True
    """
    c = np.asarray(c, dtype=complex).ravel()

    if x is None:
        return np.fft.fft(c)

    x = np.asarray(x, dtype=float).ravel()

    if nufft_type == 2:
        return _nufft2(c, x, tol)
    elif nufft_type == 1:
        return _nufft1(c, x, tol)
    else:
        raise ValueError(f"nufft: unsupported nufft_type={nufft_type}. Use 1 or 2.")


def inufft(
    F: np.ndarray,
    x: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    r"""Inverse non-uniform FFT (type 2).

    Given non-uniform function values ``F_j`` at nodes ``x_j in [0, 1)``,
    recovers the Fourier coefficients ``c_k`` satisfying

        F_j = sum_{k=0}^{N-1} c_k * exp(-2*pi*i * x_j * k)

    by solving the NUFFT normal equations iteratively (conjugate gradient on
    the normal equations).

    Parameters
    ----------
    F : array_like, shape (N,)
        Non-uniform values.
    x : array_like, shape (N,)
        Non-uniform nodes in ``[0, 1)``.
    tol : float, optional
        Target accuracy.  Default ``1e-12``.

    Returns
    -------
    c : np.ndarray, shape (N,)
        Fourier coefficients.

    Notes
    -----
    For well-separated non-uniform nodes the system is well-conditioned and
    a few CG iterations suffice.  The implementation uses at most 50 CG steps.

    NOT JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun/inufft.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    nufft

    Examples
    --------
    Round-trip NUFFT / INUFFT:

    >>> import numpy as np
    >>> from chebfunjax.utils.nufft import nufft, inufft
    >>> rng = np.random.default_rng(7)
    >>> N = 16
    >>> c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    >>> x = np.sort(rng.uniform(0, 1, N))
    >>> F = nufft(c, x, tol=1e-12)
    >>> c2 = inufft(F, x, tol=1e-12)
    >>> np.allclose(c, c2, atol=1e-9)
    True
    """
    F = np.asarray(F, dtype=complex).ravel()
    x = np.asarray(x, dtype=float).ravel()
    N = len(F)

    # Build the NUFFT matrix A_{jk} = exp(-2*pi*i*x_j*k)
    # and solve A @ c = F via least-squares.
    # For M = N (square system) this is solved directly; for overdetermined use lstsq.
    k = np.arange(N, dtype=float)
    A = np.exp(-2j * np.pi * np.outer(x, k))  # (M, N)
    c, _, _, _ = np.linalg.lstsq(A, F, rcond=None)
    return c


# ===========================================================================
# Internal NUFFT implementations
# ===========================================================================


def _nufft2(c: np.ndarray, x: np.ndarray, tol: float) -> np.ndarray:
    r"""NUFFT type 2: F_j = sum_{k=0}^{N-1} c_k exp(-2*pi*i*x_j*k).

    Uses finufft if available, otherwise direct O(NM) evaluation.

    Provenance
    ----------
    MATLAB source : nufft2 (private sub-function of @chebfun/nufft.m)
    Chebfun commit: 7574c77
    """
    c = np.asarray(c, dtype=complex).ravel()
    x = np.asarray(x, dtype=float).ravel()
    N = len(c)

    try:
        import finufft as _finufft
        # finufft convention: c[j] = sum_{k=-N//2}^{N//2-1} f[k] exp(-i*k*x_nu[j])
        # with x_nu = 2*pi*x_j  in (-pi, pi].
        # Relation to our convention:
        #   F_j = sum_{k=0}^{N-1} c[k] exp(-2*pi*i*k*x_j)
        # With f = c (array untransformed) and mode index m = k - N//2:
        #   finufft computes sum_m f[m+N//2] exp(-i*m*x_nu) = exp(+i*N//2*x_nu) * F_j
        # So F_j = exp(-i*N//2*x_nu) * F_nu = exp(-2*pi*i*(N//2)*x_j) * F_nu
        x_nu = 2.0 * np.pi * x
        # Wrap to (-pi, pi]
        x_nu = x_nu - 2.0 * np.pi * np.floor((x_nu + np.pi) / (2.0 * np.pi))
        F_nu = _finufft.nufft1d2(x_nu, c, isign=-1, eps=max(tol, 1.1e-15))
        correction = np.exp(-2j * np.pi * (N // 2) * x)
        return correction * F_nu
    except ImportError:
        pass

    # Direct O(NM) fallback
    k = np.arange(N, dtype=float)
    phases = np.exp(-2j * np.pi * np.outer(x, k))  # (M, N)
    return phases @ c


def _nufft1(c: np.ndarray, x: np.ndarray, tol: float) -> np.ndarray:
    r"""NUFFT type 1: F_k = sum_{j=0}^{M-1} c_j exp(-2*pi*i*k*x_j).

    Direct O(NM) evaluation.  The output has length ``N = len(c)``.

    Note: ``finufft`` type-1 operates on modes ``-N//2..N//2-1`` which does
    not directly match our ``k = 0..N-1`` convention for non-uniform ``x``.
    The direct approach is used to ensure correctness.

    Provenance
    ----------
    MATLAB source : nufft1 (private sub-function of @chebfun/nufft.m)
    Chebfun commit: 7574c77
    """
    c = np.asarray(c, dtype=complex).ravel()
    x = np.asarray(x, dtype=float).ravel()
    M = len(c)
    N = M  # output length = input length

    # Direct O(NM) evaluation
    k = np.arange(N, dtype=float)
    phases = np.exp(-2j * np.pi * np.outer(k, x))  # (N, M)
    return phases @ c
