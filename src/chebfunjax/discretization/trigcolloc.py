"""Trigonometric collocation discretization for periodic problems.

Provides ``TrigColloc``: spectral collocation on equidistant (Fourier) points
for periodic functions and periodic differential operators.  This is the
counterpart of ``ChebColloc2`` / ``ChebColloc1`` for non-periodic problems.

Translated from MATLAB Chebfun classes @trigcolloc and @trigspec (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.domain import Domain

# ---------------------------------------------------------------------------
# Stand-alone matrix helpers (pure functions, usable without TrigColloc)
# ---------------------------------------------------------------------------


def trig_diffmat(N: int, m: int = 1) -> jnp.ndarray:
    r"""Trigonometric Fourier differentiation matrix of order *m*.

    Returns the N×N matrix ``D`` such that ``D @ f`` gives the values of
    the *m*-th derivative of the trigonometric interpolant through ``f`` at
    the ``N`` equidistant points on ``[-1, 1)`` (i.e. the period is 2).

    The matrix is computed analytically for m = 1, 2, 3, 4 and via FFT for
    higher orders.  The final result is scaled to the interval ``[-1, 1)``
    (period 2) from the standard ``[-π, π)`` interval (period 2π).

    Parameters
    ----------
    N : int
        Number of equidistant collocation points.
    m : int, default 1
        Order of differentiation.  ``m=0`` returns the identity matrix.

    Returns
    -------
    D : jnp.ndarray, shape (N, N)
        Real-valued trigonometric differentiation matrix for the interval
        ``[-1, 1)`` (period 2).

    Notes
    -----
    The equidistant points are ``x_k = -1 + 2k/N`` for ``k = 0, ..., N-1``.

    The scaling from ``[-π, π)`` to ``[-1, 1)`` introduces a factor of
    ``π^m`` (each derivative multiplies by ``π``).

    For odd N, the first-derivative column entry uses ``csc`` (cosecant);
    for even N it uses ``cot`` (cotangent).  These are the standard formulae
    from Trefethen's "Spectral Methods in MATLAB" (SIAM, 2000), Programme 8.

    Developer notes from MATLAB Chebfun (@trigcolloc/diffmat.m):

    The higher-order (m > 4) matrices are computed as
    ``real(ifft(diag(lambda^m) * fft(eye(N))))``
    where ``lambda`` is the Fourier eigenvalue vector
    ``[0, 1, 2, ..., N/2-1, 0, -N/2+1, ..., -1]`` (even N)
    or ``[0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1]`` (odd N).
    This is both fast and numerically stable.

    References
    ----------
    .. [1] L. N. Trefethen, "Spectral Methods in MATLAB", SIAM, 2000.
    .. [2] J. A. C. Weideman and S. C. Reddy, "A MATLAB Differentiation
       Matrix Suite", ACM TOMS, Vol. 26, No. 4, pp. 465-519, 2000.

    Examples
    --------
    First-derivative matrix for N=6 equidistant points on [-1, 1):

    >>> D = trig_diffmat(6, m=1)
    >>> D.shape
    (6, 6)
    >>> # Differentiate sin(pi*x):  derivative is pi*cos(pi*x)
    >>> import jax.numpy as jnp
    >>> x = jnp.array([-1 + 2*k/6 for k in range(6)], dtype=jnp.float64)
    >>> f = jnp.sin(jnp.pi * x)
    >>> Df = D @ f
    >>> exact = jnp.pi * jnp.cos(jnp.pi * x)
    >>> float(jnp.max(jnp.abs(Df - exact))) < 1e-12
    True

    Provenance
    ----------
    MATLAB source : @trigcolloc/diffmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Trefethen [1], Weideman & Reddy [2].

    See Also
    --------
    TrigColloc, trig_cumsummat
    """
    if N == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    if N == 1:
        return jnp.zeros((1, 1), dtype=jnp.float64)
    if m == 0:
        return jnp.eye(N, dtype=jnp.float64)

    # Grid spacing on [-pi, pi)
    h = 2.0 * jnp.pi / N

    if m == 1:
        # First-order trigonometric differentiation matrix
        k = jnp.arange(1, N, dtype=jnp.float64)
        if N % 2 == 1:  # odd N
            col_entries = 0.5 / jnp.sin(k * h / 2.0)
        else:  # even N
            col_entries = 0.5 / jnp.tan(k * h / 2.0)
        # Alternate signs
        signs = jnp.where(jnp.arange(1, N) % 2 == 0, 1.0, -1.0)
        col_entries = col_entries * signs
        column = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), col_entries])
        row = column[jnp.array([0, *range(N - 1, 0, -1)])]
        D = _toeplitz(column, row)

    elif m == 2:
        # Second-order
        k = jnp.arange(1, N, dtype=jnp.float64)
        if N % 2 == 1:
            entries = 0.5 * jnp.sin(k * h / 2.0) * jnp.cos(k * h / 2.0) / jnp.sin(k * h / 2.0) ** 2
            # csc*cot = cos/(sin^2)
            entries = 0.5 * jnp.cos(k * h / 2.0) / jnp.sin(k * h / 2.0) ** 2
        else:
            entries = 0.5 / jnp.sin(k * h / 2.0) ** 2
        # For m=2, the constant diagonal term is different
        if N % 2 == 1:
            # diag = pi^2/3/h^2 - 1/12
            d0 = jnp.pi**2 / 3.0 / h**2 - 1.0 / 12.0
        else:
            d0 = jnp.pi**2 / 3.0 / h**2 + 1.0 / 6.0

        # Signs: alternate starting from d0, entries[0] (k=1) ...
        # For even indices (k=0,2,4,...) of column, sign = +; odd = -
        # MATLAB: column(1:2:end) = -column(1:2:end)  (1-indexed, so 1,3,5,...)
        # In 0-indexed terms: negate indices 0, 2, 4, ...
        signs = jnp.where(jnp.arange(N) % 2 == 0, -1.0, 1.0)
        # Actually, MATLAB code for m=2: column(1:2:end) = -column(1:2:end)
        # with column = [d0; 0.5*csc(...).^2 or similar]
        # For even N: column = [pi^2/3/h^2 + 1/6, 0.5*csc(k*h/2).^2]
        # then column(1:2:end) = -column(1:2:end)
        col = jnp.concatenate([jnp.array([d0]), entries])
        # negate indices 0, 2, 4, ... (0-indexed = positions 1, 3, 5, ... in 1-indexed)
        col = col.at[::2].multiply(-1.0)
        D = _toeplitz(col, col)

    elif m == 3:
        # Third-order
        k = jnp.arange(1, N, dtype=jnp.float64)
        csc_k = 1.0 / jnp.sin(k * h / 2.0)
        cot_k = jnp.cos(k * h / 2.0) / jnp.sin(k * h / 2.0)
        if N % 2 == 1:
            entries = 3.0 / 8.0 * csc_k * cot_k**2 + 3.0 / 8.0 * csc_k**3 - jnp.pi**2 / (2.0 * h**2) * csc_k
        else:
            entries = 3.0 / 4.0 * csc_k**2 * cot_k - jnp.pi**2 / (2.0 * h**2) * cot_k
        signs = jnp.where(jnp.arange(1, N) % 2 == 0, 1.0, -1.0)
        entries = entries * signs
        column = jnp.concatenate([jnp.array([0.0]), entries])
        row = column[jnp.array([0, *range(N - 1, 0, -1)])]
        D = _toeplitz(column, row)

    elif m == 4:
        # Fourth-order
        k = jnp.arange(1, N, dtype=jnp.float64)
        csc_k = 1.0 / jnp.sin(k * h / 2.0)
        cot_k = jnp.cos(k * h / 2.0) / jnp.sin(k * h / 2.0)
        if N % 2 == 1:
            d0 = -(jnp.pi**4) / (5.0 * h**4) + jnp.pi**2 / (6.0 * h**2) - 7.0 / 240.0
            entries = (5.0 / 4.0 * csc_k**3 * cot_k
                       + 1.0 / 4.0 * csc_k * cot_k**3
                       - (jnp.pi**2 / h**2) * csc_k * cot_k)
        else:
            d0 = -(jnp.pi**4) / (5.0 * h**4) - jnp.pi**2 / (3.0 * h**2) + 1.0 / 30.0
            entries = csc_k**2 * cot_k**2 + 0.5 * csc_k**4 - (jnp.pi**2 / h**2) * csc_k**2
        col = jnp.concatenate([jnp.array([d0]), entries])
        col = col.at[::2].multiply(-1.0)
        D = _toeplitz(col, col)

    else:
        # Higher-order: use FFT eigenvalue approach
        if N % 2 == 1:
            half = (N - 1) // 2
            lam = jnp.concatenate([
                jnp.arange(half + 1, dtype=jnp.float64),
                jnp.arange(-half, 0, dtype=jnp.float64),
            ])
        else:
            half = N // 2
            lam = jnp.concatenate([
                jnp.arange(half, dtype=jnp.float64),
                jnp.array([0.0]),
                jnp.arange(-half + 1, 0, dtype=jnp.float64),
            ])
        lam_m = (1j * lam) ** m
        eye_N = jnp.eye(N, dtype=jnp.complex128)
        D_complex = jnp.real(
            jnp.fft.ifft(lam_m[:, None] * jnp.fft.fft(eye_N, axis=0), axis=0)
        )
        D = D_complex.real.astype(jnp.float64)

    # Scale from [-pi, pi) to [-1, 1): d^m/dx^m -> pi^m * d^m/dx^m
    D = D * (jnp.pi ** m)
    return D


def trig_cumsummat(N: int) -> jnp.ndarray:
    r"""Trigonometric Fourier integration matrix.

    Returns the N×N matrix ``Q`` such that ``Q @ f`` gives the values of
    the antiderivative of the trigonometric interpolant through ``f``, with
    the zero-mean normalization (the antiderivative is defined up to an
    additive constant; this implementation zeroes the DC component of the
    antiderivative).

    Parameters
    ----------
    N : int
        Number of equidistant collocation points on ``[-1, 1)``.

    Returns
    -------
    Q : jnp.ndarray, shape (N, N)
        Integration matrix.

    Notes
    -----
    The antiderivative of a periodic function is only periodic if the DC
    (zero-frequency) component of the function is zero.  For a general
    periodic function ``f``, this integration matrix computes the integral
    of the zero-mean part of ``f``.

    The matrix is constructed via the Fourier eigenvalue approach:
    divide each non-zero Fourier mode by its eigenvalue ``i*k*pi``.
    The DC mode is set to zero.

    Unlike the Chebyshev case, indefinite integration of a periodic function
    requires the DC component to vanish.  MATLAB Chebfun's trigcolloc
    does not support indefinite integration (raises an error); we provide
    a matrix that operates on zero-mean functions.

    Provenance
    ----------
    MATLAB source : @trigcolloc/cumsummat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    trig_diffmat, TrigColloc
    """
    if N == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    if N == 1:
        return jnp.zeros((1, 1), dtype=jnp.float64)

    # Fourier eigenvalue vector (with 0 for DC mode)
    if N % 2 == 1:
        half = (N - 1) // 2
        lam = jnp.concatenate([
            jnp.arange(half + 1, dtype=jnp.float64),
            jnp.arange(-half, 0, dtype=jnp.float64),
        ])
    else:
        half = N // 2
        lam = jnp.concatenate([
            jnp.arange(half, dtype=jnp.float64),
            jnp.array([0.0]),
            jnp.arange(-half + 1, 0, dtype=jnp.float64),
        ])

    # Integration: eigenvalue 1/(i*k*pi) for k != 0; 0 for k = 0
    safe_lam = jnp.where(lam == 0, 1.0, lam)  # avoid divide-by-zero
    inv_eig = jnp.where(lam == 0, 0.0, 1.0 / (1j * safe_lam * jnp.pi))

    eye_N = jnp.eye(N, dtype=jnp.complex128)
    Q_complex = jnp.fft.ifft(inv_eig[:, None] * jnp.fft.fft(eye_N, axis=0), axis=0)
    Q = jnp.real(Q_complex).astype(jnp.float64)
    return Q


# ---------------------------------------------------------------------------
# Private helper: build Toeplitz matrix from column and row
# ---------------------------------------------------------------------------


def _toeplitz(col: jnp.ndarray, row: jnp.ndarray) -> jnp.ndarray:
    """Build a Toeplitz matrix from first column and first row.

    Parameters
    ----------
    col : jnp.ndarray, shape (n,)
        First column (including the [0,0] entry).
    row : jnp.ndarray, shape (m,)
        First row (including the [0,0] entry, which must equal col[0]).

    Returns
    -------
    T : jnp.ndarray, shape (n, m)
    """
    n = col.shape[0]
    m = row.shape[0]
    i = jnp.arange(n)
    j = jnp.arange(m)
    idx = i[:, None] - j[None, :]  # shape (n, m)
    col_ext = jnp.concatenate([col, jnp.zeros(max(m - 1, 0), dtype=col.dtype)])
    row_ext = jnp.concatenate([row, jnp.zeros(max(n - 1, 0), dtype=row.dtype)])
    # Use abs(idx) and select from col (idx>=0) or row (idx<0)
    abs_idx = jnp.abs(idx)
    T = jnp.where(idx >= 0, col_ext[abs_idx], row_ext[abs_idx])
    return T


# ---------------------------------------------------------------------------
# TrigColloc class
# ---------------------------------------------------------------------------


class TrigColloc(eqx.Module):
    """Trigonometric collocation discretization on equidistant points.

    Represents a spectral collocation discretization of a function or
    periodic differential operator on ``N`` equidistant points in ``[-1, 1)``
    (the interval wraps: ``x_k = -1 + 2k/N``, ``k = 0, ..., N-1``).

    This is the Fourier / trigonometric analogue of ``ChebColloc2`` and is
    appropriate for periodic problems.  The differentiation matrix is the
    standard Fourier pseudospectral differentiation matrix.

    Attributes
    ----------
    n : int
        Number of equidistant collocation points.
    domain : Domain
        Physical domain ``[a, b)`` on which the problem is posed.

    Examples
    --------
    Solve the periodic boundary-value problem  u'' = -4π² u,  u(0) = 1
    (i.e., u(x) = cos(2πx) on [0, 1)):

    >>> from chebfunjax.domain import Domain
    >>> disc = TrigColloc(n=16, domain=Domain((0.0, 1.0)))
    >>> D2 = disc.diffmat(k=2)           # 16×16 second-derivative matrix
    >>> x  = disc.points()               # 16 equidistant points on [0, 1)
    >>> import jax.numpy as jnp
    >>> # The operator L u = u'' + 4π² u should have cos(2πx) in its kernel.
    >>> # We enforce one BC by fixing x[0] row.
    >>> A = D2 + (2 * jnp.pi)**2 * jnp.eye(16)
    >>> rhs = jnp.zeros(16)
    >>> # Verify: A @ cos(2πx) ≈ 0
    >>> u = jnp.cos(2 * jnp.pi * x)
    >>> float(jnp.max(jnp.abs(A @ u))) < 1e-10
    True

    Provenance
    ----------
    MATLAB source : @trigcolloc/trigcolloc.m, @trigcolloc/diffmat.m,
        @trigcolloc/cumsummat.m, @trigcolloc/equationPoints.m,
        @trigcolloc/functionPoints.m, @trigcolloc/toValues.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ChebColloc2, ChebColloc1, trig_diffmat, trig_cumsummat
    """

    n: int = eqx.field(static=True)
    domain: Domain = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        domain: "Domain | tuple[float, float]" = (-1.0, 1.0),
    ) -> None:
        """Create a TrigColloc discretization.

        Parameters
        ----------
        n : int
            Number of equidistant collocation points.  Must be >= 1.
        domain : Domain or (float, float), default (-1, 1)
            Physical interval ``[a, b)``.  A tuple ``(a, b)`` is
            automatically wrapped in a ``Domain``.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError(
                f"TrigColloc requires n >= 1 collocation points, got n={n}."
            )
        if not isinstance(domain, Domain):
            domain = Domain(domain)
        self.n = n
        self.domain = domain

    # ------------------------------------------------------------------
    # Core discretization operations
    # ------------------------------------------------------------------

    def diffmat(self, k: int = 1) -> jnp.ndarray:
        r"""Trigonometric Fourier differentiation matrix of order *k*.

        Returns the ``(n, n)`` matrix ``D`` such that ``D @ f`` approximates
        the *k*-th derivative of the trigonometric interpolant through the
        function values at the ``n`` equidistant points of ``self.domain``.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.  ``k=0`` returns the identity.

        Returns
        -------
        D : jnp.ndarray, shape (n, n)
            Spectral differentiation matrix scaled to ``self.domain``.

        Notes
        -----
        The matrix for a general domain ``[a, b]`` (period ``T = b - a``)
        is obtained by scaling the standard ``[-1, 1)`` matrix (period 2)
        by ``(2 / T)^k``.

        Provenance
        ----------
        MATLAB source : @trigcolloc/diffmat.m
        Chebfun commit: 7574c77

        See Also
        --------
        trig_diffmat, cumsummat, points
        """
        a, b = self.domain.support
        D_ref = trig_diffmat(self.n, m=k)
        # Scale from [-1, 1) (period 2) to [a, b) (period b-a)
        scale = (2.0 / (b - a)) ** k
        return D_ref * scale

    def cumsummat(self) -> jnp.ndarray:
        r"""Trigonometric integration matrix (antiderivative of zero-mean part).

        Returns the ``(n, n)`` matrix ``Q`` such that ``Q @ f`` gives the
        values of the antiderivative of the zero-mean part of the
        trigonometric interpolant through ``f``.

        Returns
        -------
        Q : jnp.ndarray, shape (n, n)
            Integration matrix scaled to ``self.domain``.

        Notes
        -----
        The antiderivative of a periodic function is only periodic if the
        DC (zero-frequency) component is zero.  ``Q @ f`` computes the
        antiderivative of ``f - mean(f)``.

        Provenance
        ----------
        MATLAB source : @trigcolloc/cumsummat.m
        Chebfun commit: 7574c77

        See Also
        --------
        diffmat, points, weights
        """
        a, b = self.domain.support
        Q_ref = trig_cumsummat(self.n)
        scale = (b - a) / 2.0
        return Q_ref * scale

    def points(self) -> jnp.ndarray:
        """Equidistant collocation points on ``self.domain``.

        Returns
        -------
        x : jnp.ndarray, shape (n,)
            The ``n`` equidistant points ``a + k*(b-a)/n``
            for ``k = 0, ..., n-1`` (left-closed, right-open).

        Provenance
        ----------
        MATLAB source : @trigcolloc/functionPoints.m,
            @trigcolloc/equationPoints.m, @trigtech/trigpts.m
        Chebfun commit: 7574c77

        See Also
        --------
        weights, diffmat
        """
        a, b = self.domain.support
        T = b - a
        k = jnp.arange(self.n, dtype=jnp.float64)
        return a + k * T / self.n

    def equation_points(self) -> jnp.ndarray:
        """Points at which equations are enforced.

        For ``TrigColloc``, equation points coincide with function points
        (both are equidistant nodes).

        Returns
        -------
        x_eq : jnp.ndarray, shape (n,)
            Same as ``self.points()``.

        Provenance
        ----------
        MATLAB source : @trigcolloc/equationPoints.m
        Chebfun commit: 7574c77

        See Also
        --------
        points
        """
        return self.points()

    def weights(self) -> jnp.ndarray:
        r"""Trapezoidal quadrature weights on ``self.domain``.

        For equidistant points on a periodic domain, the trapezoidal rule
        is spectrally accurate.  Each weight equals ``(b - a) / n``.

        Returns
        -------
        w : jnp.ndarray, shape (n,)
            Quadrature weights such that ``jnp.dot(w, f_values)``
            approximates ``∫_{a}^{b} f(x) dx``.

        Notes
        -----
        The trapezoidal rule is exactly the Fourier-series quadrature rule
        for equidistant nodes on a periodic domain.  It integrates all
        trig polynomials of degree < n/2 exactly.

        Provenance
        ----------
        MATLAB source : @trigcolloc/sum.m, @valsDiscretization/points.m,
            @trigtech/quadwts.m
        Chebfun commit: 7574c77

        See Also
        --------
        points, diffmat
        """
        a, b = self.domain.support
        return jnp.full(self.n, (b - a) / self.n, dtype=jnp.float64)

    def eval_matrix(self, y: jnp.ndarray) -> jnp.ndarray:
        r"""Trigonometric interpolation matrix from grid to arbitrary points.

        Returns the ``(M, n)`` matrix ``E`` such that ``E @ f_values``
        evaluates the trigonometric polynomial interpolant through
        ``f_values`` at the ``n`` equidistant points at each of the ``M``
        target points ``y``.

        Parameters
        ----------
        y : jnp.ndarray, shape (M,) or scalar
            Target evaluation points (may lie anywhere on the real line,
            not just inside ``self.domain``).

        Returns
        -------
        E : jnp.ndarray, shape (M, n)
            Trigonometric interpolation matrix.

        Notes
        -----
        The matrix is computed via the DFT approach:
            ``E[j, k] = (1/n) * sum_{m} exp(i*lam_m*(y_j - x_k))``
        where the sum is over Fourier modes and ``x_k`` are the grid points.

        Provenance
        ----------
        MATLAB source : @trigcolloc/toValues.m, @trigtech/vals2coeffs.m,
            @trigtech/coeffs2vals.m
        Chebfun commit: 7574c77

        See Also
        --------
        points, weights
        """
        y = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float64))
        y.shape[0]
        x = self.points()
        a, b = self.domain.support
        T = b - a

        # Build E via Fourier basis evaluation
        # E[j, k] = (1/n) * real part of Fourier interpolant
        # More explicitly: use the DFT of each unit vector
        n = self.n
        if n % 2 == 1:
            half = (n - 1) // 2
            lam = jnp.concatenate([
                jnp.arange(half + 1, dtype=jnp.float64),
                jnp.arange(-half, 0, dtype=jnp.float64),
            ])
        else:
            half = n // 2
            lam = jnp.concatenate([
                jnp.arange(half, dtype=jnp.float64),
                jnp.array([0.0]),
                jnp.arange(-half + 1, 0, dtype=jnp.float64),
            ])

        # Phase: exp(i * 2*pi/T * lam_m * y_j)
        # E[j, k] = (1/n) * sum_m F_m * exp(i*2*pi/T * lam_m * (y_j - x_k))
        # = (1/n) * IDFT_{y_j} of DFT{delta_k}
        # Build matrix via Vandermonde-like construction

        # Frequency axis scaled to [0, 2pi/T]
        omega = 2.0 * jnp.pi / T  # angular frequency step

        # Phase matrix: Phi[j, m] = exp(i * omega * lam_m * y_j)
        Phi_y = jnp.exp(1j * omega * lam[None, :] * y[:, None])  # (M, n)
        # Phase for grid: Phi_x[k, m] = exp(-i * omega * lam_m * x_k)
        Phi_x = jnp.exp(-1j * omega * lam[None, :] * x[:, None])  # (n, n)

        # E = Phi_y @ Phi_x.T / n  (but we want the real part)
        E_complex = Phi_y @ Phi_x.T / n  # (M, n)
        return jnp.real(E_complex).astype(jnp.float64)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TrigColloc(n={self.n}, domain={self.domain})"
