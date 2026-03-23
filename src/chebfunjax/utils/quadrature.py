"""Quadrature points and weights.

Translated from MATLAB Chebfun (commit 7574c77): chebpts.m and related functions.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax.numpy as jnp


def chebpts(n: int, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points of the first or second kind on [-1, 1].

    CHEBPTS(N) returns N Chebyshev points of the 2nd kind in [-1, 1].
    CHEBPTS(N, 1) returns N Chebyshev points of the 1st kind.

    Parameters
    ----------
    n : int
        Number of points.
    kind : {1, 2}
        1 for roots of T_n (Gauss-Chebyshev), 2 for extrema of T_{n-1}
        (Clenshaw-Curtis / Chebyshev-Lobatto). Default is 2.

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Chebyshev points, ordered from -1 to 1.

    Provenance
    ----------
    MATLAB source : chebpts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
            quadrature rules", BIT Numerical Mathematics, 46, 2006.

    See Also
    --------
    chebweights, chebpts_ab
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([0.0], dtype=jnp.float64)

    if kind == 1:
        # Roots of T_n: cos((2k-1)*pi / (2n)), k = 1..n
        k = jnp.arange(n, 0, -1, dtype=jnp.float64)
        x = jnp.cos((2 * k - 1) * jnp.pi / (2 * n))
    elif kind == 2:
        # Extrema of T_{n-1}: cos(k*pi / (n-1)), k = 0..n-1
        k = jnp.arange(n - 1, -1, -1, dtype=jnp.float64)
        x = jnp.cos(k * jnp.pi / (n - 1))
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    return x


def chebpts_ab(n: int, a: float, b: float, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points on the interval [a, b]."""
    x = chebpts(n, kind)
    return 0.5 * ((b - a) * x + (b + a))


def chebweights(n: int, kind: int = 2) -> jnp.ndarray:
    """Clenshaw-Curtis (kind=2) or Gauss-Chebyshev (kind=1) quadrature weights.

    Parameters
    ----------
    n : int
        Number of points.
    kind : {1, 2}
        1 for Gauss-Chebyshev weights, 2 for Clenshaw-Curtis weights.

    Returns
    -------
    w : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebpts.m (weights computed alongside points)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
            quadrature rules", BIT Numerical Mathematics, 46, 2006.

    See Also
    --------
    chebpts
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)

    if kind == 1:
        # Gauss-Chebyshev: w_k = pi/n for all k
        return jnp.full(n, jnp.pi / n, dtype=jnp.float64)
    elif kind == 2:
        # Clenshaw-Curtis weights via FFT (Waldvogel's algorithm)
        return _clenshaw_curtis_weights(n)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def _clenshaw_curtis_weights(n: int) -> jnp.ndarray:
    """Clenshaw-Curtis quadrature weights for n second-kind Chebyshev points.

    Uses Waldvogel's FFT-based algorithm (BIT 2006).
    Points: x_k = cos(k*pi/(n-1)), k = 0..n-1 (descending order).
    Weights satisfy: sum(w) = 2, and integrate polynomials of degree <= n-1 exactly.
    Returns weights in ascending x order (matching chebpts output).
    """
    if n == 2:
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    N = n - 1

    # Chebyshev moments: integral of T_k(x) over [-1,1]
    # = 2/(1-k^2) for even k, 0 for odd k
    # Build the first N+1 moments
    c = jnp.zeros(N + 1, dtype=jnp.float64)
    k_even = jnp.arange(0, N + 1, 2, dtype=jnp.float64)
    c = c.at[0::2].set(2.0 / (1.0 - k_even**2))

    # Mirror to get a vector of length 2N for IFFT
    # v = [c[0], c[1], ..., c[N], c[N-1], ..., c[1]]
    v = jnp.concatenate([c, c[N - 1:0:-1]])

    # IFFT gives weights / N. We want the true CC weights which sum to 2.
    # The IFFT divides by 2N (length of v), but the DCT-I normalization
    # needs division by N only, so multiply by 2.
    w = 2.0 * jnp.real(jnp.fft.ifft(v))

    # Extract the first N+1 values (theta = 0..pi)
    w = w[:N + 1]

    # Halve the endpoints (trapezoidal rule correction)
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[N].set(w[N] / 2.0)

    # Reverse: MATLAB chebpts returns ascending order (-1 to 1)
    return w[::-1]


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------


def legpts(n: int, interval: tuple[float, float] | None = None,
           ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Legendre quadrature nodes and weights.

    LEGPTS(N) returns N Gauss-Legendre nodes in (-1, 1) and the
    corresponding quadrature weights.  The rule integrates polynomials
    of degree <= 2n-1 exactly on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 0).
    interval : (float, float) or None
        If given, rescale nodes and weights to [a, b].

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order.
    w : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : legpts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW), Nick Hale (REC/ASY),
        Ignace Bogaert (fast ASY).
    Algorithm (this implementation): Golub-Welsch eigenvalue method [1].

    References
    ----------
    [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
        rules", Math. Comp. 23, 221-230, 1969.
    [2] I. Bogaert, "Iteration-free computation of Gauss-Legendre quadrature
        nodes and weights", SIAM J. Sci. Comput., 36(3), A1008-A1026, 2014.

    See Also
    --------
    chebpts, jacpts, lobpts, radaupts
    """
    if n == 0:
        x = jnp.array([], dtype=jnp.float64)
        w = jnp.array([], dtype=jnp.float64)
        if interval is not None:
            return x, w
        return x, w

    if n == 1:
        x = jnp.array([0.0], dtype=jnp.float64)
        w = jnp.array([2.0], dtype=jnp.float64)
        if interval is not None:
            a, b = interval
            x = 0.5 * ((b - a) * x + (b + a))
            w = 0.5 * (b - a) * w
        return x, w

    x, w = _legpts_gw(n)

    if interval is not None:
        a, b = interval
        dab = b - a
        x = (x + 1.0) / 2.0 * dab + a
        w = dab * w / 2.0

    return x, w


def _legpts_gw(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Golub-Welsch eigenvalue method for Gauss-Legendre nodes and weights.

    Constructs the symmetric tridiagonal Jacobi matrix for the Legendre
    polynomials and computes its eigenvalues (nodes) and first components
    of eigenvectors (weights).
    """
    i = jnp.arange(1, n, dtype=jnp.float64)
    beta = i / jnp.sqrt(4.0 * i * i - 1.0)
    # Symmetric tridiagonal Jacobi matrix (diagonal is zero for Legendre)
    T = jnp.diag(beta, 1) + jnp.diag(beta, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    x = eigvals
    w = 2.0 * eigvecs[0, :] ** 2

    # Enforce symmetry
    m = n // 2
    x_lo = x[:m]
    w_lo = w[:m]

    if n % 2 == 1:
        x = jnp.concatenate([x_lo, jnp.array([0.0], dtype=jnp.float64),
                             -x_lo[::-1]])
        w_mid = 2.0 - 2.0 * jnp.sum(w_lo)
        w = jnp.concatenate([w_lo, jnp.array([w_mid], dtype=jnp.float64),
                             w_lo[::-1]])
    else:
        x = jnp.concatenate([x_lo, -x_lo[::-1]])
        w = jnp.concatenate([w_lo, w_lo[::-1]])

    return x, w


# ---------------------------------------------------------------------------
# Gauss-Jacobi quadrature
# ---------------------------------------------------------------------------


def jacpts(n: int, a: float, b: float,
           interval: tuple[float, float] | None = None,
           ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Jacobi quadrature nodes and weights.

    Returns the N roots of the degree-N Jacobi polynomial with parameters
    *a* (alpha) and *b* (beta), and the corresponding quadrature weights.
    The Jacobi weight function is w(x) = (1-x)^a * (1+x)^b.

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 0).
    a, b : float
        Jacobi parameters, both must be > -1.
    interval : (float, float) or None
        If given, rescale nodes and weights to [A, B].

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order.
    w : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : jacpts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW), Nick Hale (REC),
        Nick Hale & Alex Townsend (ASY).
    Algorithm (this implementation): Golub-Welsch eigenvalue method [1].

    References
    ----------
    [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
        rules", Math. Comp. 23:221-230, 1969.
    [2] N. Hale and A. Townsend, "Fast computation of Gauss-Jacobi
        quadrature nodes and weights", SISC, 2012.

    See Also
    --------
    legpts, ultrapts, lobpts, radaupts
    """
    if a <= -1.0 or b <= -1.0:
        raise ValueError("Alpha and beta must be greater than -1.")

    if n == 0:
        return (jnp.array([], dtype=jnp.float64),
                jnp.array([], dtype=jnp.float64))

    if n == 1:
        x0 = jnp.array([(b - a) / (a + b + 2.0)], dtype=jnp.float64)
        import jax.scipy.special as jsp
        w0 = jnp.array([2.0 ** (a + b + 1.0)
                         * jnp.exp(jsp.gammaln(a + 1.0)
                                   + jsp.gammaln(b + 1.0)
                                   - jsp.gammaln(a + b + 2.0))],
                        dtype=jnp.float64)
        if interval is not None:
            c1 = 0.5 * (interval[0] + interval[1])
            c2 = 0.5 * (interval[1] - interval[0])
            w0 = c2 ** (a + b + 1.0) * w0
            x0 = c1 + c2 * x0
        return x0, w0

    # Special case: alpha == beta == 0 => Legendre
    if a == 0.0 and b == 0.0:
        return legpts(n, interval=interval)

    x, w = _jacpts_gw(n, a, b)

    if interval is not None:
        c1 = 0.5 * (interval[0] + interval[1])
        c2 = 0.5 * (interval[1] - interval[0])
        w = c2 ** (a + b + 1.0) * w
        x = c1 + c2 * x

    return x, w


def _jacpts_gw(n: int, a: float, b: float,
               ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Golub-Welsch eigenvalue method for Gauss-Jacobi nodes and weights.

    The Jacobi matrix has entries:
        diagonal:    aa_k = (b^2 - a^2) / ((2k+a+b-2)(2k+a+b))
        off-diagonal: bb_k = 2/(2k+a+b) * sqrt(k(k+a)(k+b)(k+a+b) /
                                                ((2k+a+b)^2-1))
    """
    ab = a + b

    # Diagonal entries
    ii = jnp.arange(2, n, dtype=jnp.float64)
    abi = 2.0 * ii + ab
    aa_mid = (b * b - a * a) / ((abi - 2.0) * abi)
    aa_first = jnp.array([(b - a) / (2.0 + ab)], dtype=jnp.float64)
    aa_last = jnp.array([(b * b - a * a) / ((2.0 * n - 2.0 + ab) * (2.0 * n + ab))],
                         dtype=jnp.float64)
    aa = jnp.concatenate([aa_first, aa_mid, aa_last])

    # Off-diagonal entries
    j = jnp.arange(1, n, dtype=jnp.float64)
    abj = 2.0 * j + ab
    bb_vals = 2.0 * jnp.sqrt(j * (j + a) * (j + b) * (j + ab)
                              / (abj * abj - 1.0)) / abj

    # Build tridiagonal Jacobi matrix
    T = jnp.diag(aa) + jnp.diag(bb_vals, 1) + jnp.diag(bb_vals, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    x = eigvals
    import jax.scipy.special as jsp
    w = eigvecs[0, :] ** 2 * 2.0 ** (ab + 1.0) * jnp.exp(
        jsp.gammaln(a + 1.0) + jsp.gammaln(b + 1.0)
        - jsp.gammaln(ab + 2.0))

    return x, w


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


def hermpts(n: int, kind: str = "phys",
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite quadrature nodes and weights.

    HERMPTS(N) returns N Hermite points in (-inf, inf) and the
    corresponding quadrature weights.

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 0).
    kind : {"phys", "prob"}
        "phys" for physicist's Hermite (weight exp(-x^2)),
        "prob" for probabilist's Hermite (weight exp(-x^2/2)).

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order.
    w : jnp.ndarray, shape (n,)
        Quadrature weights (sum = sqrt(pi) for "phys", sqrt(2*pi) for "prob").

    Provenance
    ----------
    MATLAB source : hermpts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW), Nick Hale (GLR),
        Alex Townsend, Thomas Trogdon & Sheehan Olver (ASY).
    Algorithm (this implementation): Golub-Welsch eigenvalue method [1].

    References
    ----------
    [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
        rules", Math. Comp. 23:221-230, 1969.
    [2] A. Townsend, T. Trogdon and S. Olver, "Fast computation of Gauss
        quadrature nodes and weights on the whole real line", IMA J. Numer.
        Anal. 36(1), 337-358, 2016.

    See Also
    --------
    legpts, lagpts
    """
    if kind not in ("phys", "prob"):
        raise ValueError(f"kind must be 'phys' or 'prob', got {kind!r}")

    if n == 0:
        return (jnp.array([], dtype=jnp.float64),
                jnp.array([], dtype=jnp.float64))

    if n == 1:
        x = jnp.array([0.0], dtype=jnp.float64)
        w = jnp.array([jnp.sqrt(jnp.pi)], dtype=jnp.float64)
        if kind == "prob":
            x = x * jnp.sqrt(2.0)
            w = w * jnp.sqrt(2.0)
        return x, w

    x, w = _hermpts_gw(n)

    if kind == "prob":
        x = x * jnp.sqrt(2.0)
        w = w * jnp.sqrt(2.0)

    return x, w


def _hermpts_gw(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Golub-Welsch eigenvalue method for Gauss-Hermite nodes and weights.

    The Jacobi matrix for physicist's Hermite polynomials has:
        diagonal: 0
        off-diagonal: beta_k = sqrt(k/2), k = 1, ..., n-1
    """
    i = jnp.arange(1, n, dtype=jnp.float64)
    beta = jnp.sqrt(0.5 * i)
    T = jnp.diag(beta, 1) + jnp.diag(beta, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    x = eigvals
    w = jnp.sqrt(jnp.pi) * eigvecs[0, :] ** 2

    # Normalise so that sum(w) = sqrt(pi)
    w = (jnp.sqrt(jnp.pi) / jnp.sum(w)) * w

    # Enforce symmetry
    m = n // 2
    x_lo = x[:m]
    w_lo = w[:m]

    if n % 2 == 1:
        x = jnp.concatenate([x_lo, jnp.array([0.0], dtype=jnp.float64),
                             -x_lo[::-1]])
        w_mid = jnp.sqrt(jnp.pi) - 2.0 * jnp.sum(w_lo)
        w = jnp.concatenate([w_lo, jnp.array([w_mid], dtype=jnp.float64),
                             w_lo[::-1]])
    else:
        x = jnp.concatenate([x_lo, -x_lo[::-1]])
        w = jnp.concatenate([w_lo, w_lo[::-1]])

    return x, w


# ---------------------------------------------------------------------------
# Gauss-Laguerre quadrature
# ---------------------------------------------------------------------------


def lagpts(n: int, alpha: float = 0.0,
           interval: tuple[float, float] | None = None,
           ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Laguerre quadrature nodes and weights.

    LAGPTS(N) returns N Laguerre points in (0, inf) and the
    corresponding quadrature weights for the weight function exp(-x).

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 0).
    alpha : float
        Generalised Laguerre parameter (>= 0). Weight is x^alpha * exp(-x).
    interval : (float, float) or None
        Semi-infinite domain [a, inf) or (-inf, b].  Default is [0, inf).

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order.
    w : jnp.ndarray, shape (n,)
        Quadrature weights (sum = Gamma(alpha+1)).

    Provenance
    ----------
    MATLAB source : lagpts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW), Peter Opsomer (RH/REC).
    Algorithm (this implementation): Golub-Welsch eigenvalue method [1].

    References
    ----------
    [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
        rules", Math. Comp. 23:221-230, 1969.

    See Also
    --------
    hermpts, legpts
    """
    if n == 0:
        return (jnp.array([], dtype=jnp.float64),
                jnp.array([], dtype=jnp.float64))

    x, w = _lagpts_gw(n, alpha)

    # Normalise so that sum(w) = Gamma(alpha+1)
    import jax.scipy.special as jsp
    w = (jnp.exp(jsp.gammaln(alpha + 1.0)) / jnp.sum(w)) * w

    # Rescale to non-standard interval
    if interval is not None:
        a_int, b_int = interval
        if jnp.isinf(b_int):
            x = x + a_int
            w = w * jnp.exp(-a_int)
        else:
            x = -x + b_int
            w = w * jnp.exp(b_int)

    return x, w


def _lagpts_gw(n: int, alpha: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Golub-Welsch eigenvalue method for Gauss-Laguerre nodes and weights.

    The Jacobi matrix for generalised Laguerre polynomials has:
        diagonal: alph_k = 2k - 1 + alpha, k = 1..n
        off-diagonal: beta_k = sqrt(k * (alpha + k)), k = 1..n-1
    """
    k = jnp.arange(1, n + 1, dtype=jnp.float64)
    diag_entries = 2.0 * k - 1.0 + alpha

    k_off = jnp.arange(1, n, dtype=jnp.float64)
    off_diag = jnp.sqrt(k_off * (alpha + k_off))

    T = jnp.diag(diag_entries) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    x = eigvals
    w = eigvecs[0, :] ** 2

    return x, w


# ---------------------------------------------------------------------------
# Ultraspherical (Gegenbauer) quadrature
# ---------------------------------------------------------------------------


def ultrapts(n: int, lam: float,
             interval: tuple[float, float] | None = None,
             ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Gegenbauer (ultraspherical) quadrature nodes and weights.

    Returns the N roots of the degree-N ultraspherical polynomial C_n^{lam}
    and the corresponding quadrature weights.  The ultraspherical weight
    function is w(x) = (1 - x^2)^{lam - 1/2}.

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 0).
    lam : float
        Ultraspherical parameter (> -0.5, lam != 0).
    interval : (float, float) or None
        If given, rescale nodes and weights to [A, B].

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order.
    w : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : ultrapts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW), Lourenco Peixoto (REC/ASY).
    Algorithm (this implementation): Golub-Welsch eigenvalue method [1].

    References
    ----------
    [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
        rules", Math. Comp. 23:221-230, 1969.
    [2] L. L. Peixoto, "Fast, accurate and convergent computation of
        Gauss-Gegenbauer quadrature nodes and weights", in preparation, 2019.

    See Also
    --------
    legpts, jacpts
    """
    if lam <= -0.5:
        raise ValueError("lambda must be greater than -0.5.")

    if n == 0:
        return (jnp.array([], dtype=jnp.float64),
                jnp.array([], dtype=jnp.float64))

    if n == 1:
        import jax.scipy.special as jsp
        x = jnp.array([0.0], dtype=jnp.float64)
        w0 = jnp.sqrt(jnp.pi) * jnp.exp(
            jsp.gammaln(lam + 0.5) - jsp.gammaln(lam + 1.0))
        w = jnp.array([w0], dtype=jnp.float64)
        if interval is not None:
            x, w = _rescale_ultra(x, w, interval, lam)
        return x, w

    # Special case: lam == 0.5 => Legendre
    if lam == 0.5:
        return legpts(n, interval=interval)

    x, w = _ultrapts_gw(n, lam)

    if interval is not None:
        x, w = _rescale_ultra(x, w, interval, lam)

    return x, w


def _rescale_ultra(x, w, interval, lam):
    """Rescale ultraspherical nodes and weights to [a, b]."""
    a, b = interval
    if a == -1.0 and b == 1.0:
        return x, w
    c2 = 0.5 * (b - a)
    c1 = 0.5 * (a + b)
    w = c2 ** (2.0 * lam) * w
    x = c1 + c2 * x
    return x, w


def _ultrapts_gw(n: int, lam: float,
                 ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Golub-Welsch eigenvalue method for Gauss-Gegenbauer nodes and weights.

    The Jacobi matrix for ultraspherical polynomials has:
        diagonal: 0
        off-diagonal: bb_k = 0.5 * sqrt(k*(k+2*lam-1)/((k+lam)(k+lam-1)))
    """
    i = jnp.arange(1, n, dtype=jnp.float64)
    bb = 0.5 * jnp.sqrt(i * (i + 2.0 * lam - 1.0)
                         / ((i + lam) * (i + lam - 1.0)))
    T = jnp.diag(bb, 1) + jnp.diag(bb, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)

    x = eigvals
    import jax.scipy.special as jsp
    w = eigvecs[0, :] ** 2 * (2.0 ** (2.0 * lam)
                               * jnp.exp(2.0 * jsp.gammaln(lam + 0.5)
                                         - jsp.gammaln(2.0 * lam + 1.0)))

    return x, w


# ---------------------------------------------------------------------------
# Gauss-Radau quadrature
# ---------------------------------------------------------------------------


def radaupts(n: int, alp: float = 0.0, bet: float = 0.0,
             ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Radau quadrature nodes and weights.

    RADAUPTS(N) returns N Gauss-Legendre-Radau nodes in [-1, 1)
    (the left endpoint -1 is included) and the corresponding weights.

    The approach uses the identity that the Gauss-Radau points are
    the roots of (1+x)*P^{(alp, bet+1)}_{n-1}(x).

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 1).
    alp, bet : float
        Jacobi parameters (both > -1). Default is 0 (Legendre).

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order (x[0] = -1).
    w : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : radaupts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Uses jacpts(n-1, alp, bet+1) and the identity
        from NIST (18.9.5) and (18.9.17).

    See Also
    --------
    lobpts, jacpts, legpts
    """
    if n == 1:
        import jax.scipy.special as jsp
        x = jnp.array([-1.0], dtype=jnp.float64)
        w = jnp.array([2.0 ** (1.0 + alp + bet)
                        * jnp.exp(jsp.gammaln(1.0 + alp)
                                  + jsp.gammaln(1.0 + bet)
                                  - jsp.gammaln(2.0 + alp + bet))],
                       dtype=jnp.float64)
        return x, w

    # Interior points from Jacobi (alp, bet+1) rule
    xi, wi = jacpts(n - 1, alp, bet + 1.0)

    # Nodes: prepend -1
    x = jnp.concatenate([jnp.array([-1.0], dtype=jnp.float64), xi])

    # Weights
    wi_radau = wi / (1.0 + xi)
    if alp == 0.0 and bet == 0.0:
        w0 = jnp.array([2.0 / (n * n)], dtype=jnp.float64)
    else:
        import jax.scipy.special as jsp
        w0 = jnp.array([2.0 ** (alp + bet + 1.0)
                         * jnp.exp(jsp.gammaln(bet + 2.0)
                                   + jsp.gammaln(float(n))
                                   - jsp.gammaln(float(n) + bet + 1.0))
                         * jnp.exp(jsp.gammaln(alp + float(n))
                                   + jsp.gammaln(bet + 1.0)
                                   - jsp.gammaln(alp + float(n) + bet + 1.0))
                         * (bet + 1.0)],
                        dtype=jnp.float64)
    w = jnp.concatenate([w0, wi_radau])

    return x, w


# ---------------------------------------------------------------------------
# Gauss-Lobatto quadrature
# ---------------------------------------------------------------------------


def lobpts(n: int, alp: float = 0.0, bet: float = 0.0,
           ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Lobatto quadrature nodes and weights.

    LOBPTS(N) returns N Gauss-Legendre-Lobatto nodes in [-1, 1]
    (both endpoints included) and the corresponding weights.

    The approach uses the identity that the interior Gauss-Lobatto
    points are the roots of P'_{n-1}(x) = roots of P^{(1,1)}_{n-2}(x).

    Parameters
    ----------
    n : int
        Number of quadrature points (must be >= 2).
    alp, bet : float
        Jacobi parameters (both > -1). Default is 0 (Legendre).

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Nodes in ascending order (x[0] = -1, x[-1] = 1).
    w : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : lobpts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Uses jacpts(n-2, alp+1, bet+1) and the identity
        from NIST (18.9.15) and (18.9.16).

    See Also
    --------
    radaupts, jacpts, legpts
    """
    if n < 2:
        raise ValueError("lobpts requires n >= 2.")

    if n == 2:
        x = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        import jax.scipy.special as jsp
        w_total = 2.0 ** (1.0 + alp + bet)
        w1 = w_total * jnp.exp(jsp.gammaln(bet + 1.0)
                                + jsp.gammaln(alp + 2.0)
                                - jsp.gammaln(alp + bet + 3.0)) * (alp + bet + 2.0)
        w2 = w_total * jnp.exp(jsp.gammaln(alp + 1.0)
                                + jsp.gammaln(bet + 2.0)
                                - jsp.gammaln(alp + bet + 3.0)) * (alp + bet + 2.0)
        # For Legendre (alp=bet=0), w = [1, 1], total interval = 2, so w1=w2=1.
        # MATLAB uses beta function: 2^(1+a+b)*[beta(b+1,a+2), beta(a+1,b+2)]
        # beta(p,q) = gamma(p)*gamma(q)/gamma(p+q)
        w1_val = 2.0 ** (1.0 + alp + bet) * jnp.exp(
            jsp.gammaln(bet + 1.0) + jsp.gammaln(alp + 2.0)
            - jsp.gammaln(alp + bet + 3.0))
        w2_val = 2.0 ** (1.0 + alp + bet) * jnp.exp(
            jsp.gammaln(alp + 1.0) + jsp.gammaln(bet + 2.0)
            - jsp.gammaln(alp + bet + 3.0))
        w = jnp.array([w1_val, w2_val], dtype=jnp.float64)
        return x, w

    # Interior points from Jacobi (alp+1, bet+1)
    xi, wi = jacpts(n - 2, alp + 1.0, bet + 1.0)

    # Nodes: prepend -1, append 1
    x = jnp.concatenate([jnp.array([-1.0], dtype=jnp.float64),
                         xi,
                         jnp.array([1.0], dtype=jnp.float64)])

    # Interior weights: wi / (1 - xi^2)
    w_inner = wi / (1.0 - xi ** 2)

    # Endpoint weights
    if alp == 0.0 and bet == 0.0:
        w_end = 2.0 / (n * (n - 1.0))
        w_left = jnp.array([w_end], dtype=jnp.float64)
        w_right = jnp.array([w_end], dtype=jnp.float64)
    else:
        import jax.scipy.special as jsp
        nf = float(n)
        # Gautschi's explicit formulas for Jacobi-Lobatto endpoint weights.
        w_left_val = (2.0 ** (1.0 + alp + bet)
                      * jnp.exp(jsp.gammaln(bet + 1.0)
                                + jsp.gammaln(alp + nf)
                                - jsp.gammaln(alp + bet + nf))
                      * jnp.exp(jsp.gammaln(bet + 2.0)
                                + jsp.gammaln(nf - 2.0 + 1.0)
                                - jsp.gammaln(bet + nf))
                      * (nf - 2.0))
        w_right_val = (2.0 ** (1.0 + alp + bet)
                       * jnp.exp(jsp.gammaln(alp + 1.0)
                                 + jsp.gammaln(bet + nf)
                                 - jsp.gammaln(alp + bet + nf))
                       * jnp.exp(jsp.gammaln(alp + 2.0)
                                 + jsp.gammaln(nf - 2.0 + 1.0)
                                 - jsp.gammaln(alp + nf))
                       * (nf - 2.0))
        w_left = jnp.array([w_left_val], dtype=jnp.float64)
        w_right = jnp.array([w_right_val], dtype=jnp.float64)

    w = jnp.concatenate([w_left, w_inner, w_right])

    return x, w


# ---------------------------------------------------------------------------
# Trigonometric (equispaced) points
# ---------------------------------------------------------------------------


def trigpts(n: int, interval: tuple[float, float] | None = None,
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equispaced (trigonometric) points and trapezoidal rule weights.

    TRIGPTS(N) returns N equispaced points in [-1, 1) and the
    corresponding trapezoidal-rule weights.

    Parameters
    ----------
    n : int
        Number of points (must be >= 0).
    interval : (float, float) or None
        If given, map to [a, b).

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Equispaced points.
    w : jnp.ndarray, shape (n,)
        Trapezoidal rule weights (all equal to 2/n on [-1,1)).

    Provenance
    ----------
    MATLAB source : trigpts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpts, legpts
    """
    if n <= 0:
        return (jnp.array([], dtype=jnp.float64),
                jnp.array([], dtype=jnp.float64))

    # Equispaced points in [-1, 1) (matching MATLAB's linspace(-pi,pi,n+1)/pi)
    x = jnp.linspace(-1.0, 1.0, n + 1, dtype=jnp.float64)
    # Enforce symmetry: x = (x - x[::-1]) / 2
    x = (x - x[::-1]) / 2.0
    x = x[:-1]  # Remove last point (it is +1, which is excluded)

    # Trapezoidal weights: 2/n on [-1, 1)
    w = jnp.full(n, 2.0 / n, dtype=jnp.float64)

    if interval is not None:
        a, b = interval
        dab = b - a
        x = dab * x / 2.0 + (a + b) / 2.0
        w = w * dab / 2.0

    return x, w
