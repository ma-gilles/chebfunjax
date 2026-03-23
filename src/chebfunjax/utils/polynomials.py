"""Classical orthogonal polynomial utilities.

Chebyshev, Legendre, Jacobi, ultraspherical (Gegenbauer), Hermite, and Laguerre
polynomials: Chebyshev coefficient representations and pointwise evaluation.

Translated from MATLAB Chebfun (commit 7574c77): chebpoly.m, legpoly.m, jacpoly.m,
ultrapoly.m, hermpoly.m, lagpoly.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.utils.transforms import jac2cheb, leg2cheb

# ===========================================================================
# Chebyshev polynomials
# ===========================================================================


def chebpoly(n: int, kind: int = 1) -> jnp.ndarray:
    """Chebyshev coefficients of the Chebyshev polynomial T_n or U_n.

    Returns the Chebyshev (first-kind) coefficient vector for the degree-n
    Chebyshev polynomial of the specified kind on [-1, 1].

    For kind=1 (T_n): the coefficient vector is the unit vector e_{n+1}
    (all zeros except a 1 at position n).

    For kind=2 (U_n): the Chebyshev T-coefficients of the second-kind
    polynomial U_n are computed via the relation
        U_n(x) = sum_{k=0}^{n} c_k T_k(x),
    where c_k can be obtained from the three-term recurrence:
        U_0 = T_0, U_1 = 2*T_1,
        U_n = 2*T_n + U_{n-2}  (for n >= 2, same-parity terms).

    Parameters
    ----------
    n : int
        Polynomial degree (non-negative integer).
    kind : {1, 2}, default 1
        1 for Chebyshev polynomials of the first kind T_n,
        2 for Chebyshev polynomials of the second kind U_n.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n + 1,)
        Chebyshev (first-kind) series coefficients.

    Examples
    --------
    >>> chebpoly(0)
    Array([1.], dtype=float64)
    >>> chebpoly(3)
    Array([0., 0., 0., 1.], dtype=float64)
    >>> chebpoly(2, kind=2)  # U_2 = 4x^2 - 1 = 2*T_2 + T_0
    Array([1., 0., 2.], dtype=float64)

    Provenance
    ----------
    MATLAB source : chebpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    legpoly, jacpoly, ultrapoly
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )
    if kind not in (1, 2):
        raise ValueError(
            f"kind must be 1 (first kind, T_n) or 2 (second kind, U_n), got {kind}."
        )

    if kind == 1:
        # T_n has Chebyshev-T expansion: coefficient 1 at position n.
        c = jnp.zeros(n + 1, dtype=jnp.float64)
        c = c.at[n].set(1.0)
        return c

    # kind == 2: U_n in terms of T_k.
    # U_n(x) = sum_{k} c_k T_k(x), where the sum runs over k with same
    # parity as n, from 0 (or 1) up to n, and c_k = 2 for k > 0, c_0 = 1.
    # Derivation: U_n = 2*T_n + U_{n-2}, with U_0 = T_0, U_1 = 2*T_1.
    c = jnp.zeros(n + 1, dtype=jnp.float64)
    # Fill same-parity positions: n, n-2, n-4, ..., 0 or 1
    # All get value 2, except position 0 gets value 1.
    indices = jnp.arange(n, -1, -2)
    c = c.at[indices].set(2.0)
    # If n is even, position 0 should be 1 (not 2).
    if n % 2 == 0:
        c = c.at[0].set(1.0)
    return c


# ===========================================================================
# Legendre polynomials
# ===========================================================================


def legpoly(n: int, *, normalize: bool = False) -> jnp.ndarray:
    """Chebyshev coefficients of the Legendre polynomial P_n.

    Returns the Chebyshev (first-kind) coefficient vector for the degree-n
    Legendre polynomial on [-1, 1], computed via leg2cheb.

    Parameters
    ----------
    n : int
        Polynomial degree (non-negative integer).
    normalize : bool, default False
        If True, normalize so that the integral of P_n^2 over [-1, 1] equals 1
        (orthonormal normalization). The standard normalization has P_n(1) = 1.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n + 1,)
        Chebyshev (first-kind) series coefficients.

    Examples
    --------
    >>> legpoly(0)
    Array([1.], dtype=float64)
    >>> legpoly(1)  # P_1 = x = T_1
    Array([0., 1.], dtype=float64)

    Provenance
    ----------
    MATLAB source : legpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpoly, jacpoly, ultrapoly, cheb2leg, leg2cheb
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )

    # Legendre coefficients: unit vector e_{n+1}
    c_leg = jnp.zeros(n + 1, dtype=jnp.float64)
    c_leg = c_leg.at[n].set(1.0)

    # Convert to Chebyshev coefficients
    return leg2cheb(c_leg, normalize=normalize)


# ===========================================================================
# Jacobi polynomials
# ===========================================================================


def jacpoly(n: int, alpha: float, beta: float) -> jnp.ndarray:
    """Chebyshev coefficients of the Jacobi polynomial P_n^{(alpha, beta)}.

    Returns the Chebyshev (first-kind) coefficient vector for the degree-n
    Jacobi polynomial on [-1, 1], computed via jac2cheb.

    The Jacobi weight function is w(x) = (1-x)^alpha * (1+x)^beta.
    Normalization is consistent with NIST DLMF 18: P_n^{(a,b)}(1) = C(n+a, n),
    i.e. the rising factorial (alpha+1)_n / n!.

    Parameters
    ----------
    n : int
        Polynomial degree (non-negative integer).
    alpha : float
        Jacobi parameter alpha (exponent for 1-x). Must satisfy alpha > -1.
    beta : float
        Jacobi parameter beta (exponent for 1+x). Must satisfy beta > -1.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n + 1,)
        Chebyshev (first-kind) series coefficients.

    References
    ----------
    .. [1] F.W.J. Olver et al., editors. NIST Handbook of Mathematical
       Functions. Cambridge University Press, New York, NY, 2010. Sec. 18.

    Provenance
    ----------
    MATLAB source : jacpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpoly, legpoly, ultrapoly, jac2cheb, cheb2jac
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )

    # Jacobi coefficients: unit vector e_{n+1}
    c_jac = jnp.zeros(n + 1, dtype=jnp.float64)
    c_jac = c_jac.at[n].set(1.0)

    # Convert to Chebyshev coefficients
    return jac2cheb(c_jac, alpha, beta)


# ===========================================================================
# Ultraspherical (Gegenbauer) polynomials
# ===========================================================================


def ultrapoly(n: int, lam: float) -> jnp.ndarray:
    """Chebyshev coefficients of the ultraspherical polynomial C_n^{(lam)}.

    Returns the Chebyshev (first-kind) coefficient vector for the degree-n
    ultraspherical (Gegenbauer) polynomial on [-1, 1].

    The ultraspherical weight function is w(x) = (1 - x^2)^(lam - 1/2).
    Normalization follows NIST DLMF 18: C_n^{(lam)}(1) = (2*lam)_n / n!.

    Special cases:
        lam = 1/2  ->  Legendre polynomial P_n
        lam = 1    ->  Chebyshev polynomial of the second kind U_n

    Parameters
    ----------
    n : int
        Polynomial degree (non-negative integer).
    lam : float
        Ultraspherical parameter lambda. Must be positive.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n + 1,)
        Chebyshev (first-kind) series coefficients.

    References
    ----------
    .. [1] F.W.J. Olver et al., editors. NIST Handbook of Mathematical
       Functions. Cambridge University Press, New York, NY, 2010. Sec. 18.

    Provenance
    ----------
    MATLAB source : ultrapoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpoly, legpoly, jacpoly
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )
    if lam <= 0:
        raise ValueError(
            f"Ultraspherical parameter lam must be positive, got {lam}."
        )

    # Special case: lam = 0.5 is Legendre
    if lam == 0.5:
        return legpoly(n)

    # Special case: lam = 1 is Chebyshev U_n
    if lam == 1.0:
        return chebpoly(n, kind=2)

    # General case: C_n^(lam) = scl_n * P_n^(lam-1/2, lam-1/2)
    # where scl_n = Gamma(lam+1/2)/Gamma(2*lam) * Gamma(2*lam+n)/Gamma(lam+n+1/2)
    # We compute the scaling factors and then use jac2cheb.

    # Build the diagonal scaling: for k = 0, 1, ..., n:
    # scl_k = Gamma(lam+0.5)/Gamma(2*lam) * Gamma(2*lam+k)/Gamma(lam+k+0.5)
    # Use log-gamma to avoid overflow.
    nn = jnp.arange(n + 1, dtype=jnp.float64)
    log_scl = (
        jax.scipy.special.gammaln(lam + 0.5)
        - jax.scipy.special.gammaln(2.0 * lam)
        + jax.scipy.special.gammaln(2.0 * lam + nn)
        - jax.scipy.special.gammaln(lam + nn + 0.5)
    )
    scl = jnp.exp(log_scl)

    # Jacobi coefficients for C_n^(lam): c_jac = scl * e_n
    # (scaled identity diagonal, then pick column n)
    c_jac = jnp.zeros(n + 1, dtype=jnp.float64)
    c_jac = c_jac.at[n].set(scl[n])

    # Convert Jacobi(lam-0.5, lam-0.5) to Chebyshev
    a = lam - 0.5
    return jac2cheb(c_jac, a, a)


# ===========================================================================
# Hermite polynomials (evaluation via recurrence)
# ===========================================================================


def hermeval(x: jnp.ndarray, n: int, kind: str = "phys") -> jnp.ndarray:
    """Evaluate the Hermite polynomial H_n(x) or He_n(x) at given points.

    Computes the degree-n Hermite polynomial using the three-term recurrence
    relation. Two normalizations are supported:

    - ``'phys'`` (physicist's): weight exp(-x^2), H_0=1, H_1=2x,
      H_n = 2x*H_{n-1} - 2(n-1)*H_{n-2}.
    - ``'prob'`` (probabilist's): weight exp(-x^2/2), He_0=1, He_1=x,
      He_n = x*He_{n-1} - (n-1)*He_{n-2}. These are monic.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points (any shape).
    n : int
        Polynomial degree (non-negative integer).
    kind : {'phys', 'prob'}, default 'phys'
        Physicist's or probabilist's Hermite polynomials.

    Returns
    -------
    values : jnp.ndarray, same shape as x
        H_n(x) or He_n(x).

    Provenance
    ----------
    MATLAB source : hermpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    lageval, chebpoly, legpoly
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )
    if kind not in ("phys", "prob"):
        raise ValueError(
            f"kind must be 'phys' or 'prob', got {kind!r}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)

    if n == 0:
        return jnp.ones_like(x)

    if kind == "prob":
        H_prev = jnp.ones_like(x)        # He_0
        H_curr = x                        # He_1
        for k in range(2, n + 1):
            H_next = x * H_curr - (k - 1) * H_prev
            H_prev = H_curr
            H_curr = H_next
    else:  # 'phys'
        H_prev = jnp.ones_like(x)        # H_0
        H_curr = 2.0 * x                 # H_1
        for k in range(2, n + 1):
            H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
            H_prev = H_curr
            H_curr = H_next

    return H_curr


# ===========================================================================
# Laguerre polynomials (evaluation via recurrence)
# ===========================================================================


def lageval(x: jnp.ndarray, n: int, alpha: float = 0.0) -> jnp.ndarray:
    """Evaluate the (generalized) Laguerre polynomial L_n^{(alpha)}(x).

    Computes the degree-n Laguerre polynomial using the three-term recurrence:
        L_0 = 1, L_1 = 1 + alpha - x,
        L_k = ((2 + (alpha-1-x)/k) * L_{k-1} - (1 + (alpha-1)/k) * L_{k-2}).

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points (any shape). Typically x >= 0.
    n : int
        Polynomial degree (non-negative integer).
    alpha : float, default 0.0
        Generalized Laguerre parameter. Must be real.

    Returns
    -------
    values : jnp.ndarray, same shape as x
        L_n^{(alpha)}(x).

    Provenance
    ----------
    MATLAB source : lagpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    hermeval, chebpoly, legpoly
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)

    if n == 0:
        return jnp.ones_like(x)

    L_prev = jnp.ones_like(x)                    # L_0
    L_curr = 1.0 + alpha - x                     # L_1

    for k in range(2, n + 1):
        L_next = (
            (2.0 + (alpha - 1.0 - x) / k) * L_curr
            - (1.0 + (alpha - 1.0) / k) * L_prev
        )
        L_prev = L_curr
        L_curr = L_next

    return L_curr


# ===========================================================================
# Pointwise evaluation helpers for bounded-domain polynomials
# ===========================================================================


def chebeval(x: jnp.ndarray, n: int, kind: int = 1) -> jnp.ndarray:
    """Evaluate the Chebyshev polynomial T_n(x) or U_n(x) at given points.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points in [-1, 1] (any shape).
    n : int
        Polynomial degree (non-negative integer).
    kind : {1, 2}, default 1
        1 for first kind T_n, 2 for second kind U_n.

    Returns
    -------
    values : jnp.ndarray, same shape as x
        T_n(x) or U_n(x).

    Provenance
    ----------
    MATLAB source : chebpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpoly, legeval, jaceval
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )
    if kind not in (1, 2):
        raise ValueError(
            f"kind must be 1 (first kind, T_n) or 2 (second kind, U_n), got {kind}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)

    if kind == 1:
        # T_n(x) = cos(n * arccos(x))
        return jnp.cos(n * jnp.arccos(x))
    else:
        # U_n(x) = sin((n+1) * arccos(x)) / sin(arccos(x))
        theta = jnp.arccos(x)
        # Handle x = +/- 1 carefully via the limit:
        # U_n(1) = n+1, U_n(-1) = (-1)^n * (n+1)
        sin_theta = jnp.sin(theta)
        val = jnp.where(
            sin_theta == 0.0,
            jnp.where(x > 0, float(n + 1), ((-1.0) ** n) * (n + 1)),
            jnp.sin((n + 1) * theta) / sin_theta,
        )
        return val


def legeval(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Evaluate the Legendre polynomial P_n(x) at given points.

    Uses the three-term recurrence relation.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points in [-1, 1] (any shape).
    n : int
        Polynomial degree (non-negative integer).

    Returns
    -------
    values : jnp.ndarray, same shape as x
        P_n(x).

    Provenance
    ----------
    MATLAB source : legpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    legpoly, chebeval, jaceval
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)

    if n == 0:
        return jnp.ones_like(x)

    P_prev = jnp.ones_like(x)   # P_0
    P_curr = x                    # P_1
    for k in range(1, n):
        P_next = ((2 * k + 1) * x * P_curr - k * P_prev) / (k + 1)
        P_prev = P_curr
        P_curr = P_next

    return P_curr


def jaceval(x: jnp.ndarray, n: int, alpha: float, beta: float) -> jnp.ndarray:
    """Evaluate the Jacobi polynomial P_n^{(alpha, beta)}(x) at given points.

    Uses the standard three-term recurrence relation (NIST DLMF 18.9.2).

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points in [-1, 1] (any shape).
    n : int
        Polynomial degree (non-negative integer).
    alpha : float
        Jacobi parameter alpha. Must satisfy alpha > -1.
    beta : float
        Jacobi parameter beta. Must satisfy beta > -1.

    Returns
    -------
    values : jnp.ndarray, same shape as x
        P_n^{(alpha, beta)}(x).

    References
    ----------
    .. [1] F.W.J. Olver et al., editors. NIST Handbook of Mathematical
       Functions. Cambridge University Press, New York, NY, 2010. Sec. 18.

    Provenance
    ----------
    MATLAB source : jacpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    jacpoly, chebeval, legeval
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)
    a, b = alpha, beta
    apb = a + b

    if n == 0:
        return jnp.ones_like(x)

    P_prev = jnp.ones_like(x)
    P_curr = 0.5 * (2.0 * (a + 1.0) + (apb + 2.0) * (x - 1.0))

    aa = a * a
    bb = b * b
    for k in range(2, n + 1):
        k2 = 2 * k
        k2apb = k2 + apb
        q1 = k2 * (k + apb) * (k2apb - 2)
        q2 = (k2apb - 1) * (aa - bb)
        q3 = (k2apb - 2) * (k2apb - 1) * k2apb
        q4 = 2 * (k + a - 1) * (k + b - 1) * k2apb
        P_next = ((q2 + q3 * x) * P_curr - q4 * P_prev) / q1
        P_prev = P_curr
        P_curr = P_next

    return P_curr


def ultraeval(x: jnp.ndarray, n: int, lam: float) -> jnp.ndarray:
    """Evaluate the ultraspherical polynomial C_n^{(lam)}(x) at given points.

    Uses the three-term recurrence:
        C_0 = 1, C_1 = 2*lam*x,
        C_k = (2*(k-1+lam)*x*C_{k-1} - (k+2*lam-2)*C_{k-2}) / k.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points in [-1, 1] (any shape).
    n : int
        Polynomial degree (non-negative integer).
    lam : float
        Ultraspherical parameter lambda. Must be positive.

    Returns
    -------
    values : jnp.ndarray, same shape as x
        C_n^{(lam)}(x).

    References
    ----------
    .. [1] F.W.J. Olver et al., editors. NIST Handbook of Mathematical
       Functions. Cambridge University Press, New York, NY, 2010. Sec. 18.

    Provenance
    ----------
    MATLAB source : ultrapoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ultrapoly, chebeval, legeval, jaceval
    """
    if n < 0:
        raise ValueError(
            f"Polynomial degree n must be a non-negative integer, got {n}."
        )
    if lam <= 0:
        raise ValueError(
            f"Ultraspherical parameter lam must be positive, got {lam}."
        )

    x = jnp.asarray(x, dtype=jnp.float64)

    if n == 0:
        return jnp.ones_like(x)

    C_prev = jnp.ones_like(x)          # C_0
    C_curr = 2.0 * lam * x             # C_1

    for k in range(2, n + 1):
        C_next = (
            2.0 * (k - 1 + lam) * x * C_curr - (k + 2 * lam - 2) * C_prev
        ) / k
        C_prev = C_curr
        C_curr = C_next

    return C_curr
