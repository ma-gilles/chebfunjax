"""Quadrature points and weights.

Translated from MATLAB Chebfun (commit 7574c77): chebpts.m and related functions.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from math import isinf

import jax
import jax.numpy as jnp

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer


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
        # Roots of T_n.  MATLAB @chebtech1/chebpts.m uses the sine form
        # x = sin(pi*(-n+1:2:n-1)/(2n)) rather than cos((2k-1)pi/(2n)):
        # the sine construction is exactly antisymmetric (centre node is a
        # bit-exact 0 for odd n) and pairs small-argument sines against the
        # symmetric endpoints, matching MATLAB to the last bit.
        k = jnp.arange(-n + 1, n, 2, dtype=jnp.float64)
        x = jnp.sin(jnp.pi * k / (2 * n))
    elif kind == 2:
        # Extrema of T_{n-1}.  MATLAB @chebtech2/chebpts.m uses the sine form
        # x = sin(pi*(-m:2:m)/(2m)), m = n-1, instead of cos(k*pi/(n-1)):
        # "Use of sine enforces symmetry" (antisymmetry residual and centre
        # node are bit-exact 0, not ~1e-16 as the cosine form gives).
        m = n - 1
        k = jnp.arange(-m, m + 1, 2, dtype=jnp.float64)
        x = jnp.sin(jnp.pi * k / (2 * m))
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    return x


def chebpts_ab(n: int, a: float, b: float, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points on the interval [a, b]."""
    x = chebpts(n, kind)
    return 0.5 * ((b - a) * x + (b + a))


def chebweights(n: int, kind: int = 2) -> jnp.ndarray:
    """Quadrature weights for Chebyshev points of the 1st or 2nd kind.

    For the *plain* integral of a function over [-1, 1] (with respect to
    ``dx``):

    * ``kind=2`` gives the Clenshaw-Curtis weights on 2nd-kind points.
    * ``kind=1`` gives Fejér's first-rule weights on 1st-kind points.

    Both mirror MATLAB ``@chebtech2/quadwts.m`` and ``@chebtech1/quadwts.m``
    exactly: the weights sum to 2 and integrate polynomials up to the
    appropriate degree to machine precision.

    For the Gauss-Chebyshev rule (weights ``pi/n`` for the
    ``1/sqrt(1-x^2)``-weighted integral) use :func:`gauss_cheb_weights`.

    Parameters
    ----------
    n : int
        Number of points.
    kind : {1, 2}
        1 for Fejér-1 weights, 2 for Clenshaw-Curtis weights.

    Returns
    -------
    w : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : @chebtech1/quadwts.m, @chebtech2/quadwts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.  Variant of Waldvogel's algorithm due
        to Nick Hale.
    Algorithm:
        [1] Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
            quadrature rules", BIT Numerical Mathematics, 46, 2006.

    See Also
    --------
    chebpts, gauss_cheb_weights
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)

    if kind == 1:
        # Fejér's first rule via FFT (Waldvogel / Nick Hale)
        return _fejer_first_weights(n)
    elif kind == 2:
        # Clenshaw-Curtis weights via FFT (Waldvogel's algorithm)
        return _clenshaw_curtis_weights(n)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def gauss_cheb_weights(n: int) -> jnp.ndarray:
    """Gauss-Chebyshev (1st-kind) quadrature weights ``w_k = pi/n``.

    These integrate ``f(x) / sqrt(1 - x^2)`` over [-1, 1] exactly for
    polynomials ``f`` of degree ``<= 2n - 1`` on the 1st-kind Chebyshev
    nodes.  This is the classical Gauss-Chebyshev rule and is distinct from
    Fejér's first rule (:func:`chebweights` with ``kind=1``), which targets
    the plain ``dx`` integral.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    w : jnp.ndarray, shape (n,)
        All entries equal to ``pi / n`` (empty for ``n == 0``).

    See Also
    --------
    chebpts, chebweights
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    return jnp.full(n, jnp.pi / n, dtype=jnp.float64)


def _fejer_first_weights(n: int) -> jnp.ndarray:
    """Fejér's first-rule quadrature weights for n 1st-kind Chebyshev points.

    Ported from MATLAB ``@chebtech1/quadwts.m`` (a variant of Waldvogel's
    FFT algorithm due to Nick Hale).  The exact Chebyshev moments
    ``m_k = 2/(1-k^2)`` (even ``k``) are mirrored, rotated by
    ``exp(1i*(0:n-1)*pi/n)`` to account for the half-integer angles of the
    1st-kind grid, and inverted by a single ``ifft``.

    Weights are returned in ascending ``x`` order (matching
    ``chebpts(n, kind=1)``); they sum to 2 and integrate polynomials of
    degree ``<= n-1`` exactly.
    """
    # Moments: m = 2./[1, 1-(2:2:(n-1)).^2]  (exact integrals of even T_k).
    even = jnp.arange(2, n, 2, dtype=jnp.float64)  # 2, 4, ..., up to n-1
    m = jnp.concatenate([jnp.array([2.0], dtype=jnp.float64),
                         2.0 / (1.0 - even ** 2)])

    # Mirror the moment vector for the ifft (odd/even n handled separately,
    # matching MATLAB's [m, -m(...)] / [m, 0, -m(...)] construction).
    if n % 2 == 1:
        c = jnp.concatenate([m, -m[(n + 1) // 2 - 1:0:-1]])
    else:
        c = jnp.concatenate([m, jnp.array([0.0], dtype=jnp.float64),
                             -m[n // 2 - 1:0:-1]])

    # Rotation (weight) vector for the half-integer 1st-kind angles.
    v = jnp.exp(1j * jnp.arange(n, dtype=jnp.float64) * jnp.pi / n)
    # The source complex inverse DFT is conj(fft(conj(z)))/n. Software
    # IEEE division preserves its 1/n scale when JIT would replace division
    # with a reciprocal multiply and introduce an extra rounding. No weight
    # normalization is imposed; FFT butterfly arithmetic stays in JAX.
    transformed = jnp.conj(jnp.fft.fft(jnp.conj(c * v)))
    return _divide_binary64_by_positive_integer(jnp.real(transformed), n)


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


def _bary_weights_gauss(x, flip: bool = False):
    """Normalized barycentric weights for a sorted Gauss-type node set.

    Log-scaled products (safe for unbounded Hermite/Laguerre nodes),
    normalized to max |v| = 1 with MATLAB's sign convention:
    v_k ~ (-1)^k, or (-1)^(k+1) for Lobatto/Radau (``flip=True``), where
    MATLAB keeps the interior Gauss alternation phase.  Added by
    Claude Fable 5 (barycentric third output, MISSING_FEATURES #9).
    """
    import numpy as _onp
    xv = _onp.asarray(x, dtype=float)
    m = len(xv)
    logv = _onp.zeros(m)
    for k in range(m):
        d = xv[k] - _onp.delete(xv, k)
        logv[k] = -_onp.sum(_onp.log(_onp.abs(d)))
    logv -= _onp.max(logv)
    v = _onp.exp(logv) * (-1.0) ** (_onp.arange(m) + (1 if flip else 0))
    return jnp.asarray(v / _onp.max(_onp.abs(v)))


def _legpts_core(n: int, interval: tuple[float, float] | None = None,
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
    Algorithm: source REC for ``n < 100`` and source ASY for ``n >= 100``.
    The ASY implementation returns source angles directly for the optional
    ``newtheta=True`` result; those angles are not reconstructed with acos.

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

    # Golub-Welsch builds an n x n matrix (O(n^2) memory) — it OOMs /
    # hangs for large n. Above a threshold, use an O(n)-memory
    # vectorized Newton iteration on the Legendre recurrence instead.
    # Added by Claude Opus 4.8 (task #10). The two agree to ~1e-13.
    if n < 100:
        from chebfunjax.utils.legendre_rec import _legpts_rec

        x, w, _ = _legpts_rec(n)
    elif n > _LEGPTS_NEWTON_THRESHOLD:
        x, w = _legpts_newton(n)
    else:
        x, w = _legpts_gw(n)

    if interval is not None:
        a, b = interval
        dab = b - a
        x = (x + 1.0) / 2.0 * dab + a
        w = dab * w / 2.0

    return x, w


_LEGPTS_NEWTON_THRESHOLD = 200


def _legpts_newton(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """O(n)-memory Gauss-Legendre nodes/weights by vectorized Newton.

    Computes the non-negative roots of the Legendre polynomial P_n by
    Newton's method with a Tricomi/Gatteschi initial guess, evaluating
    P_n and P_n' via the three-term recurrence vectorized over all
    roots.  O(n) memory (no dense matrix), so it scales to very large n
    where the Golub-Welsch eigensolve is infeasible.  Matches
    Golub-Welsch / numpy.leggauss to ~1e-13.

    Added by Claude Opus 4.8 (task #10 — legpts was O(n^2) only).
    """
    import numpy as _np

    m = (n + 1) // 2  # number of non-negative roots
    k = _np.arange(1, m + 1, dtype=_np.float64)
    # Tricomi initial guess (roots near x = 1, descending)
    theta = _np.pi * (4.0 * k - 1.0) / (4.0 * n + 2.0)
    x = (1.0 - (n - 1.0) / (8.0 * n ** 3)
         - 1.0 / (384.0 * n ** 4) * (39.0 - 28.0 / _np.sin(theta) ** 2)
         ) * _np.cos(theta)

    pnp = _np.zeros_like(x)
    for _ in range(100):
        p0 = _np.ones_like(x)
        p1 = x.copy()
        for j in range(2, n + 1):
            p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        pn = p1
        pnp = n * (x * p1 - p0) / (x * x - 1.0)
        dx = pn / pnp
        x = x - dx
        if _np.max(_np.abs(dx)) < 1e-15:
            break

    w = 2.0 / ((1.0 - x * x) * pnp ** 2)

    # x descends from near 1; the smallest is ~0 for odd n.
    if n % 2 == 1:
        # last root is the centre x = 0
        pos = x[:-1]         # strictly positive roots (descending)
        posw = w[:-1]
        w_mid = w[-1]
        nodes = _np.concatenate([-pos[::-1],
                                 _np.array([0.0]), pos])
        weights = _np.concatenate([posw[::-1],
                                   _np.array([w_mid]), posw])
    else:
        nodes = _np.concatenate([-x[::-1], x])
        weights = _np.concatenate([w[::-1], w])

    # ascending order
    order = _np.argsort(nodes)
    return (jnp.asarray(nodes[order], dtype=jnp.float64),
            jnp.asarray(weights[order], dtype=jnp.float64))


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


def _jacpts_core(n: int, a: float, b: float,
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
    if abs(ab + 1.0) < 1e-14:
        # Removable 0/0 at j=1 when a+b = -1 (e.g. the Chebyshev weight
        # a=b=-1/2): (j+ab)/(2j+ab-1) -> 1, so
        # beta_1 = 2*sqrt((1+a)(1+b)/(3+ab)) / (2+ab).
        b1 = 2.0 * jnp.sqrt((1.0 + a) * (1.0 + b) / (3.0 + ab)) / (2.0 + ab)
        bb_vals = bb_vals.at[0].set(b1)

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


def _hermpts_core(n: int, kind: str = "phys",
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


def _lagpts_core(n: int, alpha: float = 0.0,
                 interval: tuple[float, float] | None = None,
                 method: str = 'gw') -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a Gauss--Laguerre rule using REC, GW, GLR or bounded RH.

    Provenance
    ----------
    MATLAB source : ``lagpts.m`` (``lag_rec``, ``gw``, ``glr``, ``newton``)
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``

    GLR requires alpha=0. RH requires concrete alpha in {0,-1/2,+1/2} and n>=3000.
    Other RH variants, EXP, and underflow-truncated RECW/RHW are unported.
    """
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return empty, empty

    if method == 'rec':
        from chebfunjax.utils.laguerre_rec import _lag_rec
        x, w = _lag_rec(n, alpha)
    elif method == 'glr':
        if isinstance(alpha, jax.core.Tracer):
            raise ValueError("lagpts: GLR requires a concrete alpha=0")
        if alpha != 0:
            raise ValueError("lagpts: GLR method not supported for nonzero alpha")
        from chebfunjax.utils.laguerre_glr import _laguerre_glr
        x, w = _laguerre_glr(n)
    elif method == 'rh':
        if n < 3000 or isinstance(alpha, jax.core.Tracer) or alpha not in (0, -0.5, 0.5):
            raise NotImplementedError("lagpts: this source RH variant is not yet supported")
        if alpha == 0:
            from chebfunjax.utils.laguerre_rh import _laguerre_rh_alpha0
            x, w = _laguerre_rh_alpha0(n)
        else:
            from chebfunjax.utils.laguerre_rh_half import _laguerre_rh_half
            x, w = _laguerre_rh_half(n, alpha)
    elif method == 'gw':
        x, w = _lagpts_gw(n, alpha)
    else:
        raise ValueError(f"_lagpts_core: unsupported method {method!r}")

    import jax.scipy.special as jsp
    w = (jnp.exp(jsp.gammaln(alpha + 1.0)) / jnp.sum(w)) * w

    if interval is not None:
        a_int, b_int = interval
        if jnp.isinf(b_int):
            x, w = x + a_int, w * jnp.exp(-a_int)
        else:
            x, w = -x + b_int, w * jnp.exp(b_int)
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


def _ultrapts_core(n: int, lam: float,
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


def _radaupts_core(n: int, alp: float = 0.0, bet: float = 0.0,
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
        # MATLAB radaupts.m: w(1) = 2^(a+b+1)*beta(b+1,n)*beta(a+n,b+1)
        #                            *(b+1)
        # (the previous code used Gamma(b+2) in the first beta, an extra
        # (b+1) factor -- found in the Fable 5 audit via moment matching)
        w0 = jnp.array([2.0 ** (alp + bet + 1.0)
                         * jnp.exp(jsp.gammaln(bet + 1.0)
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


def _lobpts_core(n: int, alp: float = 0.0, bet: float = 0.0,
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
        # MATLAB uses beta function: 2^(1+a+b)*[beta(b+1,a+2), beta(a+1,b+2)]
        # beta(p,q) = gamma(p)*gamma(q)/gamma(p+q)
        w1 = 2.0 ** (1.0 + alp + bet) * jnp.exp(
            jsp.gammaln(bet + 1.0) + jsp.gammaln(alp + 2.0)
            - jsp.gammaln(alp + bet + 3.0))
        w2 = 2.0 ** (1.0 + alp + bet) * jnp.exp(
            jsp.gammaln(alp + 1.0) + jsp.gammaln(bet + 2.0)
            - jsp.gammaln(alp + bet + 3.0))
        w = jnp.array([w1, w2], dtype=jnp.float64)
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
        # MATLAB lobpts.m endpoint weights:
        #   w(1) = 2^(1+a+b)*beta(b+1, a+n)*beta(b+2, n-2)*(n-2)
        #   w(n) = 2^(1+a+b)*beta(a+1, b+n)*beta(a+2, n-2)*(n-2)
        # (the previous derivation was off by (a+b+n)*(n-2) -- found in
        # the Fable 5 audit via moment matching)
        w_left_val = (2.0 ** (1.0 + alp + bet)
                      * jnp.exp(jsp.gammaln(bet + 1.0)
                                + jsp.gammaln(alp + nf)
                                - jsp.gammaln(alp + bet + nf + 1.0))
                      * jnp.exp(jsp.gammaln(bet + 2.0)
                                + jsp.gammaln(nf - 2.0)
                                - jsp.gammaln(bet + nf))
                      * (nf - 2.0))
        w_right_val = (2.0 ** (1.0 + alp + bet)
                       * jnp.exp(jsp.gammaln(alp + 1.0)
                                 + jsp.gammaln(bet + nf)
                                 - jsp.gammaln(alp + bet + nf + 1.0))
                       * jnp.exp(jsp.gammaln(alp + 2.0)
                                 + jsp.gammaln(nf - 2.0)
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

    from chebfunjax.utils._trigpts import global_trigpts_nodes, map_global_nodes
    x = global_trigpts_nodes(n)

    # Trapezoidal weights: 2/n on [-1, 1)
    w = jnp.full(n, 2.0 / n, dtype=jnp.float64)

    if interval is not None:
        a, b = interval
        dab = b - a
        x = map_global_nodes(x, a, b)
        w = w * dab / 2.0

    return x, w


# ---------------------------------------------------------------------------
# 2D Chebyshev tensor grid
# ---------------------------------------------------------------------------


def chebpts2(
    nx: int,
    ny: int | None = None,
    domain: tuple[float, float, float, float] | None = None,
    kind: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """2D Chebyshev tensor product grid.

    [XX, YY] = CHEBPTS2(N) constructs an N by N grid of Chebyshev tensor
    points on [-1, 1]^2.

    [XX, YY] = CHEBPTS2(NX, NY) constructs an NX by NY grid.

    [XX, YY] = CHEBPTS2(NX, NY, [a, b, c, d]) uses the rectangle [a,b] x [c,d].

    Parameters
    ----------
    nx : int
        Number of points in x direction.
    ny : int or None
        Number of points in y direction.  Defaults to nx.
    domain : (a, b, c, d) or None
        Bounding rectangle.  Defaults to [-1, 1, -1, 1].
    kind : {1, 2}, default 2
        Kind of Chebyshev points.

    Returns
    -------
    XX : jnp.ndarray, shape (ny, nx)
        x-coordinates on the tensor grid.
    YY : jnp.ndarray, shape (ny, nx)
        y-coordinates on the tensor grid.

    Provenance
    ----------
    MATLAB source : chebpts2.m (wrapper for chebfun2.chebpts2)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpts, chebpts3, paduapts
    """
    if ny is None:
        ny = nx
    if domain is None:
        ax, bx, ay, by = -1.0, 1.0, -1.0, 1.0
    else:
        ax, bx, ay, by = float(domain[0]), float(domain[1]), float(domain[2]), float(domain[3])

    x = chebpts_ab(nx, ax, bx, kind=kind)
    y = chebpts_ab(ny, ay, by, kind=kind)

    XX, YY = jnp.meshgrid(x, y)
    return XX, YY


# ---------------------------------------------------------------------------
# 3D Chebyshev tensor grid
# ---------------------------------------------------------------------------


def chebpts3(
    nx: int,
    ny: int | None = None,
    nz: int | None = None,
    domain: tuple[float, ...] | None = None,
    kind: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """3D Chebyshev tensor product grid.

    [XX, YY, ZZ] = CHEBPTS3(N) constructs an N by N by N grid of Chebyshev
    tensor points on [-1, 1]^3.

    [XX, YY, ZZ] = CHEBPTS3(NX, NY, NZ) constructs an NX by NY by NZ grid.

    [XX, YY, ZZ] = CHEBPTS3(NX, NY, NZ, DOM) uses the cube [a,b]x[c,d]x[e,g]
    where DOM = [a, b, c, d, e, g].

    Parameters
    ----------
    nx : int
        Number of points in x.
    ny : int or None
        Number of points in y.  Defaults to nx.
    nz : int or None
        Number of points in z.  Defaults to nx.
    domain : sequence of 6 floats or None
        [a, b, c, d, e, g].  Defaults to [-1,1,-1,1,-1,1].
    kind : {1, 2}, default 2
        Kind of Chebyshev points.

    Returns
    -------
    XX, YY, ZZ : jnp.ndarray, shape (ny, nx, nz)
        Coordinates on the tensor grid (ndgrid ordering).

    Provenance
    ----------
    MATLAB source : chebpts3.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpts, chebpts2
    """
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx
    if domain is None:
        dom = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    else:
        dom = tuple(float(d) for d in domain)
        if len(dom) != 6:
            raise ValueError(f"domain must have 6 elements, got {len(dom)}")

    x = chebpts_ab(nx, dom[0], dom[1], kind=kind)
    y = chebpts_ab(ny, dom[2], dom[3], kind=kind)
    z = chebpts_ab(nz, dom[4], dom[5], kind=kind)

    # Use indexing='ij' to match MATLAB ndgrid(x, y, z)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing='ij')
    return XX, YY, ZZ


# ---------------------------------------------------------------------------
# Padua points
# ---------------------------------------------------------------------------


def paduapts(
    n: int,
    domain: tuple[float, float, float, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Padua points for 2D polynomial interpolation.

    XY, IDX = PADUAPTS(N) returns the degree-N first-kind Padua points on
    [-1, 1]^2.  There are (N+1)*(N+2)/2 Padua points.

    The Padua points form a unisolvent set for total-degree polynomial
    interpolation on the square and admit an explicit interpolation formula.

    Parameters
    ----------
    n : int
        Polynomial degree.  Non-negative integer.
    domain : (a, b, c, d) or None
        Rectangle [a,b] x [c,d].  Defaults to [-1,1,-1,1].

    Returns
    -------
    xy : jnp.ndarray, shape ((n+1)*(n+2)//2, 2)
        Padua points as (x, y) pairs.
    idx : jnp.ndarray, dtype bool, shape depends on parity of n
        Logical index matrix identifying which entries of the
        (n+1) x (n+2) Chebyshev tensor grid form the Padua points.

    Notes
    -----
    The ordering is consistent with Padua2DM [1].

    References
    ----------
    .. [1] M. Caliari, S. De Marchi, A. Sommariva, M. Vianello,
       "Padua2DM: fast interpolation and cubature at the Padua points in
       Matlab/Octave", ACM TOMS 37(3), 2011.

    Provenance
    ----------
    MATLAB source : paduapts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebpts, chebpts2
    """
    import numpy as np

    if domain is None:
        dom = (-1.0, 1.0, -1.0, 1.0)
    else:
        dom = tuple(float(d) for d in domain)

    if n == 0:
        xy = jnp.array([[dom[0], dom[2]]], dtype=jnp.float64)
        idx = jnp.ones((1, 1), dtype=bool)
        return xy, idx

    # 1D Chebyshev grids (2nd-kind, descending as in MATLAB Chebfun)
    xn1_asc = chebpts_ab(n + 1, dom[0], dom[1], kind=2)
    xn2_asc = chebpts_ab(n + 2, dom[2], dom[3], kind=2)

    # Reverse for consistency with Padua2DM (MATLAB flips: xn1 = xn1(end:-1:1))
    xn1 = xn1_asc[::-1]
    xn2 = xn2_asc[::-1]

    # Full tensor grid via meshgrid (x varies along columns, y along rows)
    x1, x2 = jnp.meshgrid(xn1, xn2)  # x1: (n+2, n+1), x2: (n+2, n+1)

    # Extract every other term (alternating checkerboard)
    total = (n + 1) * (n + 2)
    idx_flat = np.ones(total, dtype=bool)
    idx_flat[::2] = False  # set even-indexed entries to False

    if n % 2 == 1:
        idx_mat = idx_flat.reshape(n + 2, n + 1)
    else:
        idx_mat = idx_flat.reshape(n + 1, n + 2).T

    idx_jnp = jnp.array(idx_mat)

    # Extract Padua points
    x1_np = np.array(x1)
    x2_np = np.array(x2)
    xy_x = x1_np[idx_mat]
    xy_y = x2_np[idx_mat]
    xy = jnp.array(np.column_stack([xy_x, xy_y]), dtype=jnp.float64)

    return xy, idx_jnp


def legpts(
    n: int,
    interval: tuple[float, float] | None = None,
    *,
    bary: bool = False,
    newtheta: bool = False,
):
    """See ``_legpts_core``.  With ``bary=True`` also returns the
    normalized barycentric weights (MATLAB's third output). With
    ``newtheta=True``, return MATLAB's four outputs ``(x, w, v, theta)``.

    Provenance
    ----------
    MATLAB source : legpts.m
    Chebfun commit: 7574c77
    """
    if n >= 100:
        from chebfunjax.utils.legendre_fast import _legpts_asy_with_theta

        x, w, v, theta = _legpts_asy_with_theta(n)
        if interval is not None:
            a, b = interval
            dab = b - a
            x = (x + 1.0) * dab / 2.0 + a
            w = dab * w / 2.0
    else:
        x, w = _legpts_core(n, interval)
        if bary or newtheta:
            from chebfunjax.utils.legendre_rec import _legpts_rec

            _x_ref, _w_ref, v = _legpts_rec(n)
            if n == 0:
                theta = jnp.empty((0,), dtype=jnp.float64)
            elif n == 1:
                theta = jnp.array([jnp.pi / 2.0], dtype=jnp.float64)
            else:
                theta = jnp.arccos(_x_ref)

    if newtheta:
        return x, w, v, theta
    if bary:
        return x, w, v
    return x, w


def jacpts(n: int, a: float, b: float, interval: tuple[float, float] | None = None, *, bary: bool = False):
    """See ``_jacpts_core``.  With ``bary=True`` also returns the
    normalized barycentric weights (MATLAB's third output)."""
    out = _jacpts_core(n, a, b, interval)
    if not bary:
        return out
    x, w = out
    return x, w, _bary_weights_gauss(x, flip=False)


def hermpts(n: int, kind: str = 'phys', *options, method: str = 'default',
            bary: bool = False):
    """Gauss--Hermite nodes, weights, and optional barycentric weights.

    This wrapper accepts MATLAB's method and type flags in any order, with
    repeated flags taking the last supplied value. Python also accepts
    ``kind=`` for the existing type argument and ``method=``/``bary=`` as
    keywords. Thus ``hermpts(42, 'REC', 'prob')`` and
    ``hermpts(42, 'prob', 'REC')`` select the same rule.

    The default follows MATLAB ``hermpts``: GW for n <= 20, REC for
    21 <= n < 200, and ASY for n >= 200. Explicit GW, REC, GLR, and ASY
    methods are supported. The Python result vectors are one-dimensional,
    adapting MATLAB's column ``x`` and ``v`` and row ``w`` outputs to this
    package's existing convention.

    Explicit REC/ASY at n=2..20 remain unsupported because the pinned source's
    initial-guess expansion has not been verified in that range. LAG is not
    yet supported for n>1 and raises ``NotImplementedError`` if requested.

    Provenance
    ----------
    MATLAB source : ``hermpts.m``
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``
    Original authors: Nick Trefethen (GW), Nick Hale (REC/GLR),
        Thomas Trogdon and Sheehan Olver (ASY).
    """
    if int(n) != n or n < 0:
        raise ValueError("hermpts: n must be a nonnegative integer")
    n = int(n)

    # MATLAB returns immediately for n=0, before validating method/type flags.
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return (empty, empty, empty) if bary else (empty, empty)

    # MATLAB processes varargin in order; later method/type flags replace earlier
    # ones. Keep kind as the Python compatibility argument but parse it first.
    selected_method = 'default'
    selected_kind = 'phys'
    for flag in (kind, *options):
        if not isinstance(flag, str):
            raise ValueError("hermpts: options must be strings")
        lowered = flag.lower()
        if lowered in ('gw', 'glr', 'rec', 'lag', 'asy'):
            selected_method = lowered
        elif lowered.startswith('phy'):
            selected_kind = 'phys'
        elif lowered.startswith('pro'):
            selected_kind = 'prob'
        else:
            raise ValueError(f"hermpts: unrecognised input string {flag!r}")

    # A non-default Python keyword is an explicit final method override.
    if not isinstance(method, str):
        raise ValueError("hermpts: method must be a string")
    if method.lower() != 'default':
        method = method.lower()
        if method not in ('gw', 'glr', 'rec', 'lag', 'asy'):
            raise ValueError(f"hermpts: unsupported method {method!r}")
        selected_method = method
    method, kind = selected_method, selected_kind

    if method not in ('default', 'gw', 'glr', 'rec', 'lag', 'asy'):
        raise ValueError(f"hermpts: unsupported method {method!r}")

    # Source's trivial singleton branch precedes algorithm selection, including LAG.
    if n == 1:
        x = jnp.zeros((1,), dtype=jnp.float64)
        w = jnp.full((1,), jnp.sqrt(jnp.pi), dtype=jnp.float64)
        v = jnp.ones((1,), dtype=jnp.float64)
    elif method == 'lag':
        from chebfunjax.utils.hermite_lag import _hermpts_lag
        x, w, v = _hermpts_lag(n)
    elif method == 'rec' or (method == 'default' and 20 < n < 200):
        from chebfunjax.utils.hermite_rec import _hermpts_rec
        x, w, v = _hermpts_rec(n)
    elif method == 'asy' or (method == 'default' and n >= 200):
        from chebfunjax.utils.hermite_asy import _hermpts_asy
        x, w, v = _hermpts_asy(n)
    elif method == 'glr':
        from chebfunjax.utils.hermite_glr import _hermpts_glr
        x, w, v = _hermpts_glr(n)
    else:
        # Explicit GW and the default n<=20 path preserve P's GW implementation.
        x, w = _hermpts_core(n, 'phys')
        v = jnp.sqrt(w / jnp.max(w)) * jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)

    # MATLAB normalizes each method's weights, then applies the prob scaling.
    w = (jnp.sqrt(jnp.pi) / jnp.sum(w)) * w
    if kind == 'prob':
        x, w = x * jnp.sqrt(2.0), w * jnp.sqrt(2.0)
    return (x, w, v) if bary else (x, w)

def lagpts(n: int, alpha: float = 0.0,
           interval: tuple[float, float] | None = None, *,
           bary: bool = False, method: str = 'default'):
    """Gauss--Laguerre nodes, weights, and optional barycentric weights.

    This implementation supports REC/GW/GLR and bounded alpha=0,+/-1/2 RH, defaulting to REC
    for n<300, GW for 300<=n<1000, GLR for 1000<=n<3000 when alpha=0, and
    RH for n>=3000 with concrete alpha in {0,-1/2,+1/2}, and GW otherwise. MATLAB uses RH
    for all alpha from n=3000; other alpha and small explicit RH are unported.
    Dynamic alpha at the GLR/RH default thresholds retains the GW path because
    source method selection is static in this Python/JAX API. Explicit GLR
    requires concrete alpha=0. The Python API returns 1D vectors in place of MATLAB's
    column-node/column-bary and row-weight convention.

    Provenance
    ----------
    MATLAB source : ``lagpts.m`` (``lag_rec``, ``gw``, ``glr``, ``newton``, dispatcher)
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``
    """
    if int(n) != n or n < 0:
        raise ValueError("lagpts: n must be a nonnegative integer")
    n = int(n)
    # MATLAB returns [] at n=0 before examining optional arguments.
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return (empty, empty, empty) if bary else (empty, empty)

    if isinstance(alpha, str):
        if method != 'default':
            raise ValueError("lagpts: method specified twice")
        method, alpha = alpha, 0.0
    if not isinstance(method, str):
        raise ValueError("lagpts: method must be a string")
    method = method.lower()
    if method == 'default':
        if n < 300:
            method = 'rec'
        elif 1000 <= n < 3000 and not isinstance(alpha, jax.core.Tracer) and alpha == 0:
            method = 'glr'
        elif n >= 3000 and not isinstance(alpha, jax.core.Tracer) and alpha in (0, -0.5, 0.5):
            method = 'rh'
        else:
            method = 'gw'
    if method not in ('rec', 'gw', 'glr', 'rh'):
        if method in ('rhw', 'exp', 'expw', 'recw'):
            raise NotImplementedError(f"lagpts: source method {method.upper()} is not yet supported")
        raise ValueError(f"lagpts: unsupported method {method!r}")

    if not jnp.isrealobj(alpha) or (
        not isinstance(alpha, jax.core.Tracer) and alpha < -1
    ):
        raise ValueError("lagpts: alpha must be real and >= -1")
    if interval is not None:
        if len(interval) > 2:
            warnings.warn("lagpts: piecewise intervals not supported and will be ignored",
                          UserWarning, stacklevel=2)
            interval = (interval[0], interval[-1])
        if len(interval) != 2 or sum(isinf(v) for v in interval) != 1:
            raise ValueError("lagpts: interval must be semi-infinite")

    x, w = _lagpts_core(n, alpha, None, method)
    if bary:
        # MATLAB computes these before affine mapping of a semi-infinite domain.
        v = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0) * jnp.sqrt(w * x)
        v = v / jnp.max(jnp.abs(v))
    if interval is not None:
        if isinf(interval[1]):
            x, w = x + interval[0], w * jnp.exp(-interval[0])
        else:
            x, w = -x + interval[1], w * jnp.exp(interval[1])
    return (x, w, v) if bary else (x, w)

def ultrapts(n: int, lam: float, interval: tuple[float, float] | None = None, *, bary: bool = False):
    """See ``_ultrapts_core``.  With ``bary=True`` also returns the
    normalized barycentric weights (MATLAB's third output)."""
    out = _ultrapts_core(n, lam, interval)
    if not bary:
        return out
    x, w = out
    return x, w, _bary_weights_gauss(x, flip=False)


def radaupts(n: int, alp: float = 0.0, bet: float = 0.0, *, bary: bool = False):
    """See ``_radaupts_core``.  With ``bary=True`` also returns the
    normalized barycentric weights (MATLAB's third output)."""
    out = _radaupts_core(n, alp, bet)
    if not bary:
        return out
    x, w = out
    return x, w, _bary_weights_gauss(x, flip=True)


def lobpts(n: int, alp: float = 0.0, bet: float = 0.0, *, bary: bool = False):
    """See ``_lobpts_core``.  With ``bary=True`` also returns the
    normalized barycentric weights (MATLAB's third output)."""
    out = _lobpts_core(n, alp, bet)
    if not bary:
        return out
    x, w = out
    return x, w, _bary_weights_gauss(x, flip=True)
