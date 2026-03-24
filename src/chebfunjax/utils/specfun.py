# uses-numpy: iterative root-finding and Stirling series — not JIT-safe
"""Special function utilities.

Translated from MATLAB Chebfun (commit 7574c77): besselroots.m, gammaratio.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.special import gammaln, jv as bessel_j, jvp as bessel_j_prime


# ===========================================================================
# Bessel roots
# ===========================================================================


def besselroots(nu: float, n: int) -> jnp.ndarray:
    """First n positive zeros of the Bessel function J_nu(x).

    BESSELROOTS(NU, N) returns the first N roots of J_nu(x) = besselj(nu, x).

    Parameters
    ----------
    nu : float
        Order of the Bessel function.  Both NU and N must be scalars;
        N must be a non-negative integer.
    n : int
        Number of zeros to compute.  Must be non-negative.

    Returns
    -------
    j : jnp.ndarray, shape (n,)
        First n positive roots of J_nu.

    Notes
    -----
    For nu == 0 the first 20 roots are precomputed from Wolfram Alpha.
    For -1 <= nu <= 5 the first 6 zeros use Piessens' Chebyshev-series
    approximations (1984) to 12 decimal figures.
    All remaining zeros use McMahon's asymptotic expansion.

    Provenance
    ----------
    MATLAB source : besselroots.m
    Chebfun commit: 7574c77
    Original authors: L. L. Peixoto, 2015.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    References
    ----------
    McMahon's expansion (1894) for large zeros of J_nu.
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)

    n = int(n)
    if n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")

    # McMahon's expansion for all n zeros (accurate for s >= 7)
    s = np.arange(1, n + 1, dtype=np.float64)
    mu = 4.0 * nu ** 2

    a1 = 1.0 / 8
    a3 = (7 * mu - 31) / 384
    a5 = 4 * (3779 + mu * (-982 + 83 * mu)) / 61440
    a7 = 6 * (-6277237 + mu * (1585743 + mu * (-153855 + 6949 * mu))) / 20643840
    a9 = (144 * (2092163573 + mu * (-512062548 + mu * (48010494 + mu * (-2479316 + 70197 * mu))))
          / 11890851840)
    a11 = (720 * (-8249725736393 + mu * (1982611456181 + mu * (-179289628602
           + mu * (8903961290 + mu * (-287149133 + 5592657 * mu))))) / 10463949619200)
    a13 = (576 * (423748443625564327 + mu * (-100847472093088506 + mu * (8929489333108377
           + mu * (-426353946885548 + mu * (13172003634537 + mu * (-291245357370
           + mu * 4148944183)))))) / 13059009124761600)

    b = 0.25 * (2 * nu + 4 * s - 1) * np.pi
    poly_coeffs = np.array([a13, 0, a11, 0, a9, 0, a7, 0, a5, 0, a3, 0, a1, 0])
    j = b - (mu - 1) * np.polyval(poly_coeffs, 1.0 / b)

    if nu == 0:
        # First 20 roots of J0 precomputed (Wolfram Alpha)
        j0_exact = np.array([
            2.4048255576957728,
            5.5200781102863106,
            8.6537279129110122,
            11.791534439014281,
            14.930917708487785,
            18.071063967910922,
            21.211636629879258,
            24.352471530749302,
            27.493479132040254,
            30.634606468431975,
            33.775820213573568,
            36.917098353664044,
            40.058425764628239,
            43.199791713176730,
            46.341188371661814,
            49.482609897397817,
            52.624051841114996,
            55.765510755019979,
            58.906983926080942,
            62.048469190227170,
        ])
        m = min(20, n)
        j[:m] = j0_exact[:m]

    elif -1 <= nu <= 5:
        # Piessens' Chebyshev series for first 6 zeros in -1 <= nu <= 5
        C = np.array([
            [2.883975316228, 8.263194332307, 11.493871452173, 14.689036505931,
             17.866882871378, 21.034784308088],
            [0.767665211539, 4.209200330779, 4.317988625384, 4.387437455306,
             4.435717974422, 4.471319438161],
            [-0.086538804759, -0.164644722483, -0.130667664397, -0.109469595763,
             -0.094492317231, -0.083234240394],
            [0.020433979038, 0.039764618826, 0.023009510531, 0.015359574754,
             0.011070071951, 0.008388073020],
            [-0.006103761347, -0.011799527177, -0.004987164201, -0.002655024938,
             -0.001598668225, -0.001042443435],
            [0.002046841322, 0.003893555229, 0.001204453026, 0.000511852711,
             0.000257620149, 0.000144611721],
            [-0.000734476579, -0.001369989689, -0.000310786051, -0.000105522473,
             -0.000044416219, -0.000021469973],
            [0.000275336751, 0.000503054700, 0.000083834770, 0.000022761626,
             0.000008016197, 0.000003337753],
            [-0.000106375704, -0.000190381770, -0.000023343325, -0.000005071979,
             -0.000001495224, -0.000000536428],
            [0.000042003336, 0.000073681222, 0.000006655551, 0.000001158094,
             0.000000285903, 0.000000088402],
            [-0.000016858623, -0.000029010830, -0.000001932603, -0.000000269480,
             -0.000000055734, -0.000000014856],
            [0.000006852440, 0.000011579131, 0.000000569367, 0.000000063657,
             0.000000011033, 0.000000002536],
            [-0.000002813300, -0.000004672877, -0.000000169722, -0.000000015222,
             -0.000000002212, -0.000000000438],
            [0.000001164419, 0.000001903082, 0.000000051084, 0.000000003677,
             0.000000000448, 0.000000000077],
            [-0.000000485189, -0.000000781030, -0.000000015501, -0.000000000896,
             -0.000000000092, -0.000000000014],
            [0.000000203309, 0.000000322648, 0.000000004736, 0.000000000220,
             0.000000000019, 0.000000000002],
            [-0.000000085602, -0.000000134047, -0.000000001456, -0.000000000054,
             -0.000000000004, 0.0],
            [0.000000036192, 0.000000055969, 0.000000000450, 0.000000000013,
             0.0, 0.0],
            [-0.000000015357, -0.000000023472, -0.000000000140, -0.000000000003,
             0.0, 0.0],
            [0.000000006537, 0.000000009882, 0.000000000043, 0.000000000001,
             0.0, 0.0],
            [-0.000000002791, -0.000000004175, -0.000000000014, 0.0, 0.0, 0.0],
            [0.000000001194, 0.000000001770, 0.000000000004, 0.0, 0.0, 0.0],
            [-0.000000000512, -0.000000000752, 0.0, 0.0, 0.0, 0.0],
            [0.000000000220, 0.000000000321, 0.0, 0.0, 0.0, 0.0],
            [-0.000000000095, -0.000000000137, 0.0, 0.0, 0.0, 0.0],
            [0.000000000041, 0.000000000059, 0.0, 0.0, 0.0, 0.0],
            [-0.000000000018, -0.000000000025, 0.0, 0.0, 0.0, 0.0],
            [0.000000000008, 0.000000000011, 0.0, 0.0, 0.0, 0.0],
            [-0.000000000003, -0.000000000005, 0.0, 0.0, 0.0, 0.0],
            [0.000000000001, 0.000000000002, 0.0, 0.0, 0.0, 0.0],
        ])
        # Evaluate via Clenshaw's algorithm (Chebyshev expansion in (nu-2)/3)
        t = (nu - 2.0) / 3.0
        six_zeros = _clenshaw(t, C)
        six_zeros[0] *= np.sqrt(nu + 1)  # Scale first root
        m = min(6, n)
        j[:m] = six_zeros[:m]

    return jnp.array(j[:n], dtype=jnp.float64)


def _clenshaw(t: float, C: np.ndarray) -> np.ndarray:
    """Evaluate a Chebyshev series in t by Clenshaw's algorithm.

    C has shape (deg+1, 6): evaluates C[:, k] as a Chebyshev series for each k.
    Returns a vector of 6 values.
    """
    m, ncols = C.shape
    bk1 = np.zeros(ncols)
    bk2 = np.zeros(ncols)
    for k in range(m - 1, 0, -1):
        bk = C[k] + 2 * t * bk1 - bk2
        bk2 = bk1
        bk1 = bk
    return C[0] + t * bk1 - bk2


# ===========================================================================
# Gamma ratio
# ===========================================================================


def gammaratio(m: float, delta: float) -> float:
    """Compute gamma(m + delta) / gamma(m) accurately.

    GAMMARATIO(M, D) accurately computes gamma(M+D)/gamma(M) using a
    Stirling-based series when M is large.  For small M, falls back to
    scipy.special.gammaln.

    Parameters
    ----------
    m : float
        Base argument.
    delta : float
        Increment.

    Returns
    -------
    ratio : float
        gamma(m + delta) / gamma(m).

    Notes
    -----
    When M > 15 and M >= delta, uses the Stirling series expansion from [1]
    to avoid catastrophic cancellation.

    References
    ----------
    .. [1] N. Hale and A. Townsend, "Fast and accurate computation of
       Gauss-Legendre and Gauss-Jacobi quadrature nodes and weights",
       SIAM J. Sci. Comp., 2013.

    Provenance
    ----------
    MATLAB source : gammaratio.m
    Chebfun commit: 7574c77
    Original authors: Nick Hale, Alex Townsend.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.
    """
    m = float(m)
    delta = float(delta)

    if m <= 15 or m < delta:
        # Fall back to log-gamma difference
        return float(np.exp(gammaln(m + delta) - gammaln(m)))

    if delta == 0.0:
        return 1.0

    # Ensure 0 < delta < 1 by stripping integer part
    fd = int(np.floor(delta))
    rd = delta - fd

    if fd >= 1:
        scl = 1.0
        for k in range(fd):
            scl *= (m + k + rd)
        return scl * gammaratio(m, rd)

    # Taylor/Stirling series for 0 < delta < 1
    ds = 0.5 * delta ** 2 / (m - 1)
    s = ds
    j_iter = 1
    while abs(ds / s) > np.finfo(float).eps / 100 and j_iter < 100:
        j_iter += 1
        ds = -delta * (j_iter - 1) / (j_iter + 1) / (m - 1) * ds
        s += ds

    p2 = np.exp(s) * np.sqrt(1 + delta / (m - 1)) * (m - 1) ** delta

    # Stirling's series
    g = np.array([1, 1 / 12, 1 / 288, -139 / 51840, -571 / 2488320,
                  163879 / 209018880, 5246819 / 75246796800,
                  -534703531 / 902961561600,
                  -4483131259 / 86684309913600,
                  432261921612371 / 514904800886784000])

    def stirling(z):
        acc = 0.0
        z_k = 1.0
        for gk in g:
            acc += gk * z_k
            z_k /= z
        return acc

    ratio = p2 * (stirling(m + delta - 1) / stirling(m - 1))
    return float(ratio)
