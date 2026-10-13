"""Special function utilities.

Translated from MATLAB Chebfun (commit 7574c77): besselroots.m, gammaratio.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from numbers import Number

import jax.numpy as jnp

# ===========================================================================
# Bessel roots
# ===========================================================================


def besselroots(nu: float, n: int) -> jnp.ndarray:
    """First n positive zeros of the Bessel function J_nu(x).

    The roots follow the MATLAB Chebfun ``besselroots.m`` source, using its
    literal order-zero table, Piessens seeds, and McMahon expansion.

    Python scalar ``n`` is a static shape parameter under JIT; host array counts
    are accepted outside JIT. MATLAB's initial ``n == 0``
    check also returns for nonempty all-zero arrays, before scalar validation.

    Provenance
    ----------
    MATLAB source : besselroots.m
    Chebfun commit: 7574c77
    Original authors: L. L. Peixoto, 2015.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.
    """
    # MATLAB returns immediately for N == 0, before validating N or NU.
    try:
        if isinstance(n, Number):
            if n == 0:
                return jnp.array([], dtype=jnp.float64)
            n_value = float(n)
        else:
            n_array = jnp.asarray(n)
            if n_array.size and bool(jnp.all(n_array == 0)):
                return jnp.array([], dtype=jnp.float64)
            if n_array.size != 1:
                raise ValueError
            n_value = float(n_array.reshape(()))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "CHEBFUN:besselroots:inputN: Input N must be a positive integer"
        ) from None

    if not math.isfinite(n_value) or n_value != round(n_value) or n_value < 0:
        raise ValueError(
            "CHEBFUN:besselroots:inputN: Input N must be a positive integer"
        )

    from chebfunjax.utils.bessel_roots_general import _bessel_roots_general

    return _bessel_roots_general(nu, int(n_value))

# ===========================================================================
# Gamma ratio
# ===========================================================================


def gammaratio(m: float, delta: float) -> jnp.ndarray:
    """Compute gamma(m + delta) / gamma(m) accurately.

    GAMMARATIO(M, D) accurately computes gamma(M+D)/gamma(M) using a
    Stirling-based series when M is large. For small M, falls back to
    JAX's log-gamma difference. Both scalar inputs may be traced under JIT.

    Parameters
    ----------
    m : float
        Base argument.
    delta : float
        Increment.

    Returns
    -------
    ratio : scalar jax.Array
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
    from chebfunjax.utils.gamma_ratio import _gamma_ratio

    return _gamma_ratio(m, delta)
