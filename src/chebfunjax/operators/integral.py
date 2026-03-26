"""Fredholm and Volterra integral operators.

Provides :func:`fred` (Fredholm integral) and :func:`volt` (Volterra integral),
which apply an integral kernel to a Chebfun function to produce a new Chebfun.

Translated from MATLAB Chebfun (commit 7574c77): @chebfun/fred.m, @chebfun/volt.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

__all__ = ["fred", "volt"]


# ===========================================================================
# Fredholm integral operator
# ===========================================================================


def fred(K: Callable, f, *, n: int = 128) -> "Chebfun":
    r"""Apply the Fredholm integral operator with kernel *K* to a Chebfun *f*.

    Computes the Chebfun representing

    .. math::
        (Kf)(x) = \int_a^b K(x, y)\, f(y)\, dy,

    where ``[a, b]`` is the domain of *f*.

    The integration over *y* uses Clenshaw-Curtis quadrature on *n* points.
    The outer function in *x* is then constructed adaptively by building a
    Chebfun that evaluates the definite integral for each *x*.

    Parameters
    ----------
    K : callable
        Kernel function ``K(x, y)``.  Must accept two scalar or 1-D array
        arguments and return an array of the same shape.  A tensor-product
        call ``K(X, Y)`` where ``X``, ``Y`` are 2-D arrays (``np.meshgrid``
        output) is used internally for efficiency.
    f : Chebfun
        Input function on domain ``[a, b]``.
    n : int, optional
        Number of Clenshaw-Curtis quadrature points for the inner integral.
        Default 128.  Increase for smooth kernels of high degree.

    Returns
    -------
    Ff : Chebfun
        Result on the same domain as *f*.

    Notes
    -----
    The integral is approximated as::

        (Kf)(x) ≈ w^T * (K(x, y_j) * f(y_j))

    where ``y_j`` are Clenshaw-Curtis nodes on ``[a, b]`` and ``w`` are the
    corresponding weights.  The outer Chebfun is then constructed adaptively.

    NOT JIT-safe (uses adaptive Chebfun construction).

    Provenance
    ----------
    MATLAB source : @chebfun/fred.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    volt

    Examples
    --------
    Identity kernel (K(x,y) = 1) integrates f over [-1, 1]:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> from chebfunjax.operators.integral import fred
    >>> f = chebfun(jnp.cos)
    >>> Ff = fred(lambda x, y: jnp.ones_like(x * y), f)
    >>> abs(float(Ff(jnp.float64(0.0))) - float(jnp.sin(jnp.float64(1.0)) - jnp.sin(jnp.float64(-1.0)))) < 1e-5
    True
    """
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun_factory
    from chebfunjax.utils.quadrature import legpts

    a = float(f.domain.a)
    b = float(f.domain.b)

    # Gauss-Legendre nodes and weights on [a, b]
    t_ref, w_ref = np.array(legpts(n))
    t_ref = np.asarray(t_ref, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)
    # Map from [-1, 1] to [a, b]
    yj = 0.5 * (b - a) * t_ref + 0.5 * (a + b)  # shape (n,)
    wj = w_ref * 0.5 * (b - a)                    # shape (n,)
    fvals = np.array(f(jnp.array(yj)), dtype=float)  # shape (n,)

    def _integrand(x_arr):
        """Evaluate (Kf)(x) for a vector of x values."""
        x_arr = np.asarray(x_arr, dtype=float)
        # Build tensor-product grid
        X, Y = np.meshgrid(x_arr, yj, indexing="ij")  # (m, n)
        Kvals = np.asarray(K(X, Y), dtype=float)       # (m, n)
        # Integrate in y: (m, n) @ (n,) = (m,)
        result = Kvals @ (wj * fvals)
        return jnp.array(result)

    return _chebfun_factory(_integrand, domain=(a, b))


# ===========================================================================
# Volterra integral operator
# ===========================================================================


def volt(K: Callable, f, *, n: int = 128) -> "Chebfun":
    r"""Apply the Volterra integral operator with kernel *K* to a Chebfun *f*.

    Computes the Chebfun representing

    .. math::
        (Kf)(x) = \int_a^x K(x, y)\, f(y)\, dy,

    where ``a`` is the left endpoint of the domain of *f*.

    At each evaluation point *x* the upper limit of integration changes,
    so the integral is computed via Gauss-Legendre quadrature with *n/2*
    nodes mapped to ``[a, x]``.

    Parameters
    ----------
    K : callable
        Kernel function ``K(x, y)``.  Must accept two scalar arguments and
        return a scalar; vectorised over the quadrature nodes.
    f : Chebfun
        Input function on domain ``[a, b]``.
    n : int, optional
        Number of Gauss-Legendre quadrature points per evaluation.
        Default 128.  For smooth kernels ``n=64`` is usually sufficient.

    Returns
    -------
    Vf : Chebfun
        Result on the same domain as *f*.

    Notes
    -----
    The outer Chebfun is constructed adaptively by calling the integral
    evaluation at Chebyshev points.  The integral at the left endpoint is
    always exactly zero (empty domain ``[a, a]``).

    NOT JIT-safe (uses adaptive Chebfun construction and Python loops).

    Provenance
    ----------
    MATLAB source : @chebfun/volt.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    fred

    Examples
    --------
    Volterra integral of f = 1 with kernel K(x, y) = 1 gives F(x) = x - a:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> from chebfunjax.operators.integral import volt
    >>> f = chebfun(lambda x: jnp.ones_like(x))
    >>> Vf = volt(lambda x, y: jnp.ones_like(x * y), f)
    >>> abs(float(Vf(jnp.float64(0.5))) - 1.5) < 1e-5
    True
    """
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun_factory
    from chebfunjax.utils.quadrature import legpts

    a = float(f.domain.a)
    b = float(f.domain.b)

    # Gauss-Legendre nodes and weights on [-1, 1]
    t_ref, w_ref = np.array(legpts(n // 2 if n > 1 else 1))

    def _volt_at_x(x_scalar: float) -> float:
        """Evaluate (Vf)(x) at a single point."""
        if x_scalar <= a + 1e-15 * (b - a):
            return 0.0
        # Map GL nodes from [-1,1] to [a, x_scalar]
        yj = 0.5 * (x_scalar - a) * t_ref + 0.5 * (x_scalar + a)  # (n/2,)
        scale = 0.5 * (x_scalar - a)
        fvals = np.asarray(f(jnp.array(yj)), dtype=float)
        Kvals = np.array([float(K(x_scalar, yj[j])) for j in range(len(yj))], dtype=float)
        return float(np.dot(w_ref * scale, Kvals * fvals))

    def _integrand(x_arr):
        """Vectorised evaluation over array of x values."""
        x_arr = np.asarray(x_arr, dtype=float)
        result = np.array([_volt_at_x(float(xi)) for xi in x_arr.ravel()],
                          dtype=float)
        return jnp.array(result.reshape(x_arr.shape))

    return _chebfun_factory(_integrand, domain=(a, b))


# ===========================================================================
# Helpers
# ===========================================================================


def _clencurt(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Curtis nodes and weights on [a, b].

    Parameters
    ----------
    n : int
        Number of quadrature points (including both endpoints).
    a, b : float
        Integration interval.

    Returns
    -------
    x : np.ndarray, shape (n,)
    w : np.ndarray, shape (n,)
    """
    if n == 1:
        return np.array([(a + b) / 2.0]), np.array([b - a])

    theta = np.pi * np.arange(n, dtype=float) / (n - 1)
    x = np.cos(theta)  # reference nodes in [-1, 1]

    # Clenshaw-Curtis weights (Waldvogel's formula)
    c = np.zeros(n)
    c[0::2] = 2.0 / (1.0 - np.arange(0, n, 2, dtype=float) ** 2)
    c = np.real(np.fft.ifft(np.concatenate([c, c[n - 2:0:-1]])))
    w_ref = np.concatenate([[c[0] / 2], c[1: n - 1], [c[0] / 2]])

    # Map to [a, b]
    x_phys = 0.5 * (b - a) * x + 0.5 * (a + b)
    # Reverse to ascending order
    x_phys = x_phys[::-1]
    w_phys = w_ref * (b - a) / 2.0

    return x_phys.copy(), w_phys.copy()
