# uses-numpy: integral-operator eigenvalue solves use numpy/scipy dense and
# ARPACK eigensolvers (not JIT-safe).
"""Fredholm and Volterra integral operators.

Provides :func:`fred` (Fredholm integral) and :func:`volt` (Volterra integral),
which apply an integral kernel to a Chebfun function to produce a new Chebfun.
Also provides :func:`fred_eigs` / :func:`volt_eigs` for the operator
eigenvalue problem (the MATLAB ``chebop(@(u) fred(K,u))`` + ``eigs`` path).

Translated from MATLAB Chebfun (commit 7574c77): @chebfun/fred.m, @chebfun/volt.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

__all__ = ["fred", "volt", "fred_eigs", "volt_eigs"]


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
        call ``K(X, Y)`` where ``X``, ``Y`` are 2-D arrays (``jnp.meshgrid``
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

    # Gauss-Legendre nodes and weights on [-1, 1]
    t_ref, w_ref = legpts(n)
    t_ref = jnp.asarray(t_ref, dtype=jnp.float64)
    w_ref = jnp.asarray(w_ref, dtype=jnp.float64)
    # Map from [-1, 1] to [a, b]
    yj = 0.5 * (b - a) * t_ref + 0.5 * (a + b)  # shape (n,)
    wj = w_ref * 0.5 * (b - a)                    # shape (n,)
    fvals = jnp.asarray(f(yj), dtype=jnp.float64)  # shape (n,)

    def _integrand(x_arr):
        """Evaluate (Kf)(x) for a vector of x values."""
        x_arr = jnp.asarray(x_arr, dtype=jnp.float64)
        # Build tensor-product grid
        X, Y = jnp.meshgrid(x_arr, yj, indexing="ij")  # (m, n)
        Kvals = jnp.asarray(K(X, Y), dtype=jnp.float64)  # (m, n)
        # Integrate in y: (m, n) @ (n,) = (m,)
        return Kvals @ (wj * fvals)

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
    t_ref, w_ref = legpts(n // 2 if n > 1 else 1)
    t_ref = jnp.asarray(t_ref, dtype=jnp.float64)
    w_ref = jnp.asarray(w_ref, dtype=jnp.float64)

    def _volt_at_x(x_scalar: float) -> float:
        """Evaluate (Vf)(x) at a single point."""
        if x_scalar <= a + 1e-15 * (b - a):
            return 0.0
        # Map GL nodes from [-1,1] to [a, x_scalar]
        yj = 0.5 * (x_scalar - a) * t_ref + 0.5 * (x_scalar + a)  # (n/2,)
        scale = 0.5 * (x_scalar - a)
        fvals = jnp.asarray(f(yj), dtype=jnp.float64)
        Kvals = jnp.asarray(
            [float(K(x_scalar, yj[j])) for j in range(yj.shape[0])],
            dtype=jnp.float64,
        )
        return float(jnp.dot(w_ref * scale, Kvals * fvals))

    def _integrand(x_arr):
        """Vectorised evaluation over array of x values."""
        x_arr = jnp.asarray(x_arr, dtype=jnp.float64)
        result = jnp.asarray(
            [_volt_at_x(float(xi)) for xi in x_arr.ravel()],
            dtype=jnp.float64,
        )
        return result.reshape(x_arr.shape)

    return _chebfun_factory(_integrand, domain=(a, b))


# ===========================================================================
# Eigenvalues of integral operators (@chebop eigs path)
# ===========================================================================


def _cheb_grid_weights(n: int, a: float, b: float):
    """2nd-kind Chebyshev points and Clenshaw-Curtis weights on ``[a, b]``."""
    from chebfunjax.utils.quadrature import chebpts, chebweights

    xr = np.asarray(chebpts(n, kind=2), dtype=np.float64)
    wr = np.asarray(chebweights(n, kind=2), dtype=np.float64)
    x = a + (b - a) * (xr + 1.0) / 2.0
    w = wr * (b - a) / 2.0
    return x, w


def _fredholm_matrix(K, a, b, n, scale):
    """Chebyshev-collocation matrix of the Fredholm operator.

    Discretises ``scale * int_a^b K(x, y) . dy`` on ``n`` 2nd-kind
    Chebyshev points via ``K(X, Y) @ diag(w)`` with Clenshaw-Curtis
    weights ``w`` (MATLAB ``@chebcolloc/fred``).
    """
    x, w = _cheb_grid_weights(n, a, b)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Kmat = np.asarray(K(X, Y), dtype=complex)
    return scale * (Kmat * w[None, :]), x


def _volterra_matrix(K, a, b, n, scale):
    """Chebyshev-collocation matrix of the Volterra operator.

    Discretises ``scale * int_a^x K(x, y) . dy`` as ``K(X, Y) .* Q`` where
    ``Q`` is the spectral cumulative-integration (cumsum) matrix on the
    2nd-kind Chebyshev grid (MATLAB ``@chebcolloc/volt``).
    """
    from chebfunjax.utils.diffmat import cumsummat

    x, _ = _cheb_grid_weights(n, a, b)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Kmat = np.asarray(K(X, Y), dtype=complex)
    Q = np.asarray(cumsummat(n, domain=(a, b), kind=2), dtype=complex)
    return scale * (Kmat * Q), x


def _k_eigs(M, k, which):
    """``k`` eigenvalues (and vectors) of ``M`` selected by ``which``.

    Uses ARPACK when ``k`` is small relative to the matrix size, otherwise a
    dense solve.  Returns ``(vals, vecs)`` with columns of ``vecs`` the
    corresponding eigenvectors.
    """
    n = M.shape[0]
    if 0 < k < n - 1 and n > 12:
        import scipy.sparse.linalg as sla

        try:
            vals, vecs = sla.eigs(M, k=k, which=which)
            return vals, vecs
        except sla.ArpackNoConvergence:
            # Clustered/near-zero spectra (e.g. quasi-nilpotent Volterra)
            # defeat ARPACK; fall back to a dense solve below.
            pass
    vals, vecs = np.linalg.eig(M)
    if which == "LM":
        order = np.argsort(-np.abs(vals))
    elif which == "SM":
        order = np.argsort(np.abs(vals))
    elif which == "LR":
        order = np.argsort(-np.real(vals))
    elif which == "SR":
        order = np.argsort(np.real(vals))
    else:
        raise ValueError(
            f"integral eigs: unknown selector which={which!r} "
            f"(use 'LM', 'SM', 'LR', or 'SR').")
    order = order[:k]
    return vals[order], vecs[:, order]


def _eigenfunctions(vecs, a, b, domain):
    """Build Chebfun eigenfunctions from collocation-point eigenvectors."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.tech.chebtech import Chebtech2

    funs = []
    for j in range(vecs.shape[1]):
        v = jnp.asarray(vecs[:, j], dtype=jnp.complex128)
        # Chebyshev-2 values are stored top-to-bottom (x descending); our
        # grid is ascending, so reverse before vals2coeffs.
        tech = Chebtech2.from_values(v[::-1])
        funs.append(Chebfun(funs=[_Piece(tech=tech, interval=(a, b))],
                            domain=domain))
    return funs


def _integral_eigs(matrix_builder, K, domain, k, which, scale, n, tol,
                   return_eigenfunctions, max_n):
    """Shared adaptive eigenvalue driver for Fredholm/Volterra operators."""
    from chebfunjax.domain import Domain

    a, b = float(domain[0]), float(domain[-1])
    dom = Domain((a, b))

    def _solve(nn):
        M, _x = matrix_builder(K, a, b, nn, scale)
        return _k_eigs(M, k, which)

    if n is not None:
        vals, vecs = _solve(int(n))
    else:
        # Adaptive: refine until the selected eigenvalues stop moving.
        nn = 32
        vals, vecs = _solve(nn)
        prev = np.sort_complex(vals)
        while nn < max_n:
            nn = min(2 * nn, max_n)
            vals, vecs = _solve(nn)
            cur = np.sort_complex(vals)
            m = min(len(prev), len(cur))
            if m > 0 and np.max(np.abs(cur[:m] - prev[:m])) < tol:
                break
            prev = cur

    vals_j = jnp.asarray(vals)
    if return_eigenfunctions:
        return vals_j, _eigenfunctions(vecs, a, b, dom)
    return vals_j


def fred_eigs(K: Callable, domain=(-1.0, 1.0), k: int = 6, *,
              which: str = "LM", scale: complex = 1.0,
              n: "int | None" = None, tol: float = 1e-10,
              return_eigenfunctions: bool = False, max_n: int = 1024):
    r"""Eigenvalues of a Fredholm integral operator.

    Solves the eigenvalue problem

    .. math::
        \mathrm{scale}\int_a^b K(x, y)\,\varphi(y)\,dy = \lambda\,\varphi(x)

    by Chebyshev collocation: the operator is discretised on 2nd-kind
    Chebyshev points as ``K(X, Y) @ diag(w)`` with Clenshaw-Curtis weights
    ``w`` (the MATLAB ``chebop(@(u) fred(K, u))`` / ``@chebcolloc/fred``
    path), and the eigenvalues of that matrix converge to those of the
    operator.

    Parameters
    ----------
    K : callable
        Kernel ``K(x, y)`` accepting tensor-product (``ndgrid``) arguments.
    domain : sequence of two floats, optional
        Interval ``[a, b]`` (default ``[-1, 1]``).
    k : int, optional
        Number of eigenvalues to return (default 6).
    which : {'LM', 'SM', 'LR', 'SR'}, optional
        Which eigenvalues: largest/smallest magnitude or real part.
    scale : complex, optional
        Multiplicative constant in front of the integral (e.g.
        ``sqrt(1j*F/pi)`` for the Fox-Li operator).
    n : int or None, optional
        Collocation size.  If ``None`` (default) the discretisation is
        refined adaptively (doubling from 32 up to ``max_n``) until the
        selected eigenvalues settle within ``tol``.
    tol : float, optional
        Convergence tolerance for the adaptive refinement (default 1e-10).
    return_eigenfunctions : bool, optional
        If True, also return a list of the eigenfunctions as Chebfuns.
    max_n : int, optional
        Maximum collocation size for the adaptive loop (default 1024).

    Returns
    -------
    lam : jnp.ndarray, shape (k,)
        The requested eigenvalues.
    funs : list of Chebfun, optional
        The eigenfunctions (only if ``return_eigenfunctions`` is True).

    Provenance
    ----------
    MATLAB source : @chebcolloc/fred.m, @chebop/eigs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    fred, volt_eigs
    """
    return _integral_eigs(_fredholm_matrix, K, domain, k, which, scale, n,
                          tol, return_eigenfunctions, max_n)


def volt_eigs(K: Callable, domain=(-1.0, 1.0), k: int = 6, *,
              which: str = "LM", scale: complex = 1.0,
              n: "int | None" = None, tol: float = 1e-10,
              return_eigenfunctions: bool = False, max_n: int = 1024):
    r"""Eigenvalues of a Volterra integral operator.

    Solves ``scale * int_a^x K(x, y) phi(y) dy = lambda phi(x)`` by
    Chebyshev collocation, discretising the operator as ``K(X, Y) .* Q``
    with ``Q`` the spectral cumulative-integration matrix (MATLAB
    ``@chebcolloc/volt``).  A Volterra operator with a bounded kernel is
    quasi-nilpotent, so its spectrum is ``{0}``; the returned eigenvalues
    of the finite discretisation cluster near zero and this routine is
    provided mainly for completeness and for generalised problems.

    Parameters and returns mirror :func:`fred_eigs`.

    Provenance
    ----------
    MATLAB source : @chebcolloc/volt.m, @chebop/eigs.m
    Chebfun commit: 7574c77

    See Also
    --------
    volt, fred_eigs
    """
    return _integral_eigs(_volterra_matrix, K, domain, k, which, scale, n,
                          tol, return_eigenfunctions, max_n)


# ===========================================================================
# Helpers
# ===========================================================================


def _clencurt(n: int, a: float, b: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clenshaw-Curtis nodes and weights on [a, b].

    Parameters
    ----------
    n : int
        Number of quadrature points (including both endpoints).
    a, b : float
        Integration interval.

    Returns
    -------
    x : jnp.ndarray, shape (n,)
    w : jnp.ndarray, shape (n,)
    """
    if n == 1:
        return jnp.array([(a + b) / 2.0]), jnp.array([b - a])

    theta = jnp.pi * jnp.arange(n, dtype=jnp.float64) / (n - 1)
    x = jnp.cos(theta)  # reference nodes in [-1, 1]

    # Clenshaw-Curtis weights (Waldvogel's formula)
    c = jnp.zeros(n)
    c = c.at[0::2].set(2.0 / (1.0 - jnp.arange(0, n, 2, dtype=jnp.float64) ** 2))
    c = jnp.real(jnp.fft.ifft(jnp.concatenate([c, c[n - 2:0:-1]])))
    w_ref = jnp.concatenate([c[0:1] / 2, c[1: n - 1], c[0:1] / 2])

    # Map to [a, b]
    x_phys = 0.5 * (b - a) * x + 0.5 * (a + b)
    # Reverse to ascending order
    x_phys = x_phys[::-1]
    w_phys = w_ref * (b - a) / 2.0

    return x_phys, w_phys
