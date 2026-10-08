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

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import numpy as np

__all__ = ["fred", "volt", "fred_eigs", "volt_eigs"]

if TYPE_CHECKING:
    from chebfunjax.chebfun1d.chebfun import Chebfun


# ===========================================================================
# Fredholm integral operator
# ===========================================================================


def _integral_result(action, domain, normv):
    """Source outer constructor resolves the result relative to norm(input)."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    pieces = [_Piece.from_function(action, a, b, vscale=normv)
              for a, b in zip(domain[:-1], domain[1:])]
    result = Chebfun(funs=pieces, domain=Domain(domain))
    return result.set_point_values(action(jnp.asarray(domain)))


def fred(K: Callable, f, onevar=None, *, n: int = 128) -> "Chebfun":
    r"""Apply int_a^b K(x,y)f(y)dy with JAX Gauss-Legendre quadrature.

    Each smooth input interval receives n nodes, retaining piecewise domains
    and real or complex scalar/array values. The outer Chebfun is adaptive.
    The fixed inner rule is an explicit numerical adaptation of the source's
    adaptive inner Chebfun construction; increase n for unresolved kernels.
    As in @chebfun/fred.m, onevar affects AD operator matrices only; direct
    action still calls the two-argument kernel. Adaptive outer construction
    is eager; the quadrature arithmetic is JAX.

    Provenance
    ----------
    MATLAB source: @chebfun/fred.m, @adchebfun/adchebfun.m (fred).
    Chebfun commit: 7574c77
    """
    from chebfunjax.autodiff.adchebfun import ADChebfun
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.operators.chebop import _FourierProxy
    from chebfunjax.utils.quadrature import legpts

    if isinstance(K, Chebfun):
        K, f = f, K
    if isinstance(f, ADChebfun):
        return f.fred(K, onevar)
    if isinstance(f, _FourierProxy):
        return f.fred(K)
    domain = tuple(float(x) for x in f.domain.breakpoints)
    column = f.transpose() if f.is_transposed else f
    normv = float(column.norm())
    nodes, weights = legpts(n)
    y = jnp.concatenate([a+(b-a)*(nodes+1)/2 for a, b in zip(domain[:-1], domain[1:])])
    w = jnp.concatenate([(b-a)*weights/2 for a, b in zip(domain[:-1], domain[1:])])
    values = column(y)
    weighted = w.reshape(w.shape+(1,)*(values.ndim-1))*values

    def action(x):
        x = jnp.asarray(x)
        X, Y = jnp.meshgrid(x.ravel(), y, indexing="ij")
        kernel = jnp.broadcast_to(jnp.asarray(K(X, Y)), X.shape)
        result = jnp.tensordot(kernel, weighted, axes=((-1,), (0,)))
        return result.reshape(x.shape+values.shape[1:])

    result = _integral_result(action, domain, normv)
    return Chebfun._as_transposed(result, f.is_transposed)


def volt(K: Callable, f, onevar=None, *, n: int = 128) -> "Chebfun":
    r"""Apply int_a^x K(x,y)f(y)dy with JAX interval-wise quadrature.

    Each smooth input interval is intersected with [a,x] and receives n/2
    Gauss-Legendre nodes (at least one). The source zero endpoint is exact.
    Complex/array values and input domain breaks are retained. As in fred,
    the inner fixed rule adapts the source adaptive Chebfun quadrature.
    Outer Chebfun construction remains eager, while numeric integration
    uses broadcast JAX operations without scalar Python evaluation loops.

    Provenance
    ----------
    MATLAB source: @chebfun/volt.m, @adchebfun/adchebfun.m (volt).
    Chebfun commit: 7574c77
    """
    from chebfunjax.autodiff.adchebfun import ADChebfun
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.utils.quadrature import legpts

    if isinstance(K, Chebfun):
        K, f = f, K
    if isinstance(f, ADChebfun):
        return f.volt(K, onevar)
    domain = tuple(float(x) for x in f.domain.breakpoints)
    column = f.transpose() if f.is_transposed else f
    normv = float(column.norm())
    nodes, weights = legpts(n//2 if n > 1 else 1)

    def action(x):
        x = jnp.asarray(x)
        flat = x.ravel()
        total = None
        for a, b in zip(domain[:-1], domain[1:]):
            half = jnp.clip(flat-a, 0, b-a)/2
            y = a+half[:, None]*(nodes[None, :]+1)
            values = column(y)
            kernel = jnp.broadcast_to(jnp.asarray(K(flat[:, None], y)), y.shape)
            weighted = half[:, None]*weights[None, :]*kernel
            weighted = weighted.reshape(weighted.shape+(1,)*(values.ndim-2))
            contribution = jnp.sum(weighted*values, axis=1)
            active = (half != 0).reshape(half.shape+(1,)*(contribution.ndim-1))
            contribution = jnp.where(active, contribution, 0)
            total = contribution if total is None else total+contribution
        return total.reshape(x.shape+total.shape[1:])

    result = _integral_result(action, domain, normv)
    return Chebfun._as_transposed(result, f.is_transposed)


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
