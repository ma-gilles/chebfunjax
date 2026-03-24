"""Chebyshev collocation discretization.

Provides ``ChebColloc2`` (2nd-kind points) and ``ChebColloc1`` (1st-kind
points) for discretizing differential operators on Chebyshev grids.  These
are the building blocks for solving ODEs and BVPs via spectral collocation
(Phase 6 / Chebop).

Translated from MATLAB Chebfun classes @chebcolloc, @chebcolloc1,
@chebcolloc2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.utils.diffmat import (
    _cheb1_barywts,
    _cheb2_barywts,
    cumsummat,
    diffmat,
)
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts, chebweights

# ---------------------------------------------------------------------------
# Quadrature weights for 1st-kind Chebyshev points (Féjer's rule)
# ---------------------------------------------------------------------------

def _chebtech1_quadwts(n: int) -> jnp.ndarray:
    """Quadrature weights for n 1st-kind Chebyshev points.

    Computes weights that integrate ``f(x) dx`` (no weight function),
    i.e., these are Féjer-type weights that sum to 2 on [-1, 1].

    Follows MATLAB Chebfun's ``@chebtech1/quadwts.m`` (commit 7574c77):
    uses a variant of Waldvogel's FFT algorithm due to Nick Hale.

    Parameters
    ----------
    n : int
        Number of 1st-kind Chebyshev points.

    Returns
    -------
    w : jnp.ndarray, shape (n,)
        Quadrature weights, ordered to match ``chebpts(n, kind=1)``
        (ascending x order, i.e. reversed from the natural cos order).

    Provenance
    ----------
    MATLAB source : @chebtech1/quadwts.m
    Chebfun commit: 7574c77
    Original authors: Nick Hale (variant of Waldvogel's algorithm, 2006).
    Algorithm: Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
        quadrature rules", BIT Numerical Mathematics 46 (2006), pp 195-202.
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)

    # Chebyshev moments: m_k = int_{-1}^{1} T_k(x) dx
    # = 2/(1 - k^2)  for even k,  0 for odd k.
    # We only need the even-indexed moments up to k = n-1.
    # m = [2, 2/(1-4), 2/(1-16), ...] for even k >= 0.
    n_even = (n - 1) // 2 + 1  # number of even indices in 0..n-1
    k_even = jnp.arange(n_even, dtype=jnp.float64) * 2.0  # 0, 2, 4, ...
    m = 2.0 / (1.0 - k_even**2)

    # Build the FFT input vector c of length n.
    # MATLAB (0-based re-index for Python):
    #   if n is odd:   c = [m, -m[(n+1)/2-1 : 0 : -1]]   length n
    #   if n is even:  c = [m, 0, -m[n/2-1 : 0 : -1]]    length n
    if n % 2 == 1:
        # odd n: c = [m[0], ..., m[(n-1)/2], -m[(n-1)/2-1], ..., -m[0]]
        c = jnp.concatenate([m, -m[n_even - 2::-1]])
    else:
        # even n: c = [m[0], ..., m[n/2-1], 0, -m[n/2-2], ..., -m[0]]
        c = jnp.concatenate([m, jnp.array([0.0]), -m[n_even - 2::-1]])

    # Apply the rotation vector v_k = exp(i*k*pi/n)
    k_idx = jnp.arange(n, dtype=jnp.float64)
    v = jnp.exp(1j * k_idx * jnp.pi / n)
    c = c.astype(jnp.complex128) * v

    # IFFT gives the weights (in the natural descending cos-order)
    w_complex = jnp.fft.ifft(c)
    w = jnp.real(w_complex)

    # chebpts(n, kind=1) returns ascending order (reversed from cos-order).
    # The weights from the IFFT are in descending order (cos-order).
    return w[::-1]


# ---------------------------------------------------------------------------
# ChebColloc2 — collocation on 2nd-kind (Chebyshev-Lobatto) points
# ---------------------------------------------------------------------------


class ChebColloc2(eqx.Module):
    """Chebyshev collocation discretization on 2nd-kind points.

    Represents a spectral collocation discretization of a function or
    differential operator on a Chebyshev grid of the 2nd kind
    (Clenshaw-Curtis / Chebyshev-Lobatto points).

    This is the standard discretization used by MATLAB Chebfun's ``chebop``
    for BVP/ODE solving.  Function values are represented at the ``n``
    2nd-kind Chebyshev points; equations are enforced at ``n`` 1st-kind
    points (to avoid boundary duplication in the reduced system).

    Attributes
    ----------
    n : int
        Number of collocation points (static).
    domain : Domain
        Physical domain ``[a, b]`` on which the problem is posed.

    Examples
    --------
    Solve u'' = -1, u(-1) = u(1) = 0 (exact solution u = (1 - x²)/2):

    >>> disc = ChebColloc2(n=10, domain=Domain((-1.0, 1.0)))
    >>> D2 = disc.diffmat(k=2)          # 10×10 second-derivative matrix
    >>> x  = disc.points()              # 10 2nd-kind Chebyshev points
    >>> # Build system: replace rows 0 and -1 with boundary conditions
    >>> import jax.numpy as jnp
    >>> A = D2.at[0, :].set(0.0).at[0, 0].set(1.0).at[-1, :].set(0.0).at[-1, -1].set(1.0)
    >>> rhs = jnp.full(10, -1.0).at[0].set(0.0).at[-1].set(0.0)
    >>> u = jnp.linalg.solve(A, rhs)
    >>> exact = (1.0 - x**2) / 2.0
    >>> float(jnp.max(jnp.abs(u - exact))) < 1e-12
    True

    Provenance
    ----------
    MATLAB source : @chebcolloc/chebcolloc.m, @chebcolloc2/chebcolloc2.m,
        @chebcolloc2/diffmat.m, @chebcolloc2/cumsummat.m,
        @chebcolloc2/equationPoints.m, @chebcolloc2/functionPoints.m,
        @chebcolloc2/toValues.m, @valsDiscretization/points.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ChebColloc1, diffmat, cumsummat
    """

    n: int = eqx.field(static=True)
    domain: Domain = eqx.field(static=True)

    def __init__(self, n: int, domain: Domain | tuple[float, float] = (-1.0, 1.0)) -> None:
        """Create a ChebColloc2 discretization.

        Parameters
        ----------
        n : int
            Number of collocation (2nd-kind Chebyshev) points.  Must be ≥ 1.
        domain : Domain or (float, float), default (-1, 1)
            Physical interval ``[a, b]``.  A tuple ``(a, b)`` is automatically
            wrapped in a ``Domain``.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError(
                f"ChebColloc2 requires n >= 1 collocation points, got n={n}."
            )
        if not isinstance(domain, Domain):
            domain = Domain(domain)
        self.n = n
        self.domain = domain

    # ------------------------------------------------------------------
    # Core discretization operations
    # ------------------------------------------------------------------

    def diffmat(self, k: int = 1) -> jnp.ndarray:
        """Chebyshev differentiation matrix of order *k*.

        Returns the ``(n, n)`` matrix ``D`` such that ``D @ f`` approximates
        the *k*-th derivative of the polynomial interpolant through the
        function values at the ``n`` 2nd-kind Chebyshev points of
        ``self.domain``.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.  ``k=0`` returns the identity matrix.

        Returns
        -------
        D : jnp.ndarray, shape (n, n)
            Spectral differentiation matrix scaled to ``self.domain``.

        Notes
        -----
        For a domain ``[a, b]`` with ``a ≠ -1`` or ``b ≠ 1``, the matrix is
        scaled by ``(2 / (b - a))^k`` (chain rule).

        Provenance
        ----------
        MATLAB source : @chebcolloc2/diffmat.m, @chebcolloc/baryDiffMat.m
        Chebfun commit: 7574c77

        See Also
        --------
        cumsummat, eval_matrix, points
        """
        return diffmat(self.n, p=k, domain=self.domain.support, kind=2)

    def cumsummat(self) -> jnp.ndarray:
        """Chebyshev integration (cumulative sum) matrix.

        Returns the ``(n, n)`` matrix ``Q`` such that ``Q @ f`` gives the
        values of the antiderivative of the interpolant, with the convention
        that the antiderivative is zero at the left endpoint of the domain.

        Returns
        -------
        Q : jnp.ndarray, shape (n, n)
            Integration matrix scaled to ``self.domain``.

        Provenance
        ----------
        MATLAB source : @chebcolloc2/cumsummat.m
        Chebfun commit: 7574c77

        See Also
        --------
        diffmat, points, weights
        """
        return cumsummat(self.n, domain=self.domain.support, kind=2)

    def points(self) -> jnp.ndarray:
        """Collocation points (2nd-kind Chebyshev points on ``self.domain``).

        Returns
        -------
        x : jnp.ndarray, shape (n,)
            The ``n`` 2nd-kind Chebyshev points (Chebyshev-Lobatto /
            Clenshaw-Curtis nodes) mapped to ``self.domain``.
            Ordered from ``domain.a`` to ``domain.b``.

        Provenance
        ----------
        MATLAB source : @chebcolloc2/functionPoints.m,
            @valsDiscretization/points.m, @chebtech2/chebpts.m
        Chebfun commit: 7574c77

        See Also
        --------
        weights, diffmat, eval_matrix
        """
        x_ref = chebpts(self.n, kind=2)
        return self.domain.forward_map(x_ref)

    def equation_points(self) -> jnp.ndarray:
        """Points at which equations are enforced (1st-kind Chebyshev).

        In MATLAB Chebfun's CHEBCOLLOC2, *functions* are represented at 2nd-kind
        points but *equations* are enforced at 1st-kind points to avoid
        boundary-row duplication when imposing BCs.

        Returns
        -------
        x_eq : jnp.ndarray, shape (n,)
            The ``n`` 1st-kind Chebyshev points mapped to ``self.domain``.

        Provenance
        ----------
        MATLAB source : @chebcolloc2/equationPoints.m
        Chebfun commit: 7574c77

        See Also
        --------
        points, eval_matrix
        """
        x_ref = chebpts(self.n, kind=1)
        return self.domain.forward_map(x_ref)

    def weights(self) -> jnp.ndarray:
        """Clenshaw-Curtis quadrature weights on ``self.domain``.

        Returns
        -------
        w : jnp.ndarray, shape (n,)
            Quadrature weights such that ``jnp.dot(w, f_values)`` approximates
            ``∫_{a}^{b} f(x) dx``.

        Notes
        -----
        The weights are the standard Clenshaw-Curtis weights scaled to the
        length of the domain: ``w_physical = w_ref * (b - a) / 2``.

        Provenance
        ----------
        MATLAB source : @chebcolloc/sum.m, @valsDiscretization/points.m,
            @chebtech2/quadwts.m
        Chebfun commit: 7574c77

        See Also
        --------
        points, diffmat, cumsummat
        """
        w_ref = chebweights(self.n, kind=2)
        a, b = self.domain.support
        return w_ref * ((b - a) / 2.0)

    def bary_weights(self) -> jnp.ndarray:
        """Barycentric interpolation weights for the collocation points.

        Returns
        -------
        v : jnp.ndarray, shape (n,)
            Barycentric weights for the 2nd-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech2/barywts.m
        Chebfun commit: 7574c77

        See Also
        --------
        eval_matrix, points
        """
        return _cheb2_barywts(self.n)

    def eval_matrix(self, y: jnp.ndarray) -> jnp.ndarray:
        """Barycentric interpolation matrix from grid to arbitrary points.

        Returns the ``(M, n)`` matrix ``E`` such that ``E @ f_values``
        evaluates the polynomial interpolant through ``f_values`` at the
        ``n`` collocation points at each of the ``M`` target points ``y``.

        Parameters
        ----------
        y : jnp.ndarray, shape (M,) or scalar
            Target evaluation points, which must lie in ``self.domain``.

        Returns
        -------
        E : jnp.ndarray, shape (M, n)
            Barycentric interpolation matrix.

        Notes
        -----
        Uses the standard barycentric formula with 2nd-kind Chebyshev weights.
        Evaluation exactly at collocation nodes returns the appropriate
        identity row (no 0/0 cancellation issues).

        Provenance
        ----------
        MATLAB source : @chebcolloc/feval.m, barymat.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        points, bary_weights, barymat
        """
        y = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float64))
        x = self.points()
        w = self.bary_weights()
        return barymat(y, x, w)

    # ------------------------------------------------------------------
    # Convenience: repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChebColloc2(n={self.n}, domain={self.domain})"
        )


# ---------------------------------------------------------------------------
# ChebColloc1 — collocation on 1st-kind (Gauss-Chebyshev) points
# ---------------------------------------------------------------------------


class ChebColloc1(eqx.Module):
    """Chebyshev collocation discretization on 1st-kind points.

    Represents a spectral collocation discretization of a function or
    differential operator on a Chebyshev grid of the 1st kind
    (Gauss-Chebyshev nodes, interior points only).

    Unlike ``ChebColloc2``, both *function* and *equation* points coincide
    with 1st-kind Chebyshev nodes (no boundary nodes are included).

    Attributes
    ----------
    n : int
        Number of collocation points (static).
    domain : Domain
        Physical domain ``[a, b]`` on which the problem is posed.

    Provenance
    ----------
    MATLAB source : @chebcolloc1/chebcolloc1.m,
        @chebcolloc1/diffmat.m, @chebcolloc1/cumsummat.m,
        @chebcolloc1/equationPoints.m, @chebcolloc1/functionPoints.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ChebColloc2, diffmat, cumsummat
    """

    n: int = eqx.field(static=True)
    domain: Domain = eqx.field(static=True)

    def __init__(self, n: int, domain: Domain | tuple[float, float] = (-1.0, 1.0)) -> None:
        """Create a ChebColloc1 discretization.

        Parameters
        ----------
        n : int
            Number of collocation (1st-kind Chebyshev) points.  Must be ≥ 1.
        domain : Domain or (float, float), default (-1, 1)
            Physical interval ``[a, b]``.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError(
                f"ChebColloc1 requires n >= 1 collocation points, got n={n}."
            )
        if not isinstance(domain, Domain):
            domain = Domain(domain)
        self.n = n
        self.domain = domain

    # ------------------------------------------------------------------
    # Core discretization operations
    # ------------------------------------------------------------------

    def diffmat(self, k: int = 1) -> jnp.ndarray:
        """Chebyshev differentiation matrix of order *k*.

        Returns the ``(n, n)`` matrix ``D`` such that ``D @ f`` approximates
        the *k*-th derivative at the ``n`` 1st-kind Chebyshev points of
        ``self.domain``.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        D : jnp.ndarray, shape (n, n)
            Spectral differentiation matrix.

        Provenance
        ----------
        MATLAB source : @chebcolloc1/diffmat.m, @chebcolloc/baryDiffMat.m
        Chebfun commit: 7574c77

        See Also
        --------
        cumsummat, eval_matrix
        """
        return diffmat(self.n, p=k, domain=self.domain.support, kind=1)

    def cumsummat(self) -> jnp.ndarray:
        """Chebyshev integration (cumulative sum) matrix.

        Returns the ``(n, n)`` matrix ``Q`` such that ``Q @ f`` gives the
        values of the antiderivative of the interpolant, with the convention
        that the antiderivative is zero at the left endpoint of the domain.

        Returns
        -------
        Q : jnp.ndarray, shape (n, n)
            Integration matrix.

        Provenance
        ----------
        MATLAB source : @chebcolloc1/cumsummat.m
        Chebfun commit: 7574c77

        See Also
        --------
        diffmat, points, weights
        """
        return cumsummat(self.n, domain=self.domain.support, kind=1)

    def points(self) -> jnp.ndarray:
        """Collocation points (1st-kind Chebyshev points on ``self.domain``).

        Returns
        -------
        x : jnp.ndarray, shape (n,)
            The ``n`` 1st-kind Chebyshev points (Gauss-Chebyshev nodes)
            mapped to ``self.domain``, ordered from ``domain.a`` to
            ``domain.b``.

        Provenance
        ----------
        MATLAB source : @chebcolloc1/functionPoints.m,
            @valsDiscretization/points.m, @chebtech1/chebpts.m
        Chebfun commit: 7574c77

        See Also
        --------
        weights, diffmat, eval_matrix
        """
        x_ref = chebpts(self.n, kind=1)
        return self.domain.forward_map(x_ref)

    def equation_points(self) -> jnp.ndarray:
        """Points at which equations are enforced.

        For CHEBCOLLOC1, equation points coincide with function points
        (both are 1st-kind Chebyshev nodes).

        Returns
        -------
        x_eq : jnp.ndarray, shape (n,)
            Same as ``self.points()``.

        Provenance
        ----------
        MATLAB source : @chebcolloc1/equationPoints.m
        Chebfun commit: 7574c77

        See Also
        --------
        points
        """
        return self.points()

    def weights(self) -> jnp.ndarray:
        """Féjer-type quadrature weights on ``self.domain``.

        Returns the weights that integrate ``f(x) dx`` (no weight function),
        matching MATLAB Chebfun's ``chebtech1.quadwts``.  They sum to
        ``b - a`` on the domain ``[a, b]``.

        Returns
        -------
        w : jnp.ndarray, shape (n,)
            Quadrature weights such that ``jnp.dot(w, f_values)`` approximates
            ``∫_{a}^{b} f(x) dx``.

        Notes
        -----
        These are **not** the Gauss-Chebyshev weights ``π/n`` (which integrate
        ``f(x)/√(1-x²) dx``).  They are the Féjer weights that integrate
        ``f(x) dx`` exactly for polynomials of degree ≤ n-1.

        Provenance
        ----------
        MATLAB source : @chebcolloc/sum.m, @valsDiscretization/points.m,
            @chebtech1/quadwts.m
        Chebfun commit: 7574c77
        Original authors: Nick Hale (Waldvogel FFT algorithm variant).

        See Also
        --------
        points, diffmat
        """
        w_ref = _chebtech1_quadwts(self.n)
        a, b = self.domain.support
        return w_ref * ((b - a) / 2.0)

    def bary_weights(self) -> jnp.ndarray:
        """Barycentric interpolation weights for the collocation points.

        Returns
        -------
        v : jnp.ndarray, shape (n,)
            Barycentric weights for the 1st-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech1/barywts.m
        Chebfun commit: 7574c77

        See Also
        --------
        eval_matrix, points
        """
        return _cheb1_barywts(self.n)

    def eval_matrix(self, y: jnp.ndarray) -> jnp.ndarray:
        """Barycentric interpolation matrix from grid to arbitrary points.

        Returns the ``(M, n)`` matrix ``E`` such that ``E @ f_values``
        evaluates the polynomial interpolant through ``f_values`` at the
        ``n`` 1st-kind collocation points at each of the ``M`` target
        points ``y``.

        Parameters
        ----------
        y : jnp.ndarray, shape (M,) or scalar
            Target evaluation points.

        Returns
        -------
        E : jnp.ndarray, shape (M, n)
            Barycentric interpolation matrix.

        Provenance
        ----------
        MATLAB source : @chebcolloc/feval.m, barymat.m
        Chebfun commit: 7574c77

        See Also
        --------
        points, bary_weights, barymat
        """
        y = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float64))
        x = self.points()
        w = self.bary_weights()
        return barymat(y, x, w)

    def __repr__(self) -> str:
        return (
            f"ChebColloc1(n={self.n}, domain={self.domain})"
        )
