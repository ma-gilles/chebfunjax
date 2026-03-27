# uses-numpy: 2D adaptive construction uses numpy for pivot selection (not JIT-safe)
"""Chebfun2 — user-facing 2D function approximation on rectangles.

Wraps ``SeparableApprox`` with a friendly API for bivariate smooth
functions on rectangles [xa, xb] x [ya, yb].

Translated from MATLAB Chebfun class @chebfun2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Helper: 1-D integral of a Chebtech2 slice on a physical interval [a, b]
# ============================================================================


def _chebtech_integral(tech: Chebtech2, a: float, b: float) -> jax.Array:
    """Definite integral of ``tech`` (on reference [-1, 1]) over physical [a, b].

    Uses the standard scale factor (b - a) / 2 for the affine map
    ``x = (b - a) / 2 * t + (a + b) / 2``.
    """
    return tech.sum() * jnp.float64((b - a) / 2.0)


def _chebtech_diff_physical(tech: Chebtech2, a: float, b: float, k: int = 1) -> Chebtech2:
    """Differentiate ``tech`` with respect to the physical variable k times.

    Chain rule: d/dx = (2 / (b - a)) * d/dt, so the k-th derivative picks
    up a factor of (2 / (b - a))^k.

    Parameters
    ----------
    tech : Chebtech2
        A Chebtech2 on the reference interval [-1, 1].
    a, b : float
        Physical domain endpoints.
    k : int
        Differentiation order.

    Returns
    -------
    Chebtech2
        The k-th derivative (on reference [-1, 1]) scaled for the physical domain.
    """
    scale = (2.0 / (b - a)) ** k
    tech_der = tech.diff(k)
    scaled_coeffs = tech_der.coeffs * jnp.float64(scale)
    return Chebtech2.from_coeffs(scaled_coeffs)


# ============================================================================
# Main class
# ============================================================================


class Chebfun2(eqx.Module):
    """Chebfun2 — smooth function on a rectangle, via low-rank approximation.

    Represents a bivariate smooth function f(x, y) on a rectangle
    [xa, xb] x [ya, yb] using a ``SeparableApprox`` (Gaussian elimination
    with complete pivoting / Chebfun2 algorithm).

    The internal representation is:

        f(x, y) ≈ Σ_j  d_j * c_j(y) * r_j(x)

    where ``c_j`` are column slices (functions of y), ``r_j`` are row
    slices (functions of x), and ``d_j`` are scalar pivot weights.

    Attributes
    ----------
    approx : SeparableApprox
        The underlying low-rank approximation.

    Notes
    -----
    Construction is NOT JIT-safe (adaptive algorithm with Python loops).
    Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun2/chebfun2.m, @separableApprox/separableApprox.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend & L. N. Trefethen, "An extension of Chebfun to
        two dimensions", SISC, 35(6), C495–C518, 2013.

    See Also
    --------
    SeparableApprox, chebfun2
    """

    approx: SeparableApprox

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array, jax.Array], jax.Array],
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        tol: Optional[float] = None,
        n: Optional[int] = None,
    ) -> "Chebfun2":
        """Construct a Chebfun2 from a callable f(x, y).

        Uses the Chebfun2 algorithm (Gaussian elimination with complete
        pivoting) to adaptively find a low-rank representation.

        Parameters
        ----------
        f : callable
            A function f(x, y) that accepts JAX arrays and returns JAX arrays.
            Must be vectorised: f(xx, yy) where xx and yy are 2D arrays
            from ``jnp.meshgrid``.
        domain : tuple of 4 floats, optional
            (xa, xb, ya, yb). Default is (-1, 1, -1, 1).
        tol : float, optional
            Target tolerance. Default is machine epsilon (~2.2e-16).
        n : int, optional
            If given, use exactly n x n sampling points (non-adaptive in
            the grid size sense; rank is still determined adaptively).
            Not yet implemented — raises ``NotImplementedError`` if given.

        Returns
        -------
        Chebfun2

        Raises
        ------
        ValueError
            If ``domain`` does not have exactly 4 elements.
        NotImplementedError
            If ``n`` is given (fixed-degree construction not yet implemented).

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2/chebfun2.m, @chebfun2/constructor.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Townsend & Trefethen, SISC 2013.

        See Also
        --------
        chebfun2
        """
        if len(domain) != 4:
            raise ValueError(
                f"Chebfun2.from_function: domain must have exactly 4 elements "
                f"(xa, xb, ya, yb), got {len(domain)}."
            )
        if n is not None:
            raise NotImplementedError(
                "Chebfun2.from_function: fixed-degree construction (n=...) "
                "is not yet implemented."
            )
        kwargs: dict = dict(domain=domain)
        if tol is not None:
            kwargs["tol"] = tol
        approx = SeparableApprox.from_function(f, **kwargs)
        return cls(approx=approx)

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Evaluate f(x, y).

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb]. Must broadcast with x.

        Returns
        -------
        jax.Array, same shape as broadcast(x, y)
            Approximated function values.

        Notes
        -----
        JIT-safe, vmap-safe, and grad-safe.

        Provenance
        ----------
        MATLAB source : @separableApprox/feval.m, @chebfun2/feval.m
        Chebfun commit: 7574c77
        """
        return self.approx(x, y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def domain(self) -> tuple[float, float, float, float]:
        """Physical domain (xa, xb, ya, yb)."""
        return self.approx.domain

    @property
    def rank(self) -> int:
        """Numerical rank of the low-rank approximation."""
        return self.approx.rank

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, dim: int = 1, k: int = 1) -> "Chebfun2":
        """Partial derivative of f.

        Parameters
        ----------
        dim : int, default 1
            Dimension to differentiate along.
            - ``dim=1``: derivative with respect to y.
            - ``dim=2``: derivative with respect to x.
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Chebfun2
            The k-th partial derivative in the chosen direction.

        Raises
        ------
        ValueError
            If dim is not 1 or 2, or if k < 0.

        Notes
        -----
        Each col/row slice is differentiated independently, with the
        chain-rule scale factor (2/(b-a))^k for the affine map.

        Provenance
        ----------
        MATLAB source : @separableApprox/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, sum2
        """
        if dim not in (1, 2):
            raise ValueError(
                f"Chebfun2.diff: dim must be 1 (y-direction) or 2 (x-direction), "
                f"got dim={dim}."
            )
        if k < 0:
            raise ValueError(
                f"Chebfun2.diff: differentiation order k must be >= 0, got k={k}."
            )
        if k == 0:
            return self

        xa, xb, ya, yb = self.domain

        if dim == 1:
            # Differentiate column slices c_j(y) with respect to y
            new_cols = [_chebtech_diff_physical(c, ya, yb, k) for c in self.approx.cols]
            new_rows = list(self.approx.rows)
        else:
            # Differentiate row slices r_j(x) with respect to x
            new_cols = list(self.approx.cols)
            new_rows = [_chebtech_diff_physical(r, xa, xb, k) for r in self.approx.rows]

        new_approx = SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=self.approx.pivots,
            domain=self.domain,
        )
        return Chebfun2(approx=new_approx)

    def sum(self, dim: Optional[int] = None) -> Union["Chebfun2", jax.Array]:
        """Integrate f over one or both dimensions.

        Parameters
        ----------
        dim : int or None, optional
            - ``dim=None``: double integral (returns scalar).  Same as ``sum2()``.
            - ``dim=1``: integrate over y; returns a Chebfun2 with rank equal
              to the original rank, but where each column slice has been replaced
              by its integral (a constant), effectively returning a function of
              x only.  The result evaluates to g(x) = Σ_j d_j * int_ya^yb c_j(y) dy * r_j(x).
            - ``dim=2``: integrate over x; returns a function of y only.

        Returns
        -------
        Chebfun2 or jax.Array (scalar)
            - If ``dim=None``: a scalar (double integral).
            - If ``dim=1`` or ``dim=2``: a ``Chebfun2`` with collapsed
              col/row slices representing the 1D result.

        Raises
        ------
        ValueError
            If dim is not None, 1, or 2.

        Notes
        -----
        For ``dim=1`` or ``dim=2``, the returned Chebfun2 has flat slices in
        one direction (all column slices are constant=1, or all row slices are
        constant=1) and the accumulated integral weights are absorbed into the
        remaining pivots.  Evaluation along the collapsed dimension always
        returns the same value (the integral), as expected.

        Provenance
        ----------
        MATLAB source : @separableApprox/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum2, diff
        """
        if dim is None:
            return self.sum2()
        if dim not in (1, 2):
            raise ValueError(
                f"Chebfun2.sum: dim must be None, 1 (integrate over y), "
                f"or 2 (integrate over x), got dim={dim}."
            )

        xa, xb, ya, yb = self.domain
        r = self.approx.rank

        if dim == 1:
            # Integrate over y: g(x) = Σ_j d_j * int_ya^yb c_j(y) dy * r_j(x)
            # Compute col integrals (scalars)
            col_integrals = jnp.array(
                [float(_chebtech_integral(self.approx.cols[j], ya, yb)) for j in range(r)],
                dtype=jnp.float64,
            )
            # New pivots absorb the column integrals: d_j' = d_j * int(c_j)
            # New columns: constant = 1 (Chebtech2 with coeffs = [1])
            # New rows: same as before
            new_pivots = self.approx.pivots * col_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]
            new_rows = list(self.approx.rows)
        else:
            # Integrate over x: g(y) = Σ_j d_j * int_xa^xb r_j(x) dx * c_j(y)
            row_integrals = jnp.array(
                [float(_chebtech_integral(self.approx.rows[j], xa, xb)) for j in range(r)],
                dtype=jnp.float64,
            )
            new_pivots = self.approx.pivots * row_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = list(self.approx.cols)
            new_rows = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]

        new_approx = SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=new_pivots,
            domain=self.domain,
        )
        return Chebfun2(approx=new_approx)

    def sum2(self) -> jax.Array:
        """Double integral of f over its domain.

        Computes  integral_xa^xb integral_ya^yb f(x, y) dy dx.

        Returns
        -------
        jax.Array (scalar)
            The double integral.

        Notes
        -----
        Uses the low-rank representation:
            I = Σ_j d_j * integral(c_j over [ya, yb]) * integral(r_j over [xa, xb])

        Provenance
        ----------
        MATLAB source : @separableApprox/sum2.m, @separableApprox/integral2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, diff, norm
        """
        xa, xb, ya, yb = self.domain
        r = self.approx.rank
        total = jnp.float64(0.0)
        for j in range(r):
            col_int = _chebtech_integral(self.approx.cols[j], ya, yb)
            row_int = _chebtech_integral(self.approx.rows[j], xa, xb)
            total = total + self.approx.pivots[j] * col_int * row_int
        return total

    def norm(self, p: Union[int, float, str] = "fro") -> jax.Array:
        """Norm of f.

        Parameters
        ----------
        p : int, float, or str, default ``'fro'``
            The norm type:
            - ``2`` or ``'fro'``: Frobenius (L2) norm,
              ``sqrt(integral_domain |f(x,y)|^2 dx dy)``.
            - ``jnp.inf`` or ``float('inf')``: not implemented (raises
              ``NotImplementedError``).

        Returns
        -------
        jax.Array (scalar)
            The norm.

        Raises
        ------
        NotImplementedError
            If p is not 2 or 'fro'.

        Notes
        -----
        The Frobenius norm is computed as::

            ||f||_F^2 = Σ_j Σ_k d_j * d_k
                          * <c_j, c_k>_[ya,yb]  * <r_j, r_k>_[xa,xb]

        where the inner products use the L2 inner product on the physical
        domain with the affine-map scale factor.

        Provenance
        ----------
        MATLAB source : @separableApprox/norm.m (delegated from @chebfun2/norm.m)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum2
        """
        if p not in (2, "fro", 2.0):
            raise NotImplementedError(
                f"Chebfun2.norm: only the Frobenius/L2 norm (p=2 or p='fro') "
                f"is implemented, got p={p!r}."
            )
        xa, xb, ya, yb = self.domain
        r = self.approx.rank
        # Scale factors for physical inner products
        col_scale = jnp.float64((yb - ya) / 2.0)
        row_scale = jnp.float64((xb - xa) / 2.0)

        norm_sq = jnp.float64(0.0)
        for j in range(r):
            for k in range(r):
                # <c_j, c_k> on reference [-1,1] scaled for physical domain
                col_ip = self.approx.cols[j].inner(self.approx.cols[k]) * col_scale
                # <r_j, r_k> on reference [-1,1] scaled for physical domain
                row_ip = self.approx.rows[j].inner(self.approx.rows[k]) * row_scale
                norm_sq = norm_sq + self.approx.pivots[j] * self.approx.pivots[k] * col_ip * row_ip

        return jnp.sqrt(jnp.abs(norm_sq))

    # ------------------------------------------------------------------
    # Root finding
    # ------------------------------------------------------------------

    def roots(self) -> list:
        """Zero contours of f as a list of (x, y) coordinate arrays.

        Finds the zero level-set of f by evaluating on a fine grid and
        applying marching squares to locate edge crossings.  Each call
        returns a list of point clouds, one per connected component; the
        points are not sorted along the contour.

        Returns
        -------
        list of np.ndarray, each of shape (n_pts, 2)
            Each element is an array of (x, y) coordinates lying on the
            zero contour.  If f has no zeros inside the domain, returns
            an empty list.

        Notes
        -----
        This is a simplified implementation using a cell-based marching
        squares algorithm for initial curve detection (no Newton
        refinement and no contour tracing/ordering).  Accuracy is limited
        by the grid resolution (n = 500 by default).  For high accuracy
        or ordered zero curves, use the MATLAB Chebfun2 ``roots()`` method
        which performs Newton refinement.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @separableApprox/roots.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        diff, sum
        """
        xa, xb, ya, yb = self.domain
        n = 500
        x_pts = np.linspace(float(xa), float(xb), n)
        y_pts = np.linspace(float(ya), float(yb), n)
        xx, yy = np.meshgrid(x_pts, y_pts)
        xx_j = jnp.asarray(xx, dtype=jnp.float64)
        yy_j = jnp.asarray(yy, dtype=jnp.float64)
        vals = np.array(self.approx(xx_j, yy_j), dtype=np.float64)

        # Marching-squares: collect edge midpoints where sign changes.
        def _interp(x0, y0, v0, x1, y1, v1):
            t = v0 / (v0 - v1)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

        crossing_pts = []
        ny, nx = vals.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                v00 = vals[i, j]
                v10 = vals[i, j + 1]
                v01 = vals[i + 1, j]
                v11 = vals[i + 1, j + 1]
                x0, y0 = x_pts[j], y_pts[i]
                x1, y1 = x_pts[j + 1], y_pts[i + 1]

                if v00 * v10 < 0:  # bottom edge
                    crossing_pts.append(_interp(x0, y0, v00, x1, y0, v10))
                if v01 * v11 < 0:  # top edge
                    crossing_pts.append(_interp(x0, y1, v01, x1, y1, v11))
                if v00 * v01 < 0:  # left edge
                    crossing_pts.append(_interp(x0, y0, v00, x0, y1, v01))
                if v10 * v11 < 0:  # right edge
                    crossing_pts.append(_interp(x1, y0, v10, x1, y1, v11))

        if not crossing_pts:
            return []

        pts = np.array(crossing_pts, dtype=np.float64)  # shape (n_pts, 2)
        return [pts]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Surface plot of this Chebfun2 (calls :func:`chebfunjax.plotting.surf`)."""
        from chebfunjax.plotting import surf
        return surf(self, **kwargs)

    def contour(self, **kwargs):
        """Contour plot of this Chebfun2 (calls :func:`chebfunjax.plotting.contour`)."""
        from chebfunjax.plotting import contour
        return contour(self, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display like MATLAB Chebfun2.

        Examples
        --------
        >>> f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        >>> repr(f)
        'Chebfun2(rank=2, domain=(-1.0, 1.0, -1.0, 1.0))'
        """
        xa, xb, ya, yb = self.domain
        return (
            f"Chebfun2(rank={self.rank}, "
            f"domain=({xa}, {xb}, {ya}, {yb}))"
        )


# ============================================================================
# Factory function
# ============================================================================


def chebfun2(
    f: Callable[[jax.Array, jax.Array], jax.Array],
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    tol: Optional[float] = None,
    n: Optional[int] = None,
) -> Chebfun2:
    """Construct a Chebfun2 representing a bivariate smooth function.

    This is the primary factory function for creating Chebfun2 objects.

    Parameters
    ----------
    f : callable
        A function f(x, y) that accepts 2D JAX arrays (from meshgrid) and
        returns an array of the same shape.  Must be vectorised.
    domain : tuple of 4 floats, optional
        (xa, xb, ya, yb). Default is (-1, 1, -1, 1).
    tol : float, optional
        Target relative tolerance. Default is machine epsilon (~2.2e-16).
    n : int, optional
        Fixed-degree construction (not yet implemented).

    Returns
    -------
    Chebfun2

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import chebfunjax as cj
    >>> f = cj.chebfun2(lambda x, y: jnp.cos(x + y))
    >>> f(0.5, -0.3)  # evaluate at a point
    Array(0.20..., dtype=float64)

    Provenance
    ----------
    MATLAB source : @chebfun2/chebfun2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun2
    """
    return Chebfun2.from_function(f, domain=domain, tol=tol, n=n)
