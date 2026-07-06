"""Ballfunv — vector field on the unit ball (thin wrapper over Ballfun components).

A Ballfunv holds three Ballfun scalar components (f, g, h), representing
a 3-component vector field on the unit ball.  This mirrors the MATLAB
@ballfunv class.

Provenance
----------
MATLAB source : @ballfunv (commit 7574c77)
Original: Copyright 2019 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun


class Ballfunv(eqx.Module):
    """Vector field on the unit ball with 3 scalar Ballfun components.

    Represents [f(lam, theta, r), g(lam, theta, r), h(lam, theta, r)]
    where f, g, h are Ballfun objects on the same domain.

    Attributes
    ----------
    components : list
        [f, g, h] — three Ballfun scalar fields.

    Provenance
    ----------
    MATLAB source : @ballfunv/ballfunv.m
    Chebfun commit: 7574c77
    """

    components: list  # [f, g, h]

    def __init__(self, f: Ballfun, g: Ballfun, h: Ballfun) -> None:
        """Create a Ballfunv from three Ballfun scalar components.

        Parameters
        ----------
        f : Ballfun
            First component (e.g. lam-direction).
        g : Ballfun
            Second component (e.g. theta-direction).
        h : Ballfun
            Third component (e.g. r-direction).
        """
        self.components = [f, g, h]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_functions(
        cls,
        f: Callable,
        g: Callable,
        h: Callable,
        *,
        spherical: bool = False,
        fixed_size: "tuple[int, int, int] | None" = None,
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Ballfunv":
        """Construct a Ballfunv from three callables.

        Parameters
        ----------
        f, g, h : callable
            The three scalar components. By default each accepts CARTESIAN
            coordinates ``(x, y, z)``; with ``spherical=True`` each accepts
            spherical coordinates ``(r, lam, th)`` — the same convention as
            :meth:`Ballfun.from_function`.
        spherical : bool, optional
            If True, the callables are in spherical coordinates.
        fixed_size : tuple of int, optional
            Fixed (m, n, p) grid instead of adaptive construction.
        tol : float, optional
            Tolerance for Ballfun construction.

        Returns
        -------
        Ballfunv

        Provenance
        ----------
        MATLAB source : @ballfunv/ballfunv.m
        Chebfun commit: 7574c77
        """
        kw = dict(spherical=spherical, fixed_size=fixed_size, tol=tol)
        return cls(
            Ballfun.from_function(f, **kw),
            Ballfun.from_function(g, **kw),
            Ballfun.from_function(h, **kw),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(
        self, r: jax.Array, lam: jax.Array, th: jax.Array
    ) -> tuple:
        """Evaluate all three components at spherical (r, lam, th).

        Same argument order as :meth:`Ballfun.__call__`.

        Parameters
        ----------
        r : jax.Array
            Radius/radii in [0, 1].
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        th : jax.Array
            Colatitude(s) in [0, pi].

        Returns
        -------
        tuple
            (f_val, g_val, h_val) evaluated at (r, lam, th).
        """
        f, g, h = self.components
        return (f(r, lam, th), g(r, lam, th), h(r, lam, th))

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Ballfunv") -> Ballfun:
        """Dot product: f1*f2 + g1*g2 + h1*h2.

        Parameters
        ----------
        other : Ballfunv
            Second vector field.

        Returns
        -------
        Ballfun
            Scalar dot product field.
        """
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        return f1 * f2 + g1 * g2 + h1 * h2

    def cross(self, other: "Ballfunv") -> "Ballfunv":
        """Cross product of two 3D Ballfunv fields.

        Returns (g1*h2 - h1*g2, h1*f2 - f1*h2, f1*g2 - g1*f2).

        Parameters
        ----------
        other : Ballfunv
            Second vector field.

        Returns
        -------
        Ballfunv
            Cross product vector field.
        """
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        return Ballfunv(
            g1 * h2 - h1 * g2,
            h1 * f2 - f1 * h2,
            f1 * g2 - g1 * f2,
        )

    def div(self) -> Ballfun:
        r"""Divergence :math:`\\nabla \\cdot (f, g, h) = f_x + g_y + h_z`.

        Uses the Cartesian derivatives ``Ballfun.diff`` (dim 1/2/3 = x/y/z).
        Added by Claude Opus 4.8 (task #17).

        Provenance
        ----------
        MATLAB source : @ballfunv/div.m
        Chebfun commit: 7574c77
        """
        f, g, h = self.components
        return f.diff(1) + g.diff(2) + h.diff(3)

    divergence = div

    def curl(self) -> "Ballfunv":
        r"""Curl :math:`\\nabla \\times (f, g, h)`.

        Returns ``(h_y - g_z, f_z - h_x, g_x - f_y)``.  Added by Claude
        Opus 4.8 (task #17).

        Provenance
        ----------
        MATLAB source : @ballfunv/curl.m
        Chebfun commit: 7574c77
        """
        f, g, h = self.components
        return Ballfunv(
            h.diff(2) - g.diff(3),
            f.diff(3) - h.diff(1),
            g.diff(1) - f.diff(2),
        )

    def norm(self) -> float:
        """L2 norm of the field: sqrt(norm(f)^2 + norm(g)^2 + norm(h)^2).

        Matches MATLAB semantics — a scalar, not a pointwise field. For
        the pointwise magnitude use ``v.dot(v)`` and take a sqrt of its
        evaluations.

        Provenance
        ----------
        MATLAB source : @ballfunv/norm.m
        Chebfun commit: 7574c77
        """
        f, g, h = self.components
        return float(jnp.sqrt(f.norm() ** 2 + g.norm() ** 2 + h.norm() ** 2))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Ballfunv") -> "Ballfunv":
        """Componentwise addition."""
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        return Ballfunv(f1 + f2, g1 + g2, h1 + h2)

    def __mul__(self, scalar: float) -> "Ballfunv":
        """Scalar multiplication (componentwise)."""
        f, g, h = self.components
        return Ballfunv(f * scalar, g * scalar, h * scalar)

    def __rmul__(self, scalar: float) -> "Ballfunv":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __neg__(self) -> "Ballfunv":
        """Negation."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "Ballfunv") -> "Ballfunv":
        """Componentwise subtraction."""
        return self.__add__(other.__neg__())

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Quiver plot inside the ball (calls :func:`chebfunjax.plotting.quiver_ball`)."""
        from chebfunjax.plotting import quiver_ball
        return quiver_ball(self, **kwargs)

    def quiver(self, **kwargs):
        """Quiver plot inside the ball (calls :func:`chebfunjax.plotting.quiver_ball`)."""
        from chebfunjax.plotting import quiver_ball
        return quiver_ball(self, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        f, g, h = self.components
        return (
            f"Ballfunv with 3 components:\n"
            f"  [0]: {f!r}\n"
            f"  [1]: {g!r}\n"
            f"  [2]: {h!r}"
        )
