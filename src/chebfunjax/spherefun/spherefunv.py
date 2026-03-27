"""Spherefunv — vector field on the unit sphere (thin wrapper over Spherefun components).

A Spherefunv holds a pair of Spherefun scalar components (f, g), representing
a 2-component vector field on the unit sphere (e.g. the longitudinal and
latitudinal components of a surface vector field).  This mirrors the MATLAB
@spherefunv class.

Provenance
----------
MATLAB source : @spherefunv (commit 7574c77)
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun


class Spherefunv(eqx.Module):
    """Vector field on the unit sphere with 2 scalar Spherefun components.

    Represents [f(lam, theta), g(lam, theta)] where f and g are Spherefun
    objects and (lam, theta) are longitude and colatitude.

    Attributes
    ----------
    components : list
        [f, g] — two Spherefun scalar fields.

    Provenance
    ----------
    MATLAB source : @spherefunv/spherefunv.m
    Chebfun commit: 7574c77
    """

    components: list  # [f, g]

    def __init__(self, f: Spherefun, g: Spherefun) -> None:
        """Create a Spherefunv from two Spherefun scalar components.

        Parameters
        ----------
        f : Spherefun
            First component (e.g. longitudinal component).
        g : Spherefun
            Second component (e.g. latitudinal component).
        """
        self.components = [f, g]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_functions(
        cls,
        f: Callable,
        g: Callable,
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Spherefunv":
        """Construct a Spherefunv from two callables.

        Parameters
        ----------
        f : callable
            f(lam, theta) — first scalar component.
        g : callable
            g(lam, theta) — second scalar component.
        tol : float, optional
            Tolerance for Spherefun construction.

        Returns
        -------
        Spherefunv
        """
        return cls(
            Spherefun.from_function(f, tol=tol),
            Spherefun.from_function(g, tol=tol),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, lam: jax.Array, theta: jax.Array) -> tuple:
        """Evaluate both components at spherical coordinates (lam, theta).

        Parameters
        ----------
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        theta : jax.Array
            Colatitude(s) in [0, pi].

        Returns
        -------
        tuple
            (f_val, g_val) evaluated at (lam, theta).
        """
        f, g = self.components
        return (f(lam, theta), g(lam, theta))

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Spherefunv") -> Spherefun:
        """Dot product of two Spherefunv: f1*f2 + g1*g2.

        Parameters
        ----------
        other : Spherefunv
            Second vector field.

        Returns
        -------
        Spherefun
            Scalar dot product.
        """
        f1, g1 = self.components
        f2, g2 = other.components
        return Spherefun.from_function(
            lambda lam, th: f1(lam, th) * f2(lam, th) + g1(lam, th) * g2(lam, th)
        )

    def norm(self) -> Spherefun:
        """Pointwise Euclidean norm: sqrt(f^2 + g^2).

        Returns
        -------
        Spherefun
            Scalar norm field.
        """
        f, g = self.components
        return Spherefun.from_function(
            lambda lam, th: jnp.sqrt(f(lam, th) ** 2 + g(lam, th) ** 2)
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Spherefunv") -> "Spherefunv":
        """Componentwise addition."""
        f1, g1 = self.components
        f2, g2 = other.components
        return Spherefunv(
            Spherefun.from_function(lambda lam, th: f1(lam, th) + f2(lam, th)),
            Spherefun.from_function(lambda lam, th: g1(lam, th) + g2(lam, th)),
        )

    def __mul__(self, scalar: float) -> "Spherefunv":
        """Scalar multiplication (componentwise)."""
        f, g = self.components
        s = float(scalar)
        return Spherefunv(
            Spherefun.from_function(lambda lam, th: s * f(lam, th)),
            Spherefun.from_function(lambda lam, th: s * g(lam, th)),
        )

    def __rmul__(self, scalar: float) -> "Spherefunv":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __neg__(self) -> "Spherefunv":
        """Negation."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "Spherefunv") -> "Spherefunv":
        """Componentwise subtraction."""
        return self.__add__(other.__neg__())

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Quiver plot of this vector field on the sphere (calls :func:`chebfunjax.plotting.quiver_sphere`)."""
        from chebfunjax.plotting import quiver_sphere
        return quiver_sphere(self, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        f, g = self.components
        return (
            f"Spherefunv with 2 components:\n"
            f"  [0]: {f!r}\n"
            f"  [1]: {g!r}"
        )
