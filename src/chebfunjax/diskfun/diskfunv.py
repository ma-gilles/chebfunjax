"""Diskfunv — vector field on the unit disk (thin wrapper over Diskfun components).

A Diskfunv holds a pair of Diskfun scalar components (f, g), representing
a 2-component vector field on the unit disk.  This mirrors the MATLAB
@diskfunv class.

Provenance
----------
MATLAB source : @diskfunv (commit 7574c77)
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


class Diskfunv(eqx.Module):
    """Vector field on the unit disk with 2 scalar Diskfun components.

    Represents [f(theta, r), g(theta, r)] where f and g are Diskfun objects.

    Attributes
    ----------
    components : list
        [f, g] — two Diskfun scalar fields.

    Provenance
    ----------
    MATLAB source : @diskfunv/diskfunv.m
    Chebfun commit: 7574c77
    """

    components: list  # [f, g]

    def __init__(self, f: Diskfun, g: Diskfun) -> None:
        """Create a Diskfunv from two Diskfun scalar components.

        Parameters
        ----------
        f : Diskfun
            First component (e.g. radial or x-component).
        g : Diskfun
            Second component (e.g. angular or y-component).
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
    ) -> "Diskfunv":
        """Construct a Diskfunv from two callables.

        Parameters
        ----------
        f : callable
            f(theta, r) — first scalar component.
        g : callable
            g(theta, r) — second scalar component.
        tol : float, optional
            Tolerance for Diskfun construction.

        Returns
        -------
        Diskfunv
        """
        return cls(
            Diskfun.from_function(f, tol=tol),
            Diskfun.from_function(g, tol=tol),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, theta: jax.Array, r: jax.Array) -> tuple:
        """Evaluate both components at polar coordinates (theta, r).

        Parameters
        ----------
        theta : jax.Array
            Angle(s) in [-pi, pi].
        r : jax.Array
            Radius/radii in [0, 1].

        Returns
        -------
        tuple
            (f_val, g_val) evaluated at (theta, r).
        """
        f, g = self.components
        return (f(theta, r), g(theta, r))

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Diskfunv") -> Diskfun:
        """Dot product of two Diskfunv: f1*f2 + g1*g2.

        Parameters
        ----------
        other : Diskfunv
            Second vector field.

        Returns
        -------
        Diskfun
            Scalar dot product.
        """
        f1, g1 = self.components
        f2, g2 = other.components
        # Build dot product as a new Diskfun via lambda
        return Diskfun.from_function(
            lambda th, r: f1(th, r) * f2(th, r) + g1(th, r) * g2(th, r)
        )

    def norm(self) -> Diskfun:
        """Pointwise Euclidean norm: sqrt(f^2 + g^2).

        Returns
        -------
        Diskfun
            Scalar norm field.
        """
        f, g = self.components
        return Diskfun.from_function(
            lambda th, r: jnp.sqrt(f(th, r) ** 2 + g(th, r) ** 2)
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Diskfunv") -> "Diskfunv":
        """Componentwise addition."""
        f1, g1 = self.components
        f2, g2 = other.components
        return Diskfunv(
            Diskfun.from_function(lambda th, r: f1(th, r) + f2(th, r)),
            Diskfun.from_function(lambda th, r: g1(th, r) + g2(th, r)),
        )

    def __mul__(self, scalar: float) -> "Diskfunv":
        """Scalar multiplication (componentwise)."""
        f, g = self.components
        s = float(scalar)
        return Diskfunv(
            Diskfun.from_function(lambda th, r: s * f(th, r)),
            Diskfun.from_function(lambda th, r: s * g(th, r)),
        )

    def __rmul__(self, scalar: float) -> "Diskfunv":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __neg__(self) -> "Diskfunv":
        """Negation."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "Diskfunv") -> "Diskfunv":
        """Componentwise subtraction."""
        return self.__add__(other.__neg__())

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Quiver plot on the disk (calls :func:`chebfunjax.plotting.quiver_disk`)."""
        from chebfunjax.plotting import quiver_disk
        return quiver_disk(self, **kwargs)

    def quiver(self, **kwargs):
        """Quiver plot on the disk (calls :func:`chebfunjax.plotting.quiver_disk`)."""
        from chebfunjax.plotting import quiver_disk
        return quiver_disk(self, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        f, g = self.components
        return (
            f"Diskfunv with 2 components:\n"
            f"  [0]: {f!r}\n"
            f"  [1]: {g!r}"
        )
