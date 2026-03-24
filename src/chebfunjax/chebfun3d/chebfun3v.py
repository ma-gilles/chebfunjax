"""Chebfun3v — 3D vector field (thin wrapper over Chebfun3 components).

A Chebfun3v holds three Chebfun3 scalar components (f, g, h), representing
a 3-component vector field on a 3D cuboid.  This mirrors the MATLAB
@chebfun3v class.

Provenance
----------
MATLAB source : @chebfun3v (commit 7574c77)
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class Chebfun3v(eqx.Module):
    """Vector field on a 3D cuboid with 3 scalar Chebfun3 components.

    Represents [f(x,y,z), g(x,y,z), h(x,y,z)] where f, g, h are Chebfun3
    objects on the same domain.

    Attributes
    ----------
    components : list
        [f, g, h] — three Chebfun3 scalar fields.

    Provenance
    ----------
    MATLAB source : @chebfun3v/chebfun3v.m
    Chebfun commit: 7574c77
    """

    components: list  # [f, g, h]

    def __init__(self, f: Chebfun3, g: Chebfun3, h: Chebfun3) -> None:
        """Create a Chebfun3v from three Chebfun3 scalar components.

        Parameters
        ----------
        f : Chebfun3
            x-component.
        g : Chebfun3
            y-component.
        h : Chebfun3
            z-component.
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
        domain: tuple = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Chebfun3v":
        """Construct a Chebfun3v from three callables.

        Parameters
        ----------
        f : callable
            f(x, y, z) — first scalar component.
        g : callable
            g(x, y, z) — second scalar component.
        h : callable
            h(x, y, z) — third scalar component.
        domain : 6-tuple, optional
            (xa, xb, ya, yb, za, zb).  Default is the unit cube.
        tol : float, optional
            Tolerance for Chebfun3 construction.

        Returns
        -------
        Chebfun3v
        """
        return cls(
            Chebfun3.from_function(f, domain=domain, tol=tol),
            Chebfun3.from_function(g, domain=domain, tol=tol),
            Chebfun3.from_function(h, domain=domain, tol=tol),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(
        self, x: jax.Array, y: jax.Array, z: jax.Array
    ) -> tuple:
        """Evaluate all three components at (x, y, z).

        Parameters
        ----------
        x : jax.Array
            x-coordinate(s) in [xa, xb].
        y : jax.Array
            y-coordinate(s) in [ya, yb].
        z : jax.Array
            z-coordinate(s) in [za, zb].

        Returns
        -------
        tuple
            (f_val, g_val, h_val) evaluated at (x, y, z).
        """
        f, g, h = self.components
        return (f(x, y, z), g(x, y, z), h(x, y, z))

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Chebfun3v") -> Chebfun3:
        """Dot product of two Chebfun3v: f1*f2 + g1*g2 + h1*h2.

        Parameters
        ----------
        other : Chebfun3v
            Second vector field.

        Returns
        -------
        Chebfun3
            Scalar dot product field.
        """
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        dom = f1.domain
        return Chebfun3.from_function(
            lambda x, y, z: (
                f1(x, y, z) * f2(x, y, z)
                + g1(x, y, z) * g2(x, y, z)
                + h1(x, y, z) * h2(x, y, z)
            ),
            domain=dom,
        )

    def cross(self, other: "Chebfun3v") -> "Chebfun3v":
        """Cross product of two 3D Chebfun3v fields.

        Returns the vector field (g1*h2 - h1*g2, h1*f2 - f1*h2, f1*g2 - g1*f2).

        Parameters
        ----------
        other : Chebfun3v
            Second vector field.

        Returns
        -------
        Chebfun3v
            Cross product vector field.
        """
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        dom = f1.domain
        cx = Chebfun3.from_function(
            lambda x, y, z: g1(x, y, z) * h2(x, y, z) - h1(x, y, z) * g2(x, y, z),
            domain=dom,
        )
        cy = Chebfun3.from_function(
            lambda x, y, z: h1(x, y, z) * f2(x, y, z) - f1(x, y, z) * h2(x, y, z),
            domain=dom,
        )
        cz = Chebfun3.from_function(
            lambda x, y, z: f1(x, y, z) * g2(x, y, z) - g1(x, y, z) * f2(x, y, z),
            domain=dom,
        )
        return Chebfun3v(cx, cy, cz)

    def norm(self) -> Chebfun3:
        """Pointwise Euclidean norm: sqrt(f^2 + g^2 + h^2).

        Returns
        -------
        Chebfun3
            Scalar norm field.
        """
        f, g, h = self.components
        dom = f.domain
        return Chebfun3.from_function(
            lambda x, y, z: jnp.sqrt(
                f(x, y, z) ** 2 + g(x, y, z) ** 2 + h(x, y, z) ** 2
            ),
            domain=dom,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Chebfun3v") -> "Chebfun3v":
        """Componentwise addition."""
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        dom = f1.domain
        return Chebfun3v(
            Chebfun3.from_function(lambda x, y, z: f1(x, y, z) + f2(x, y, z), domain=dom),
            Chebfun3.from_function(lambda x, y, z: g1(x, y, z) + g2(x, y, z), domain=dom),
            Chebfun3.from_function(lambda x, y, z: h1(x, y, z) + h2(x, y, z), domain=dom),
        )

    def __mul__(self, scalar: float) -> "Chebfun3v":
        """Scalar multiplication (componentwise)."""
        f, g, h = self.components
        s = float(scalar)
        dom = f.domain
        return Chebfun3v(
            Chebfun3.from_function(lambda x, y, z: s * f(x, y, z), domain=dom),
            Chebfun3.from_function(lambda x, y, z: s * g(x, y, z), domain=dom),
            Chebfun3.from_function(lambda x, y, z: s * h(x, y, z), domain=dom),
        )

    def __rmul__(self, scalar: float) -> "Chebfun3v":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __neg__(self) -> "Chebfun3v":
        """Negation."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "Chebfun3v") -> "Chebfun3v":
        """Componentwise subtraction."""
        return self.__add__(other.__neg__())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        f, g, h = self.components
        return (
            f"Chebfun3v with 3 components:\n"
            f"  [0]: {f!r}\n"
            f"  [1]: {g!r}\n"
            f"  [2]: {h!r}"
        )
