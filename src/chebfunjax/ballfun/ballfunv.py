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
        n: int = 15,
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Ballfunv":
        """Construct a Ballfunv from three callables.

        Parameters
        ----------
        f : callable
            f(lam, theta, r) — first scalar component.
        g : callable
            g(lam, theta, r) — second scalar component.
        h : callable
            h(lam, theta, r) — third scalar component.
        n : int, optional
            Grid size per direction. Default 15.
        tol : float, optional
            Tolerance for Ballfun construction.

        Returns
        -------
        Ballfunv
        """
        return cls(
            Ballfun.from_function(f, n=n, tol=tol),
            Ballfun.from_function(g, n=n, tol=tol),
            Ballfun.from_function(h, n=n, tol=tol),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(
        self, lam: jax.Array, theta: jax.Array, r: jax.Array
    ) -> tuple:
        """Evaluate all three components at (lam, theta, r).

        Parameters
        ----------
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        theta : jax.Array
            Colatitude(s) in [0, pi].
        r : jax.Array
            Radius/radii in [0, 1].

        Returns
        -------
        tuple
            (f_val, g_val, h_val) evaluated at (lam, theta, r).
        """
        f, g, h = self.components
        return (f(lam, theta, r), g(lam, theta, r), h(lam, theta, r))

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
        n = f1.n_r
        return Ballfun.from_function(
            lambda lam, th, r: (
                f1(lam, th, r) * f2(lam, th, r)
                + g1(lam, th, r) * g2(lam, th, r)
                + h1(lam, th, r) * h2(lam, th, r)
            ),
            n=n,
        )

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
        n = f1.n_r
        cx = Ballfun.from_function(
            lambda lam, th, r: g1(lam, th, r) * h2(lam, th, r) - h1(lam, th, r) * g2(lam, th, r),
            n=n,
        )
        cy = Ballfun.from_function(
            lambda lam, th, r: h1(lam, th, r) * f2(lam, th, r) - f1(lam, th, r) * h2(lam, th, r),
            n=n,
        )
        cz = Ballfun.from_function(
            lambda lam, th, r: f1(lam, th, r) * g2(lam, th, r) - g1(lam, th, r) * f2(lam, th, r),
            n=n,
        )
        return Ballfunv(cx, cy, cz)

    def norm(self) -> Ballfun:
        """Pointwise Euclidean norm: sqrt(f^2 + g^2 + h^2).

        Returns
        -------
        Ballfun
            Scalar norm field.
        """
        f, g, h = self.components
        n = f.n_r
        return Ballfun.from_function(
            lambda lam, th, r: jnp.sqrt(
                f(lam, th, r) ** 2 + g(lam, th, r) ** 2 + h(lam, th, r) ** 2
            ),
            n=n,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Ballfunv") -> "Ballfunv":
        """Componentwise addition."""
        f1, g1, h1 = self.components
        f2, g2, h2 = other.components
        n = f1.n_r
        return Ballfunv(
            Ballfun.from_function(lambda lam, th, r: f1(lam, th, r) + f2(lam, th, r), n=n),
            Ballfun.from_function(lambda lam, th, r: g1(lam, th, r) + g2(lam, th, r), n=n),
            Ballfun.from_function(lambda lam, th, r: h1(lam, th, r) + h2(lam, th, r), n=n),
        )

    def __mul__(self, scalar: float) -> "Ballfunv":
        """Scalar multiplication (componentwise)."""
        f, g, h = self.components
        s = float(scalar)
        n = f.n_r
        return Ballfunv(
            Ballfun.from_function(lambda lam, th, r: s * f(lam, th, r), n=n),
            Ballfun.from_function(lambda lam, th, r: s * g(lam, th, r), n=n),
            Ballfun.from_function(lambda lam, th, r: s * h(lam, th, r), n=n),
        )

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
