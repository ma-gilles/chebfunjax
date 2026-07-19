"""Spherefunv — vector field on the unit sphere (Spherefun components).

A Spherefunv holds an ordered list of :class:`Spherefun` scalar components.
Two representations coexist:

- The MATLAB-faithful **3-Cartesian-component** field ``[fx, fy, fz]`` — the
  x/y/z components of a vector field on (or tangent to) the unit sphere.
  This is what ``Spherefun.gradient`` returns and what the surface
  differential operators (curl, divergence, vorticity, cross, normal,
  tangent) act on.
- A **2-component** field ``[f, g]`` — a lighter intrinsic-tangent pair kept
  for backward compatibility with existing callers.

All componentwise operations (arithmetic, real/imag/conj, dot, norm,
evaluation) work for any number of components; the surface differential
operators require the 3-Cartesian form.

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


def _sphere_xyz() -> tuple[Spherefun, Spherefun, Spherefun]:
    """The three Cartesian coordinate functions x, y, z as Spherefuns.

    On the unit sphere x = cos(lam) sin(th), y = sin(lam) sin(th),
    z = cos(th); together they form the outward unit normal field.
    """
    x = Spherefun.from_function(lambda lam, th: jnp.cos(lam) * jnp.sin(th))
    y = Spherefun.from_function(lambda lam, th: jnp.sin(lam) * jnp.sin(th))
    z = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    return x, y, z


class Spherefunv(eqx.Module):
    """Vector field on the unit sphere with 2 or 3 scalar Spherefun components.

    Constructed either as ``Spherefunv(fx, fy, fz)`` (MATLAB 3-Cartesian
    form) or ``Spherefunv(f, g)`` (2-component form).  Each component is a
    :class:`Spherefun` in intrinsic coordinates ``(lambda, theta)`` with
    lambda the longitude in [-pi, pi] and theta the colatitude in [0, pi].

    Attributes
    ----------
    components : list of Spherefun
        The scalar components (length 2 or 3).

    Provenance
    ----------
    MATLAB source : @spherefunv/spherefunv.m
    Chebfun commit: 7574c77
    """

    @classmethod
    def empty(cls) -> "Spherefunv":
        """The empty Spherefunv (MATLAB spherefunv()).

        Provenance
        ----------
        MATLAB source : @spherefunv/spherefunv.m (empty branch)
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Spherefunv (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @spherefunv/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    components: list  # [fx, fy, fz] or [f, g]

    def __init__(self, *components: Spherefun) -> None:
        """Create a Spherefunv from 2 or 3 Spherefun scalar components.

        Parameters
        ----------
        *components : Spherefun
            The scalar components.  Pass three (fx, fy, fz) for the MATLAB
            Cartesian form or two (f, g) for the intrinsic-tangent form.
        """
        if len(components) not in (2, 3):
            raise ValueError(
                "Spherefunv takes 2 or 3 Spherefun components, got "
                f"{len(components)}.")
        self.components = list(components)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_functions(
        cls,
        *fns: Callable,
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Spherefunv":
        """Construct a Spherefunv from 2 or 3 callables ``f(lam, theta)``.

        Parameters
        ----------
        *fns : callable
            Component callables ``f(lam, theta)``.
        tol : float, optional
            Tolerance for each Spherefun construction.

        Returns
        -------
        Spherefunv
        """
        return cls(*[Spherefun.from_function(fn, tol=tol) for fn in fns])

    # ------------------------------------------------------------------
    # Static constructors
    # ------------------------------------------------------------------

    @staticmethod
    def unormal() -> "Spherefunv":
        """Unit outward normal vector field of the unit sphere ``(x, y, z)``.

        Provenance
        ----------
        MATLAB source : @spherefunv/unormal.m
        Chebfun commit: 7574c77
        """
        x, y, z = _sphere_xyz()
        return Spherefunv(x, y, z)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, lam: jax.Array, theta: jax.Array) -> tuple:
        """Evaluate every component at spherical coordinates (lam, theta).

        Returns a tuple with one entry per component (length 2 or 3).

        Parameters
        ----------
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        theta : jax.Array
            Colatitude(s) in [0, pi].
        """
        return tuple(c(lam, theta) for c in self.components)

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Spherefunv") -> Spherefun:
        """Vector dot product ``sum_j f_j * g_j`` (MATLAB dot).

        Provenance
        ----------
        MATLAB source : @spherefunv/dot.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return Spherefun.empty()
        prods = [a * b for a, b in zip(self.components, other.components)]
        out = prods[0]
        for c in prods[1:]:
            out = out + c
        return out

    def cross(self, other: "Spherefunv") -> "Spherefunv":
        """3D vector cross product (MATLAB cross).

        Both fields must have three Cartesian components.

        Provenance
        ----------
        MATLAB source : @spherefunv/cross.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return Spherefunv.empty()
        if len(self.components) != 3 or len(other.components) != 3:
            raise ValueError("cross requires 3-component Spherefunv inputs.")
        a1, a2, a3 = self.components
        b1, b2, b3 = other.components
        return Spherefunv(
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1,
        )

    def norm(self) -> Spherefun:
        """Pointwise Euclidean magnitude ``sqrt(sum_j f_j^2)`` as a Spherefun.

        Note: chebfunjax returns the pointwise magnitude field (a Spherefun),
        not MATLAB's scalar Frobenius norm.
        """
        comps = self.components
        return Spherefun.from_function(
            lambda lam, th: jnp.sqrt(
                sum(c(lam, th) ** 2 for c in comps)))

    # ------------------------------------------------------------------
    # Surface differential operators (3-Cartesian-component fields)
    # ------------------------------------------------------------------

    def _require3(self, name: str) -> tuple:
        if len(self.components) != 3:
            raise ValueError(
                f"Spherefunv.{name} requires a 3-Cartesian-component field.")
        return tuple(self.components)

    def curl(self) -> "Spherefunv":
        r"""Surface curl of a (tangential) vector field (MATLAB curl).

        Returns the SPHEREFUNV whose components are the tangential
        derivatives ``(d_y f_z - d_z f_y, d_z f_x - d_x f_z,
        d_x f_y - d_y f_x)``.

        Provenance
        ----------
        MATLAB source : @spherefunv/curl.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Spherefunv.empty()
        fx, fy, fz = self._require3("curl")
        gx = fz.diff(2) - fy.diff(3)
        gy = fx.diff(3) - fz.diff(1)
        gz = fy.diff(1) - fx.diff(2)
        return Spherefunv(gx, gy, gz)

    def divergence(self) -> Spherefun:
        r"""Surface divergence ``d_x f_x + d_y f_y + d_z f_z`` (MATLAB
        divergence).  Only meaningful for tangential fields.

        Provenance
        ----------
        MATLAB source : @spherefunv/divergence.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Spherefun.empty()
        fx, fy, fz = self._require3("divergence")
        return fx.diff(1) + fy.diff(2) + fz.diff(3)

    def div(self) -> Spherefun:
        """Shorthand for :meth:`divergence` (MATLAB div).

        Provenance
        ----------
        MATLAB source : @spherefunv/div.m
        Chebfun commit: 7574c77
        """
        return self.divergence()

    def vorticity(self) -> Spherefun:
        r"""Surface vorticity ``N . curl(F)`` — the normal component of the
        surface curl (MATLAB vorticity).

        Provenance
        ----------
        MATLAB source : @spherefunv/vorticity.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Spherefun.empty()
        return Spherefunv.unormal().dot(self.curl())

    def vort(self) -> Spherefun:
        """Shorthand for :meth:`vorticity` (MATLAB vort).

        Provenance
        ----------
        MATLAB source : @spherefunv/vort.m
        Chebfun commit: 7574c77
        """
        return self.vorticity()

    def normal(self) -> "Spherefunv":
        """Projection of the field onto the sphere's normal direction
        ``(F . N) N`` (MATLAB normal).

        Provenance
        ----------
        MATLAB source : @spherefunv/normal.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Spherefunv.empty()
        self._require3("normal")
        N = Spherefunv.unormal()
        f = self.dot(N)
        return Spherefunv(*[c * f for c in N.components])

    def tangent(self) -> "Spherefunv":
        """Projection of the field onto the tangent plane ``F - normal(F)``
        (MATLAB tangent).

        Provenance
        ----------
        MATLAB source : @spherefunv/tangent.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return self - self.normal()

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Spherefunv") -> "Spherefunv":
        """Componentwise addition."""
        return Spherefunv(*[a + b for a, b in
                            zip(self.components, other.components)])

    def times(self, other) -> "Spherefunv":
        """Componentwise product with another Spherefunv, or scaling by
        a scalar field / scalar (MATLAB times).

        Provenance
        ----------
        MATLAB source : @spherefunv/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Spherefunv):
            return Spherefunv(*[a * b for a, b in
                                zip(self.components, other.components)])
        return Spherefunv(*[a * other for a in self.components])

    def power(self, n: int) -> "Spherefunv":
        """Componentwise power (MATLAB power).

        Provenance
        ----------
        MATLAB source : @spherefunv/power.m
        Chebfun commit: 7574c77
        """
        return Spherefunv(*[a ** n for a in self.components])

    def real(self) -> "Spherefunv":
        """Componentwise real part (MATLAB real).

        Provenance
        ----------
        MATLAB source : @spherefunv/real.m
        Chebfun commit: 7574c77
        """
        return Spherefunv(*[c.real() for c in self.components])

    def imag(self) -> "Spherefunv":
        """Componentwise imaginary part (MATLAB imag).

        Provenance
        ----------
        MATLAB source : @spherefunv/imag.m
        Chebfun commit: 7574c77
        """
        return Spherefunv(*[c.imag() for c in self.components])

    def conj(self) -> "Spherefunv":
        """Componentwise complex conjugate (MATLAB conj).

        Provenance
        ----------
        MATLAB source : @spherefunv/conj.m
        Chebfun commit: 7574c77
        """
        return Spherefunv(*[c.conj() for c in self.components])

    def iszero(self) -> bool:
        """True iff every component is the zero function (MATLAB iszero).

        Provenance
        ----------
        MATLAB source : @spherefunv/iszero.m
        Chebfun commit: 7574c77
        """
        return all(c.iszero() for c in self.components)

    def __mul__(self, scalar: float) -> "Spherefunv":
        """Scalar multiplication (componentwise)."""
        s = float(scalar)
        return Spherefunv(*[c * s for c in self.components])

    def __rmul__(self, scalar: float) -> "Spherefunv":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __neg__(self) -> "Spherefunv":
        """Negation."""
        return Spherefunv(*[-c for c in self.components])

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

    def quiver(self, **kwargs):
        """Quiver plot of this vector field on the sphere (calls :func:`chebfunjax.plotting.quiver_sphere`)."""
        from chebfunjax.plotting import quiver_sphere
        return quiver_sphere(self, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.components)
        lines = [f"Spherefunv with {n} components:"]
        for i, c in enumerate(self.components):
            lines.append(f"  [{i}]: {c!r}")
        return "\n".join(lines)
