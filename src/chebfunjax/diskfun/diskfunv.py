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

    @classmethod
    def empty(cls) -> "Diskfunv":
        """The empty Diskfunv (MATLAB diskfunv()).

        Provenance
        ----------
        MATLAB source : @diskfunv/diskfunv.m (empty branch)
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Diskfunv (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @diskfunv/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

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

    def div(self) -> Diskfun:
        r"""Divergence :math:`\\nabla \\cdot (f, g) = f_x + g_y`.

        Uses the Cartesian derivatives ``Diskfun.diffx`` / ``diffy``.
        Added by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @diskfunv/div.m
        Chebfun commit: 7574c77
        """
        f, g = self.components
        fx = f.diffx()
        gy = g.diffy()
        return Diskfun.from_function(
            lambda th, r: fx(th, r) + gy(th, r)
        )

    divergence = div

    def curl(self) -> Diskfun:
        r"""Scalar curl :math:`g_x - f_y` of the 2-D vector field.

        Added by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @diskfunv/curl.m
        Chebfun commit: 7574c77
        """
        f, g = self.components
        gx = g.diffx()
        fy = f.diffy()
        return Diskfun.from_function(
            lambda th, r: gx(th, r) - fy(th, r)
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

    def minandmax2est(self, n: int = 33) -> jax.Array:
        """Estimate the range of each component on the disk.

        ``mM = minandmax2est(F)`` returns a length-4 array
        ``[min(f), max(f), min(g), max(g)]`` obtained from each component's
        :meth:`Diskfun.minandmax2est`.

        Provenance
        ----------
        MATLAB source : @diskfunv/minandmax2est.m
        Chebfun commit: 7574c77
        """
        f, g = self.components
        return jnp.concatenate(
            [f.minandmax2est(n), g.minandmax2est(n)]).astype(jnp.float64)

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

    def times(self, other) -> "Diskfunv":
        """Componentwise product with another Diskfunv, or scaling by
        a scalar field / scalar (MATLAB times).

        Provenance
        ----------
        MATLAB source : @diskfunv/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Diskfunv):
            return Diskfunv(*[a * b for a, b in
                           zip(self.components, other.components)])
        return Diskfunv(*[a * other for a in self.components])

    def power(self, n: int) -> "Diskfunv":
        """Componentwise power (MATLAB power).

        Provenance
        ----------
        MATLAB source : @diskfunv/power.m
        Chebfun commit: 7574c77
        """
        return Diskfunv(*[a ** n for a in self.components])

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

    @property
    def size(self):
        """MATLAB ``size``: ``(2, inf, inf)``, transposed
        ``(inf, inf, 2)``.

        Provenance
        ----------
        MATLAB source : @diskfunv/size.m
        Chebfun commit: 7574c77
        """
        import math
        if getattr(self, "_row", False):
            return (math.inf, math.inf, len(self.components))
        return (len(self.components), math.inf, math.inf)

    def transpose(self) -> "Diskfunv":
        """MATLAB ``F.'`` (row form; components unchanged).

        Provenance
        ----------
        MATLAB source : @diskfunv/transpose.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        out = type(self)(*self.components)
        object.__setattr__(out, "_row",
                           not getattr(self, "_row", False))
        return out

    def ctranspose(self) -> "Diskfunv":
        """MATLAB ``F'`` (empty in, empty out).

        Provenance
        ----------
        MATLAB source : @diskfunv/ctranspose.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return self.conj().transpose()

    def conj(self) -> "Diskfunv":
        """Componentwise conjugate (MATLAB conj).

        Provenance
        ----------
        MATLAB source : @diskfunv/conj.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return type(self)(*[c.conj() if hasattr(c, "conj") else c
                            for c in self.components])

    def real(self) -> "Diskfunv":
        """Componentwise real part.

        Provenance
        ----------
        MATLAB source : @diskfunv/real.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return type(self)(*[c.real() if hasattr(c, "real") else c
                            for c in self.components])

    def imag(self) -> "Diskfunv":
        """Componentwise imaginary part.

        Provenance
        ----------
        MATLAB source : @diskfunv/imag.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return type(self)(*[c.imag() if hasattr(c, "imag")
                            else c * 0.0 for c in self.components])

    def vscale(self) -> float:
        """The largest component vertical scale (MATLAB vscale).

        Provenance
        ----------
        MATLAB source : @diskfunv/vscale.m
        Chebfun commit: 7574c77
        """
        import numpy as _onp
        worst = 0.0
        for c in self.components:
            t = _onp.linspace(-_onp.pi, _onp.pi, 33)
            r = _onp.linspace(0.0, 1.0, 17)
            T, R = _onp.meshgrid(t, r)
            worst = max(worst, float(_onp.max(_onp.abs(_onp.asarray(
                c(jnp.asarray(T), jnp.asarray(R)))))))
        return worst

    def diffx(self, k: int = 1) -> "Diskfunv":
        """Componentwise d/dx (MATLAB diffx).

        Provenance
        ----------
        MATLAB source : @diskfunv/diffx.m
        Chebfun commit: 7574c77
        """
        out = self
        for _ in range(k):
            out = type(self)(*[c.diffx() for c in out.components])
        return out

    def diffy(self, k: int = 1) -> "Diskfunv":
        """Componentwise d/dy (MATLAB diffy).

        Provenance
        ----------
        MATLAB source : @diskfunv/diffy.m
        Chebfun commit: 7574c77
        """
        out = self
        for _ in range(k):
            out = type(self)(*[c.diffy() for c in out.components])
        return out

    def jacobian(self):
        """The Jacobian determinant ``Fx(1) Fy(2) - Fy(1) Fx(2)``
        (MATLAB jacobian); empty in, empty Diskfun-slot out.

        Provenance
        ----------
        MATLAB source : @diskfunv/jacobian.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            from chebfunjax.diskfun.diskfun import Diskfun
            return Diskfun.empty()
        Fx = self.diffx()
        Fy = self.diffy()
        return (Fx.components[0] * Fy.components[1]
                - Fy.components[0] * Fx.components[1])

    def divgrad(self):
        """``d2F1/dx2 + d2F2/dy2`` (MATLAB divgrad).

        Provenance
        ----------
        MATLAB source : @diskfunv/divgrad.m
        Chebfun commit: 7574c77
        """
        return (self.components[0].diffx().diffx()
                + self.components[1].diffy().diffy())

    def cross(self, other: "Diskfunv"):
        """2-D cross product ``F1 G2 - F2 G1`` (a scalar Diskfun);
        empty in, empty out.

        Provenance
        ----------
        MATLAB source : @diskfunv/cross.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return self if self.isempty() else other
        return (self.components[0] * other.components[1]
                - self.components[1] * other.components[0])

    def compose(self, g):
        """Composition ``g(F)`` with a Chebfun2 (-> Diskfun) or
        Chebfun2v (-> Diskfunv).

        Provenance
        ----------
        MATLAB source : @diskfunv/compose.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        from chebfunjax.diskfun.diskfun import Diskfun
        f1, f2 = self.components

        def disk_of(gg):
            g2 = gg if callable(gg) and not hasattr(gg, "approx")                 else Chebfun2(approx=getattr(gg, "approx", gg))                 if not isinstance(gg, Chebfun2) else gg

            def h(t, r):
                return g2(f1(t, r), f2(t, r))
            return Diskfun.from_function(h)

        if hasattr(g, "components"):
            return type(self)(*[
                disk_of(Chebfun2(approx=c) if not hasattr(c, "domain")
                        else c) for c in g.components])
        return disk_of(g)

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
