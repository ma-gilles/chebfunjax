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


def _source_isempty(value) -> bool:
    method = getattr(value, "isempty", None)
    if callable(method):
        return bool(method())
    try:
        return jnp.asarray(value).size == 0
    except (TypeError, ValueError):
        return False


def _source_scalar(value):
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size != 1 or not jnp.issubdtype(array.dtype, jnp.number):
        return None
    # All numeric size-one shapes use the same JAX factor arithmetic.
    return array.reshape(())


def _source_double_array(value):
    """Return a numeric MATLAB-double analogue, excluding logical arrays."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    if jnp.issubdtype(array.dtype, jnp.number):
        return array
    return None


def _source_linear_values(array):
    """MATLAB linear indexing order for a numeric array."""
    if array.ndim < 2:
        return jnp.reshape(array, (-1,))
    axes = tuple(range(array.ndim - 1, -1, -1))
    return jnp.transpose(array, axes).reshape(-1)


def _source_rdivide(left, right):
    """Evaluate one source ``rdivide`` branch for numeric/Spherefun values."""
    if isinstance(left, Spherefun):
        return left / right
    if isinstance(right, Spherefun):
        return Spherefun.from_function(
            lambda lam, th: jnp.asarray(left) / right(lam, th))
    return jnp.asarray(left) / jnp.asarray(right)


def _source_factor_rdivide(function, divisor):
    """Spherefun ``rdivide`` by a scalar, retaining source factors.

    The pinned separableApprox path is ``f .* (1/g)``.  Spherefun's scalar
    ``times`` path updates pivots without reconstructing the function.
    """
    scalar = _source_scalar(divisor)
    if scalar is None:
        raise ValueError(
            "MATLAB:dimagree: Matrix dimensions must agree.")
    if scalar == 0:
        raise ValueError(
            "CHEBFUN:SEPARABLEAPPROX:rdivide:divByZero: "
            "Division by zero or near zero.")
    from chebfunjax.utils._ordered_binary import binary
    return function * binary(jnp.asarray(1.0), scalar, 3)


def _source_factor_mrdivide(function, divisor):
    """Spherefun ``mrdivide``'s literal pivotValues *= divisor operation."""
    return Spherefun(
        cols=function.cols, rows=function.rows,
        pivots=function.pivots * divisor,
        idx_plus=function.idx_plus, idx_minus=function.idx_minus,
        pivot_locations=function.pivot_locations,
        nonzero_poles=function.nonzero_poles)


def _source_vector_like(template, components):
    """Build result components while preserving MATLAB isTransposed state."""
    out = Spherefunv(*components)
    if isinstance(template, Spherefunv):
        object.__setattr__(out, "_row", getattr(template, "_row", False))
    return out


def _source_transpose_operand(value):
    """MATLAB nonconjugating transpose for mtimes dispatch operands."""
    if isinstance(value, Spherefunv):
        return value.transpose()
    try:
        return jnp.transpose(jnp.asarray(value))
    except (TypeError, ValueError):
        return value.transpose()


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
        object.__setattr__(obj, "components", [])
        object.__setattr__(obj, "_is_empty_object", True)
        object.__setattr__(obj, "_row", False)
        return obj

    def isempty(self) -> bool:
        """True for the empty Spherefunv (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @spherefunv/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    def __iter__(self):
        """Iterate over components, preserving gradient tuple unpacking.

        The empty Spherefunv has no components and therefore iterates as an
        empty sequence.
        """
        return iter(()) if self.isempty() else iter(self.components)

    def __len__(self) -> int:
        """Number of components (zero for an empty Spherefunv)."""
        return 0 if self.isempty() else len(self.components)

    components: list  # [fx, fy, fz] or [f, g]
    _is_empty_object: bool = eqx.field(static=True, default=False)
    _row: bool = eqx.field(static=True, default=False)

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
        self._is_empty_object = False
        self._row = False

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
    def __call__(self, lam: jax.Array, theta: jax.Array) -> tuple | jax.Array:
        """Evaluate every component at spherical coordinates (lam, theta).

        Returns a tuple with one entry per component (length 2 or 3).
        An empty field returns the source-compatible empty ``(0, 0)`` array.

        Parameters
        ----------
        lam : jax.Array
            Longitude(s) in [-pi, pi].
        theta : jax.Array
            Colatitude(s) in [0, pi].
        """
        if self.isempty():
            return jnp.empty((0, 0), dtype=jnp.float64)
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

    def norm(self):
        """Global norm ``sqrt(sum_j norm(F_j, 2)^2)`` (MATLAB
        @spherefunv/norm.m); the pointwise magnitude field is
        :meth:`magnitude`.

        Provenance
        ----------
        MATLAB source : @spherefunv/norm.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.zeros((0,), dtype=jnp.float64)
        component_norms = jnp.stack(tuple(
            jnp.asarray(c.norm(), dtype=jnp.float64)
            for c in self.components
        ))
        return jnp.sqrt(jnp.sum(component_norms ** 2))

    def magnitude(self) -> Spherefun:
        """Pointwise magnitude ``sqrt(F1^2 + F2^2 + F3^2)`` as a
        Spherefun."""
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

    def plus(self, other):
        """MATLAB ``plus`` dispatch for vector, scalar, array, or Spherefun operands.

        Provenance
        ----------
        MATLAB source : @spherefunv/plus.m
        Chebfun commit: 7574c77
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        if not isinstance(self, Spherefunv):
            if isinstance(other, Spherefunv):
                return other.plus(self)
            raise ValueError(
                "SPHEREFUN:SPHEREFUNV:plus:type: Unrecognized input arguments.")
        if isinstance(other, Spherefunv):
            if getattr(self, "_row", False) != getattr(other, "_row", False):
                raise ValueError(
                    "SPHEREFUN:SPHEREFUNV:plus:transposed: Dimension mismatch.")
            return _source_vector_like(
                self, [a + b for a, b in zip(self.components, other.components)])
        if isinstance(other, Spherefun):
            return _source_vector_like(self, [a + other for a in self.components])
        scalar = _source_scalar(other)
        if scalar is not None:
            return _source_vector_like(self, [a + scalar for a in self.components])
        array = _source_double_array(other)
        if array is not None:
            if array.size == 3:
                values = _source_linear_values(array)
                return _source_vector_like(
                    self, [component + values[j]
                           for j, component in enumerate(self.components)])
            raise ValueError(
                "SPHEREFUN:SPHEREFUNV:plus:doubleSize: Dimension mismatch.")
        raise ValueError(
            "SPHEREFUN:SPHEREFUNV:plus:type: Unrecognized input arguments.")

    def __add__(self, other) -> "Spherefunv":
        return self.plus(other)

    def __radd__(self, other) -> "Spherefunv":
        return self.plus(other)

    def times(self, other) -> "Spherefunv":
        """MATLAB componentwise ``.*`` for vector and scalar operands.

        Provenance
        ----------
        MATLAB source : @spherefunv/times.m
        Chebfun commit: 7574c77
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        if not isinstance(self, Spherefunv):
            if isinstance(other, Spherefunv):
                # Literal @spherefunv/times.m left-operand swap.
                return other.times(self)
            raise ValueError("SPHEREFUN:SPHEREFUNV:times:inputs: Unrecognized input arguments.")
        if isinstance(other, Spherefunv):
            return _source_vector_like(
                self, [a * b for a, b in zip(self.components, other.components)])
        if isinstance(other, Spherefun):
            return _source_vector_like(self, [a * other for a in self.components])
        scalar = _source_scalar(other)
        if scalar is not None:
            return _source_vector_like(self, [a * scalar for a in self.components])
        array = _source_double_array(other)
        if array is not None:
            row = getattr(self, "_row", False)
            valid = ((array.ndim == 1 and array.shape[0] == 3)
                     or (array.ndim >= 2 and array.shape[0] == 3)
                     or (row and array.ndim >= 2 and array.shape[1] == 3))
            if valid:
                vals = _source_linear_values(array)
                return _source_vector_like(
                    self, [c * vals[j] for j, c in enumerate(self.components)])
            raise ValueError(
                "SPHEREFUN:SPHEREFUNV:times:double: SPHEREFUNV and double size mismatch.")
        raise ValueError("SPHEREFUN:SPHEREFUNV:times:inputs: Unrecognized input arguments.")

    def power(self, other):
        """MATLAB ``power`` dispatch for scalar/vector operands.

        Provenance
        ----------
        MATLAB source : @spherefunv/power.m
        Chebfun commit: 7574c77
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        if isinstance(self, Spherefunv):
            if isinstance(other, Spherefunv):
                raise ValueError(
                    "SPHEREFUN:SPHEREFUNV:power:size: SPHEREFUNV dimension mismatch.")
            scalar = _source_scalar(other)
            if scalar is not None:
                return _source_vector_like(
                    self, [component ** scalar for component in self.components])
            if _source_double_array(other) is not None:
                raise ValueError(
                    "SPHEREFUN:SPHEREFUNV:power:double: "
                    "SPHEREFUNV and double size mismatch.")
            raise ValueError(
                "SPHEREFUN:SPHEREFUNV:power:inputs: Unrecognized input arguments.")
        if isinstance(other, Spherefunv):
            scalar = _source_scalar(self)
            if scalar is not None:
                return _source_vector_like(
                    other, [scalar ** component for component in other.components])
            if _source_double_array(self) is not None:
                raise ValueError(
                    "SPHEREFUN:SPHEREFUNV:power:double: Dimension mismatch.")
        raise ValueError(
            "SPHEREFUN:SPHEREFUNV:power:inputs: Unrecognized input arguments.")

    def __pow__(self, other):
        return self.power(other)

    def __rpow__(self, base):
        return Spherefunv.power(base, self)

    def mrdivide(self, other) -> "Spherefunv":
        """MATLAB ``F/G`` for supported scalar numeric denominators.

        Native accepts Chebfun2 denominators and delegates each component to
        separable-approximation division after checking the separated domains.
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        if isinstance(other, (Spherefunv, Spherefun)):
            raise ValueError("SPHEREFUN:SPHEREFUNV:mrdivide:nonScalar: Division must be scalar valued.")
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        if isinstance(other, Chebfun2):
            expected = (-float(jnp.pi), float(jnp.pi),
                        -float(jnp.pi), float(jnp.pi))
            domain = tuple(float(v) for v in other.approx.domain)
            def source_hscale(interval):
                scale = max(abs(float(v)) for v in interval)
                return 1.0 if scale == float("inf") else scale

            domain_matches = len(domain) == 4
            if domain_matches:
                for first, last in ((0, 2), (2, 4)):
                    tol = 1e-15 * max(
                        source_hscale(expected[first:last]),
                        source_hscale(domain[first:last]))
                    if not all(abs(a - b) < tol for a, b in
                               zip(domain[first:last], expected[first:last])):
                        domain_matches = False
                        break
            if not domain_matches:
                raise ValueError(
                    "CHEBFUN:SEPARABLEAPPROX:rdivide:domains: Domains inconsistent.")
            return _source_vector_like(self, [
                Spherefun.from_function(
                    lambda lam, th, c=c: c(lam, th) / other(lam, th))
                for c in self.components
            ])
        divisor = _source_scalar(other)
        if divisor is None:
            raise ValueError("SPHEREFUN:SPHEREFUNV:mrdivide:nonScalar: Division must be scalar valued.")
        return _source_vector_like(
            self, [_source_factor_mrdivide(c, divisor) for c in self.components])

    def rdivide(self, other) -> "Spherefunv":
        """MATLAB ``A./B`` componentwise right division."""
        left, right = self, other
        if _source_isempty(left) or _source_isempty(right):
            return Spherefunv.empty()
        if isinstance(left, Spherefunv):
            if isinstance(right, Spherefunv):
                return _source_vector_like(
                    left, [a / b for a, b in zip(left.components, right.components)])
            if isinstance(right, Spherefun):
                return _source_vector_like(left, [a / right for a in left.components])
            divisor = _source_double_array(right)
            if divisor is not None:
                scalar = _source_scalar(right)
                if scalar is not None:
                    return _source_vector_like(
                        left, [_source_factor_rdivide(c, scalar)
                               for c in left.components])
                return _source_vector_like(left, [a / right for a in left.components])
        elif isinstance(right, Spherefunv):
            if isinstance(left, Spherefun):
                return _source_vector_like(
                    right, [_source_rdivide(left, b) for b in right.components])
            numerator = _source_double_array(left)
            if numerator is not None:
                return _source_vector_like(
                    right, [_source_rdivide(left, b) for b in right.components])
        raise ValueError("SPHEREFUN:SPHEREFUNV:rdivide:inputs: Unrecognized input arguments.")

    def ldivide(self, other) -> "Spherefunv":
        r"""MATLAB ``A.\B`` componentwise left division (``B./A``)."""
        left, right = self, other
        if _source_isempty(left) or _source_isempty(right):
            return Spherefunv.empty()
        if isinstance(left, Spherefunv) and isinstance(right, Spherefunv):
            return _source_vector_like(
                left, [_source_rdivide(b, a)
                       for a, b in zip(left.components, right.components)])
        if isinstance(right, Spherefunv) and _source_double_array(left) is not None:
            scalar = _source_scalar(left)
            if scalar is not None:
                return _source_vector_like(
                    right, [_source_factor_rdivide(c, scalar)
                            for c in right.components])
            return _source_vector_like(
                right, [_source_rdivide(c, left) for c in right.components])
        if isinstance(left, Spherefunv) and _source_double_array(right) is not None:
            scalar = _source_scalar(right)
            if scalar is not None:
                return _source_vector_like(
                    left, [_source_factor_rdivide(c, scalar)
                           for c in left.components])
            return _source_vector_like(
                left, [_source_rdivide(right, c) for c in left.components])
        raise ValueError("SPHEREFUN:SPHEREFUNV:ldivide:inputs: Unrecognized input arguments.")

    def mldivide(self, other) -> "Spherefunv":
        r"""MATLAB ``A\B``: divide each component of ``B`` by scalar ``A``."""
        left, right = self, other
        if _source_isempty(left) or _source_isempty(right):
            return Spherefunv.empty()
        if not isinstance(right, Spherefunv):
            raise ValueError("SPHEREFUN:SPHEREFUNV:mldivide:inputs: Unrecognized input arguments.")
        divisor = _source_double_array(left)
        if divisor is None:
            raise ValueError("SPHEREFUN:SPHEREFUNV:mldivide:nonScalar: Division must be by a scalar.")
        return _source_vector_like(
            right, [_source_factor_rdivide(c, left) for c in right.components])

    def real(self) -> "Spherefunv":
        """Componentwise real part (MATLAB real).

        Provenance
        ----------
        MATLAB source : @spherefunv/real.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return _source_vector_like(self, [c.real() for c in self.components])

    def imag(self) -> "Spherefunv":
        """Componentwise imaginary part (MATLAB imag).

        Provenance
        ----------
        MATLAB source : @spherefunv/imag.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return _source_vector_like(self, [c.imag() for c in self.components])

    def conj(self) -> "Spherefunv":
        """Componentwise complex conjugate (MATLAB conj).

        Provenance
        ----------
        MATLAB source : @spherefunv/conj.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return _source_vector_like(self, [c.conj() for c in self.components])

    def __pos__(self) -> "Spherefunv":
        """MATLAB ``uplus`` is the identity, including for empty vectors.

        Provenance
        ----------
        MATLAB source : @spherefunv/uplus.m
        Chebfun commit: 7574c77
        """
        return self

    def iszero(self) -> bool:
        """True iff every component is the zero function (MATLAB iszero).

        Provenance
        ----------
        MATLAB source : @spherefunv/iszero.m
        Chebfun commit: 7574c77
        """
        return all(c.iszero() for c in self.components)

    def __matmul__(self, other):
        """``u' * v``: a row field times a column field contracts to
        the dot product (MATLAB mtimes).

        Provenance
        ----------
        MATLAB source : @spherefunv/mtimes.m
        Chebfun commit: 7574c77
        """
        return self.mtimes(other)

    def mtimes(self, other):
        """MATLAB ``*`` scalar scaling and one-row matrix contraction.

        The pinned source's numeric-left multirow branch forms a cell array
        of row contractions and passes it to ``spherefun(store)``. MATLAB
        R2025b then errors because the constructor expects a function handle.
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        # Match the two transpose rewrites before considering operand types.
        # MATLAB row/column orientation is part of the function object state.
        if (isinstance(self, Spherefunv)
                and not isinstance(other, Spherefunv)
                and getattr(self, "_row", False)):
            return Spherefunv.mtimes(_source_transpose_operand(other),
                                     self.transpose())
        if (isinstance(other, Spherefunv)
                and not isinstance(self, Spherefunv)
                and getattr(other, "_row", False)):
            return Spherefunv.mtimes(
                other.transpose(), _source_transpose_operand(self)).transpose()
        if isinstance(self, Spherefunv):
            if isinstance(other, Spherefunv):
                if getattr(self, "_row", False) and not getattr(other, "_row", False):
                    return self.dot(other)
                raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:sizes: Dimensions mismatch.")
            if isinstance(other, Spherefun):
                # Native swaps operands and then reaches its unsupported-input
                # branch for Spherefun * Spherefunv.
                raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double")
            value = _source_scalar(other)
            if value is not None:
                return _source_vector_like(self, [c * value for c in self.components])
            try:
                array = jnp.asarray(other)
            except (TypeError, ValueError) as exc:
                raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double") from exc
            if jnp.issubdtype(array.dtype, jnp.number) and array.size == 1:
                value = array.reshape(())
                return _source_vector_like(self, [c * value for c in self.components])
            if jnp.issubdtype(array.dtype, jnp.number):
                raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:double2: SPHEREFUNV and double size mismatch.")
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double")

        # Numeric-left multiplication (MATLAB A*F). A scalar scales each
        # component; a 1-by-3 matrix contracts to a scalar Spherefun.
        if not isinstance(other, Spherefunv):
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double")
        if isinstance(self, Spherefun):
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double")
        value = _source_scalar(self)
        if value is not None:
            return _source_vector_like(other, [value * c for c in other.components])
        try:
            matrix = jnp.asarray(self)
        except (TypeError, ValueError) as exc:
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:double1: Dimension mismatch.") from exc
        if not jnp.issubdtype(matrix.dtype, jnp.number):
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:inputs: SPHEREFUNV can only mtimes to SPHEREFUNV or double")
        if matrix.size == 1:
            scalar = matrix.reshape(())
            return _source_vector_like(other, [scalar * c for c in other.components])
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise ValueError("SPHEREFUN:SPHEREFUNV:mtimes:double1: Dimension mismatch.")
        if matrix.shape[0] == 1:
            result = matrix[0, 0] * other.components[0]
            for j in range(1, 3):
                result = result + matrix[0, j] * other.components[j]
            return result
        raise ValueError(
            "MATLAB:feval:argMustBeStringOrHandle: Function to evaluate must "
            "be represented as a string scalar, character vector, or "
            "function_handle object.")

    def __mul__(self, scalar: float) -> "Spherefunv":
        """Scalar multiplication (componentwise)."""
        return self.mtimes(scalar)

    def __rmul__(self, scalar: float):
        """Dispatch reflected multiplication with MATLAB operand order."""
        return Spherefunv.mtimes(scalar, self)

    def __truediv__(self, other) -> "Spherefunv":
        return self.mrdivide(other)

    def __neg__(self) -> "Spherefunv":
        """Negation."""
        if self.isempty():
            return Spherefunv.empty()
        return _source_vector_like(self, [-c for c in self.components])

    def __sub__(self, other) -> "Spherefunv":
        """MATLAB ``minus`` via ``plus(self, uminus(other))``.

        Provenance
        ----------
        MATLAB source : @spherefunv/minus.m
        Chebfun commit: 7574c77
        """
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        numeric_other = _source_double_array(other)
        negated = -numeric_other if numeric_other is not None else -other
        return self.plus(negated)

    def __rsub__(self, other) -> "Spherefunv":
        """Reflected subtraction follows MATLAB ``plus(other, -self)``."""
        if _source_isempty(self) or _source_isempty(other):
            return Spherefunv.empty()
        return (-self).plus(other)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @property
    def size(self):
        """MATLAB ``size``: ``(3, inf, inf)`` for the three-component
        field.

        Provenance
        ----------
        MATLAB source : @spherefunv/size.m
        Chebfun commit: 7574c77
        """
        import math
        if getattr(self, "_row", False):
            return (math.inf, math.inf, len(self.components))
        return (len(self.components), math.inf, math.inf)

    def minandmax2est(self, N: int = 33):
        """Estimated per-component [min max ...] range vector on an
        N x N sample grid (MATLAB ``minandmax2est``).

        Provenance
        ----------
        MATLAB source : @spherefunv/minandmax2est.m
        Chebfun commit: 7574c77
        """
        import numpy as _onp
        lam = _onp.linspace(-_onp.pi, _onp.pi, N)
        th = _onp.linspace(0.0, _onp.pi, N)
        L, T = _onp.meshgrid(lam, th)
        out = []
        for c in self.components:
            V = _onp.asarray(c(jnp.asarray(L), jnp.asarray(T)))
            out.extend([float(V.min()), float(V.max())])
        return out

    def transpose(self) -> "Spherefunv":
        """MATLAB ``F.'``: the row form (components unchanged; size
        reports the transposed ordering and ``row * col`` contracts to
        the dot product).

        Provenance
        ----------
        MATLAB source : @spherefunv/transpose.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        out = type(self)(*self.components)
        object.__setattr__(out, "_row",
                           not getattr(self, "_row", False))
        return out

    def ctranspose(self) -> "Spherefunv":
        """MATLAB ``F'`` (empty in, empty out).

        Provenance
        ----------
        MATLAB source : @spherefunv/ctranspose.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return self.conj().transpose()

    def compose(self, g):
        """Composition ``g(F)`` with a Chebfun3 (-> Spherefun) or
        Chebfun3v (-> Spherefunv); the range of F lies on the unit
        sphere inside g's domain.

        Provenance
        ----------
        MATLAB source : @spherefunv/compose.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun3d.chebfun3 import Chebfun3
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        from chebfunjax.spherefun.spherefun import Spherefun
        if not isinstance(g, (Chebfun3, Chebfun3v)):
            raise ValueError('CHEBFUN:SPHEREFUNV:COMPOSE:OP: '
                             'Can compose a SPHEREFUNV with a CHEBFUN3 or CHEBFUN3V.')
        if isinstance(g, Chebfun3):
            from chebfunjax.chebfun3d._subsref import check_sphere_composition
            check_sphere_composition(self, g)
        fx, fy, fz = self.components

        def sph_of(gg):
            def h(lam, th):
                return gg(fx(lam, th), fy(lam, th), fz(lam, th))
            return Spherefun.from_function(h)

        if hasattr(g, "components"):
            return type(self)(*[sph_of(c) for c in g.components])
        return sph_of(g)

    def helmholtzdecomp(self):
        """Source Helmholtz wrapper: project tangent, warn, and solve twice.

        Returns two scalar Spherefuns. The Poisson dimensions preserve the
        pinned source's rows-first ``length`` ordering literally. Existing
        differential and Poisson implementations are reused; their numerical
        backend/source qualification is separate from this wrapper.

        Provenance
        ----------
        MATLAB source : @spherefunv/helmholtzdecomp.m,
            @separableApprox/length.m
        Chebfun commit: 7574c77
        """
        import warnings

        from chebfunjax.chebpref import ChebfunPref

        if self.isempty():
            return Spherefun.empty(), Spherefun.empty()
        tangentf = self.tangent()
        eps = ChebfunPref().cheb2Prefs.chebfun2eps
        tol1 = 100 * jnp.asarray(self.components[0].vscale()) * eps
        tol2 = 100 * jnp.asarray(self.components[1].vscale()) * eps
        tol3 = 100 * jnp.asarray(self.components[2].vscale()) * eps
        if bool((self - tangentf).norm() > tol1 + tol2 + tol3):
            warnings.warn(
                "SPHEREFUNV:HELMHOLTZDECOMPOSITON:TANGENT: "
                "The vector field needs to be tangent to the surface of the"
                "sphere, taking f = tangent(f).",
                stacklevel=2,
            )
        divf = tangentf.div()
        m, n = divf.length()
        u = Spherefun.poisson(divf, 0, max(2*m, 50), max(2*n, 50))
        vortf = tangentf.vort()
        m, n = vortf.length()
        v = Spherefun.poisson(vortf, 0, max(2*m, 50), max(2*n, 50))
        return u, v

    def coeffs2(self, m=None, n=None):
        """Per-component 2-D Fourier coefficient matrices (MATLAB
        ``coeffs2``).

        Provenance
        ----------
        MATLAB source : @spherefunv/coeffs2.m
        Chebfun commit: 7574c77
        """
        return tuple(c.coeffs2(m, n) for c in self.components)

    @classmethod
    def coeffs2spherefunv(cls, *coeff_mats) -> "Spherefunv":
        """Assemble a Spherefunv from per-component coefficient
        matrices (MATLAB ``spherefunv.coeffs2spherefunv``).

        Provenance
        ----------
        MATLAB source : @spherefunv/coeffs2spherefunv.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.spherefun.spherefun import Spherefun
        return cls(*[Spherefun.coeffs2spherefun(X)
                     for X in coeff_mats])

    @staticmethod
    def coeffs2vals(*coeff_mats):
        """Per-component coeffs -> values (MATLAB
        ``spherefunv.coeffs2vals``).

        Provenance
        ----------
        MATLAB source : @spherefunv/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.spherefun.spherefun import Spherefun
        return tuple(Spherefun.coeffs2vals(X) for X in coeff_mats)

    @staticmethod
    def vals2coeffs(*val_mats):
        """Per-component values -> coeffs (MATLAB
        ``spherefunv.vals2coeffs``).

        Provenance
        ----------
        MATLAB source : @spherefunv/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.spherefun.spherefun import Spherefun
        return tuple(Spherefun.vals2coeffs(V) for V in val_mats)

    def quiver3(self, **kwargs):
        """3-D quiver of the field on the sphere surface (MATLAB
        ``quiver3``).

        Provenance
        ----------
        MATLAB source : @spherefunv/quiver3.m
        Chebfun commit: 7574c77
        """
        import matplotlib.pyplot as plt
        import numpy as _onp
        lam = _onp.linspace(-_onp.pi, _onp.pi, 16)
        th = _onp.linspace(0.05, _onp.pi - 0.05, 12)
        L, T = _onp.meshgrid(lam, th)
        X = _onp.cos(L) * _onp.sin(T)
        Y = _onp.sin(L) * _onp.sin(T)
        Z = _onp.cos(T)
        U = _onp.asarray(self.components[0](jnp.asarray(L),
                                            jnp.asarray(T)))
        V = _onp.asarray(self.components[1](jnp.asarray(L),
                                            jnp.asarray(T)))
        W = _onp.asarray(self.components[2](jnp.asarray(L),
                                            jnp.asarray(T)))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.quiver(X, Y, Z, U, V, W, length=0.15)
        return ax

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
