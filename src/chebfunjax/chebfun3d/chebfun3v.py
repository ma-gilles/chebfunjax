"""Chebfun3v — 3D vector field of Chebfun3 components.

A Chebfun3v holds one, two, or three Chebfun3 scalar components,
representing a vector field on a 3D cuboid.  This mirrors the MATLAB
@chebfun3v class and, structurally, the 2D analog
:class:`chebfunjax.chebfun2d.chebfun2v.Chebfun2v`.

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

_INF = float("inf")


class Chebfun3v(eqx.Module):
    """Vector field on a 3D cuboid with 1, 2, or 3 Chebfun3 components.

    Represents ``[f_1; f_2; f_3]`` where each ``f_j`` is a Chebfun3 object
    on the same domain.  Immutable: every operation returns a new object.

    Attributes
    ----------
    components : list of Chebfun3
        The scalar components. Length 0 (empty), 1, 2, or 3.
    is_transposed : bool
        Whether the field is a row (transposed) vector. Static field.

    Provenance
    ----------
    MATLAB source : @chebfun3v/chebfun3v.m
    Chebfun commit: 7574c77
    """

    components: list
    is_transposed: bool = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *components, is_transposed: bool = False) -> None:
        """Construct from Chebfun3 components.

        Accepts either a single list/tuple of Chebfun3 objects or the
        components as separate positional arguments.  No arguments builds
        the empty Chebfun3v.

        Parameters
        ----------
        *components : Chebfun3, or a single list/tuple of Chebfun3
            The 1, 2, or 3 scalar components (all sharing one domain).
        is_transposed : bool, optional
            Row-vector flag. Default False.

        Provenance
        ----------
        MATLAB source : @chebfun3v/chebfun3v.m
        Chebfun commit: 7574c77
        """
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            comps = list(components[0])
        else:
            comps = list(components)

        if len(comps) > 3:
            raise ValueError(
                f"Chebfun3v supports at most 3 components, got {len(comps)}."
            )
        # Check domains match (non-empty components only).
        if comps:
            dom0 = comps[0].domain
            for j, c in enumerate(comps[1:], 1):
                if tuple(c.domain) != tuple(dom0):
                    raise ValueError(
                        f"Chebfun3v: component {j} has domain {c.domain} but "
                        f"component 0 has domain {dom0}. All components must "
                        "share the same domain."
                    )
        object.__setattr__(self, "components", comps)
        object.__setattr__(self, "is_transposed", bool(is_transposed))

    @classmethod
    def from_functions(
        cls,
        *fns: Callable,
        domain: tuple = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        tol: float = float(jnp.finfo(jnp.float64).eps),
    ) -> "Chebfun3v":
        """Construct a Chebfun3v from 1, 2, or 3 callables.

        Parameters
        ----------
        *fns : callables
            1, 2, or 3 functions of ``(x, y, z)``.
        domain : 6-tuple, optional
            ``(xa, xb, ya, yb, za, zb)``.  Default is the unit cube.
        tol : float, optional
            Tolerance for Chebfun3 construction.

        Returns
        -------
        Chebfun3v

        Provenance
        ----------
        MATLAB source : @chebfun3v/chebfun3v.m
        Chebfun commit: 7574c77
        """
        if len(fns) < 1 or len(fns) > 3:
            raise ValueError(
                f"Chebfun3v.from_functions needs 1-3 callables, got {len(fns)}."
            )
        comps = [Chebfun3.from_function(f, domain=domain, tol=tol) for f in fns]
        return cls(comps)

    @staticmethod
    def gradient(f: Chebfun3) -> "Chebfun3v":
        """Gradient of a scalar Chebfun3 as a 3-component Chebfun3v.

        ``gradient(f) = [f_x; f_y; f_z]``.  Mirrors MATLAB's
        ``chebfun3/gradient`` (also spelled ``grad``), which returns a
        Chebfun3v.

        Provenance
        ----------
        MATLAB source : @chebfun3/gradient.m
        Chebfun commit: 7574c77
        """
        return Chebfun3v([f.diff(1), f.diff(2), f.diff(3)])

    grad = gradient

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of components (0, 1, 2, or 3)."""
        return len(self.components)

    @property
    def domain(self):
        """Shared domain of the components, or None if empty."""
        return self.components[0].domain if self.components else None

    def isempty(self) -> bool:
        """True for the empty Chebfun3v (MATLAB isempty)."""
        return len(self.components) == 0

    def __getitem__(self, k: int) -> Chebfun3:
        """Component access ``F[k]`` (0-based), returning a Chebfun3."""
        return self.components[k]

    def size(self, dim: int | None = None):
        """Size of the vector field (MATLAB size).

        Returns ``(K, M, N, P)`` where, for a column field, ``K`` is the
        number of components and ``M = N = P = inf``; for a row
        (transposed) field the number of components appears in ``P``.

        Provenance
        ----------
        MATLAB source : @chebfun3v/size.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return (None, None, None, None) if dim is None else None
        nF = self.n_components
        if not self.is_transposed:
            dims = (nF, _INF, _INF, _INF)
        else:
            dims = (_INF, _INF, _INF, nF)
        if dim is None:
            return dims
        return dims[dim - 1]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def feval(self, x, y, z) -> jax.Array:
        """Evaluate all components at ``(x, y, z)``.

        Returns the component values stacked along a leading axis (a MATLAB
        column ``[f_1; f_2; f_3]`` for scalar inputs).

        Provenance
        ----------
        MATLAB source : @chebfun3v/feval.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.asarray([], dtype=jnp.float64)
        return jnp.stack([c(x, y, z) for c in self.components], axis=0)

    def __call__(self, x, y, z) -> jax.Array:
        """Alias for :meth:`feval`."""
        return self.feval(x, y, z)

    # ------------------------------------------------------------------
    # Additive arithmetic
    # ------------------------------------------------------------------

    def _like(self, comps) -> "Chebfun3v":
        return Chebfun3v(list(comps), is_transposed=self.is_transposed)

    def __neg__(self) -> "Chebfun3v":
        """Negate: ``-F`` (MATLAB uminus)."""
        if self.isempty():
            return self
        return self._like([-c for c in self.components])

    def __pos__(self) -> "Chebfun3v":
        """Unary plus: ``+F`` (MATLAB uplus)."""
        return self

    def _as_vector(self, other):
        """Return a length-n python list if ``other`` is a numeric vector
        matching the component count, else None."""
        if isinstance(other, Chebfun3v) or isinstance(other, Chebfun3):
            return None
        try:
            seq = list(other)
        except TypeError:
            return None
        if len(seq) == self.n_components:
            return [complex(v) if isinstance(v, complex) else float(v)
                    for v in seq]
        return None

    def __add__(self, other) -> "Chebfun3v":
        """Componentwise addition (MATLAB plus).

        ``other`` may be another Chebfun3v, a scalar (added to every
        component), a length-n numeric vector (added componentwise), or a
        Chebfun3 (added to every component).

        Provenance
        ----------
        MATLAB source : @chebfun3v/plus.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        if isinstance(other, Chebfun3v):
            if other.n_components != self.n_components:
                raise ValueError(
                    "Chebfun3v + Chebfun3v: component count mismatch "
                    f"({self.n_components} vs {other.n_components})."
                )
            return self._like(
                [a + b for a, b in zip(self.components, other.components)]
            )
        if isinstance(other, Chebfun3):
            return self._like([c + other for c in self.components])
        if isinstance(other, (int, float, complex)):
            return self._like([c + other for c in self.components])
        vec = self._as_vector(other)
        if vec is not None:
            return self._like([c + v for c, v in zip(self.components, vec)])
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other) -> "Chebfun3v":
        """Componentwise subtraction (MATLAB minus)."""
        if isinstance(other, Chebfun3v):
            return self.__add__(other.__neg__())
        neg = self.__neg__()
        res = neg.__add__(other)
        return res if res is NotImplemented else res.__neg__()

    def __rsub__(self, other) -> "Chebfun3v":
        return self.__neg__().__add__(other)

    # ------------------------------------------------------------------
    # Multiplicative arithmetic
    # ------------------------------------------------------------------

    def __mul__(self, other) -> "Chebfun3v":
        """Pointwise / scalar multiplication (MATLAB times, ``.*``).

        ``other`` may be a scalar (scale every component), a length-n
        numeric vector (scale componentwise), a Chebfun3 (scale every
        component), or a Chebfun3v (componentwise product).

        Provenance
        ----------
        MATLAB source : @chebfun3v/times.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        if isinstance(other, Chebfun3v):
            if other.n_components != self.n_components:
                raise ValueError("Chebfun3v .* Chebfun3v: component mismatch.")
            return self._like(
                [a * b for a, b in zip(self.components, other.components)]
            )
        if isinstance(other, Chebfun3):
            return self._like([c * other for c in self.components])
        if isinstance(other, (int, float, complex)):
            return self._like([c * other for c in self.components])
        vec = self._as_vector(other)
        if vec is not None:
            return self._like([c * v for c, v in zip(self.components, vec)])
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Chebfun3v":
        """Right divide (MATLAB rdivide/mrdivide).

        ``F / c`` divides every component by a scalar; ``F / g`` divides
        every component by a Chebfun3.

        Provenance
        ----------
        MATLAB source : @chebfun3v/rdivide.m, mrdivide.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        if isinstance(other, (int, float, complex)):
            return self._like([c / other for c in self.components])
        if isinstance(other, Chebfun3):
            return self._like([c / other for c in self.components])
        return NotImplemented

    def __pow__(self, p) -> "Chebfun3v":
        """Componentwise power ``F .^ p`` for a scalar ``p`` (MATLAB power)."""
        if self.isempty():
            return self
        if isinstance(p, (int, float, complex)):
            return self._like([c ** p for c in self.components])
        return NotImplemented

    def __matmul__(self, other):
        """Inner product ``F' * v`` / ``F' * G`` (MATLAB mtimes).

        With a length-n numeric vector, returns the Chebfun3
        ``sum_j v_j * F_j``.  With another Chebfun3v, returns the Chebfun3
        ``sum_j F_j * G_j``.  (Conjugation, if wanted, is applied through
        :meth:`ctranspose` before the product, matching MATLAB ``F' * G``.)

        Provenance
        ----------
        MATLAB source : @chebfun3v/mtimes.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        if isinstance(other, Chebfun3v):
            if other.n_components != self.n_components:
                raise ValueError("Chebfun3v inner product: component mismatch.")
            acc = self.components[0] * other.components[0]
            for a, b in zip(self.components[1:], other.components[1:]):
                acc = acc + a * b
            return acc
        vec = self._as_vector(other)
        if vec is not None:
            acc = self.components[0] * vec[0]
            for c, v in zip(self.components[1:], vec[1:]):
                acc = acc + c * v
            return acc
        return NotImplemented

    def __rmatmul__(self, other):
        """A left matrix/vector product ``v * F`` is a dimension mismatch
        for a column Chebfun3v (MATLAB errors on ``[1 2 3]' * G``)."""
        raise ValueError(
            "Chebfun3v: left matrix-vector product is a dimension mismatch."
        )

    # ------------------------------------------------------------------
    # Transpose / conjugate
    # ------------------------------------------------------------------

    def transpose(self) -> "Chebfun3v":
        """Toggle the row/column flag (MATLAB ``.'``)."""
        if self.isempty():
            return self
        return Chebfun3v(list(self.components),
                         is_transposed=not self.is_transposed)

    @property
    def T(self) -> "Chebfun3v":
        """Transpose (see :meth:`transpose`)."""
        return self.transpose()

    def ctranspose(self) -> "Chebfun3v":
        """Conjugate transpose (MATLAB ``'``): transpose, then conjugate."""
        return self.transpose().conj()

    # ------------------------------------------------------------------
    # Complex parts
    # ------------------------------------------------------------------

    def real(self) -> "Chebfun3v":
        """Real part, componentwise (MATLAB real)."""
        if self.isempty():
            return self
        return self._like([c.real() for c in self.components])

    def imag(self) -> "Chebfun3v":
        """Imaginary part, componentwise (MATLAB imag)."""
        if self.isempty():
            return self
        return self._like([c.imag() for c in self.components])

    def conj(self) -> "Chebfun3v":
        """Complex conjugate, componentwise (MATLAB conj)."""
        if self.isempty():
            return self
        return self._like([c.conj() for c in self.components])

    def isreal(self) -> bool:
        """True if every component is real-valued (MATLAB isreal)."""
        if self.isempty():
            return True
        return all(c.isreal() for c in self.components)

    def isPeriodicTech(self) -> bool:
        """True if the components use a periodic tech.

        chebfunjax Chebfun3 is always built on a Chebyshev tech, so this is
        always False (chebfunjax has no trigonometric tech).

        Provenance
        ----------
        MATLAB source : @chebfun3v/isPeriodicTech.m
        Chebfun commit: 7574c77
        """
        return False

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def dot(self, other: "Chebfun3v") -> Chebfun3:
        """Dot product ``sum_j conj(F_j) * G_j`` (MATLAB dot, same as
        ``F' * G``).

        Provenance
        ----------
        MATLAB source : @chebfun3v/dot.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return Chebfun3.empty()
        if other.n_components != self.n_components:
            raise ValueError("Chebfun3v.dot: component count mismatch.")
        fc = [c.conj() for c in self.components]
        acc = fc[0] * other.components[0]
        for a, b in zip(fc[1:], other.components[1:]):
            acc = acc + a * b
        return acc

    def cross(self, other: "Chebfun3v") -> "Chebfun3v":
        """Cross product of two 3-component fields (MATLAB cross).

        ``F x G = [F_2 G_3 - F_3 G_2; F_3 G_1 - F_1 G_3;
        F_1 G_2 - F_2 G_1]``.

        Provenance
        ----------
        MATLAB source : @chebfun3v/cross.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return Chebfun3v()
        if self.n_components != 3 or other.n_components != 3:
            raise ValueError("Chebfun3v.cross needs 3 components on each side.")
        f1, f2, f3 = self.components
        g1, g2, g3 = other.components
        return Chebfun3v([
            f2 * g3 - f3 * g2,
            f3 * g1 - f1 * g3,
            f1 * g2 - f2 * g1,
        ])

    def norm(self):
        """Frobenius norm ``sqrt(sum_j norm(F_j)^2)`` (MATLAB norm).

        Provenance
        ----------
        MATLAB source : @chebfun3v/norm.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return None
        total = 0.0
        for c in self.components:
            total += float(jnp.real(c.norm())) ** 2
        return jnp.sqrt(jnp.asarray(total, dtype=jnp.float64))

    def magnitude(self) -> Chebfun3:
        """Pointwise Euclidean magnitude ``sqrt(sum_j |F_j|^2)`` as a
        Chebfun3.

        This is not a MATLAB @chebfun3v method (there ``norm`` is the
        Frobenius scalar); it exposes the pointwise magnitude field a
        vector norm induces, useful for plotting and for the golden-ref
        parity fixture.
        """
        if self.isempty():
            return Chebfun3.empty()
        dom = self.domain
        comps = self.components
        return Chebfun3.from_function(
            lambda x, y, z: jnp.sqrt(
                sum(jnp.abs(c(x, y, z)) ** 2 for c in comps)),
            domain=dom)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1, dim: int = 1) -> "Chebfun3v":
        """Componentwise ``k``-th derivative in direction ``dim`` (MATLAB
        diff; dim 1/2/3 = x/y/z).

        Provenance
        ----------
        MATLAB source : @chebfun3v/diff.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return self._like([c.diff(dim, k) for c in self.components])

    def diffx(self, k: int = 1) -> "Chebfun3v":
        """Componentwise ``k``-th x-derivative (MATLAB diffx)."""
        return self.diff(k, 1)

    def diffy(self, k: int = 1) -> "Chebfun3v":
        """Componentwise ``k``-th y-derivative (MATLAB diffy)."""
        return self.diff(k, 2)

    def diffz(self, k: int = 1) -> "Chebfun3v":
        """Componentwise ``k``-th z-derivative (MATLAB diffz)."""
        return self.diff(k, 3)

    def divergence(self) -> Chebfun3:
        """Divergence of the field (MATLAB divergence).

        For 2 components: ``F_1x + F_2y``.  For 3: ``F_1x + F_2y + F_3z``.

        Provenance
        ----------
        MATLAB source : @chebfun3v/divergence.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Chebfun3.empty()
        c = self.components
        if self.n_components == 2:
            return c[0].diff(1) + c[1].diff(2)
        if self.n_components == 3:
            return c[0].diff(1) + c[1].diff(2) + c[2].diff(3)
        raise ValueError("Chebfun3v.divergence needs 2 or 3 components.")

    def div(self) -> Chebfun3:
        """Alias for :meth:`divergence` (MATLAB div)."""
        return self.divergence()

    def curl(self) -> "Chebfun3v":
        """Curl of a 3-component field (MATLAB curl).

        ``curl(F) = [F_3y - F_2z; F_1z - F_3x; F_2x - F_1y]``.

        Provenance
        ----------
        MATLAB source : @chebfun3v/curl.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Chebfun3v()
        if self.n_components != 3:
            raise ValueError("Chebfun3v.curl needs 3 components.")
        f1, f2, f3 = self.components
        return Chebfun3v([
            f3.diff(2) - f2.diff(3),
            f1.diff(3) - f3.diff(1),
            f2.diff(1) - f1.diff(2),
        ])

    def divgrad(self) -> Chebfun3:
        """``F_1xx + F_2yy + F_3zz`` for a 3-component field (MATLAB
        divgrad).

        Provenance
        ----------
        MATLAB source : @chebfun3v/divgrad.m
        Chebfun commit: 7574c77
        """
        if self.n_components != 3:
            raise ValueError("Chebfun3v.divgrad needs 3 components.")
        f1, f2, f3 = self.components
        return f1.diff(1, 2) + f2.diff(2, 2) + f3.diff(3, 2)

    def laplacian(self) -> "Chebfun3v":
        """Vector Laplacian: scalar Laplacian of each component (MATLAB
        laplacian).

        Provenance
        ----------
        MATLAB source : @chebfun3v/laplacian.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        return self._like([c.laplacian() for c in self.components])

    def lap(self) -> "Chebfun3v":
        """Alias for :meth:`laplacian` (MATLAB lap)."""
        return self.laplacian()

    def jacobian(self) -> Chebfun3:
        """Determinant of the Jacobian of a 3-component field (MATLAB
        jacobian).

        Provenance
        ----------
        MATLAB source : @chebfun3v/jacobian.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Chebfun3.empty()
        if self.n_components != 3:
            raise ValueError("Chebfun3v.jacobian: Jacobian is not square.")
        f1, f2, f3 = self.components
        f1x, f1y, f1z = f1.diff(1), f1.diff(2), f1.diff(3)
        f2x, f2y, f2z = f2.diff(1), f2.diff(2), f2.diff(3)
        f3x, f3y, f3z = f3.diff(1), f3.diff(2), f3.diff(3)
        return (
            f1x * (f2y * f3z - f3y * f2z)
            - f2x * (f1y * f3z - f3y * f1z)
            + f3x * (f1y * f2z - f2y * f1z)
        )

    # ------------------------------------------------------------------
    # Range estimate
    # ------------------------------------------------------------------

    def minandmax3est(self, N: int = 25):
        """Estimate the range of each component on an ``N x N x N`` grid
        (MATLAB minandmax3est).

        Returns a length ``2 * n_components`` array of alternating minimum
        and maximum estimates.

        Provenance
        ----------
        MATLAB source : @chebfun3v/minandmax3est.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.asarray([], dtype=jnp.float64)
        import numpy as _np

        out = []
        for c in self.components:
            xa, xb, ya, yb, za, zb = c.domain
            # Second-kind Chebyshev points include the endpoints.
            t = _np.cos(_np.pi * _np.arange(N) / (N - 1))
            gx = 0.5 * (xa + xb) + 0.5 * (xb - xa) * t
            gy = 0.5 * (ya + yb) + 0.5 * (yb - ya) * t
            gz = 0.5 * (za + zb) + 0.5 * (zb - za) * t
            XX, YY, ZZ = _np.meshgrid(gx, gy, gz, indexing="ij")
            V = _np.asarray(
                c(jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()),
                  jnp.asarray(ZZ.ravel()))
            )
            out.append(float(_np.real(V).min()))
            out.append(float(_np.real(V).max()))
        return jnp.asarray(out, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Root finding
    # ------------------------------------------------------------------

    def root(self, ngrid: int = 81):
        """Find one common zero of a 3-component field (MATLAB root).

        A dense grid seeds the minimum of ``F_1^2 + F_2^2 + F_3^2``, then
        Newton's method on the 3x3 Jacobian polishes it to convergence.

        Provenance
        ----------
        MATLAB source : @chebfun3v/root.m, @chebfun3/root.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.asarray([], dtype=jnp.float64)
        if self.n_components != 3:
            raise ValueError("Chebfun3v.root needs 3 components.")
        import numpy as _np

        f, g, h = self.components
        xa, xb, ya, yb, za, zb = f.domain

        # Step 1: grid seed on second-kind Chebyshev points.
        t = _np.cos(_np.pi * _np.arange(ngrid) / (ngrid - 1))
        gx = 0.5 * (xa + xb) + 0.5 * (xb - xa) * t
        gy = 0.5 * (ya + yb) + 0.5 * (yb - ya) * t
        gz = 0.5 * (za + zb) + 0.5 * (zb - za) * t
        XX, YY, ZZ = _np.meshgrid(gx, gy, gz, indexing="ij")
        xf = jnp.asarray(XX.ravel())
        yf = jnp.asarray(YY.ravel())
        zf = jnp.asarray(ZZ.ravel())
        T = (_np.asarray(f(xf, yf, zf)) ** 2
             + _np.asarray(g(xf, yf, zf)) ** 2
             + _np.asarray(h(xf, yf, zf)) ** 2)
        idx = int(_np.argmin(_np.abs(T)))
        r = _np.array([float(XX.ravel()[idx]), float(YY.ravel()[idx]),
                       float(ZZ.ravel()[idx])])

        # Step 2: Newton's method on [f; g; h] = 0.
        grads = [(c.diff(1), c.diff(2), c.diff(3)) for c in (f, g, h)]

        def _feval(c, p):
            return float(_np.real(
                c(jnp.asarray(p[0]), jnp.asarray(p[1]), jnp.asarray(p[2]))))

        for _ in range(20):
            fv = _np.array([_feval(c, r) for c in (f, g, h)])
            J = _np.array([[_feval(d, r) for d in grads[i]] for i in range(3)])
            try:
                update = _np.linalg.solve(J, fv)
            except _np.linalg.LinAlgError:
                break
            r = r - update
            if _np.linalg.norm(update) < 1e-13:
                break
        return jnp.asarray(r, dtype=jnp.float64)

    def roots(self):
        """Common zeros of a 3-component field (MATLAB roots).

        Only the empty case and a single seeded root (via :meth:`root`) are
        provided; full multi-root continuation is not ported.

        Provenance
        ----------
        MATLAB source : @chebfun3v/roots.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.asarray([], dtype=jnp.float64)
        return self.root().reshape(1, 3)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integral(self, curve, domain=None):
        """Line integral ``int_C F_1 dx + F_2 dy + F_3 dz`` (MATLAB
        integral).

        ``curve`` is a callable ``t -> (x, y, z)`` with ``domain=(t0, t1)``,
        or a 1D array-valued Chebfun with three columns.

        Provenance
        ----------
        MATLAB source : @chebfun3v/integral.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain

        if self.n_components != 3:
            raise ValueError("Chebfun3v.integral needs 3 components.")

        if hasattr(curve, "funs") or isinstance(curve, Chebfun):
            cdom = curve.domain
            # ``cdom`` may be a Domain object (exposing .a/.b) or a plain
            # sequence of breakpoints; support both.
            if isinstance(cdom, Domain):
                t0, t1 = float(cdom.a), float(cdom.b)
            else:
                t0, t1 = float(cdom[0]), float(cdom[-1])

            def _comp(k):
                return Chebfun.from_function(
                    lambda t, k=k: jnp.asarray(curve(t))[..., k],
                    Domain((t0, t1)))
        else:
            if domain is None:
                raise ValueError("A callable curve requires domain=(t0, t1).")
            t0, t1 = float(domain[0]), float(domain[1])

            def _comp(k):
                return Chebfun.from_function(
                    lambda t, k=k: jnp.asarray(curve(t))[k],
                    Domain((t0, t1)))

        cx, cy, cz = _comp(0), _comp(1), _comp(2)
        dx, dy, dz = cx.diff(), cy.diff(), cz.diff()
        p, q, r = self.components

        def _integrand(t):
            xt, yt, zt = cx(t), cy(t), cz(t)
            return (p(xt, yt, zt) * dx(t) + q(xt, yt, zt) * dy(t)
                    + r(xt, yt, zt) * dz(t))

        return Chebfun.from_function(_integrand, Domain((t0, t1))).sum()

    def integral2(self, surface):
        """Flux integral ``int int_S <F, dS>`` through a parametric surface
        (MATLAB integral2).

        ``surface`` is a Chebfun2v with 3 components ``(x(u,v), y(u,v),
        z(u,v))``.  The flux is
        ``int int <F(S), S_u x S_v> du dv``.

        Provenance
        ----------
        MATLAB source : @chebfun3v/integral2.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d import chebfun2

        if self.n_components != 3:
            raise ValueError("Chebfun3v.integral2 needs 3 components.")
        comps = surface.components
        if len(comps) != 3:
            raise ValueError("integral2 surface must have 3 components.")
        s1, s2, s3 = comps[0], comps[1], comps[2]
        sdom = s1.approx.domain if hasattr(s1, "approx") else s1.domain
        f1, f2, f3 = self.components

        def _stacked(uv):
            return jnp.stack([s1(uv[0], uv[1]), s2(uv[0], uv[1]),
                              s3(uv[0], uv[1])])

        jac = jax.jacfwd(_stacked)

        def _one(u, v):
            jm = jac(jnp.stack([u, v]))          # (3, 2): columns S_u, S_v
            cr = jnp.cross(jm[:, 0], jm[:, 1])
            x, y, z = s1(u, v), s2(u, v), s3(u, v)
            fv = jnp.stack([f1(x, y, z), f2(x, y, z), f3(x, y, z)])
            return jnp.sum(fv * cr)

        def _integrand(u, v):
            u = jnp.asarray(u, dtype=jnp.float64)
            v = jnp.asarray(v, dtype=jnp.float64)
            return jnp.vectorize(_one)(u, v)

        return chebfun2(_integrand, domain=sdom).sum2()

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, op) -> "Chebfun3 | Chebfun3v":
        """Compose ``op(F)`` of this (real) field with a suitable operand.

        Following MATLAB @chebfun3v/compose:

        - 3 components with a Chebfun3 ``g`` -> Chebfun3 ``g(F_1, F_2, F_3)``.
        - 3 components with a 3-component Chebfun3v ``G`` -> Chebfun3v of
          the componentwise compositions.
        - 2 components with a Chebfun2 ``g`` -> Chebfun3 ``g(F_1, F_2)``.
        - 2 components with a Chebfun2v ``G`` -> Chebfun3v of the
          componentwise compositions.

        Provenance
        ----------
        MATLAB source : @chebfun3v/compose.m
        Chebfun commit: 7574c77
        """
        dom = self.domain
        n = self.n_components

        if n == 3:
            f1, f2, f3 = self.components
            if isinstance(op, Chebfun3):
                return Chebfun3.from_function(
                    lambda x, y, z: op(f1(x, y, z), f2(x, y, z), f3(x, y, z)),
                    domain=dom)
            if isinstance(op, Chebfun3v):
                return Chebfun3v([self.compose(g) for g in op.components])
            raise ValueError(
                "Chebfun3v.compose: a 3-component field composes with a "
                "Chebfun3 or a 3-component Chebfun3v.")

        if n == 2:
            f1, f2 = self.components
            comps = getattr(op, "components", None)
            if comps is not None:
                # Chebfun2v: compose each (callable) component in turn.
                return Chebfun3v([self.compose(g) for g in comps])
            # A scalar Chebfun2 / SeparableApprox (a callable of two args).
            return Chebfun3.from_function(
                lambda x, y, z: op(f1(x, y, z), f2(x, y, z)), domain=dom)

        raise ValueError("Chebfun3v.compose supports 2- or 3-component fields.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def quiver3(self, n: int = 9):
        """Sample the vector field on an ``n x n x n`` grid for a 3D quiver
        plot (MATLAB quiver3).

        Returns ``(X, Y, Z, U, V, W)`` arrays.  A thin, dependency-free
        sampler so that plotting harnesses (and the MATLAB "does not crash"
        test) have data to draw.

        Provenance
        ----------
        MATLAB source : @chebfun3v/quiver3.m
        Chebfun commit: 7574c77
        """
        if self.n_components != 3:
            raise ValueError("Chebfun3v.quiver3 needs 3 components.")
        xa, xb, ya, yb, za, zb = self.domain
        gx = jnp.linspace(xa, xb, n)
        gy = jnp.linspace(ya, yb, n)
        gz = jnp.linspace(za, zb, n)
        XX, YY, ZZ = jnp.meshgrid(gx, gy, gz, indexing="ij")
        f1, f2, f3 = self.components
        U = f1(XX, YY, ZZ)
        V = f2(XX, YY, ZZ)
        W = f3(XX, YY, ZZ)
        return XX, YY, ZZ, U, V, W

    def quiver(self, **kwargs):
        """Alias for :meth:`quiver3`."""
        return self.quiver3(**kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.isempty():
            return "Chebfun3v (empty)"
        lines = [f"Chebfun3v with {self.n_components} components:"]
        for k, c in enumerate(self.components):
            lines.append(f"  [{k}]: {c!r}")
        return "\n".join(lines)
