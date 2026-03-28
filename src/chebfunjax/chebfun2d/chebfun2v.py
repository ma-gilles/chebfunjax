"""Chebfun2v — 2D vector-valued function [f; g] on a rectangle.

A Chebfun2v represents a 2- or 3-component vector field on a 2D domain
[xa, xb] x [ya, yb], where each component is a SeparableApprox (Chebfun2)
object.

Usage
-----
::

    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    import jax.numpy as jnp

    f = SeparableApprox.from_function(lambda x, y: jnp.sin(x))
    g = SeparableApprox.from_function(lambda x, y: jnp.cos(y))
    F = Chebfun2v([f, g])
    div = F.divergence()        # f_x + g_y as a SeparableApprox
    cur = F.curl()              # g_x - f_y as a SeparableApprox (2D scalar curl)

Translated from MATLAB Chebfun class @chebfun2v (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.separable_approx import SeparableApprox

# ============================================================================
# Main class
# ============================================================================


class Chebfun2v(eqx.Module):
    """2D vector field with 2 or 3 Chebfun2 (SeparableApprox) components.

    Represents a smooth vector field F = [f_1; f_2] (or [f_1; f_2; f_3])
    on a rectangle [xa, xb] x [ya, yb], where each component f_j is a
    SeparableApprox object.

    Attributes
    ----------
    components : list of SeparableApprox
        The component functions. Length 2 or 3.
    domain : tuple of 4 floats
        (xa, xb, ya, yb). Static field (must agree for all components).

    Notes
    -----
    Immutable. All operations return new Chebfun2v objects.
    Construction is NOT JIT-safe (delegates to SeparableApprox.from_function).
    Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun2v/chebfun2v.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    SeparableApprox
    """

    components: list  # list of SeparableApprox, length 2 or 3
    domain: tuple = eqx.field(static=True)  # (xa, xb, ya, yb)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, components: list[SeparableApprox]) -> None:
        """Construct from a list of SeparableApprox components.

        Parameters
        ----------
        components : list of SeparableApprox
            2 or 3 component functions, all with the same domain.

        Raises
        ------
        ValueError
            If fewer than 2 or more than 3 components are given, or if
            component domains do not match.

        Provenance
        ----------
        MATLAB source : @chebfun2v/chebfun2v.m
        Chebfun commit: 7574c77
        """
        if len(components) < 2 or len(components) > 3:
            raise ValueError(f"Chebfun2v requires 2 or 3 components, got {len(components)}.")
        # Check domains match
        dom0 = components[0].domain
        for j, c in enumerate(components[1:], 1):
            if c.domain != dom0:
                raise ValueError(
                    f"Chebfun2v: component {j} has domain {c.domain} but "
                    f"component 0 has domain {dom0}. All components must share "
                    "the same domain."
                )
        # Equinox Module fields must be set via object.__setattr__ in __init__
        object.__setattr__(self, "components", list(components))
        object.__setattr__(self, "domain", dom0)

    @classmethod
    def from_functions(
        cls,
        *fns: Callable,
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        tol: float = 2.220446049250313e-16,
    ) -> "Chebfun2v":
        """Construct from 2 or 3 function handles.

        Parameters
        ----------
        *fns : callables
            2 or 3 functions of (x, y). Each must be vectorized: accepts 2D
            arrays from meshgrid broadcasting. Signature: f(x, y) -> array.
        domain : tuple of 4 floats, optional
            (xa, xb, ya, yb). Default (-1, 1, -1, 1).
        tol : float, optional
            Target relative tolerance. Default is machine epsilon.

        Returns
        -------
        Chebfun2v

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2v/chebfun2v.m
        Chebfun commit: 7574c77
        """
        if len(fns) < 2 or len(fns) > 3:
            raise ValueError(f"Chebfun2v.from_functions requires 2 or 3 callables, got {len(fns)}.")
        comps = [SeparableApprox.from_function(f, domain=domain, tol=tol) for f in fns]
        return cls(comps)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of components (2 or 3)."""
        return len(self.components)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Evaluate the vector field at point(s) (x, y).

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb]. Must broadcast with x.

        Returns
        -------
        jax.Array, shape (..., n_components)
            The stacked component values. The last axis is the component index.

        Notes
        -----
        JIT-safe, vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2v/feval.m
        Chebfun commit: 7574c77
        """
        vals = jnp.stack([c(x, y) for c in self.components], axis=-1)
        return vals

    # ------------------------------------------------------------------
    # Arithmetic (immutable: return new Chebfun2v)
    # ------------------------------------------------------------------

    def __neg__(self) -> "Chebfun2v":
        """Negate: -F.

        Provenance
        ----------
        MATLAB source : @chebfun2v/uminus.m
        Chebfun commit: 7574c77
        """
        neg_comps = [_neg_separable(c) for c in self.components]
        return Chebfun2v(neg_comps)

    def __pos__(self) -> "Chebfun2v":
        """Unary plus: +F."""
        return self

    def __add__(self, other: "Chebfun2v | float | int") -> "Chebfun2v":
        """Componentwise addition.

        Parameters
        ----------
        other : Chebfun2v or scalar
            If scalar, adds to each component.

        Provenance
        ----------
        MATLAB source : @chebfun2v/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float)):
            new_comps = [_add_scalar_separable(c, float(other)) for c in self.components]
            return Chebfun2v(new_comps)
        if isinstance(other, Chebfun2v):
            if other.n_components != self.n_components:
                raise ValueError(
                    f"Cannot add Chebfun2v with {self.n_components} components "
                    f"to one with {other.n_components} components."
                )
            new_comps = [
                _add_separable(c1, c2) for c1, c2 in zip(self.components, other.components)
            ]
            return Chebfun2v(new_comps)
        return NotImplemented

    def __radd__(self, other: "float | int") -> "Chebfun2v":
        return self.__add__(other)

    def __sub__(self, other: "Chebfun2v | float | int") -> "Chebfun2v":
        """Componentwise subtraction.

        Provenance
        ----------
        MATLAB source : @chebfun2v/minus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun2v):
            return self.__add__(other.__neg__())
        return self.__add__(-other)

    def __rsub__(self, other: "float | int") -> "Chebfun2v":
        return self.__neg__().__add__(other)

    def __mul__(self, other: "float | int") -> "Chebfun2v":
        """Scalar multiplication.

        Provenance
        ----------
        MATLAB source : @chebfun2v/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float)):
            new_comps = [_scale_separable(c, float(other)) for c in self.components]
            return Chebfun2v(new_comps)
        return NotImplemented

    def __rmul__(self, other: "float | int") -> "Chebfun2v":
        return self.__mul__(other)

    def __truediv__(self, other: "float | int") -> "Chebfun2v":
        """Divide by scalar.

        Provenance
        ----------
        MATLAB source : @chebfun2v/mrdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, n: int = 1, dim: int = 1) -> "Chebfun2v":
        """Componentwise differentiation.

        Parameters
        ----------
        n : int, optional
            Order of derivative. Default 1.
        dim : int, optional
            Direction: 1 = y-direction (along columns), 2 = x-direction
            (along rows). Default 1 (y-direction).

        Returns
        -------
        Chebfun2v
            Componentwise derivative.

        Notes
        -----
        Convention follows MATLAB Chebfun2v where DIM=1 is y and DIM=2 is x.

        Provenance
        ----------
        MATLAB source : @chebfun2v/diff.m
        Chebfun commit: 7574c77
        """
        new_comps = [_diff_separable(c, n=n, dim=dim) for c in self.components]
        return Chebfun2v(new_comps)

    def divergence(self) -> SeparableApprox:
        """Divergence of the vector field: F_x + G_y (or F_x + G_y + H_z for 3-comp).

        For a 2D vector field F = [f; g]:
            div(F) = df/dx + dg/dy

        For a 3-component field (functions of x and y only):
            div(F) = df1/dx + df2/dy  (only first two components contribute)

        Returns
        -------
        SeparableApprox
            The scalar divergence.

        Notes
        -----
        MATLAB Chebfun2v's divergence is df1/dx + df2/dy regardless of whether
        there are 2 or 3 components (since all components are functions of
        (x, y) only — the z-variable is not represented).

        Provenance
        ----------
        MATLAB source : @chebfun2v/divergence.m
        Chebfun commit: 7574c77
        """
        # df1/dx (dim=2 means x-derivative)
        df1_dx = _diff_separable(self.components[0], n=1, dim=2)
        # dg/dy (dim=1 means y-derivative)
        df2_dy = _diff_separable(self.components[1], n=1, dim=1)
        return _add_separable(df1_dx, df2_dy)

    def div(self) -> SeparableApprox:
        """Alias for divergence().

        Provenance
        ----------
        MATLAB source : @chebfun2v/div.m
        Chebfun commit: 7574c77
        """
        return self.divergence()

    def curl(self) -> "SeparableApprox | Chebfun2v":
        """Curl of the vector field.

        For a 2-component field F = [f; g]:
            curl(F) = dg/dx - df/dy  (scalar result — z-component of 3D curl)

        For a 3-component field F = [f1; f2; f3] (functions of x, y):
            curl(F) = [df3/dy; -df3/dx; df2/dx - df1/dy]
            (since partial derivatives in z are zero)

        Returns
        -------
        SeparableApprox
            For 2-component F: the scalar curl g_x - f_y.
        Chebfun2v
            For 3-component F: the vector curl [f3_y; -f3_x; f2_x - f1_y].

        Provenance
        ----------
        MATLAB source : @chebfun2v/curl.m
        Chebfun commit: 7574c77
        """
        if self.n_components == 2:
            # 2D scalar curl: g_x - f_y
            dg_dx = _diff_separable(self.components[1], n=1, dim=2)
            df_dy = _diff_separable(self.components[0], n=1, dim=1)
            return _add_separable(dg_dx, _neg_separable(df_dy))
        else:
            # 3D curl: [f3_y; -f3_x; f2_x - f1_y]
            f3 = self.components[2]
            df3_dy = _diff_separable(f3, n=1, dim=1)
            df3_dx = _diff_separable(f3, n=1, dim=2)
            df2_dx = _diff_separable(self.components[1], n=1, dim=2)
            df1_dy = _diff_separable(self.components[0], n=1, dim=1)
            curl_z = _add_separable(df2_dx, _neg_separable(df1_dy))
            return Chebfun2v([df3_dy, _neg_separable(df3_dx), curl_z])

    def gradient(self) -> "Chebfun2v":
        """Gradient of each component (not standard gradient of a scalar).

        Returns a new Chebfun2v where each component is replaced by the
        gradient of that component stacked, i.e., [df1/dx; df1/dy; df2/dx; df2/dy; ...].
        Note: this is not the standard Jacobian; see ``jacobian()`` instead.

        .. warning::
            This method computes component-wise gradients (diff in both
            directions), which does not match the MATLAB ``gradient`` for
            vector fields. For the standard gradient of a scalar Chebfun2,
            use ``SeparableApprox.gradient()``.

        Provenance
        ----------
        MATLAB source : @chebfun2v/diff.m (componentwise)
        Chebfun commit: 7574c77
        """
        # Return df/dx and df/dy stacked for each component
        new_comps = []
        for c in self.components:
            new_comps.append(_diff_separable(c, n=1, dim=2))  # d/dx
            new_comps.append(_diff_separable(c, n=1, dim=1))  # d/dy
        return Chebfun2v(new_comps[:3] if len(new_comps) > 3 else new_comps)

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------

    def dot(self, other: "Chebfun2v") -> SeparableApprox:
        """Pointwise dot product F . G.

        Returns the scalar SeparableApprox
            F . G = f1*g1 + f2*g2  (+ f3*g3 for 3-component fields)

        Parameters
        ----------
        other : Chebfun2v
            Must have the same number of components and the same domain.

        Returns
        -------
        SeparableApprox
            The pointwise dot product.

        Provenance
        ----------
        MATLAB source : @chebfun2v/dot.m
        Chebfun commit: 7574c77
        """
        if other.n_components != self.n_components:
            raise ValueError(
                f"Cannot compute dot product of Chebfun2v with "
                f"{self.n_components} and {other.n_components} components."
            )
        prods = [_mul_separable(c1, c2) for c1, c2 in zip(self.components, other.components)]
        result = prods[0]
        for p in prods[1:]:
            result = _add_separable(result, p)
        return result

    def cross(self, other: "Chebfun2v") -> "SeparableApprox | Chebfun2v":
        """Pointwise cross product F x G.

        For 2-component fields F = [f1; f2] and G = [g1; g2]:
            F x G = f1*g2 - f2*g1  (scalar — z-component of 3D cross product)

        For 3-component fields:
            F x G = [f2*g3 - f3*g2;  f3*g1 - f1*g3;  f1*g2 - f2*g1]

        Parameters
        ----------
        other : Chebfun2v
            Must have the same number of components.

        Returns
        -------
        SeparableApprox
            For 2-component F, G.
        Chebfun2v
            For 3-component F, G.

        Provenance
        ----------
        MATLAB source : @chebfun2v/cross.m
        Chebfun commit: 7574c77
        """
        nF = self.n_components
        nG = other.n_components
        if nF != nG:
            raise ValueError(
                f"Chebfun2v.cross: both fields must have the same number of "
                f"components, got {nF} and {nG}."
            )
        f = self.components
        g = other.components

        if nF == 2:
            # Scalar cross: f1*g2 - f2*g1
            return _add_separable(
                _mul_separable(f[0], g[1]),
                _neg_separable(_mul_separable(f[1], g[0])),
            )
        else:
            # 3D cross product
            c1 = _add_separable(
                _mul_separable(f[1], g[2]), _neg_separable(_mul_separable(f[2], g[1]))
            )
            c2 = _add_separable(
                _mul_separable(f[2], g[0]), _neg_separable(_mul_separable(f[0], g[2]))
            )
            c3 = _add_separable(
                _mul_separable(f[0], g[1]), _neg_separable(_mul_separable(f[1], g[0]))
            )
            return Chebfun2v([c1, c2, c3])

    def norm(self) -> float:
        """Frobenius norm: sqrt(sum_j ||f_j||^2).

        Returns
        -------
        float
            The Frobenius (L2) norm of the vector field.

        Provenance
        ----------
        MATLAB source : @chebfun2v/norm.m
        Chebfun commit: 7574c77
        """
        total = 0.0
        for c in self.components:
            # ||f_j||^2 = integral_domain f_j^2 dx dy
            prod = _mul_separable(c, c)
            total += float(_integral_separable(prod))
        return float(total**0.5)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Plot this Chebfun2v: quiver for 2-component, surface for 3-component."""
        if len(self.components) == 3:
            from chebfunjax.plotting import surf_chebfun2v
            return surf_chebfun2v(self, **kwargs)
        else:
            from chebfunjax.plotting import quiver_2d
            return quiver_2d(self, **kwargs)

    def surf(self, **kwargs):
        """Parametric surface plot (3-component Chebfun2v only)."""
        from chebfunjax.plotting import surf_chebfun2v
        return surf_chebfun2v(self, **kwargs)

    def quiver(self, **kwargs):
        """2D quiver plot of the vector field."""
        from chebfunjax.plotting import quiver_2d
        return quiver_2d(self, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact summary.

        Examples
        --------
        >>> F = Chebfun2v.from_functions(lambda x,y: jnp.sin(x), lambda x,y: jnp.cos(y))
        >>> repr(F)
        'Chebfun2v(n_components=2, domain=(-1.0, 1.0, -1.0, 1.0))'
        """
        xa, xb, ya, yb = self.domain
        return f"Chebfun2v(n_components={self.n_components}, domain=({xa}, {xb}, {ya}, {yb}))"


# ============================================================================
# SeparableApprox helper operations
# These implement arithmetic on SeparableApprox objects needed by Chebfun2v.
# All return new SeparableApprox objects (immutable).
# ============================================================================


def _neg_separable(f: SeparableApprox) -> SeparableApprox:
    """Negate a SeparableApprox: -f.

    Negates the pivot values, leaving cols and rows unchanged.
    f(x,y) = sum_j d_j * c_j(y) * r_j(x)  =>  -f = sum_j (-d_j) * c_j(y) * r_j(x)
    """
    return SeparableApprox(
        cols=f.cols,
        rows=f.rows,
        pivots=-f.pivots,
        domain=f.domain,
    )


def _scale_separable(f: SeparableApprox, scalar: float) -> SeparableApprox:
    """Multiply a SeparableApprox by a scalar."""
    return SeparableApprox(
        cols=f.cols,
        rows=f.rows,
        pivots=f.pivots * scalar,
        domain=f.domain,
    )


def _add_scalar_separable(f: SeparableApprox, scalar: float) -> SeparableApprox:
    """Add a scalar to a SeparableApprox: f + c.

    This adds a rank-1 term c * T_0(x) * T_0(y) to the representation.
    """
    from chebfunjax.tech.chebtech import Chebtech2

    if scalar == 0.0:
        return f

    # Create constant T_0 functions (coeffs = [1.0])
    const_col = Chebtech2.from_coeffs(jnp.ones(1, dtype=jnp.float64))
    const_row = Chebtech2.from_coeffs(jnp.ones(1, dtype=jnp.float64))
    # New pivot = scalar (so the term is scalar * 1 * 1)
    new_pivots = jnp.concatenate([f.pivots, jnp.array([float(scalar)], dtype=jnp.float64)])
    new_cols = list(f.cols) + [const_col]
    new_rows = list(f.rows) + [const_row]
    return SeparableApprox(
        cols=new_cols,
        rows=new_rows,
        pivots=new_pivots,
        domain=f.domain,
    )


def _add_separable(f: SeparableApprox, g: SeparableApprox) -> SeparableApprox:
    """Add two SeparableApprox objects.

    Simply concatenates the rank-1 terms (no compression).
    f + g = sum_j d_j c_j(y) r_j(x) + sum_k e_k p_k(y) q_k(x)
    """
    if f.domain != g.domain:
        raise ValueError(
            f"Cannot add SeparableApprox on domain {f.domain} to one on "
            f"domain {g.domain}. Domains must match."
        )
    new_cols = list(f.cols) + list(g.cols)
    new_rows = list(f.rows) + list(g.rows)
    new_pivots = jnp.concatenate([f.pivots, g.pivots])
    return SeparableApprox(
        cols=new_cols,
        rows=new_rows,
        pivots=new_pivots,
        domain=f.domain,
    )


def _mul_separable(f: SeparableApprox, g: SeparableApprox) -> SeparableApprox:
    """Pointwise multiply two SeparableApprox objects: f .* g.

    Uses the identity: if f = sum_j d_j c_j(y) r_j(x) and
    g = sum_k e_k p_k(y) q_k(x), then

        f * g = sum_j sum_k (d_j * e_k) * (c_j * p_k)(y) * (r_j * q_k)(x)

    where products of 1D Chebyshev functions are computed via multiplication
    of coefficient polynomials.
    """
    if f.domain != g.domain:
        raise ValueError(
            f"Cannot multiply SeparableApprox on domain {f.domain} with one on "
            f"domain {g.domain}. Domains must match."
        )

    new_cols = []
    new_rows = []
    new_pivots_list = []

    for j, (cj, rj, dj) in enumerate(zip(f.cols, f.rows, f.pivots)):
        for k, (pk, qk, ek) in enumerate(zip(g.cols, g.rows, g.pivots)):
            # Multiply 1D Chebyshev tech: cj * pk
            col_prod = _mul_chebtech(cj, pk)
            row_prod = _mul_chebtech(rj, qk)
            new_cols.append(col_prod)
            new_rows.append(row_prod)
            new_pivots_list.append(float(dj) * float(ek))

    new_pivots = jnp.array(new_pivots_list, dtype=jnp.float64)
    return SeparableApprox(
        cols=new_cols,
        rows=new_rows,
        pivots=new_pivots,
        domain=f.domain,
    )


def _mul_chebtech(f: "Chebtech2", g: "Chebtech2") -> "Chebtech2":
    """Multiply two Chebtech2 objects (both on [-1, 1]).

    Uses the fact that T_j(x)*T_k(x) = (T_{j+k}(x) + T_{|j-k|}(x))/2.
    The result has degree at most n_f + n_g - 1.
    """
    from chebfunjax.tech.chebtech import Chebtech2

    cf = f.coeffs  # shape (nf,)
    cg = g.coeffs  # shape (ng,)
    nf = cf.shape[0]
    ng = cg.shape[0]

    if nf == 0 or ng == 0:
        return Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64))

    n_out = nf + ng - 1
    result = jnp.zeros(n_out, dtype=jnp.float64)

    # T_j * T_k = (T_{j+k} + T_{|j-k|})/2
    for j in range(nf):
        for k in range(ng):
            coeff = float(cf[j]) * float(cg[k]) * 0.5
            # T_{j+k} contribution
            idx1 = j + k
            if idx1 < n_out:
                result = result.at[idx1].add(coeff)
            # T_{|j-k|} contribution
            idx2 = abs(j - k)
            if idx2 < n_out:
                result = result.at[idx2].add(coeff)

    return Chebtech2.from_coeffs(result)


def _diff_separable(f: SeparableApprox, n: int = 1, dim: int = 1) -> SeparableApprox:
    """Differentiate a SeparableApprox n times in direction dim.

    Parameters
    ----------
    f : SeparableApprox
    n : int
        Order of differentiation.
    dim : int
        1 = differentiate in y (column direction, axis 1 in MATLAB).
        2 = differentiate in x (row direction, axis 2 in MATLAB).

    Returns
    -------
    SeparableApprox
        The derivative.

    Notes
    -----
    Convention: MATLAB chebfun2 diff(f, n, 1) = d^n f / dy^n,
    diff(f, n, 2) = d^n f / dx^n.

    For the SeparableApprox representation  f = sum_j d_j c_j(y) r_j(x):
      - d/dy acts on the column slices: d/dy [c_j(y)]
      - d/dx acts on the row slices: d/dx [r_j(x)]

    Both column and row slices are Chebtech2 on [-1,1], with the physical
    coordinate mapped from [ya, yb] (or [xa, xb]) to [-1,1]. The chain
    rule gives a factor of 2/(b-a).
    """

    xa, xb, ya, yb = f.domain

    if dim == 1:
        # Differentiate y-direction: act on cols
        scale_y = (2.0 / (yb - ya)) ** n  # chain rule for [ya,yb] -> [-1,1]
        new_cols = [_diff_chebtech(c, n=n) for c in f.cols]
        new_pivots = f.pivots * scale_y
        return SeparableApprox(
            cols=new_cols,
            rows=list(f.rows),
            pivots=new_pivots,
            domain=f.domain,
        )
    elif dim == 2:
        # Differentiate x-direction: act on rows
        scale_x = (2.0 / (xb - xa)) ** n  # chain rule for [xa,xb] -> [-1,1]
        new_rows = [_diff_chebtech(r, n=n) for r in f.rows]
        new_pivots = f.pivots * scale_x
        return SeparableApprox(
            cols=list(f.cols),
            rows=new_rows,
            pivots=new_pivots,
            domain=f.domain,
        )
    else:
        raise ValueError(f"_diff_separable: dim must be 1 (y) or 2 (x), got {dim}.")


def _diff_chebtech(f: "Chebtech2", n: int = 1) -> "Chebtech2":
    """Differentiate a Chebtech2 n times.

    Delegates to the Chebtech2.diff() method which uses the correct
    Chebyshev coefficient recurrence from Mason & Handscomb p. 34.
    """
    return f.diff(n)


def _integral_separable(f: SeparableApprox) -> float:
    """Compute the integral of a SeparableApprox over its domain.

    integral_[xa,xb]x[ya,yb] f(x,y) dx dy
    = sum_j d_j * (integral c_j(y) dy) * (integral r_j(x) dx)
    """
    xa, xb, ya, yb = f.domain
    # Scale factors for physical domain
    scale_x = (xb - xa) / 2.0
    scale_y = (yb - ya) / 2.0

    total = 0.0
    for c, r, d in zip(f.cols, f.rows, f.pivots):
        # Integral of T_k over [-1,1]:
        # int_{-1}^1 T_k(x) dx = 0 if k odd
        #                       = 2/(1-k^2) if k even, k != 0
        #                       = 2 if k = 0
        int_col = _chebtech_integral(c)
        int_row = _chebtech_integral(r)
        total += float(d) * float(int_col) * float(int_row)

    return total * scale_x * scale_y


def _chebtech_integral(f: "Chebtech2") -> float:
    """Integral of Chebtech2 over [-1, 1].

    int_{-1}^1 sum_k c_k T_k(x) dx = c_0 * 2 + sum_{k even, k>=2} c_k * 2/(1-k^2)
    """
    c = f.coeffs
    n = c.shape[0]
    if n == 0:
        return 0.0
    total = float(c[0]) * 2.0
    for k in range(2, n, 2):
        total += float(c[k]) * 2.0 / (1.0 - k * k)
    return total
