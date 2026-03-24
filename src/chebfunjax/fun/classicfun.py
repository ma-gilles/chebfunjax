"""Abstract base class for functions on arbitrary intervals [a, b].

Translated from MATLAB Chebfun class @classicfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


class Classicfun(eqx.Module):
    """Abstract base class for smooth functions on a bounded interval [a, b].

    A Classicfun represents a smooth function on an interval [a, b] by
    wrapping a Chebtech2 (which lives on the standard interval [-1, 1]) with
    an affine domain mapping.  Concrete subclasses (``Bndfun``) handle
    different types of domain maps.

    The separation is: the ``onefun`` holds all function approximation logic
    (coefficient representation, evaluation at [-1,1], arithmetic, calculus),
    while the Classicfun layer applies the affine map so everything is
    expressed in physical coordinates [a, b].

    Attributes
    ----------
    onefun : Chebtech2
        The underlying Chebyshev representation on [-1, 1].
    domain : Domain
        The bounded interval [a, b] (exactly one sub-interval).

    Notes
    -----
    All binary operations assume the two Classicfun objects share the same
    domain.  This is checked at the Python level and raises ``ValueError``
    if the domains differ.

    Provenance
    ----------
    MATLAB source : @classicfun/classicfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Bndfun, Chebtech2, Domain
    """

    onefun: Chebtech2
    domain: Domain = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Construction (class methods — NOT __init__)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        *,
        n: int | None = None,
    ) -> "Classicfun":
        """Construct from a callable on the given domain.

        Parameters
        ----------
        f : callable
            Function accepting and returning ``jax.Array``.  Must be
            vectorised (accept an array of points and return an array of
            values of the same shape).
        domain : Domain
            A single-interval domain [a, b].
        n : int or None, optional
            Fixed number of Chebyshev points.  If ``None`` (default), use
            the adaptive algorithm.

        Returns
        -------
        Classicfun
            A new instance.
        """
        ...

    @classmethod
    @abstractmethod
    def from_chebtech(cls, tech: Chebtech2, domain: Domain) -> "Classicfun":
        """Wrap an existing Chebtech2 in a domain mapping.

        Parameters
        ----------
        tech : Chebtech2
            An already-constructed Chebtech2 on [-1, 1].
        domain : Domain
            A single-interval domain [a, b].

        Returns
        -------
        Classicfun
            A new instance.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the function at physical point(s) x in [a, b].

        Maps x from [a, b] to [-1, 1] via the inverse affine map and
        delegates to ``self.onefun``.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s) in [a, b].

        Returns
        -------
        y : jax.Array, same shape as x
            Function values.

        Notes
        -----
        JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @bndfun/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        # Map from [a, b] to [-1, 1]
        y = self.domain.inverse_map(x)
        return self.onefun(y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients."""
        return self.onefun.n

    @property
    def coeffs(self) -> jax.Array:
        """Chebyshev coefficients of the underlying onefun."""
        return self.onefun.coeffs

    @property
    def values(self) -> jax.Array:
        """Function values at the Chebyshev-2 points on [a, b]."""
        return self.onefun.values

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute function value."""
        return self.onefun.vscale

    @property
    def ishappy(self) -> bool:
        """True if the representation is adaptively resolved."""
        return self.onefun.ishappy

    def __len__(self) -> int:
        """Number of Chebyshev coefficients."""
        return self.n

    # ------------------------------------------------------------------
    # Arithmetic (delegate to onefun, check domains match)
    # ------------------------------------------------------------------

    def _check_domain(self, other: "Classicfun") -> None:
        """Raise ValueError if two Classicfuns have different domains."""
        if self.domain != other.domain:
            raise ValueError(
                f"Cannot perform arithmetic on Classicfun on {self.domain} "
                f"and Classicfun on {other.domain}: domains do not match. "
                f"Use f.restrict({other.domain.a}, {other.domain.b}) to "
                f"restrict the domain first."
            )

    def __add__(self, other) -> "Classicfun":
        """Add two Classicfuns or a Classicfun and a scalar.

        Provenance
        ----------
        MATLAB source : @classicfun/plus.m (delegates to @chebtech/plus.m)
        Chebfun commit: 7574c77
        """
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun + other.onefun, self.domain)
        else:
            return self.__class__(self.onefun + other, self.domain)

    def __radd__(self, other) -> "Classicfun":
        return self.__add__(other)

    def __sub__(self, other) -> "Classicfun":
        """Subtract two Classicfuns or subtract a scalar.

        Provenance
        ----------
        MATLAB source : @classicfun/minus.m (delegates to @chebtech/minus.m)
        Chebfun commit: 7574c77
        """
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun - other.onefun, self.domain)
        else:
            return self.__class__(self.onefun - other, self.domain)

    def __rsub__(self, other) -> "Classicfun":
        return -(self - other)

    def __neg__(self) -> "Classicfun":
        """Unary negation.

        Provenance
        ----------
        MATLAB source : @classicfun/uminus.m
        Chebfun commit: 7574c77
        """
        return self.__class__(-self.onefun, self.domain)

    def __pos__(self) -> "Classicfun":
        """Unary plus (returns self)."""
        return self.__class__(self.onefun, self.domain)

    def __mul__(self, other) -> "Classicfun":
        """Pointwise multiplication.

        Provenance
        ----------
        MATLAB source : @classicfun/times.m (delegates to @chebtech/times.m)
        Chebfun commit: 7574c77
        """
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun * other.onefun, self.domain)
        else:
            return self.__class__(self.onefun * other, self.domain)

    def __rmul__(self, other) -> "Classicfun":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Classicfun":
        """Division.

        Provenance
        ----------
        MATLAB source : @classicfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun / other.onefun, self.domain)
        else:
            return self.__class__(self.onefun / other, self.domain)

    def __rtruediv__(self, other) -> "Classicfun":
        """Scalar divided by Classicfun."""
        return self.__class__(other / self.onefun, self.domain)

    def __pow__(self, exponent) -> "Classicfun":
        """Raise to a power.

        Provenance
        ----------
        MATLAB source : @classicfun/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, Classicfun):
            self._check_domain(exponent)
            return self.__class__(self.onefun ** exponent.onefun, self.domain)
        else:
            return self.__class__(self.onefun ** exponent, self.domain)

    def __abs__(self) -> "Classicfun":
        """Absolute value.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @classicfun/abs.m
        Chebfun commit: 7574c77
        """
        return self.__class__(abs(self.onefun), self.domain)

    # ------------------------------------------------------------------
    # Calculus (with domain scaling)
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "Classicfun":
        """Differentiate *k* times with respect to x in [a, b].

        Applies the chain rule: the derivative with respect to x is the
        derivative with respect to the mapped variable (in [-1, 1])
        divided by the Jacobian ``(b - a) / 2`` raised to the *k*-th power.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Classicfun
            The k-th derivative, still on the same domain [a, b].

        Provenance
        ----------
        MATLAB source : @bndfun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return self.__class__(self.onefun, self.domain)
        # Jacobian raised to the k-th power
        rescale = (self.domain.map_derivative()) ** k  # ((b-a)/2)^k
        new_onefun = self.onefun.diff(k) * (1.0 / rescale)
        return self.__class__(new_onefun, self.domain)

    def cumsum(self) -> "Classicfun":
        """Indefinite integral (antiderivative) with F(a) = 0.

        Scales the output of the underlying Chebtech2 cumsum (which
        satisfies F(-1) = 0 on [-1, 1]) by the Jacobian ``(b - a) / 2``
        to account for the change-of-variables.

        Returns
        -------
        Classicfun
            The antiderivative on the same domain [a, b].

        Provenance
        ----------
        MATLAB source : @bndfun/cumsum.m
        Chebfun commit: 7574c77
        """
        rescale = self.domain.map_derivative()  # (b-a)/2
        new_onefun = self.onefun.cumsum() * rescale
        return self.__class__(new_onefun, self.domain)

    def sum(self) -> jax.Array:
        """Definite integral over [a, b].

        Scales the integral of the underlying Chebtech2 (over [-1, 1])
        by ``(b - a) / 2``.

        Returns
        -------
        jax.Array, scalar
            The integral ``∫_a^b f(x) dx``.

        Provenance
        ----------
        MATLAB source : @bndfun/sum.m
        Chebfun commit: 7574c77
        """
        rescale = self.domain.map_derivative()  # (b-a)/2
        return self.onefun.sum() * jnp.float64(rescale)

    def inner(self, other: "Classicfun") -> jax.Array:
        """L2 inner product ⟨f, g⟩ = ∫_a^b f(x) g(x) dx.

        Parameters
        ----------
        other : Classicfun
            Must have the same domain.

        Returns
        -------
        jax.Array, scalar
            The inner product.

        Provenance
        ----------
        MATLAB source : @bndfun/innerProduct.m (delegates to @chebtech/innerProduct.m)
        Chebfun commit: 7574c77
        """
        self._check_domain(other)
        rescale = self.domain.map_derivative()  # (b-a)/2
        return self.onefun.inner(other.onefun) * jnp.float64(rescale)

    def norm(self, p: float = 2.0) -> jax.Array:
        """L-p norm on [a, b].

        For p=2, computes ``sqrt(∫_a^b f(x)² dx)``.

        Parameters
        ----------
        p : float, default 2.0
            The norm order.  Only p=2 uses the Chebyshev inner product.
            For p=1 or p=inf it delegates to pointwise evaluation.

        Returns
        -------
        jax.Array, scalar
            The norm value.

        Provenance
        ----------
        MATLAB source : @classicfun/normest.m
        Chebfun commit: 7574c77
        """
        if p == 2.0:
            rescale = self.domain.map_derivative()
            return (self.onefun.norm(p=2.0) ** 2 * jnp.float64(rescale)) ** 0.5
        elif p == jnp.inf or p == float("inf"):
            return jnp.array(self.onefun.vscale, dtype=jnp.float64)
        elif p == 1.0:
            return abs(self).sum()
        else:
            raise ValueError(
                f"norm(p={p}) is not supported. Use p=1, p=2, or p=inf."
            )

    def mean(self) -> jax.Array:
        """Mean value of the function over [a, b].

        Computes ``(1 / (b - a)) * ∫_a^b f(x) dx``.

        Returns
        -------
        jax.Array, scalar
        """
        a, b = self.domain.a, self.domain.b
        return self.sum() / jnp.float64(b - a)

    # ------------------------------------------------------------------
    # Rootfinding and extrema
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """Real roots in [a, b].

        Delegates to the underlying Chebtech2 rootfinder (which returns
        roots in [-1, 1]) and maps them back to [a, b].

        NOT JIT-safe (variable output size).

        Returns
        -------
        jax.Array, shape (n_roots,)
            Sorted roots in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/roots.m
        Chebfun commit: 7574c77
        """
        # Roots in [-1, 1]
        onefun_roots = self.onefun.roots()
        # Map to [a, b]
        return self.domain.forward_map(onefun_roots)

    def minandmax(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Global minimum and maximum on [a, b].

        Computes extrema of ``onefun`` on [-1, 1] and maps the positions
        back to [a, b] via the forward affine map.

        NOT JIT-safe.

        Returns
        -------
        (min_val, min_pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position in [a, b].
        (max_val, max_pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/minandmax.m
        Chebfun commit: 7574c77
        """
        (min_val, min_y), (max_val, max_y) = self.onefun.minandmax()
        # Map positions from [-1, 1] to [a, b]
        min_pos = self.domain.forward_map(min_y)
        max_pos = self.domain.forward_map(max_y)
        return (min_val, min_pos), (max_val, max_pos)

    def min(self) -> tuple[jax.Array, jax.Array]:
        """Global minimum on [a, b].

        NOT JIT-safe.

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/min.m
        Chebfun commit: 7574c77
        """
        (min_val, min_pos), _ = self.minandmax()
        return min_val, min_pos

    def max(self) -> tuple[jax.Array, jax.Array]:
        """Global maximum on [a, b].

        NOT JIT-safe.

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/max.m
        Chebfun commit: 7574c77
        """
        _, (max_val, max_pos) = self.minandmax()
        return max_val, max_pos

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    @abstractmethod
    def restrict(self, a: float, b: float) -> "Classicfun":
        """Restrict to a sub-interval [a, b].

        Parameters
        ----------
        a : float
            Left endpoint of the sub-interval.
        b : float
            Right endpoint of the sub-interval.

        Returns
        -------
        Classicfun
            A new instance representing the same function on [a, b].

        Raises
        ------
        ValueError
            If [a, b] is not a sub-interval of the current domain.
        """
        ...

    # ------------------------------------------------------------------
    # Simplify
    # ------------------------------------------------------------------

    def simplify(self, tol: float | None = None) -> "Classicfun":
        """Return a simplified (potentially shorter) representation.

        Delegates to ``self.onefun.simplify()``.

        Parameters
        ----------
        tol : float or None, optional
            Tolerance for coefficient chopping.

        Returns
        -------
        Classicfun
            Simplified instance.

        Provenance
        ----------
        MATLAB source : @classicfun/simplify.m
        Chebfun commit: 7574c77
        """
        return self.__class__(self.onefun.simplify(tol), self.domain)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        a, b = self.domain.a, self.domain.b
        lval = float(self.onefun(jnp.float64(-1.0)))
        rval = float(self.onefun(jnp.float64(1.0)))
        return (
            f"{self.__class__.__name__}("
            f"[{a:.4g}, {b:.4g}], n={self.n}, "
            f"lval={lval:.4g}, rval={rval:.4g})"
        )
