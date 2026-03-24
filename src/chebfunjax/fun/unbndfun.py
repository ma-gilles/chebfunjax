"""Unbounded-domain function (Unbndfun) — smooth functions on semi-infinite or
doubly-infinite intervals.

Translated from MATLAB Chebfun class @unbndfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import legpts

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)

# Scale factor used in the MATLAB unbounded map (s=1, c=0 default).
# For [a, ∞) or (-∞, b]:  scale = 15
# For (-∞, ∞):            scale = 5
_SCALE_SEMI = 15.0
_SCALE_BOTH = 5.0


# ============================================================================
# Pure-function mapping helpers (JIT-safe, grad-safe, vmap-safe)
# ============================================================================


def _forward_right(y: jax.Array, a: float) -> jax.Array:
    """Map s ∈ [-1, 1] → x ∈ [a, +∞).

    Formula (MATLAB: ``ForHandle = @(y) 15*s*(y+1)/(1-y) + a``):
        x = 15*(y + 1) / (1 - y) + a

    At y = -1: x = a.  At y → 1: x → +∞.

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (b==inf branch)
    Chebfun commit: 7574c77
    """
    return _SCALE_SEMI * (y + 1.0) / (1.0 - y) + a


def _inverse_right(x: jax.Array, a: float) -> jax.Array:
    """Map x ∈ [a, +∞) → s ∈ [-1, 1].

    Formula (MATLAB: ``InvHandle = @(x) (-15*s + x - a)./(15*s + x - a)``):
        s = (x - a - 15) / (x - a + 15)

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (b==inf branch)
    Chebfun commit: 7574c77
    """
    u = x - a
    return (u - _SCALE_SEMI) / (u + _SCALE_SEMI)


def _derivative_right(y: jax.Array) -> jax.Array:
    """Derivative dx/dy for the right semi-infinite map.

    Formula (MATLAB: ``DerHandle = @(y) 15*s*2./(y-1).^2``):
        dx/dy = 30 / (y - 1)²

    Note: (y-1)² = (1-y)², so the sign is positive everywhere on [-1,1).

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (b==inf branch)
    Chebfun commit: 7574c77
    """
    return _SCALE_SEMI * 2.0 / (y - 1.0) ** 2


def _forward_left(y: jax.Array, b: float) -> jax.Array:
    """Map s ∈ [-1, 1] → x ∈ (-∞, b].

    Formula (MATLAB: ``ForHandle = @(y) 15*s*(y-1)./(y+1) + b``):
        x = 15*(y - 1) / (y + 1) + b

    At y = 1: x = b.  At y → -1: x → -∞.

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf branch)
    Chebfun commit: 7574c77
    """
    return _SCALE_SEMI * (y - 1.0) / (y + 1.0) + b


def _inverse_left(x: jax.Array, b: float) -> jax.Array:
    """Map x ∈ (-∞, b] → s ∈ [-1, 1].

    Formula (MATLAB: ``InvHandle = @(x) (15*s + x - b)./(15*s - x + b)``):
        s = (15 + x - b) / (15 - x + b)

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf branch)
    Chebfun commit: 7574c77
    """
    u = x - b
    return (_SCALE_SEMI + u) / (_SCALE_SEMI - u)


def _derivative_left(y: jax.Array) -> jax.Array:
    """Derivative dx/dy for the left semi-infinite map.

    Formula (MATLAB: ``DerHandle = @(y) 15*s*2./(y+1).^2``):
        dx/dy = 30 / (y + 1)²

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf branch)
    Chebfun commit: 7574c77
    """
    return _SCALE_SEMI * 2.0 / (y + 1.0) ** 2


def _forward_both(y: jax.Array) -> jax.Array:
    """Map s ∈ (-1, 1) → x ∈ (-∞, +∞).

    Formula (MATLAB: ``ForHandle = @(y) 5*s*y./(1 - min(y.^2, 1))``):
        x = 5*y / (1 - y²)

    The ``min(y², 1)`` clamp prevents division by zero when |y|=1;
    at those endpoints the function diverges to ±∞.

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf, b==inf branch)
    Chebfun commit: 7574c77
    """
    y2 = jnp.minimum(y ** 2, jnp.float64(1.0))
    return _SCALE_BOTH * y / (1.0 - y2)


def _inverse_both(x: jax.Array) -> jax.Array:
    """Map x ∈ (-∞, +∞) → s ∈ (-1, 1).

    Formula (MATLAB: ``InvHandle = @(x) 2*x./(5*s + sqrt(25*s^2 + 4*x.^2))``):
        s = 2*x / (5 + sqrt(25 + 4*x²))

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf, b==inf branch)
    Chebfun commit: 7574c77
    """
    s = _SCALE_BOTH
    return 2.0 * x / (s + jnp.sqrt(s ** 2 + 4.0 * x ** 2))


def _derivative_both(y: jax.Array) -> jax.Array:
    """Derivative dx/dy for the doubly-infinite map.

    Formula (MATLAB: ``DerHandle = @(y) 5*s*(1 + y.^2)./(1 - y.^2).^2``):
        dx/dy = 5*(1 + y²) / (1 - y²)²

    Provenance
    ----------
    MATLAB source : @mapping/mapping.m ``unbounded`` static method (a==-inf, b==inf branch)
    Chebfun commit: 7574c77
    """
    return _SCALE_BOTH * (1.0 + y ** 2) / (1.0 - y ** 2) ** 2


# ============================================================================
# Unbndfun class
# ============================================================================


class Unbndfun(eqx.Module):
    """Smooth function on a semi-infinite or doubly-infinite interval.

    ``Unbndfun`` represents a smooth function on an unbounded interval by
    mapping to the standard interval [-1, 1] via a nonlinear algebraic map
    and storing the mapped function as a :class:`~chebfunjax.tech.chebtech.Chebtech2`.

    The three supported domain types and their forward maps (from reference
    variable ``y ∈ [-1, 1]`` to physical variable ``x``) are:

    * ``'right_inf'`` — domain ``[a, +∞)``:
        ``x = 15*(y + 1) / (1 - y) + a``
    * ``'left_inf'``  — domain ``(-∞, b]``:
        ``x = 15*(y - 1) / (y + 1) + b``
    * ``'both_inf'``  — domain ``(-∞, +∞)``:
        ``x = 5*y / (1 - y²)``

    Evaluation, differentiation, and integration all account for the Jacobian
    ``dx/dy`` of the forward map.

    Attributes
    ----------
    onefun : Chebtech2
        Chebyshev representation of the mapped function on [-1, 1].
    domain : Domain
        The unbounded domain (a single-interval Domain with at least one
        infinite endpoint).
    mapping_type : str
        One of ``'right_inf'``, ``'left_inf'``, or ``'both_inf'`` (static).

    Examples
    --------
    Integral of exp(-x²) over (-∞, ∞) equals √π:

    >>> import math, jax.numpy as jnp
    >>> from chebfunjax.fun.unbndfun import Unbndfun
    >>> from chebfunjax.domain import Domain
    >>> d = Domain((-jnp.inf, jnp.inf))
    >>> f = Unbndfun.from_function(lambda x: jnp.exp(-x**2), d)
    >>> abs(float(f.sum()) - math.sqrt(math.pi)) < 1e-12
    True

    Integral of exp(-x) over [0, ∞) equals 1:

    >>> d = Domain((0.0, jnp.inf))
    >>> g = Unbndfun.from_function(jnp.exp, d)      # construct exp
    >>> g_neg = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
    >>> abs(float(g_neg.sum()) - 1.0) < 1e-12
    True

    Provenance
    ----------
    MATLAB source : @unbndfun/unbndfun.m, @mapping/mapping.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2, Domain, Bndfun
    """

    onefun: Chebtech2
    domain: Domain = eqx.field(static=True)
    mapping_type: str = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        *,
        n: int | None = None,
    ) -> "Unbndfun":
        """Construct an Unbndfun from a callable on an unbounded domain.

        The callable ``f`` is evaluated at Chebyshev-2 points mapped from
        [-1, 1] to the physical unbounded domain.  If ``n`` is ``None``
        (default), an adaptive algorithm doubles the grid size until the
        Chebyshev coefficients decay to machine precision.

        Parameters
        ----------
        f : callable
            Vectorised function accepting and returning ``jax.Array``.
        domain : Domain
            A single-interval domain with at least one infinite endpoint.
            Use ``Domain((-jnp.inf, jnp.inf))``, ``Domain((a, jnp.inf))``,
            or ``Domain((-jnp.inf, b))``.
        n : int or None, optional
            Fixed number of Chebyshev points.  ``None`` triggers adaptive
            construction.

        Returns
        -------
        Unbndfun
            A new Unbndfun instance.

        Raises
        ------
        ValueError
            If ``domain`` is not a single-interval domain, or if neither
            endpoint is infinite.

        Examples
        --------
        >>> from chebfunjax.domain import Domain
        >>> import jax.numpy as jnp
        >>> d = Domain((0.0, jnp.inf))
        >>> f = Unbndfun.from_function(lambda x: jnp.exp(-x), d)
        >>> f.mapping_type
        'right_inf'

        Provenance
        ----------
        MATLAB source : @unbndfun/unbndfun.m
        Chebfun commit: 7574c77
        """
        _validate_unbounded_domain(domain)
        mtype = _mapping_type(domain)
        a = domain.a
        b = domain.b

        # Build the composed function: evaluate f at the forward-mapped points.
        # At y=±1 the forward map sends points to ±∞, and f(±∞) may be NaN
        # in floating-point (e.g. x*exp(-x²) = ∞*0 = NaN at x=∞).
        # We sanitise the output with nan_to_num so that the Chebtech2
        # constructor receives clean values (NaN → 0 = the physical limit).
        if mtype == "right_inf":
            mapped_f = lambda y: jnp.nan_to_num(  # noqa: E731
                f(_forward_right(y, a)), nan=0.0, posinf=0.0, neginf=0.0
            )
        elif mtype == "left_inf":
            mapped_f = lambda y: jnp.nan_to_num(  # noqa: E731
                f(_forward_left(y, b)), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:  # both_inf
            mapped_f = lambda y: jnp.nan_to_num(  # noqa: E731
                f(_forward_both(y)), nan=0.0, posinf=0.0, neginf=0.0
            )

        onefun = Chebtech2.from_function(mapped_f, n=n)
        return cls(onefun=onefun, domain=domain, mapping_type=mtype)

    @classmethod
    def from_chebtech(
        cls,
        tech: Chebtech2,
        domain: Domain,
    ) -> "Unbndfun":
        """Wrap an existing Chebtech2 in an unbounded domain mapping.

        Parameters
        ----------
        tech : Chebtech2
            An already-constructed Chebtech2 on [-1, 1].  Its values are
            interpreted as ``f(map(y))`` for the appropriate ``map``.
        domain : Domain
            A single-interval domain with at least one infinite endpoint.

        Returns
        -------
        Unbndfun
            A new Unbndfun instance.

        Raises
        ------
        ValueError
            If ``domain`` is not a single-interval domain, or if neither
            endpoint is infinite.

        Provenance
        ----------
        MATLAB source : @unbndfun/unbndfun.m
        Chebfun commit: 7574c77
        """
        _validate_unbounded_domain(domain)
        mtype = _mapping_type(domain)
        return cls(onefun=tech, domain=domain, mapping_type=mtype)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the function at physical point(s) x.

        Maps x to the reference interval [-1, 1] via the inverse map and
        delegates to ``self.onefun``.  Points at ±∞ are mapped to ±1.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s) in the physical domain.

        Returns
        -------
        y : jax.Array, same shape as x
            Function values.

        Notes
        -----
        JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @unbndfun/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        # domain.a and domain.b are Python floats (static=True on domain),
        # so they are concrete constants at trace time — safe to use as scalars.
        a: float = self.domain.a
        b: float = self.domain.b
        mtype = self.mapping_type  # static string, safe in JIT

        if mtype == "right_inf":
            y = _inverse_right(x, a)
        elif mtype == "left_inf":
            y = _inverse_left(x, b)
        else:
            y = _inverse_both(x)

        # Clamp infinite inputs to ±1 (MATLAB: z(mask) = sign(x(mask)))
        y = jnp.where(jnp.isinf(x), jnp.sign(x), y)
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
        """Function values at the Chebyshev-2 points on the reference interval."""
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
    # Mapping accessors
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def forward_map(self, y: jax.Array) -> jax.Array:
        """Forward map from [-1, 1] to the physical domain.

        Parameters
        ----------
        y : jax.Array
            Reference points in [-1, 1].

        Returns
        -------
        jax.Array
            Physical points.

        Notes
        -----
        JIT-safe.
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        a = float(self.domain.a)
        b = float(self.domain.b)
        if self.mapping_type == "right_inf":
            return _forward_right(y, a)
        elif self.mapping_type == "left_inf":
            return _forward_left(y, b)
        else:
            return _forward_both(y)

    @eqx.filter_jit
    def map_derivative(self, y: jax.Array) -> jax.Array:
        """Derivative dx/dy of the forward map at reference points y.

        The Jacobian factor required for change-of-variables in integration
        and differentiation.

        Parameters
        ----------
        y : jax.Array
            Reference points in [-1, 1].

        Returns
        -------
        jax.Array
            Jacobian values dx/dy.

        Notes
        -----
        JIT-safe.

        Provenance
        ----------
        MATLAB source : @mapping/mapping.m (DerHandle definitions)
        Chebfun commit: 7574c77
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        if self.mapping_type == "right_inf":
            return _derivative_right(y)
        elif self.mapping_type == "left_inf":
            return _derivative_left(y)
        else:
            return _derivative_both(y)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "Unbndfun":
        """Differentiate *k* times with respect to physical variable x.

        By the chain rule, ``df/dx = (df/dy) / (dx/dy)`` where y is the
        reference variable and ``dx/dy`` is the map derivative.  This is
        applied iteratively for higher-order derivatives.

        The approach mirrors MATLAB's @unbndfun/diff.m:
        * Build a representation of ``1 / (dx/dy)`` on [-1, 1].
        * For each differentiation order: differentiate the onefun in
          reference coordinates, then multiply by the inverse-Jacobian.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Unbndfun
            The k-th derivative on the same domain.

        Notes
        -----
        Construction is NOT JIT-safe (calls ``Chebtech2.from_function``).
        The resulting Unbndfun can be evaluated under JIT.

        Provenance
        ----------
        MATLAB source : @unbndfun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return self

        # Build onefun representation of 1/(dx/dy) on [-1,1]
        inv_der_onefun = Chebtech2.from_function(
            lambda y: jnp.float64(1.0) / self.map_derivative(y)
        )
        inv_der = Unbndfun(
            onefun=inv_der_onefun,
            domain=self.domain,
            mapping_type=self.mapping_type,
        )

        result = self
        for _ in range(k):
            # Differentiate in reference coordinates (onefun.diff())
            diff_onefun = result.onefun.diff()
            result_diff = Unbndfun(
                onefun=diff_onefun,
                domain=self.domain,
                mapping_type=self.mapping_type,
            )
            # Apply chain rule: multiply by 1/(dx/dy)
            result = result_diff * inv_der

        return result

    def cumsum(self) -> "Unbndfun":
        """Indefinite integral with the constant chosen so that F(left endpoint) = 0.

        Uses the substitution rule:
            ∫ f(x) dx = ∫ f(map(y)) * (dx/dy) dy

        The integrand ``f(map(y)) * (dx/dy)`` is a smooth function on [-1, 1]
        and is represented as a new Chebtech2, whose ``cumsum`` (on [-1, 1])
        gives the antiderivative in reference coordinates.

        Returns
        -------
        Unbndfun
            The antiderivative on the same domain.

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @unbndfun/cumsum.m
        Chebfun commit: 7574c77
        """
        # Build onefun for the integrand: f(map(y)) * (dx/dy)
        # = self.onefun(y) * map_derivative(y).
        # At boundaries y=±1, map_derivative → ∞ while onefun → 0;
        # IEEE 754 gives 0 * ∞ = NaN, so we sanitise with nan_to_num.
        def integrand_fn(y: jax.Array) -> jax.Array:
            raw = self.onefun(y) * self.map_derivative(y)
            return jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        integrand_onefun = Chebtech2.from_function(
            integrand_fn,
            n=self.n,  # same resolution as self
        )
        cumsum_onefun = integrand_onefun.cumsum()
        return Unbndfun(
            onefun=cumsum_onefun,
            domain=self.domain,
            mapping_type=self.mapping_type,
        )

    def sum(self) -> jax.Array:
        """Definite integral over the (unbounded) domain.

        Uses the substitution rule:
            ∫_{domain} f(x) dx = ∫_{-1}^{1} f(map(y)) * (dx/dy) dy

        The integrand is constructed as a new Chebtech2 of the same length
        (following MATLAB's heuristic in @unbndfun/sum.m: fixed-length
        construction to avoid issues with the nonlinear map near ±1).

        Returns
        -------
        jax.Array, scalar
            The definite integral.

        Notes
        -----
        Construction is NOT JIT-safe.  The resulting scalar can be used
        inside JIT if the Unbndfun is already constructed outside.

        Provenance
        ----------
        MATLAB source : @unbndfun/sum.m, ``unbndfunIntegrand``
        Chebfun commit: 7574c77
        """
        # Integrate via Gauss-Legendre quadrature on the reference interval [-1,1].
        #
        # The substitution rule gives:
        #   ∫_{domain} f(x) dx = ∫_{-1}^{1} f(map(y)) * (dx/dy) dy
        #
        # Gauss-Legendre nodes are strictly interior to (-1, 1), which avoids the
        # endpoint singularity where dx/dy → ∞.  The Chebyshev-2 extreme points
        # include y=±1 exactly, causing 0*∞ = NaN/Inf; GL quadrature is free of
        # this issue.
        #
        # We use n_gl ≥ n to ensure sufficient quadrature accuracy.
        n_gl = max(self.n, 64)
        pts_gl, wts_gl = legpts(n_gl)
        # Evaluate integrand at GL nodes (all interior, so der is finite)
        fy = self.onefun(pts_gl)
        der = self.map_derivative(pts_gl)
        return jnp.dot(wts_gl, fy * der)

    def inner(self, other: "Unbndfun") -> jax.Array:
        """L2 inner product ⟨f, g⟩ = ∫_{domain} f(x) g(x) dx.

        Parameters
        ----------
        other : Unbndfun
            Must have the same domain and mapping_type.

        Returns
        -------
        jax.Array, scalar
            The inner product.

        Notes
        -----
        Construction is NOT JIT-safe.
        """
        self._check_domain(other)

        def integrand_fn(y: jax.Array) -> jax.Array:
            raw = self.onefun(y) * other.onefun(y) * self.map_derivative(y)
            return jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        n = max(self.n, other.n)
        integrand_onefun = Chebtech2.from_function(integrand_fn, n=n)
        return integrand_onefun.sum()

    def norm(self, p: float = 2.0) -> jax.Array:
        """L-p norm on the unbounded domain.

        For p=2, computes ``sqrt(∫_{domain} f(x)² dx)``.

        Parameters
        ----------
        p : float, default 2.0
            The norm order.  Only p=2 and p=∞ are supported.

        Returns
        -------
        jax.Array, scalar
            The norm value.
        """
        if p == 2.0:
            return jnp.sqrt(self.inner(self))
        elif p == jnp.inf or p == float("inf"):
            return jnp.array(self.onefun.vscale, dtype=jnp.float64)
        else:
            raise ValueError(
                f"norm(p={p}) is not supported for Unbndfun. Use p=2 or p=inf."
            )

    # ------------------------------------------------------------------
    # Arithmetic (pointwise; domains must match)
    # ------------------------------------------------------------------

    def _check_domain(self, other: "Unbndfun") -> None:
        """Raise ValueError if two Unbndfuns have different domains."""
        if self.domain != other.domain:
            raise ValueError(
                f"Cannot perform arithmetic on Unbndfun on {self.domain} "
                f"and Unbndfun on {other.domain}: domains do not match."
            )

    def __add__(self, other) -> "Unbndfun":
        """Pointwise addition.

        Provenance
        ----------
        MATLAB source : @classicfun/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Unbndfun):
            self._check_domain(other)
            return Unbndfun(self.onefun + other.onefun, self.domain, self.mapping_type)
        return Unbndfun(self.onefun + other, self.domain, self.mapping_type)

    def __radd__(self, other) -> "Unbndfun":
        return self.__add__(other)

    def __sub__(self, other) -> "Unbndfun":
        """Pointwise subtraction.

        Provenance
        ----------
        MATLAB source : @classicfun/minus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Unbndfun):
            self._check_domain(other)
            return Unbndfun(self.onefun - other.onefun, self.domain, self.mapping_type)
        return Unbndfun(self.onefun - other, self.domain, self.mapping_type)

    def __rsub__(self, other) -> "Unbndfun":
        return -(self - other)

    def __neg__(self) -> "Unbndfun":
        """Unary negation."""
        return Unbndfun(-self.onefun, self.domain, self.mapping_type)

    def __pos__(self) -> "Unbndfun":
        """Unary plus (returns self)."""
        return Unbndfun(self.onefun, self.domain, self.mapping_type)

    def __mul__(self, other) -> "Unbndfun":
        """Pointwise multiplication.

        Provenance
        ----------
        MATLAB source : @classicfun/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Unbndfun):
            self._check_domain(other)
            return Unbndfun(self.onefun * other.onefun, self.domain, self.mapping_type)
        return Unbndfun(self.onefun * other, self.domain, self.mapping_type)

    def __rmul__(self, other) -> "Unbndfun":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Unbndfun":
        """Division.

        Provenance
        ----------
        MATLAB source : @classicfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Unbndfun):
            self._check_domain(other)
            return Unbndfun(self.onefun / other.onefun, self.domain, self.mapping_type)
        return Unbndfun(self.onefun / other, self.domain, self.mapping_type)

    def __rtruediv__(self, other) -> "Unbndfun":
        return Unbndfun(other / self.onefun, self.domain, self.mapping_type)

    def __pow__(self, exponent) -> "Unbndfun":
        """Raise to a power."""
        if isinstance(exponent, Unbndfun):
            self._check_domain(exponent)
            return Unbndfun(
                self.onefun ** exponent.onefun, self.domain, self.mapping_type
            )
        return Unbndfun(self.onefun ** exponent, self.domain, self.mapping_type)

    def __abs__(self) -> "Unbndfun":
        """Absolute value (NOT JIT-safe)."""
        return Unbndfun(abs(self.onefun), self.domain, self.mapping_type)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display.

        Examples
        --------
        >>> f = Unbndfun.from_function(lambda x: jnp.exp(-x**2), Domain((-jnp.inf, jnp.inf)))
        >>> "Unbndfun" in repr(f)
        True
        """
        a_inf = math.isinf(self.domain.a) and self.domain.a < 0
        a_str = "-inf" if a_inf else f"{self.domain.a:.4g}"
        b_inf = math.isinf(self.domain.b) and self.domain.b > 0
        b_str = "inf" if b_inf else f"{self.domain.b:.4g}"
        lval = float(self.onefun(jnp.float64(-1.0)))
        rval = float(self.onefun(jnp.float64(1.0)))
        return (
            f"Unbndfun([{a_str}, {b_str}], n={self.n}, "
            f"lval={lval:.4g}, rval={rval:.4g})"
        )


# ============================================================================
# Module-level helpers
# ============================================================================


def _validate_unbounded_domain(domain: Domain) -> None:
    """Raise ValueError if domain is not a valid single-interval unbounded domain.

    Parameters
    ----------
    domain : Domain
        Domain to validate.

    Raises
    ------
    ValueError
        If the domain has more than one interval, or if neither endpoint is
        infinite.
    """
    if domain.n_intervals != 1:
        raise ValueError(
            f"Unbndfun requires a single-interval domain, but got a domain "
            f"with {domain.n_intervals} intervals: {domain}. "
            f"Use a Domain with exactly 2 breakpoints."
        )
    a, b = domain.a, domain.b
    if not (math.isinf(a) or math.isinf(b)):
        raise ValueError(
            f"Unbndfun requires at least one infinite endpoint, but the domain "
            f"[{a}, {b}] is bounded. Use Bndfun for bounded domains."
        )
    if math.isinf(a) and a > 0:
        raise ValueError(
            f"Left endpoint must be -inf (not +inf): got a={a}."
        )
    if math.isinf(b) and b < 0:
        raise ValueError(
            f"Right endpoint must be +inf (not -inf): got b={b}."
        )


def _mapping_type(domain: Domain) -> str:
    """Return the mapping type string for a validated unbounded domain.

    Parameters
    ----------
    domain : Domain
        A validated unbounded single-interval domain.

    Returns
    -------
    str
        One of ``'right_inf'``, ``'left_inf'``, or ``'both_inf'``.
    """
    a, b = domain.a, domain.b
    if math.isinf(a) and math.isinf(b):
        return "both_inf"
    elif math.isinf(b):
        return "right_inf"
    else:
        return "left_inf"
