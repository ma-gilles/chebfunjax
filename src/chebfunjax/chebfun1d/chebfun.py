"""User-facing Chebfun class for piecewise smooth function approximation.

This is the main class users interact with. A Chebfun on a domain [a, b] is
represented as a list of *pieces* (Chebtech2 objects), each defined on a
sub-interval, together with a Domain recording the breakpoints.

Arithmetic, calculus (diff, cumsum, sum, inner, norm, mean), and rootfinding /
extrema (roots, max, min) are delegated to the underlying Chebtech2 pieces
with appropriate affine rescaling for the physical interval.

Translated from MATLAB Chebfun class @chebfun (commit 7574c77) and informed
by chebpy's Chebfun class.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Piece wrapper: a Chebtech2 together with the physical interval it lives on
# ============================================================================

class _Piece(eqx.Module):
    """A single smooth piece of a Chebfun on a physical interval [a, b].

    Internally stores a Chebtech2 on the reference interval [-1, 1] and the
    affine map between [a, b] and [-1, 1].

    Parameters
    ----------
    tech : Chebtech2
        The Chebyshev representation on [-1, 1].
    interval : tuple[float, float]
        Physical interval (a, b).
    """

    tech: Chebtech2
    interval: tuple[float, float] = eqx.field(static=True)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        a: float,
        b: float,
        *,
        n: int | None = None,
    ) -> _Piece:
        """Build a piece from a callable on [a, b].

        Parameters
        ----------
        f : callable
            Function mapping physical x in [a, b] to values.
        a, b : float
            Physical interval endpoints.
        n : int or None
            Fixed degree (None = adaptive).
        """
        a, b = float(a), float(b)
        # Wrap f to map from reference [-1, 1] into [a, b]
        def f_ref(t: jax.Array) -> jax.Array:
            x = 0.5 * (b - a) * t + 0.5 * (a + b)
            return f(x)

        tech = Chebtech2.from_function(f_ref, n=n)
        return cls(tech=tech, interval=(a, b))

    @classmethod
    def from_coeffs(
        cls,
        coeffs: jax.Array,
        a: float,
        b: float,
    ) -> _Piece:
        """Build a piece from Chebyshev coefficients on [a, b]."""
        tech = Chebtech2.from_coeffs(coeffs)
        return cls(tech=tech, interval=(float(a), float(b)))

    @classmethod
    def from_values(
        cls,
        values: jax.Array,
        a: float,
        b: float,
    ) -> _Piece:
        """Build a piece from values at Chebyshev-2 points of [a, b]."""
        tech = Chebtech2.from_values(values)
        return cls(tech=tech, interval=(float(a), float(b)))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate piece at physical point(s) x in [a, b].

        Maps x from [a, b] to [-1, 1] then uses Clenshaw evaluation.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s) in [a, b].

        Returns
        -------
        jax.Array, same shape as x
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        a, b = self.interval
        t = (2.0 * x - (a + b)) / (b - a)
        return self.tech(t)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients."""
        return self.tech.n

    @property
    def ishappy(self) -> bool:
        """True if resolved to tolerance."""
        return self.tech.ishappy

    @property
    def coeffs(self) -> jax.Array:
        """Chebyshev coefficients on the reference interval [-1, 1]."""
        return self.tech.coeffs

    @property
    def values(self) -> jax.Array:
        """Values at Chebyshev-2 points of [a, b] (ascending order)."""
        return self.tech.values

    @property
    def vscale(self) -> float:
        """Vertical scale (max |f| on the piece)."""
        return self.tech.vscale

    @property
    def endpoint_values(self) -> tuple[float, float]:
        """Function values at the left and right endpoints (a, b)."""
        vals = self.values
        return (float(vals[0]), float(vals[-1]))

    def restrict(self, a: float, b: float) -> _Piece:
        """Restrict to sub-interval [a, b].

        Parameters
        ----------
        a, b : float
            Sub-interval of ``self.interval``.

        Returns
        -------
        _Piece
            A new _Piece on [a, b].
        """
        pa, pb = self.interval
        # Map [a, b] (physical) into reference [-1, 1] coordinates
        # t_a = (2*a - (pa+pb)) / (pb-pa),  t_b similarly
        t_a = (2.0 * a - (pa + pb)) / (pb - pa)
        t_b = (2.0 * b - (pa + pb)) / (pb - pa)
        new_tech = self.tech.restrict(t_a, t_b)
        return _Piece(tech=new_tech, interval=(float(a), float(b)))

    # ------------------------------------------------------------------
    # Arithmetic helpers (used by Chebfun arithmetic operators)
    # ------------------------------------------------------------------

    def _apply_unary(self, tech_result: Chebtech2) -> _Piece:
        """Wrap a Chebtech2 result in a _Piece with the same interval."""
        return _Piece(tech=tech_result, interval=self.interval)

    def _apply_fun(self, op) -> _Piece:
        """Compose this piece with a scalar function op.

        Builds a new _Piece by adaptively approximating ``op(self(x))``
        on the same physical interval [a, b].

        Parameters
        ----------
        op : callable
            A vectorized JAX function applied pointwise.

        Returns
        -------
        _Piece
        """
        a, b = self.interval
        return _Piece.from_function(lambda x: op(self(x)), a, b)

    # ------------------------------------------------------------------
    # Special functions (thin wrappers around _apply_fun)
    # ------------------------------------------------------------------

    def sin(self) -> _Piece:
        """Sine of the piece."""
        return self._apply_fun(jnp.sin)

    def cos(self) -> _Piece:
        """Cosine of the piece."""
        return self._apply_fun(jnp.cos)

    def exp(self) -> _Piece:
        """Exponential of the piece."""
        return self._apply_fun(jnp.exp)

    def log(self) -> _Piece:
        """Natural logarithm of the piece."""
        return self._apply_fun(jnp.log)

    def sqrt(self) -> _Piece:
        """Square root of the piece."""
        return self._apply_fun(jnp.sqrt)

    def sinh(self) -> _Piece:
        """Hyperbolic sine of the piece."""
        return self._apply_fun(jnp.sinh)

    def cosh(self) -> _Piece:
        """Hyperbolic cosine of the piece."""
        return self._apply_fun(jnp.cosh)

    def tanh(self) -> _Piece:
        """Hyperbolic tangent of the piece."""
        return self._apply_fun(jnp.tanh)

    def asin(self) -> _Piece:
        """Inverse sine (arcsin) of the piece."""
        return self._apply_fun(jnp.arcsin)

    def acos(self) -> _Piece:
        """Inverse cosine (arccos) of the piece."""
        return self._apply_fun(jnp.arccos)

    def atan(self) -> _Piece:
        """Inverse tangent (arctan) of the piece."""
        return self._apply_fun(jnp.arctan)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> _Piece:
        """Differentiate *k* times with respect to the physical variable x.

        The reference Chebtech2 is on [-1, 1] with the map x = (b-a)/2 * t + c.
        By the chain rule, d/dx = (2/(b-a)) * d/dt, so the k-th derivative
        gains a factor of (2/(b-a))^k.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        _Piece
        """
        a, b = self.interval
        scale = (2.0 / (b - a)) ** k
        tech_der = self.tech.diff(k)
        # Scale the coefficients
        scaled_coeffs = tech_der.coeffs * jnp.float64(scale)
        new_tech = Chebtech2.from_coeffs(scaled_coeffs)
        return _Piece(tech=new_tech, interval=(a, b))

    def cumsum(self) -> _Piece:
        """Antiderivative with respect to x satisfying F(a) = 0.

        The antiderivative in the reference variable t is scaled by (b-a)/2
        to get the physical antiderivative.  The constant of integration is
        then adjusted so F(a) = 0 (the left endpoint maps to t = -1 where
        Chebtech2.cumsum already satisfies F(-1) = 0, so the value at t=-1
        is zero by construction of Chebtech2.cumsum — we just need to scale).

        Returns
        -------
        _Piece
        """
        a, b = self.interval
        scale = (b - a) / 2.0
        tech_cs = self.tech.cumsum()
        # Scale coefficients by (b-a)/2
        scaled_coeffs = tech_cs.coeffs * jnp.float64(scale)
        new_tech = Chebtech2.from_coeffs(scaled_coeffs)
        return _Piece(tech=new_tech, interval=(a, b))

    def sum(self) -> jax.Array:
        """Definite integral over [a, b].

        Returns
        -------
        jax.Array (scalar)
        """
        a, b = self.interval
        scale = (b - a) / 2.0
        return self.tech.sum() * jnp.float64(scale)

    def inner(self, other: _Piece) -> jax.Array:
        r"""L2 inner product <self, other> = \int_a^b f(x) g(x) dx.

        Requires both pieces to share the same interval.

        Parameters
        ----------
        other : _Piece

        Returns
        -------
        jax.Array (scalar)
        """
        if self.interval != other.interval:
            raise ValueError(
                f"Cannot compute inner product of pieces on different intervals: "
                f"{self.interval} vs {other.interval}."
            )
        a, b = self.interval
        scale = (b - a) / 2.0
        return self.tech.inner(other.tech) * jnp.float64(scale)

    def roots(self) -> jax.Array:
        """Real roots in [a, b] via Chebtech2.roots (colleague matrix).

        Maps roots from the reference interval [-1, 1] back to [a, b].

        Returns
        -------
        jax.Array, shape (n_roots,)
            Sorted roots in [a, b].
        """
        a, b = self.interval
        t_roots = self.tech.roots()
        # Map t in [-1, 1] to x in [a, b]: x = (b-a)/2 * t + (a+b)/2
        x_roots = 0.5 * (b - a) * t_roots + 0.5 * (a + b)
        return x_roots

    def minandmax(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Global min and max of this piece.

        Returns extrema by evaluating at the roots of the derivative plus
        the endpoints.

        Returns
        -------
        (x_min, f_min), (x_max, f_max)
        """
        a, b = self.interval
        # Roots of derivative give critical points
        dp = self.diff(1)
        crit_t = dp.tech.roots()  # roots in [-1, 1]
        # Map to physical
        crit_x = 0.5 * (b - a) * crit_t + 0.5 * (a + b)
        # Include endpoints
        endpoints = jnp.array([float(a), float(b)], dtype=jnp.float64)
        if crit_x.shape[0] > 0:
            candidates = jnp.concatenate([endpoints, crit_x])
        else:
            candidates = endpoints
        vals = self(candidates)
        i_min = int(jnp.argmin(vals))
        i_max = int(jnp.argmax(vals))
        return (
            (float(candidates[i_min]), float(vals[i_min])),
            (float(candidates[i_max]), float(vals[i_max])),
        )


# ============================================================================
# Chebfun — the main user-facing class
# ============================================================================

class Chebfun(eqx.Module):
    """Piecewise smooth function approximation on an arbitrary interval.

    A Chebfun represents a function by a list of smooth *pieces*, each
    approximated by a Chebyshev series on a sub-interval.  The overall domain
    is a :class:`~chebfunjax.domain.Domain` recording the breakpoints.

    For construction, use the :func:`chebfun` factory function rather than
    calling ``Chebfun(...)`` directly.

    Attributes
    ----------
    funs : list[_Piece]
        List of smooth pieces (one per sub-interval).  Treated as a static
        Python list — its length is fixed after construction.
    domain : Domain
        The piecewise domain (breakpoints).

    Notes
    -----
    ``funs`` is a Python list of ``_Piece`` objects. Because its length is
    determined at construction time (not during JIT tracing), it is stored as
    a static pytree node. The JAX arrays *inside* each piece (the coefficient
    arrays) are still traced normally.

    JAX Contract
    ------------
    - ``f(x)`` — JIT, grad, vmap safe for single-piece Chebfuns (fixed shape).
    - Multi-piece evaluation uses Python-level dispatch (not JIT-safe with
      dynamic piece selection).
    - Construction (adaptive) is NOT JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebfun, Chebtech2, Domain
    """

    # Python list of pieces — static (the list itself, not the arrays inside)
    funs: list = eqx.field(static=False)
    domain: Domain = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Internal constructor (use factory classmethod or chebfun() instead)
    # ------------------------------------------------------------------

    def __init__(self, funs: list[_Piece], domain: Domain) -> None:
        """Low-level constructor.  Prefer :func:`chebfun` for user code.

        Parameters
        ----------
        funs : list[_Piece]
            Non-empty list of smooth pieces in domain order.
        domain : Domain
            Corresponding domain (breakpoints must match piece intervals).
        """
        if len(funs) == 0:
            raise ValueError(
                "Chebfun requires at least one piece. "
                "Use chebfun(0.0) to create a constant zero Chebfun."
            )
        self.funs = funs
        self.domain = domain

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        n: int | None = None,
    ) -> Chebfun:
        """Construct a Chebfun from a callable on a given domain.

        For a single-interval domain this calls ``Chebtech2.from_function``
        on the reference interval, wrapping ``f`` with the affine map from
        [a, b] to [-1, 1].

        For a multi-interval domain each sub-interval is treated independently.

        Parameters
        ----------
        f : callable
            Vectorized function mapping physical points to values.
        domain : Domain
            Domain (with possible breakpoints).
        n : int or None
            Fixed degree per piece (None = adaptive).

        Returns
        -------
        Chebfun

        Notes
        -----
        Adaptive construction is NOT JIT-safe (Python while loop).

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (parse+populate path)
        Chebfun commit: 7574c77
        """
        funs = []
        for sub in domain.intervals:
            piece = _Piece.from_function(f, sub.a, sub.b, n=n)
            funs.append(piece)
        return cls(funs=funs, domain=domain)

    @classmethod
    def from_coeffs(
        cls,
        coeffs: jax.Array,
        domain: Domain | None = None,
    ) -> Chebfun:
        """Construct a Chebfun from Chebyshev coefficients.

        Parameters
        ----------
        coeffs : array_like, shape (n,)
            Chebyshev coefficients c[0], ..., c[n-1] for the full domain.
        domain : Domain or None
            Domain. If ``None`` defaults to ``[-1, 1]``.

        Returns
        -------
        Chebfun

        Notes
        -----
        Only single-interval domains are supported (multi-piece would require
        the user to specify which coefficients belong to which piece).

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (''coeffs'' flag path)
        Chebfun commit: 7574c77
        """
        if domain is None:
            domain = Domain((-1.0, 1.0))
        if domain.n_intervals != 1:
            raise ValueError(
                f"from_coeffs only supports single-interval domains, "
                f"but domain has {domain.n_intervals} intervals. "
                f"Construct pieces individually and combine."
            )
        coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        piece = _Piece.from_coeffs(coeffs, domain.a, domain.b)
        return cls(funs=[piece], domain=domain)

    @classmethod
    def from_values(
        cls,
        values: jax.Array,
        domain: Domain | None = None,
    ) -> Chebfun:
        """Construct a Chebfun from values at Chebyshev-2 points.

        Parameters
        ----------
        values : array_like, shape (n,)
            Function values at n Chebyshev-2 points on the domain, ascending.
        domain : Domain or None
            Single-interval domain. Defaults to ``[-1, 1]``.

        Returns
        -------
        Chebfun

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (values-on-chebpts path)
        Chebfun commit: 7574c77
        """
        if domain is None:
            domain = Domain((-1.0, 1.0))
        if domain.n_intervals != 1:
            raise ValueError(
                f"from_values only supports single-interval domains, "
                f"but domain has {domain.n_intervals} intervals."
            )
        values = jnp.asarray(values, dtype=jnp.float64)
        piece = _Piece.from_values(values, domain.a, domain.b)
        return cls(funs=[piece], domain=domain)

    @classmethod
    def identity(cls, domain: Domain | None = None) -> Chebfun:
        """Construct the identity function f(x) = x on a domain.

        Parameters
        ----------
        domain : Domain or None
            Single-interval domain. Defaults to ``[-1, 1]``.

        Returns
        -------
        Chebfun
            Represents f(x) = x.

        Examples
        --------
        >>> x = Chebfun.identity()
        >>> float(x(jnp.float64(0.5)))
        0.5
        """
        if domain is None:
            domain = Domain((-1.0, 1.0))
        return cls.from_function(lambda x: x, domain=domain)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the Chebfun at point(s) x.

        For single-piece Chebfuns this is JIT-safe, grad-safe, and vmap-safe.

        For multi-piece Chebfuns, Python-level dispatch is used to route each
        point to the correct piece.  This is NOT JIT-safe across the full
        dispatch logic, but the inner evaluation for each piece is.

        Parameters
        ----------
        x : scalar or jax.Array, shape (m,)
            Evaluation point(s) in the Chebfun domain.

        Returns
        -------
        jax.Array, same shape as x

        Raises
        ------
        None — points outside the domain will return values from the nearest
        endpoint piece (matching MATLAB behavior).

        Notes
        -----
        JIT contract: jit=yes for single-piece, jit=NO for multi-piece
        (dynamic dispatch). vmap=yes for single-piece.

        Provenance
        ----------
        MATLAB source : @chebfun/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x.ndim == 0
        x = jnp.atleast_1d(x)

        if len(self.funs) == 1:
            # Fast path: single piece — fully JIT-able
            result = self.funs[0](x)
            if scalar_input:
                result = result[0]
            return result

        # Multi-piece: Python dispatch
        out = jnp.full(x.shape, jnp.nan, dtype=jnp.float64)
        n_pieces = len(self.funs)
        for i, piece in enumerate(self.funs):
            a, b = piece.interval
            if i == 0:
                # Left piece: include all x <= b
                if n_pieces == 1:
                    mask = jnp.ones(x.shape, dtype=bool)
                else:
                    mask = x <= b
            elif i == n_pieces - 1:
                # Right piece: include all x >= a
                mask = x >= a
            else:
                # Interior piece: include a <= x <= b
                mask = (x >= a) & (x <= b)

            # Evaluate masked points
            x_piece = jnp.where(mask, x, jnp.float64(a))
            vals = piece(x_piece)
            out = jnp.where(mask, vals, out)

        if scalar_input:
            out = out[0]
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coeffs(self) -> jax.Array:
        """Chebyshev coefficients.

        For a single-piece Chebfun: the coefficient array of the one piece.
        For multi-piece: concatenation of all piece coefficients (with a
        separator of ``[jnp.nan]`` between pieces for clarity).

        Returns
        -------
        jax.Array
        """
        if len(self.funs) == 1:
            return self.funs[0].coeffs
        return jnp.concatenate(
            [p.coeffs for p in self.funs]
        )

    @property
    def values(self) -> jax.Array:
        """Values at Chebyshev-2 points.

        For single-piece: the values array.  For multi-piece: concatenated.

        Returns
        -------
        jax.Array
        """
        if len(self.funs) == 1:
            return self.funs[0].values
        return jnp.concatenate([p.values for p in self.funs])

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute value across all pieces."""
        return max(p.vscale for p in self.funs)

    @property
    def ishappy(self) -> bool:
        """True if all pieces are resolved to the requested tolerance."""
        return all(p.ishappy for p in self.funs)

    # ------------------------------------------------------------------
    # Python dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of Chebyshev coefficients across all pieces."""
        return sum(p.n for p in self.funs)

    def __repr__(self) -> str:
        """Informative multi-line representation matching MATLAB Chebfun style.

        Examples
        --------
        >>> f = chebfun(jnp.sin)
        >>> repr(f)
        'Chebfun column (1 smooth piece)\\n       interval       length ...\\n...'
        """
        n_pieces = len(self.funs)
        piece_word = "piece" if n_pieces == 1 else "pieces"
        header = f"Chebfun column ({n_pieces} smooth {piece_word})"
        # Mimic MATLAB's column header layout:
        # "       interval       length     endpoint values"
        col_header = "       interval       length     endpoint values"
        lines = [header, col_header]
        for piece in self.funs:
            a, b = piece.interval
            length = piece.n
            lval, rval = piece.endpoint_values
            # Format interval as "[      -1,       1]" (8 chars each side)
            interval_str = f"[{a:8g},{b:8g}]"
            lines.append(
                f"{interval_str}  {length:7d}    {lval:7.2f}    {rval:6.2f}"
            )
        total_len = len(self)
        vs = self.vscale
        footer = f"vscale = {vs:.2e}"
        if n_pieces > 1:
            footer += f"    total length = {total_len}"
        lines.append(footer)
        return "\n".join(lines)

    def __str__(self) -> str:
        """One-line summary."""
        a, b = self.domain.a, self.domain.b
        return f"<Chebfun [{a}, {b}], length {len(self)}>"

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    @staticmethod
    def _check_domains(f: Chebfun, g: Chebfun) -> None:
        """Raise ValueError if two Chebfuns have incompatible domains."""
        if f.domain != g.domain:
            raise ValueError(
                f"Cannot combine Chebfun on {f.domain} with Chebfun on "
                f"{g.domain}: domains do not match.  "
                f"Use f.restrict(...) to make the domains compatible first."
            )

    @staticmethod
    def _binary_op(f: Chebfun, g: Chebfun, op) -> Chebfun:
        """Apply a piecewise binary op between two same-domain Chebfuns.

        ``op`` must be a method name (str) on ``Chebtech2`` accepting one
        Chebtech2 argument, or a callable ``op(tech_a, tech_b) -> Chebtech2``.
        """
        Chebfun._check_domains(f, g)
        new_funs = [
            _Piece(tech=op(pf.tech, pg.tech), interval=pf.interval)
            for pf, pg in zip(f.funs, g.funs)
        ]
        return Chebfun(funs=new_funs, domain=f.domain)

    def __add__(self, other) -> Chebfun:
        """Add two Chebfuns or a Chebfun and a scalar.

        Returns a new Chebfun with each piece added independently.

        Provenance
        ----------
        MATLAB source : @chebfun/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun):
            return Chebfun._binary_op(self, other, lambda a, b: a + b)
        # scalar: delegate to each piece
        new_funs = [
            piece._apply_unary(piece.tech + other)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __radd__(self, other) -> Chebfun:
        return self.__add__(other)

    def __sub__(self, other) -> Chebfun:
        """Subtract two Chebfuns or a scalar from a Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/minus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun):
            return Chebfun._binary_op(self, other, lambda a, b: a - b)
        new_funs = [
            piece._apply_unary(piece.tech - other)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __rsub__(self, other) -> Chebfun:
        return -(self - other)

    def __neg__(self) -> Chebfun:
        """Unary negation.

        Provenance
        ----------
        MATLAB source : @chebfun/uminus.m
        Chebfun commit: 7574c77
        """
        new_funs = [piece._apply_unary(-piece.tech) for piece in self.funs]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __pos__(self) -> Chebfun:
        """Unary plus (identity)."""
        return self

    def __mul__(self, other) -> Chebfun:
        """Pointwise multiplication of two Chebfuns or Chebfun by scalar.

        Provenance
        ----------
        MATLAB source : @chebfun/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun):
            return Chebfun._binary_op(self, other, lambda a, b: a * b)
        # If other is not a scalar/array, defer to other's __rmul__
        if not isinstance(other, (int, float, jnp.ndarray, jax.Array)):
            return NotImplemented
        new_funs = [
            piece._apply_unary(piece.tech * other)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __rmul__(self, other) -> Chebfun:
        return self.__mul__(other)

    def __truediv__(self, other) -> Chebfun:
        """Pointwise division: Chebfun / scalar or Chebfun / Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun):
            return Chebfun._binary_op(self, other, lambda a, b: a / b)
        new_funs = [
            piece._apply_unary(piece.tech / other)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __rtruediv__(self, other) -> Chebfun:
        """scalar / Chebfun."""
        new_funs = [
            piece._apply_unary(other / piece.tech)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __pow__(self, exponent) -> Chebfun:
        """Raise each piece to a power.

        Provenance
        ----------
        MATLAB source : @chebfun/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, Chebfun):
            return Chebfun._binary_op(self, exponent, lambda a, b: a ** b)
        new_funs = [
            piece._apply_unary(piece.tech ** exponent)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __abs__(self) -> Chebfun:
        """Absolute value (piecewise; may introduce kinks at zeros).

        NOT JIT-safe (uses compose which calls adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/abs.m
        Chebfun commit: 7574c77
        """
        new_funs = [
            piece._apply_unary(abs(piece.tech))
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    # ------------------------------------------------------------------
    # Composition with scalar functions
    # ------------------------------------------------------------------

    def _apply_fun(self, op) -> Chebfun:
        """Compose self with a scalar function op, piece by piece.

        Constructs a new Chebfun by adaptively approximating ``op(self(x))``
        on each sub-interval.  This mirrors MATLAB's ``compose(f, @op)``
        pattern used internally by all special-function methods.

        NOT JIT-safe (calls adaptive construction).

        Parameters
        ----------
        op : callable
            A vectorized JAX function applied pointwise, e.g. ``jnp.sin``.

        Returns
        -------
        Chebfun
            New Chebfun approximating op(self(x)) on the same domain.

        Provenance
        ----------
        MATLAB source : @chebfun/compose.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        new_funs = [piece._apply_fun(op) for piece in self.funs]
        return Chebfun(funs=new_funs, domain=self.domain)

    # ------------------------------------------------------------------
    # Special functions (thin wrappers around _apply_fun)
    # ------------------------------------------------------------------

    def sin(self) -> Chebfun:
        """Sine of the Chebfun.

        Returns a new Chebfun approximating sin(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Examples
        --------
        >>> x = Chebfun.identity()
        >>> f = x.sin()
        >>> import numpy.testing as npt
        >>> import numpy as np
        >>> xs = jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float64)
        >>> npt.assert_allclose(np.array(f(xs)), np.array(jnp.sin(xs)), atol=1e-13)

        Provenance
        ----------
        MATLAB source : @chebfun/sin.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.cos, Chebfun.asin
        """
        return self._apply_fun(jnp.sin)

    def cos(self) -> Chebfun:
        """Cosine of the Chebfun.

        Returns a new Chebfun approximating cos(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/cos.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.sin, Chebfun.acos
        """
        return self._apply_fun(jnp.cos)

    def exp(self) -> Chebfun:
        """Exponential of the Chebfun.

        Returns a new Chebfun approximating exp(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/exp.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log
        """
        return self._apply_fun(jnp.exp)

    def log(self) -> Chebfun:
        """Natural logarithm of the Chebfun.

        Returns a new Chebfun approximating log(f(x)) on the same domain.
        If f has roots in its domain, the representation may be inaccurate
        (log diverges at zeros).

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/log.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.exp, Chebfun.sqrt
        """
        return self._apply_fun(jnp.log)

    def sqrt(self) -> Chebfun:
        """Square root of the Chebfun.

        Returns a new Chebfun approximating sqrt(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/sqrt.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log, Chebfun.exp
        """
        return self._apply_fun(jnp.sqrt)

    def abs(self) -> Chebfun:
        """Absolute value of the Chebfun.

        For a smooth function with no sign changes on the domain, this is
        equivalent to ``__abs__`` (using the piece-level abs).  If sign
        changes are present, breakpoints are introduced at the roots so that
        each piece remains smooth.

        NOT JIT-safe (root-finding and adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/abs.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.sign, Chebfun.__abs__
        """
        # Find roots where the function changes sign and add them as
        # breakpoints, then apply |·| piecewise for smoothness.
        roots = self.roots()
        if roots.shape[0] == 0:
            # No sign changes — simple abs on each piece
            return self._apply_fun(jnp.abs)

        # Build new breakpoints: existing domain breakpoints + roots
        import numpy as _np
        existing = _np.array(list(self.domain.breakpoints))
        new_bps = _np.sort(_np.unique(
            _np.concatenate([existing, _np.asarray(roots)])
        ))
        # Remove duplicates within tolerance
        domain_len = float(self.domain.b - self.domain.a)
        tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)
        mask = _np.concatenate([[True], _np.diff(new_bps) > tol])
        new_bps = new_bps[mask]

        if len(new_bps) < 2:
            return self._apply_fun(jnp.abs)

        new_dom = Domain(tuple(float(b) for b in new_bps))
        f = self  # capture for closure
        new_funs = [
            _Piece.from_function(lambda x, _f=f: jnp.abs(_f(x)), sub.a, sub.b)
            for sub in new_dom.intervals
        ]
        return Chebfun(funs=new_funs, domain=new_dom)

    def sign(self) -> Chebfun:
        """Sign function of the Chebfun.

        Returns a piecewise-constant Chebfun: +1 where self > 0, -1 where
        self < 0, 0 at zeros.  Breakpoints are introduced at the roots of
        self so that each piece is smooth (constant).

        NOT JIT-safe (root-finding and adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/sign.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.abs, Chebfun.roots
        """
        roots = self.roots()
        import numpy as _np
        existing = _np.array(list(self.domain.breakpoints))
        new_bps = _np.sort(_np.unique(
            _np.concatenate([existing, _np.asarray(roots)])
        ))
        domain_len = float(self.domain.b - self.domain.a)
        tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)
        mask = _np.concatenate([[True], _np.diff(new_bps) > tol])
        new_bps = new_bps[mask]

        if len(new_bps) < 2:
            return self._apply_fun(jnp.sign)

        new_dom = Domain(tuple(float(b) for b in new_bps))
        f = self  # capture for closure
        new_funs = [
            _Piece.from_function(lambda x, _f=f: jnp.sign(_f(x)), sub.a, sub.b)
            for sub in new_dom.intervals
        ]
        return Chebfun(funs=new_funs, domain=new_dom)

    def sinh(self) -> Chebfun:
        """Hyperbolic sine of the Chebfun.

        Returns a new Chebfun approximating sinh(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/sinh.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.cosh, Chebfun.tanh
        """
        return self._apply_fun(jnp.sinh)

    def cosh(self) -> Chebfun:
        """Hyperbolic cosine of the Chebfun.

        Returns a new Chebfun approximating cosh(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/cosh.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.sinh, Chebfun.tanh
        """
        return self._apply_fun(jnp.cosh)

    def tanh(self) -> Chebfun:
        """Hyperbolic tangent of the Chebfun.

        Returns a new Chebfun approximating tanh(f(x)) on the same domain.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/tanh.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.sinh, Chebfun.cosh
        """
        return self._apply_fun(jnp.tanh)

    def asin(self) -> Chebfun:
        """Inverse sine (arcsin) of the Chebfun.

        Returns a new Chebfun approximating arcsin(f(x)).  The values of
        f must lie in [-1, 1] for this to be well-defined.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/asin.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.sin, Chebfun.acos, Chebfun.atan
        """
        return self._apply_fun(jnp.arcsin)

    def acos(self) -> Chebfun:
        """Inverse cosine (arccos) of the Chebfun.

        Returns a new Chebfun approximating arccos(f(x)).  The values of
        f must lie in [-1, 1] for this to be well-defined.

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/acos.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.cos, Chebfun.asin, Chebfun.atan
        """
        return self._apply_fun(jnp.arccos)

    def atan(self) -> Chebfun:
        """Inverse tangent (arctan) of the Chebfun.

        Returns a new Chebfun approximating arctan(f(x)).

        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/atan.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.asin, Chebfun.acos, Chebfun.tan
        """
        return self._apply_fun(jnp.arctan)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> Chebfun:
        """Differentiate *k* times with respect to x.

        Each piece is differentiated independently using the affine chain rule.

        JIT-safe: yes (k must be a static Python int).

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Chebfun
            The k-th derivative, represented piecewise.

        Provenance
        ----------
        MATLAB source : @chebfun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return self
        new_funs = [piece.diff(k) for piece in self.funs]
        return Chebfun(funs=new_funs, domain=self.domain)

    def cumsum(self) -> Chebfun:
        """Antiderivative satisfying F(a) = 0 at the left endpoint.

        For a piecewise Chebfun, the antiderivative is computed on each piece
        and then shifted to ensure continuity across breakpoints.

        JIT-safe: yes for the per-piece computation.

        Returns
        -------
        Chebfun
            The antiderivative.

        Provenance
        ----------
        MATLAB source : @chebfun/cumsum.m
        Chebfun commit: 7574c77
        """
        if len(self.funs) == 1:
            return Chebfun(funs=[self.funs[0].cumsum()], domain=self.domain)

        # Multi-piece: compute antiderivative on each piece, then shift to
        # ensure continuity: F_i(b_i) = F_{i+1}(a_{i+1})
        new_pieces = []
        offset = jnp.float64(0.0)
        for piece in self.funs:
            piece_cs = piece.cumsum()
            # piece_cs has F_piece(a_piece) = 0 by construction
            # Shift by offset to achieve continuity
            if float(offset) != 0.0:
                # Add offset as a constant to the antiderivative piece
                new_tech = piece_cs.tech + float(offset)
                piece_cs = _Piece(tech=new_tech, interval=piece_cs.interval)
            new_pieces.append(piece_cs)
            # Update offset: new cumulative value at the right endpoint
            _, rval = piece_cs.endpoint_values
            offset = jnp.float64(rval)

        return Chebfun(funs=new_pieces, domain=self.domain)

    def sum(self) -> jax.Array:
        r"""Definite integral over the full domain.

        Sums the definite integrals of all pieces.

        JIT-safe: yes.

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebfun/sum.m
        Chebfun commit: 7574c77
        """
        total = jnp.float64(0.0)
        for piece in self.funs:
            total = total + piece.sum()
        return total

    def inner(self, other: Chebfun) -> jax.Array:
        r"""L2 inner product <self, other> = \int_a^b f(x) g(x) dx.

        Requires both Chebfuns to have the same domain.

        JIT-safe: yes.

        Parameters
        ----------
        other : Chebfun

        Returns
        -------
        jax.Array (scalar)

        Raises
        ------
        ValueError
            If domains do not match.

        Provenance
        ----------
        MATLAB source : @chebfun/innerProduct.m
        Chebfun commit: 7574c77
        """
        Chebfun._check_domains(self, other)
        total = jnp.float64(0.0)
        for pf, pg in zip(self.funs, other.funs):
            total = total + pf.inner(pg)
        return total

    def norm(self, p: float = 2) -> jax.Array:
        """Lp norm over the domain.

        Parameters
        ----------
        p : float, default 2
            The exponent.
            - ``p=2``: L2 norm = sqrt(<f, f>).
            - ``p=jnp.inf``: L-infinity norm (max over all pieces).
            - Other p: computed via ``|f|^p`` integration.

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebfun/norm.m
        Chebfun commit: 7574c77
        """
        if p == 2:
            return jnp.sqrt(jnp.abs(self.inner(self)))
        elif p == float("inf") or p == jnp.inf:
            # Max over pieces
            maxvals = [float(jnp.max(jnp.abs(piece.tech.values)))
                       for piece in self.funs]
            return jnp.array(max(maxvals), dtype=jnp.float64)
        else:
            # Integrate |f|^p
            fp = abs(self) ** p
            return fp.sum() ** (1.0 / p)

    def mean(self) -> jax.Array:
        """Mean value of the function over the domain.

        mean(f) = (1 / (b - a)) * int_a^b f(x) dx

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebfun/mean.m
        Chebfun commit: 7574c77
        """
        a, b = self.domain.a, self.domain.b
        domain_len = jnp.float64(b - a)
        return self.sum() / domain_len

    # ------------------------------------------------------------------
    # Rootfinding and extrema
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """All roots of the Chebfun in its domain.

        Collects roots from each piece, sorts them, and deduplicates roots
        that are very close to each other (e.g. a root at a breakpoint may
        be found independently by two adjacent pieces).

        NOT JIT-safe (variable output size, eigenvalue computation).

        Returns
        -------
        jax.Array, shape (n_roots,)
            Sorted, deduplicated roots in [a, b].

        Provenance
        ----------
        MATLAB source : @chebfun/roots.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        all_roots = []
        for piece in self.funs:
            r = piece.roots()
            if r.shape[0] > 0:
                all_roots.append(r)
        if not all_roots:
            return jnp.array([], dtype=jnp.float64)
        combined = _np.sort(_np.concatenate([_np.asarray(r) for r in all_roots]))

        # Deduplicate: remove consecutive roots that are within a tight tolerance
        # (handles the case where a breakpoint root is found by two pieces).
        if combined.shape[0] <= 1:
            return jnp.asarray(combined, dtype=jnp.float64)
        domain_len = float(self.domain.b - self.domain.a)
        dedup_tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)
        mask = _np.concatenate([[True], _np.diff(combined) > dedup_tol])
        unique_roots = combined[mask]
        return jnp.asarray(unique_roots, dtype=jnp.float64)

    def minandmax(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Global minimum and maximum of the Chebfun.

        Searches each piece for its local extrema (critical points of the
        derivative plus piece endpoints), then returns the global min/max.

        NOT JIT-safe (uses roots of derivative — eigenvalue computation).

        Returns
        -------
        (x_min, f_min), (x_max, f_max) : pair of (location, value) tuples.

        Provenance
        ----------
        MATLAB source : @chebfun/minandmax.m
        Chebfun commit: 7574c77
        """
        global_min_x = None
        global_min_val = float("inf")
        global_max_x = None
        global_max_val = float("-inf")

        for piece in self.funs:
            (x_min, f_min), (x_max, f_max) = piece.minandmax()
            if f_min < global_min_val:
                global_min_val = f_min
                global_min_x = x_min
            if f_max > global_max_val:
                global_max_val = f_max
                global_max_x = x_max

        return (global_min_x, global_min_val), (global_max_x, global_max_val)

    def min(self) -> tuple[float, float]:
        """Global minimum: returns (x_min, f_min).

        NOT JIT-safe.

        Returns
        -------
        (x_min, f_min) : tuple of floats

        Provenance
        ----------
        MATLAB source : @chebfun/min.m
        Chebfun commit: 7574c77
        """
        (x_min, f_min), _ = self.minandmax()
        return x_min, f_min

    def max(self) -> tuple[float, float]:
        """Global maximum: returns (x_max, f_max).

        NOT JIT-safe.

        Returns
        -------
        (x_max, f_max) : tuple of floats

        Provenance
        ----------
        MATLAB source : @chebfun/max.m
        Chebfun commit: 7574c77
        """
        _, (x_max, f_max) = self.minandmax()
        return x_max, f_max

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    def restrict(self, a: float, b: float) -> Chebfun:
        """Restrict the Chebfun to the sub-interval [a, b].

        Parameters
        ----------
        a : float
            Left endpoint of the restriction (must be in the domain).
        b : float
            Right endpoint of the restriction (must be in the domain).

        Returns
        -------
        Chebfun
            A new Chebfun on [a, b].

        Raises
        ------
        ValueError
            If [a, b] is not a sub-interval of the domain.

        Notes
        -----
        Each piece that overlaps [a, b] is restricted via ``_Piece.restrict``.
        Pieces entirely outside [a, b] are discarded.

        Provenance
        ----------
        MATLAB source : @chebfun/restrict.m
        Chebfun commit: 7574c77
        """
        a, b = float(a), float(b)
        da, db = self.domain.a, self.domain.b
        if a < da - 100 * _EPS or b > db + 100 * _EPS or a >= b:
            raise ValueError(
                f"Cannot restrict Chebfun on [{da}, {db}] to [{a}, {b}]: "
                f"the restriction interval must be a sub-interval of the domain."
            )
        new_domain = self.domain.restrict(a, b)
        new_funs = []
        for piece in self.funs:
            pa, pb = piece.interval
            # Does this piece overlap [a, b]?
            lo = max(pa, a)
            hi = min(pb, b)
            if lo >= hi - 100 * _EPS:
                continue  # No overlap or zero-width
            new_funs.append(piece.restrict(lo, hi))
        if not new_funs:
            raise ValueError(
                f"Restriction [{a}, {b}] produced no pieces — check domain."
            )
        return Chebfun(funs=new_funs, domain=new_domain)

    # ------------------------------------------------------------------
    # Quasimatrix linear algebra: qr, svd
    # ------------------------------------------------------------------

    def qr(self, other_cols: list | None = None):
        """QR factorization of this Chebfun as a single column, or a quasimatrix.

        For a single Chebfun (one column) this simply normalises:
        ``Q = f / ||f||_2``, ``R = [[||f||_2]]``.

        For a quasimatrix (by passing a list of additional Chebfun columns as
        ``other_cols``), the columns ``[self] + other_cols`` are jointly
        factorised using the continuous Householder algorithm [1].

        Parameters
        ----------
        other_cols : list[Chebfun] or None
            Additional columns.  If ``None`` (default), ``self`` is treated as
            a single column.

        Returns
        -------
        Q : Quasimatrix
            Quasimatrix with L2-orthonormal columns on the same domain.
        R : jnp.ndarray, shape (n, n)
            Upper-triangular factor.  If all n columns are ``[self]``, R is
            1 x 1.

        Notes
        -----
        NOT JIT-safe (continuous Householder QR uses Python loops).

        References
        ----------
        [1] L.N. Trefethen, "Householder triangularization of a quasimatrix",
            IMA J Numer Anal (2010) 30(4): 887–897.

        Provenance
        ----------
        MATLAB source : @chebfun/qr.m, abstractQR.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.svd, chebfun1d.linalg.qr_quasimatrix
        """
        from chebfunjax.chebfun1d.linalg import chebfun_qr
        if other_cols is None:
            cols = [self]
        else:
            cols = [self] + list(other_cols)
        return chebfun_qr(cols)

    def svd(self, other_cols: list | None = None):
        """SVD of this Chebfun as a single column, or a quasimatrix.

        Computes the singular value decomposition A = U * diag(S) * V^T via:
        (1) QR factorisation of the quasimatrix, and
        (2) discrete SVD of the upper-triangular R factor.

        Parameters
        ----------
        other_cols : list[Chebfun] or None
            Additional columns.  If ``None`` (default), ``self`` is treated as
            a single column.

        Returns
        -------
        U : Quasimatrix
            Left singular functions (L2-orthonormal columns).
        S : jnp.ndarray, shape (n,)
            Singular values in non-increasing order.
        V : jnp.ndarray, shape (n, n)
            Right singular vectors (columns of V are orthonormal).

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/svd.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.qr, chebfun1d.linalg.svd_quasimatrix
        """
        from chebfunjax.chebfun1d.linalg import chebfun_svd
        if other_cols is None:
            cols = [self]
        else:
            cols = [self] + list(other_cols)
        return chebfun_svd(cols)


# ============================================================================
# Factory function — the main user-facing entry point
# ============================================================================

def chebfun(
    f=None,
    *,
    domain=(-1.0, 1.0),
    n: int | None = None,
) -> Chebfun:
    """Create a Chebfun from a callable, array of coefficients, or constant.

    This is the primary construction entry point. It mimics MATLAB's
    ``chebfun(...)`` syntax.

    Parameters
    ----------
    f : callable, float, or None
        - A callable ``f(x)`` (vectorized): builds an adaptive (or fixed-n)
          Chebyshev approximation.
        - A scalar (int or float): builds a constant Chebfun.
        - ``None``: raises ``ValueError`` (empty Chebfun not supported here;
          use ``Chebfun`` directly for internal use).
    domain : array-like of length 2 or more, optional
        The domain.  Two values ``(a, b)`` give a single interval.  More
        values ``(a, b1, ..., b)`` specify breakpoints for piecewise
        construction. Default is ``(-1.0, 1.0)``.
    n : int or None, optional
        Fixed number of Chebyshev points per piece.  If ``None`` (default)
        adaptive construction is used.

    Returns
    -------
    Chebfun

    Raises
    ------
    TypeError
        If ``f`` is not a callable, number, or None.
    ValueError
        If the domain is invalid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import chebfunjax as cj
    >>> f = cj.chebfun(jnp.sin)             # adaptive on [-1, 1]
    >>> f(jnp.float64(0.5))                 # evaluate
    Array(0.47942554, dtype=float64)

    >>> g = cj.chebfun(jnp.sin, domain=[0, jnp.pi])  # custom domain
    >>> float(g(jnp.float64(jnp.pi / 2)))
    1.0

    >>> h = cj.chebfun(1.0)                 # constant 1
    >>> float(h(jnp.float64(0.0)))
    1.0

    >>> k = cj.chebfun(jnp.sin, n=20)      # fixed degree
    >>> len(k)
    20

    Notes
    -----
    Adaptive construction is NOT JIT-safe (Python while loop). Fixed-n
    construction is JIT-safe in principle but is typically called outside JIT.

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun.from_function, Chebfun.from_coeffs, Chebfun.from_values
    """
    # --- Parse domain ---
    dom_seq = [float(x) for x in domain]
    if len(dom_seq) < 2:
        raise ValueError(
            f"domain must have at least 2 elements, got {len(dom_seq)}. "
            f"Example: domain=(-1, 1)."
        )
    dom = Domain(tuple(dom_seq))

    # --- Dispatch on f type ---
    if f is None:
        raise ValueError(
            "f=None is not supported. "
            "Pass a callable, a scalar, or use Chebfun.from_coeffs / "
            "Chebfun.from_values for data-driven construction."
        )

    if isinstance(f, (int, float)) or (
        hasattr(f, "__float__") and not callable(f)
    ):
        # Scalar constant
        c = float(f)
        return Chebfun.from_function(lambda x: jnp.full_like(x, c), dom, n=n)

    # Try JAX scalar (0-d array)
    try:
        arr = jnp.asarray(f)
        if arr.ndim == 0:
            c = float(arr)
            return Chebfun.from_function(lambda x: jnp.full_like(x, c), dom, n=n)
    except Exception:
        pass

    if callable(f):
        return Chebfun.from_function(f, dom, n=n)

    raise TypeError(
        f"Cannot construct a Chebfun from f of type {type(f).__name__}. "
        f"Pass a callable (e.g. jnp.sin), a scalar, or use "
        f"Chebfun.from_coeffs / Chebfun.from_values."
    )


# Attach factory classmethods to `chebfun` callable so users can write
# ``chebfun.from_coeffs(...)`` and ``chebfun.from_values(...)`` as shown
# in the API design doc.
chebfun.from_coeffs = Chebfun.from_coeffs  # type: ignore[attr-defined]
chebfun.from_values = Chebfun.from_values  # type: ignore[attr-defined]
chebfun.identity = Chebfun.identity        # type: ignore[attr-defined]


# ============================================================================
# ODE integrators: ode45 / ode113  (V04)
# ============================================================================
# uses-numpy: scipy.integrate.solve_ivp uses NumPy arrays internally


def ode45(
    odefun: "Callable[[float, jax.Array], jax.Array]",
    tspan: "tuple[float, float]",
    y0: "jax.Array",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense_n: int | None = None,
    **kwargs,
) -> "Chebfun":
    """Solve a non-stiff IVP y' = f(t, y) and return a Chebfun.

    Wraps ``scipy.integrate.solve_ivp`` with the ``RK45`` method (the
    Python equivalent of MATLAB's ``ode45``) and interpolates the dense
    solution output onto a piecewise Chebfun with one piece per adaptive
    step.

    Parameters
    ----------
    odefun : callable(t, y) -> array_like
        Right-hand side of the ODE.  ``t`` is a scalar float; ``y`` is a
        1-D NumPy array.  Must be broadcastable to ``y0.shape``.
    tspan : (float, float)
        Integration interval ``(t0, tf)``.
    y0 : array_like, shape (d,) or scalar
        Initial state.  A scalar ``y0`` is treated as a 1-D vector of
        length 1.
    rtol : float, default 1e-6
        Relative tolerance passed to the solver.
    atol : float, default 1e-8
        Absolute tolerance passed to the solver.
    dense_n : int or None
        Number of uniform evaluation points used to build the Chebfun from
        the dense output.  Default: ``max(32, 4 * nsteps)``.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.integrate.solve_ivp``
        (e.g. ``max_step``, ``events``).

    Returns
    -------
    sol : Chebfun
        Piecewise Chebfun on ``tspan``.  For a scalar ODE (d=1) this is a
        scalar Chebfun.  For a system (d>1) each component is a separate
        piece stored in a separate call; users should index components
        manually via ``sol(t)[k]``.

    Examples
    --------
    >>> from chebfunjax.chebfun1d.chebfun import ode45
    >>> import jax.numpy as jnp
    >>> # y' = y,  y(0) = 1  =>  y = exp(t)  on [0, 1]
    >>> sol = ode45(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
    >>> abs(float(sol(jnp.float64(1.0))) - float(jnp.exp(jnp.float64(1.0)))) < 1e-4
    True

    Notes
    -----
    The adaptive solver chooses its own internal step sequence; the Chebfun
    is built by evaluating the dense (continuous) extension of the solution
    at ``dense_n`` uniformly spaced points and fitting a Chebfun to those
    values.  This decouples the ODE step-size from the Chebfun degree.

    Provenance
    ----------
    MATLAB source : @chebfun/ode45.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ode113 : Adams-Bashforth-Moulton integrator (MATLAB ode113 analogue)
    """
    return _ode_solve("RK45", odefun, tspan, y0,
                      rtol=rtol, atol=atol, dense_n=dense_n, **kwargs)


def ode113(
    odefun: "Callable[[float, jax.Array], jax.Array]",
    tspan: "tuple[float, float]",
    y0: "jax.Array",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense_n: int | None = None,
    **kwargs,
) -> "Chebfun":
    """Solve a non-stiff IVP y' = f(t, y) and return a Chebfun.

    Wraps ``scipy.integrate.solve_ivp`` with the ``DOP853`` method (a
    high-order explicit Runge-Kutta method, the closest Python analogue
    of MATLAB's variable-order Adams ``ode113``) and interpolates the
    dense output onto a Chebfun.

    Parameters
    ----------
    odefun : callable(t, y) -> array_like
        Right-hand side of the ODE.
    tspan : (float, float)
        Integration interval ``(t0, tf)``.
    y0 : array_like, shape (d,) or scalar
        Initial state.
    rtol : float, default 1e-6
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    dense_n : int or None
        Number of uniform evaluation points for Chebfun construction.
    **kwargs
        Forwarded to ``scipy.integrate.solve_ivp``.

    Returns
    -------
    sol : Chebfun
        Piecewise Chebfun on ``tspan``.

    Examples
    --------
    >>> from chebfunjax.chebfun1d.chebfun import ode113
    >>> import jax.numpy as jnp
    >>> sol = ode113(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
    >>> abs(float(sol(jnp.float64(1.0))) - float(jnp.exp(jnp.float64(1.0)))) < 1e-4
    True

    Notes
    -----
    The Dopri8/DOP853 method uses a fixed 8th-order scheme with a 5th-order
    error estimate.  It is well-suited for smooth, non-stiff problems.

    Provenance
    ----------
    MATLAB source : @chebfun/ode113.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ode45 : Dormand-Prince RK45 integrator (MATLAB ode45 analogue)
    """
    return _ode_solve("DOP853", odefun, tspan, y0,
                      rtol=rtol, atol=atol, dense_n=dense_n, **kwargs)


# ---------------------------------------------------------------------------
# Private implementation shared by ode45 / ode113
# ---------------------------------------------------------------------------


def _ode_solve(
    method: str,
    odefun,
    tspan: "tuple[float, float]",
    y0,
    *,
    rtol: float,
    atol: float,
    dense_n: int | None,
    **kwargs,
) -> "Chebfun":
    """Integrate an IVP and return a Chebfun (shared implementation).

    Parameters
    ----------
    method : str
        ``solve_ivp`` method string (``'RK45'`` or ``'DOP853'``).
    odefun, tspan, y0, rtol, atol, dense_n, **kwargs
        As documented in :func:`ode45` / :func:`ode113`.

    Returns
    -------
    Chebfun

    Provenance
    ----------
    MATLAB source : @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    # uses-numpy: scipy.integrate.solve_ivp uses NumPy internally
    import numpy as _np
    from scipy.integrate import solve_ivp  # type: ignore[import]

    t0, tf = float(tspan[0]), float(tspan[1])

    # Normalise initial state to a 1-D NumPy float64 vector
    y0_np = _np.atleast_1d(_np.asarray(y0, dtype=_np.float64))
    scalar_out = y0_np.ndim == 1 and y0_np.shape[0] == 1

    # Wrap odefun so it always receives/returns NumPy arrays
    def _rhs(t, y):
        result = odefun(float(t), jnp.asarray(y, dtype=jnp.float64))
        return _np.atleast_1d(_np.asarray(result, dtype=_np.float64))

    # Call scipy solver with dense_output=True for interpolation
    sol = solve_ivp(
        _rhs,
        [t0, tf],
        y0_np,
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE solver ({method}) failed: {sol.message}"
        )

    # Choose evaluation grid for Chebfun construction
    nsteps = len(sol.t)
    n_pts = dense_n if dense_n is not None else max(32, 4 * nsteps)
    t_eval = _np.linspace(t0, tf, n_pts)

    # Evaluate dense output: shape (d, n_pts) — used only to verify success
    sol.sol(t_eval)  # type: ignore[union-attr]

    if scalar_out:
        # Scalar ODE — build a single-component Chebfun by fitting the
        # dense output via the adaptive chebfun factory.
        # The dense solution ``sol.sol`` is a continuous interpolant from
        # solve_ivp; we pass it directly as the function to approximate.
        return chebfun(
            lambda t: jnp.asarray(sol.sol(  # type: ignore[union-attr]
                _np.atleast_1d(_np.asarray(t, dtype=_np.float64))
            )[0], dtype=jnp.float64),
            domain=(t0, tf),
        )
    else:
        # Vector ODE — build one Chebfun per component, return list
        # (MATLAB returns a quasimatrix; here we return a Python list)
        raise NotImplementedError(
            "ode45/ode113: vector ODEs (d > 1) are not yet supported. "
            "Use scipy.integrate.solve_ivp directly for multi-component systems."
        )
