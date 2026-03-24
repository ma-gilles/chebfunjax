"""User-facing Chebfun class for piecewise smooth function approximation.

This is the main class users interact with. A Chebfun on a domain [a, b] is
represented as a list of *pieces* (Chebtech2 objects), each defined on a
sub-interval, together with a Domain recording the breakpoints.

For U40 (construction + evaluation only), single-piece Chebfuns on
arbitrary intervals [a, b] are supported directly via Chebtech2 with an
affine change of variables. Multi-piece (piecewise) Chebfuns are structured
but only single-piece construction is exposed through the main factory.

Translated from MATLAB Chebfun class @chebfun (commit 7574c77) and informed
by chebpy's Chebfun class.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
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
