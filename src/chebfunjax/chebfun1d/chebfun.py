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

import math
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)


# One-sided-evaluation recording: while a list is installed here, every
# ``Chebfun(x, side)`` call (and hence every ``jump``) appends its evaluation
# point ``x``, so a chebop can discover the interior breakpoints an interior
# jump / one-sided boundary condition refers to before solving.
_SIDE_EVAL_RECORD: "list | None" = None


def _record_side_eval(x) -> None:
    if _SIDE_EVAL_RECORD is not None:
        try:
            _SIDE_EVAL_RECORD.append(float(jnp.asarray(x).reshape(())))
        except Exception:
            pass


def start_side_eval_record() -> None:
    """Begin recording the points passed to one-sided evaluations."""
    global _SIDE_EVAL_RECORD
    _SIDE_EVAL_RECORD = []


def stop_side_eval_record() -> "list":
    """Stop recording and return the collected one-sided evaluation points."""
    global _SIDE_EVAL_RECORD
    rec = _SIDE_EVAL_RECORD or []
    _SIDE_EVAL_RECORD = None
    return rec


def jump(f: "Chebfun", x, c: float = 0.0):
    """The jump in ``f`` across the breakpoint ``x`` (MATLAB ``jump``).

    ``jump(f, x, c) = f(x, 'right') - f(x, 'left') - c``; with two arguments
    ``c`` defaults to zero.  For a smooth ``f`` the result is zero.

    Provenance
    ----------
    MATLAB source : @chebfun/jump.m
    Chebfun commit: 7574c77
    """
    return f(x, "right") - f(x, "left") - c


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
        maxpow2: int = 16,
        tol: float | None = None,
        turbo: bool = False,
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
        maxpow2 : int, default 16
            Max adaptive grid power (``max_length = 2**maxpow2 + 1``).
        tol : float or None
            Construction tolerance (``eps``); None uses machine epsilon.
        turbo : bool, default False
            Recompute the coefficients to high accuracy via the turbo
            contour integral (MATLAB ``'turbo'`` flag).
        """
        a, b = float(a), float(b)
        # Wrap f to map from reference [-1, 1] into [a, b]
        def f_ref(t: jax.Array) -> jax.Array:
            x = 0.5 * (b - a) * t + 0.5 * (a + b)
            return f(x)

        tech = Chebtech2.from_function(f_ref, n=n, maxpow2=maxpow2, tol=tol,
                                       turbo=turbo)
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
        # Preserve a complex argument (the affine [a, b] -> [-1, 1] map and
        # the Clenshaw recurrence are both valid for complex x); everything
        # else is promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
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

    def __len__(self) -> int:
        """Polynomial length (number of Chebyshev coefficients).

        Matches MATLAB Chebfun's ``length(fun)`` so that ``len(piece)``
        works on the pieces returned by ``Chebfun.funs``.

        Added by Claude Opus 4.8 (flagged by Claude Fable 5 during the
        example-page campaign: several snippets had to use
        ``len(piece.tech.coeffs)`` because this was missing).
        """
        return int(self.tech.n)

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
        """Function values at the left and right endpoints (a, b).

        A Singfun piece has no plain ``values`` grid; its endpoint value
        is +/-Inf where the endpoint exponent is negative (sign taken from
        the smooth part, mirroring MATLAB's ``get(f, 'lval')``), 0 where
        it is positive, and the smooth-part value where it vanishes.
        """
        exps = getattr(self.tech, "exponents", None)
        if exps is not None:
            sm = self.tech.smoothPart.values
            out = []
            for exp, v in ((float(exps[0]), float(sm[0])),
                           (float(exps[1]), float(sm[-1]))):
                if exp < -1e-14:
                    out.append(math.copysign(math.inf, v if v != 0 else 1.0))
                elif exp > 1e-14:
                    out.append(0.0)
                else:
                    out.append(v)
            return (out[0], out[1])
        vals = self.values
        return (float(vals[0]), float(vals[-1]))

    def with_tech(self, tech) -> _Piece:
        """Return a new piece with the same interval but a new tech.

        Type-preserving rebuild so column operations can share one code path
        across bounded pieces and :class:`~chebfunjax.fun.unbndfun.Unbndfun`
        (which overrides this to keep its unbounded mapping).
        """
        return _Piece(tech=tech, interval=self.interval)

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
        # t_a = (2*a - (pa+pb)) / (pb-pa),  t_b similarly.  Clamp to [-1, 1]:
        # when [a, b] is (essentially) the whole piece, floating-point in the
        # affine map -- or a breakpoint merged from another operand that lands a
        # rounding step outside this piece -- can push t just past +/-1, which
        # the tech-level restrict rejects.  Restricting to marginally more than
        # the piece is the piece itself, so clamping is the correct guard.
        t_a = min(1.0, max(-1.0, (2.0 * a - (pa + pb)) / (pb - pa)))
        t_b = min(1.0, max(-1.0, (2.0 * b - (pa + pb)) / (pb - pa)))
        new_tech = self.tech.restrict(t_a, t_b)
        return _Piece(tech=new_tech, interval=(float(a), float(b)))

    # ------------------------------------------------------------------
    # Arithmetic helpers (used by Chebfun arithmetic operators)
    # ------------------------------------------------------------------

    def _apply_unary(self, tech_result: Chebtech2) -> _Piece:
        """Wrap a Chebtech2 result in a piece of the same kind.

        Goes through :meth:`with_tech` so an unbounded piece keeps its
        mapping (scalar arithmetic on an ``Unbndfun`` piece must not collapse
        it to a bounded ``_Piece`` on an infinite interval).
        """
        return self.with_tech(tech_result)

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
        # Preserve the Fourier representation: composing a periodic
        # piece yields a periodic result, and rebuilding it as a
        # Chebtech would poison later arithmetic with mixed techs.
        from chebfunjax.tech.trigtech import Trigtech
        if isinstance(self.tech, Trigtech):
            import numpy as _np
            m = max(4 * len(_np.asarray(self.tech.coeffs)), 64)
            xs = a + (b - a) * _np.arange(m) / m
            vals = op(self(jnp.asarray(xs)))
            tech = Trigtech.from_values(
                jnp.asarray(vals)).simplify()
            return _Piece(tech=tech, interval=(a, b))
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

    def abs(self) -> _Piece:
        """Absolute value of the piece (no interior sign change assumed).

        A Singfun piece keeps its exponents: the singular factors
        ``(1+x)^a (1-x)^b`` are positive on the interior, so
        ``|f| = |s| * (1+x)^a (1-x)^b``, and after root-splitting the
        smooth part has one sign — ``|s| = sign(s(0)) * s`` exactly.
        Re-approximating through ``_apply_fun`` instead silently drops
        the exponents (a pole's |f| would come back as a finite,
        unhappy interpolant).

        Provenance
        ----------
        MATLAB source : @singfun/abs.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import Singfun
        tech = self.tech
        if isinstance(tech, Singfun):
            mid = float(jnp.asarray(
                tech.smoothPart(jnp.asarray([0.0], dtype=jnp.float64)))[0])
            sgn = -1.0 if mid < 0 else 1.0
            return _Piece(tech=Singfun(tech.smoothPart * sgn,
                                       tech.exponents),
                          interval=self.interval)
        return self._apply_fun(jnp.abs)

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
        # Scale the coefficients; rebuild with the SAME tech class — the
        # previous hard-coded Chebtech2 reinterpreted Fourier coefficients
        # as Chebyshev ones for trig pieces.
        scaled_coeffs = tech_der.coeffs * jnp.float64(scale)
        new_tech = type(self.tech).from_coeffs(scaled_coeffs)
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
        try:
            tech_cs = self.tech.cumsum()
        except ValueError:
            # A Trigtech antiderivative of a non-zero-mean periodic function
            # is not periodic; MATLAB casts to the chebtech basis rather than
            # error (test_trigcasting pass 19).  Zero-mean trig stays trig.
            from chebfunjax.tech.trigtech import Trigtech
            if not isinstance(self.tech, Trigtech):
                raise
            cheb = Chebtech2.from_function(lambda t, _s=self.tech: _s(t))
            scaled = cheb.cumsum().coeffs * jnp.float64(scale)
            return _Piece(tech=Chebtech2.from_coeffs(scaled), interval=(a, b))
        # Scale coefficients by (b-a)/2
        scaled_coeffs = tech_cs.coeffs * jnp.float64(scale)
        new_tech = type(self.tech).from_coeffs(scaled_coeffs)
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
        from chebfunjax.fun.singfun import Singfun
        if isinstance(self.tech, Singfun) or isinstance(other.tech, Singfun):
            # <f, g> = \int conj(f) g over the reference interval; the
            # SingFun product folds the endpoint exponents together and
            # integrates them exactly (Gauss-Jacobi).  Promote a smooth
            # partner to a trivial-exponent SingFun so the product is defined.
            sf = self.tech if isinstance(self.tech, Singfun) \
                else Singfun.from_chebtech(self.tech, (0.0, 0.0))
            og = other.tech if isinstance(other.tech, Singfun) \
                else Singfun.from_chebtech(other.tech, (0.0, 0.0))
            # Conjugate the left factor, keeping it a Singfun (Singfun.conj
            # downgrades a trivial-exponent case to its smooth part).
            conj_sf = Singfun(sf.smoothPart.conj(), sf.exponents)
            return (conj_sf * og).sum() * jnp.float64(scale)
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
        if self.tech.coeffs.ndim == 2 or (
                jnp.iscomplexobj(self.tech.coeffs)
                and not getattr(self.tech, "is_real", False)):
            # Array-valued and/or complex: delegate to the tech (which
            # handles per-column extrema and MATLAB's complex |f|^2
            # path); map positions from the reference interval to
            # [a, b].
            (mn, mnp), (mx, mxp) = self.tech.minandmax()
            xmn = 0.5 * (b - a) * mnp + 0.5 * (a + b)
            xmx = 0.5 * (b - a) * mxp + 0.5 * (a + b)
            return (xmn, mn), (xmx, mx)
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


def _is_empty_operand(x) -> bool:
    """True if ``x`` is an empty Chebfun or an empty numeric array/list
    (MATLAB propagates emptiness through arithmetic: ``f + [] == []``)."""
    import numpy as _np
    if isinstance(x, Chebfun):
        return x.isempty()
    if x is None:
        return True
    if not callable(x) and hasattr(x, "__len__"):
        try:
            return len(_np.ravel(_np.asarray(x, dtype=object))) == 0
        except (TypeError, ValueError):
            return False
    return False


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
    # Dirac delta functions carried alongside the smooth part, as a tuple
    # of (location, magnitude) pairs.  Static metadata (hashable), so the
    # JIT/vmap pytree structure is unchanged when there are no deltas.
    # Added by Claude Opus 4.8 (task #9).
    deltas: tuple = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Internal constructor (use factory classmethod or chebfun() instead)
    # ------------------------------------------------------------------

    def __init__(self, funs: list[_Piece], domain: Domain,
                 deltas: tuple = ()) -> None:
        """Low-level constructor.  Prefer :func:`chebfun` for user code.

        Parameters
        ----------
        funs : list[_Piece]
            Non-empty list of smooth pieces in domain order.
        domain : Domain
            Corresponding domain (breakpoints must match piece intervals).
        deltas : tuple, optional
            ``((location, magnitude), ...)`` Dirac deltas (e.g. from
            differentiating across a jump).  Ignored by point evaluation
            (measure zero); contribute to :meth:`sum`.
        """
        # An empty funs list is the MATLAB empty chebfun (chebfun());
        # isempty() is True and most operations are undefined on it.
        self.funs = funs
        self.domain = domain
        self.deltas = tuple(deltas)

    @classmethod
    def empty(cls) -> "Chebfun":
        """The empty Chebfun (MATLAB ``chebfun()``): no pieces; isempty() is
        True and every operation propagates emptiness.

        Provenance
        ----------
        MATLAB source : @chebfun/isempty.m
        Chebfun commit: 7574c77
        """
        return cls(funs=[], domain=Domain((-1.0, 1.0)))

    # ------------------------------------------------------------------
    # Orientation (row vs column chebfun) — the MATLAB isTransposed flag
    # ------------------------------------------------------------------
    #
    # A column Chebfun is an Inf-by-n object; its transpose is an n-by-Inf
    # *row* Chebfun.  MATLAB records the orientation in an ``isTransposed``
    # property and dispatches size(), mtimes(), fliplr(), etc. on it.
    #
    # We store the flag as a private marker attribute set via
    # ``object.__setattr__`` (the same bypass of eqx's frozen ``__setattr__``
    # that the delta machinery uses for ``_delta_locs``).  It is therefore
    # NOT part of the equinox pytree: it is Python-side dispatch metadata that
    # does not survive ``jax.jit``/``vmap`` flattening.  This is deliberate --
    # orientation is never consulted inside a JIT hot path, and keeping it off
    # the pytree means every existing (column) Chebfun keeps the identical
    # pytree structure, so ``is_transposed`` defaults to False everywhere.

    @property
    def is_transposed(self) -> bool:
        """True for a row (transposed) Chebfun, False for a column.

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (isTransposed property)
        Chebfun commit: 7574c77
        """
        return bool(getattr(self, "_is_transposed", False))

    @staticmethod
    def _as_transposed(obj: "Chebfun", flag: bool) -> "Chebfun":
        """Tag ``obj`` with orientation ``flag`` (in place) and return it."""
        if flag:
            object.__setattr__(obj, "_is_transposed", True)
        return obj

    def transpose(self) -> "Chebfun":
        """Non-conjugate transpose ``F.'``: swap row/column orientation.

        Converts a column Chebfun to a row Chebfun and vice versa without
        conjugating.  The underlying pieces are shared unchanged.

        Provenance
        ----------
        MATLAB source : @chebfun/transpose.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.ctranspose
        """
        new = Chebfun(funs=self.funs, domain=self.domain, deltas=self.deltas)
        return Chebfun._as_transposed(new, not self.is_transposed)

    @property
    def T(self) -> "Chebfun":
        """The non-conjugate transpose ``F.'`` (see :meth:`transpose`)."""
        return self.transpose()

    def ctranspose(self) -> "Chebfun":
        """Complex-conjugate transpose ``F'`` = ``transpose(conj(F))``.

        Provenance
        ----------
        MATLAB source : @chebfun/ctranspose.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.transpose
        """
        return self.conj().transpose()

    @property
    def H(self) -> "Chebfun":
        """The complex-conjugate transpose ``F'`` (see :meth:`ctranspose`)."""
        return self.ctranspose()

    def permute(self, order) -> "Chebfun":
        """Permute the two Chebfun array dimensions.

        Since a Chebfun has exactly two dimensions, ``order`` must be
        ``[1, 2]`` (identity) or ``[2, 1]`` (transpose, ``F.'``).

        Provenance
        ----------
        MATLAB source : @chebfun/permute.m
        Chebfun commit: 7574c77
        """
        order = tuple(int(o) for o in order)
        if order == (1, 2):
            return self
        if order == (2, 1):
            return self.transpose()
        raise ValueError(
            "ORDER must be a permutation of [1, 2] for a Chebfun "
            f"(got {list(order)})."
        )

    # ------------------------------------------------------------------
    # pointValues — the MATLAB explicit-value-at-breakpoints field
    # ------------------------------------------------------------------
    #
    # MATLAB Chebfun carries a ``pointValues`` array: the function value AT
    # each breakpoint, which for a kink / jump may differ from either
    # one-sided limit (e.g. ``abs`` and ``sign`` record ``|f|`` / ``sign(f)``
    # of the stored value there).  It is metadata, not part of the smooth
    # pieces, so -- like the ``_is_transposed`` orientation flag -- we store
    # any explicit override off the equinox pytree via ``object.__setattr__``
    # and default to the endpoint feval when none is set.  This keeps the
    # pytree structure of every existing Chebfun untouched.

    @property
    def point_values(self) -> jax.Array:
        """Function values at the breakpoints (MATLAB ``pointValues``).

        Returns the explicit override set by :meth:`set_point_values` when
        present, else the default: the value of the Chebfun evaluated at each
        breakpoint (shape ``(n_ends,)``, or ``(n_ends, n_cols)`` when
        array-valued).

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (pointValues property)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.set_point_values
        """
        override = getattr(self, "_point_values", None)
        if override is not None:
            return override
        import numpy as _np
        bps = _np.asarray(list(self.domain.breakpoints), dtype=float)
        return self(jnp.asarray(bps))

    def set_point_values(self, values) -> "Chebfun":
        """Return a copy carrying explicit ``pointValues`` (MATLAB
        ``f.pointValues = values``).

        The stored values must have one entry per breakpoint (an
        ``(n_ends,)`` vector, or ``(n_ends, n_cols)`` for an array-valued
        Chebfun).  They are metadata: point evaluation away from the
        breakpoints is unaffected; :meth:`abs` and :meth:`sign` propagate
        them element-wise (as MATLAB's ``abs``/``sign`` do).

        Provenance
        ----------
        MATLAB source : @chebfun/chebfun.m (pointValues assignment)
        Chebfun commit: 7574c77

        See Also
        --------
        Chebfun.point_values
        """
        new = Chebfun(funs=self.funs, domain=self.domain, deltas=self.deltas)
        object.__setattr__(new, "_point_values", jnp.asarray(values))
        if self.is_transposed:
            object.__setattr__(new, "_is_transposed", True)
        return new

    def _propagate_point_values(self, result: "Chebfun", op) -> "Chebfun":
        """Carry an explicit ``pointValues`` override through a pointwise op.

        MATLAB's ``abs``/``sign`` record ``op(pointValues)`` at the
        breakpoints.  Applies only when an override was explicitly set; the
        default (endpoint feval) is recomputed from ``result`` on demand.
        """
        override = getattr(self, "_point_values", None)
        if override is not None:
            object.__setattr__(result, "_point_values", op(override))
        return result

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        n: int | None = None,
        *,
        maxpow2: int = 16,
        tol: float | None = None,
        turbo: bool = False,
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
            piece = _Piece.from_function(f, sub.a, sub.b, n=n,
                                         maxpow2=maxpow2, tol=tol,
                                         turbo=turbo)
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

    def __call__(self, x: jax.Array, side: "str | None" = None) -> jax.Array:
        """Evaluate the Chebfun at point(s) x.

        If ``x`` is itself a Chebfun ``g``, returns the composition
        ``f(g)`` as a new Chebfun (MATLAB f(g) semantics).

        The optional ``side`` selects a one-sided limit at a breakpoint:
        ``'left'`` / ``'-'`` / ``'start'`` uses the piece to the left of
        ``x``; ``'right'`` / ``'+'`` / ``'end'`` uses the piece to the
        right (MATLAB ``feval(f, x, 'left')`` etc.).  This matters only at
        interior breakpoints of a piecewise Chebfun, where the value may
        differ from either side.

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
        if isinstance(x, Chebfun):
            return self.compose_chebfun(x)
        if side is not None:
            _record_side_eval(x)
            return self._feval_side(x, side)
        # Preserve a complex argument (MATLAB feval evaluates a real Chebfun
        # at complex points, routing pieces by real(x)); everything else is
        # promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
        scalar_input = x.ndim == 0
        x = jnp.atleast_1d(x)

        if len(self.funs) == 1:
            # Fast path: single piece — fully JIT-able
            result = self.funs[0](x)
            if scalar_input:
                result = result[0]
            return self._orient_values(result)

        # Multi-piece: Python dispatch.  Piece selection uses real(x) so a
        # complex evaluation point is routed by its real part (MATLAB
        # xReal = real(x) in columnFeval).
        # (array-valued pieces add a trailing column axis to the output;
        # masks broadcast over it.  jnp.where promotes to complex when
        # a piece evaluates complex, as before.)
        xr = jnp.real(x)
        cols = self.funs[0].tech.coeffs.shape[1:]
        out = jnp.full(x.shape + cols, jnp.nan, dtype=x.dtype)
        n_pieces = len(self.funs)
        for i, piece in enumerate(self.funs):
            a, b = piece.interval
            if i == 0:
                # Left piece: include all x <= b
                if n_pieces == 1:
                    mask = jnp.ones(x.shape, dtype=bool)
                else:
                    mask = xr <= b
            elif i == n_pieces - 1:
                # Right piece: include all x >= a
                mask = xr >= a
            else:
                # Interior piece: include a <= x <= b
                mask = (xr >= a) & (xr <= b)

            # Evaluate masked points
            x_piece = jnp.where(mask, x, jnp.asarray(a, dtype=x.dtype))
            vals = piece(x_piece)
            maskE = mask.reshape(mask.shape + (1,) * len(cols))
            out = jnp.where(maskE, vals, out)

        if scalar_input:
            out = out[0]
        return self._orient_values(out)

    def _feval_side(self, x, side: str):
        """One-sided evaluation at ``x`` (see :meth:`__call__`).

        Selects the piece on the requested side of every point and
        evaluates that piece's polynomial there, so a breakpoint returns
        the left- or right-hand limit rather than the default (which favours
        the right piece).

        Provenance
        ----------
        MATLAB source : @chebfun/feval.m (one-sided evaluation)
        Chebfun commit: 7574c77
        """
        key = str(side).lower()
        if key in ("left", "-", "start"):
            want_left = True
        elif key in ("right", "+", "end"):
            want_left = False
        else:
            raise ValueError(
                f"Chebfun side {side!r}: expected 'left'/'-'/'start' or "
                f"'right'/'+'/'end'.")
        xa = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = xa.ndim == 0
        xa = jnp.atleast_1d(xa)
        tol = 1e-12 * max(1.0, abs(float(self.domain.b) - float(self.domain.a)))
        n_pieces = len(self.funs)

        def pick_piece(xi: float) -> int:
            # First pass: an interior breakpoint match on the requested side
            # (left limit -> piece whose right end is xi; right limit ->
            # piece whose left end is xi) takes priority over containment.
            for i, piece in enumerate(self.funs):
                a, b = float(piece.interval[0]), float(piece.interval[1])
                if want_left and abs(xi - b) <= tol:
                    return i
                if not want_left and abs(xi - a) <= tol:
                    return i
            # Second pass: the piece containing xi, else the nearest endpoint.
            for i, piece in enumerate(self.funs):
                a, b = float(piece.interval[0]), float(piece.interval[1])
                if a - tol <= xi <= b + tol:
                    return i
            return n_pieces - 1 if xi >= float(self.funs[-1].interval[1]) else 0

        vals = []
        for xi in xa:
            i = pick_piece(float(xi))
            vals.append(self.funs[i](jnp.asarray(xi, dtype=jnp.float64)))
        out = jnp.stack(vals)
        if scalar_input:
            out = out[0]
        return self._orient_values(out)

    def _orient_values(self, vals):
        """Transpose an evaluation result for a row (transposed) Chebfun.

        MATLAB ``feval(f.', x)`` returns ``feval(f, x).'``.  For an
        array-valued row Chebfun the (n_points, n_cols) column result becomes
        (n_cols, n_points).  A scalar-valued (1-D) result is unchanged (its
        transpose is itself, matching MATLAB where a 1-by-m row stays 1-by-m).
        Column Chebfuns (the default) are returned untouched.
        """
        if not self.is_transposed:
            return vals
        v = jnp.asarray(vals)
        if v.ndim >= 2:
            return jnp.swapaxes(v, -1, -2)
        return v

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
        orientation = "row" if self.is_transposed else "column"
        header = f"Chebfun {orientation} ({n_pieces} smooth {piece_word})"
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
        """Raise ValueError if two Chebfuns live on different intervals.

        Interior breakpoints may differ — :meth:`_overlap` merges those.
        Only the outer endpoints must agree.
        """
        hscale = max(abs(float(f.domain.a)), abs(float(f.domain.b)), 1.0)
        tol = 1e-14 * hscale
        if (abs(float(f.domain.a) - float(g.domain.a)) > tol
                or abs(float(f.domain.b) - float(g.domain.b)) > tol):
            raise ValueError(
                f"Cannot combine Chebfun on [{f.domain.a}, {f.domain.b}] "
                f"with Chebfun on [{g.domain.a}, {g.domain.b}]: intervals "
                f"do not match.  Use f.restrict(...) first."
            )

    def _with_breakpoints(self, bps: "tuple[float, ...]") -> Chebfun:
        """Exactly re-break onto a refined breakpoint list.

        ``bps`` must contain the current breakpoints (up to rounding).
        Each new sub-interval is cut from its containing piece via the
        exact coefficient-space restriction, so no re-approximation
        error is introduced.
        """
        new_dom = Domain(tuple(float(b) for b in bps))
        hscale = max(abs(float(self.domain.a)), abs(float(self.domain.b)), 1.0)
        tol = 1e-12 * hscale
        new_funs = []
        for sub in new_dom.intervals:
            mid = 0.5 * (float(sub.a) + float(sub.b))
            piece = next(
                p for p in self.funs
                if float(p.interval[0]) - tol <= mid <= float(p.interval[1]) + tol
            )
            new_funs.append(piece.restrict(float(sub.a), float(sub.b)))
        return Chebfun(funs=new_funs, domain=new_dom)

    @staticmethod
    def _overlap(f: Chebfun, g: Chebfun) -> "tuple[Chebfun, Chebfun]":
        """Re-break both Chebfuns onto the union of their breakpoints.

        MATLAB Chebfun arithmetic accepts operands with different interior
        breakpoints and merges them (@chebfun/overlap.m); requiring
        identical piecewise structure broke e.g. ``abs(x) * u`` inside
        operators when one operand had root-splitting breakpoints.

        Provenance
        ----------
        MATLAB source : @chebfun/overlap.m
        Chebfun commit: 7574c77
        """
        Chebfun._check_domains(f, g)
        if f.domain == g.domain:
            return f, g
        hscale = max(abs(float(f.domain.a)), abs(float(f.domain.b)), 1.0)
        tol = 1e-14 * hscale
        merged: list[float] = []
        for x in sorted(
            {float(b) for b in f.domain.breakpoints}
            | {float(b) for b in g.domain.breakpoints}
        ):
            if not merged or x - merged[-1] > tol:
                merged.append(x)
        # Pin the outer endpoints to f's exact values.
        merged[0] = float(f.domain.a)
        merged[-1] = float(f.domain.b)
        bps = tuple(merged)
        return f._with_breakpoints(bps), g._with_breakpoints(bps)

    @staticmethod
    def _binary_op(f: Chebfun, g: Chebfun, op) -> Chebfun:
        """Apply a piecewise binary op between two same-domain Chebfuns.

        ``op`` must be a method name (str) on ``Chebtech2`` accepting one
        Chebtech2 argument, or a callable ``op(tech_a, tech_b) -> Chebtech2``.
        """
        f, g = Chebfun._overlap(f, g)
        new_funs = [
            pf.with_tech(op(*_cast_tech_pair(pf.tech, pg.tech)))
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
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
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
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
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
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)

    def __pos__(self) -> Chebfun:
        """Unary plus (identity)."""
        return self

    def __mul__(self, other) -> Chebfun:
        """Matrix-style multiplication (MATLAB ``*``/``mtimes``).

        Dispatch mirrors @chebfun/mtimes.m:

        - Chebfun * scalar: pointwise scaling (orientation preserved).
        - Chebfun * Chebfun with the SAME orientation: pointwise product
          ``f .* g`` (both column or both row).
        - row * column: the L2 inner product ``\\int f g dx`` (a scalar).
          MATLAB computes ``innerProduct(conj(f), g)`` to undo the
          conjugation that ``innerProduct`` applies to its first factor; the
          equivalent here is ``conj(f).inner(g)``.
        - column * row: the rank-1 outer product (a Chebfun2), which
          chebfunjax does not yet provide.

        Provenance
        ----------
        MATLAB source : @chebfun/mtimes.m, @chebfun/times.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
        if isinstance(other, Chebfun):
            f_trans = self.is_transposed
            g_trans = other.is_transposed
            if f_trans == g_trans:
                # Both column or both row: pointwise product, orientation kept.
                return Chebfun._as_transposed(
                    Chebfun._binary_op(self, other, lambda a, b: a * b),
                    f_trans)
            if f_trans and not g_trans:
                # Row * column -> inner product scalar.  inner() conjugates
                # its first argument, so conj(f).inner(g) == int f*g.
                return self.conj().inner(other)
            # Column * row -> rank-1 outer product (a Chebfun2).
            raise NotImplementedError(
                "column * row (rank-1 outer product / Chebfun2) is not "
                "supported in chebfunjax."
            )
        # If other is not a scalar/array, defer to other's __rmul__
        if not isinstance(other, (int, float, complex, jnp.ndarray, jax.Array)):
            return NotImplemented
        new_funs = [
            piece._apply_unary(piece.tech * other)
            for piece in self.funs
        ]
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)

    def __rmul__(self, other) -> Chebfun:
        return self.__mul__(other)

    def __truediv__(self, other) -> Chebfun:
        """Pointwise division: Chebfun / scalar or Chebfun / Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
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
        """Absolute value — delegates to :meth:`abs`.

        MATLAB's abs(f) introduces breakpoints at the roots so each piece
        stays smooth; ``abs(f)`` (this dunder) must behave identically to
        ``f.abs()``, otherwise the builtin silently returns a non-split,
        under-resolved representation.

        NOT JIT-safe (root-finding and adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/abs.m
        Chebfun commit: 7574c77
        """
        return self.abs()

    def _abs_piecewise_raw(self) -> Chebfun:
        """Piece-level |f| without root-splitting (internal fallback)."""
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
        return self._root_power(0.5, jnp.sqrt)

    def _root_power(self, b: float, smooth_op) -> Chebfun:
        """Raise to a real power ``b`` producing singular reps at roots.

        Mirrors MATLAB @chebfun/power.m (``columnPower``, general case):
        breakpoints are added at the interior roots of ``f``, then each piece
        is raised to the power at the fun level.  A piece that vanishes at a
        breakpoint carries the root into a fractional (branch-point) exponent
        via :class:`Singfun`; pieces with no roots stay smooth.

        Provenance
        ----------
        MATLAB source : @chebfun/power.m (columnPower), @singfun/power.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.fun.singfun import Singfun
        # No roots anywhere -> smooth composition (fast path, matches the
        # positive-function tests exactly).
        r = _np.asarray(self.roots(), dtype=float).ravel()
        r = r[_np.isfinite(r)]
        if len(r) == 0:
            # Exponent-aware even without interior roots: a Singfun piece
            # must scale its exponents (f^b keeps the singularity), not be
            # re-approximated smoothly.
            if any(isinstance(p.tech, Singfun) for p in self.funs):
                new_funs = [
                    _Piece(tech=p.tech ** b, interval=p.interval)
                    if isinstance(p.tech, Singfun) else p._apply_fun(smooth_op)
                    for p in self.funs
                ]
                return Chebfun(funs=new_funs, domain=self.domain)
            return self._apply_fun(smooth_op)
        # Split at interior roots so every remaining root sits on a breakpoint.
        fbr = self.addBreaksAtRoots()
        new_funs = []
        for p in fbr.funs:
            tech = p.tech
            if isinstance(tech, Singfun):
                sf = tech ** b
            else:
                sf = (Singfun.from_chebtech(tech, (0.0, 0.0))
                      .extractBoundaryRoots() ** b).simplifyExponents()
            # A piece with no boundary roots collapses to trivial exponents;
            # keep it smooth to avoid needless Singfun overhead downstream.
            if all(abs(e) < 1e-14 for e in sf.exponents):
                new_funs.append(_Piece.from_function(
                    lambda x, _p=p: smooth_op(_p(x)),
                    p.interval[0], p.interval[1]))
            else:
                new_funs.append(_Piece(tech=sf, interval=p.interval))
        return Chebfun(funs=new_funs, domain=fbr.domain)

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
        return self._propagate_point_values(self._abs_core(), jnp.abs)

    def _abs_core(self) -> Chebfun:
        """Root-splitting ``|f|`` without pointValues propagation."""
        # Find roots where the function changes sign and add them as
        # breakpoints, then apply |·| piecewise for smoothness.
        import numpy as _np
        roots = _np.asarray(self.roots())
        if roots.ndim == 2:
            # Array-valued: the union of every column's roots splits
            # ALL columns (MATLAB @chebfun/abs.m uses roots(f) which
            # gathers all columns); drop the NaN padding.
            roots = _np.sort(roots[_np.isfinite(roots)])
        if roots.shape[0] == 0:
            # No sign changes — simple abs on each piece (exponent-
            # preserving for Singfun pieces, see _Piece.abs).
            return Chebfun(funs=[p.abs() for p in self.funs],
                           domain=self.domain)

        # Build new breakpoints: existing domain breakpoints + roots
        existing = _np.array(list(self.domain.breakpoints))
        new_bps = _np.sort(_np.unique(
            _np.concatenate([existing, roots])
        ))
        # Remove duplicates within tolerance
        domain_len = float(self.domain.b - self.domain.a)
        tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)
        mask = _np.concatenate([[True], _np.diff(new_bps) > tol])
        new_bps = new_bps[mask]

        if len(new_bps) < 2:
            return Chebfun(funs=[p.abs() for p in self.funs],
                           domain=self.domain)

        new_dom = Domain(tuple(float(b) for b in new_bps))
        f = self  # capture for closure
        new_funs = [
            _Piece.from_function(lambda x, _f=f: jnp.abs(_f(x)), sub.a, sub.b)
            for sub in new_dom.intervals
        ]
        return Chebfun(funs=new_funs, domain=new_dom)

    # ------------------------------------------------------------------
    # Assorted MATLAB utilities (added by Claude Fable 5,
    # MISSING_FEATURES named-utilities sweep).
    # ------------------------------------------------------------------

    def prod(self) -> jax.Array:
        """Integral product: exp(sum(log(f))) (MATLAB prod).

        Provenance
        ----------
        MATLAB source : @chebfun/prod.m
        Chebfun commit: 7574c77
        """
        return jnp.exp(self.log().sum())

    def compose(self, op, g: "Chebfun" = None) -> "Chebfun":
        """Compose with a pointwise operator: op(f) or op(f, g)
        (MATLAB compose).  Falls back to splitting when the result
        is not smooth.

        Provenance
        ----------
        MATLAB source : @chebfun/compose.m
        Chebfun commit: 7574c77
        """
        import warnings as _w
        a, b = float(self.domain.a), float(self.domain.b)
        if g is None:
            def h(x):
                return op(self(x))
        else:
            def h(x):
                return op(self(x), g(x))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            try:
                out = chebfun(h, domain=(a, b))
                if all(p.tech.ishappy for p in out.funs):
                    return out
            except Exception:
                pass
            return chebfun(h, domain=(a, b), splitting=True)

    def compose_chebfun(self, g: "Chebfun") -> "Chebfun":
        """Composition f(g): evaluate self at the values of g
        (MATLAB compose(g, f) / f(g) syntax).

        Provenance
        ----------
        MATLAB source : @chebfun/compose.m (chebfun-of-chebfun branch)
        Chebfun commit: 7574c77
        """
        import warnings as _w

        import numpy as _np
        a, b = float(g.domain.a), float(g.domain.b)

        def h(x):
            return self(g(x))

        # Kinks of f(g) occur where g crosses a GENUINE kink of f, so
        # place breakpoints at the roots of g - c (as MATLAB compose
        # does) instead of bisection splitting; spurious breaks with
        # no derivative jump are filtered out.
        breaks = {a, b}
        breaks.update(float(p.interval[0]) for p in g.funs[1:])
        if len(self.funs) > 1:
            fd = self.diff()
            vs = max(float(self.vscale), 1.0)
            for i in range(len(self.funs) - 1):
                c = float(self.funs[i + 1].interval[0])
                jump = abs(
                    float(fd.funs[i](jnp.asarray(c)))
                    - float(fd.funs[i + 1](jnp.asarray(c))))
                if jump > 1e-7 * vs:
                    r = _np.asarray((g - c).roots(),
                                    dtype=float).ravel()
                    breaks.update(
                        float(t) for t in r if a < t < b)
        pts = sorted(breaks)

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if len(pts) == 2:
                return chebfun(h, domain=(a, b))
            out = chebfun(h, domain=(pts[0], pts[1]))
            for i in range(1, len(pts) - 1):
                out = out.join(
                    chebfun(h, domain=(pts[i], pts[i + 1])))
            return out

    def legcoeffs(self, n: int | None = None) -> jax.Array:
        """Legendre expansion coefficients of a single-piece chebfun
        (MATLAB legcoeffs), via the cheb2leg transform.

        Provenance
        ----------
        MATLAB source : @chebfun/legcoeffs.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.transforms import cheb2leg
        if len(self.funs) != 1:
            raise ValueError("legcoeffs requires a single-piece chebfun")
        c = cheb2leg(self.funs[0].tech.coeffs)
        if n is not None:
            m = len(jnp.asarray(c))
            if n <= m:
                c = c[:n]
            else:
                c = jnp.concatenate(
                    [c, jnp.zeros(n - m, dtype=c.dtype)])
        return c

    def jaccoeffs(self, n: int | None = None, alpha: float = 0.0,
                  beta: float = 0.0) -> jax.Array:
        """Jacobi expansion coefficients (MATLAB jaccoeffs), via the
        cheb2jac transform.

        Provenance
        ----------
        MATLAB source : @chebfun/jaccoeffs.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.transforms import cheb2jac
        if len(self.funs) != 1:
            raise ValueError("jaccoeffs requires a single-piece chebfun")
        c = cheb2jac(self.funs[0].tech.coeffs, alpha, beta)
        if n is not None:
            m = len(jnp.asarray(c))
            if n <= m:
                c = c[:n]
            else:
                c = jnp.concatenate(
                    [c, jnp.zeros(n - m, dtype=c.dtype)])
        return c

    def addBreaks(self, breaks) -> "Chebfun":
        """Introduce new interior breakpoints (MATLAB addBreaks).

        Provenance
        ----------
        MATLAB source : @chebfun/addBreaks.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        a, b = float(self.domain.a), float(self.domain.b)
        old = [float(p.interval[0]) for p in self.funs] + [b]
        pts = sorted(set(old)
                     | {float(t) for t in _np.atleast_1d(
                         _np.asarray(breaks, dtype=float))
                        if a < float(t) < b})
        out = self.restrict(pts[0], pts[1])
        for i in range(1, len(pts) - 1):
            out = out.join(self.restrict(pts[i], pts[i + 1]))
        return out

    def addBreaksAtRoots(self, tol: float = 0.0) -> "Chebfun":
        """Introduce breakpoints at the interior roots of f
        (MATLAB addBreaksAtRoots).

        Provenance
        ----------
        MATLAB source : @chebfun/addBreaksAtRoots.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        r = _np.asarray(self.roots(), dtype=float).ravel()
        a, b = float(self.domain.a), float(self.domain.b)
        eps_ = float(_np.finfo(float).eps)
        gap = max(tol, 100 * eps_ * max(abs(a), abs(b), 1.0))
        r = r[(r > a + gap) & (r < b - gap)]
        if len(r) == 0:
            return self
        return self.addBreaks(r)

    def var(self) -> jax.Array:
        """Variance over the domain: mean(|f - mean(f)|^2)
        (MATLAB var).

        Provenance
        ----------
        MATLAB source : @chebfun/var.m
        Chebfun commit: 7574c77
        """
        m = self.mean()
        d = self - complex(m) if jnp.iscomplexobj(jnp.asarray(m)) \
            else self - float(m)
        if any(jnp.iscomplexobj(p.tech.coeffs) for p in self.funs):
            return jnp.real((d * d.conj()).mean())
        return (d * d).mean()

    def std(self) -> jax.Array:
        """Standard deviation: sqrt(var(f)) (MATLAB std).

        Provenance
        ----------
        MATLAB source : @chebfun/std.m
        Chebfun commit: 7574c77
        """
        return jnp.sqrt(self.var())

    def merge(self) -> "Chebfun":
        """Remove unnecessary interior breakpoints (MATLAB merge):
        re-approximate globally and keep the merged representation if
        it matches the piecewise one.

        Provenance
        ----------
        MATLAB source : @chebfun/merge.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        if len(self.funs) == 1:
            return self
        a, b = float(self.domain.a), float(self.domain.b)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            cand = Chebfun.from_function(
                lambda x: self(x), Domain((a, b)))
        xs = jnp.asarray(_np.linspace(a + 1e-9 * (b - a),
                                      b - 1e-9 * (b - a), 201))
        err = float(jnp.max(jnp.abs(cand(xs) - self(xs))))
        tol = 1e3 * float(_np.finfo(float).eps) * max(self.vscale, 1.0)
        return cand if (err < tol and cand.funs[0].tech.ishappy) \
            else self

    def rem(self, g) -> "Chebfun":
        """Remainder after division: f - fix(f/g)*g (MATLAB rem).

        Provenance
        ----------
        MATLAB source : @chebfun/rem.m
        Chebfun commit: 7574c77
        """
        q = (self * (1.0 / float(g))) if isinstance(g, (int, float)) \
            else (self / g)
        return self - q.fix() * g

    def deriv(self, x, k: int = 1):
        """Evaluate the k-th derivative at x (MATLAB deriv).

        Provenance
        ----------
        MATLAB source : @chebfun/deriv.m
        Chebfun commit: 7574c77
        """
        return self.diff(k)(jnp.asarray(x))

    def nextpow2(self) -> "Chebfun":
        """ceil(log2(|f|)) as a piecewise-constant chebfun
        (MATLAB nextpow2; f must be positive).

        Provenance
        ----------
        MATLAB source : @chebfun/nextpow2.m
        Chebfun commit: 7574c77
        """
        lg = Chebfun.from_function(
            lambda x: jnp.log2(self(x)),
            Domain((float(self.domain.a), float(self.domain.b))))
        return lg.ceil()

    def realsqrt(self) -> "Chebfun":
        """sqrt with a realness guard (MATLAB realsqrt).

        Provenance
        ----------
        MATLAB source : @chebfun/realsqrt.m
        Chebfun commit: 7574c77
        """
        (_, fmin), _ = self.minandmax()
        if float(fmin) < -1e-12 * max(self.vscale, 1.0):
            raise ValueError("realsqrt: function is negative somewhere")
        return self.sqrt()

    def realpow(self, p) -> "Chebfun":
        """f**p with a realness guard (MATLAB realpow)."""
        if float(p) != int(p):
            (_, fmin), _ = self.minandmax()
            if float(fmin) < -1e-12 * max(self.vscale, 1.0):
                raise ValueError(
                    "realpow: fractional power of a negative function")
        return self ** p

    def arclength(self) -> jax.Array:
        """Arc length of the graph: int sqrt(1 + f'(x)^2) dx.

        Provenance
        ----------
        MATLAB source : @chebfun/arcLength.m
        Chebfun commit: 7574c77
        """
        d = self.diff()
        integrand = Chebfun.from_function(
            lambda x, _d=d: jnp.sqrt(1.0 + _d(x) ** 2),
            Domain((float(self.domain.a), float(self.domain.b))))
        return integrand.sum()

    def hypot(self, other: "Chebfun") -> "Chebfun":
        """sqrt(f^2 + g^2) computed robustly (MATLAB hypot).

        Provenance
        ----------
        MATLAB source : @chebfun/hypot.m
        Chebfun commit: 7574c77
        """
        f = self

        def op(x):
            return jnp.hypot(f(x), other(x))

        return Chebfun.from_function(
            op, Domain((float(self.domain.a), float(self.domain.b))))

    def fix(self) -> "Chebfun":
        """Round toward zero (MATLAB fix): floor for f >= 0, ceil for
        f < 0, as an exact piecewise-constant chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/fix.m
        Chebfun commit: 7574c77
        """
        fl = self.floor()
        ce = self.ceil()
        # combine: pieces where the function is negative use ceil
        import numpy as _np
        bps = sorted(set([float(v) for v in fl.domain.breakpoints]
                         + [float(v) for v in ce.domain.breakpoints]
                         + [float(r) for r in _np.asarray(self.roots())]))
        funs = []
        for a_, b_ in zip(bps[:-1], bps[1:]):
            mid = 0.5 * (a_ + b_)
            v = float(self(jnp.asarray(mid)))
            const = float(_np.floor(v) if v >= 0 else _np.ceil(v))
            funs.append(_Piece.from_coeffs(
                jnp.asarray([const], dtype=jnp.float64), a_, b_))
        return Chebfun(funs=funs, domain=Domain(tuple(bps)))

    def _cummax_or_min(self, use_max: bool) -> "Chebfun":
        import numpy as _np
        a, b = float(self.domain.a), float(self.domain.b)
        # running extremum: piecewise -- alternates between copies of f
        # (where f is the running extremum) and constants (where the
        # past extremum dominates).  Build by dense scan + refinement.
        xs = _np.linspace(a, b, 2049)
        vals = _np.asarray(self(jnp.asarray(xs)))
        run = (_np.maximum if use_max else _np.minimum).accumulate(vals)

        def op(x):
            xq = _np.asarray(x)
            idx = _np.clip(_np.searchsorted(xs, xq.ravel()), 1,
                           len(xs) - 1)
            base = run[idx - 1]
            cur = _np.asarray(self(jnp.asarray(xq.ravel())))
            out = (_np.maximum(base, cur) if use_max
                   else _np.minimum(base, cur))
            return jnp.asarray(out.reshape(xq.shape))

        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            return chebfun(op, domain=(a, b), splitting=True)

    def cummax(self) -> "Chebfun":
        """Running maximum (MATLAB cummax).

        Provenance
        ----------
        MATLAB source : @chebfun/cummax.m
        Chebfun commit: 7574c77
        """
        return self._cummax_or_min(True)

    def cummin(self) -> "Chebfun":
        """Running minimum (MATLAB cummin)."""
        return self._cummax_or_min(False)

    def join(self, other: "Chebfun") -> "Chebfun":
        """Concatenate with a chebfun on an adjacent domain
        (MATLAB join).

        Provenance
        ----------
        MATLAB source : @chebfun/join.m
        Chebfun commit: 7574c77
        """
        if abs(float(self.domain.b) - float(other.domain.a)) > 1e-14:
            raise ValueError("join: domains must be adjacent")
        bps = tuple([float(v) for v in self.domain.breakpoints]
                    + [float(v) for v in other.domain.breakpoints][1:])
        return Chebfun(funs=list(self.funs) + list(other.funs),
                       domain=Domain(bps))

    def inv(self) -> "Chebfun":
        """Compositional inverse of a monotonic chebfun (MATLAB inv):
        g with g(f(x)) = x.

        Provenance
        ----------
        MATLAB source : @chebfun/inv.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        a, b = float(self.domain.a), float(self.domain.b)
        fa = float(self(jnp.asarray(a)))
        fb = float(self(jnp.asarray(b)))
        if fa == fb:
            raise ValueError("inv: function must be monotonic")
        lo, hi = (fa, fb) if fa < fb else (fb, fa)

        def g(y):
            yq = _np.asarray(y, dtype=float).ravel()
            out = _np.empty_like(yq)
            for i, yv in enumerate(yq):
                x0, x1 = a, b
                # bisection + Newton polish
                for _ in range(80):
                    xm = 0.5 * (x0 + x1)
                    fv = float(self(jnp.asarray(xm))) - yv
                    flo = float(self(jnp.asarray(x0))) - yv
                    if fv * flo <= 0:
                        x1 = xm
                    else:
                        x0 = xm
                out[i] = 0.5 * (x0 + x1)
            return jnp.asarray(out.reshape(_np.shape(y)))

        return Chebfun.from_function(g, Domain((lo, hi)))

    # ------------------------------------------------------------------
    # Logical (indicator) chebfuns -- MATLAB ==, <, <=, ~, &, |
    # (added by Claude Fable 5, MISSING_FEATURES logical-chebfun gap).
    # ------------------------------------------------------------------

    def _indicator(self, other, positive: bool) -> "Chebfun":
        """Indicator chebfun of {self < other} (positive=False) or
        {self > other} (positive=True), built from the exact constant
        pieces of sign(other - self)."""
        diff = (other - self) if not isinstance(other, (int, float))             else (-self + float(other))
        sgn = diff.sign() if not positive else (-diff).sign()
        # map pieces: +1 -> 1, else -> 0 (exact constant pieces)
        new_funs = []
        for piece in sgn.funs:
            val = float(piece(jnp.asarray(
                0.5 * (piece.interval[0] + piece.interval[1]))))
            const = 1.0 if val > 0.5 else 0.0
            new_funs.append(_Piece.from_coeffs(
                jnp.asarray([const], dtype=jnp.float64),
                piece.interval[0], piece.interval[1]))
        return Chebfun(funs=new_funs, domain=sgn.domain)

    def lt(self, other) -> "Chebfun":
        """Indicator chebfun of {f < g} (MATLAB f < g).

        Provenance
        ----------
        MATLAB source : @chebfun/lt.m
        Chebfun commit: 7574c77
        """
        return self._indicator(other, positive=False)

    def gt(self, other) -> "Chebfun":
        """Indicator chebfun of {f > g} (MATLAB f > g)."""
        return self._indicator(other, positive=True)

    le = lt   # measure-zero boundary: same indicator a.e.
    ge = gt

    def logical_eq(self, other) -> "Chebfun":
        """Indicator of {f == g}: 1 if identically equal, else 0 a.e.

        Provenance
        ----------
        MATLAB source : @chebfun/eq.m
        Chebfun commit: 7574c77
        """
        diff = self - other if not isinstance(other, (int, float))             else self - float(other)
        val = 1.0 if bool(diff.iszero()) else 0.0
        a, b = float(self.domain.a), float(self.domain.b)
        return Chebfun(funs=[_Piece.from_coeffs(
            jnp.asarray([val], dtype=jnp.float64), a, b)],
            domain=Domain((a, b)))

    def logical_ne(self, other) -> "Chebfun":
        """Indicator of {f ~= g} a.e. (MATLAB ne)."""
        return 1.0 - self.logical_eq(other)

    def logical_not(self) -> "Chebfun":
        """Indicator of {f == 0} (MATLAB ~f)."""
        return self.logical_eq(0.0)

    def logical_and(self, other) -> "Chebfun":
        """Indicator of {f ~= 0 and g ~= 0} via the product of
        nonzero-indicators (MATLAB &)."""
        return self.logical_ne(0.0) * other.logical_ne(0.0)

    def logical_or(self, other) -> "Chebfun":
        """Indicator of {f ~= 0 or g ~= 0} (MATLAB |)."""
        p = self.logical_ne(0.0) + other.logical_ne(0.0)             - self.logical_and(other)
        return p

    def __lt__(self, other):
        return self.lt(other)

    def __gt__(self, other):
        return self.gt(other)

    def __le__(self, other):
        return self.le(other)

    def __ge__(self, other):
        return self.ge(other)

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
        return self._propagate_point_values(self._sign_core(), jnp.sign)

    def _sign_core(self) -> Chebfun:
        """Root-splitting ``sign(f)`` without pointValues propagation."""
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
        # Each piece is exactly constant: evaluate sign at the interval
        # MIDPOINT and build an explicit constant piece.  Sampling
        # sign(f) across the piece hits the roots at the endpoints
        # (sign(0) = 0) and pollutes the interpolant -- same bug class
        # as floor/ceil/round before their fix.  (Fable 5 audit.)
        new_funs = []
        for sub in new_dom.intervals:
            mid = 0.5 * (sub.a + sub.b)
            const = float(jnp.sign(self(jnp.asarray(mid))))
            new_funs.append(_Piece.from_coeffs(
                jnp.asarray([const], dtype=jnp.float64), sub.a, sub.b))
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
        # (orientation is preserved below via _as_transposed)
        # Differentiating across a jump discontinuity produces a Dirac
        # delta of magnitude equal to the jump (task #9, Opus 4.8).  We
        # attach deltas only for the first derivative of a genuine jump;
        # they are carried through and picked up by sum().
        deltas = ()
        if k == 1 and len(self.funs) > 1:
            dlist = []
            for i in range(len(self.funs) - 1):
                loc = float(self.funs[i].interval[1])
                left = float(self.funs[i](jnp.array(loc)))
                right = float(self.funs[i + 1](jnp.array(loc)))
                jump = right - left
                if abs(jump) > 1e-11 * (abs(left) + abs(right) + 1.0):
                    dlist.append((loc, jump))
            deltas = tuple(dlist)
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain, deltas=deltas),
            self.is_transposed)

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
        if len(self.funs) == 1 and not self.deltas:
            return Chebfun._as_transposed(
                Chebfun(funs=[self.funs[0].cumsum()], domain=self.domain),
                self.is_transposed)

        # Multi-piece: compute antiderivative on each piece, then shift to
        # ensure continuity: F_i(b_i) = F_{i+1}(a_{i+1})
        # (offset is a per-column row for array-valued chebfuns)
        #
        # Dirac deltas carried on this Chebfun (e.g. from diff() across a
        # jump) integrate to Heaviside steps: a delta of magnitude ``m`` at
        # location ``loc`` adds ``m`` to the antiderivative for all x > loc.
        # MATLAB source: @deltafun/cumsum.m (deltas -> jumps in the funPart).
        delta_by_loc = {}
        for loc, mag in self.deltas:
            delta_by_loc[float(loc)] = delta_by_loc.get(float(loc), 0.0) + float(mag)
        new_pieces = []
        offset = None
        for piece in self.funs:
            piece_cs = piece.cumsum()
            # piece_cs has F_piece(a_piece) = 0 by construction
            # Shift by offset to achieve continuity
            if offset is not None and bool(jnp.any(offset != 0)):
                # Add offset as a constant to the antiderivative piece
                new_tech = piece_cs.tech + offset
                piece_cs = _Piece(tech=new_tech, interval=piece_cs.interval)
            new_pieces.append(piece_cs)
            # Update offset: new cumulative value at the right endpoint
            offset = piece_cs.values[-1]
            # A delta sitting on this piece's right breakpoint adds a step
            # (Heaviside jump) to every subsequent piece.
            if delta_by_loc:
                right = float(piece.interval[1])
                for loc, mag in delta_by_loc.items():
                    if abs(loc - right) <= 1e-10 * (abs(right) + 1.0):
                        offset = offset + mag

        return Chebfun._as_transposed(
            Chebfun(funs=new_pieces, domain=self.domain), self.is_transposed)

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
        # Dirac deltas contribute their magnitude to the integral (#9).
        for _loc, _mag in getattr(self, "deltas", ()):
            total = total + jnp.float64(_mag)
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
        f, g = Chebfun._overlap(self, other)
        total = jnp.float64(0.0)
        for pf, pg in zip(f.funs, g.funs):
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
            # MATLAB: [normF, ~] = minandmax(f); max(abs(normF)) — the true
            # extremum via rootfinding on f'. Taking max|values at the
            # Chebyshev nodes| instead misses peaks that fall between nodes
            # (e.g. sin on a shifted domain gave 0.99084 instead of 1.0).
            # Complex chebfuns: minandmax needs an ordered field, so work
            # with |f|^2 (a real chebfun) and take the sqrt of its max —
            # the previous code crashed with `lt on complex128`
            # (Fable 5 audit, bug #5).
            if any(jnp.iscomplexobj(piece.tech.coeffs)
                   for piece in self.funs):
                mag2 = (self.real() * self.real()
                        + self.imag() * self.imag())
                (_, _), (_, m_max) = mag2.minandmax()
                return jnp.sqrt(jnp.array(max(float(m_max), 0.0),
                                          dtype=jnp.float64))
            (_, f_min), (_, f_max) = self.minandmax()
            return jnp.array(
                max(abs(float(f_min)), abs(float(f_max))), dtype=jnp.float64
            )
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

    def roots(self, complex_roots: bool = False) -> jax.Array:
        """All roots of the Chebfun in its domain.

        With ``complex_roots=True`` returns *all* roots of the Chebyshev
        series of each piece (real and complex, mapped to the physical
        interval), like MATLAB's ``roots(f, 'complex')`` — added by
        Claude Opus 4.8 (task #14).

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

        if complex_roots:
            croots = []
            for piece in self.funs:
                a, b = piece.interval
                c = _np.asarray(piece.tech.coeffs, dtype=complex)
                if c.size < 2:
                    continue
                t = _np.polynomial.chebyshev.chebroots(c)
                # map reference [-1,1] -> physical [a, b]
                croots.append(0.5 * (b - a) * t + 0.5 * (a + b))
            if not croots:
                return jnp.array([], dtype=jnp.complex128)
            return jnp.asarray(_np.concatenate(croots), dtype=jnp.complex128)

        domain_len = float(self.domain.b - self.domain.a)
        dedup_tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)

        def _dedup(combined):
            if combined.shape[0] <= 1:
                return combined
            mask = _np.concatenate(
                [[True], _np.diff(combined) > dedup_tol])
            return combined[mask]

        if any(p.tech.coeffs.ndim == 2 for p in self.funs):
            # Array-valued: per-column collection, NaN-padded to equal
            # length (MATLAB @chebfun/roots.m convention).
            m = max(p.tech.coeffs.shape[1] for p in self.funs
                    if p.tech.coeffs.ndim == 2)
            cols = []
            for j in range(m):
                rj = []
                for piece in self.funs:
                    r = _np.asarray(piece.roots())
                    col = r[:, j] if r.ndim == 2 else r
                    rj.append(col[_np.isfinite(col)])
                combined = (_np.sort(_np.concatenate(rj))
                            if rj else _np.zeros(0))
                cols.append(_dedup(combined))
            nmax = max((len(c) for c in cols), default=0)
            out = _np.full((nmax, m), _np.nan)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)

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
        unique_roots = _dedup(combined)
        return jnp.asarray(unique_roots, dtype=jnp.float64)

    def minandmax(self, flag: "str | None" = None):
        """Global minimum and maximum of the Chebfun.

        Searches each piece for its local extrema (critical points of the
        derivative plus piece endpoints), then returns the global min/max.

        With ``flag='local'`` returns *all* local extrema instead of the
        global pair: ``(x, y)`` where ``x`` are the extrema locations (the
        interior critical points where ``f'=0`` together with the two domain
        endpoints, which MATLAB always includes) sorted ascending, and
        ``y = f(x)``.  For an array-valued Chebfun the columns of ``x``/``y``
        correspond to the columns of ``f`` and are padded with ``NaN`` to a
        common length.

        NOT JIT-safe (uses roots of derivative — eigenvalue computation).

        Returns
        -------
        (x_min, f_min), (x_max, f_max) : pair of (location, value) tuples
            (default).
        (x, y) : tuple of arrays
            All local extrema locations and values (``flag='local'``).

        Provenance
        ----------
        MATLAB source : @chebfun/minandmax.m
        Chebfun commit: 7574c77
        """
        if flag is not None:
            if str(flag).lower() != "local":
                raise ValueError(
                    f"minandmax: unknown flag {flag!r} (expected 'local').")
            return self._local_minandmax()
        if self.isempty():
            e = jnp.asarray([], dtype=jnp.float64)
            return (e, e), (e, e)
        if any(p.tech.coeffs.ndim == 2 for p in self.funs):
            # Array-valued: elementwise per-column comparison across
            # pieces (MATLAB returns 2 x m values/positions).
            import numpy as _np

            gmin_x = gmin_v = gmax_x = gmax_v = None
            for piece in self.funs:
                (x_min, f_min), (x_max, f_max) = piece.minandmax()
                x_min, f_min = _np.asarray(x_min), _np.asarray(f_min)
                x_max, f_max = _np.asarray(x_max), _np.asarray(f_max)
                if gmin_v is None:
                    gmin_x, gmin_v = x_min, f_min
                    gmax_x, gmax_v = x_max, f_max
                else:
                    take = f_min < gmin_v
                    gmin_x = _np.where(take, x_min, gmin_x)
                    gmin_v = _np.where(take, f_min, gmin_v)
                    take = f_max > gmax_v
                    gmax_x = _np.where(take, x_max, gmax_x)
                    gmax_v = _np.where(take, f_max, gmax_v)
            return ((jnp.asarray(gmin_x), jnp.asarray(gmin_v)),
                    (jnp.asarray(gmax_x), jnp.asarray(gmax_v)))

        global_min_x = None
        global_min_val = float("inf")
        global_max_x = None
        global_max_val = float("-inf")
        # Complex-valued pieces order by |f| (MATLAB minandmax.m); the
        # returned values stay complex.
        global_min_key = float("inf")
        global_max_key = float("-inf")

        for piece in self.funs:
            (x_min, f_min), (x_max, f_max) = piece.minandmax()
            k_min = abs(f_min) if isinstance(f_min, complex) or \
                jnp.iscomplexobj(jnp.asarray(f_min)) else f_min
            k_max = abs(f_max) if isinstance(f_max, complex) or \
                jnp.iscomplexobj(jnp.asarray(f_max)) else f_max
            if k_min < global_min_key:
                global_min_key = k_min
                global_min_val = f_min
                global_min_x = x_min
            if k_max > global_max_key:
                global_max_key = k_max
                global_max_val = f_max
                global_max_x = x_max

        return (global_min_x, global_min_val), (global_max_x, global_max_val)

    def local_extrema(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """All interior local extrema of the Chebfun.

        Returns ``(x, v, kind)`` where ``x`` are the interior critical
        points (roots of ``f'``), ``v = f(x)`` the values, and ``kind``
        is ``+1`` at local maxima, ``-1`` at local minima, ``0`` at
        inflection/degenerate points (classified by the sign of ``f''``).
        Added by Claude Opus 4.8 (task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/minandmax.m ('local' flag)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        df = self.diff()
        d2f = df.diff()
        a = float(self.domain.a)
        b = float(self.domain.b)
        r = _np.asarray(df.roots())
        r = _np.unique(r[(r > a + 1e-12) & (r < b - 1e-12)])
        if r.size == 0:
            empty = jnp.array([], dtype=jnp.float64)
            return empty, empty, empty
        rj = jnp.asarray(r, dtype=jnp.float64)
        v = _np.asarray(self(rj))
        curv = _np.asarray(d2f(rj))
        kind = _np.where(curv < -1e-10, 1,
                         _np.where(curv > 1e-10, -1, 0))
        return (jnp.asarray(r, dtype=jnp.float64),
                jnp.asarray(v, dtype=jnp.float64),
                jnp.asarray(kind, dtype=jnp.float64))

    def _local_minandmax_scalar(self):
        """All local extrema ``(x, y)`` of a scalar-valued Chebfun.

        Interior critical points (roots of ``f'``) together with the two
        domain endpoints (always included, per MATLAB ``localMinAndMax``),
        sorted ascending, with ``y = f(x)``.
        """
        import numpy as _np

        df = self.diff()
        a = float(self.domain.a)
        b = float(self.domain.b)
        r = _np.asarray(df.roots()).ravel()
        r = _np.real(r[_np.abs(_np.imag(r)) < 1e-12]) \
            if _np.iscomplexobj(r) else r
        r = r[(r > a + 1e-12) & (r < b - 1e-12)]
        r = _np.unique(r)
        x = _np.sort(_np.concatenate([[a], r, [b]]))
        y = _np.asarray(self(jnp.asarray(x, dtype=jnp.float64)))
        return x, y

    def _local_min_or_max_scalar(self, which: str):
        """Local minima (``which='min'``) or maxima of a scalar Chebfun.

        Interior extrema are classified by the sign of ``f''``; endpoints by
        the sign of ``f'`` (falling back to ``f''`` when ``f'`` is negligible
        there), matching MATLAB ``localMin``/``localMax``.
        """
        import numpy as _np

        x, y = self._local_minandmax_scalar()
        df = self.diff()
        d2f = df.diff()
        a = float(self.domain.a)
        b = float(self.domain.b)
        xj = jnp.asarray(x, dtype=jnp.float64)
        d1 = _np.asarray(df(xj)).real
        d2 = _np.asarray(d2f(xj)).real
        dfvs = float(df.vscale)
        eps = float(_np.finfo(_np.float64).eps)
        keep = _np.zeros(len(x), dtype=bool)
        want_min = which == "min"
        for i, xi in enumerate(x):
            small = abs(d1[i]) < 1e3 * dfvs * eps
            if abs(xi - a) <= 1e-12:            # left endpoint
                if want_min:
                    keep[i] = (d2[i] > 0) if small else (d1[i] > 0)
                else:
                    keep[i] = (d2[i] < 0) if small else (d1[i] < 0)
            elif abs(xi - b) <= 1e-12:          # right endpoint
                if want_min:
                    keep[i] = (d2[i] > 0) if small else (d1[i] < 0)
                else:
                    keep[i] = (d2[i] < 0) if small else (d1[i] > 0)
            else:                               # interior
                keep[i] = (d2[i] > 0) if want_min else (d2[i] < 0)
        return x[keep], y[keep]

    def _stack_local_columns(self, per_col):
        """Pad per-column ``(x, y)`` lists to a common length with NaN.

        ``per_col`` is a list of ``(x, y)`` numpy arrays, one per column.
        Returns ``(X, Y)`` of shape ``(maxlen, ncols)`` (or 1-D for a single
        column), matching MATLAB's NaN-padding of ragged local-extrema
        columns.
        """
        import numpy as _np

        if len(per_col) == 1:
            x, y = per_col[0]
            return jnp.asarray(x), jnp.asarray(y)
        maxlen = max(len(x) for x, _ in per_col) if per_col else 0
        ncols = len(per_col)
        X = _np.full((maxlen, ncols), _np.nan)
        Y = _np.full((maxlen, ncols), _np.nan, dtype=complex)
        for j, (x, y) in enumerate(per_col):
            X[: len(x), j] = x
            Y[: len(y), j] = y
        # Collapse to real when no column carried a complex value.
        if _np.all(_np.nan_to_num(Y.imag) == 0.0):
            Y = Y.real
        return jnp.asarray(X), jnp.asarray(Y)

    def _is_array_valued(self) -> bool:
        return any(getattr(p.tech, "coeffs", jnp.zeros(1)).ndim == 2
                   for p in self.funs)

    def _local_minandmax(self):
        """All local extrema; scalar or array-valued (NaN-padded columns)."""
        if not self._is_array_valued():
            return self._stack_local_columns([self._local_minandmax_scalar()])
        cols = [self.extract_columns(j) for j in range(self.n_columns)]
        return self._stack_local_columns(
            [c._local_minandmax_scalar() for c in cols])

    def _local_min_or_max(self, which: str):
        if not self._is_array_valued():
            return self._stack_local_columns(
                [self._local_min_or_max_scalar(which)])
        cols = [self.extract_columns(j) for j in range(self.n_columns)]
        return self._stack_local_columns(
            [c._local_min_or_max_scalar(which) for c in cols])

    def min(self, flag: "str | None" = None):
        """Global minimum ``(x_min, f_min)``, or all local minima.

        With ``flag='local'`` returns ``(x, y)`` of every local minimum
        (interior minima where ``f'' > 0`` plus any endpoint that is a local
        minimum), sorted ascending.

        NOT JIT-safe.

        Returns
        -------
        (x_min, f_min) : tuple of floats (default).
        (x, y) : tuple of arrays (``flag='local'``).

        Provenance
        ----------
        MATLAB source : @chebfun/min.m
        Chebfun commit: 7574c77
        """
        if flag is not None:
            if str(flag).lower() != "local":
                raise ValueError(
                    f"min: unknown flag {flag!r} (expected 'local').")
            return self._local_min_or_max("min")
        (x_min, f_min), _ = self.minandmax()
        return x_min, f_min

    def max(self, flag: "str | None" = None):
        """Global maximum ``(x_max, f_max)``, or all local maxima.

        With ``flag='local'`` returns ``(x, y)`` of every local maximum
        (interior maxima where ``f'' < 0`` plus any endpoint that is a local
        maximum), sorted ascending.

        NOT JIT-safe.

        Returns
        -------
        (x_max, f_max) : tuple of floats (default).
        (x, y) : tuple of arrays (``flag='local'``).

        Provenance
        ----------
        MATLAB source : @chebfun/max.m
        Chebfun commit: 7574c77
        """
        if flag is not None:
            if str(flag).lower() != "local":
                raise ValueError(
                    f"max: unknown flag {flag!r} (expected 'local').")
            return self._local_min_or_max("max")
        _, (x_max, f_max) = self.minandmax()
        return x_max, f_max

    def maximum(self, other) -> Chebfun:
        """Pointwise maximum of two Chebfuns (or a Chebfun and a scalar).

        Introduces breakpoints at the crossing points (roots of
        ``self - other``) so each returned piece is smooth, matching
        MATLAB's ``max(f, g)``.  Added by Claude Opus 4.8 (two-arg
        max/min, task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/max.m (two-argument form)
        Chebfun commit: 7574c77
        """
        return _two_arg_extremum(self, other, jnp.maximum)

    def minimum(self, other) -> Chebfun:
        """Pointwise minimum of two Chebfuns (or a Chebfun and a scalar).

        See :meth:`maximum`.  Added by Claude Opus 4.8 (task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/min.m (two-argument form)
        Chebfun commit: 7574c77
        """
        return _two_arg_extremum(self, other, jnp.minimum)

    def floor(self) -> Chebfun:
        """Pointwise floor, as a piecewise-constant Chebfun.

        Breakpoints are inserted where ``self`` crosses an integer, so
        each piece is constant.  Added by Claude Opus 4.8 (task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/floor.m
        Chebfun commit: 7574c77
        """
        return _integer_step(self, jnp.floor)

    def ceil(self) -> Chebfun:
        """Pointwise ceiling, as a piecewise-constant Chebfun.

        Added by Claude Opus 4.8 (task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/ceil.m
        Chebfun commit: 7574c77
        """
        return _integer_step(self, jnp.ceil)

    def round(self) -> Chebfun:
        """Pointwise round-to-nearest-integer, piecewise-constant Chebfun.

        Added by Claude Opus 4.8 (task #14).

        Provenance
        ----------
        MATLAB source : @chebfun/round.m
        Chebfun commit: 7574c77
        """
        return _integer_step(self, jnp.round, half_offset=True)

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
        if self.isempty():
            return Chebfun.empty()
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

    def diag(self):
        """Multiplication-by-self operator ``D`` with ``D*g == self.*g``
        (MATLAB ``diag(f)``): returns an ``OperatorBlock``.

        Provenance
        ----------
        MATLAB source : @chebfun/diag.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.operators.blocks import diag as _diag
        return _diag(self)

    # ------------------------------------------------------------------
    # V08 — Quasimatrix ops: horzcat, vertcat, size, __getitem__
    # ------------------------------------------------------------------

    @staticmethod
    def horzcat(chebfuns: list[Chebfun]) -> list[Chebfun]:
        """Horizontal concatenation: return a list (quasimatrix column list).

        All Chebfuns must share the same domain endpoints. Returns the input
        list, validating domain compatibility. In Python there is no native
        quasimatrix type; the list-of-Chebfun convention is used here and
        throughout the linalg module.

        Parameters
        ----------
        chebfuns : list[Chebfun]
            Column Chebfuns to concatenate.

        Returns
        -------
        list[Chebfun]
            The same list (quasimatrix representation).

        Raises
        ------
        ValueError
            If domain endpoints differ between any two inputs.

        Provenance
        ----------
        MATLAB source : @chebfun/horzcat.m
        Chebfun commit: 7574c77
        """
        if not chebfuns:
            return []
        a0, b0 = chebfuns[0].domain.a, chebfuns[0].domain.b
        for i, f in enumerate(chebfuns[1:], start=1):
            if abs(f.domain.a - a0) > 100 * _EPS or abs(f.domain.b - b0) > 100 * _EPS:
                raise ValueError(
                    f"horzcat: column {i} has domain [{f.domain.a}, {f.domain.b}] "
                    f"which is inconsistent with [{a0}, {b0}]."
                )
        return list(chebfuns)

    @staticmethod
    def vertcat(chebfuns: list[Chebfun]) -> list[Chebfun]:
        """Vertical concatenation: concatenate Chebfuns by stacking domains.

        Each successive Chebfun is appended after the previous one in the
        x-direction.  The domains must be compatible:
        ``chebfuns[k].domain.b == chebfuns[k+1].domain.a``.

        Parameters
        ----------
        chebfuns : list[Chebfun]
            Row Chebfuns (in domain order) to concatenate.

        Returns
        -------
        Chebfun
            A single piecewise Chebfun on the union domain.

        Raises
        ------
        ValueError
            If successive domains are not contiguous.

        Provenance
        ----------
        MATLAB source : @chebfun/vertcat.m
        Chebfun commit: 7574c77
        """
        if not chebfuns:
            raise ValueError("vertcat: input list is empty.")
        if len(chebfuns) == 1:
            return chebfuns[0]
        # Validate contiguity and collect all pieces
        all_funs: list[_Piece] = []
        all_bps: list[float] = [chebfuns[0].domain.a]
        for k, f in enumerate(chebfuns):
            if k > 0:
                prev_b = all_bps[-1]
                if abs(f.domain.a - prev_b) > 100 * _EPS:
                    raise ValueError(
                        f"vertcat: chebfuns[{k}].domain.a = {f.domain.a} does not "
                        f"match chebfuns[{k-1}].domain.b = {prev_b}."
                    )
            for piece in f.funs:
                all_funs.append(piece)
            # Append internal breakpoints except the first one (already added)
            bps = list(f.domain.breakpoints)
            all_bps.extend(bps[1:])
        new_domain = Domain(tuple(float(x) for x in all_bps))
        return Chebfun(funs=all_funs, domain=new_domain)

    def size(self, dim: int | None = None):
        """Size of the Chebfun (quasimatrix notion).

        A column Chebfun with ``n`` columns has size ``(inf, n)``; a row
        (transposed) Chebfun has size ``(n, inf)``.  A scalar column Chebfun
        is thus ``(inf, 1)``.

        Parameters
        ----------
        dim : int or None
            If 1, return the first dimension; if 2, the second; if None,
            return the ``(d1, d2)`` tuple.

        Returns
        -------
        tuple[float, int] or float or int
            The size in the requested sense.

        Provenance
        ----------
        MATLAB source : @chebfun/size.m
        Chebfun commit: 7574c77
        """
        inf_dim = float("inf")
        n_cols = 0 if self.isempty() else self.n_columns
        # Column: (inf, n_cols); row (transposed): swap the two.
        if self.is_transposed:
            d1, d2 = n_cols, inf_dim
        else:
            d1, d2 = inf_dim, n_cols
        if dim is None:
            return (d1, d2)
        elif dim == 1:
            return d1
        elif dim == 2:
            return d2
        else:
            # Higher dimensions are 1 (no tensor Chebfuns)
            return 1

    def __getitem__(self, idx):
        """Column indexing for quasimatrix-style access.

        For a scalar Chebfun (single column), ``f[0]`` or ``f[:]`` returns
        ``self``.  Slices and integer indices follow standard Python
        conventions: only index 0 is valid for a single-column Chebfun.

        Parameters
        ----------
        idx : int or slice
            Column index.

        Returns
        -------
        Chebfun

        Raises
        ------
        IndexError
            If the index is out of range.

        Provenance
        ----------
        MATLAB source : @chebfun/subsref.m
        Chebfun commit: 7574c77
        """
        n_cols = 1  # scalar Chebfun
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n_cols)
            cols = list(range(start, stop, step))
            if not cols:
                raise IndexError("slice results in empty selection.")
            return self  # single column — return self
        else:
            idx = int(idx)
            if idx < 0:
                idx = n_cols + idx
            if idx != 0:
                raise IndexError(
                    f"index {idx} is out of bounds for a scalar Chebfun (1 column)."
                )
            return self

    # ------------------------------------------------------------------
    # V09 — Interpolation / fitting
    # ------------------------------------------------------------------

    def polyfit(self, n: int) -> Chebfun:
        """Polynomial fit of degree n (least-squares projection).

        Computes the degree-n polynomial that best approximates self in the
        L2 sense on the Chebfun's domain, by truncating the Chebyshev series
        to the first n+1 coefficients.

        For a single-piece Chebfun whose polynomial degree is already <= n,
        the input is returned unchanged.

        Parameters
        ----------
        n : int
            Degree of the approximating polynomial (>= 0).

        Returns
        -------
        Chebfun
            The degree-n least-squares polynomial fit, on the same domain.

        Raises
        ------
        ValueError
            If n is not a non-negative integer.

        Provenance
        ----------
        MATLAB source : @chebfun/polyfit.m
        Chebfun commit: 7574c77
        """
        if not isinstance(n, (int,)) or n < 0:
            raise ValueError(f"polyfit: n must be a non-negative integer, got {n!r}.")
        from chebfunjax.utils.transforms import cheb2leg, leg2cheb

        def _l2_truncate(coeffs):
            """Degree-n L2 best fit: truncate the LEGENDRE series.

            MATLAB @chebfun/polyfit.m truncates legcoeffs(y, n+1) and maps
            back with leg2cheb — the true least-squares projection.
            Truncating the CHEBYSHEV series instead gives a different
            (weighted-L2) polynomial: exp(x) at degree 5 differs by 1.7e-5.
            """
            cleg = cheb2leg(coeffs)[: n + 1]
            return leg2cheb(cleg)

        if len(self.funs) == 1:
            piece = self.funs[0]
            coeffs = piece.coeffs  # Chebyshev coefficients, length = piece.n
            if piece.n <= n + 1:
                # Already degree <= n, nothing to truncate
                return self
            truncated = _l2_truncate(coeffs)
            new_piece = _Piece.from_coeffs(truncated, piece.interval[0], piece.interval[1])
            return Chebfun(funs=[new_piece], domain=self.domain)
        # Multi-piece: fit each piece independently
        new_funs = []
        for piece in self.funs:
            coeffs = piece.coeffs
            if piece.n <= n + 1:
                new_funs.append(piece)
            else:
                truncated = _l2_truncate(coeffs)
                new_funs.append(
                    _Piece.from_coeffs(truncated, piece.interval[0], piece.interval[1])
                )
        return Chebfun(funs=new_funs, domain=self.domain)

    @staticmethod
    def interp1(
        x: jax.Array,
        y: jax.Array,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Polynomial interpolant through data (x, y).

        Builds a Chebfun by constructing the polynomial interpolant through
        the data points ``(x[j], y[j])``.  The interpolation is performed
        using barycentric weights, matching MATLAB Chebfun's ``interp1``
        default ``'poly'`` method.

        Parameters
        ----------
        x : array_like, shape (n,)
            Distinct, sorted interpolation sites.
        y : array_like, shape (n,)
            Function values at the sites.
        domain : (float, float) or None
            Domain for the resulting Chebfun.  Defaults to
            ``(x[0], x[-1])``.

        Returns
        -------
        Chebfun
            The polynomial interpolant on ``domain``.

        Notes
        -----
        Uses barycentric Lagrange interpolation with second-kind barycentric
        weights, which is numerically stable for any node distribution.
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/interp1.m (interp1Poly subfunction)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        # Sort nodes
        order = jnp.argsort(x)
        x = x[order]
        y = y[order]
        xa, xb = float(x[0]), float(x[-1])
        if domain is None:
            domain = (xa, xb)
        dom = Domain(domain)

        # Compute second-kind barycentric weights (Chebyshev-like, safe for
        # arbitrary nodes via the standard alternating-sign formula)
        n = x.shape[0]
        x_np = _np.asarray(x)
        w = _np.ones(n)
        for j in range(n):
            for k in range(n):
                if k != j:
                    w[j] /= (x_np[j] - x_np[k])

        x_ref = jnp.asarray(x_np)
        y_ref = y
        w_ref = jnp.asarray(w)

        def interpolant(z: jax.Array) -> jax.Array:
            """Evaluate barycentric interpolant at points z (column-wise
            for array-valued (n, m) data)."""
            z = jnp.atleast_1d(z)
            # Compute w_j / (z - x_j) for each z, then sum
            diffs = z[:, None] - x_ref[None, :]   # shape (nz, n)
            # Handle exact hits (z == x_j)
            hit = jnp.abs(diffs) < 1e-14
            safe_diffs = jnp.where(hit, jnp.ones_like(diffs), diffs)
            terms = w_ref[None, :] / safe_diffs    # shape (nz, n)
            numer = terms @ y_ref                  # (nz,) or (nz, m)
            denom = jnp.sum(terms, axis=1)         # (nz,)
            any_hit = jnp.any(hit, axis=1)
            hit_idx = jnp.argmax(hit, axis=1)
            hit_val = y_ref[hit_idx]               # (nz,) or (nz, m)
            if y_ref.ndim == 2:
                return jnp.where(any_hit[:, None], hit_val,
                                 numer / denom[:, None])
            return jnp.where(any_hit, hit_val, numer / denom)

        return Chebfun.from_function(interpolant, dom)

    @staticmethod
    def spline(
        x: jax.Array,
        y: jax.Array,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Piecewise cubic spline interpolant (not-a-knot conditions).

        Wraps ``scipy.interpolate.CubicSpline`` to construct a piecewise
        cubic Chebfun through the data ``(x, y)``.  The domain is
        partitioned at the knot sites ``x``.

        Parameters
        ----------
        x : array_like, shape (n,)
            Sorted interpolation sites (knots).
        y : array_like, shape (n,)
            Function values at the knots.
        domain : (float, float) or None
            Domain for the result; defaults to ``(x[0], x[-1])``.

        Returns
        -------
        Chebfun
            Piecewise cubic Chebfun interpolant.

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/spline.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        from scipy.interpolate import CubicSpline
        x_np = _np.asarray(x, dtype=_np.float64)
        y_np = _np.asarray(y, dtype=_np.float64)
        order = _np.argsort(x_np)
        x_np = x_np[order]
        y_np = y_np[order]
        if domain is None:
            domain = (float(x_np[0]), float(x_np[-1]))
        cs = CubicSpline(x_np, y_np, bc_type="not-a-knot")
        # Use x nodes as breakpoints (unique, sorted)
        breakpoints = _np.unique(_np.concatenate([[domain[0]], x_np, [domain[1]]]))
        dom = Domain(tuple(float(b) for b in breakpoints))
        return Chebfun.from_function(lambda z: jnp.asarray(cs(jnp.asarray(z)), dtype=jnp.float64), dom)

    @staticmethod
    def pchip(
        x: jax.Array,
        y: jax.Array,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Piecewise cubic Hermite interpolant (shape-preserving).

        Wraps ``scipy.interpolate.PchipInterpolator`` to build a
        shape-preserving piecewise cubic Chebfun.

        Parameters
        ----------
        x : array_like, shape (n,)
            Sorted knot sites.
        y : array_like, shape (n,)
            Function values at knots.
        domain : (float, float) or None
            Domain for the result; defaults to ``(x[0], x[-1])``.

        Returns
        -------
        Chebfun
            Shape-preserving piecewise cubic interpolant.

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/pchip.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        from scipy.interpolate import PchipInterpolator
        x_np = _np.asarray(x, dtype=_np.float64)
        y_np = _np.asarray(y, dtype=_np.float64)
        order = _np.argsort(x_np)
        x_np = x_np[order]
        y_np = y_np[order]
        if domain is None:
            domain = (float(x_np[0]), float(x_np[-1]))
        ph = PchipInterpolator(x_np, y_np)
        breakpoints = _np.unique(_np.concatenate([[domain[0]], x_np, [domain[1]]]))
        dom = Domain(tuple(float(b) for b in breakpoints))
        return Chebfun.from_function(lambda z: jnp.asarray(ph(jnp.asarray(z)), dtype=jnp.float64), dom)

    # ------------------------------------------------------------------
    # V10 — Convolution, flip
    # ------------------------------------------------------------------

    def conv(self, g: Chebfun) -> Chebfun:
        r"""Convolution of two Chebfuns.

        Computes

        .. math::
            h(x) = \int f(t)\, g(x - t)\, dt,
            \quad x \in [a+c,\, b+d],

        where ``self`` is on ``[a, b]`` and ``g`` is on ``[c, d]``.

        The convolution is computed by numerical quadrature on each pair of
        sub-intervals, then summed up.  This is the "brute force" / ``'old'``
        algorithm from MATLAB Chebfun, which works for all piecewise-smooth
        functions (not only single-piece Chebyshev expansions).

        Parameters
        ----------
        g : Chebfun
            The second operand.  Must be on a bounded domain.

        Returns
        -------
        Chebfun
            Convolution h = f * g on [a+c, b+d].

        Raises
        ------
        ValueError
            If either domain is unbounded.

        Notes
        -----
        NOT JIT-safe (uses adaptive quadrature and Chebfun construction).

        Provenance
        ----------
        MATLAB source : @chebfun/conv.m (oldConv subfunction)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        f = self
        a, b = float(f.domain.a), float(f.domain.b)
        c, d = float(g.domain.a), float(g.domain.b)
        if not all(_np.isfinite([a, b, c, d])):
            raise ValueError("conv: only bounded domains are supported.")

        # All pairwise sums of breakpoints give the convolution breakpoints
        f_bps = _np.array(list(f.domain.breakpoints))
        g_bps = _np.array(list(g.domain.breakpoints))
        A, B = _np.meshgrid(f_bps, g_bps)
        dom_pts = _np.unique(A.ravel() + B.ravel())
        # Remove near-duplicate breakpoints
        tol = 10.0 * _np.finfo(_np.float64).eps * max(abs(dom_pts[[0, -1]]))
        if tol == 0:
            tol = 1e-14
        keep = _np.concatenate([[True], _np.diff(dom_pts) > tol])
        dom_pts = dom_pts[keep]

        def _conv_at(x_val: float) -> float:
            """Evaluate convolution integral at a single point x."""
            from scipy import integrate as _scint
            A_lim = max(a, x_val - d)
            B_lim = min(b, x_val - c)
            if A_lim >= B_lim:
                return 0.0
            # Build integration sub-intervals from breakpoints of f and g
            ends_g = x_val - g_bps  # maps g breakpoints to t-space
            int_bps = _np.union1d(f_bps, ends_g)
            int_bps = int_bps[(int_bps >= A_lim) & (int_bps <= B_lim)]
            sub_dom = _np.unique(_np.concatenate([[A_lim], int_bps, [B_lim]]))
            result = 0.0
            x_jax = jnp.float64(x_val)
            for j in range(len(sub_dom) - 1):
                def integrand(t):
                    t_arr = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64))
                    ft = f(t_arr)
                    gt = g(x_jax - t_arr)
                    return float((ft * gt)[0])
                val, _ = _scint.quad(integrand, sub_dom[j], sub_dom[j + 1],
                                     epsabs=1e-13, epsrel=1e-13, limit=100)
                result += val
            return result

        conv_dom = Domain((float(dom_pts[0]), float(dom_pts[-1])))
        if len(dom_pts) > 2:
            conv_dom = Domain(tuple(float(p) for p in dom_pts))

        h = Chebfun.from_function(
            lambda x: jnp.array(
                [_conv_at(float(xi)) for xi in jnp.atleast_1d(x)],
                dtype=jnp.float64,
            ),
            conv_dom,
        )
        return h

    def circconv(self, g: Chebfun) -> Chebfun:
        """Circular convolution of two Chebfuns on a shared domain.

        Computes the circular convolution using the DFT trick:
        evaluating both functions on an equi-spaced grid, multiplying their
        DFTs, then building a new Chebfun from the result.

        Parameters
        ----------
        g : Chebfun
            Must be on the same domain as ``self``.

        Returns
        -------
        Chebfun
            Circular convolution on the shared domain.

        Raises
        ------
        ValueError
            If domains do not match.

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/circconv.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or _is_empty_operand(g):
            return Chebfun.empty()
        Chebfun._check_domains(self, g)
        a, b = float(self.domain.a), float(self.domain.b)
        L = b - a
        # Fine equi-spaced grid for the DFT-based circular convolution
        n = max(len(self), len(g)) * 2 + 1
        import math
        n = 2 ** math.ceil(math.log2(n + 1))
        x = jnp.linspace(jnp.float64(a), jnp.float64(b), n, endpoint=False)
        dx = L / n
        fv = self(x)
        # h(x_j) = dx * sum_k f(x_k) g((j-k) dx), so g must be sampled
        # at s_m = m*dx (wrapped periodically into [a, b)), NOT at
        # x_m = a + m*dx.  The previous code sampled at x_m, shifting
        # the result by exactly 'a' (half the period on symmetric
        # domains).  (Fable 5 audit, bug #3a.)
        import numpy as _np
        m = _np.arange(n)
        s = a + _np.mod(m * dx - a, L)
        gv = g(jnp.asarray(s))
        h_vals = jnp.real(
            jnp.fft.ifft(jnp.fft.fft(fv) * jnp.fft.fft(gv))) * dx
        # The convolution of periodic functions is periodic: rebuild as
        # a Fourier series (the previous global-polynomial interp1
        # through equi-spaced points Runge-diverged to NaN on wide
        # domains -- bug #3b).
        c = _np.fft.fft(_np.asarray(h_vals)) / n
        cpos = c[: n // 2]

        def h_eval(t):
            tau = 2.0 * _np.pi * (jnp.asarray(t) - a) / L
            out = jnp.zeros_like(jnp.asarray(t, dtype=jnp.float64))
            # k = 0 and positive k (conjugate symmetry doubles k > 0)
            out = out + jnp.real(jnp.asarray(cpos[0]))
            for k in range(1, n // 2):
                ck = complex(cpos[k])
                out = out + 2.0 * (ck.real * jnp.cos(k * tau)
                                   - ck.imag * jnp.sin(k * tau))
            # Nyquist term (n even)
            cN = complex(c[n // 2])
            out = out + cN.real * jnp.cos((n // 2) * tau)
            return out

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        return _cf(h_eval, domain=(a, b), trig=True)

    def flipud(self) -> Chebfun:
        """Reverse the Chebfun: ``g(x) = f(a + b - x)``.

        Returns a new Chebfun on the same domain ``[a, b]`` satisfying
        ``g(x) = f(a + b - x)``, i.e., the function reflected about the
        mid-point of the domain.

        Returns
        -------
        Chebfun

        Provenance
        ----------
        MATLAB source : @chebfun/flipud.m
        Chebfun commit: 7574c77
        """
        a, b = float(self.domain.a), float(self.domain.b)
        mid = a + b  # a + b - x maps [a,b] -> [a,b]
        # Reverse the order of pieces and flip each piece's interval
        f = self  # capture for closure
        new_funs = []
        # Reversed piece intervals
        for piece in reversed(self.funs):
            pa, pb = piece.interval
            new_a = mid - pb
            new_b = mid - pa
            new_funs.append(
                _Piece.from_function(
                    lambda x, _f=f, _m=mid: _f(_m - x),
                    new_a,
                    new_b,
                )
            )
        # Rebuild domain from reversed pieces
        bps = tuple(new_funs[0].interval[0:1])
        for p in new_funs:
            bps = bps + (p.interval[1],)
        new_domain = Domain(bps)
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=new_domain), self.is_transposed)

    def fliplr(self) -> Chebfun:
        """Flip/reverse a Chebfun.

        For an array-valued COLUMN Chebfun (``~isTransposed``), this reverses
        the order of the columns -- the identity for a scalar (single-column)
        Chebfun.  For a ROW (transposed) Chebfun, ``fliplr`` reflects the
        function about the domain mid-point, i.e. ``columnFliplr`` computes
        ``flipud(f.').'`` so that ``g(x) = f(a + b - x)``.

        (An earlier port aliased the column branch to flipud, which is the
        row-chebfun branch; fixed by Claude Fable 5, Big-Three array-valued
        epic.  The row branch was added with the transpose feature.)

        Returns
        -------
        Chebfun

        Provenance
        ----------
        MATLAB source : @chebfun/fliplr.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return self
        # Row (transposed) chebfun: columnFliplr == flipud(f.').'
        # (reflect about the domain mid-point, preserving row orientation).
        if self.is_transposed:
            return self.transpose().flipud().transpose()
        # Column chebfun: reverse the column order of each piece
        # (the identity for a scalar single-column chebfun).
        new_funs = [
            _Piece(tech=piece.tech.fliplr(), interval=piece.interval)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    # ------------------------------------------------------------------
    # Array-valued column manipulation (Fable 5, Big-Three epic)
    # ------------------------------------------------------------------

    @property
    def n_columns(self) -> int:
        """Number of columns (1 for a scalar-valued Chebfun).

        Provenance
        ----------
        MATLAB source : @chebfun/numColumns.m
        Chebfun commit: 7574c77
        """
        c = self.funs[0].tech.coeffs
        return int(c.shape[1]) if c.ndim == 2 else 1

    @staticmethod
    def _tech_with_coeffs(tech, coeffs):
        """Rebuild a tech of the same class around new coefficients."""
        kwargs = {"coeffs": coeffs, "ishappy": tech.ishappy}
        if hasattr(tech, "is_real"):
            kwargs["is_real"] = tech.is_real
        return type(tech)(**kwargs)

    def extract_columns(self, cols) -> "Chebfun":
        """Return the sub-Chebfun made of the 0-based columns ``cols``
        (MATLAB extractColumns / ``f(:, cols)``); a single index gives
        a scalar-valued Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/extractColumns.m
        Chebfun commit: 7574c77
        """
        single = isinstance(cols, int)
        idx = [cols] if single else list(cols)
        new_funs = []
        for piece in self.funs:
            t = piece.tech
            c = t.coeffs if t.coeffs.ndim == 2 else t.coeffs[:, None]
            block = c[:, jnp.asarray(idx)]
            if single:
                block = block[:, 0]
            new_funs.append(piece.with_tech(
                self._tech_with_coeffs(t, block)))
        return Chebfun(funs=new_funs, domain=self.domain)

    def assign_columns(self, cols, g) -> "Chebfun":
        """Overwrite the 0-based columns ``cols`` with the columns of
        ``g`` (MATLAB assignColumns); ``g=None`` deletes them.  ``g``
        must share this Chebfun's breakpoints.

        Provenance
        ----------
        MATLAB source : @chebfun/assignColumns.m
        Chebfun commit: 7574c77
        """
        if g is not None and tuple(g.domain.breakpoints) != tuple(
                self.domain.breakpoints):
            raise ValueError(
                "assign_columns requires matching breakpoints")
        new_funs = []
        for k, piece in enumerate(self.funs):
            gt = None if g is None else g.funs[k].tech
            new_funs.append(piece.with_tech(
                piece.tech.assign_columns(cols, gt)))
        return Chebfun(funs=new_funs, domain=self.domain)

    def mat2cell(self, sizes=None) -> list:
        """Split an array-valued Chebfun by column counts (MATLAB
        ``mat2cell(f, 1, sizes)``).  An empty Chebfun returns a single-cell
        list holding the empty Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/mat2cell.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return [Chebfun.empty()]
        # For a row (transposed) chebfun the split is along the rows; operate
        # on the underlying columns and re-tag every cell as a row chebfun so
        # each cell's size is (size(k), inf) rather than (inf, size(k)).
        row = self.is_transposed
        base = self.transpose() if row else self
        if sizes is None:
            # MATLAB mat2cell(F): split into single components (ones vector).
            sizes = [1] * base.n_columns
        out = []
        j = 0
        for s in sizes:
            cols = j if s == 1 else list(range(j, j + s))
            cell = base.extract_columns(cols)
            out.append(Chebfun._as_transposed(cell, row))
            j += s
        return out

    def repmat(self, k: int) -> "Chebfun":
        """Horizontally tile the columns ``k`` times (MATLAB
        ``repmat(f, 1, k)``).

        Provenance
        ----------
        MATLAB source : @chebfun/repmat.m
        Chebfun commit: 7574c77
        """
        new_funs = []
        for piece in self.funs:
            t = piece.tech
            c = t.coeffs if t.coeffs.ndim == 2 else t.coeffs[:, None]
            tiled = jnp.tile(c, (1, k))
            new_funs.append(piece.with_tech(
                self._tech_with_coeffs(t, tiled)))
        return Chebfun(funs=new_funs, domain=self.domain)

    # ------------------------------------------------------------------
    # V11 — Special functions: Bessel, Airy, elliptic, erf family
    # ------------------------------------------------------------------

    def besselj(self, nu: float) -> Chebfun:
        r"""Bessel function of the first kind :math:`J_\nu(f(x))`.

        Parameters
        ----------
        nu : float
            Order (real).

        Returns
        -------
        Chebfun
            Approximation to :math:`J_\nu(f(x))` on the same domain.

        Notes
        -----
        Uses ``jax.scipy.special.bessel_jn`` when ``nu`` is a non-negative
        integer, and falls back to ``scipy.special.jv`` otherwise.
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/besselj.m
        Chebfun commit: 7574c77
        """
        try:
            # jax.scipy.special.bessel_jn requires non-negative integer order
            n_int = int(round(nu))
            if abs(nu - n_int) < 1e-12 and n_int >= 0:
                return self._apply_fun(
                    lambda x: jax.scipy.special.bessel_jn(x, n=n_int, maxiter=100)[-1]
                    if hasattr(jax.scipy.special, "bessel_jn")
                    else jnp.asarray(
                        __import__("scipy").special.jv(nu, jnp.asarray(x)),
                        dtype=jnp.float64,
                    )
                )
        except Exception:
            pass
        import scipy.special as _ss
        return self._apply_fun(
            lambda x: jnp.asarray(_ss.jv(nu, jnp.asarray(x)), dtype=jnp.float64)
        )

    def bessely(self, nu: float) -> Chebfun:
        r"""Bessel function of the second kind :math:`Y_\nu(f(x))`.

        Parameters
        ----------
        nu : float
            Order (real).

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``scipy.special.yv``.  NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/bessely.m
        Chebfun commit: 7574c77
        """
        import scipy.special as _ss
        return self._apply_fun(
            lambda x: jnp.asarray(_ss.yv(nu, jnp.asarray(x)), dtype=jnp.float64)
        )

    def airy(self, k: int = 0) -> Chebfun:
        """Airy function :math:`\\mathrm{Ai}` or :math:`\\mathrm{Bi}` of the Chebfun.

        Parameters
        ----------
        k : int
            Which Airy function:
            - 0 : :math:`\\mathrm{Ai}(f(x))`
            - 1 : :math:`\\mathrm{Ai}'(f(x))`
            - 2 : :math:`\\mathrm{Bi}(f(x))`
            - 3 : :math:`\\mathrm{Bi}'(f(x))`

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/airy.m
        Chebfun commit: 7574c77
        """
        import scipy.special as _ss
        return self._apply_fun(
            lambda x: jnp.asarray(_ss.airy(jnp.asarray(x))[k], dtype=jnp.float64)
        )

    def besselh(self, nu: float, k: int = 1, *, scale: int = 0) -> "tuple[Chebfun, Chebfun]":
        r"""Hankel (Bessel of the third kind) function :math:`H^{(k)}_\nu(f(x))`.

        Because jaxchebfun uses real float64 storage, the complex Hankel
        function is returned as a *pair* ``(H_re, H_im)`` of real Chebfuns
        representing the real and imaginary parts respectively.  This follows
        the relationship :math:`H^{(1)}_\nu = J_\nu + i Y_\nu` and
        :math:`H^{(2)}_\nu = J_\nu - i Y_\nu`.

        Parameters
        ----------
        nu : float
            Order (real).
        k : int, default 1
            Which Hankel function: 1 for :math:`H^{(1)}_\nu`, 2 for
            :math:`H^{(2)}_\nu`.
        scale : int, default 0
            Scaling flag (reserved; currently ignored for the real/imag split).

        Returns
        -------
        H_re : Chebfun
            Real part of :math:`H^{(k)}_\nu(f(x))` — equals
            :math:`J_\nu(f(x))`.
        H_im : Chebfun
            Imaginary part of :math:`H^{(k)}_\nu(f(x))` — equals
            :math:`\pm Y_\nu(f(x))` (``+`` for k=1, ``-`` for k=2).

        Raises
        ------
        ValueError
            If ``k`` is not 1 or 2.
        ValueError
            If the Chebfun passes through zero.

        Notes
        -----
        Uses ``scipy.special.jv`` / ``scipy.special.yv``.
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/besselh.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        besselj, bessely, besselk
        """
        # uses-numpy: scipy.special.jv / yv use NumPy arrays
        import numpy as _np
        import scipy.special as _ss

        if k not in (1, 2):
            raise ValueError("besselh: k must be 1 or 2.")

        # Check for zeros in the domain (Hankel undefined at origin)
        r = self.roots()
        if r.shape[0] > 0:
            raise ValueError(
                "besselh: the Chebfun passes through zero in its domain; "
                "Hankel functions are undefined at the origin."
            )

        # H^(1)_nu = J_nu + i Y_nu,  H^(2)_nu = J_nu - i Y_nu
        # Real part is always J_nu
        H_re = self._apply_fun(
            lambda x: jnp.asarray(_ss.jv(nu, _np.asarray(x)), dtype=jnp.float64)
        )
        # Imaginary part: +Y for k=1, -Y for k=2
        sign = 1.0 if k == 1 else -1.0
        H_im = self._apply_fun(
            lambda x: jnp.asarray(sign * _ss.yv(nu, _np.asarray(x)), dtype=jnp.float64)
        )
        return H_re, H_im

    def besselk(self, nu: float, *, scale: int = 0) -> "Chebfun":
        r"""Modified Bessel function of the second kind :math:`K_\nu(f(x))`.

        Parameters
        ----------
        nu : float
            Order (real).
        scale : int, default 0
            Scaling flag.  ``scale=1`` multiplies the result by ``exp(f)``.

        Returns
        -------
        Chebfun
            Chebfun approximating :math:`K_\nu(f(x))`.

        Notes
        -----
        Uses ``scipy.special.kv``.
        Raises ``ValueError`` if the Chebfun passes through zero.
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/besselk.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        besselj, bessely, besselh
        """
        # uses-numpy: scipy.special.kv uses NumPy arrays
        import numpy as _np
        import scipy.special as _ss

        r = self.roots()
        if r.shape[0] > 0:
            raise ValueError(
                "besselk: the Chebfun passes through zero; K_nu is undefined at x=0."
            )

        result = self._apply_fun(
            lambda x: jnp.asarray(_ss.kv(nu, _np.asarray(x)), dtype=jnp.float64)
        )

        if scale == 1:
            a, b = self.domain.a, self.domain.b
            scl = self._apply_fun(lambda x: jnp.exp(x))
            return chebfun(lambda x: result(x) * scl(x), domain=(a, b))

        return result

    def ellipke(self) -> "tuple[Chebfun, Chebfun]":
        r"""Complete elliptic integrals K(m) and E(m) of the Chebfun.

        Computes the complete elliptic integral of the first kind K(m) and
        the second kind E(m) where m is the Chebfun representing the
        parameter.  The parameter m must satisfy :math:`0 \le m \le 1`.

        Returns
        -------
        K : Chebfun
            First complete elliptic integral :math:`K(f(x))`.
        E : Chebfun
            Second complete elliptic integral :math:`E(f(x))`.

        Notes
        -----
        Uses ``scipy.special.ellipk`` and ``scipy.special.ellipe``.
        Values of ``self`` outside [0, 1] will produce NaN.
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/ellipke.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.ellipj
        """
        # uses-numpy: scipy.special.ellipk/ellipe use NumPy arrays
        import numpy as _np
        import scipy.special as _ss

        K = self._apply_fun(
            lambda x: jnp.asarray(_ss.ellipk(_np.asarray(x)), dtype=jnp.float64)
        )
        E = self._apply_fun(
            lambda x: jnp.asarray(_ss.ellipe(_np.asarray(x)), dtype=jnp.float64)
        )
        return K, E

    def dirac(self) -> "Chebfun":
        r"""Dirac delta distribution centred at the roots of the Chebfun.

        Returns a Chebfun whose (distributional) value is a sum of Dirac
        deltas placed at each simple zero :math:`r_i` of ``self``, with
        weights :math:`1 / |f'(r_i)|`.  Interior deltas are stored as
        impulse coefficients in the piece containing the root; boundary
        roots get half-weight.

        Returns
        -------
        Chebfun
            A zero Chebfun with delta-impulse metadata at each root.

        Raises
        ------
        ValueError
            If the Chebfun has a non-simple zero.

        Notes
        -----
        This is a *distributional* representation.  The returned object
        supports ``sum()`` (integration), which recovers the correct weight.
        The Chebfun is otherwise zero everywhere except at the delta locations.
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/dirac.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.heaviside
        """
        # uses-numpy: rootfinding and evaluation use NumPy arrays
        import numpy as _np

        a = float(self.domain.a)
        b = float(self.domain.b)

        # Find all interior roots
        r_jax = self.roots()
        r = _np.sort(_np.asarray(r_jax, dtype=_np.float64))

        # Compute derivative for checking simple-root condition and weights
        fp = self.diff()
        tol = 100.0 * _EPS * max(float(self.vscale), 1.0)

        # Start with a zero Chebfun on the same domain
        result = chebfun(lambda x: jnp.zeros_like(x, dtype=jnp.float64),
                         domain=(a, b))

        if r.shape[0] == 0:
            object.__setattr__(result, "_delta_locs", [])
            object.__setattr__(result, "_delta_weights", [])
            return result

        fpvals = _np.asarray(fp(jnp.array(r, dtype=jnp.float64)), dtype=_np.float64)
        if _np.any(_np.abs(fpvals) < tol):
            raise ValueError(
                "dirac: the Chebfun has a non-simple zero; "
                "Dirac delta is not defined in this case."
            )

        # Each delta has weight 1/|f'(r_i)|
        weights = 1.0 / _np.abs(fpvals)

        # Store delta metadata as a list of (location, weight) on the result.
        # Chebfun is a frozen equinox module so we use object.__setattr__ to
        # attach distributional metadata outside the pytree.
        object.__setattr__(result, "_delta_locs", r.tolist())
        object.__setattr__(result, "_delta_weights", weights.tolist())
        return result

    def unwrap(self, jump_tol: float | None = None) -> "Chebfun":
        """Phase-unwrap a real Chebfun by removing jumps of 2*pi.

        Adjusts each piece after the first by adding a multiple of
        ``2 * jump_tol`` (default: ``pi``) to remove discontinuities at
        breakpoints that are multiples of ``2 * jump_tol``.

        Parameters
        ----------
        jump_tol : float or None
            Jump tolerance in radians.  Absolute jumps at breakpoints that
            are within ``jump_tol`` of a multiple of ``2*jump_tol`` are
            unwrapped.  Default: ``pi`` (standard phase unwrap).

        Returns
        -------
        Chebfun
            Unwrapped version of the Chebfun.

        Notes
        -----
        For smooth single-piece Chebfuns this is a no-op.
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/unwrap.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.angle

        Examples
        --------
        >>> import jax.numpy as jnp, numpy as np
        >>> from chebfunjax.chebfun1d.chebfun import chebfun
        >>> # A smooth phase – unwrap should leave it unchanged
        >>> f = chebfun(lambda x: x * 2.0 * float(jnp.pi))
        >>> g = f.unwrap()
        >>> xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        >>> np.testing.assert_allclose(
        ...     np.array(f(xs)), np.array(g(xs)), atol=1e-12)
        """
        # uses-numpy: breakpoint evaluation uses NumPy
        import numpy as _np

        if jump_tol is None:
            jump_tol = float(_np.pi)

        # Single-piece: nothing to unwrap
        if len(self.funs) == 1:
            return self

        # Evaluate left- and right-limits at each internal breakpoint
        bps = list(self.domain.breakpoints)
        n_pieces = len(self.funs)

        # Right-limit value of piece j at breakpoint j+1 (ascending order)
        # Left-limit value of piece j+1 at the same breakpoint
        rvals = _np.array([float(self.funs[j](jnp.float64(bps[j + 1])))
                           for j in range(n_pieces - 1)])
        lvals = _np.array([float(self.funs[j + 1](jnp.float64(bps[j + 1])))
                           for j in range(n_pieces - 1)])

        jumps = lvals - rvals  # raw jump at each internal breakpoint
        two_jump = 2.0 * jump_tol

        # Cumulative shift to apply to each piece after the first
        shifts = _np.zeros(n_pieces)
        for j in range(n_pieces - 1):
            # Nearest multiple of two_jump to the jump
            k = _np.round(jumps[j] / two_jump)
            shifts[j + 1] = shifts[j] - k * two_jump

        if _np.all(shifts == 0.0):
            return self  # nothing to do

        # Build shifted pieces
        new_funs = []
        for j, piece in enumerate(self.funs):
            s = float(shifts[j])
            if s == 0.0:
                new_funs.append(piece)
            else:
                a_p, b_p = piece.interval
                new_funs.append(
                    _Piece.from_function(
                        lambda x, _p=piece, _s=s: _p(x) + _s,
                        a_p, b_p,
                    )
                )

        return Chebfun(funs=new_funs, domain=self.domain)

    def iszero(self) -> bool:
        """True if the Chebfun is identically zero (within tolerance).

        Returns
        -------
        bool
            True if ``vscale < eps``, meaning the function is indistinguishable
            from the zero function at machine precision.

        Examples
        --------
        >>> from chebfunjax.chebfun1d.chebfun import chebfun
        >>> import jax.numpy as jnp
        >>> chebfun(lambda x: jnp.zeros_like(x)).iszero()
        True
        >>> chebfun(lambda x: jnp.ones_like(x)).iszero()
        False

        Provenance
        ----------
        MATLAB source : @chebfun/iszero.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return True
        return float(self.vscale) < _EPS

    # ------ innerProduct alias -----------------------------------------------

    def innerProduct(self, other: "Chebfun") -> "jax.Array":
        r"""L2 inner product alias for :meth:`inner`.

        ``innerProduct(f, g)`` computes :math:`\int_a^b f(x)\,g(x)\,dx`.

        Parameters
        ----------
        other : Chebfun

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebfun/innerProduct.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.inner, Chebfun.norm
        """
        return self.inner(other)

    def ellipj(self, m: float) -> tuple[Chebfun, Chebfun, Chebfun]:
        """Jacobi elliptic functions sn, cn, dn of the Chebfun.

        Parameters
        ----------
        m : float
            Parameter (0 <= m <= 1).

        Returns
        -------
        (sn, cn, dn) : tuple of three Chebfuns
            The three Jacobi elliptic functions of ``self`` with parameter m.

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/ellipj.m
        Chebfun commit: 7574c77
        """
        import scipy.special as _ss
        sn = self._apply_fun(
            lambda x: jnp.asarray(_ss.ellipj(jnp.asarray(x), m)[0], dtype=jnp.float64)
        )
        cn = self._apply_fun(
            lambda x: jnp.asarray(_ss.ellipj(jnp.asarray(x), m)[1], dtype=jnp.float64)
        )
        dn = self._apply_fun(
            lambda x: jnp.asarray(_ss.ellipj(jnp.asarray(x), m)[2], dtype=jnp.float64)
        )
        return sn, cn, dn

    def erf(self) -> Chebfun:
        """Error function :math:`\\mathrm{erf}(f(x))`.

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``jax.scipy.special.erf``.  NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/erf.m
        Chebfun commit: 7574c77
        """
        return self._apply_fun(jax.scipy.special.erf)

    def erfc(self) -> Chebfun:
        """Complementary error function :math:`\\mathrm{erfc}(f(x))`.

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``jax.scipy.special.erfc``.  NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/erfc.m
        Chebfun commit: 7574c77
        """
        return self._apply_fun(jax.scipy.special.erfc)

    def erfinv(self) -> Chebfun:
        r"""Inverse error function :math:`\\mathrm{erf}^{-1}(f(x))`.

        Parameters
        ----------
        (none)

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``jax.scipy.special.erfinv``.  Values of ``f`` must lie in
        (-1, 1). NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/erfinv.m
        Chebfun commit: 7574c77
        """
        return self._apply_fun(jax.scipy.special.erfinv)

    def gamma(self) -> Chebfun:
        r"""Gamma function :math:`\Gamma(f(x))`.

        Computes the composition of the gamma function with ``f``.  For
        example, a Chebfun of the gamma function on ``[0.1, 3]`` is

        >>> import chebfunjax as cj
        >>> x = cj.chebfun(lambda t: t, domain=[0.1, 3.0])
        >>> g = x.gamma()

        This does not introduce poles: the range of ``f`` must avoid the
        non-positive integers where :math:`\Gamma` is singular.  (To get a
        Chebfun with poles, construct ``gamma`` directly with the
        ``splitting``/``blowup`` options.)

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``jax.scipy.special.gamma``.  NOT JIT-safe (adaptive
        construction).

        Provenance
        ----------
        MATLAB source : @chebfun/gamma.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.exp, Chebfun.log
        """
        return self._apply_fun(jax.scipy.special.gamma)

    # ------------------------------------------------------------------
    # V12 — Type / logical ops
    # ------------------------------------------------------------------

    def isnan(self) -> bool:
        """True if any coefficient of any piece is NaN.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isnan.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import Singfun
        for piece in self.funs:
            tech = piece.tech
            coeffs = tech.smoothPart.coeffs if isinstance(tech, Singfun) \
                else tech.coeffs
            if bool(jnp.any(jnp.isnan(coeffs))):
                return True
        return False

    def isfinite(self) -> bool:
        """True if the Chebfun is bounded everywhere.

        A piece backed by a :class:`Singfun` with any negative endpoint
        exponent is unbounded (a pole/blowup), so the Chebfun is not
        finite.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isfinite.m, @singfun/isfinite.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import _EXP_TOL, Singfun
        for piece in self.funs:
            tech = piece.tech
            if isinstance(tech, Singfun):
                if any(e < -_EXP_TOL for e in tech.exponents):
                    return False
                coeffs = tech.smoothPart.coeffs
            else:
                coeffs = tech.coeffs
            if bool(jnp.any(~jnp.isfinite(coeffs))):
                return False
        return True

    def isinf(self) -> bool:
        """True if the Chebfun has any infinite values.

        The negation of :meth:`isfinite`: a Singfun piece with a negative
        endpoint exponent (a pole/blowup) makes the Chebfun infinite.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isinf.m
        Chebfun commit: 7574c77
        """
        return not self.isfinite()

    def isreal(self) -> bool:
        """True if all coefficients are real-valued (no imaginary part).

        In jaxchebfun all Chebfuns use float64 storage, so this always
        returns True for the standard scalar Chebfun.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isreal.m
        Chebfun commit: 7574c77
        """
        for piece in self.funs:
            if jnp.iscomplexobj(piece.coeffs):
                return False
        return True

    def real(self) -> Chebfun:
        """Real part of the Chebfun.

        Exact in coefficient space: the Chebyshev basis is real, so
        Re(f) has coefficients Re(c).

        Provenance
        ----------
        MATLAB source : @chebfun/real.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import Singfun

        def _real_tech(tech):
            if isinstance(tech, Chebtech2):
                return Chebtech2.from_coeffs(jnp.real(tech.coeffs))
            if isinstance(tech, Singfun):
                # Real part acts on the smooth factor; the singular
                # factors (1±x)^e are real (@singfun real via smoothPart).
                return Singfun(_real_tech(tech.smoothPart), tech.exponents)
            # Fourier coefficients of a real function are
            # conjugate-symmetric, not real — go through values.
            return type(tech).from_values(jnp.real(tech.values))

        new_funs = [p.with_tech(_real_tech(p.tech)) for p in self.funs]
        return Chebfun(funs=new_funs, domain=self.domain)

    def imag(self) -> Chebfun:
        """Imaginary part of the Chebfun (a real Chebfun).

        Provenance
        ----------
        MATLAB source : @chebfun/imag.m
        Chebfun commit: 7574c77
        """
        new_funs = [
            p.with_tech(
                Chebtech2.from_coeffs(jnp.imag(p.tech.coeffs))
                if isinstance(p.tech, Chebtech2)
                # Fourier coefficients of a real function are
                # conjugate-symmetric, not real — go through values.
                else type(p.tech).from_values(jnp.imag(p.tech.values))
            )
            for p in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def conj(self) -> Chebfun:
        """Complex conjugate of the Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/conj.m
        Chebfun commit: 7574c77
        """
        new_funs = [
            p.with_tech(
                Chebtech2.from_coeffs(jnp.conj(p.tech.coeffs))
                if isinstance(p.tech, Chebtech2)
                # Fourier coefficients of a real function are
                # conjugate-symmetric, not real — go through values.
                else type(p.tech).from_values(jnp.conj(p.tech.values))
            )
            for p in self.funs
        ]
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)

    def angle(self) -> Chebfun:
        """Phase angle atan2(imag f, real f), constructed adaptively.

        Provenance
        ----------
        MATLAB source : @chebfun/angle.m
        Chebfun commit: 7574c77
        """
        f = self
        new_funs = [
            _Piece.from_function(
                lambda x, _f=f: jnp.angle(_f(x)),
                float(p.interval[0]), float(p.interval[1]),
            )
            for p in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def logical(self) -> Chebfun:
        """Convert to a logical (0/1) Chebfun.

        Returns a piecewise Chebfun that is 1 wherever ``self`` is non-zero
        and 0 at the zeros of ``self`` (breakpoints are added at the roots).

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe (root-finding).

        Provenance
        ----------
        MATLAB source : @chebfun/logical.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        roots = self.roots()
        existing = _np.array(list(self.domain.breakpoints))
        if roots.shape[0] > 0:
            new_bps = _np.sort(_np.unique(
                _np.concatenate([existing, _np.asarray(roots)])
            ))
        else:
            new_bps = existing
        domain_len = float(self.domain.b - self.domain.a)
        tol = 1e6 * _np.finfo(_np.float64).eps * max(domain_len, 1.0)
        mask = _np.concatenate([[True], _np.diff(new_bps) > tol])
        new_bps = new_bps[mask]
        if len(new_bps) < 2:
            return self._apply_fun(lambda x: jnp.where(x != 0.0, jnp.ones_like(x), jnp.zeros_like(x)))
        new_dom = Domain(tuple(float(bp) for bp in new_bps))
        f = self
        eps = float(self.vscale) * _EPS if self.vscale > 0 else _EPS
        new_funs = [
            _Piece.from_function(
                lambda x, _f=f, _e=eps: jnp.where(jnp.abs(_f(x)) > _e, jnp.ones_like(x), jnp.zeros_like(x)),
                sub.a, sub.b,
            )
            for sub in new_dom.intervals
        ]
        return Chebfun(funs=new_funs, domain=new_dom)

    def any(self):
        """True if the Chebfun is non-zero anywhere on its domain; a
        per-column boolean row for array-valued input (MATLAB returns
        a 1 x m logical).

        Returns
        -------
        bool or jax.Array of bool, shape (m,)

        Provenance
        ----------
        MATLAB source : @chebfun/any.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return False
        if self.n_columns > 1:
            import numpy as _np
            col_max = _np.max(
                _np.stack([
                    _np.max(_np.abs(_np.asarray(p.tech.values)), axis=0)
                    for p in self.funs
                ]), axis=0)
            return jnp.asarray(col_max > _EPS)
        return self.vscale > _EPS

    def all(self):
        """True if the Chebfun is non-zero *everywhere* on its domain;
        a per-column boolean row for array-valued input.

        Returns True where the column has no roots.

        Returns
        -------
        bool or jax.Array of bool, shape (m,)

        Notes
        -----
        NOT JIT-safe (root-finding via eigenvalue computation).

        Provenance
        ----------
        MATLAB source : @chebfun/all.m
        Chebfun commit: 7574c77
        """
        roots = self.roots()
        if self.n_columns > 1:
            import numpy as _np
            r = _np.asarray(roots)
            if r.size == 0:
                return jnp.ones(self.n_columns, dtype=bool)
            return jnp.asarray(~_np.any(_np.isfinite(r), axis=0))
        return roots.shape[0] == 0

    def isempty(self) -> bool:
        """True if the Chebfun has no pieces.

        In practice, the standard constructor always creates at least one
        piece, so this is always False for valid Chebfuns.  It is kept for
        API compatibility.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isempty.m
        Chebfun commit: 7574c77
        """
        return len(self.funs) == 0

    def isequal(self, other: Chebfun) -> bool:
        """Equality test: True if self and other have identical coefficients.

        Checks domain equality, the number of pieces, and then compares
        Chebyshev coefficients of each corresponding piece.

        Parameters
        ----------
        other : Chebfun

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isequal.m
        Chebfun commit: 7574c77
        """
        if self.is_transposed != other.is_transposed:
            # A column and a row Chebfun are never equal (MATLAB isequal).
            return False
        if self.domain != other.domain:
            return False
        if len(self.funs) != len(other.funs):
            return False
        tol = 10.0 * _EPS
        for pf, pg in zip(self.funs, other.funs):
            cf = pf.coeffs
            cg = pg.coeffs
            if cf.shape[1:] != cg.shape[1:]:
                # Different column counts -> not equal (array-valued mismatch).
                return False
            # Pad the shorter coefficient array with zeros along the *degree*
            # axis only (axis 0).  A bare ``(0, k)`` pad width applies to every
            # axis, which for array-valued (n, m) coeffs wrongly grows the
            # column axis too -- e.g. (13, 2) -> (14, 3) -- and raised a
            # broadcasting error whenever two pieces had different lengths.
            n = max(cf.shape[0], cg.shape[0])
            pad_f = [(0, n - cf.shape[0])] + [(0, 0)] * (cf.ndim - 1)
            pad_g = [(0, n - cg.shape[0])] + [(0, 0)] * (cg.ndim - 1)
            cf_pad = jnp.pad(cf, pad_f)
            cg_pad = jnp.pad(cg, pad_g)
            if float(jnp.max(jnp.abs(cf_pad - cg_pad))) > tol:
                return False
        return True

    def __eq__(self, other) -> bool:
        """Equality shortcut: delegates to :meth:`isequal`."""
        if isinstance(other, Chebfun):
            return self.isequal(other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Fractional calculus
    # ------------------------------------------------------------------

    def fracInt(self, mu: float) -> "Chebfun":
        r"""Riemann-Liouville fractional integral of order *mu*.

        Computes the fractional integral

        .. math::
            I^\mu f(x) = \frac{1}{\Gamma(\mu)}
                \int_a^x (x - t)^{\mu - 1} f(t)\, dt.

        For ``mu = n`` (positive integer) this reduces to *n* repeated
        applications of ``cumsum``.

        Parameters
        ----------
        mu : float
            Order of integration (>= 0).  Must be ``>= 0``.

        Returns
        -------
        Chebfun

        Notes
        -----
        The fractional integral is computed via quadrature on a Chebyshev
        grid using the kernel ``(x - t)^{mu-1} / Gamma(mu)``.  Only single-
        piece Chebfuns are supported.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/fracInt.m, @fun/fracInt.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.fracDiff, Chebfun.cumsum

        Examples
        --------
        Fractional integral of order 0.5 of the constant function 1:

        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun1d.chebfun import chebfun
        >>> f = chebfun(lambda x: jnp.ones_like(x))
        >>> g = f.fracInt(0.5)  # I^{0.5}[1](x) = 2*sqrt(x+1)/Gamma(1.5) on [-1,1]
        >>> g(jnp.float64(0.0)) is not None  # smoke test
        True
        """
        # uses-numpy: scipy.special.gamma for non-integer orders
        import numpy as _np
        from scipy.special import gamma as _gamma

        mu = float(mu)
        if mu < 0:
            raise ValueError("fracInt: mu must be >= 0.")

        # Integer part: repeated cumsum
        mu_int = int(_np.floor(mu))
        mu_frac = mu - mu_int

        f = self
        for _ in range(mu_int):
            f = f.cumsum()

        if mu_frac == 0.0:
            return f

        # Fractional part via Volterra integral operator with kernel (x-t)^{mu_frac-1}/Gamma(mu_frac)
        if len(f.funs) > 1:
            raise ValueError(
                "fracInt: fractional integral only supported for single-piece Chebfuns. "
                "Use a Chebfun with one interval."
            )

        gam = float(_gamma(mu_frac))
        a = float(f.domain.a)
        b = float(f.domain.b)

        # Gauss-JACOBI quadrature absorbing the (x - t)^(mu-1) endpoint
        # singularity into the weight (Fable 5 fix -- plain Gauss-
        # Legendre on the singular kernel converged only algebraically,
        # giving ~4-digit fracInt and a systematically biased fracDiff):
        #   I(x) = (x-a)^mu / (Gamma(mu) 2^mu)
        #          * sum_j w_j f(a + (x-a)(xi_j+1)/2),
        # with (xi_j, w_j) = jacpts(n, mu-1, 0) so that the weight
        # (1-xi)^(mu-1) is exactly the kernel's singular factor.
        from chebfunjax.utils.quadrature import jacpts as _jacpts
        n_q = 60
        xi_ref, w_ref = (_np.asarray(v)
                         for v in _jacpts(n_q, mu_frac - 1.0, 0.0))

        def _frac_int_at_x(x_scalar: float) -> float:
            if x_scalar <= a + 1e-15 * (b - a):
                return 0.0
            t_phys = a + (x_scalar - a) * (xi_ref + 1.0) / 2.0
            fvals = _np.asarray(f(jnp.array(t_phys)), dtype=float)
            pref = (x_scalar - a) ** mu_frac / (gam * 2.0 ** mu_frac)
            return float(pref * _np.dot(w_ref, fvals))

        def _frac_integrand(x_arr):
            x_arr = _np.asarray(x_arr, dtype=float)
            result = _np.array([_frac_int_at_x(float(xi)) for xi in x_arr.ravel()],
                               dtype=float)
            return jnp.array(result.reshape(x_arr.shape))

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        return _cf(_frac_integrand, domain=(a, b))

    def fracDiff(self, mu: float, kind: str = "RL") -> "Chebfun":
        r"""Fractional derivative of order *mu* (Riemann-Liouville or Caputo).

        Computes the order-*mu* fractional derivative using either the
        Riemann-Liouville or Caputo definition.

        **Riemann-Liouville** (default)::

            D^mu f = D^n I^{n-mu} f,   n = ceil(mu)

        **Caputo**::

            D^mu f = I^{n-mu} D^n f,   n = ceil(mu)

        where *D^n* denotes the *n*-th classical derivative and *I^alpha*
        denotes :meth:`fracInt` of order *alpha*.

        Parameters
        ----------
        mu : float
            Fractional order (>= 0).
        kind : {'RL', 'Caputo'}
            Definition to use.  Default ``'RL'`` (Riemann-Liouville).

        Returns
        -------
        Chebfun

        Notes
        -----
        For integer *mu* both definitions agree with the classical
        *n*-th derivative (computed via repeated differentiation).
        Only single-piece Chebfuns are supported for the fractional part.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun/fracDiff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.fracInt, Chebfun.diff

        Examples
        --------
        Fractional derivative of order 1 equals the classical derivative:

        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun1d.chebfun import chebfun
        >>> f = chebfun(jnp.sin)
        >>> df_frac = f.fracDiff(1.0)
        >>> df_class = f.diff(1)
        >>> err = float(jnp.max(jnp.abs(df_frac(jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)) -
        ...                             df_class(jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)))))
        >>> err < 1e-8
        True
        """
        import math as _math

        mu = float(mu)
        if mu < 0:
            raise ValueError("fracDiff: mu must be >= 0.")

        n = _math.ceil(mu)

        if n == mu:
            # Integer order: classical derivative
            return self.diff(int(n))

        if kind.upper() in ("RL", "RIEMANNLIOUVILLE"):
            # Riemann-Liouville: I^{n-mu} first, then D^n
            g = self.fracInt(n - mu)
            return g.diff(n)
        elif kind.upper() == "CAPUTO":
            # Caputo: D^n first, then I^{n-mu}
            g = self.diff(n)
            return g.fracInt(n - mu)
        else:
            raise ValueError(
                f"fracDiff: unknown kind '{kind}'. Use 'RL' or 'Caputo'."
            )

    # ------------------------------------------------------------------
    # L1 polynomial fitting
    # ------------------------------------------------------------------

    def polyfitL1(self, n: int) -> "Chebfun":
        """Best polynomial approximation of degree *n* in the L1 norm.

        Computes the degree-*n* polynomial *p* that minimises
        ``|| f - p ||_1 = int_a^b |f(x) - p(x)| dx``
        using Watson's iterative algorithm.

        Parameters
        ----------
        n : int
            Degree of the approximating polynomial.

        Returns
        -------
        Chebfun
            The best L1 polynomial approximant of degree *n*.

        Notes
        -----
        Watson's algorithm is a damped Newton iteration that constructs a
        sequence of polynomial approximations converging to the L1 best
        approximant.  Unlike the L-infinity case (Remez), the L1 optimum
        may not be unique.

        This implementation delegates to :func:`chebfunjax.utils.minimax.minimax`
        for the initial polynomial interpolant and then runs Watson's update
        loop.  For smooth *f* with ``len(f) <= n+1`` the result is just *f*
        itself.

        NOT JIT-safe (iterative construction).

        Provenance
        ----------
        MATLAB source : @chebfun/polyfitL1.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            [1] G. A. Watson, "An algorithm for linear L1 approximation of
                continuous functions", IMA J. Numer. Anal., 1, 1981.
            [2] Y. Nakatsukasa and A. Townsend, arXiv:1902.02664, 2019.

        See Also
        --------
        chebfunjax.utils.minimax.minimax

        Examples
        --------
        L1 best polynomial approximation to |x| of degree 10:

        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun1d.chebfun import chebfun
        >>> f = chebfun(jnp.abs)
        >>> p = f.polyfitL1(10)
        >>> float(abs(p).sum()) > 0  # smoke test: returns a valid Chebfun
        True
        """
        # uses-numpy: Watson-Newton iteration uses NumPy linear algebra
        import numpy as _np
        from numpy.polynomial import chebyshev as _C

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        a = float(self.domain.a)
        b = float(self.domain.b)
        if len(self.funs) == 1 and self.funs[0].n <= n + 1:
            return self

        # Work in the normalized variable s in [-1, 1].
        def fs(s):
            x = 0.5 * (b - a) * _np.asarray(s) + 0.5 * (a + b)
            return _np.asarray(self(jnp.asarray(x)), dtype=float)

        def _int_T(k, lo, hi):
            # exact integral of T_k on [lo, hi] (normalized variable)
            if k == 0:
                return hi - lo
            if k == 1:
                return 0.5 * (hi ** 2 - lo ** 2)
            def A(t):
                return 0.5 * (_C.chebval(t, _np.eye(k + 2)[k + 1])
                              / (k + 1)
                              - _C.chebval(t, _np.eye(k)[k - 1])
                              / (k - 1))
            return A(hi) - A(lo)

        # start from the L2 projection (Chebyshev-Gauss quadrature fit)
        ng = max(8 * (n + 1), 64)
        sg = _np.cos(_np.pi * (_np.arange(ng) + 0.5) / ng)
        c = _C.chebfit(sg, fs(sg), n)

        sfine = _np.linspace(-1.0, 1.0, 4001)
        ffine = fs(sfine)
        for _ in range(60):
            e = ffine - _C.chebval(sfine, c)
            # sign-change locations (bisection-refined from the grid)
            sgn = _np.sign(e)
            idx = _np.nonzero(_np.diff(sgn) != 0)[0]
            roots = []
            for i in idx:
                lo, hi = sfine[i], sfine[i + 1]
                flo = e[i]
                for _b in range(60):
                    mid = 0.5 * (lo + hi)
                    fm = (fs(_np.array([mid]))[0]
                          - _C.chebval(mid, c))
                    if flo * fm <= 0:
                        hi = mid
                    else:
                        lo, flo = mid, fm
                roots.append(0.5 * (lo + hi))
            roots = _np.asarray(roots)
            # optimality residual G_k = int T_k sign(e)
            seg = _np.concatenate([[-1.0], roots, [1.0]])
            mids = 0.5 * (seg[:-1] + seg[1:])
            sig = _np.sign(fs(mids) - _C.chebval(mids, c))
            G = _np.array([
                sum(sig[j] * _int_T(k, seg[j], seg[j + 1])
                    for j in range(len(sig)))
                for k in range(n + 1)])
            if _np.max(_np.abs(G)) < 1e-11:
                break
            # Watson Jacobian: J_kj = 2 sum_r T_k(r) T_j(r) / |e'(r)|
            dedx = (_np.gradient(ffine, sfine)
                    - _C.chebval(sfine, _C.chebder(c)))
            eprime = _np.abs(_np.interp(roots, sfine, dedx))
            eprime = _np.maximum(eprime, 1e-8)
            Tk = _np.vstack([_C.chebval(roots, _np.eye(n + 1)[k])
                             for k in range(n + 1)])
            J = 2.0 * (Tk / eprime) @ Tk.T
            J += 1e-14 * _np.eye(n + 1)
            step = _np.linalg.solve(J, G)
            # damped Newton for robustness far from the optimum
            nrm = _np.max(_np.abs(step))
            if nrm > 0.5:
                step *= 0.5 / nrm
            c = c + step


        def p_eval(x):
            s = (2.0 * jnp.asarray(x) - (a + b)) / (b - a)
            return jnp.asarray(_C.chebval(_np.asarray(s),
                                          _np.asarray(c)))

        return _cf(p_eval, domain=(a, b))

    # ------------------------------------------------------------------
    # Plotting method on the Chebfun object
    # ------------------------------------------------------------------

    def plot(self, ax=None, **kw):
        """Plot this Chebfun.  Delegates to :func:`chebfunjax.plotting.plot`.

        Returns ``(fig, ax)`` so callers can customise the axes or overlay
        additional plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        **kw
            Additional keyword arguments forwarded to :func:`~chebfunjax.plotting.plot`.

        Returns
        -------
        fig, ax
        """
        from chebfunjax.plotting import plot as _plot
        return _plot(self, ax=ax, **kw)


# ============================================================================
# Factory function — the main user-facing entry point
# ============================================================================

def getValuesAtBreakpoints(f: "Chebfun", op=None) -> jax.Array:
    """Values at the breakpoints of f (MATLAB
    chebfun.getValuesAtBreakpoints): op (default f itself) evaluated
    at every breakpoint.

    Provenance
    ----------
    MATLAB source : @chebfun/getValuesAtBreakpoints.m
    Chebfun commit: 7574c77
    """
    breaks = [float(p.interval[0]) for p in f.funs] \
        + [float(f.domain.b)]
    xb = jnp.asarray(breaks, dtype=jnp.float64)
    if op is None:
        return f(xb)
    return jnp.asarray(op(xb), dtype=xb.dtype)


def kron(f, g, mode=None):
    """Kronecker product of two chebfuns.

    ``kron(f, g)`` (default) is the rank-1 Chebfun2
    ``kron(f, g)(x, y) = f(x) * g(y)`` -- MATLAB ``kron(f.', g)`` with ``f``
    a row chebfun.

    ``kron(f, g, 'op')`` builds the rank-1 integral OPERATOR
    ``A = f (g' .)``: ``A*h = f * <g, h>`` (see
    :class:`chebfunjax.operators.blocks.KronOp`).  ``f`` and ``g`` may be
    array-valued (given as a list of column chebfuns), in which case the
    operator is a sum of rank-1 terms.

    Provenance
    ----------
    MATLAB source : @chebfun/kron.m
    Chebfun commit: 7574c77
    """
    if mode == "op":
        from chebfunjax.operators.blocks import KronOp
        fs = list(f) if isinstance(f, (list, tuple)) else [f]
        gs = list(g) if isinstance(g, (list, tuple)) else [g]
        dom = (float(fs[0].domain.a), float(fs[0].domain.b))
        return KronOp(fs, gs, dom)
    if mode is not None and mode != "op":
        raise ValueError(
            "CHEBFUN:CHEBFUN:kron:sizes -- unknown kron mode "
            f"{mode!r} (expected 'op').")

    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    fa, fb = float(f.domain.a), float(f.domain.b)
    ga, gb = float(g.domain.a), float(g.domain.b)
    return Chebfun2.from_function(
        lambda x, y: f(x) * g(y), domain=(fa, fb, ga, gb))


def _qm_gram(cols):
    """Gram matrix ``G[i,j] = <cols_i, cols_j>`` of a list of chebfuns."""
    import numpy as _np
    m = len(cols)
    G = _np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            G[i, j] = complex((cols[i].conj() * cols[j]).sum())
    return G


def _lincomb(cols, coeffs):
    """Chebfun linear combination ``sum_k coeffs[k] * cols[k]``."""
    out = cols[0] * complex(coeffs[0])
    for k in range(1, len(cols)):
        out = out + cols[k] * complex(coeffs[k])
    return out


def mldivide(A, B):
    """Left matrix division ``A \\ B`` (least squares) -- MATLAB mldivide.

    Supports the mixed scalar / numeric-matrix / quasimatrix cases used by
    ``@chebfun/mldivide``:

    - scalar ``A``: ``B / A`` (elementwise);
    - numeric ``A`` (m x n) and a row-quasimatrix ``B`` (a list of m
      chebfuns): ``X = pinv(A) @ B``, a list of n chebfuns;
    - a row-quasimatrix ``A`` (a list of m chebfuns) and numeric ``B``
      (m x k): the least-squares chebfun(s) ``X`` with ``<A_i, X> = B_i``
      via the Gram system.

    Provenance
    ----------
    MATLAB source : @chebfun/mldivide.m
    Chebfun commit: 7574c77
    """
    import numpy as _np
    if isinstance(A, (int, float, complex)):
        if isinstance(B, list):
            return [b * (1.0 / A) for b in B]
        return B * (1.0 / A)
    if isinstance(A, list) and not isinstance(B, list):
        # Row-quasimatrix A, numeric B: Gram solve <A_i, X> = B_i.
        Bv = _np.atleast_2d(_np.asarray(B, dtype=complex))
        if Bv.shape[0] != len(A):
            Bv = Bv.T
        G = _qm_gram(A)
        C = _np.linalg.solve(G, Bv)          # (m, k)
        cols = [_lincomb(A, C[:, j]) for j in range(C.shape[1])]
        return cols[0] if len(cols) == 1 else cols
    # Numeric A, row-quasimatrix B: X = pinv(A) @ B.
    Am = _np.atleast_2d(_np.asarray(A, dtype=float))
    Bl = B if isinstance(B, list) else [B]
    # Orient A so its row count matches the number of B chebfuns.
    if Am.shape[0] != len(Bl) and Am.shape[1] == len(Bl):
        Am = Am.T
    P = _np.linalg.pinv(Am)                   # (n, m)
    out = [_lincomb(Bl, P[j, :]) for j in range(P.shape[0])]
    return out[0] if len(out) == 1 else out


def mrdivide(A, B):
    """Right matrix division ``A / B`` (least squares) -- MATLAB mrdivide.

    ``A / B`` solves ``X B = A``; it is the transpose-dual of
    :func:`mldivide` (``X B = A  <=>  B' X' = A'``).

    Provenance
    ----------
    MATLAB source : @chebfun/mrdivide.m
    Chebfun commit: 7574c77
    """
    if isinstance(B, (int, float, complex)):
        if isinstance(A, list):
            return [a * (1.0 / B) for a in A]
        return A * (1.0 / B)
    # X B = A  <=>  B' \ A'  (mldivide with the orientations swapped).
    return mldivide(B, A)


def gmres(L, f, tol: float = 1e-10, maxiter: int = 36):
    """Solve the linear operator equation ``L(u) = f`` for a Chebfun ``u``
    by GMRES with Chebfun inner products (MATLAB ``@chebfun/gmres``).

    Parameters
    ----------
    L : callable
        A linear operator ``u -> L(u)`` mapping a Chebfun to a Chebfun.
    f : Chebfun
        Right-hand side.
    tol : float, default 1e-10
        Relative-residual convergence tolerance.
    maxiter : int, default 36
        Maximum number of Arnoldi iterations.

    Returns
    -------
    (u, flag) : (Chebfun, int)
        The solution and a convergence flag (0 = converged, 1 = not).

    Provenance
    ----------
    MATLAB source : @chebfun/gmres.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    def _ip(a, b):
        return complex((a.conj() * b).sum())

    def _nrm(a):
        return float(_np.sqrt(abs(_ip(a, a))))

    normb = _nrm(f)
    if normb == 0.0:
        return f * 0.0, 0
    beta = _nrm(f)
    Q = [f * (1.0 / beta)]
    H = _np.zeros((maxiter + 2, maxiter + 1), dtype=complex)
    flag = 1
    y = _np.array([beta], dtype=complex)
    n_used = 0
    for n in range(maxiter):
        n_used = n
        v = L(Q[n])
        for k in range(n + 1):
            H[k, n] = _ip(Q[k], v)
            v = v - Q[k] * complex(H[k, n])
        H[n + 1, n] = _nrm(v)
        if H[n + 1, n] > 1e-300:
            Q.append(v * (1.0 / H[n + 1, n]))
        rhs = _np.zeros(n + 2, dtype=complex)
        rhs[0] = beta
        y, *_ = _np.linalg.lstsq(H[:n + 2, :n + 1], rhs, rcond=None)
        res = _np.linalg.norm(H[:n + 2, :n + 1] @ y - rhs) / normb
        if res < tol:
            flag = 0
            break
    u = Q[0] * complex(y[0])
    for k in range(1, min(len(y), n_used + 2)):
        u = u + Q[k] * complex(y[k])
    return u, flag


def wronskian(*args) -> Chebfun:
    """Wronskian determinant of n chebfuns (MATLAB wronskian).

    ``wronskian(f1, ..., fn)`` or ``wronskian(L, f1, ..., fn)`` (a
    leading chebop is accepted and ignored -- it only fixes n in
    MATLAB).  Returns det([f_i^(j)]) as a chebfun.

    Provenance
    ----------
    MATLAB source : @chebop/wronskian.m
    Chebfun commit: 7574c77
    """
    funs = [a for a in args if isinstance(a, Chebfun)]
    if not funs:
        raise ValueError("wronskian: no chebfun inputs")
    n = len(funs)
    a, b = float(funs[0].domain.a), float(funs[0].domain.b)
    ders = [[f if j == 0 else f.diff(j) for j in range(n)]
            for f in funs]

    def w(x):
        rows = [jnp.stack([ders[i][j](x) for i in range(n)], axis=-1)
                for j in range(n)]
        M = jnp.stack(rows, axis=-2)     # (..., n, n)
        return jnp.linalg.det(M)

    return Chebfun.from_function(w, Domain((a, b)))


def complex_fun(f: Chebfun, g: Chebfun) -> Chebfun:
    """complex(f, g) = f + 1i*g for real chebfuns (MATLAB complex).

    Provenance
    ----------
    MATLAB source : @chebfun/complex.m
    Chebfun commit: 7574c77
    """
    for h in (f, g) if isinstance(g, Chebfun) else (f,):
        if any(jnp.iscomplexobj(p.tech.coeffs) for p in h.funs):
            raise ValueError("complex: inputs must be real")
    return f + g * 1j


def cell2quasi(cells):
    """Build a Quasimatrix from a list of chebfuns (MATLAB cell2quasi).

    Provenance
    ----------
    MATLAB source : @chebfun/cell2quasi.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    if not cells:
        raise ValueError("cell2quasi: empty input")
    return Quasimatrix(list(cells), cells[0].domain)


def overlap(f: Chebfun, g: Chebfun) -> tuple[Chebfun, Chebfun]:
    """Return copies of f and g with identical breakpoints
    (MATLAB overlap).

    Provenance
    ----------
    MATLAB source : @chebfun/overlap.m
    Chebfun commit: 7574c77
    """
    return Chebfun._overlap(f, g)


def atan2(y: Chebfun, x: Chebfun) -> Chebfun:
    r"""Two-argument arctangent of two Chebfuns: ``atan2(y, x)``.

    Computes the four-quadrant inverse tangent of the Chebfun pair ``(y, x)``,
    returning values in ``(-pi, pi]``.  Breakpoints are introduced where
    ``y = 0`` and ``x < 0`` (the cut of the standard ``atan2``).

    Parameters
    ----------
    y : Chebfun
        Numerator (the "y" component).
    x : Chebfun
        Denominator (the "x" component).

    Returns
    -------
    Chebfun
        ``atan2(y, x)`` on the same domain.

    Notes
    -----
    Implementation follows MATLAB Chebfun's ``@chebfun/atan2.m``:

    1. Find roots of ``y`` where ``x < 0`` (discontinuities of atan2).
    2. Add those as breakpoints to the domain.
    3. Compose with ``jnp.arctan2`` pointwise, extrapolating the piece
       endpoints (which lie on the roots of ``y``) so the branch cut is
       not sampled directly.  This mirrors MATLAB's
       ``pref.techPrefs.extrapolate = true`` and is required for
       eps-level accuracy: at a root ``y`` is numerically +/-0 and
       ``atan2`` returns the wrong branch, injecting a spurious ``2*pi``
       jump that otherwise defeats the adaptive constructor.

    NOT JIT-safe (root-finding and adaptive construction).

    Provenance
    ----------
    MATLAB source : @chebfun/atan2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun.atan

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.chebfun import chebfun, atan2
    >>> x = chebfun(lambda t: jnp.cos(t))
    >>> y = chebfun(lambda t: jnp.sin(t))
    >>> f = atan2(y, x)
    >>> abs(float(f(jnp.float64(0.0)))) < 1e-12  # atan2(0, 1) = 0
    True
    """
    import numpy as _np

    if y.domain != x.domain:
        raise ValueError(
            "atan2: y and x must have the same domain. "
            f"Got {y.domain} and {x.domain}."
        )

    # Find breakpoints: roots of y where x < 0
    ry = _np.asarray(y.roots(), dtype=_np.float64)
    # Keep only those where x(ry) < 0
    if len(ry) > 0:
        xvals_at_ry = _np.asarray(x(jnp.array(ry)), dtype=_np.float64)
        tol = 2.0 * _EPS * max(float(y.vscale), float(x.vscale))
        ry = ry[xvals_at_ry < tol]  # keep roots where x <= 0

    # Also find roots of x where y changes sign (kinks in angle)
    # Simplified: just use the roots of y (main discontinuities)
    existing = _np.array(list(y.domain.breakpoints), dtype=_np.float64)
    if len(ry) > 0:
        new_bps = _np.sort(_np.unique(_np.concatenate([existing, ry])))
        tol_merge = 1e6 * _EPS * max(float(y.domain.b - y.domain.a), 1.0)
        mask = _np.concatenate([[True], _np.diff(new_bps) > tol_merge])
        new_bps = new_bps[mask]
    else:
        new_bps = existing

    if len(new_bps) < 2:
        new_bps = existing

    new_dom = Domain(tuple(float(b) for b in new_bps))
    _y = y
    _x = x

    def _piece_fun(sub_a, sub_b):
        # MATLAB's atan2.m sets pref.techPrefs.extrapolate = true: the
        # subinterval endpoints are the roots of y where the atan2 branch
        # cut lives, so y is numerically +/-0 there and atan2 returns the
        # WRONG branch (e.g. -pi instead of the +pi the interior
        # approaches) whenever the root's rounding sign disagrees with the
        # limit.  Sampled directly, that single flipped endpoint value
        # injects a spurious 2*pi jump at the piece boundary, giving 1/n
        # Chebyshev decay and a construction that never resolves.  We
        # emulate extrapolation by pulling the (only troublesome) endpoint
        # samples an infinitesimal step inside the subinterval, so atan2 is
        # evaluated on the correct branch; the shift is O(1e-11) of the
        # width, far below the construction tolerance.
        width = sub_b - sub_a
        # Nudge ONLY the two endpoint nodes off the roots of y where the
        # atan2 branch cut lives.  A tolerance band (1e-10 of the width)
        # catches them robustly despite the reference->physical mapping
        # rounding, yet is far tighter than the gap to the nearest interior
        # Chebyshev node at the degrees needed here, so the interior samples
        # -- including atan2's steep g=0 crossings -- are left untouched.
        # The inward step (1e-12 of the width) lifts y well above its
        # roundoff floor while shifting the endpoint value by < ~1e-12.
        tol_e = 1e-10 * width
        step = 1e-12 * width

        def fn(t, sa=sub_a, sb=sub_b, te=tol_e, d=step):
            x = jnp.where(jnp.abs(t - sa) < te, sa + d, t)
            x = jnp.where(jnp.abs(x - sb) < te, sb - d, x)
            return jnp.arctan2(_y(x), _x(x))

        return _Piece.from_function(fn, sub_a, sub_b)

    new_funs = [_piece_fun(sub.a, sub.b) for sub in new_dom.intervals]
    return Chebfun(funs=new_funs, domain=new_dom)


# ============================================================================
# Singular-exponent factory helpers ('exps' / 'blowup' — SingFun wiring)
# ============================================================================
# uses-numpy: exponent parsing/reshaping is pure Python/NumPy bookkeeping on
# small user-supplied vectors, never on library array data.


def _parse_exps(exps, n_int: int) -> "list[tuple[float, float]]":
    """Expand a user ``exps`` vector into one ``(left, right)`` pair per interval.

    Mirrors the MATLAB @chebfun/chebfun.m parsing (``parseInputs``):

    * 1 value        -> broadcast to every endpoint;
    * 2 values       -> the two entire-domain endpoints, interior breaks 0;
    * ``n_int + 1``  -> one shared value per breakpoint;
    * ``2*n_int``    -> a ``(left, right)`` pair per interval.

    ``NaN`` entries are preserved so the caller can autodetect that endpoint.

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m (parseInputs, exps reshaping)
    Chebfun commit: 7574c77
    """
    import numpy as _np
    if exps is None:
        flat = [float("nan")] * (2 * n_int)
    else:
        e = [float(v) for v in _np.atleast_1d(
            _np.asarray(exps, dtype=float)).ravel()]
        ne = len(e)
        if ne == 1:
            flat = e * (2 * n_int)
        elif ne == 2:
            flat = [e[0]] + [0.0] * (2 * (n_int - 1)) + [e[1]]
        elif ne == n_int + 1:
            flat = []
            for j in range(n_int):
                flat += [e[j], e[j + 1]]
        elif ne == 2 * n_int:
            flat = e
        else:
            raise ValueError(
                f"chebfun: {ne} exponents supplied for {n_int} interval(s); "
                f"expected 1, 2, {n_int + 1}, or {2 * n_int}.")
    return [(flat[2 * j], flat[2 * j + 1]) for j in range(n_int)]


def _parse_singtype(singType, n_int: int, blowup: bool
                    ) -> "list[tuple[str | None, str | None]]":
    """Expand a user ``singType`` into one ``(left, right)`` pair per interval.

    Each entry is ``'pole'``, ``'sing'``, or ``'none'`` (or ``None`` to defer
    to the exponent value).  Two entries are read as the entire-domain
    endpoints with interior breaks ``'none'``.  When ``blowup`` is set with no
    ``singType`` the default is ``'sing'`` at every endpoint.

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m (parseInputs, singType handling)
    Chebfun commit: 7574c77
    """
    if singType is None:
        default = "sing" if blowup else None
        return [(default, default)] * n_int
    st = list(singType)
    if len(st) == 2 and n_int >= 1:
        flat = [st[0]] + ["none"] * (2 * (n_int - 1)) + [st[1]]
    elif len(st) == 2 * n_int:
        flat = st
    else:
        raise ValueError(
            f"chebfun: {len(st)} singType entries for {n_int} interval(s).")
    return [(flat[2 * j], flat[2 * j + 1]) for j in range(n_int)]


def _cast_tech_pair(a, b):
    """Return ``(a, b)`` with a common tech type for cross-tech arithmetic.

    MATLAB casts mixed TRIGTECH + CHEBTECH arithmetic to the CHEBTECH basis
    (``@chebfun/plus.m`` and friends promote a periodic operand to the
    non-periodic tech before combining).  When the two operands have the
    same tech type (the common case) they are returned unchanged, so the
    hot path is untouched.

    Provenance
    ----------
    MATLAB source : @chebfun/plus.m, @chebfun/times.m (tech casting)
    Chebfun commit: 7574c77
    """
    if type(a) is type(b):
        return a, b
    from chebfunjax.tech.trigtech import Trigtech
    if isinstance(a, Trigtech) and not isinstance(b, Trigtech):
        return Chebtech2.from_function(lambda t, _a=a: _a(t)), b
    if isinstance(b, Trigtech) and not isinstance(a, Trigtech):
        return a, Chebtech2.from_function(lambda t, _b=b: _b(t))
    return a, b


def _build_exps_piece(op, a: float, b: float, el, er, stl, str_,
                      turbo: bool = False) -> _Piece:
    """Build one Chebfun piece on ``[a, b]`` honouring endpoint exponents.

    ``op`` is the physical function on ``[a, b]``.  Each exponent is either
    given (``el``/``er``), autodetected when ``NaN``/``None``, or forced to 0
    by a ``'none'`` singType.  A ``'pole'`` hint uses the integer pole-order
    finder; otherwise the fractional singularity finder (which also recovers
    integer poles) is used.  When both exponents vanish a smooth piece is
    returned; otherwise a :class:`Singfun` piece.

    Provenance
    ----------
    MATLAB source : @classicfun/constructor.m -> @singfun/singfun.m
    Chebfun commit: 7574c77
    """
    import math as _m

    from chebfunjax.fun.singfun import Singfun, _find_pole_order, _find_sing_order

    def _full(t):
        x = a + (b - a) * (jnp.asarray(t) + 1.0) / 2.0
        return op(x)

    def _resolve(e, st, end):
        if st == "none":
            return 0.0
        detect = e is None or (isinstance(e, float) and _m.isnan(e))
        if detect:
            if st == "pole":
                return _find_pole_order(_full, end)
            return _find_sing_order(_full, end)
        return float(e)

    el = _resolve(el, stl, "left")
    er = _resolve(er, str_, "right")
    if abs(el) < 1e-14 and abs(er) < 1e-14:
        return _Piece.from_function(op, a, b, turbo=turbo)
    sf = Singfun.from_function(_full, exponents=(el, er), turbo=turbo)
    return _Piece(tech=sf, interval=(float(a), float(b)))


def chebfun(
    f=None,
    *,
    domain=(-1.0, 1.0),
    n: int | None = None,
    trig: bool = False,
    eps: float | None = None,
    max_length: int | None = None,
    splitting: bool = False,
    exps: tuple[float, float] | None = None,
    blowup: bool | int = False,
    singType: "list | tuple | None" = None,
    turbo: bool = False,
    equi: bool = False,
) -> Chebfun:
    """Create a Chebfun from a callable, array of coefficients, or constant.

    This is the primary construction entry point. It mimics MATLAB's
    ``chebfun(...)`` syntax.

    With ``trig=True`` the function is represented in the Fourier basis
    (MATLAB's ``chebfun(f, dom, 'trig')``): f must be smooth and periodic
    on the domain, which must be a single interval.

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
    # Empty chebfun (MATLAB chebfun(), chebfun([]), chebfun([], dom),
    # chebfun(@sin, 0)): no data / a zero-length domain -> the empty object.
    import numpy as _np
    _empty_f = f is None or (
        not callable(f) and hasattr(f, "__len__")
        and len(_np.ravel(_np.asarray(f, dtype=object))) == 0)
    try:
        _dv = ([float(x) for x in domain] if hasattr(domain, "__len__")
               else [float(domain)])
    except (TypeError, ValueError):
        _dv = [0.0]
    # An EMPTY domain gives the empty object; a DEGENERATE domain
    # (repeated endpoints) with actual data is an error, matching
    # MATLAB's 'Domain intervals must be of positive length' (pinned
    # by test_invalid_domain).
    _empty_dom = len(_dv) == 0
    if _empty_f or _empty_dom or n == 0:
        return Chebfun.empty()
    if len(_dv) < 2 or len(set(_dv)) < len(_dv):
        raise ValueError(
            "chebfun: domain intervals must be of positive length")

    # Endpoint singularities (MATLAB 'exps'/'blowup' flags): each interval
    # becomes a Singfun piece s(x)*(1+x)^a*(1-x)^b (Fable 5, wiring the
    # SingFun class into the chebfun factory).  Exponents are either given
    # explicitly (``exps``), autodetected (``blowup``/NaN entries), or a
    # mix.  ``singType`` hints the detector (``'pole'``/``'sing'``/``'none'``).
    if exps is not None or blowup:
        if trig or n is not None:
            raise ValueError(
                "chebfun: exps/blowup cannot be combined with trig/n")
        if not callable(f):
            raise ValueError("chebfun: exps/blowup requires a callable.")
        dom_vals = [float(v) for v in (domain if hasattr(domain, "__len__")
                                       else (domain,))]
        if len(dom_vals) < 2:
            raise ValueError("chebfun: exps/blowup requires a domain.")
        n_int = len(dom_vals) - 1
        pairs = _parse_exps(exps, n_int)
        stypes = _parse_singtype(singType, n_int, bool(blowup))
        # ``splitting`` only affects detection of INTERIOR singularities
        # (poles/branch points away from the breakpoints), which the
        # singular factory does not yet locate; the given/detected endpoint
        # exponents are still honoured on each supplied interval.
        funs = []
        for j in range(n_int):
            a_, b_ = dom_vals[j], dom_vals[j + 1]
            piece = _build_exps_piece(f, a_, b_, pairs[j][0], pairs[j][1],
                                      stypes[j][0], stypes[j][1], turbo=turbo)
            funs.append(piece)
        return Chebfun(funs=funs, domain=Domain(tuple(dom_vals)))

    # --- Preferences (task #11): eps -> chop tolerance, max_length ->
    #     maximum adaptive length (2**maxpow2 + 1). ---
    _tol = None if eps is None else float(eps)
    if max_length is None:
        _maxpow2 = 16
    else:
        _maxpow2 = max(4, int(math.ceil(math.log2(max(int(max_length) - 1, 2)))))

    # --- 'equi' flag: data sampled on an equispaced grid ---
    # The values are interpreted as coming from linspace(a, b, N); a
    # Floater-Hormann rational interpolant (FUNQUI) is built and then
    # resolved adaptively as a Chebfun (MATLAB @chebfun/chebfun.m
    # 'equi' -> chebfunpref.enableFunqui -> @smoothfun funqui).
    if equi:
        if callable(f):
            raise ValueError(
                "chebfun(..., equi=True): the 'equi' flag requires numeric "
                "data sampled on an equispaced grid; adaptive construction "
                "from a function handle is not supported "
                "(MATLAB CHEBFUN:CHEBFUN:parseInputs:equi).")
        from chebfunjax.utils.interpolation import funqui

        handle = funqui(jnp.asarray(f))
        _de = [float(v) for v in (domain if hasattr(domain, "__len__")
                                  else (domain,))]
        if len(_de) < 2:
            _de = [-1.0, 1.0]
        a_e, b_e = _de[0], _de[-1]
        if (a_e, b_e) == (-1.0, 1.0):
            op_e = handle
        else:
            def op_e(x, h=handle, a=a_e, b=b_e):
                return h((2.0 * x - (a + b)) / (b - a))
        return Chebfun.from_function(
            op_e, Domain((a_e, b_e)), n=n, maxpow2=_maxpow2, tol=_tol)

    # --- Parse domain ---
    dom_seq = [float(x) for x in domain]
    if len(dom_seq) < 2:
        raise ValueError(
            f"domain must have at least 2 elements, got {len(dom_seq)}. "
            f"Example: domain=(-1, 1)."
        )
    dom = Domain(tuple(dom_seq))

    # --- Dispatch on f type ---
    # Empty chebfun: chebfun(), chebfun([]), or n=0 (MATLAB isempty
    # semantics -- no pieces; most operations are undefined on it).
    is_empty_arg = (
        f is None
        or (hasattr(f, "__len__") and not callable(f) and len(f) == 0)
        or (n is not None and n == 0)
    )
    if is_empty_arg:
        return Chebfun(funs=[], domain=dom)

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

    _dom_arr = [float(v) for v in (domain if hasattr(domain, "__len__")
                                    else (domain,))]
    if any(not math.isfinite(v) for v in _dom_arr):
        from chebfunjax.fun.unbndfun import Unbndfun

        if trig:
            raise ValueError("trig=True is not supported on unbounded domains.")
        if not callable(f):
            raise ValueError("Unbounded domains require a callable.")
        if len(_dom_arr) != 2:
            raise ValueError(
                "Unbounded domains support a single interval only."
            )
        dom_u = Domain((_dom_arr[0], _dom_arr[1]))
        fun_u = Unbndfun.from_function(f, domain=dom_u, n=n)
        return Chebfun(funs=[fun_u], domain=dom_u)

    if trig:
        from chebfunjax.tech.trigtech import Trigtech

        if not callable(f):
            raise ValueError("chebfun(..., trig=True) requires a callable.")
        dom_arr = tuple(float(v) for v in domain)
        if len(dom_arr) != 2:
            raise ValueError(
                "chebfun(..., trig=True) supports a single interval only."
            )
        a, b = dom_arr

        def f_ref(x):
            return f(a + (b - a) * (x + 1.0) / 2.0)

        tech = Trigtech.from_function(f_ref, n=n)
        piece = _Piece(tech=tech, interval=(a, b))
        return Chebfun(funs=[piece], domain=Domain((a, b)))

    if callable(f):
        if splitting and n is None:
            return _construct_with_splitting(f, float(dom_seq[0]),
                                             float(dom_seq[-1]),
                                             _maxpow2, tol=_tol, turbo=turbo)
        return Chebfun.from_function(f, dom, n=n, maxpow2=_maxpow2,
                                     tol=_tol, turbo=turbo)

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


def _two_arg_extremum(f: "Chebfun", other, pick):
    """Pointwise max/min of two chebfuns via breakpoints at crossings.

    ``pick`` is ``jnp.maximum`` or ``jnp.minimum``.  Crossings are the
    roots of ``f - other``; splitting the domain there makes each piece
    smooth.  Added by Claude Opus 4.8 (task #14).
    """
    import numpy as _np

    a = float(f.domain.a)
    b = float(f.domain.b)

    if isinstance(other, Chebfun):
        def g_eval(x):
            return other(x)
        diff = f - other
    else:
        c = float(other)

        def g_eval(x):
            return jnp.full_like(jnp.asarray(x, dtype=jnp.float64), c)
        diff = f - c

    # interior crossing points
    roots = _np.asarray(diff.roots())
    roots = roots[(roots > a + 1e-13) & (roots < b - 1e-13)]
    # also carry over any existing breakpoints of the inputs
    brks = set(float(x) for x in f.domain.breakpoints)
    if isinstance(other, Chebfun):
        brks |= set(float(x) for x in other.domain.breakpoints)
    brks |= set(float(r) for r in roots)
    brks = sorted(x for x in brks if a + 1e-13 < x < b - 1e-13)
    domain = [a, *brks, b]

    def ev(x):
        return pick(f(x), g_eval(x))

    return chebfun(ev, domain=tuple(domain))


def _split_breakpoints(f, a: float, b: float, maxpow2: int,
                       depth: int = 0, max_depth: int = 45,
                       min_w: float = 1e-10) -> list:
    """Recursively find interior breakpoints for splitting-on (Opus 4.8).

    Detection is capped at 2^12 points: a piece containing a
    singularity does not resolve below that length, so it is flagged
    unhappy and split, instead of being accepted as a ~10k-coefficient
    grind that straddles the edge.  The final smooth pieces are still
    built at the caller's full ``maxpow2``.
    """
    import warnings as _warnings
    # MATLAB splitting caps each FUN at pref.splitPrefs.splitLength (= 257,
    # i.e. 2**8 + 1): a piece that is not resolved by 257 points is declared
    # sad and split, rather than ground up to 2**16 + 1 = 65537 points.  The
    # previous cap of 2**12 = 4097 made an endpoint branch-point singularity
    # (e.g. sqrt(4-(x-1)^2), which is ~2*sqrt(1+x) at x=-1) thrash: every
    # detection and edge-bisection construction ran to thousands of points and
    # the recursion hung for minutes.  MATLAB source: @chebfunpref splitLength.
    det = min(maxpow2, 8)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        p = _Piece.from_function(f, a + 1e-9 * (b - a), b - 1e-9 * (b - a),
                                 maxpow2=det)
    if p.ishappy or (b - a) < min_w or depth > max_depth:
        return []
    # Locate the singularity with a cheap finite-difference scan (the spirit of
    # MATLAB @fun/detectEdge: sample on a grid, take the largest first
    # difference) instead of the old 80-iteration construction-based bisection
    # (_split_find_edge did 160 adaptive constructions PER recursion level,
    # which -- combined with the recursion toward an endpoint branch point --
    # made splitting-on hang for minutes).
    e = _split_edge_fd(f, a, b)
    w = b - a
    htol = 1e-12 * max(abs(a), abs(b), 1.0)
    if e <= a + htol:
        # Singularity on the LEFT boundary (e.g. the sqrt branch point of
        # sqrt(4-(x-1)^2) at x=-1).  MATLAB detectEdge moves a boundary edge in
        # by diff(dom)/100; the boundary-adjacent child then peels off ~1% at a
        # time (geometric, a handful of levels) instead of halving ~35 times.
        e = a + w / 100
    elif e >= b - htol:
        e = b - w / 100
    elif not (a < e < b):
        e = 0.5 * (a + b)
    return (_split_breakpoints(f, a, e, maxpow2, depth + 1, max_depth, min_w)
            + [e]
            + _split_breakpoints(f, e, b, maxpow2, depth + 1, max_depth,
                                 min_w))


def _split_edge_fd(f, a: float, b: float, n: int = 17) -> float:
    """Cheap finite-difference singularity locator for splitting (Fable 5).

    Iteratively zooms into the sub-interval carrying the largest first
    difference of ``f`` -- a fast stand-in for @fun/detectEdge that needs only
    ``O(n)`` function evaluations per zoom (no adaptive constructions).  A jump
    or branch point produces the dominant first difference, so the bracket
    closes geometrically on the singularity; the returned point is precise
    enough (~1e-14) to place a breakpoint exactly on a jump (so ``sign(x)``
    splits into exactly two pieces), while a boundary singularity converges to
    the endpoint and is handled by the caller's move-in rule.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m (findMaxDer / findJump)
    Chebfun commit: 7574c77
    """
    import numpy as _np
    lo, hi = float(a), float(b)
    scale = max(abs(a), abs(b), 1.0)
    for _ in range(80):
        if hi - lo < 1e-15 * scale:
            break
        xs = _np.linspace(lo, hi, n)
        ys = _np.asarray(f(jnp.asarray(xs)), dtype=_np.float64)
        ys = _np.where(_np.isfinite(ys), ys, 0.0)
        # A jump (0th-derivative discontinuity) shows an ISOLATED spike in the
        # first difference and localises cleanly there; a kink (1st-derivative
        # discontinuity, e.g. abs(x-c)) leaves the first difference ~uniform but
        # spikes the second difference.  Pick the first difference when it
        # already isolates a jump, otherwise the second.
        d1 = _np.abs(_np.diff(ys))
        med1 = _np.median(d1) if d1.size else 0.0
        if d1.size and _np.max(d1) > 4.0 * med1 + 1e-300:
            centres = 0.5 * (xs[:-1] + xs[1:])
            score = d1
        else:
            score = _np.abs(_np.diff(ys, n=2))
            centres = xs[1:-1]
        i = int(_np.argmax(score))
        step = (hi - lo) / (n - 1)
        nlo = max(lo, centres[i] - step)
        nhi = min(hi, centres[i] + step)
        if (nhi - nlo) >= (hi - lo) * (1.0 - 1e-12):
            # Bracket no longer shrinking: settle on the peak location.
            lo, hi = float(nlo), float(nhi)
            break
        lo, hi = float(nlo), float(nhi)
    return 0.5 * (lo + hi)


def _construct_with_splitting(f, a: float, b: float, maxpow2: int,
                              tol=None, turbo: bool = False):
    """Build a piecewise Chebfun, auto-detecting breakpoints (Opus 4.8, #12).

    Each piece is constructed on a slightly-shrunk interval so that at a
    jump the piece captures the one-sided limit (not the ambiguous value
    exactly at the breakpoint, e.g. sign(0)=0).
    """
    import warnings as _warnings
    brks = _split_breakpoints(f, a, b, maxpow2)
    # Always keep the true domain endpoints a and b; merge only INTERIOR
    # breakpoints, and drop any interior point that lands within the merge
    # tolerance of EITHER neighbour (previously a geometric peel breakpoint a
    # rounding step inside b could displace b itself, yielding a domain like
    # [-1, -1e-12] instead of [-1, 0]).
    interior = sorted(float(x) for x in brks if a < float(x) < b)
    cleaned = [float(a)]
    for x in interior:
        if x - cleaned[-1] > 1e-11 and (b - x) > 1e-11:
            cleaned.append(x)
    cleaned.append(float(b))

    funs = []
    for i in range(len(cleaned) - 1):
        ai, bi = cleaned[i], cleaned[i + 1]
        w = bi - ai
        # shrink only at interior breakpoints (jumps); keep true domain
        # endpoints exact.
        lo = ai + (1e-11 * w if i > 0 else 0.0)
        hi = bi - (1e-11 * w if i < len(cleaned) - 2 else 0.0)

        def f_ref(t, _lo=lo, _hi=hi):
            x = 0.5 * (_hi - _lo) * t + 0.5 * (_lo + _hi)
            return f(x)

        # Splitting-mode pieces are capped at MATLAB's splitLength (2**8 + 1
        # = 257): the recursion has already subdivided until each piece either
        # resolves within that budget or is a minimal-width singular piece that
        # is accepted unresolved.  Building at the caller's full maxpow2 (2**16)
        # would re-grind the near-singular pieces to 65537 points.
        piece_maxpow2 = min(maxpow2, 8)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            tech = Chebtech2.from_function(f_ref, maxpow2=piece_maxpow2,
                                           tol=tol, turbo=turbo)
        funs.append(_Piece(tech=tech, interval=(float(ai), float(bi))))
    return Chebfun(funs=funs, domain=Domain(tuple(cleaned)))


def _integer_step(f: "Chebfun", op, half_offset: bool = False):
    """Piecewise-constant floor/ceil/round of a Chebfun (Opus 4.8, #14).

    Breakpoints are the points where ``f`` (or ``f - 1/2`` for round)
    crosses an integer; between them ``op(f)`` is constant.
    """
    if f.isempty():
        return Chebfun.empty()
    import numpy as _np

    a = float(f.domain.a)
    b = float(f.domain.b)
    # sample to bound the range of f
    xs = _np.linspace(a, b, 257)
    fv = _np.asarray(f(jnp.asarray(xs)))
    lo = int(_np.floor(fv.min())) - 1
    hi = int(_np.ceil(fv.max())) + 1

    brks = set(float(x) for x in f.domain.breakpoints)
    shift = 0.5 if half_offset else 0.0
    for k in range(lo, hi + 1):
        # crossings of f = k + shift
        r = _np.asarray((f - (k + shift)).roots())
        for rr in r:
            rr = float(rr)
            if a + 1e-12 < rr < b - 1e-12:
                brks.add(rr)
    brks = sorted(x for x in brks if a + 1e-12 < x < b - 1e-12)
    domain = _np.array([a, *brks, b])

    # Each piece is exactly constant = op(f(midpoint)); build the pieces
    # directly as degree-0 Chebtechs so shared breakpoints (which sit on
    # the jump) don't corrupt the fit.
    mids = 0.5 * (domain[:-1] + domain[1:])
    consts = _np.asarray(op(f(jnp.asarray(mids))), dtype=_np.float64)
    funs = [
        _Piece.from_coeffs(jnp.array([float(consts[i])], dtype=jnp.float64),
                           float(domain[i]), float(domain[i + 1]))
        for i in range(len(consts))
    ]
    return Chebfun(funs=funs, domain=Domain(tuple(float(x) for x in domain)))


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


# ============================================================================
# Higher-order ODE integrators: ode78 / ode89
# ============================================================================


def ode78(
    odefun: "Callable[[float, jax.Array], jax.Array]",
    tspan: "tuple[float, float]",
    y0: "jax.Array",
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    dense_n: int | None = None,
    **kwargs,
) -> "Chebfun":
    """Solve a non-stiff IVP using a 7(8)-order Runge-Kutta method.

    Wraps ``scipy.integrate.solve_ivp`` with the ``DOP853`` (8th-order
    Dormand-Prince) method — the closest available Python analogue of
    MATLAB's ``ode78`` — and interpolates the dense output onto a Chebfun.

    Parameters
    ----------
    odefun : callable(t, y) -> array_like
        Right-hand side of the ODE.
    tspan : (float, float)
        Integration interval ``(t0, tf)``.
    y0 : array_like, shape (d,) or scalar
        Initial state.
    rtol : float, default 1e-8
        Relative tolerance (tighter than ode45/ode113 defaults).
    atol : float, default 1e-10
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
    >>> from chebfunjax.chebfun1d.chebfun import ode78
    >>> import jax.numpy as jnp
    >>> sol = ode78(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
    >>> abs(float(sol(jnp.float64(1.0))) - float(jnp.exp(jnp.float64(1.0)))) < 1e-6
    True

    Notes
    -----
    MATLAB's ``ode78`` uses a specific 7(8)-order pair by Fehlberg.  Python's
    SciPy does not provide this exact method; ``DOP853`` is an 8th-order
    Dormand-Prince scheme that offers equivalent or better accuracy.

    Provenance
    ----------
    MATLAB source : @chebfun/ode78.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ode45 : Dormand-Prince RK45
    ode89 : Verner 8(9) integrator
    ode113 : Adams/DOP853 integrator
    """
    return _ode_solve("DOP853", odefun, tspan, y0,
                      rtol=rtol, atol=atol, dense_n=dense_n, **kwargs)


def ode89(
    odefun: "Callable[[float, jax.Array], jax.Array]",
    tspan: "tuple[float, float]",
    y0: "jax.Array",
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    dense_n: int | None = None,
    **kwargs,
) -> "Chebfun":
    """Solve a non-stiff IVP using an 8(9)-order Runge-Kutta method.

    Wraps ``scipy.integrate.solve_ivp`` with the ``DOP853`` method at
    very tight tolerances — the closest available Python analogue of
    MATLAB's ``ode89`` (Verner 8(9) method).

    Parameters
    ----------
    odefun : callable(t, y) -> array_like
        Right-hand side of the ODE.
    tspan : (float, float)
        Integration interval ``(t0, tf)``.
    y0 : array_like, shape (d,) or scalar
        Initial state.
    rtol : float, default 1e-10
        Relative tolerance (tighter than ode78).
    atol : float, default 1e-12
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
    >>> from chebfunjax.chebfun1d.chebfun import ode89
    >>> import jax.numpy as jnp
    >>> sol = ode89(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
    >>> abs(float(sol(jnp.float64(1.0))) - float(jnp.exp(jnp.float64(1.0)))) < 1e-8
    True

    Notes
    -----
    MATLAB's ``ode89`` uses the Verner 8(9) pair.  SciPy does not expose
    this specific pair; we use DOP853 (Dormand-Prince 8th-order) with
    very tight tolerances as the closest analogue.

    Provenance
    ----------
    MATLAB source : @chebfun/ode89.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ode45, ode78, ode113
    """
    return _ode_solve("DOP853", odefun, tspan, y0,
                      rtol=rtol, atol=atol, dense_n=dense_n, **kwargs)


# ============================================================================
# Module-level convenience wrappers for Chebfun methods
# ============================================================================


def innerProduct(f: "Chebfun", g: "Chebfun") -> "jax.Array":
    r"""L2 inner product of two Chebfuns.

    Computes :math:`\langle f, g \rangle = \int_a^b f(x)\,g(x)\,dx`.

    This is a module-level alias for :meth:`Chebfun.inner` (which is also
    accessible as :meth:`Chebfun.innerProduct`).

    Parameters
    ----------
    f : Chebfun
    g : Chebfun

    Returns
    -------
    jax.Array (scalar)

    Provenance
    ----------
    MATLAB source : @chebfun/innerProduct.m
    Chebfun commit: 7574c77
    """
    return f.inner(g)


# ============================================================================
# Lagrange interpolation basis
# ============================================================================


def lagrange(
    x: "jax.Array | list[float]",
    domain: "tuple[float, float] | None" = None,
) -> "list[Chebfun]":
    r"""Compute the Lagrange basis polynomials for interpolation nodes ``x``.

    Returns a list of ``n`` Chebfuns ``[L_0, L_1, ..., L_{n-1}]`` where
    ``n = len(x)``.  Each :math:`L_j` is the unique polynomial of degree
    ``n-1`` satisfying:

    .. math::

        L_j(x_k) = \delta_{jk}  \quad (k = 0, \ldots, n-1)

    Parameters
    ----------
    x : array_like, shape (n,)
        Interpolation nodes.  Must be distinct.
    domain : (float, float) or None
        Spatial domain for the Chebfun.  If ``None``, uses
        ``[min(x), max(x)]``.  Must be supplied when ``x`` has length 1.

    Returns
    -------
    basis : list of Chebfun, length n
        The Lagrange basis polynomials.

    Raises
    ------
    ValueError
        If nodes are not distinct or ``x`` is a scalar without a domain.

    Examples
    --------
    >>> import jax.numpy as jnp, numpy as np
    >>> from chebfunjax.chebfun1d.chebfun import lagrange
    >>> nodes = [-1.0, 0.0, 1.0]
    >>> basis = lagrange(nodes)
    >>> len(basis)
    3
    >>> # L_0(-1) == 1,  L_0(0) == 0,  L_0(1) == 0
    >>> abs(float(basis[0](jnp.float64(-1.0))) - 1.0) < 1e-12
    True
    >>> abs(float(basis[0](jnp.float64(0.0)))) < 1e-12
    True

    Notes
    -----
    Each basis polynomial is built by constructing the identity matrix
    ``y = eye(n)`` column-by-column and calling ``chebfun`` via barycentric
    interpolation at the nodes.

    NOT JIT-safe (adaptive construction).

    Provenance
    ----------
    MATLAB source : @chebfun/lagrange.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    # uses-numpy: barycentric interpolation uses NumPy arrays
    import numpy as _np

    x_np = _np.asarray(x, dtype=_np.float64).ravel()
    n = x_np.shape[0]

    if n == 0:
        return []

    if n == 1 and domain is None:
        raise ValueError(
            "lagrange: must supply ``domain`` when x is a scalar."
        )

    # Uniqueness check
    if _np.unique(x_np).shape[0] != n:
        raise ValueError("lagrange: interpolation nodes must be distinct.")

    if domain is None:
        a, b = float(x_np.min()), float(x_np.max())
    else:
        a, b = float(domain[0]), float(domain[1])

    # Sort nodes so that barycentric weights are well-defined
    idx = _np.argsort(x_np)
    x_sorted = x_np[idx]

    # Barycentric weights w_j = prod_{k != j} 1/(x_j - x_k)
    w = _np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (x_sorted[j] - x_sorted[k])

    # Build each basis polynomial via a Chebfun that passes through the
    # j-th column of the n×n identity matrix
    basis_sorted = []
    for j_sorted in range(n):
        yj = _np.zeros(n)
        yj[j_sorted] = 1.0

        # Barycentric interpolation as a callable.
        # Standard Type-II barycentric formula with exact-node handling:
        # For t not equal to any x_k: L_j(t) = [w_j/(t-x_j)] / sum_k [w_k/(t-x_k)]
        # For t == x_k: L_j(t) = delta_{jk}  (exact node — return y_j[k] directly)
        def _Lj(t, _x=x_sorted, _w=w, _y=yj):
            t_np = _np.asarray(t, dtype=_np.float64).ravel()
            m = t_np.shape[0]
            result = _np.empty(m)
            for i in range(m):
                ti = t_np[i]
                # Check if ti coincides with any node
                diffs = ti - _x
                close_mask = _np.abs(diffs) < 1e-14 * max(1.0, _np.max(_np.abs(_x)))
                if _np.any(close_mask):
                    # At an exact node: return y_j at that node
                    result[i] = float(_y[_np.argmax(close_mask)])
                else:
                    # Standard barycentric formula
                    terms = _w / diffs
                    result[i] = float(_np.dot(_y, terms) / _np.sum(terms))
            return jnp.asarray(result, dtype=jnp.float64)

        basis_sorted.append(chebfun(_Lj, domain=(a, b)))

    # Invert the sort permutation to return basis in original node order
    inv_idx = _np.argsort(idx)
    return [basis_sorted[inv_idx[j]] for j in range(n)]


# ============================================================================
# Subspace angle
# ============================================================================


def subspace(
    A: "list[Chebfun]",
    B: "list[Chebfun]",
) -> float:
    """Principal angle between two quasimatrix subspaces.

    Computes the smallest principal angle (in radians) between the subspaces
    spanned by the columns of quasimatrix ``A`` and quasimatrix ``B``.  Both
    inputs are lists of Chebfuns on the same domain.

    Parameters
    ----------
    A : list of Chebfun
        Columns of the first quasimatrix.
    B : list of Chebfun
        Columns of the second quasimatrix.

    Returns
    -------
    theta : float
        Smallest principal angle in radians.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` are empty or have mismatched domains.

    Examples
    --------
    >>> import jax.numpy as jnp, numpy as np
    >>> from chebfunjax.chebfun1d.chebfun import chebfun, subspace
    >>> # Two identical 1-D subspaces → angle = 0
    >>> f = chebfun(jnp.sin)
    >>> theta = subspace([f], [f])
    >>> theta < 1e-10
    True

    Notes
    -----
    Algorithm (Bjorck & Golub 1973, Knyazev & Argentati 2002):

    1. Orthonormalise each collection via the continuous QR factorisation.
    2. Compute the Gram matrix :math:`C = Q_A^T Q_B` (inner-product matrix).
    3. The smallest singular value of C gives :math:`\\cos\\theta`.
    4. For small angles recompute via the sine formulation for accuracy.

    NOT JIT-safe (QR uses adaptive construction).

    Provenance
    ----------
    MATLAB source : @chebfun/subspace.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    References
    ----------
    [1] A. Bjorck & G. Golub, *Numerical methods for computing angles between
        linear subspaces*, Math. Comp. 27 (1973), pp. 579–594.
    [2] A. V. Knyazev and M. E. Argentati, *Principal Angles between Subspaces
        in an A-Based Scalar Product*, SIAM J. Sci. Comput., 23 (2002), 2009–2041.
    """
    # uses-numpy: QR / SVD use NumPy internally
    import numpy as _np

    from chebfunjax.chebfun1d.linalg import Quasimatrix, qr_quasimatrix

    if not A or not B:
        raise ValueError("subspace: A and B must be non-empty lists of Chebfun.")

    # Orthonormalise via continuous QR
    # Quasimatrix requires a Domain argument — extract from the first column
    domA = A[0].domain
    domB = B[0].domain
    qA = Quasimatrix(A, domA)
    qB = Quasimatrix(B, domB)
    QA, _ = qr_quasimatrix(qA)
    QB, _ = qr_quasimatrix(qB)

    pA = len(QA.cols)
    pB = len(QB.cols)

    # Build Gram matrix C_ij = <QA_i, QB_j>
    C = _np.zeros((pA, pB), dtype=_np.float64)
    for i, colA in enumerate(QA.cols):
        for j, colB in enumerate(QB.cols):
            C[i, j] = float(colA.inner(colB))

    # Singular values of C
    sv = _np.linalg.svd(C, compute_uv=False)
    cos_theta = float(_np.clip(sv.min(), 0.0, 1.0))

    if cos_theta < 0.8:
        return float(_np.arccos(cos_theta))
    else:
        # Sine formulation for small angles
        if pA <= pB:
            # sin_theta = ||QA - QB * C.T||
            # Compute QA - projection of QA onto QB
            # recontruct vector norms
            diff_cols = []
            for i, colA in enumerate(QA.cols):
                proj = None
                for j, colB in enumerate(QB.cols):
                    c_ij = C[i, j]
                    if proj is None:
                        proj = colB * c_ij
                    else:
                        proj = proj + colB * c_ij
                if proj is not None:
                    diff_cols.append(colA - proj)
            sin_theta = max(float(col.norm()) for col in diff_cols) if diff_cols else 0.0
        else:
            diff_cols = []
            for j, colB in enumerate(QB.cols):
                proj = None
                for i, colA in enumerate(QA.cols):
                    c_ij = C[i, j]
                    if proj is None:
                        proj = colA * c_ij
                    else:
                        proj = proj + colA * c_ij
                if proj is not None:
                    diff_cols.append(colB - proj)
            sin_theta = max(float(col.norm()) for col in diff_cols) if diff_cols else 0.0
        return float(_np.arcsin(_np.clip(sin_theta, 0.0, 1.0)))


# ============================================================================
# Quantum states (Schrödinger eigenstates)
# ============================================================================


def quantumstates(
    V: "Chebfun",
    n: int = 10,
    h: float = 0.1,
) -> "tuple[jax.Array, list[Chebfun]]":
    """Compute eigenstates of the time-independent Schrödinger equation.

    Solves :math:`Lu = \\lambda u` where the Schrödinger operator is
    :math:`L u(x) = -h^2 u''(x) + V(x)\\,u(x)` with zero (Dirichlet)
    boundary conditions at both ends of the domain of ``V``.

    Parameters
    ----------
    V : Chebfun
        Potential function.  The domain of ``V`` sets the spatial domain.
    n : int, default 10
        Number of eigenstates to compute.
    h : float, default 0.1
        Reduced Planck constant (small parameter).

    Returns
    -------
    eigenvalues : jax.Array, shape (n,)
        Eigenvalues (energy levels) in ascending order.
    eigenfunctions : list of Chebfun, length n
        Corresponding normalised eigenfunctions.

    Examples
    --------
    >>> import jax.numpy as jnp, numpy as np
    >>> from chebfunjax.chebfun1d.chebfun import chebfun, quantumstates
    >>> # Harmonic oscillator: V = x^2
    >>> x = chebfun(lambda t: t, domain=(-3.0, 3.0))
    >>> V = x ** 2
    >>> evals, efuns = quantumstates(V, n=3, h=0.1)
    >>> len(efuns)
    3
    >>> float(evals[0]) > 0  # ground state energy > 0
    True

    Notes
    -----
    The operator is discretised on a Chebyshev collocation grid of size
    ``max(n_grid, 2*(n+1))`` using the Chebyshev differentiation matrix.
    Eigenvalues are computed by ``scipy.linalg.eigh`` on the resulting
    generalised eigenvalue problem.  The boundary conditions are enforced
    by row replacement.

    NOT JIT-safe (uses NumPy/SciPy linear algebra).

    Provenance
    ----------
    MATLAB source : @chebfun/quantumstates.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (January 2012), University of Oxford.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.
    """
    # uses-numpy: Chebyshev differentiation matrix and scipy.linalg.eigh
    import numpy as _np
    from scipy.linalg import eigh as _eigh  # type: ignore[import]

    a = float(V.domain.a)
    b = float(V.domain.b)

    # Grid size — must resolve the potential and have enough eigenvalue room.
    # Nodes are Chebyshev-2 nodes in *ascending* order on [a, b].
    n_grid = max(100, 4 * n)
    N = n_grid - 1  # polynomial degree

    # Chebyshev-2 nodes in ascending order: x_k = cos(pi*(N-k)/N), k=0..N
    k = _np.arange(N + 1)
    x_ref = _np.cos(_np.pi * (N - k) / N)          # ascending: -1 to 1
    x_phys = 0.5 * (b - a) * x_ref + 0.5 * (a + b)  # ascending: a to b

    # Chebyshev differentiation matrix on [-1, 1] for *ascending* nodes.
    # c_k: endpoint nodes get weight 2, interior nodes get weight 1,
    # with alternating signs in ascending order: c_0 = 2*(-1)^0 = 2,
    # c_N = 2*(-1)^N, c_k = (-1)^k for interior k.
    c = _np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** k

    Xm = _np.tile(x_ref, (N + 1, 1))
    dX = Xm - Xm.T                          # dX[i,j] = x_i - x_j (ref)
    D_ref = (c[:, None] / c[None, :]) / _np.where(_np.abs(dX) < 1e-15, 1.0, dX)
    _np.fill_diagonal(D_ref, 0.0)
    _np.fill_diagonal(D_ref, -D_ref.sum(axis=1))

    # Scale to physical domain [a, b]
    scale = 2.0 / (b - a)
    D1 = scale * D_ref
    D2 = D1 @ D1  # second-derivative matrix on [a, b]

    # Schrödinger operator on the interior nodes (index 1..N-1).
    # Dirichlet BCs u(a)=u(b)=0 are imposed by restricting to interior nodes.
    D2_int = D2[1:-1, 1:-1]                        # (N-1) × (N-1)
    x_int = x_phys[1:-1]                           # interior physical nodes

    # Potential values at interior nodes
    V_vals = _np.asarray(V(jnp.array(x_int)), dtype=_np.float64)

    # Schrödinger operator restricted to interior: L = -h² D2_int + diag(V)
    L_int = -(h ** 2) * D2_int + _np.diag(V_vals)

    # L_int is symmetric by construction (D2 of Chebfun spec collocation is
    # symmetric up to floating-point error); symmetrise explicitly.
    L_sym = 0.5 * (L_int + L_int.T)

    # Solve the symmetric eigenvalue problem for the n smallest eigenvalues
    n_int = L_sym.shape[0]
    n_req = min(n, n_int)
    evals_n, evecs_int = _eigh(L_sym, subset_by_index=[0, n_req - 1])

    # Pad eigenvectors with zeros at boundary nodes
    evecs_full = _np.zeros((n_grid, n_req))
    evecs_full[1:-1, :] = evecs_int

    # Build Chebfun for each eigenfunction via barycentric interpolation
    # x_phys is in ascending order; jnp.interp needs ascending xp
    efuns = []
    for j in range(n_req):
        v = evecs_full[:, j].copy()
        # Normalise: discrete trapezoid rule as approximation to L2 norm
        norm_v = _np.sqrt(_np.trapezoid(v ** 2, x_phys))
        if norm_v > 1e-15:
            v /= norm_v
        # Make the dominant lobe positive (sign convention)
        if v[_np.argmax(_np.abs(v))] < 0:
            v = -v
        x_p_jax = jnp.array(x_phys)
        v_jax = jnp.array(v)
        efuns.append(
            chebfun(
                lambda t, _xp=x_p_jax, _vj=v_jax: jnp.interp(t, _xp, _vj),
                domain=(a, b),
            )
        )

    return jnp.array(evals_n, dtype=jnp.float64), efuns
