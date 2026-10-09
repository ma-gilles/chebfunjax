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
from typing import TYPE_CHECKING, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain, _linear_inverse_map
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.elementary import _atanh_log1p_real

if TYPE_CHECKING:
    from chebfunjax.chebfun1d.linalg import Quasimatrix

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)


def _delta_row(row):
    """Normalize a ``deltas`` row to ``(loc, mag, order)``.

    A 2-tuple is a plain Dirac impulse (order 0); a 3-tuple carries the
    distributional-derivative order (MATLAB @deltafun deltaMag rows).
    """
    if len(row) == 2:
        return float(row[0]), row[1], 0
    return float(row[0]), row[1], int(row[2])


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
    from chebfunjax.autodiff.adchebfun import ADChebfun

    if isinstance(f, ADChebfun):
        return f.jump(x, c)
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
        start_pow2: int = 4,
        extrapolate: bool = False,
        vscale: float = 0.0,
        sample_test: bool = True,
        refinement_function: str | Callable | None = None,
        min_samples: int | None = None,
        max_length: int | None = None,
        check: str = "standard",
        hscale: float = 1.0,
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
            Legacy adaptive grid power, used when max_length is omitted.
        max_length : int or None
            Raw source grid cap, including an oversized initial min_samples.
        tol : float or None
            Construction tolerance (``eps``); None uses machine epsilon.
        turbo : bool, default False
            Recompute the coefficients to high accuracy via the turbo
            contour integral (MATLAB ``'turbo'`` flag).
        """
        a, b = float(a), float(b)
        # Wrap f to map from reference [-1, 1] into [a, b]
        def f_ref(t: jax.Array) -> jax.Array:
            # @bndfun/bndfun.m skips the map on the reference interval.
            if a == -1.0 and b == 1.0:
                return f(t)
            # Literal @mapping/mapping.m linear.For preserves both endpoints.
            x = b * (t + 1) / 2 + a * (1 - t) / 2
            return f(x)

        tech_options = {}
        if refinement_function is not None:
            tech_options["refinement_function"] = refinement_function
        tech = Chebtech2.from_function(
            f_ref, n=n, maxpow2=maxpow2, tol=tol, start_pow2=start_pow2,
            turbo=turbo, extrapolate=extrapolate, vscale=vscale,
            sample_test=sample_test, min_samples=min_samples,
            max_length=max_length,
            check=check, hscale=hscale, **tech_options)
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

        Maps x from [a, b] to [-1, 1] with the source inverse map, then
        evaluates the tech. MATLAB source: @bndfun/feval.m and
        @mapping/mapping.m (linear), Chebfun commit 7574c77.

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
        t = _linear_inverse_map(x, a, b)
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
        from chebfunjax.tech.trigtech import Trigtech
        if isinstance(self.tech, Trigtech):
            # The trig grid excludes the right endpoint; by periodicity
            # f(b) == f(a), and MATLAB's get(f,'lval'/'rval') evaluates
            # the FUN at the endpoints.
            v0 = float(jnp.real(jnp.asarray(self.tech.values).ravel()[0]))
            return (v0, v0)
        exps = getattr(self.tech, "exponents", None)
        if exps is not None:
            sm = self.tech.smoothPart.values
            a_exp, b_exp = float(exps[0]), float(exps[1])
            out = []
            # At each endpoint the OTHER end's singular factor is finite
            # and equals 2**other_exp: f(-1) = s(-1)*2^b, f(1) = s(1)*2^a.
            for exp, v, other in ((a_exp, float(sm[0]), b_exp),
                                  (b_exp, float(sm[-1]), a_exp)):
                if exp < -1e-14:
                    out.append(math.copysign(math.inf, v if v != 0 else 1.0))
                elif exp > 1e-14:
                    out.append(0.0)
                else:
                    out.append(v * 2.0 ** other)
            return (out[0], out[1])
        vals = self.values
        # A complex-dtype piece with identically-zero imaginary part
        # (e.g. one straight horizontal scribble stroke) is displayed as
        # real by MATLAB; take the real part so float() accepts the dtype.
        if jnp.ndim(vals) == 2:
            # Array-valued piece: display the first column's endpoints.
            vals = vals[:, 0]
        v0, v1 = complex(vals[0]), complex(vals[-1])
        if v0.imag == 0 and v1.imag == 0:
            return (v0.real, v1.real)
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
        # A sub-interval that reaches the piece's own endpoint must map
        # EXACTLY onto +/-1: @singfun/restrict keeps the endpoint exponent
        # only when s(end) == 1 (s(1) == -1), and the affine map above can
        # round 7.0 -> 0.9999999999999999, which silently rebuilt the
        # singular tail as a 65537-point smooth piece (innerProduct test).
        _htol = 4.0 * _EPS * max(abs(pa), abs(pb), 1.0)
        if abs(a - pa) <= _htol:
            t_a = -1.0
        if abs(b - pb) <= _htol:
            t_b = 1.0
        from chebfunjax.fun.singfun import Singfun
        if isinstance(self.tech, Singfun):
            # @singfun/restrict takes a subinterval SEQUENCE.
            new_tech = self.tech.restrict([t_a, t_b])
        else:
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

    def _apply_fun(self, op, *, extrapolate: bool = False) -> _Piece:
        """Compose this piece with a scalar function op.

        Bounded Chebyshev pieces compose their canonical tech, preserving
        its kind and MATLAB's minimum-sample / sampleTest=false policy.
        The physical interval is retained.

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
            return _Piece(tech=self.tech.compose(op), interval=(a, b))
        from chebfunjax.tech.chebtech import Chebtech1
        if isinstance(self.tech, (Chebtech1, Chebtech2)):
            # @bndfun/compose.m delegates to its onefun. @chebtech/compose.m
            # covers the operand length and disables off-grid sampleTest.
            # Reconstructing in physical coordinates loses that contract
            # and the first-kind representation.
            if isinstance(self.tech, Chebtech2):
                return self.with_tech(
                    self.tech.compose(op, extrapolate=extrapolate))
            # Chebtech1's source resampling grid omits both endpoints and its
            # constructor has no extrapolate option. Preserve its exact
            # default compose path for either flag value.
            return self.with_tech(self.tech.compose(op))
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
        MATLAB source : @singfun/abs.m, @chebtech/abs.m
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
        from chebfunjax.tech.chebtech import Chebtech1
        if isinstance(tech, (Chebtech1, Chebtech2)) and (
                bool(jnp.all(jnp.imag(tech.coeffs) == 0))
                or bool(jnp.all(jnp.real(tech.coeffs) == 0))):
            # @chebtech/abs.m transforms the existing grid for real or pure
            # imaginary data after roots have been introduced as breaks.
            # Adaptive composition amplifies near-zero cancellation noise.
            cls = type(tech)
            values = cls.coeffs2vals(tech.coeffs)
            result = cls(coeffs=cls.vals2coeffs(jnp.abs(values)),
                         ishappy=tech.ishappy)
            return self.with_tech(result)
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
        from chebfunjax.fun.singfun import Singfun
        if isinstance(tech_der, Singfun):
            # Constant scaling of a Singfun acts on its smooth factor;
            # the exponents (already adjusted by @singfun/diff) stay.
            new_tech = Singfun(tech_der.smoothPart * jnp.float64(scale),
                               tech_der.exponents)
            return _Piece(tech=new_tech, interval=(a, b))
        # Scale the coefficients; rebuild with the DERIVATIVE's tech class
        # (not self's: a Singfun piece whose diff demoted to a smooth
        # Chebtech2 must rebuild as Chebtech2), which also keeps trig
        # pieces from reinterpreting Fourier coefficients as Chebyshev.
        scaled_coeffs = tech_der.coeffs * jnp.float64(scale)
        from chebfunjax.tech.trigtech import Trigtech
        if isinstance(tech_der, Trigtech):
            # MATLAB @trigtech/diff.m retains isReal and realifies values
            # for derivative columns that are known to represent reals.
            new_tech = type(tech_der).from_coeffs(
                scaled_coeffs, is_real=tech_der.is_real)
        else:
            new_tech = type(tech_der).from_coeffs(scaled_coeffs)
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
        import numpy as _np
        a, b = self.interval
        t_roots = self.tech.roots()
        # Map t in [-1, 1] to x in [a, b]: x = (b-a)/2 * t + (a+b)/2.
        # The map runs in numpy: rootfinding is not JIT-safe, and jnp
        # arithmetic here compiled one program per distinct root count.
        x_roots = (0.5 * (b - a) * _np.asarray(t_roots)
                   + 0.5 * (a + b))
        return jnp.asarray(x_roots)

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


def _real_simple_roots(den: "Chebfun"):
    """Interior real roots of a denominator chebfun (pole locations).

    Returns a sorted numpy array of the roots of ``den`` strictly
    inside its domain; endpoint roots are kept too (they become
    endpoint poles of the quotient).  Complex-valued denominators
    return an empty array (their zeros are generically off the real
    line, and MATLAB rdivide only singularises real crossings).
    """
    import numpy as _np
    if any(_np.iscomplexobj(_np.asarray(p.tech.coeffs))
           for p in den.funs):
        return _np.zeros(0)
    try:
        r = _np.asarray(den.roots())
    except Exception:
        return _np.zeros(0)
    if r.ndim != 1:
        return _np.zeros(0)
    return _np.sort(_np.real(r))


def _divide_with_poles(num, den: "Chebfun", poles) -> "Chebfun":
    """num/den where den vanishes at ``poles`` (MATLAB rdivide + singfun).

    Breakpoints are inserted at the poles and each piece is built as a
    SingFun whose (negative, integer or detected) endpoint exponents
    absorb the blow-up, mirroring @chebfun/rdivide.m with the default
    'blowup' behaviour used by operator arithmetic like 1./(1-x).

    Provenance
    ----------
    MATLAB source : @chebfun/rdivide.m, @singfun/singfun.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    from chebfunjax.fun.singfun import Singfun

    a0, b0 = float(den.domain.a), float(den.domain.b)
    tol = 1e-10 * max(b0 - a0, 1.0)
    # cluster nearly-equal roots: pole location + multiplicity
    raw = sorted(float(p) for p in _np.asarray(poles, dtype=float))
    clusters: list[list[float]] = [[raw[0]]]
    for v in raw[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    pole_locs = [float(_np.mean(c)) for c in clusters]
    pole_mult = {loc: len(c) for loc, c in zip(pole_locs, clusters)}

    bps = sorted(
        set(float(v) for v in den.domain.breakpoints)
        | set(pole_locs))
    merged = [bps[0]]
    for v in bps[1:]:
        if v - merged[-1] > tol:
            merged.append(v)
    merged[0], merged[-1] = a0, b0

    def _mult_at(x):
        for loc, m in pole_mult.items():
            if abs(x - loc) <= tol:
                return m
        return 0

    def _num_eval(x):
        if isinstance(num, Chebfun):
            return num(x)
        return jnp.asarray(num) * jnp.ones_like(jnp.asarray(x))

    new_funs = []
    for a, b in zip(merged[:-1], merged[1:]):
        half = (b - a) / 2.0
        mid = (a + b) / 2.0
        ml, mr = _mult_at(a), _mult_at(b)
        if ml == 0 and mr == 0:
            new_funs.append(_Piece.from_function(
                lambda x: _num_eval(x) / den(x), a, b))
            continue

        # smooth part: (num/den) * (1+t)^ml * (1-t)^mr on [-1, 1];
        # the singular factors are known structurally from the root
        # multiplicities (like extractBoundaryRoots in MATLAB), so no
        # eps-scale sampling detection is needed.
        def op(t, _half=half, _mid=mid):
            x = _mid + _half * t
            return _num_eval(x) / den(x)

        sf = Singfun.from_function(op, exponents=(-float(ml),
                                                  -float(mr)))
        new_funs.append(_Piece(tech=sf, interval=(a, b)))
    return Chebfun(funs=new_funs, domain=Domain(tuple(merged)))


def _np_vals2coeffs2(v):
    """2nd-kind Chebyshev values (ascending x) -> coefficients (numpy)."""
    import numpy as _np
    v = _np.asarray(v, dtype=float)
    n = len(v)
    if n <= 1:
        return v.copy()
    tmp = _np.concatenate([v[n - 1:0:-1], v[:n - 1]])
    c = _np.real(_np.fft.ifft(tmp))
    c = c[:n]
    c[1:n - 1] *= 2.0
    return c


def _np_leg2cheb(cl):
    """Legendre -> Chebyshev coefficients (numpy for small n; the
    lax.scan transform for large n)."""
    import numpy as _np
    cl = _np.asarray(cl, dtype=float).ravel()
    n = len(cl)
    if n == 0:
        return cl
    if n > 256:
        from chebfunjax.utils.transforms import leg2cheb as _l2c
        return _np.asarray(_l2c(jnp.asarray(cl)))
    tk = _np.cos(_np.pi * _np.arange(n - 1, -1, -1) / max(n - 1, 1))
    vals = _np.polynomial.legendre.legval(tk, cl)
    return _np_vals2coeffs2(_np.atleast_1d(vals))


def _np_cheb2leg(cc):
    """Chebyshev -> Legendre coefficients (numpy for small n; the
    O(n^2) lax.scan transform for large n — numpy leggauss is an
    O(n^3) eigensolve and hangs on unresolved 65537-point pieces)."""
    import numpy as _np
    cc = _np.asarray(cc, dtype=float).ravel()
    n = len(cc)
    if n <= 1:
        return cc.copy()
    if n > 256:
        from chebfunjax.utils.transforms import cheb2leg as _c2l
        return _np.asarray(_c2l(jnp.asarray(cc)))
    xg, wg = _np.polynomial.legendre.leggauss(n)
    fv = _np.polynomial.chebyshev.chebval(xg, cc)
    P = _np.zeros((n, len(xg)))
    P[0] = 1.0
    P[1] = xg
    for k in range(2, n):
        P[k] = ((2 * k - 1) * xg * P[k - 1]
                - (k - 1) * P[k - 2]) / k
    scal = (2 * _np.arange(n) + 1) / 2.0
    return scal * (P @ (wg * fv))


def _clenshaw_legendre(x, alpha):
    """Evaluate a Legendre series at points x (bndfun/conv.m)."""
    import numpy as _np
    n = len(alpha)
    b_old = _np.zeros_like(x)
    b_cur = _np.zeros_like(x)
    for k in range(n - 1, 0, -1):
        b_new = (alpha[k] + (2 * k + 1) / (k + 1) * x * b_cur
                 - (k + 1) / (k + 2) * b_old)
        b_old = b_cur
        b_cur = b_new
    return alpha[0] + x * b_cur - 0.5 * b_old


def _easy_conv(alpha, beta):
    """Legendre coefficients of the convolution of two Legendre series
    on the two triangular pieces (bndfun/conv.m easyConv; Hale &
    Townsend, Theorem 4.1)."""
    import numpy as _np
    alpha = _np.asarray(alpha, dtype=_np.float64).ravel()
    beta = _np.asarray(beta, dtype=_np.float64).ravel()
    if len(beta) > len(alpha):
        alpha, beta = beta, alpha
    MN = len(alpha) + len(beta)
    alpha = _np.concatenate([alpha, _np.zeros(MN - len(alpha))])
    # tridiagonal S (spdiags layout: band d, element row i col i+d = B[i])
    S = _np.zeros((MN, MN))
    sub = _np.concatenate([[1.0], 1.0 / (2 * _np.arange(1, MN) + 1)])
    sup = -1.0 / (2 * _np.arange(0, MN) + 1)
    for i in range(MN - 1):
        S[i + 1, i] = sub[i]        # subdiagonal: B[i] at (i+1, i)
        S[i, i + 1] = sup[i + 1]    # superdiagonal: B[i+1] at (i, i+1)
    S[0, 0] = 1.0

    def rec(S, alpha, beta, sgn):
        N = len(beta)
        scl = 1.0 / (2 * _np.arange(1, N + 1) - 1)
        scl[1::2] = -scl[1::2]
        vNew = S @ alpha
        v = vNew.copy()
        gamma = beta[0] * vNew
        beta_scl = scl * beta
        beta_scl[0] = 0.0
        gamma[0] += vNew[:N] @ beta_scl
        if N == 1:
            return gamma
        vNew = S @ v + sgn * v
        vOld = v
        v = vNew.copy()
        vNew = vNew.copy()
        vNew[0] = 0.0
        gamma = gamma + beta[1] * vNew
        beta_scl = -beta_scl * ((2 - 0.5) / (2 - 1.5))
        beta_scl[1] = 0.0
        gamma[1] += vNew[:N] @ beta_scl
        for n in range(3, N + 1):
            vNew = (2 * n - 3) * (S @ v) + vOld
            vNew[:n - 1] = 0.0
            gamma = gamma + vNew * beta[n - 1]
            beta_scl = -beta_scl * ((n - 0.5) / (n - 1.5))
            beta_scl[n - 1] = 0.0
            gamma[n - 1] += vNew[:N] @ beta_scl
            vOld = v
            v = vNew
        ag = _np.abs(gamma)
        mg = ag.max() if ag.size else 0.0
        if mg > 0:
            nz = _np.where(ag > _np.finfo(float).eps * mg)[0]
            gamma = gamma[:nz[-1] + 1] if nz.size else gamma[:1]
        return gamma

    gammaL = rec(S, alpha, beta, -1)
    S2 = S.copy()
    S2[0, 0] = -1.0
    gammaR = rec(S2, -alpha, beta, 1)
    return gammaL, gammaR


def _chebval_interval(c, interval, x):
    """Evaluate Chebyshev coeffs c on `interval` at physical points x."""
    import numpy as _np
    aa, bb = interval
    t = (2.0 * _np.asarray(x, dtype=float) - (aa + bb)) / (bb - aa)
    return _np.polynomial.chebyshev.chebval(t, _np.asarray(c))


def _restrict_cheb_coeffs(c, src, dst, nout=None):
    """Chebyshev coeffs of the restriction of (c on src) to dst."""
    import numpy as _np
    n = nout or max(len(c), 2)
    tk = _np.cos(_np.pi * _np.arange(n - 1, -1, -1) / (n - 1))
    xk = dst[0] + (dst[1] - dst[0]) * (tk + 1) / 2
    vals = _chebval_interval(c, src, xk)
    return _np_vals2coeffs2(_np.atleast_1d(vals))


def _cheb1_vals2coeffs(v):
    """First-kind Chebyshev values -> Chebyshev coefficients."""
    import numpy as _np
    v = _np.asarray(v, dtype=float)
    n = len(v)
    if n <= 1:
        return v.copy()
    # DCT-III inverse: c_k = (2/n) sum v_j cos(k (2j+1) pi / (2n)),
    # with points ordered ascending (theta descending)
    theta = (2 * _np.arange(n)[::-1] + 1) * _np.pi / (2 * n)
    K = _np.arange(n)
    C = _np.cos(_np.outer(K, theta))
    c = (2.0 / n) * (C @ v)
    c[0] *= 0.5
    return c


def _fun_conv_ht(fc, fint, gc, gint):
    """Convolution of two single smooth pieces (Chebyshev coeffs on
    intervals), returning [(interval, cheb_coeffs), ...].

    Faithful port of the Hale-Townsend algorithm of @bndfun/conv.m
    (patch decomposition; Legendre-coefficient triangle convolutions).
    """
    import numpy as _np

    a, b = float(fint[0]), float(fint[1])
    c, d = float(gint[0]), float(gint[1])
    if (b - a) > (d - c):
        return _fun_conv_ht(gc, gint, fc, fint)
    fc = _np.asarray(fc, dtype=float).ravel()
    gc = _np.asarray(gc, dtype=float).ravel()
    if len(fc) + len(gc) > 8200:
        import warnings as _w
        _w.warn("conv: truncating an unresolved piece "
                f"(lengths {len(fc)}, {len(gc)}) to 4096 "
                "coefficients; the input was not fully resolved.",
                stacklevel=3)
        fc = fc[:4096]
        gc = gc[:4096]
    N = max(len(gc), 2)
    numPatches = int(_np.floor((d - c) / (b - a)))
    # first-kind Chebyshev grid on interior [b+c, a+d]
    t1 = _np.cos((2 * _np.arange(N)[::-1] + 1) * _np.pi / (2 * N))
    x = (b + c) + ((a + d) - (b + c)) * (t1 + 1) / 2
    y = _np.zeros(N)

    def _map(xv, aa, bb):
        return (2.0 * _np.asarray(xv) - (aa + bb)) / (bb - aa)

    def _trim(cv):
        cv = _np.asarray(cv)
        acv = _np.abs(cv)
        m = acv.max() if acv.size else 0.0
        if m == 0:
            return cv[:1]
        nz = _np.where(acv > 10 * _np.finfo(float).eps * m)[0]
        return cv[:nz[-1] + 1] if nz.size else cv[:1]

    f_leg = _np_cheb2leg(_trim(fc))
    doms = c + (b - a) * _np.arange(numPatches + 1)
    pieces = []
    h_left = None
    h_right = None
    hLegR_last = None
    for k in range(numPatches):
        dk = (doms[k], doms[k + 1])
        dk_left = a + dk[0]
        dk_mid = a + dk[1]
        dk_right = b + dk[1]
        gk = _trim(_restrict_cheb_coeffs(gc, (c, d), dk))
        gk_leg = _np_cheb2leg(gk)
        hLegL, hLegR = _easy_conv(f_leg, gk_leg)
        hLegR_last = hLegR
        ind = (x >= dk_left) & (x < dk_mid)
        if k == 0:
            hL_cheb = _np_leg2cheb(hLegL)
            h_left = ((dk_left, dk_mid), hL_cheb)
        else:
            z = _map(x[ind], dk_left, dk_mid)
            y[ind] += _clenshaw_legendre(z, hLegL)
        if k < numPatches - 1:
            ind = (x >= dk_mid) & (x < dk_right)
            z = _map(x[ind], dk_mid, dk_right)
            y[ind] += _clenshaw_legendre(z, hLegR)
    hs = max(abs(a), abs(b), abs(c), abs(d), 1.0)
    if abs((b - a) - (d - c)) < 10 * _np.finfo(float).eps * hs:
        hR_cheb = _np_leg2cheb(hLegR_last)
        h_right = ((d + a, d + b), hR_cheb)
        h_mid = None
    else:
        finish = a + c + numPatches * (b - a)
        gk = _trim(_restrict_cheb_coeffs(gc, (c, d), (d - (b - a), d)))
        gk_leg = _np_cheb2leg(gk)
        hLegL, hLegR = _easy_conv(f_leg, gk_leg)
        hR_cheb = _np_leg2cheb(hLegR)
        h_right = ((d + a, d + b), hR_cheb)
        remainderWidth = d + a - finish
        domfk = (max(b - remainderWidth, a), b)
        domgk = (max(finish - b, c), min(d + a - b, d))
        if max(domfk[1] - domfk[0],
               domgk[1] - domgk[0]) > _np.finfo(float).eps:
            ind = x >= finish
            z = _map(x[ind], d - b + 2 * a, d + a)
            y[ind] = _clenshaw_legendre(z, hLegL)
            fk = _trim(_restrict_cheb_coeffs(fc, (a, b), domfk))
            fk_leg = _np_cheb2leg(fk)
            gk2 = _trim(_restrict_cheb_coeffs(gc, (c, d), domgk))
            gk2_leg = _np_cheb2leg(gk2)
            _, hLegR2 = _easy_conv(fk_leg, gk2_leg)
            z = _map(x[ind], finish, d + a)
            y[ind] += (_clenshaw_legendre(z, hLegR2)
                       * remainderWidth / (b - a))
        y_cheb = _trim(_cheb1_vals2coeffs(y))
        h_mid = ((b + c, a + d), y_cheb)
    scale = (b - a) / 2.0
    for piece in (h_left, h_mid, h_right):
        if piece is not None:
            pieces.append((piece[0], piece[1] * scale))
    return pieces


def _assemble_pieces(contribs, out_dom=None):
    """Sum piecewise-polynomial contributions [(interval, coeffs)]
    into a single Chebfun (exact polynomial resampling per cell)."""
    import numpy as _np
    tol0 = 1e-13
    bps = set()
    for (aa, bb), _cv in contribs:
        bps.add(float(aa))
        bps.add(float(bb))
    if out_dom is not None:
        bps.add(float(out_dom[0]))
        bps.add(float(out_dom[1]))
    bps = sorted(bps)
    hs = max(abs(bps[0]), abs(bps[-1]), 1.0)
    merged = [bps[0]]
    for v in bps[1:]:
        if v - merged[-1] > tol0 * hs:
            merged.append(v)
    funs = []
    for aa, bb in zip(merged[:-1], merged[1:]):
        nmax = 2
        for (ca, cb), cv in contribs:
            if ca <= aa + tol0 * hs and cb >= bb - tol0 * hs:
                nmax = max(nmax, len(cv))
        tk = _np.cos(_np.pi * _np.arange(nmax - 1, -1, -1)
                     / (nmax - 1))
        xk = aa + (bb - aa) * (tk + 1) / 2
        vals = _np.zeros(nmax)
        for (ca, cb), cv in contribs:
            if ca <= aa + tol0 * hs and cb >= bb - tol0 * hs:
                vals += _chebval_interval(cv, (ca, cb), xk)
        cc = _np_vals2coeffs2(vals)
        acc = _np.abs(cc)
        mcc = acc.max() if acc.size else 0.0
        if mcc > 0:
            nz = _np.where(acc > 1e-15 * mcc)[0]
            cc = cc[:nz[-1] + 1] if nz.size else cc[:1]
        tech = Chebtech2.from_coeffs(jnp.asarray(cc))
        funs.append(_Piece(tech=tech, interval=(float(aa), float(bb))))
    dom = Domain(tuple(float(v) for v in merged))
    return Chebfun(funs=funs, domain=dom)


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
    # Stored breakpoint values are numerical state, so they must survive
    # passing a Chebfun through JAX transforms as a PyTree argument.
    _point_values: jax.Array | None = None

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
        self._point_values = None

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

    @staticmethod
    def tol_union(A, B, tol=None):
        """Tolerance-aware union used when assembling inverse breakpoints.

        This lazily imports the implementation to avoid a module cycle with
        the inverse constructor.

        Provenance
        ----------
        MATLAB source : @chebfun/tolUnion.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.

        JAX note
        --------
        The returned array has data-dependent length, so this helper is eager
        only; its arithmetic and result are JAX arrays.
        """
        from chebfunjax.chebfun1d.inverse import tol_union

        return tol_union(A, B, tol)

    tolUnion = tol_union

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
        _pv = getattr(self, "_point_values", None)
        if _pv is not None:
            object.__setattr__(new, "_point_values", _pv)
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

    def define_point(self, s, v) -> "Chebfun":
        """Assign function value(s) at point(s) (MATLAB ``f(s) = v``).

        Introduces breakpoints at each ``s`` (splitting the containing
        pieces; smooth pieces are restricted exactly) and records ``v``
        in the ``pointValues`` metadata at those breakpoints.  Point
        evaluation away from the assigned points is unchanged.

        Parameters
        ----------
        s : float or sequence of float
            Assignment point(s); must lie inside the domain.
        v : float or sequence of float
            Value(s); a scalar broadcasts over all points.

        Returns
        -------
        Chebfun

        Provenance
        ----------
        MATLAB source : @chebfun/definePoint.m (columnDefinePoint)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.fun.unbndfun import Unbndfun as _Ub
        s_arr = _np.atleast_1d(_np.asarray(s, dtype=_np.float64))
        v_arr = _np.asarray(v, dtype=float)
        ncols = self.n_columns
        if v_arr.size == 0:
            raise ValueError(
                "CHEBFUN:CHEBFUN:definePoint:columnDefinePoint:conversion: "
                "Cannot assign empty values to points.")
        # MATLAB columnDefinePoint value-shape rules: a scalar fills
        # every point and column; a row of numCols values is repeated
        # for every point; for a scalar-valued chebfun a vector matches
        # the points one-to-one; otherwise numel(s) x numCols is required.
        if v_arr.size == 1:
            v_arr = _np.full((s_arr.size, ncols), float(v_arr.reshape(-1)[0]))
        elif v_arr.ndim == 1 and ncols > 1 and v_arr.size == ncols:
            v_arr = _np.tile(v_arr[None, :], (s_arr.size, 1))
        elif ncols == 1 and v_arr.size == s_arr.size:
            v_arr = v_arr.reshape(-1, 1)
        elif v_arr.ndim == 2 and v_arr.shape == (s_arr.size, ncols):
            pass
        else:
            raise ValueError(
                "CHEBFUN:CHEBFUN:definePoint:columnDefinePoint:dimensions: "
                "Subscripted assignment dimension mismatch.")
        a, b = float(self.domain.a), float(self.domain.b)
        if _np.min(s_arr) < a or _np.max(s_arr) > b:
            raise ValueError(
                "define_point: cannot introduce points outside the domain.")

        # Introduce the new breakpoints piece by piece.
        new_funs: list = []
        new_bps: list[float] = [float(self.domain.breakpoints[0])]
        for p_ in self.funs:
            pa, pb = float(p_.interval[0]) if hasattr(p_, "interval") \
                else float(p_.domain.a), None
            if hasattr(p_, "interval"):
                pa, pb = float(p_.interval[0]), float(p_.interval[1])
            else:
                pa, pb = float(p_.domain.a), float(p_.domain.b)
            cuts = sorted({float(t) for t in s_arr if pa < t < pb})
            if not cuts:
                new_funs.append(p_)
                new_bps.append(pb)
                continue
            edges = [pa] + cuts + [pb]
            if isinstance(p_, _Ub):
                for rp in p_.restrict(tuple(edges)):
                    if isinstance(rp, _Ub):
                        new_funs.append(rp)
                    else:
                        # Bndfun -> _Piece (same Chebtech onefun + affine
                        # interval protocol).
                        new_funs.append(_Piece(
                            tech=rp.onefun,
                            interval=(float(rp.domain.a),
                                      float(rp.domain.b))))
            else:
                for ea, eb in zip(edges[:-1], edges[1:]):
                    new_funs.append(p_.restrict(ea, eb))
            new_bps.extend(edges[1:])
        out = Chebfun(funs=new_funs, domain=Domain(tuple(new_bps)),
                      deltas=self.deltas)

        # Record the assigned values in pointValues metadata (existing
        # overrides at untouched breakpoints are kept).
        pv = _np.asarray(out.point_values, dtype=float).copy()
        # MATLAB keeps f.pointValues at the untouched breakpoints (restrict
        # copies them); re-evaluating the split pieces there could differ
        # in the last bits (an Unbndfun at +/-inf).
        oldv = _np.asarray(self.point_values, dtype=float)
        old_bps = _np.asarray(list(self.domain.breakpoints), dtype=float)
        for k, t in enumerate(old_bps):
            j = int(_np.argmin(_np.abs(_np.asarray(new_bps) - t)))
            pv[j] = oldv[k]
        bps = _np.asarray(new_bps, dtype=float)
        if pv.ndim == 1 and ncols > 1:
            pv = pv[:, None]
        for t, val in zip(s_arr, v_arr):
            idx = int(_np.argmin(_np.abs(bps - t)))
            pv[idx] = val if pv.ndim == 2 else float(val[0])
        return out.set_point_values(jnp.asarray(pv))

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
        if override is None:
            return result
        import numpy as _np
        old_bps = [float(v) for v in self.domain.breakpoints]
        new_bps = [float(v) for v in result.domain.breakpoints]
        if old_bps == new_bps:
            object.__setattr__(result, "_point_values", op(override))
            return result
        # The op introduced breakpoints (sign/abs at roots): keep the
        # mapped values at the old breakpoints, the result's own values
        # at the new ones.
        pv = _np.array(result.point_values)
        mapped = _np.asarray(op(override))
        if _np.iscomplexobj(mapped) and not _np.iscomplexobj(pv):
            pv = pv.astype(_np.complex128)
        for k, t in enumerate(old_bps):
            if t in new_bps:
                pv[new_bps.index(t)] = mapped[k]
        object.__setattr__(result, "_point_values", jnp.asarray(pv))
        return result

    def _restrict_breaks(self, pts) -> "Chebfun":
        """MATLAB ``restrict(f, [s1 s2 ... sk])``: the restriction to
        ``[s1, sk]`` with breakpoints introduced at every ``s_j``."""
        pts = [float(v) for v in pts]
        funs: list = []
        bps: list[float] = [pts[0]]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            sub = self.restrict(a_, b_)
            funs.extend(sub.funs)
            bps.extend(float(v) for v in sub.domain.breakpoints[1:])
        return Chebfun(funs=funs, domain=Domain(tuple(bps)))

    def define_interval(self, sub_int, g) -> "Chebfun":
        """Redefine ``f`` on a subinterval (MATLAB ``f{a, b} = g`` /
        ``defineInterval``).

        ``g`` may be a Chebfun (restricted to ``sub_int``), a number or
        vector of numbers (one constant per column), or ``None``/empty,
        which REMOVES the subinterval and closes the gap by shifting the
        right part left.  A subinterval outside the current domain
        extends it, padding any gap with zero.

        Provenance
        ----------
        MATLAB source : @chebfun/defineInterval.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        sub = [float(v) for v in sub_int]
        if any(b_ - a_ <= 0 for a_, b_ in zip(sub[:-1], sub[1:])):
            raise ValueError(
                "CHEBFUN:CHEBFUN:defineInterval:invalidDomain: "
                "Not a valid domain.")
        ncols = self.n_columns if not self.isempty() else None
        remove = g is None or (
            not isinstance(g, Chebfun) and hasattr(g, "__len__")
            and len(_np.ravel(_np.asarray(g, dtype=object))) == 0)
        if not remove:
            if isinstance(g, Chebfun):
                g = g._restrict_breaks(sub)
            else:
                vals = _np.atleast_1d(_np.asarray(g, dtype=float)).ravel()
                if vals.size == 1 and ncols is not None and ncols > 1:
                    vals = _np.full(ncols, float(vals[0]))
                if vals.size == 1:
                    c0 = float(vals[0])
                    g = chebfun(lambda x, _c=c0: jnp.full_like(x, _c),
                                domain=tuple(sub))
                else:
                    cv = [float(v) for v in vals]
                    g = chebfun(
                        lambda x, _cv=cv: jnp.stack(
                            [jnp.full_like(x, c) for c in _cv], axis=-1),
                        domain=tuple(sub))
        if self.isempty():
            return g if not remove else self
        if not remove and g.n_columns != ncols:
            raise ValueError(
                "CHEBFUN:CHEBFUN:defineInterval:numCols: Dimensions of "
                "matrices being concatenated are not consistent.")

        bp = [float(v) for v in self.domain.breakpoints]
        fa, fb = bp[0], bp[-1]
        if not remove:
            gb = [float(v) for v in g.domain.breakpoints]
            if sub[-1] < fa:
                # Extension to the left (zero padding over the gap).
                funs = list(g.funs)
                dom = list(gb)
                if gb[-1] < fa:
                    pad = chebfun(lambda x: jnp.zeros_like(x),
                                  domain=(gb[-1], fa))
                    funs.extend(pad.funs)
                    dom.append(fa)
                funs.extend(self.funs)
                dom.extend(bp[1:])
                return Chebfun(funs=funs, domain=Domain(tuple(dom)))
            if sub[0] > fb:
                funs = list(self.funs)
                dom = list(bp)
                if fb < gb[0]:
                    pad = chebfun(lambda x: jnp.zeros_like(x),
                                  domain=(fb, gb[0]))
                    funs.extend(pad.funs)
                    dom.append(gb[0])
                funs.extend(g.funs)
                dom.extend(gb[1:] if dom[-1] == gb[0] else gb)
                return Chebfun(funs=funs, domain=Domain(tuple(dom)))
            funs, dom = [], []
            if sub[0] > fa:
                left = self.restrict(fa, sub[0])
                funs.extend(left.funs)
                dom.extend(float(v) for v in left.domain.breakpoints[:-1])
            funs.extend(g.funs)
            dom.extend(gb)
            if sub[-1] < fb:
                right = self.restrict(sub[-1], fb)
                funs.extend(right.funs)
                dom.extend(float(v) for v in right.domain.breakpoints[1:])
            return Chebfun(funs=funs, domain=Domain(tuple(dom)))

        # Removal of a subinterval.
        if sub[-1] < fa or sub[0] > fb:
            raise ValueError(
                "CHEBFUN:CHEBFUN:defineInterval:badremoveinterval: "
                "Interval to be removed is outside the domain.")
        left = self.restrict(fa, sub[0]) if sub[0] > fa else None
        right = self.restrict(sub[-1], fb) if sub[-1] < fb else None
        if right is None:
            return left if left is not None else Chebfun.empty()
        if left is None:
            return right
        lb = [float(v) for v in left.domain.breakpoints]
        rb = [float(v) for v in right.domain.breakpoints]
        shift = lb[-1] - rb[0]
        new_ends = [v + shift for v in rb]
        shifted = [
            _Piece(tech=p.tech, interval=(new_ends[k], new_ends[k + 1]))
            for k, p in enumerate(right.funs)
        ]
        return Chebfun(funs=list(left.funs) + shifted,
                       domain=Domain(tuple(lb[:-1] + new_ends)))

    def find(self, return_cols: bool = False):
        """Locations where a logical Chebfun is nonzero (MATLAB ``find``).

        The nonzero set of a logical Chebfun (e.g. ``f == 1/2``) must be a
        finite set of breakpoints, recorded in ``point_values``; otherwise
        ``CHEBFUN:CHEBFUN:find:infset`` is raised.  For an array-valued
        Chebfun ``return_cols=True`` is required (MATLAB's two-output
        form) and ``(x, col)`` is returned with 0-based column indices.

        Provenance
        ----------
        MATLAB source : @chebfun/find.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        if self.isempty():
            e = jnp.asarray([], dtype=jnp.float64)
            return (e, e) if return_cols else e
        ncols = self.n_columns
        if ncols > 1 and not return_cols:
            raise ValueError(
                "CHEBFUN:CHEBFUN:find:arrout: Use two output arguments "
                "for array-valued CHEBFUN objects.")
        pv_all = _np.asarray(self.point_values, dtype=float)
        if pv_all.ndim == 1:
            pv_all = pv_all[:, None]
        bps = _np.asarray(list(self.domain.breakpoints), dtype=float)
        xs, cols = [], []
        for j in range(ncols):
            for piece in self.funs:
                c = _np.asarray(piece.tech.coeffs)
                cj = c[:, j] if c.ndim == 2 else c
                if _np.any(cj != 0):
                    raise ValueError(
                        "CHEBFUN:CHEBFUN:find:infset: Nonzero locations "
                        "are not a finite set.")
            xnew = bps[pv_all[:, j] != 0]
            xs.extend(float(v) for v in xnew)
            cols.extend([j] * xnew.size)
        x = jnp.asarray(xs, dtype=jnp.float64)
        if return_cols:
            return x, jnp.asarray(cols, dtype=jnp.int64)
        return x

    def points(self) -> jax.Array:
        """The grid points underlying each piece (MATLAB ``f.points``):
        Chebyshev points of the piece's kind (or equispaced points for
        a trig piece) mapped to the piece's interval, concatenated.

        Provenance
        ----------
        MATLAB source : @chebfun/subsref.m (points), @chebtech/points.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.chebtech import Chebtech1
        from chebfunjax.tech.trigtech import Trigtech
        from chebfunjax.utils.quadrature import chebpts
        out = []
        for piece in self.funs:
            a_, b_ = float(piece.interval[0]), float(piece.interval[1])
            n_ = int(piece.tech.coeffs.shape[0])
            if isinstance(piece.tech, Trigtech):
                y = -1.0 + 2.0 * jnp.arange(n_, dtype=jnp.float64) / n_
            elif isinstance(piece.tech, Chebtech1):
                y = chebpts(n_, kind=1)
            else:
                y = chebpts(n_, kind=2)
            out.append(a_ + (b_ - a_) * (y + 1.0) / 2.0)
        return jnp.concatenate(out) if out else jnp.asarray([])

    def range(self, dim: int | None = None):
        """``max - min`` of the Chebfun (MATLAB ``range``).

        Along the continuous dimension (the default for a column
        Chebfun) the result is a number, or one number per column; along
        the discrete dimension of an array-valued Chebfun it is the
        pointwise range across the columns, a Chebfun.  ``dim >= 3``
        gives the zero Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/range.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.asarray([], dtype=jnp.float64)
        ncols = self.n_columns
        if dim is None:
            dim = 1 if ncols > 1 else (2 if self.is_transposed else 1)
        if dim >= 3:
            return 0.0 * self
        along_x = (dim == 1) != self.is_transposed
        if along_x:
            (_xmin, fmin), (_xmax, fmax) = self.minandmax()
            r = jnp.asarray(fmax) - jnp.asarray(fmin)
            return r if ncols > 1 else float(r)
        if ncols == 1:
            return 0.0 * self
        cols = [self.extract_columns(j) for j in range(ncols)]
        mx, mn = cols[0], cols[0]
        for c in cols[1:]:
            mx = mx.maximum(c)
            mn = mn.minimum(c)
        out = mx - mn
        return out.transpose() if self.is_transposed else out

    def truncate(self, n: int) -> "Chebfun":
        """Truncate to ``n`` terms of the underlying series (MATLAB
        ``truncate``): the first ``n`` Chebyshev coefficients (computed
        by projection for a piecewise Chebfun) or, for a trig Chebfun,
        the ``n`` central Fourier coefficients; the result is a global
        polynomial / trigonometric polynomial on the same domain.

        Provenance
        ----------
        MATLAB source : @chebfun/truncate.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.trigtech import Trigtech
        bp = [float(v) for v in self.domain.breakpoints]
        a_, b_ = bp[0], bp[-1]
        if isinstance(self.funs[0].tech, Trigtech):
            c = self.trigcoeffs(int(n))
            return chebfun(c, domain=(a_, b_), coeffs=True, trig=True)
        c = self.chebcoeffs(int(n))
        return chebfun(c, domain=(a_, b_), coeffs=True)

    def ultracoeffs(self, *args) -> jax.Array:
        """Ultraspherical (Gegenbauer) expansion coefficients (MATLAB
        ``ultracoeffs(f, lam)`` or ``ultracoeffs(f, n, lam)``): the
        coefficients of ``f`` in the polynomials ``C^{(lam)}_k``,
        obtained from the Jacobi coefficients with
        ``alpha = beta = lam - 1/2`` and the standard rescaling.

        Provenance
        ----------
        MATLAB source : ultracoeffs.m
        Chebfun commit: 7574c77
        """
        from jax.scipy.special import gammaln
        if len(args) == 1:
            n, lam = None, float(args[0])
        elif len(args) == 2:
            n, lam = args
            lam = float(lam)
        else:
            raise TypeError("ultracoeffs(f, lam) or ultracoeffs(f, n, lam)")
        if lam <= 0:
            raise ValueError(
                "CHEBFUN:chebfun:ultrapoly:invalidLam: Ultraspherical "
                "polynomials are not defined for lambda <= 0.")
        if lam == 0.5:
            return self.legcoeffs(n)
        if lam == 1.0:
            return self.chebcoeffs(n, kind=2)
        ab = lam - 0.5
        c = self.jaccoeffs(n, ab, ab)
        N = int(c.shape[0]) - 1
        nn = jnp.arange(N + 1, dtype=jnp.float64)
        scl = (jnp.exp(gammaln(2 * lam) - gammaln(lam + 0.5))
               * jnp.exp(gammaln(lam + 0.5 + nn) - gammaln(2 * lam + nn)))
        return (scl[:, None] * c) if c.ndim == 2 else scl * c

    def heaviside(self) -> "Chebfun":
        """Heaviside step of the Chebfun, ``0.5 * (sign(f) + 1)``
        (MATLAB ``heaviside``).

        Provenance
        ----------
        MATLAB source : @chebfun/heaviside.m
        Chebfun commit: 7574c77
        """
        return 0.5 * (self.sign() + 1.0)

    def remove_deltas(self) -> "Chebfun":
        """Strip every Dirac-delta component, keeping the smooth part
        (MATLAB ``removeDeltas``).

        Provenance
        ----------
        MATLAB source : @chebfun/removeDeltas.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.deltafun import Deltafun
        funs = []
        for piece in self.funs:
            t = piece.tech
            if isinstance(t, Deltafun):
                t = t.funPart if hasattr(t, "funPart") else t
                piece = piece.with_tech(t) if hasattr(piece, "with_tech") \
                    else _Piece(tech=t, interval=piece.interval)
            funs.append(piece)
        out = Chebfun(funs=funs, domain=self.domain, deltas=())
        if self.is_transposed:
            object.__setattr__(out, "_is_transposed", True)
        return out

    def change_tech(self, tech) -> "Chebfun":
        """Rebuild ``f`` with a different underlying representation
        (MATLAB ``changeTech``): ``'trig'``/``'trigtech'`` for the
        Fourier basis, ``'chebtech2'`` / ``'chebtech1'`` for Chebyshev
        points of the second / first kind (a tech class is accepted
        too).  Returns ``f`` itself when it already uses ``tech``.

        Provenance
        ----------
        MATLAB source : @chebfun/changeTech.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.chebtech import Chebtech1
        from chebfunjax.tech.trigtech import Trigtech
        if self.isempty():
            return self
        if isinstance(tech, type):
            key = tech.__name__.lower()
        else:
            key = str(tech).lower().lstrip("@")
        if key in ("trig", "trigtech", "periodic"):
            target = Trigtech
        elif key in ("chebtech1", "1", "1st"):
            target = Chebtech1
        elif key in ("chebtech", "chebtech2", "2", "2nd"):
            target = Chebtech2
        else:
            raise ValueError(f"change_tech: unknown tech {tech!r}")
        cur = type(self.funs[0].tech)
        if cur is target:
            return self
        bp = tuple(float(v) for v in self.domain.breakpoints)
        if target is Trigtech:
            return chebfun(lambda x: self(x), domain=(bp[0], bp[-1]),
                           trig=True)
        return chebfun(lambda x: self(x), domain=bp,
                       chebkind=1 if target is Chebtech1 else 2)

    @property
    def end(self):
        """The value at the right endpoint (MATLAB ``f(end)``): a number
        for a scalar Chebfun, one number per column otherwise.

        Provenance
        ----------
        MATLAB source : @chebfun/end.m, @chebfun/subsref.m
        Chebfun commit: 7574c77
        """
        b_ = float(self.domain.breakpoints[-1])
        out = self(jnp.asarray(b_, dtype=jnp.float64))
        if jnp.ndim(out) != 0:
            return out
        return complex(out) if jnp.iscomplexobj(out) else float(out)

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
        extrapolate: bool = False,
        start_pow2: int = 4,
        sample_test: bool = True,
        refinement_function: str | Callable | None = None,
        min_samples: int | None = None,
        max_length: int | None = None,
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
        sample_test : bool, optional
            Enable the Chebtech off-grid construction check.
        refinement_function : str or callable, optional
            Adaptive refinement method forwarded to each piece.
        min_samples : int, optional
            Minimum initial adaptive grid size.
        max_length : int or None
            Raw maximum adaptive grid length, forwarded without rounding.

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
                                         turbo=turbo,
                                         extrapolate=extrapolate,
                                         start_pow2=start_pow2,
                                         sample_test=sample_test,
                                         refinement_function=refinement_function,
                                         min_samples=min_samples,
                                         max_length=max_length)
            funs.append(piece)
        out = cls(funs=funs, domain=domain)
        if all(math.isfinite(x) for x in domain.breakpoints):
            object.__setattr__(out, "_point_values",
                               _source_breakpoint_values(funs, domain.breakpoints, f))
        return out

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
        elif not isinstance(domain, Domain):
            domain = Domain(tuple(float(v) for v in domain))
        if domain.n_intervals != 1:
            raise ValueError(
                f"from_coeffs only supports single-interval domains, "
                f"but domain has {domain.n_intervals} intervals. "
                f"Construct pieces individually and combine."
            )
        # Chebtech2.from_coeffs promotes real/complex data without narrowing.
        coeffs = jnp.asarray(coeffs)
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
        elif not isinstance(domain, Domain):
            domain = Domain(tuple(float(v) for v in domain))
        if domain.n_intervals != 1:
            raise ValueError(
                f"from_values only supports single-interval domains, "
                f"but domain has {domain.n_intervals} intervals."
            )
        # Preserve complex data (MATLAB chebfuns are natively complex);
        # forcing float64 here silently discarded imaginary parts, e.g.
        # solutions of complex-shifted operators.
        values = jnp.asarray(values)
        if not jnp.iscomplexobj(values):
            values = values.astype(jnp.float64)
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
        from .linalg import Quasimatrix
        if isinstance(x, (Chebfun, Quasimatrix)):
            return self.compose_chebfun(x)
        # MATLAB deltafun/feval: the value AT a (zeroth-order) delta
        # location is +-inf (sign of the magnitude); one-sided limits
        # ignore the delta.
        if self.deltas and side is None and not isinstance(x, jax.core.Tracer):
            import numpy as _np
            xq = x
            if isinstance(x, str):
                xq = (float(self.domain.a) if x.lower() == "left"
                      else float(self.domain.b))
            base = Chebfun(funs=self.funs, domain=self.domain, deltas=())
            _pv = getattr(self, "_point_values", None)
            if _pv is not None:
                object.__setattr__(base, "_point_values", _pv)
            res = _np.array(base(xq))
            xn = _np.asarray(xq, dtype=float)
            for row in self.deltas:
                loc, mag, order = _delta_row(row)
                if order != 0:
                    continue
                mg = _np.asarray(mag)
                if mg.ndim > 0:
                    mg = mg.reshape(-1)[0]
                if mg == 0:
                    continue
                mask = xn == float(loc)
                if _np.any(mask):
                    val = _np.inf * _np.sign(float(_np.real(mg)))
                    if xn.ndim == 0:
                        res[...] = val
                    else:
                        res[mask] = val
            return jnp.asarray(res)
        if isinstance(x, str):
            # MATLAB feval(f, 'left'/'start'/'-') and
            # feval(f, 'right'/'end'/'+'): the value at an endpoint.
            key = x.lower()
            if key in ("left", "start", "-"):
                return self(jnp.asarray(float(self.domain.a)))
            if key in ("right", "end", "+"):
                return self(jnp.asarray(float(self.domain.b)))
            raise ValueError(
                f"Chebfun evaluation point {x!r}: expected "
                f"'left'/'start'/'-' or 'right'/'end'/'+'.")
        if side is not None:
            _record_side_eval(x)
            return self._feval_side(x, side)
        # MATLAB feval returns the stored pointValues AT the breakpoints
        # (definePoint / f(s) = v); away from them the pieces are used.
        _pv = getattr(self, "_point_values", None)
        if (_pv is not None and not isinstance(x, jax.core.Tracer)
                and not isinstance(_pv, jax.core.Tracer)):
            import numpy as _np
            base = Chebfun(funs=self.funs, domain=self.domain,
                           deltas=self.deltas)
            res = _np.array(base(x))
            xn = _np.asarray(x)
            pvn = _np.asarray(_pv)
            if pvn.ndim == 1 and res.ndim == xn.ndim + 1:
                pvn = pvn[:, None]
            if _np.iscomplexobj(pvn) and not _np.iscomplexobj(res):
                res = res.astype(_np.complex128)
            for k, t in enumerate(self.domain.breakpoints):
                mask = xn == float(t)
                if _np.any(mask):
                    if xn.ndim == 0:
                        res[...] = pvn[k]
                    else:
                        res[mask] = pvn[k]
            return self._orient_values(jnp.asarray(res))
        # Concrete input, concrete coefficients: stay in numpy end to end
        # (dtype promotion, affine maps, piece binning).  The jitted path
        # below pays per-call dispatch (~0.5 ms) that dominates ODE-marcher
        # RHS evaluation of chebfun coefficients one scalar at a time.
        if not isinstance(x, jax.core.Tracer) and \
                not isinstance(_pv, jax.core.Tracer) and \
                not any(isinstance(p.tech.coeffs, jax.core.Tracer)
                        for p in self.funs):
            import numpy as _np

            xn = _np.asarray(x)
            xn = xn.astype(_np.complex128 if _np.iscomplexobj(xn)
                           else _np.float64)
            scalar_input = xn.ndim == 0

            if len(self.funs) == 1:
                p = self.funs[0]
                a_, b_ = p.interval
                if type(p) is _Piece and _np.isfinite(a_) \
                        and _np.isfinite(b_):
                    # A scalar point stays 0-d so the tech's scalar-
                    # Clenshaw branch (Python-float recurrence) can
                    # take it.  Only the plain bounded piece uses this
                    # inline affine map — an Unbndfun piece owns its
                    # (nonlinear, infinite-interval) map.
                    result = jnp.asarray(
                        p.tech(_linear_inverse_map(xn, a_, b_)))
                else:
                    result = p(jnp.asarray(_np.atleast_1d(xn)))
                    if scalar_input:
                        result = result[0]
                return self._orient_values(result)

            xn = _np.atleast_1d(xn)

            # Multi-piece: bin the points to their pieces with
            # searchsorted and evaluate each piece ONLY at its own
            # points -- the masked full sweep below costs
            # O(pieces x points x degree) and dominated many-piece
            # chebfun evaluation.  Breakpoint points route to the RIGHT
            # piece, the same winner as the masked sweep's last-write
            # order.
            xrn = _np.real(xn)
            breaks = _np.array([p.interval[1] for p in self.funs[:-1]],
                               dtype=_np.float64)
            idx = _np.searchsorted(breaks, xrn, side="right")
            cols_np = self.funs[0].tech.coeffs.shape[1:]
            out_dtype = _np.complex128 if (
                _np.iscomplexobj(xn)
                or any(_np.iscomplexobj(_np.asarray(p.tech.coeffs))
                       for p in self.funs)) else _np.float64
            out_np = _np.empty(xn.shape + cols_np, dtype=out_dtype)
            for i in _np.unique(idx):
                sel = idx == i
                p = self.funs[int(i)]
                a_, b_ = p.interval
                if type(p) is _Piece and _np.isfinite(a_) \
                        and _np.isfinite(b_):
                    tn = _linear_inverse_map(xn[sel], a_, b_)
                    out_np[sel] = _np.asarray(p.tech(tn))
                else:
                    # Unbndfun (or other) pieces own their map.
                    out_np[sel] = _np.asarray(p(jnp.asarray(xn[sel])))
            result = jnp.asarray(out_np)
            result = self._apply_point_values(result, jnp.asarray(xn))
            if scalar_input:
                result = result[0]
            return self._orient_values(result)

        # Traced input (or traced coefficients): promote via jnp.
        # Preserve a complex argument (MATLAB feval evaluates a real
        # Chebfun at complex points, routing pieces by real(x));
        # everything else is promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
        scalar_input = x.ndim == 0
        x = jnp.atleast_1d(x)

        if len(self.funs) == 1:
            # Single piece — fully JIT-able.
            result = self.funs[0](x)
            result = self._apply_point_values(result, x)
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

        out = self._apply_point_values(out, x)
        if scalar_input:
            out = out[0]
        return self._orient_values(out)

    def _apply_point_values(self, values, x):
        """Apply stored breakpoint values in a traced evaluation.

        Provenance
        ----------
        MATLAB source : @chebfun/feval.m (columnFeval pointValues override)
        Chebfun commit: 7574c77
        """
        if self._point_values is None and len(self.funs) <= 1:
            return values
        point_values = self._breakpoint_values()
        for index, breakpoint in enumerate(self.domain.breakpoints):
            mask = x == breakpoint
            mask = mask.reshape(mask.shape + (1,) * (values.ndim - x.ndim))
            values = jnp.where(mask, point_values[index], values)
        return values

    def _breakpoint_values(self):
        """JAX breakpoint values from stored data or neighboring FUN limits.

        Provenance
        ----------
        MATLAB source : @chebfun/getValuesAtBreakpoints.m
        Chebfun commit: 7574c77
        """
        if self._point_values is not None:
            return jnp.asarray(self._point_values)
        if not self.funs:
            return jnp.empty((0,), dtype=jnp.float64)
        breaks = self.domain.breakpoints
        values = [self.funs[0](jnp.asarray(breaks[0]))]
        for index, breakpoint in enumerate(breaks[1:-1], 1):
            values.append((self.funs[index - 1](jnp.asarray(breakpoint))
                           + self.funs[index](jnp.asarray(breakpoint))) / 2)
        values.append(self.funs[-1](jnp.asarray(breaks[-1])))
        return jnp.stack(values)

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
        """MATLAB-faithful multi-line display (@chebfun/disp.m).

        Reproduces MATLAB's format exactly: the ``chebfun column (N smooth
        pieces)`` header, the per-piece ``[a, b]  length  lval  rval`` rows
        (``%8.2g`` fields, ``complex values`` for complex pieces, an
        ``endpoint exponents`` column when any piece is singular), and the
        ``vertical scale = ... Total length = ...`` footer.

        Provenance
        ----------
        MATLAB source : @chebfun/disp.m, @chebfun/dispData.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        Examples
        --------
        >>> f = chebfun(jnp.sin)
        >>> repr(f)
        '   chebfun column (1 smooth piece)\\n       interval ...'
        """
        n_pieces = len(self.funs)
        piece_word = "piece" if n_pieces == 1 else "pieces"
        orientation = "row" if self.is_transposed else "column"
        s = f"   chebfun {orientation} ({n_pieces} smooth {piece_word})"

        # dispData: an 'endpoint exponents' column when any piece has one.
        all_exps = [tuple(float(e) for e in
                          getattr(p.tech, "exponents", (0.0, 0.0)))
                    for p in self.funs]
        has_exps = any(e != (0.0, 0.0) for e in all_exps)
        extra_item = "  endpoint exponents" if has_exps else " "
        # MATLAB's disp appends a 'trig' marker for periodic reps.
        from chebfunjax.tech.trigtech import Trigtech as _Trig
        if any(isinstance(p.tech, _Trig) for p in self.funs):
            extra_item = (extra_item.rstrip() + " trig"
                          if extra_item.strip() else "trig")
        s += ("\n       interval       length     endpoint values"
              f" {extra_item}\n")

        total = 0
        vs_sup = 0.0
        for piece, exps in zip(self.funs, all_exps):
            a, b = piece.interval
            length = piece.n
            total += length
            extra = (f"        [{exps[0]:2.2g}      {exps[1]:2.2g}]  "
                     if has_exps else "")
            cf = piece.coeffs
            # A real trigfun's FOURIER coefficients are complex; MATLAB's
            # disp checks the FUN's realness (isreal(f.funs{j})), so honour
            # the tech's is_real flag before inspecting coefficients.
            piece_real = bool(getattr(piece.tech, "is_real", False))
            if (not piece_real) and bool(jnp.iscomplexobj(cf)) and float(
                    jnp.max(jnp.abs(jnp.imag(cf)))) > 0:
                s += ("[%8.2g,%8.2g]   %6i     complex values %s\n"
                      % (a, b, length, extra))
                vs_sup = max(vs_sup, piece.vscale)
            else:
                lval, rval = piece.endpoint_values
                # Prevent -0/+0 (MATLAB's endvals tweak).
                if not (math.isnan(lval) or math.isnan(rval)):
                    lval = 0.0 if lval == 0 else lval
                    rval = 0.0 if rval == 0 else rval
                s += ("[%8.2g,%8.2g]   %6i  %8.2g %8.2g %s\n"
                      % (a, b, length, lval, rval, extra))
                for v in (piece.vscale, abs(lval), abs(rval)):
                    if not math.isnan(v):
                        vs_sup = max(vs_sup, v)
        s += "vertical scale = %3.2g " % vs_sup
        if n_pieces > 1:
            s += "   Total length = %i" % total
        # MATLAB capitalises Inf/NaN; Python's %g prints them lowercase.
        return s.replace("inf", "Inf").replace("nan", "NaN")

    def __str__(self) -> str:
        """One-line summary."""
        a, b = self.domain.a, self.domain.b
        return f"<Chebfun [{a}, {b}], length {len(self)}>"

    @staticmethod
    def odesol(sol, domain, options=None, *, return_time=False):
        """Convert dense ODE output; see :func:`chebfunjax.odesol`.

        Provenance
        ----------
        MATLAB source : @chebfun/odesol.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.ode_solution import odesol
        return odesol(sol, domain, options, return_time=return_time)


    @staticmethod
    def constructODEsol(solver, odefun, tspan, y0, *solver_args, return_time=False):
        """Run source ODE construction around a supplied dense-output solver.

        Provenance
        ----------
        MATLAB source : @chebfun/constructODEsol.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.construct_ode_solution import constructODEsol
        return constructODEsol(solver, odefun, tspan, y0, *solver_args,
                               return_time=return_time)

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
        new_funs = []
        for sub in new_dom.intervals:
            sa, sb = float(sub.a), float(sub.b)
            # owner = the piece with maximal overlap: midpoint-with-
            # tolerance selection could pick a piece the cell only
            # touches at a rounded endpoint, whose restriction then
            # degenerates to a zero-width reference interval.
            piece = max(
                self.funs,
                key=lambda p: (min(float(p.interval[1]), sb)
                               - max(float(p.interval[0]), sa)))
            new_funs.append(piece.restrict(sa, sb))
        return Chebfun(funs=new_funs, domain=new_dom)

    @staticmethod
    def _overlap(f: Chebfun, g: Chebfun) -> "tuple[Chebfun, Chebfun]":
        """Source domain check, tweak, exact union, and restriction.

        Provenance
        ----------
        MATLAB source : @chebfun/overlap.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.restriction import overlap
        return overlap(f, g)

    @staticmethod
    def _binary_op(f: Chebfun, g: Chebfun, op) -> Chebfun:
        """Apply a piecewise binary op between two same-domain Chebfuns.

        ``op`` must be a method name (str) on ``Chebtech2`` accepting one
        Chebtech2 argument, or a callable ``op(tech_a, tech_b) -> Chebtech2``.
        """
        f0, g0 = f, g
        f, g = Chebfun._overlap(f, g)
        new_funs = [
            pf.with_tech(op(*_cast_tech_pair(pf.tech, pg.tech)))
            for pf, pg in zip(f.funs, g.funs)
        ]
        out = Chebfun(funs=new_funs, domain=f.domain)
        # MATLAB: pointValues of a binary operation are the operation
        # applied to the operands' pointValues (feval at a breakpoint
        # returns the stored value, e.g. sign(x)(0) = 0).
        if (getattr(f0, "_point_values", None) is not None
                or getattr(g0, "_point_values", None) is not None):
            try:
                bps = jnp.asarray([float(v) for v in out.domain.breakpoints])
                pv = op(f0(bps), g0(bps))
                if getattr(pv, "shape", None) is not None \
                        and pv.shape[0] == bps.shape[0]:
                    object.__setattr__(out, "_point_values", jnp.asarray(pv))
            except Exception:
                pass
        return out

    @staticmethod
    def _merge_deltas(d1, d2, s1: float = 1.0, s2: float = 1.0) -> tuple:
        """Combine two ``deltas`` tuples with scalar weights.

        Coincident (location, order) rows accumulate; zero magnitudes
        drop.  Rows are ``(loc, mag)`` (a plain Dirac) or
        ``(loc, mag, order)`` (the order-th distributional derivative,
        MATLAB deltaMag row ``order + 1``).  Mirrors the delta
        bookkeeping of MATLAB @deltafun/plus.m.
        """
        acc: dict = {}
        for scale, ds in ((s1, d1), (s2, d2)):
            for row in ds:
                loc, mag, order = _delta_row(row)
                key = (loc, order)
                acc[key] = acc.get(key, 0.0 * mag) + scale * mag
        out = []
        # MATLAB @deltafun/simplify: magnitudes below deltaTol (1e-9) are
        # rounding residue of cancelled deltas and are dropped.
        from chebfunjax.chebpref import ChebfunPref
        _dtol = float(ChebfunPref().deltaPrefs.deltaTol)
        for (loc, order), mag in acc.items():
            if abs(mag) > _dtol:
                out.append((loc, mag) if order == 0 else (loc, mag, order))
        return tuple(sorted(out, key=lambda r: (r[0], _delta_row(r)[2])))

    def _attach_deltas(self, result: "Chebfun", deltas: tuple) -> "Chebfun":
        """Return ``result`` carrying ``deltas`` and its existing metadata.

        Provenance
        ----------
        MATLAB source : @deltafun/times.m, @chebfun/times.m
        Chebfun commit: 7574c77
        """
        if not deltas:
            return result
        out = Chebfun(funs=result.funs, domain=result.domain,
                      deltas=deltas)
        pv = getattr(result, "_point_values", None)
        if pv is not None:
            object.__setattr__(out, "_point_values", pv)
        return Chebfun._as_transposed(out, result.is_transposed)

    @staticmethod
    def _finish_plus(out: Chebfun, transposed: bool) -> Chebfun:
        """Apply source breakpoint threshold and retain operand orientation.

        Provenance
        ----------
        MATLAB source : @chebfun/plus.m, @chebfun/thresholdBreakpointValues.m
        Chebfun commit: 7574c77
        """
        values = jnp.asarray(out.point_values)
        scale = jnp.max(jnp.stack([jnp.max(jnp.abs(p.values)) for p in out.funs]))
        values = jnp.where(jnp.abs(values) < 10*scale*_EPS, 0., values)
        object.__setattr__(out, "_point_values", values)
        return Chebfun._as_transposed(out, transposed)

    def __add__(self, other) -> Chebfun:
        """Add Chebfuns or numeric constants with source column expansion.

        Provenance
        ----------
        MATLAB source : @chebfun/plus.m, @chebfun/dimCheck.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
        if isinstance(other, Chebfun):
            if self.is_transposed != other.is_transposed:
                raise ValueError("CHEBFUN:CHEBFUN:plus:matdim: "
                                 "Matrix dimensions must agree. (One input is transposed).")
            if (self.n_columns != other.n_columns
                    and self.n_columns != 1 and other.n_columns != 1):
                raise ValueError("CHEBFUN:CHEBFUN:dimCheck:dim: "
                                 "Matrix dimensions must agree.")
            out = Chebfun._binary_op(self, other, lambda a, b: a+b)
            if (getattr(self, "_point_values", None) is not None
                    or getattr(other, "_point_values", None) is not None):
                points = jnp.asarray(out.domain.breakpoints)
                left = self.T if self.is_transposed else self
                right = other.T if other.is_transposed else other
                fv, gv = left(points), right(points)
                if fv.ndim == 1 and gv.ndim == 2:
                    fv = fv[:, None]
                elif gv.ndim == 1 and fv.ndim == 2:
                    gv = gv[:, None]
                object.__setattr__(out, "_point_values", fv+gv)
            out = self._attach_deltas(out, Chebfun._merge_deltas(
                getattr(self, "deltas", ()), getattr(other, "deltas", ())))
            return self._finish_plus(out, self.is_transposed)
        if not isinstance(other, (int, float, complex, list, tuple, jnp.ndarray,
                                  jax.Array)):
            if not (hasattr(other, "dtype")
                    and getattr(other, "ndim", None) is not None):
                return NotImplemented
        value = jnp.asarray(other)
        # Keep the established signed Python integer adapter. Explicit unsigned
        # MATLAB operands are not doubles; the bounded leaf rejects them.
        if (jnp.issubdtype(value.dtype, jnp.unsignedinteger)
                and all(isinstance(p.tech, Chebtech2) for p in self.funs)):
            raise TypeError("CHEBFUN:CHEBTECH:plus:typeMismatch: "
                            "Incompatible operation between objects.\n"
                            "Make sure functions are of the same type.")
        if value.ndim == 2:
            value = value.T if self.is_transposed else value
            if value.shape[0] != 1:
                raise ValueError("CHEBFUN:CHEBFUN:plus:dims: Matrix dimensions must agree.")
            value = value[0]
        if value.ndim > 1 or (value.ndim == 1 and value.size not in (1, self.n_columns)
                             and self.n_columns != 1):
            raise ValueError("CHEBFUN:CHEBFUN:plus:dims: Matrix dimensions must agree.")
        if value.size == 1:
            value = value.reshape(())
        new_funs = [piece._apply_unary(piece.tech+value) for piece in self.funs]
        out = Chebfun(funs=new_funs, domain=self.domain)

        def add_points(points):
            if points.ndim == 1 and value.ndim == 1 and value.size > 1:
                points = points[:, None]
            return points+value

        out = self._propagate_point_values(out, add_points)
        out = self._attach_deltas(out, getattr(self, "deltas", ()))
        return self._finish_plus(out, self.is_transposed)

    # numpy must not try to broadcast a Chebfun when it appears on the
    # right of a numpy scalar.  Without this, `np.float64(2) - f` (and
    # so any ported expression like `2*sin(psi) - b*U(1)`, since np.sin
    # returns np.float64) raises "setting an array element with a
    # sequence" instead of deferring to __rsub__.  Setting
    # __array_ufunc__ to None makes numpy return NotImplemented, so
    # Python falls back to this class's reflected operators.
    __array_ufunc__ = None

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
            out = Chebfun._binary_op(self, other, lambda a, b: a - b)
            return self._attach_deltas(out, Chebfun._merge_deltas(
                getattr(self, "deltas", ()), getattr(other, "deltas", ()),
                1.0, -1.0))
        if not isinstance(other, (int, float, complex, jnp.ndarray,
                                  jax.Array)):
            # Defer to the other type's reflected operator (see __add__).
            if not (hasattr(other, "dtype")
                    and getattr(other, "ndim", None) is not None):
                return NotImplemented
        new_funs = [
            piece._apply_unary(piece.tech - other)
            for piece in self.funs
        ]
        out = Chebfun(funs=new_funs, domain=self.domain)
        return self._attach_deltas(out, getattr(self, "deltas", ()))

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
        out = Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
        return self._attach_deltas(out, Chebfun._merge_deltas(
            (), getattr(self, "deltas", ()), 1.0, -1.0))

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
                out = Chebfun._as_transposed(
                    Chebfun._binary_op(self, other, lambda a, b: a * b),
                    f_trans)
                # Dirac deltas multiply by the cofactor via Leibniz
                # (@deltafun/times.m funTimesDelta):
                #   g(x) d^(k)(x-u) =
                #     sum_j (-1)^(k-j) C(k,j) g^(k-j)(u) d^(j)(x-u).
                ds = ()
                from math import comb as _comb
                for me, oth in ((self, other), (other, self)):
                    rows = [_delta_row(r)
                            for r in getattr(me, "deltas", ())]
                    if not rows:
                        continue
                    kmax = max(o for _l, _m, o in rows)
                    derivs = [oth]
                    for _ in range(kmax):
                        derivs.append(derivs[-1].diff())
                    for loc, mag, order in rows:
                        for j in range(order + 1):
                            val = float(jnp.real(jnp.asarray(
                                derivs[order - j](jnp.float64(loc)))))
                            coeff = ((-1.0) ** (order - j)
                                     * _comb(order, j) * mag * val)
                            ds = Chebfun._merge_deltas(
                                ds, ((loc, coeff) if j == 0
                                     else (loc, coeff, j),))
                return self._attach_deltas(out, ds)
            if f_trans and not g_trans:
                # Row * column -> inner product scalar.  inner() conjugates
                # its first argument, so conj(f).inner(g) == int f*g.
                return self.conj().inner(other)
            # Column * row -> rank-1 outer product (a Chebfun2).
            raise NotImplementedError(
                "column * row (rank-1 outer product / Chebfun2) is not "
                "supported in chebfunjax."
            )
        # If other is not a scalar/array, defer to other's __rmul__.
        # np.float64 subclasses float and so passed already, but
        # np.int64 and a 0-d np.ndarray did not, and with
        # __array_ufunc__ = None numpy now hands those straight here --
        # rejecting them turns `np.int64(2) * f` into a TypeError.
        # Duck-typed so this module still imports no numpy.
        if not isinstance(other, (int, float, complex, jnp.ndarray,
                                  jax.Array)):
            if not (hasattr(other, "dtype")
                    and getattr(other, "ndim", None) == 0):
                return NotImplemented
        # The Python Chebfun API accepts scalar integers, including typed
        # zero-dimensional scalars. Adapt them as MATLAB double literals at
        # this boundary; explicit non-scalar arrays and Chebtech dtype/class
        # diagnostics retain their own source dispatch.
        if (getattr(other, "ndim", None) == 0 and hasattr(other, "dtype")
                and jnp.issubdtype(other.dtype, jnp.integer)):
            other = jnp.asarray(other, dtype=jnp.float64)
        new_funs = [
            piece._apply_unary(piece.tech * other)
            for piece in self.funs
        ]
        out = Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
        out = self._propagate_point_values(out, lambda v, _o=other: v * _o)
        # Scalar scaling also scales any Dirac deltas (@deltafun/mtimes.m).
        ds = getattr(self, "deltas", ())
        if ds:
            try:
                sc = complex(other)
                sc = sc.real if sc.imag == 0 else sc
                out = self._attach_deltas(out, Chebfun._merge_deltas(
                    (), ds, 1.0, sc))
            except TypeError:
                pass
        return out

    def __matmul__(self, other):
        """MATLAB matrix multiplication, exposed as Python ``@``.

        Provenance
        ----------
        MATLAB source : @chebfun/mtimes.m
        Chebfun commit: 7574c77
        """
        from .mtimes import mtimes
        return mtimes(self, other)

    def __rmatmul__(self, other):
        """MATLAB multiplication with a numeric left operand.

        Provenance
        ----------
        MATLAB source : @chebfun/mtimes.m
        Chebfun commit: 7574c77
        """
        from .mtimes import mtimes
        return mtimes(other, self)

    def __rmul__(self, other) -> Chebfun:
        return self.__mul__(other)

    def _check_zero_denominator_funs(self) -> bool:
        """Reject exactly zero polynomial FUNs before division root finding.

        Return whether every piece used the exact ordinary polynomial check.
        Other representations and distributions retain their existing adapters.
        This construction-stage check is eager, like adaptive division itself.

        Provenance
        ----------
        MATLAB source : @chebfun/rdivide.m, @classicfun/iszero.m,
            @chebtech/iszero.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.chebtech import Chebtech1

        if self.deltas:
            return False
        complete = True
        for piece in self.funs:
            if not isinstance(piece, _Piece) or not isinstance(
                    piece.tech, (Chebtech1, Chebtech2)):
                complete = False
                continue
            # Equality is exact: tiny coefficients and NaNs are not zero.
            # MATLAB's IF requires every column's ISZERO result to be true.
            if bool(jnp.all(piece.tech.coeffs == 0)):
                raise ValueError(
                    "CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun: "
                    "Division by CHEBFUN with identically zero FUN.")
        return complete

    def _check_rdivide_compatibility(self, other: Chebfun) -> None:
        """Check division domains, then orientation, after denominator roots.

        Provenance
        ----------
        MATLAB source : @chebfun/rdivide.m, @chebfun/domainCheck.m,
            @chebfun/hscale.m
        Chebfun commit: 7574c77
        """
        hs = max(_hscale(self), _hscale(other))
        ends = jnp.asarray([self.domain.a, self.domain.b])
        other_ends = jnp.asarray([other.domain.a, other.domain.b])
        err = ends - other_ends
        if not bool(jnp.all((jnp.abs(err) < 1e-15 * hs) | jnp.isnan(err))):
            raise ValueError(
                "CHEBFUN:CHEBFUN:rdivide:columnRdivide:domain: "
                "Inconsistent domains.")
        if self.is_transposed != other.is_transposed:
            raise ValueError(
                "CHEBFUN:CHEBFUN:rdivide:columnRdivide:dim: "
                "Matrix dimension do not agree (transposed)")

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
            other._check_zero_denominator_funs()
            poles = _real_simple_roots(other)
            self._check_rdivide_compatibility(other)
            f, g = self, other
            if f.domain.breakpoints != g.domain.breakpoints:
                f, g, _, _ = tweak_domain(f, g)
                if poles.size and g.domain.breakpoints != other.domain.breakpoints:
                    # Map changes move physical roots with their FUNs.
                    poles = _real_simple_roots(g)
            # Stored point values use breakpoint rows, irrespective of the
            # output orientation. Evaluate the scalar/array columns as columns.
            f = f.T if f.is_transposed else f
            g = g.T if g.is_transposed else g
            if poles.size:
                out = _divide_with_poles(f, g, poles)
            else:
                out = Chebfun._binary_op(f, g, lambda a, b: a / b)
            return Chebfun._as_transposed(out, self.is_transposed)
        if not isinstance(other, (int, float, complex, jnp.ndarray,
                                  jax.Array)):
            # Defer to the other type's reflected operator (see __add__).
            if not (hasattr(other, "dtype")
                    and getattr(other, "ndim", None) is not None):
                return NotImplemented
        if getattr(other, "ndim", 0) == 0:
            # @chebfun/rdivide columnRdivide uses TIMES with the reciprocal,
            # preserving orientation, stored point values and delta terms.
            # A traced scalar cannot raise a value-dependent Python error.
            if not isinstance(other, jax.core.Tracer) and bool(other == 0):
                raise ValueError(
                    "CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero: "
                    "Division by zero.")
            return self * (1 / other)
        new_funs = [
            piece._apply_unary(piece.tech / other)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __rtruediv__(self, other) -> Chebfun:
        """scalar / Chebfun (MATLAB rdivide: denominator roots become
        poles represented by SingFun pieces with negative exponents).

        Provenance
        ----------
        MATLAB source : @chebfun/rdivide.m, @deltafun/rdivide.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or _is_empty_operand(other):
            return Chebfun.empty()
        if (getattr(other, "ndim", 0) == 0
                and not isinstance(other, jax.core.Tracer) and bool(other == 0)):
            # The source zero-numerator shortcut precedes denominator checks.
            return self * 0
        if any(row[1] != 0 for row in self.deltas):
            raise ValueError(
                "CHEBFUN:DELTAFUN:rdivide:rdivide: "
                "Division by delta functions is not defined.")
        if not self._check_zero_denominator_funs():
            try:
                _is_zero = float(self.norm(jnp.inf)) == 0.0
            except Exception:
                _is_zero = False
            if _is_zero:
                raise ValueError(
                    "CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun: "
                    "Division by CHEBFUN with identically zero FUN.")
        poles = _real_simple_roots(self)
        if poles.size:
            return _divide_with_poles(other, self, poles)
        new_funs = [
            piece._apply_unary(other / piece.tech)
            for piece in self.funs
        ]
        out = Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
        return self._propagate_point_values(out, lambda v: other / v)

    def __pow__(self, exponent) -> Chebfun | Quasimatrix:
        """Raise scalar or matched columns to numeric powers.

        A numeric exponent vector expands a scalar-valued base. Smooth
        columns combine into an array-valued Chebfun; columns with singular
        representations remain a Quasimatrix. Matched multi-column powers
        return a Quasimatrix, retaining each column's piece partition.

        Bounded positive integer powers above two compose adaptively, as in
        MATLAB columnPower. That construction is eager; evaluation of the
        constructed bounded result supports JAX tracing.

        Provenance
        ----------
        MATLAB source : @chebfun/power.m
        Chebfun commit: 7574c77
        """
        # MATLAB class dispatch routes CHEBFUN.^ADCHEBFUN to AD power.
        from chebfunjax.autodiff.adchebfun import ADChebfun
        if isinstance(exponent, ADChebfun):
            return exponent.__rpow__(self)
        if self.isempty():
            return type(self).empty()
        exp_array = None
        if not isinstance(exponent, Chebfun):
            try:
                exp_array = jnp.asarray(exponent)
            except (TypeError, ValueError):
                pass
        if exp_array is not None and exp_array.ndim > 0:
            if exp_array.dtype.kind not in "biufc":
                return NotImplemented
            # Source b(k) uses MATLAB column-major linear indexing, including
            # shaped row/column vectors and numeric exponent matrices.
            exp_array = jnp.ravel(exp_array, order="F")
            if exp_array.size == 0:
                return type(self).empty()
            if exp_array.size == 1:
                return self ** exp_array[0]
            elif self.n_columns == 1:
                # MATLAB @chebfun/power.m then quasi2cheb: build each scalar
                # power separately, and align/concatenate through assignColumns.
                powered = [None] * exp_array.size
                for k in range(exp_array.size - 1, -1, -1):
                    powered[k] = self ** exp_array[k]
                from chebfunjax.tech.chebtech import Chebtech1
                from chebfunjax.tech.trigtech import Trigtech

                # quasi2cheb calls cat using the first result's orientation.
                # Native zero powers are fresh columns, even for a row base;
                # mixing those with nonzero row powers fails concatenation.
                if any(column.is_transposed != powered[0].is_transposed
                       for column in powered[1:]):
                    operation = "vertcat" if powered[0].is_transposed else "horzcat"
                    raise ValueError(
                        f"CHEBFUN:CHEBFUN:{operation}:transpose: "
                        "Dimensions of matrices being concatenated are not consistent.")
                # @chebfun/cat cannot collate singular funs into one array.
                if not all(isinstance(p.tech, (Chebtech1, Chebtech2, Trigtech))
                           for column in powered for p in column.funs):
                    from chebfunjax.chebfun1d.linalg import Quasimatrix

                    return Quasimatrix(powered, Domain((self.domain.a, self.domain.b)))
                out = powered[0]
                for k, column in enumerate(powered[1:], start=1):
                    out = out.assign_columns(k, column)
                return out
            else:
                if exp_array.size != self.n_columns:
                    raise ValueError(
                        "CHEBFUN:CHEBFUN:power:dim: "
                        "Chebfun quasimatrix dimensions must agree.")
                from chebfunjax.chebfun1d.linalg import Quasimatrix

                columns = self.mat2cell()
                powered = [None] * self.n_columns
                for k in range(self.n_columns - 1, -1, -1):
                    powered[k] = columns[k] ** exp_array[k]
                return Quasimatrix(
                    powered, Domain((self.domain.a, self.domain.b)),
                )
        if isinstance(exponent, Chebfun):
            if self.isempty() or exponent.isempty():
                return type(self).empty()
            result = Chebfun._binary_op(self, exponent, lambda a, b: a ** b)
            from chebfunjax.tech.chebtech import Chebtech1

            if all(isinstance(piece.tech, (Chebtech1, Chebtech2))
                   for operand in (self, exponent) for piece in operand.funs):
                # Source binary compose maps both stored pointValues after
                # overlap, with MATLAB's singleton-column broadcasting.
                # Evaluate as columns: row orientation is output metadata.
                base_column = self.T if self.is_transposed else self
                exp_column = exponent.T if exponent.is_transposed else exponent
                breaks = jnp.asarray(result.domain.breakpoints)
                base_values = base_column(breaks)
                exp_values = exp_column(breaks)
                if base_values.ndim < exp_values.ndim:
                    base_values = base_values[..., None]
                elif exp_values.ndim < base_values.ndim:
                    exp_values = exp_values[..., None]
                result = result.set_point_values(jnp.power(base_values, exp_values))
                result = Chebfun._as_transposed(result, self.is_transposed)
            return result
        # A non-integer scalar power of a function with roots produces
        # branch-point singularities: route through the singularity-aware
        # path (MATLAB @chebfun/power.m columnPower general case) so e.g.
        # (1+x)**0.3 yields a compact Singfun with exps [0.3, 0] instead
        # of an unhappy 65537-point smooth representation.
        # Source equality branches also accept complex numeric scalars
        # with zero imaginary part. This concrete scalar conversion does
        # not change the later general-power dispatch.
        try:
            scalar_value = complex(exponent)
        except (TypeError, ValueError):
            scalar_value = None
        # Source columnPower dispatches by scalar value, including numeric
        # array scalars. Zero constructs a constant on the support domain;
        # one preserves the object and its metadata; two calls TIMES.
        if scalar_value == 0:
            values = (jnp.asarray(1.) if self.n_columns == 1
                      else jnp.ones((1, self.n_columns)))
            # Native constructor orientation is column; power does not
            # restore a row base's orientation in this branch.
            return chebfun(values, domain=(self.domain.a, self.domain.b))
        if scalar_value == 1:
            return self
        if scalar_value == 2:
            return self * self
        try:
            exp_f = float(exponent)
        except (TypeError, ValueError):
            exp_f = None
        if exp_f is not None and exp_f != int(exp_f):
            return self._root_power(exp_f, lambda v, _b=exp_f: v ** _b)
        if exp_f is not None and exp_f >= 3 and exp_f == int(exp_f):
            from chebfunjax.tech.chebtech import Chebtech1

            if all(isinstance(piece.tech, (Chebtech1, Chebtech2))
                   for piece in self.funs):
                # Source columnPower retains TIMES for the square, but
                # composes smooth positive integer powers above two.
                integer_power = int(exp_f)
                op = lambda values: values ** integer_power  # noqa: E731
                result = self._apply_fun(op).set_point_values(op(self._breakpoint_values()))
                return Chebfun._as_transposed(result, self.is_transposed)
        new_funs = [
            piece._apply_unary(piece.tech ** exponent)
            for piece in self.funs
        ]
        return Chebfun(funs=new_funs, domain=self.domain)

    def __rpow__(self, base) -> Chebfun:
        """Compute a concrete numeric scalar base to this bounded Chebfun power.

        MATLAB source @chebfun/power.m handles ``constant .^ CHEBFUN`` by
        composing over the exponent Chebfun. Complex promotion for a negative
        real scalar base gives MATLAB's principal complex power for real-valued
        exponents; without it, real JAX power yields NaNs at fractional values.

        Provenance
        ----------
        MATLAB source : @chebfun/power.m (columnPower), @chebtech/power.m
        Chebfun commit: 7574c77

        Adaptive reconstruction is not JIT-safe. This method handles
        finite Chebtech1/2 pieces only; arrays/quasimatrices and Singfun,
        Unbndfun, and Trigtech composition are not claimed.
        """
        from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

        if self.isempty():
            return type(self).empty()

        try:
            base_array = jnp.asarray(base)
        except (TypeError, ValueError):
            return NotImplemented
        if base_array.ndim != 0 or base_array.dtype.kind not in "biufc":
            return NotImplemented

        if not all(isinstance(piece.tech, (Chebtech1, Chebtech2))
                   for piece in self.funs):
            return NotImplemented

        # MATLAB power uses the complex principal branch for negative real bases.
        # This concrete host-side branch is acceptable because composition itself
        # adaptively builds a Chebfun and is not JIT-safe.
        if base_array.dtype.kind != "c" and bool(base_array < 0):
            base_array = base_array.astype(jnp.complex128)

        op = lambda exponent: jnp.power(base_array, exponent)  # noqa: E731
        # Source compose maps the stored breakpoint values (including means
        # at jumps), then keeps the exponent Chebfun's row/column orientation.
        result = self._apply_fun(op).set_point_values(op(self._breakpoint_values()))
        return Chebfun._as_transposed(result, self.is_transposed)

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
        if not self.funs:
            return self
        # @chebfun/compose.m transforms stored pointValues separately from
        # smooth one-sided limits, and forces extrapolation for numInts > 1.
        point_values = op(self.point_values)
        extrapolate = len(self.funs) > 1
        new_funs = []
        for piece in self.funs:
            if isinstance(piece, _Piece) and isinstance(piece.tech, Chebtech2):
                new_funs.append(piece._apply_fun(op, extrapolate=extrapolate))
            else:
                # Other representation adapters retain their existing path;
                # their endpoint preference dispatch remains a separate gap.
                new_funs.append(piece._apply_fun(op))
        result = Chebfun(funs=new_funs, domain=self.domain)
        result = result.set_point_values(point_values)
        return Chebfun._as_transposed(result, self.is_transposed)

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

    def expm1(self) -> Chebfun:
        """Compute ``exp(f) - 1`` accurately for small ``f``.

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/expm1.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.exp, Chebfun.log1p
        """
        return self._apply_fun(jnp.expm1)

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

    def log10(self) -> Chebfun:
        """Base-10 logarithm of the Chebfun.

        Breakpoints are introduced at the interior roots of ``f`` before
        composing, since ``log10`` is singular there.

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/log10.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log, Chebfun.log2
        """
        return self.addBreaksAtRoots()._apply_fun(jnp.log10)

    def log2(self) -> Chebfun:
        """Base-2 logarithm of the Chebfun.

        Breakpoints are introduced at the interior roots of ``f`` before
        composing, since ``log2`` is singular there.

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/log2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log, Chebfun.log10
        """
        return self.addBreaksAtRoots()._apply_fun(jnp.log2)

    def log1p(self) -> Chebfun:
        """Compute ``log(1 + f)`` accurately for small ``f``.

        Breakpoints are introduced at the interior roots of ``f + 1``
        before composing (MATLAB ``addBreaks(f, getRootsForBreaks(f+1))``).

        Returns
        -------
        Chebfun

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/log1p.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log, Chebfun.expm1
        """
        import numpy as _np
        r = _np.asarray((self + 1.0).roots(nojump=True), dtype=float).ravel()
        a, b = float(self.domain.a), float(self.domain.b)
        gap = 100 * float(_np.finfo(float).eps) * max(abs(a), abs(b), 1.0)
        r = r[(r > a + gap) & (r < b - gap)]
        base = self if len(r) == 0 else self.addBreaks(r)
        return base._apply_fun(jnp.log1p)

    def reallog(self) -> Chebfun:
        """Real natural logarithm of the Chebfun.

        Errors if ``f`` is complex-valued (MATLAB raises
        ``CHEBFUN:CHEBFUN:reallog:complex``).

        Returns
        -------
        Chebfun

        Raises
        ------
        ValueError
            If the Chebfun is not real-valued.

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/reallog.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.log, Chebfun.realsqrt
        """
        if not self.isreal():
            raise ValueError("reallog produced complex result.")
        return self.addBreaksAtRoots()._apply_fun(jnp.log)

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
        MATLAB source : @chebfun/power.m (columnPower), @classicfun/power.m,
                        @singfun/power.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.fun.singfun import Singfun
        # The identically-zero function: 0**b = 0 for b > 0.  Falling
        # through to the root-splitting path below would ask roots()
        # for the roots of the zero function -- every point is one --
        # which hangs.  (Operator probing evaluates ops on the zero
        # function, so a fractional power in the op hit this.)
        try:
            _is_zero = float(self.norm(jnp.inf)) == 0.0
        except Exception:
            _is_zero = False
        if _is_zero and b > 0:
            return self * 0.0
        if _is_zero and b < 0:
            # 0**b is Inf everywhere: MATLAB's constructor cannot
            # extrapolate a grid with no finite sample (chebop's
            # linearize maps this to invalidInitialGuess).
            raise ValueError(
                "CHEBFUN:CHEBTECH:extrapolate:nansInfs: "
                "Too many NaNs/Infs to handle.")

        # MATLAB @chebfun/power.m partitions noninteger complex powers at
        # roots of imag(f), including crossings where real(f) is positive.
        # Finite endpoint Singfuns retain their singular factors while their
        # smooth parts are composed. Unbounded and periodic representations
        # retain their existing dispatch until their source paths qualify.
        from chebfunjax.tech.chebtech import Chebtech1
        complex_bounded = (
            bool(self.funs)
            and
            not self.isreal()
            and b != int(b)
            and all(isinstance(piece.tech, (Chebtech1, Chebtech2, Singfun))
                    and all(math.isfinite(float(end))
                            for end in piece.interval)
                    for piece in self.funs)
        )
        if complex_bounded:
            domain_points = [float(value) for value in self.domain.breakpoints]
            # This uses the current public roots() result. Its broader
            # existing deduplication may already have merged very close
            # roots, which this source-level filter cannot recover.
            raw_roots = jnp.ravel(
                self.imag().roots(nojump=True, nozerofun=True))
            roots = sorted(
                float(value) for value in raw_roots
                if math.isfinite(float(value)))

            # Literal getRootsForBreaks neighbor filter. The comparison is
            # against the prior root in the sorted, finite, unfiltered list,
            # matching MATLAB's simultaneous logical-index deletion.
            hscale = max(abs(domain_points[0]), abs(domain_points[-1]))
            root_tol = _EPS * hscale
            separated_roots = [
                value for index, value in enumerate(roots)
                if index == 0 or value - roots[index - 1] >= root_tol
            ]

            # Literal addBreaks proximity rule on a finite domain. It uses
            # the smallest existing panel width, floored at one.
            min_width = min(
                right - left
                for left, right in zip(domain_points[:-1], domain_points[1:])
            )
            break_tol = 100 * _EPS * max(min_width, 1.0)
            added_roots = [
                value for value in separated_roots
                if domain_points[0] < value < domain_points[-1]
                and all(abs(value - point) >= break_tol
                        for point in domain_points)
            ]
            fbr = (
                self._with_breakpoints(tuple(sorted(set(domain_points + added_roots))))
                if added_roots else self
            )
            op = lambda value: jnp.power(value, b)  # noqa: E731

            def _piece_power(piece):
                tech = piece.tech
                # @classicfun/power.m casts endpoint zeros to SINGFUN
                # before powering its onefun, so fractional roots retain
                # their algebraic exponents instead of a smooth interpolant.
                if isinstance(tech, (Chebtech1, Chebtech2)):
                    root_tol = 1e3 * _EPS * tech.vscale
                    if (bool(jnp.any(jnp.abs(tech(jnp.asarray(-1.0))) < root_tol))
                            or bool(jnp.any(jnp.abs(tech(jnp.asarray(1.0))) < root_tol))):
                        tech = Singfun(tech, (0.0, 0.0))
                if isinstance(tech, Singfun):
                    return piece.with_tech(tech ** b)
                return piece._apply_fun(op, extrapolate=True)

            new_funs = [_piece_power(piece) for piece in fbr.funs]
            result = Chebfun(funs=new_funs, domain=fbr.domain)
            result = Chebfun._as_transposed(result, self.is_transposed)
            # MATLAB maps the stored value at every breakpoint through the
            # principal complex power after composing the smooth pieces.
            # Source restrict() derives new values from the restricted FUNs
            # and then restores stored values at every old breakpoint.
            point_values = fbr._breakpoint_values().astype(jnp.complex128)
            old_values = self._breakpoint_values()
            refined_points = [float(value) for value in fbr.domain.breakpoints]
            for index, point in enumerate(domain_points):
                point_values = point_values.at[refined_points.index(point)].set(
                    old_values[index])
            return result.set_point_values(jnp.power(point_values, b))

        # No roots anywhere -> smooth composition (fast path, matches the
        # positive-function tests exactly).
        r = _np.asarray(self.roots(nojump=True), dtype=float).ravel()
        r = r[_np.isfinite(r)]
        if len(r) == 0:
            # Exponent-aware even without interior roots: a Singfun piece
            # must scale its exponents (f^b keeps the singularity), not be
            # re-approximated smoothly.
            if any(isinstance(p.tech, Singfun) for p in self.funs):
                new_funs = [
                    p.with_tech(p.tech ** b)
                    if isinstance(p.tech, Singfun) else p._apply_fun(smooth_op)
                    for p in self.funs
                ]
                result = Chebfun(funs=new_funs, domain=self.domain)
                result = Chebfun._as_transposed(result, self.is_transposed)
                return result.set_point_values(jnp.power(self._breakpoint_values(), b))
            def principal_smooth(value):
                # MATLAB POWER promotes negative real bases to the principal
                # complex branch for noninteger powers (CHEBTECH/POWER).
                if b != int(b) and not jnp.iscomplexobj(value):
                    if bool(jnp.any(value < 0)):
                        value = value.astype(jnp.complex128)
                return smooth_op(value)

            return self._apply_fun(principal_smooth)
        # Split at interior roots so every remaining root sits on a breakpoint.
        fbr = self.addBreaksAtRoots()
        new_funs = []
        for p in fbr.funs:
            tech = p.tech
            # A piece on which a real f is negative: MATLAB's power takes
            # the principal branch, (-a)^b = a^b exp(i pi b), so the
            # result is complex rather than NaN (chebop Newton iterates
            # of u^1.5 dip below zero near a root of u -- Lane-Emden).
            _neg = False
            if not jnp.iscomplexobj(getattr(tech, "coeffs", jnp.zeros(1))):
                try:
                    _mid = 0.5 * (p.interval[0] + p.interval[1])
                    _neg = float(_np.real(_np.asarray(
                        p(jnp.asarray(_mid))))) < 0
                except Exception:
                    _neg = False
            if _neg:
                tech = -tech
            if isinstance(tech, Singfun):
                sf = tech ** b
            else:
                sf = (Singfun.from_chebtech(tech, (0.0, 0.0))
                      .extractBoundaryRoots() ** b).simplifyExponents()
            if _neg:
                _ph = complex(_np.exp(1j * _np.pi * b))
                if all(abs(e) < 1e-14 for e in sf.exponents):
                    new_funs.append(_Piece.from_function(
                        lambda x, _p=p: _ph * smooth_op(-_p(x)),
                        p.interval[0], p.interval[1]))
                else:
                    new_funs.append(_Piece(tech=sf * _ph, interval=p.interval))
                continue
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
        from ._array_abs import bounded_real_array, source_array_abs
        if bounded_real_array(self):
            return source_array_abs(self)
        return self._propagate_point_values(self._abs_core(), jnp.abs)

    def _abs_core(self) -> Chebfun:
        """Root-splitting ``|f|`` without pointValues propagation."""
        # Bounded scalar real polynomials retain their representation through
        # root partitioning. Other representations keep their existing path.
        from chebfunjax.tech.chebtech import Chebtech1
        if self.isempty():
            return self
        bounded_real = self.n_columns == 1 and all(
            isinstance(p.tech, (Chebtech1, Chebtech2))
            and all(math.isfinite(float(v)) for v in p.interval)
            and bool(jnp.all(jnp.imag(p.tech.coeffs) == 0))
            for p in self.funs)
        if bounded_real:
            # @chebfun/getRootsForBreaks.m uses exact-zero suppression;
            # small nonzero polynomials must retain their roots and scale.
            r = jnp.ravel(self.roots(nojump=True, nozerofun=True))
            roots = sorted(float(v) for v in r if math.isfinite(float(v)))
            old = [float(v) for v in self.domain.breakpoints]
            root_tol = _EPS * max(abs(old[0]), abs(old[-1]))
            # Source discards each original neighbor gap simultaneously.
            roots = [v for i, v in enumerate(roots)
                     if i == 0 or v - roots[i - 1] >= root_tol]
            break_tol = 100 * _EPS * max(min(b - a for a, b in
                                            zip(old[:-1], old[1:])), 1.0)
            added = [v for v in roots if old[0] < v < old[-1]
                     and all(abs(v - b) >= break_tol for b in old)]
            bps = tuple(sorted(set(old + added)))
            restricted = (self if bps == tuple(old)
                          else self._with_breakpoints(bps))
            out = Chebfun._as_transposed(Chebfun(
                funs=[p.abs() for p in restricted.funs],
                domain=restricted.domain).simplify(), self.is_transposed)
            # Keep old pointValues and force exactly zero at newly inserted
            # roots, as @chebfun/addBreaksAtRoots.m requires.
            point_values = jnp.abs(self(jnp.asarray(bps)))
            for i, v in enumerate(bps):
                if v in added:
                    point_values = point_values.at[i].set(0.0)
            return out.set_point_values(point_values)

        # Find roots where the function changes sign and add them as
        # breakpoints, then apply |·| piecewise for smoothness.  Jump
        # roots are excluded: a sign flip through a jump or pole sits
        # at an existing breakpoint, and re-splitting there would
        # rebuild singular pieces as (unhappy) smooth ones.
        import numpy as _np
        roots = _np.asarray(self.roots(nojump=True))
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

    def cumprod(self) -> "Chebfun":
        """Compute the indefinite product integral exp(cumsum(log(f))).

        Provenance
        ----------
        MATLAB source: @chebfun/cumprod.m
        Chebfun commit: 7574c77
        """
        return self.log().cumsum().exp()

    def prod(self) -> jax.Array:
        """Integral product: exp(sum(log(f))) (MATLAB prod).

        Provenance
        ----------
        MATLAB source : @chebfun/prod.m
        Chebfun commit: 7574c77
        """
        return jnp.exp(self.log().sum())

    def compose(self, op, g: "Chebfun" = None, *, pref=None) -> "Chebfun":
        """Compose a pointwise operator on each overlapping input interval.

        Stored breakpoint values are transformed independently of piece
        limits. Preferences select approximation tolerance; numeric elliptic
        tolerances do not change these composition preferences.

        Provenance
        ----------
        MATLAB source : @chebfun/compose.m
        Chebfun commit: 7574c77
        """
        import math

        from chebfunjax.chebfun1d.linalg import Quasimatrix
        from chebfunjax.chebpref import ChebfunPref
        from chebfunjax.fun.singfun import Singfun
        from chebfunjax.tech.trigtech import Trigtech

        from ._composition import compose_object

        handled, result = compose_object(self, op, pref)
        if handled:
            return result
        if isinstance(g, Quasimatrix):
            # Source mixed array/quasimatrix binary path extracts columns.
            from chebfunjax.chebfun1d._composition import _columns
            return Quasimatrix(_columns(self), self.domain).compose(op, g, pref=pref)
        if g is None and pref is None:
            return self._apply_fun(op)
        # Keep explicit periodic options distinct from the Chebyshev factory
        # maxLength. Trigtech's own factory cap is 65536.
        trig_pref = {}
        if isinstance(pref, dict):
            trig_pref.update(pref.get("techPrefs", {}))
            trig_pref.update({k: v for k, v in pref.items() if k != "techPrefs"})
        elif pref is not None:
            factory = ChebfunPref().techPrefs
            trig_pref.update(pref.techPrefs)
            if trig_pref.get("maxLength") == factory.get("maxLength"):
                trig_pref.pop("maxLength", None)
        pref = ChebfunPref(pref) if pref is not None else ChebfunPref()
        if not self.funs:
            return self
        if g is not None and self.is_transposed != g.is_transposed:
            raise ValueError("Cannot compose row and column Chebfuns")
        original_g = g
        f, g = self._overlap(self, g) if g is not None else (self, None)
        # The existing overlap adapter only rebreaks smooth pieces. Recover
        # original stored values at the union, including isolated point jumps.
        sites = jnp.asarray(f.domain.breakpoints)
        values = self.point_values if g is None else self(sites)
        if g is None:
            values = op(values)
        else:
            other = original_g(sites)
            if values.ndim < other.ndim:
                values = values[..., None]
            elif other.ndim < values.ndim:
                other = other[..., None]
            values = op(values, other)
        pieces = []
        new_breaks = [f.domain.breakpoints[0]]
        new_values = [values[0]]
        maxpow2 = (int(math.floor(math.log2(max(int(pref.splitPrefs.splitLength)-1, 2))))
                   if pref.splitting else 16)
        for k, piece in enumerate(f.funs):
            if isinstance(piece.tech, Chebtech2):
                tech = piece.tech.compose(
                    op, None if g is None else g.funs[k].tech,
                    extrapolate=len(f.funs) > 1 or pref.extrapolate,
                    tol=pref.chebfuneps, maxpow2=maxpow2)
                pieces.append(piece.with_tech(tech))
            elif isinstance(piece.tech, Trigtech):
                tech = piece.tech.compose(op, None if g is None else g.funs[k].tech,
                                          pref=trig_pref)
                pieces.append(piece.with_tech(tech))
            elif isinstance(piece.tech, Singfun):
                pieces.append(piece.with_tech(piece.tech.compose(
                    op, None if g is None else g.funs[k].tech)))
            elif g is None:
                pieces.append(piece._apply_fun(op))
            else:
                a, b = piece.interval
                other_piece = g.funs[k]
                pieces.append(_Piece.from_function(
                    lambda x, p=piece, q=other_piece: op(p(x), q(x)), a, b))
            if pref.splitting and not pieces[-1].ishappy:
                # Source columnCompose retries an unresolved interval through
                # the splitting-enabled Chebfun constructor. Old endpoints
                # retain the independently transformed stored pointValues.
                def evaluate(x):
                    left = f(x)
                    if g is None:
                        return op(left)
                    right = g(x)
                    if left.ndim < right.ndim:
                        left = left[..., None]
                    elif right.ndim < left.ndim:
                        right = right[..., None]
                    return op(left, right)
                a, b = piece.interval
                sub = chebfun(evaluate, domain=(a, b), splitting=True,
                              split_length=pref.splitPrefs.splitLength,
                              split_max_length=pref.splitPrefs.splitMaxLength,
                              eps=pref.chebfuneps, extrapolate=bool(pref.extrapolate),
                              sample_test=bool(pref.sampleTest))
                pieces.pop()
                pieces.extend(sub.funs)
                new_breaks.extend(sub.domain.breakpoints[1:-1])
                new_values.extend(sub.point_values[1:-1])
            new_breaks.append(f.domain.breakpoints[k+1])
            new_values.append(values[k+1])
        result = Chebfun(funs=pieces, domain=Domain(new_breaks)).set_point_values(jnp.stack(new_values))
        return Chebfun._as_transposed(result, self.is_transposed)

    def compose_chebfun(self, g):
        """Return self(g) through the source typed composition dispatch.

        Provenance
        ----------
        MATLAB source : @chebfun/compose.m, @chebfun/subsref.m
        Chebfun commit: 7574c77
        """
        return g.compose(self)

    def isPeriodicTech(self):
        """Whether the first FUN uses the source periodic technology.

        Provenance
        ----------
        MATLAB source : @chebfun/isPeriodicTech.m
        Chebfun commit: 7574c77
        """
        from ._composition import periodic
        return periodic(self)

    @staticmethod
    def _chebT2U(c):
        """Convert first-kind (T) coefficients to second-kind (U) ones.

        Runs the recurrence ``T_n = (U_n - U_{n-2})/2`` column-wise, so the
        coefficient of ``U_n`` needs those of ``T_n`` and ``T_{n+2}``.

        Provenance
        ----------
        MATLAB source : @chebtech/chebTcoeffs2chebUcoeffs.m
        Chebfun commit: 7574c77
        """
        c = jnp.asarray(c)
        if c.shape[0] == 0:
            return c
        pad = jnp.zeros((2,) + c.shape[1:], dtype=c.dtype)
        cu = jnp.concatenate([c, pad], axis=0)
        cu = cu.at[0].set(2.0 * c[0])
        return 0.5 * (cu[:-2] - cu[2:])

    def chebcoeffs(self, n: int | None = None, kind: int = 1) -> jax.Array:
        """Chebyshev expansion coefficients (MATLAB ``chebcoeffs``).

        With no arguments this returns the coefficients of a single-piece
        (global) Chebfun, so that ``f = a[0] T_0 + ... + a[N-1] T_{N-1}``
        on the Chebfun's domain transplanted to ``[-1, 1]``.

        Parameters
        ----------
        n : int or None, optional
            Number of coefficients to return; the array is zero-padded or
            truncated to this length.  ``n`` is REQUIRED for a piecewise
            Chebfun, whose coefficients are then obtained from weighted
            inner products rather than from a global expansion.
        kind : {1, 2}, optional
            ``1`` (default) expands in the Chebyshev polynomials of the
            first kind ``T_k``; ``2`` expands in those of the second kind
            ``U_k``.

        Returns
        -------
        jax.Array, shape (n,) or (n, n_columns)

        Raises
        ------
        ValueError
            If ``n`` is missing for a piecewise Chebfun, if the domain is
            unbounded, or if ``kind`` is not 1 or 2.

        Notes
        -----
        For a piecewise Chebfun MATLAB evaluates
        ``(2/pi) * innerProduct(f*w, T_k)`` with the Chebyshev weight
        ``w``.  Substituting ``x = (a+b)/2 + (b-a)/2 cos(theta)`` turns
        that singular integral into a smooth one over ``theta``, which is
        evaluated here by Gauss-Legendre quadrature on each smooth
        ``theta``-subinterval.

        Provenance
        ----------
        MATLAB source : @chebfun/chebcoeffs.m, @chebtech/chebcoeffs.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.legcoeffs, Chebfun.trigcoeffs
        """
        import numpy as _np
        if kind not in (1, 2):
            raise ValueError("chebcoeffs: 'kind' input must be 1 or 2.")
        if self.isempty():
            return jnp.zeros((0,), dtype=jnp.float64)
        f = self if len(self.funs) == 1 else self.merge()
        if n is None and len(f.funs) > 1:
            raise ValueError(
                "chebcoeffs: input N is required for piecewise Chebfun "
                "objects.")
        a, b = float(f.domain.a), float(f.domain.b)
        if not (_np.isfinite(a) and _np.isfinite(b)):
            raise ValueError(
                "chebcoeffs: infinite intervals are not supported here.")

        if len(f.funs) == 1:
            if kind == 2:
                # The U_n coefficient needs T_n and T_{n+2}, so ask for two
                # extra first-kind coefficients before converting.
                c = Chebfun._chebT2U(
                    f.chebcoeffs(None if n is None else n + 2, 1))
            else:
                c = jnp.asarray(f.funs[0].tech.coeffs)
            if n is not None:
                m = int(c.shape[0])
                if n > m:
                    pad = jnp.zeros((n - m,) + c.shape[1:], dtype=c.dtype)
                    c = jnp.concatenate([c, pad], axis=0)
                c = c[:n]
            return c

        # Piecewise: weighted inner products.  With x = mid + half*cos(th)
        # the first-kind integral is  (2/pi) * int_0^pi F(th) cos(k th) dth
        # (the T_0 term is halved), and the second-kind one is
        # (2/pi) * half^2 * int_0^pi F(th) sin((k+1) th) sin(th) dth.
        from chebfunjax.utils.quadrature import legpts

        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        n_cols = f.n_columns
        # theta breakpoints: theta = arccos((x - mid)/half), decreasing in x.
        xs = [float(p.interval[0]) for p in f.funs] + [b]
        th = _np.arccos(
            _np.clip((_np.asarray(xs) - mid) / half, -1.0, 1.0))[::-1]
        out = _np.zeros((n, n_cols)) if n_cols > 1 else _np.zeros(n)
        for i in range(len(th) - 1):
            t0, t1 = float(th[i]), float(th[i + 1])
            if t1 <= t0:
                continue
            piece = f.funs[len(th) - 2 - i]
            deg = int(_np.asarray(piece.tech.coeffs).shape[0])
            # The integrand is a trigonometric polynomial in theta of
            # degree <= deg + n; Gauss-Legendre with this many nodes
            # resolves it to machine precision.
            nq = min(2 * (deg + n) + 20, 20000)
            tq, wq = legpts(nq, (t0, t1))
            tq = _np.asarray(tq)
            wq = _np.asarray(wq)
            fq = _np.asarray(f(jnp.asarray(mid + half * _np.cos(tq))))
            kk = _np.arange(n)
            if kind == 1:
                basis = _np.cos(_np.outer(kk, tq))
            else:
                basis = (half ** 2) * _np.sin(_np.outer(kk + 1, tq)) \
                    * _np.sin(tq)[None, :]
            out = out + (basis * wq[None, :]) @ fq
        if kind == 1:
            out[0] = out[0] / 2.0
        out = (2.0 / _np.pi) * out
        return jnp.asarray(out, dtype=jnp.float64)

    def chebpoly(self, n: int | None = None, kind: int = 1) -> jax.Array:
        """Chebyshev coefficients, highest degree first (deprecated).

        MATLAB keeps ``chebpoly`` only as a deprecated accessor for
        :meth:`chebcoeffs`: the coefficients are reversed and transposed,
        so a scalar-valued Chebfun gives a row of length ``n`` and an
        array-valued one gives an ``(n_columns, n)`` matrix.

        Parameters
        ----------
        n : int or None, optional
            Number of coefficients; see :meth:`chebcoeffs`.
        kind : {1, 2}, optional
            Chebyshev polynomial kind; see :meth:`chebcoeffs`.

        Returns
        -------
        jax.Array, shape (n,) or (n_columns, n)

        Provenance
        ----------
        MATLAB source : @chebfun/chebpoly.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.chebcoeffs
        """
        c = self.chebcoeffs(n, kind)
        return jnp.flip(c, axis=0).T

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

    def addBreaks(self, breaks, tol: float = 0.0) -> "Chebfun":
        """Introduce new interior breakpoints (MATLAB addBreaks).

        Provenance
        ----------
        MATLAB source : @chebfun/addBreaks.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        a, b = float(self.domain.a), float(self.domain.b)
        old = jnp.asarray(self.domain.breakpoints)
        new = jnp.unique(jnp.asarray(breaks, dtype=jnp.float64).reshape(-1))
        new = new[jnp.isfinite(new)]
        finite = old[jnp.isfinite(old)]
        # Source addBreaks.m uses the shortest finite existing interval,
        # and tests distance to every existing breakpoint before restricting.
        if finite.size > 1:
            break_tol = max(100 * float(jnp.finfo(jnp.float64).eps)
                            * max(float(jnp.min(jnp.diff(finite))), 1.0), tol)
            distance = jnp.abs(new[:, None] - finite[None, :])
            new = new[~jnp.any(distance < break_tol, axis=1)]
        if new.size == 0:
            return self
        pts = sorted(set(_np.asarray(old).tolist())
                     | {float(t) for t in new if a < float(t) < b})
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
        r_all = jnp.asarray(self.roots(nojump=True, nozerofun=True))
        r = _np.asarray(r_all, dtype=float).ravel()
        # getRootsForBreaks.m discards NaNs and adjacent roots closer
        # than eps*hscale; addBreaks handles proximity to existing breaks.
        a, b = float(self.domain.a), float(self.domain.b)
        hscale = max(abs(a), abs(b))
        if not _np.isfinite(hscale):
            hscale = 1.0
        gap = max(tol, float(_np.finfo(float).eps) * hscale)
        r = _np.sort(r[~_np.isnan(r)])
        if len(r):
            r = r[_np.r_[True, _np.diff(r) >= gap]]
        if len(r) == 0:
            return self
        out = self.addBreaks(r, tol)
        if tuple(out.domain.breakpoints) != tuple(self.domain.breakpoints):
            # Source addBreaksAtRoots sets pointValues=0 at original roots,
            # only after an actual new breakpoint was introduced.
            ends = jnp.asarray(out.domain.breakpoints)
            values = _source_breakpoint_values(out.funs, out.domain.breakpoints, op=self)
            values2 = values[:, None] if values.ndim == 1 else values
            roots2 = r_all[:, None] if r_all.ndim == 1 else r_all
            at_root = jnp.any(ends[:, None, None] == roots2[None, :, :], axis=1)
            values2 = jnp.where(at_root, 0.0, values2)
            object.__setattr__(out, '_point_values', values2[:, 0] if values.ndim == 1 else values2)
        return out

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

    def merge(self, index=None, *, maxpow2: int | None = None,
              tol: float | None = None, sample_test: bool = True,
              min_samples: int | None = None,
              refinement_function: str | Callable | None = None,
              max_length: int | None = None, splitting: bool | None = None,
              turbo: bool = False, check: str = "standard") -> "Chebfun":
        """Remove supported smooth or singular breakpoints in one source-ordered pass.

        Python index values remain breakpoint LOCATIONS, preserving this public
        API. Source breakpoint indices are resolved internally by exact equality.
        Periodic pieces retain the existing adapter.
        Raw max_length and explicit splitting carry constructor preferences;
        legacy maxpow2 still implies splitting when splitting is omitted.

        Provenance
        ----------
        MATLAB source : @chebfun/merge.m, @fun/merge.m
        Chebfun commit: 7574c77
        """
        import warnings

        from chebfunjax.fun.singfun import Singfun
        from chebfunjax.tech.chebtech import Chebtech1

        if len(self.funs) < 2:
            return self
        from chebfunjax.fun.unbndfun import Unbndfun
        if any(not isinstance(p.tech, (Chebtech1, Chebtech2, Singfun))
               or (not all(math.isfinite(t) for t in p.interval)
                   and not isinstance(p, Unbndfun)) for p in self.funs):
            # Periodic and unsupported representation adapters remain separate.
            if max_length is not None or splitting is not None or turbo or check != "standard":
                raise NotImplementedError(
                    "Source merge preferences require smooth or singular polynomial technologies.")
            return self._merge_representation_adapter(
                index, maxpow2=maxpow2, tol=tol, sample_test=sample_test,
                min_samples=min_samples, refinement_function=refinement_function)
        old_ends = tuple(float(x) for x in self.domain.breakpoints)
        if index is None or isinstance(index, str) and index.lower() == "all":
            selected = list(range(1, len(old_ends) - 1))
        else:
            allowed = {float(x) for x in index}
            selected = [k for k in range(1, len(old_ends) - 1)
                        if old_ends[k] in allowed]
        if not selected:
            return self
        cap = (2 ** (16 if maxpow2 is None else int(maxpow2)) + 1
               if max_length is None else int(max_length))
        if cap < 1:
            raise ValueError("max_length must be a positive integer")
        splitting = maxpow2 is not None if splitting is None else bool(splitting)
        eps = float(jnp.finfo(jnp.float64).eps)
        tolerance = jnp.maximum(eps, jnp.asarray(eps if tol is None else tol))
        vs = jnp.asarray(self.vscale, dtype=jnp.float64)
        hs = max(abs(old_ends[0]), abs(old_ends[-1]))
        if math.isinf(hs):
            hs = 1.0  # @chebfun/hscale.m, including bounded intermediate trials.
        old_funs = tuple(self.funs)
        explicit = getattr(self, "_point_values", None)
        point_values = (_source_breakpoint_values(old_funs, old_ends)
                        if explicit is None else jnp.asarray(explicit))
        old_values = point_values[:, None] if point_values.ndim == 1 else point_values
        funs = list(old_funs)
        ends = list(old_ends)
        retained_rows = list(range(len(old_ends)))
        for k in selected:
            j = ends.index(old_ends[k])
            left, right = funs[j - 1], funs[j]
            if left.n + right.n >= 1.2 * cap:
                continue
            # Original point values and original one-sided limits are invariant
            # throughout the pass, even when earlier current neighbors changed.
            limits = jnp.stack((_merge_limit_row(old_funs[k - 1], True),
                                _merge_limit_row(old_funs[k], False)))
            differences = old_values[k][None, :] - limits
            # MATLAB matrix norm(...,inf) is max row SUM, not max entry.
            jumps = jnp.max(jnp.sum(jnp.abs(differences), axis=1)) / vs
            if (bool(jnp.all(jumps >= 1e3 * tolerance))
                    or bool(jnp.any(jnp.isinf(jnp.concatenate(
                        (old_values[k][None, :], limits), axis=0))))):
                continue
            if abs(left.interval[1] - right.interval[0]) > hs * float(jnp.max(tolerance)):
                raise ValueError("F and G must be on consecutive domains.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trial = _merge_fun_source(
                    left, right,
                    maxpow2=16 if maxpow2 is None else int(maxpow2),
                    max_length=cap, tol=tolerance, splitting=splitting,
                    vscale=float(vs), hscale=hs, sample_test=sample_test,
                    min_samples=min_samples, refinement_function=refinement_function,
                    turbo=turbo, check=check)
            if not trial.ishappy:
                continue
            funs[j - 1:j + 1] = [trial]
            del ends[j]
            del retained_rows[j]
        if len(funs) == len(old_funs):
            return self
        result = Chebfun(funs=funs, domain=Domain(tuple(ends)), deltas=self.deltas)
        kept = point_values[jnp.asarray(retained_rows, dtype=jnp.int32)]
        object.__setattr__(result, "_point_values", kept)
        if self.is_transposed:
            object.__setattr__(result, "_is_transposed", True)
        return result

    def _merge_representation_adapter(self, index=None, *, maxpow2: int | None = None,
              tol: float | None = None, sample_test: bool = True,
              min_samples: int | None = None,
              refinement_function: str | Callable | None = None) -> "Chebfun":
        """Remove unnecessary interior breakpoints (MATLAB merge):
        re-approximate globally and keep the merged representation if
        it matches the piecewise one.

        ``index`` restricts the attempt to the given interior breakpoint
        LOCATIONS (MATLAB ``merge(f, index)``; the constructor merges only
        the breakpoints it introduced itself).  ``maxpow2`` caps the
        merged piece at ``2**maxpow2 + 1`` points (MATLAB caps at
        ``splitLength`` under splitting).

        ``tol``, ``sample_test``, ``min_samples`` and
        ``refinement_function`` control both the global fit and pairwise
        merge trials.

        Provenance
        ----------
        MATLAB source : @chebfun/merge.m
        Chebfun commit: 7574c77
        """
        import warnings as _w

        import numpy as _np
        if len(self.funs) == 1:
            return self
        a, b = float(self.domain.a), float(self.domain.b)
        _mp2 = 16 if maxpow2 is None else int(maxpow2)
        _allowed = None
        if index is not None:
            _allowed = [float(v) for v in index]
        merge_tol = 1e3 * float(_np.finfo(float).eps) * max(self.vscale, 1.0)
        _all_interior = [float(v) for v in self.domain.breakpoints[1:-1]]
        # fast path: a single global piece (only when every interior
        # breakpoint is up for removal)
        if _allowed is None or all(
                any(abs(t - u) <= 1e-14 * max(abs(t), 1.0) for u in _allowed)
                for t in _all_interior):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                cand = Chebfun.from_function(
                    lambda x: self(x), Domain((a, b)), maxpow2=_mp2,
                    extrapolate=maxpow2 is not None, tol=tol,
                    sample_test=sample_test, min_samples=min_samples,
                    refinement_function=refinement_function)
            xs = jnp.asarray(_np.linspace(a + 1e-9 * (b - a),
                                          b - 1e-9 * (b - a), 201))
            err = float(jnp.max(jnp.abs(cand(xs) - self(xs))))
            if err < merge_tol and cand.funs[0].tech.ishappy:
                return cand
        # MATLAB merge.m removes breakpoints ONE AT A TIME: each
        # interior breakpoint is dropped if the union of its two
        # neighbouring pieces re-approximates happily (true jumps and
        # singular pieces keep their breakpoints).
        funs = list(self.funs)
        changed = True
        while changed:
            changed = False
            for k in range(len(funs) - 1):
                p, q = funs[k], funs[k + 1]
                from chebfunjax.fun.singfun import Singfun
                if isinstance(p.tech, Singfun) or isinstance(
                        q.tech, Singfun):
                    continue
                aa = float(p.interval[0])
                bb = float(q.interval[1])
                _bp = float(p.interval[1])
                if _allowed is not None and not any(
                        abs(_bp - u) <= 1e-14 * max(abs(_bp), 1.0)
                        for u in _allowed):
                    continue
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    trial = _Piece.from_function(
                        lambda x: self(x), aa, bb,
                        maxpow2=min(10, _mp2) if maxpow2 is None else _mp2,
                        extrapolate=maxpow2 is not None, tol=tol,
                        sample_test=sample_test, min_samples=min_samples,
                        refinement_function=refinement_function)
                if not getattr(trial.tech, "ishappy", False):
                    continue
                t = _np.linspace(aa + 1e-9 * (bb - aa),
                                 bb - 1e-9 * (bb - aa), 101)
                e = float(_np.max(_np.abs(
                    _np.asarray(trial(jnp.asarray(t)))
                    - _np.asarray(self(jnp.asarray(t))))))
                if e < merge_tol:
                    funs[k:k + 2] = [trial]
                    changed = True
                    break
        if len(funs) == len(self.funs):
            return self
        bps = [funs[0].interval[0]] + [pc.interval[1] for pc in funs]
        return Chebfun(funs=funs,
                       domain=Domain(tuple(float(v) for v in bps)))

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
        """Arc length (alias of :meth:`arc_length`; the older
        real-graph-only implementation broke on complex paths and
        piecewise inputs).

        Provenance
        ----------
        MATLAB source : @chebfun/arcLength.m
        Chebfun commit: 7574c77
        """
        return jnp.asarray(self.arc_length())

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
                         + [float(r) for r in _np.asarray(self.roots(nojump=True))]))
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

    def join(self, *others: "Chebfun") -> "Chebfun":
        """Join pieces with source interval-length accumulation and orientation.

        Every interval after the first begins at its predecessor's right
        endpoint; its right endpoint adds its original interval length.
        Equal column counts and transposition states are required.

        Provenance
        ----------
        MATLAB source : @chebfun/join.m (columnJoin and array-valued branch)
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.unbndfun import Unbndfun

        if not others:
            return self
        inputs = (self, *others)
        if any(f.is_transposed != self.is_transposed for f in inputs):
            raise ValueError("All inputs to join must have the same transposition state")
        counts = {f.n_columns for f in inputs}
        if len(counts) > 1:
            raise ValueError("join: Matrix dimensions must agree")
        funs = [piece for f in inputs for piece in f.funs]
        if not funs:
            return Chebfun._as_transposed(Chebfun.empty(), self.is_transposed)
        pieces = [funs[0]]
        for piece in funs[1:]:
            left = pieces[-1].interval[1]
            right = left + (piece.interval[1]-piece.interval[0])
            interval = (left, right)
            if isinstance(piece, _Piece):
                remapped = _Piece(tech=piece.tech, interval=interval)
            elif isinstance(piece, Unbndfun):
                # Retain the nonlinear map, including its singular onefun.
                remapped = Unbndfun.from_chebtech(piece.onefun, Domain(interval))
            elif hasattr(piece, "change_map"):
                remapped = piece.change_map(interval)
            else:
                raise NotImplementedError(
                    f"join cannot remap {type(piece).__name__}")
            pieces.append(remapped)
        domain = Domain((pieces[0].interval[0],) + tuple(p.interval[1] for p in pieces))
        return Chebfun._as_transposed(Chebfun(funs=pieces, domain=domain), self.is_transposed)

    def inv(self, pref=None, *, algorithm="brent", eps=None,
            splitting=None, monocheck=False, rangecheck=False) -> "Chebfun":
        """Compositional inverse of a real monotonic chebfun.

        ``algorithm`` selects ``'roots'``, ``'newton'``, ``'bisection'``,
        ``'regulafalsi'``, ``'illinois'`` or the default ``'brent'``.
        ``eps`` sets construction tolerance, ``splitting`` enables edge
        detection, ``monocheck`` checks derivative roots, and ``rangecheck``
        adjusts the inverse's range to this function's domain. Boolean
        options also accept MATLAB's ``'on'`` and ``'off'`` strings.
        An optional ChebfunPref supplies construction preferences.

        Provenance
        ----------
        MATLAB source : @chebfun/inv.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.inverse import _inverse

        return _inverse(self, pref, algorithm=algorithm, eps=eps,
                        splitting=splitting, monocheck=monocheck,
                        rangecheck=rangecheck)

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
        """Pointwise equality as a logical Chebfun (MATLAB ``f == g``).

        Returns a piecewise-constant Chebfun that is 1 on any subinterval
        where ``f`` and ``g`` agree identically and 0 elsewhere.  Isolated
        crossings, where the two agree at a single point, show up in
        :meth:`point_values` as a 1 at the corresponding breakpoint --
        exactly how MATLAB records them.

        Parameters
        ----------
        other : Chebfun or float
            Right-hand side of the comparison.

        Returns
        -------
        Chebfun
            The logical Chebfun, carrying explicit ``pointValues``.

        Raises
        ------
        ValueError
            If either operand is array-valued (MATLAB raises
            ``CHEBFUN:CHEBFUN:eq:array``).

        Notes
        -----
        Follows MATLAB: ``h = sign(f - g)``, each FUN is replaced by the
        constant 1 (if it was identically zero) or 0, the pointValues are
        logically negated, and the result is merged.  As in MATLAB's
        ``merge``, a breakpoint whose pointValue differs from the function
        value on both sides is retained.

        Provenance
        ----------
        MATLAB source : @chebfun/eq.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.logical_ne, Chebfun.roots, Chebfun.sign
        """
        import numpy as _np
        if self.isempty():
            return self
        if isinstance(other, Chebfun):
            if other.isempty():
                return other
            if self.n_columns > 1 or other.n_columns > 1:
                raise ValueError(
                    "==  does not support array-valued Chebfun objects.")
            diff = self - other
        else:
            if self.n_columns > 1:
                raise ValueError(
                    "==  does not support array-valued Chebfun objects.")
            diff = self - float(other)

        h = diff.sign()
        bps = _np.asarray(list(h.domain.breakpoints), dtype=float)
        # A FUN that sign() left identically zero means f == g there.
        consts = []
        for piece in h.funs:
            mid = 0.5 * (float(piece.interval[0]) + float(piece.interval[1]))
            consts.append(1.0 if abs(float(piece(jnp.asarray(mid))))
                          < 0.5 else 0.0)
        # pointValues: sign(f-g) vanishes exactly at the roots that sign()
        # broke the domain at, and MATLAB negates it (~0 == 1).
        r = _np.asarray(diff.roots(nojump=True), dtype=float).ravel()
        htol = 1e6 * float(_np.finfo(float).eps) * max(
            abs(float(self.domain.a)), abs(float(self.domain.b)), 1.0)
        pv = _np.array([
            1.0 if (r.size and _np.min(_np.abs(r - t)) <= htol) else 0.0
            for t in bps])
        # Points inside an identically-equal FUN are equal too.
        for k, c in enumerate(consts):
            if c == 1.0:
                pv[k] = 1.0
                pv[k + 1] = 1.0

        # merge (MATLAB @chebfun/merge.m): drop an interior breakpoint when
        # its pointValue matches the constant on at least one side.
        keep_a, keep_c, keep_pv = [float(bps[0])], [consts[0]], [pv[0]]
        for k in range(1, len(consts)):
            same = consts[k] == keep_c[-1]
            if same and pv[k] == consts[k]:
                continue
            keep_a.append(float(bps[k]))
            keep_c.append(consts[k])
            keep_pv.append(pv[k])
        keep_a.append(float(bps[-1]))
        keep_pv.append(pv[-1])

        funs = [_Piece.from_coeffs(
            jnp.asarray([c], dtype=jnp.float64), keep_a[k], keep_a[k + 1])
            for k, c in enumerate(keep_c)]
        out = Chebfun(funs=funs, domain=Domain(tuple(keep_a)))
        return out.set_point_values(
            jnp.asarray(_np.asarray(keep_pv), dtype=jnp.float64))

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
        out = self._sign_core()
        # MATLAB @chebfun/sign.m: the point values of sign(f) are
        # sign(f(breakpoints)), so a root that becomes a breakpoint has
        # value 0 there (not a one-sided limit).
        bps = jnp.asarray([float(v) for v in out.domain.breakpoints])
        try:
            pv = jnp.sign(jnp.real(self(bps)))
            object.__setattr__(out, "_point_values", pv)
        except Exception:
            out = self._propagate_point_values(out, jnp.sign)
        return out

    def _sign_core(self) -> Chebfun:
        """Root-splitting ``sign(f)`` without pointValues propagation."""
        roots = self.roots(nojump=True)
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
        k : int or float, default 1
            Order of differentiation.  A non-integer order computes the
            Riemann-Liouville fractional derivative (MATLAB
            ``diff(f, alpha)`` semantics, dispatching to
            :meth:`fracDiff`).

        Returns
        -------
        Chebfun
            The k-th derivative, represented piecewise.

        Provenance
        ----------
        MATLAB source : @chebfun/diff.m
        Chebfun commit: 7574c77
        """
        if float(k) != int(k):
            return self.fracDiff(float(k))
        k = int(k)
        if k == 0:
            return self
        if k > 1 and (len(self.funs) > 1 or self.deltas):
            # Iterate so each stage's jump deltas are promoted to
            # higher-order rows by the next stage (MATLAB @deltafun/diff).
            out = self
            for _ in range(int(k)):
                out = out.diff(1)
            return out
        new_funs = [piece.diff(k) for piece in self.funs]
        # (orientation is preserved below via _as_transposed)
        # Differentiating across a jump discontinuity produces a Dirac
        # delta of magnitude equal to the jump (task #9, Opus 4.8); a
        # carried delta row is promoted to its distributional derivative
        # (order + 1).
        deltas = ()
        if k == 1 and len(self.funs) > 1:
            dlist = []
            for i in range(len(self.funs) - 1):
                loc = float(self.funs[i].interval[1])
                left_value = jnp.asarray(self.funs[i](jnp.array(loc)))
                right_value = jnp.asarray(self.funs[i + 1](jnp.array(loc)))
                from chebfunjax.chebpref import ChebfunPref as _CP
                _dtol = float(_CP().deltaPrefs.deltaTol)
                if left_value.size > 1:
                    # Source diffContinuousDim/getDeltaMag handles continuous
                    # columns individually. makeDeltaFun warns and omits
                    # unsupported array-valued deltas at actual jumps.
                    if bool(jnp.any(jnp.abs(right_value - left_value) > _dtol)):
                        import warnings
                        warnings.warn(
                            "CHEBFUN:CHEBFUN:diff:diffContinuousDim:makeDeltaFun:array: "
                            "No support here for array-valued delta functions.",
                            RuntimeWarning, stacklevel=2)
                    continue
                left = complex(left_value.reshape(()))
                right = complex(right_value.reshape(()))
                jump = right - left
                # max(|re|, |im|) rather than abs(): Python's complex
                # abs (hypot) overflows for ~1e308 parts (gamma's poles,
                # approx/GammaFun); MATLAB's abs(jmp) > deltaTol simply
                # yields an Inf-magnitude delta.
                if max(abs(jump.real), abs(jump.imag)) > _dtol:
                    dlist.append((loc, jump if jump.imag else jump.real))
            deltas = tuple(dlist)
        if k == 1 and self.deltas:
            promoted = tuple(
                (loc, mag, order + 1)
                for loc, mag, order in map(_delta_row, self.deltas))
            deltas = Chebfun._merge_deltas(deltas, promoted)
        return Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain, deltas=deltas),
            self.is_transposed)

    def trigcoeffs(self, n: int | None = None, form: str = "exp"):
        """Fourier coefficients of the Chebfun (MATLAB trigcoeffs).

        With ``form='exp'`` (default) returns the complex-exponential
        coefficients ``c_k`` for modes ``k = -(N-1)/2 .. (N-1)/2``
        (ascending; even ``N`` uses ``-N/2 .. N/2-1``).  With
        ``form='cos_sin'`` returns the pair ``(a, b)`` of cosine/sine
        coefficients (MATLAB's two-output form).  For a single-piece
        periodic (trig) chebfun ``n`` defaults to ``len(f)``; piecewise
        or non-periodic chebfuns require ``n`` and are integrated
        against ``exp(-1i k omega x)`` mode by mode.

        Provenance
        ----------
        MATLAB source : @chebfun/trigcoeffs.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        import numpy as _np

        from chebfunjax.tech.trigtech import Trigtech

        num_funs = len(self.funs)
        is_trig = num_funs == 1 and isinstance(self.funs[0].tech, Trigtech)
        if n is None:
            if num_funs > 1:
                raise ValueError(
                    "trigcoeffs: input N is required for piecewise "
                    "chebfuns.")
            if not is_trig:
                raise ValueError(
                    "trigcoeffs(f, N) is allowed but not trigcoeffs(f) "
                    "for non-periodic chebfuns.")
            n = len(self)
        n = int(n)
        if n <= 0:
            return jnp.asarray([], dtype=jnp.complex128)
        a_d = float(self.domain.a)
        b_d = float(self.domain.b)
        L = b_d - a_d
        if n % 2 == 1:
            modes = _np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
        else:
            modes = _np.arange(-n // 2, n // 2)
        if is_trig:
            c_tech = _np.asarray(self.funs[0].tech.coeffs,
                                 dtype=_np.complex128)
            m = c_tech.shape[0]
            if m % 2 == 1:
                tech_modes = _np.arange(-(m - 1) // 2, (m - 1) // 2 + 1)
            else:
                tech_modes = _np.arange(-m // 2, m // 2)
            C = _np.zeros(n, dtype=_np.complex128)
            for i, k in enumerate(modes):
                j = _np.where(tech_modes == k)[0]
                if j.size:
                    C[i] = c_tech[int(j[0])]
        else:
            omega = 2.0 * _np.pi / L
            C = _np.zeros(n, dtype=_np.complex128)
            for i, k in enumerate(modes):
                Fc = chebfun(
                    lambda x, _k=k: jnp.exp(-1j * _k * omega * x),
                    domain=tuple(float(v)
                                 for v in self.domain.breakpoints))
                C[i] = complex(_np.asarray((self * Fc).sum())) / L
        change = _np.exp(-1j * modes * 2.0 * _np.pi
                         * (a_d + L / 2.0) / L)
        C = C * change
        if form == "exp":
            return jnp.asarray(C)
        if form != "cos_sin":
            raise ValueError("trigcoeffs: form must be 'exp' or 'cos_sin'.")
        if n % 2 == 1:
            z = (n - 1) // 2
            A = _np.concatenate([[C[z]], C[z - 1::-1] + C[z + 1:]])
            B = 1j * (C[z + 1:] - C[z - 1::-1])
        else:
            z = n // 2
            A = _np.concatenate([[C[z]], C[z - 1:0:-1] + C[z + 1:],
                                 [C[0]]])
            B = 1j * (C[z + 1:] - C[z - 1:0:-1])
        if self.isreal():
            A = _np.real(A)
            B = _np.real(B)
        return jnp.asarray(A), jnp.asarray(B)

    def simplify(self, tol: float | None = None) -> "Chebfun":
        """Chop negligible trailing coefficients from every piece.

        Each fun is simplified against the GLOBAL vertical scale: the
        tolerance passed to the tech is ``tol * vscale_global /
        vscale_local`` so small pieces of a large function are chopped
        relative to the whole (MATLAB @chebfun/simplify.m).

        Provenance
        ----------
        MATLAB source : @chebfun/simplify.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        base = 2.220446049250313e-16 if tol is None else float(tol)
        # Native default simplify scales each array column independently
        # across pieces; a common scale requires the separate globaltol flag.
        from chebfunjax.tech.chebtech import Chebtech1
        if self.n_columns > 1 and all(
                isinstance(piece.tech, (Chebtech1, Chebtech2))
                for piece in self.funs):
            if tol is None:
                from chebfunjax.chebpref import ChebfunPref
                base = float(ChebfunPref().techPrefs.chebfuneps)
            local = jnp.stack([piece.tech.vscale_columns for piece in self.funs])
            global_scale = jnp.max(local, axis=0)
            new_funs = [piece.with_tech(piece.tech.simplify(base * global_scale / scale))
                        for piece, scale in zip(self.funs, local)]
            out = Chebfun._as_transposed(
                Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
            out = out.set_point_values(self.point_values)
            return self._attach_deltas(out, getattr(self, "deltas", ()))
        vloc = [max(float(p.vscale), 0.0) for p in self.funs]
        vglob = max(vloc) if vloc else 0.0
        new_funs = []
        for piece, vl in zip(self.funs, vloc):
            t = piece.tech
            if hasattr(t, "simplify"):
                scaled = base * (vglob / vl) if (vl > 0 and vglob > 0) else base
                try:
                    t = t.simplify(scaled)
                except TypeError:
                    t = t.simplify()
            new_funs.append(_Piece(tech=t, interval=piece.interval))
        out = Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
        # Source simplify modifies FUNs while retaining pointValues.
        # Preserve stored discontinuous values as representation metadata.
        if self._point_values is not None:
            object.__setattr__(out, "_point_values", self._point_values)
        return self._attach_deltas(out, getattr(self, "deltas", ()))

    def cumsum(self, k: float = 1) -> Chebfun:
        """Antiderivative satisfying F(a) = 0 at the left endpoint.

        For a piecewise Chebfun, the antiderivative is computed on each piece
        and then shifted to ensure continuity across breakpoints.

        JIT-safe: yes for the per-piece computation.

        Parameters
        ----------
        k : int or float, default 1
            Order of integration.  An integer ``k`` applies ``cumsum``
            *k* times; a non-integer order computes the
            Riemann-Liouville fractional integral (MATLAB
            ``cumsum(f, alpha)`` semantics, dispatching to
            :meth:`fracInt`).

        Returns
        -------
        Chebfun
            The antiderivative.

        Provenance
        ----------
        MATLAB source : @chebfun/cumsum.m
        Chebfun commit: 7574c77
        """
        if float(k) != int(k):
            return self.fracInt(float(k))
        if int(k) != 1:
            out = self
            for _ in range(int(k)):
                out = out.cumsum()
            return out
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
        # A derivative row integrates to the next-lower order delta.
        # MATLAB source: @deltafun/cumsum.m (deltas -> jumps in the funPart).
        delta_by_loc = {}
        lowered = []
        for row in self.deltas:
            loc, mag, order = _delta_row(row)
            if order == 0:
                delta_by_loc[loc] = delta_by_loc.get(loc, 0.0) + mag
            else:
                lowered.append((loc, mag) if order == 1
                               else (loc, mag, order - 1))
        # An order-0 delta INSIDE a piece must become a breakpoint so the
        # Heaviside step can be represented (MATLAB @deltafun/cumsum.m).
        base = self
        if delta_by_loc:
            _bps = [float(v) for v in self.domain.breakpoints]
            _extra = [loc for loc in delta_by_loc
                      if _bps[0] < loc < _bps[-1]
                      and min(abs(loc - t) for t in _bps)
                      > 1e-10 * (abs(loc) + 1.0)]
            if _extra:
                base = self._restrict_breaks(sorted(set(_bps) | set(_extra)))
        new_pieces = []
        offset = None
        for piece in base.funs:
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
            Chebfun(funs=new_pieces, domain=base.domain,
                    deltas=tuple(lowered)), self.is_transposed)

    def sum(self, *args, dim=None):
        r"""Definite integral, subdomain integral, or sum across columns.

        With no arguments, sum the definite integrals of all pieces.
        ``sum(a, b)`` or ``sum([a, b])`` integrates between numeric or
        Chebfun limits; ``sum(dim)``/``sum(dim=dim)`` reduces dimension 1 or 2.
        Numeric row results have shape ``(n, 1)``; column results retain
        Python's one-dimensional vector convention.

        JIT-safe: yes.

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebfun/sum.m
        Chebfun commit: 7574c77
        """
        if args or dim is not None:
            from .summation import sum_dispatch
            return sum_dispatch(self, args, dim)
        total = jnp.float64(0.0)
        for piece in self.funs:
            total = total + piece.sum()
        # Dirac deltas contribute their magnitude to the integral (#9);
        # derivative rows integrate to zero over the whole domain.
        for row in getattr(self, "deltas", ()):
            _loc, _mag, _order = _delta_row(row)
            if _order == 0:
                total = total + jnp.float64(_mag)
        if self.is_transposed and self.n_columns > 1:
            return jnp.reshape(total, (-1, 1))
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

    def norm(self, p: float | str | None = None, *, return_location: bool = False):
        """Lp norm over the domain.

        Parameters
        ----------
        p : float or str, optional
            Default (or "fro") is the Frobenius norm for array-valued
            functions and the L2 norm for scalar functions.
            - ``p=2``: L2 norm for scalar functions; spectral norm for arrays.
            - ``p=jnp.inf``: scalar L-infinity norm, or for array-valued
              Chebfuns ``max_x sum_j |f_j(x)|`` (MATLAB matrix infinity norm).
            - Other p: scalar integral or array maximum row power sum.
        return_location : bool, optional
            Request MATLAB's second output, with source arity errors.

        Returns
        -------
        jax.Array (scalar), or (value, location) when return_location=True.
        Array1-norm locations are one-based column indices.

        Provenance
        ----------
        MATLAB source : @chebfun/norm.m
        Chebfun commit: 7574c77
        """
        # Dirac deltas (@deltafun semantics): the 1-norm adds the total
        # delta mass; every other norm of a genuine delta is infinite.
        _ds = getattr(self, "deltas", ())
        if _ds:
            import numpy as _np
            rows = [_delta_row(r) for r in _ds]
            base = Chebfun(funs=self.funs, domain=self.domain)
            if p == 1 and all(o == 0 for _l, _m, o in rows):
                mags = _np.asarray([m for _l, m, _o in rows], dtype=float)
                return jnp.asarray(float(base.norm(1))
                                   + float(_np.sum(_np.abs(mags))))
            return jnp.asarray(_np.inf)

        from .norms import continuous_norm
        return continuous_norm(self, p, return_location=return_location)

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

    def roots(self, complex_roots: bool = False,
              all_roots: bool = False,
              nojump: bool = False,
              nozerofun: bool = False) -> jax.Array:
        """All roots of the Chebfun in its domain.

        With ``complex_roots=True`` returns the complex roots of the
        Chebyshev series of each piece pruned to the region where the
        chebfun has some accuracy, like MATLAB's ``roots(f,
        'complex')``: a root is kept when its Bernstein-ellipse
        parameter ``|r + sqrt(r^2-1)|`` (symmetrized to >= 1) is at
        most ``sqrt(eps)^(-1/n)`` with ``n`` the piece length
        (chebtech/roots.m).  With ``all_roots=True`` no pruning is
        done, like MATLAB's ``roots(f, 'all')``.

        Collects roots from each piece, sorts them, and deduplicates roots
        that are very close to each other (e.g. a root at a breakpoint may
        be found independently by two adjacent pieces).

        NOT JIT-safe (variable output size, eigenvalue computation).

        Parameters
        ----------
        complex_roots, all_roots : bool, optional
            MATLAB's ``'complex'`` and ``'all'`` flags (see above).
        nojump : bool, optional
            MATLAB's ``'nojump'``: suppress the roots reported at
            breakpoints where the Chebfun changes sign through a jump
            discontinuity.
        nozerofun : bool, optional
            MATLAB's ``'nozerofun'``: suppress the single root reported at
            the midpoint of any piece on which the Chebfun is identically
            zero.

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

        if complex_roots or all_roots:
            croots = []
            for piece in self.funs:
                a, b = piece.interval
                if piece.tech.coeffs.shape[0] < 2:
                    continue
                # full chebtech machinery: 'complex' recurses with
                # per-leaf Bernstein-ellipse pruning; 'all' disables
                # recursion so a degree-n piece yields exactly n roots
                # (@chebfun/roots.m flag table)
                t = _np.asarray(piece.tech.roots(
                    complex_roots=(complex_roots and not all_roots),
                    all_roots=all_roots,
                    recurse=not all_roots))
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
                    pc = _np.asarray(piece.tech.coeffs)
                    if pc.ndim == 2 and j < pc.shape[1]:
                        if nozerofun and bool(jnp.all(
                                jnp.asarray(pc[:, j]) == 0)):
                            # 'nozerofun': an identically-zero column
                            # sends the subdivision rootfinder into an
                            # every-point-is-a-root recursion (observed
                            # as an XLA compile stall) — and it must be
                            # skipped BEFORE the tech-level call, which
                            # otherwise root-finds every column.
                            continue
                        ct = type(piece.tech)(
                            coeffs=jnp.asarray(pc[:, j]))
                        a_, b_ = piece.interval
                        t_r = _np.asarray(ct.roots())
                        col = a_ + (b_ - a_) * (t_r + 1.0) / 2.0
                    else:
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
            if nozerofun:
                # @chebfun/roots.m 'nozerofun': a FUN that is identically
                # zero contributes no midpoint root.
                pc = _np.asarray(piece.tech.coeffs)
                if pc.size and bool(jnp.all(jnp.asarray(pc) == 0)):
                    continue
            r = piece.roots()
            if r.shape[0] > 0:
                all_roots.append(r)
        combined = (_np.sort(_np.concatenate(
            [_np.asarray(r) for r in all_roots]))
            if all_roots else _np.zeros(0))

        # Jump roots (@chebfun/roots.m, jumpRoot default on): an
        # interior breakpoint is a root when the one-sided values
        # rval(fun_k) * lval(fun_{k+1}) <= 0 (sign change through a
        # jump; a pole flip gives -inf and qualifies), unless it is
        # already within tolerance of a computed root.
        if not nojump and len(self.funs) > 1:
            htol = 100 * _np.finfo(_np.float64).eps * max(
                abs(float(self.domain.a)), abs(float(self.domain.b)),
                1.0)
            jumps = []
            with _np.errstate(all="ignore"):
                for k in range(len(self.funs) - 1):
                    bp = float(self.funs[k].interval[1])
                    lv = float(_np.real(_np.asarray(
                        self.funs[k](jnp.asarray(bp)))))
                    rv = float(_np.real(_np.asarray(
                        self.funs[k + 1](jnp.asarray(bp)))))
                    if _np.isnan(lv) or _np.isnan(rv):
                        continue
                    if lv * rv <= 0 or (_np.isinf(lv) and
                                        _np.isinf(rv) and
                                        _np.sign(lv) != _np.sign(rv)):
                        if (combined.size == 0
                                or _np.min(_np.abs(combined - bp))
                                > htol):
                            jumps.append(bp)
            if jumps:
                combined = _np.sort(_np.concatenate(
                    [combined, _np.asarray(jumps)]))
        if combined.size == 0:
            return jnp.array([], dtype=jnp.float64)

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
            if bool(jnp.all(jnp.isnan(piece.tech.coeffs))):
                # MATLAB max/min ignore NaN: the NaN padding after a
                # chebop maxnorm blowup must not poison the extrema
                # (and rootfinding on NaN coefficients would crash).
                continue
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
        r = _np.asarray(df.roots(nojump=True))
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
        # MATLAB roots(df) with the default jump detection: a breakpoint
        # where f' changes sign through a jump (a kink of f) is a local
        # extremum too (Checkmark example: min(E_3, 'local') at alpha = 0).
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
        # MATLAB feval(diff(f, 2), x) at a breakpoint returns the
        # pointValue there, the average of the left and right limits
        # (@chebfun/getValuesAtBreakpoints).
        d2 = 0.5 * (_np.asarray(d2f(xj, "left")).real
                    + _np.asarray(d2f(xj, "right")).real)
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
        # A negative Dirac delta (or any derivative row, unbounded both
        # ways) makes the min infinite (@deltafun).
        _ds = [_delta_row(r) for r in getattr(self, "deltas", ())]
        if any(m < 0 or o > 0 for _l, m, o in _ds):
            loc = [_l for _l, m, o in _ds if m < 0 or o > 0][0]
            return (float(loc), float("-inf"))

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
        # A positive Dirac delta (or any derivative row, unbounded both
        # ways) makes the max infinite (@deltafun).
        _ds = [_delta_row(r) for r in getattr(self, "deltas", ())]
        if any(m > 0 or o > 0 for _l, m, o in _ds):
            loc = [_l for _l, m, o in _ds if m > 0 or o > 0][0]
            return (float(loc), float("+inf"))

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

    def restrict(self, a, b=None) -> Chebfun:
        """Restrict to endpoints or a breakpoint vector, preserving old breaks.

        Provenance
        ----------
        MATLAB source : @chebfun/restrict.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.restriction import restrict
        return restrict(self, a if b is None else (a, b))

    # ------------------------------------------------------------------
    # Quasimatrix linear algebra: qr, svd
    # ------------------------------------------------------------------

    def cond(self):
        """Two-norm condition number, @chebfun/cond.m (7574c77)."""
        _, singular_values, _ = self.svd()
        return singular_values[0] / singular_values[-1]

    def rank(self, tol=None):
        """Numerical rank, with the native length/vscale tolerance."""
        _, singular_values, _ = self.svd()
        if tol is None:
            largest = jnp.max(singular_values)
            spacing = jnp.nextafter(largest, jnp.inf) - largest
            tol = jnp.maximum(len(self) * spacing,
                              self.vscale * jnp.finfo(jnp.float64).eps)
        return jnp.sum(singular_values > tol)

    def normest(self):
        """Sum FUN norm estimates, including all array-valued columns.

        MATLAB @chebfun/normest delegates to each FUN; polynomial and
        trigonometric technologies use max(abs(values)), not pointValues.
        """
        from chebfunjax.tech.trigtech import Trigtech

        out = 0.0
        for piece in self.funs:
            tech = piece.tech
            if isinstance(tech, Trigtech):
                out = out + jnp.max(jnp.abs(tech.values))
            else:
                out = out + tech.normest()
        return out

    def qr(self, other_cols: list | None = None):
        """QR factorization of this Chebfun as a single column, or a quasimatrix.

        For a single Chebfun (one column) this simply normalises:
        ``Q = f / ||f||_2``, ``R = [[||f||_2]]``.

        For a quasimatrix (by passing a list of additional Chebfun columns as
        ``other_cols``), the columns ``[self] + other_cols`` are jointly
        factorised through polynomial FUN QR or the continuous Householder
        fallback [1]. Array-valued input is split into its existing columns.

        Parameters
        ----------
        other_cols : list[Chebfun] or None
            Additional columns.  If ``None`` (default), ``self`` is treated as
            a single column unless it is array-valued.

        Returns
        -------
        Q : Chebfun or Quasimatrix
            Array-valued Chebfun for collatable columns, otherwise a
            Quasimatrix, with L2-orthonormal columns on the same domain.
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
        if self.is_transposed:
            raise ValueError("CHEBFUN:CHEBFUN:qr:transpose: "
                             "CHEBFUN QR works only for column CHEBFUN objects.")
        inputs = [self] if other_cols is None else [self] + list(other_cols)
        cols = [col for f in inputs for col in
                (f.mat2cell() if f.n_columns > 1 else [f])]
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
        U : Chebfun or Quasimatrix
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
        from chebfunjax.chebfun1d.mtimes import _columns
        inputs = [self] if other_cols is None else [self] + list(other_cols)
        cols = [col for f in inputs for col in
                _columns(f.H if f.is_transposed else f)]
        U, S, V = chebfun_svd(cols)
        return (V, S, U) if self.is_transposed else (U, S, V)

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
        # Row chebfuns (MATLAB [x.'; x.']): stacking transposed chebfuns
        # on a SHARED domain builds an array-valued row chebfun; mixing
        # a row with a column is an error (@chebfun/vertcat.m).
        trans = [bool(getattr(f, "is_transposed", False)) for f in chebfuns]
        if any(trans):
            if not all(trans):
                raise ValueError(
                    "vertcat: cannot concatenate a column chebfun with "
                    "a row chebfun.")
            cols = [f.transpose() for f in chebfuns]
            dom = cols[0].domain
            for c in cols[1:]:
                if (float(c.domain.a) != float(dom.a)
                        or float(c.domain.b) != float(dom.b)):
                    raise ValueError(
                        "vertcat: row chebfuns must share a domain.")
            arr = Chebfun.from_function(
                lambda x: jnp.stack([c(x) for c in cols], axis=-1), dom)
            return arr.transpose()
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
        import numpy as _np
        n_cols = self.n_columns
        if isinstance(idx, tuple):
            # MATLAB f(:, cols) / f(rows, :): two-dimensional selection.
            if len(idx) != 2:
                raise IndexError(
                    f"Chebfun indexing takes at most 2 subscripts, got "
                    f"{len(idx)}.")
            cont, cols = (idx[0], idx[1]) if not self.is_transposed \
                else (idx[1], idx[0])
            if not (isinstance(cont, slice)
                    and cont == slice(None, None, None)):
                raise IndexError(
                    "Chebfun: the continuous dimension must be indexed "
                    "with ':'.")
            idx = cols
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n_cols)
            cols = list(range(start, stop, step))
            if not cols:
                raise IndexError("slice results in empty selection.")
            if cols == list(range(n_cols)):
                return self
        elif isinstance(idx, (list, tuple, _np.ndarray, jax.Array)):
            cols = [int(c) + (n_cols if int(c) < 0 else 0)
                    for c in _np.asarray(idx).ravel()]
        else:
            j = int(idx)
            if j < 0:
                j = n_cols + j
            if not 0 <= j < n_cols:
                raise IndexError(
                    f"index {j} is out of bounds for a Chebfun with "
                    f"{n_cols} column(s).")
            if n_cols == 1:
                return self
            cols = j
            return Chebfun._as_transposed(
                self._columns_base().extract_columns(cols),
                self.is_transposed)
        for c in (cols if isinstance(cols, list) else [cols]):
            if not 0 <= c < n_cols:
                raise IndexError(
                    f"index {c} is out of bounds for a Chebfun with "
                    f"{n_cols} column(s).")
        return Chebfun._as_transposed(
            self._columns_base().extract_columns(cols), self.is_transposed)

    def _columns_base(self) -> "Chebfun":
        """The column-oriented view used by the column selectors."""
        return self.transpose() if self.is_transposed else self

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

        # Multi-piece: the GLOBAL least-squares polynomial (MATLAB
        # @chebfun/polyfit.m projects onto Legendre polynomials of the
        # whole domain; fitting each piece separately returns the input
        # whenever every piece already has degree <= n, e.g. |x|).
        # Legendre coefficients via exact per-piece Gauss-Legendre
        # quadrature: c_k = (2k+1)/2 * int f(x) P_k(xhat) dxhat.
        import numpy as _np

        from chebfunjax.utils.quadrature import legpts as _legpts

        a = float(self.domain.a)
        b = float(self.domain.b)
        c_leg = _np.zeros(n + 1)
        # One quadrature size for ALL pieces, and evaluate each PIECE
        # rather than the whole chebfun: evaluating ``self`` inside the
        # loop is O(pieces^2) and, with a different node count per
        # piece, compiles a fresh kernel per piece (a few hundred
        # pieces exhausted LLVM's section memory outright).
        deg_max = max(int(piece.n) for piece in self.funs)
        nq = max(4, (n + deg_max) // 2 + 2)
        xq, wq = _legpts(nq)
        xq = _np.asarray(xq, dtype=_np.float64)
        wq = _np.asarray(wq, dtype=_np.float64)
        for piece in self.funs:
            pa, pb = float(piece.interval[0]), float(piece.interval[1])
            # physical nodes on the piece; weights scaled to xhat measure
            xp = 0.5 * (pb - pa) * xq + 0.5 * (pa + pb)
            w_hat = wq * (pb - pa) / (b - a)
            fv = _np.asarray(piece(jnp.asarray(xp)), dtype=_np.float64)
            xhat = 2.0 * (xp - a) / (b - a) - 1.0
            # Legendre-Vandermonde on xhat via the three-term recurrence
            P = _np.zeros((len(xhat), n + 1))
            P[:, 0] = 1.0
            if n >= 1:
                P[:, 1] = xhat
            for k in range(1, n):
                P[:, k + 1] = ((2 * k + 1) * xhat * P[:, k]
                               - k * P[:, k - 1]) / (k + 1)
            c_leg += P.T @ (w_hat * fv)
        c_leg *= (_np.arange(n + 1) + 0.5)
        ccheb = leg2cheb(jnp.asarray(c_leg))
        new_piece = _Piece.from_coeffs(jnp.asarray(ccheb), a, b)
        return Chebfun(funs=[new_piece], domain=Domain((a, b)))

    @staticmethod
    def interp1(
        x: jax.Array,
        y: jax.Array,
        method: "str | tuple | None" = None,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Interpolant through data (x, y).

        With the default ``'poly'`` method a single global polynomial
        interpolant through the data points ``(x[j], y[j])`` is built using
        barycentric weights.  With ``'linear'`` the result is instead the
        piecewise-linear interpolant, one two-point piece per data
        interval.

        Parameters
        ----------
        x : array_like, shape (n,)
            Interpolation sites (sorted internally).
        y : array_like, shape (n,) or (n, m)
            Function values at the sites; columns are interpolated
            independently.
        method : {'poly', 'linear'} or None, optional
            Interpolation method.  ``None`` (default) means ``'poly'``.  A
            non-string value is taken as ``domain``, matching MATLAB's
            ``interp1(x, y, dom)`` syntax.
        domain : (float, float) or None
            Domain for the resulting Chebfun.  Defaults to
            ``(x[0], x[-1])``; a narrower domain restricts the result.

        Returns
        -------
        Chebfun
            The interpolant on ``domain``.

        Notes
        -----
        The polynomial method uses barycentric Lagrange interpolation with
        second-kind barycentric weights, which is numerically stable for
        any node distribution.  As in MATLAB, ``'linear'`` evaluation
        outside ``[x[0], x[-1]]`` gives NaN.  NOT JIT-safe (adaptive
        construction).

        Provenance
        ----------
        MATLAB source : @chebfun/interp1.m (interp1Poly, interp1Linear
            subfunctions)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        if method is not None and not isinstance(method, str):
            # MATLAB interp1(x, y, dom): a non-char third argument is the
            # domain, not a method name.
            method, domain = None, method
        method = "poly" if method is None else str(method).lower()
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

        if method == "linear":
            return Chebfun._interp1_linear(x, y, domain)
        if method != "poly":
            raise ValueError(f"interp1: unknown method {method!r}.")

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
    def _interp1_linear(x, y, domain) -> Chebfun:
        """Piecewise-linear interpolant through sorted data ``(x, y)``.

        Breakpoints are the union of the data sites and the requested
        domain endpoints; each interval carries a two-point (linear)
        piece.  Points outside ``[x[0], x[-1]]`` evaluate to NaN, as in
        MATLAB's built-in ``interp1``.

        Provenance
        ----------
        MATLAB source : @chebfun/interp1.m (interp1Linear subfunction)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        x_np = _np.asarray(x, dtype=float)
        y_np = _np.asarray(y, dtype=float)
        dom_vals = [float(v) for v in domain]
        # MATLAB builds on unique([dom; x]) and then restricts to dom;
        # dropping the out-of-domain sites up front is equivalent and keeps
        # every piece at its full two-point length.
        lo, hi = dom_vals[0], dom_vals[-1]
        breaks = _np.unique(_np.concatenate(
            [_np.asarray([lo, hi]), x_np[(x_np >= lo) & (x_np <= hi)]]))

        def op(t):
            tn = _np.atleast_1d(_np.asarray(t, dtype=float))
            outside = (tn < x_np[0]) | (tn > x_np[-1])
            if y_np.ndim == 2:
                cols = [_np.where(outside, _np.nan,
                                  _np.interp(tn, x_np, y_np[:, j]))
                        for j in range(y_np.shape[1])]
                return jnp.asarray(_np.stack(cols, axis=-1))
            return jnp.asarray(
                _np.where(outside, _np.nan, _np.interp(tn, x_np, y_np)))

        return Chebfun.from_function(
            op, Domain(tuple(float(v) for v in breaks)), n=2)

    @staticmethod
    def spline(
        x: jax.Array,
        y: jax.Array,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Piecewise cubic spline with not-a-knot or supplied endpoint slopes.

        Samples may be scalar, complex, or array-valued; their site axis
        is accepted in either matrix orientation. With two extra sites in
        ``y``, the first/last rows specify endpoint slopes. The domain may
        restrict or extend the data interval and include additional breaks.
        The public adapter is eager; spline arithmetic is compiled JAX.

        Provenance
        ----------
        MATLAB source : @chebfun/spline.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils._spline import spline_coefficients, spline_evaluate

        x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(y)
        n = x.size
        if n < 2 or not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError("spline requires at least two finite sites")
        if y.ndim == 0 or y.ndim > 2:
            raise ValueError("spline samples must be a vector or matrix")
        if y.ndim == 2:
            # Source first accepts the last axis as the site dimension,
            # then forgives transposed input. Internally use sites first.
            if y.shape[1] in (n, n + 2):
                y = y.T
            if y.shape[1] == 1:
                y = y[:, 0]
        if y.shape[0] not in (n, n + 2):
            raise ValueError("spline samples must match sites, optionally with two slopes")
        slopes = None
        if y.shape[0] == n + 2:
            slopes, y = y[jnp.asarray([0, n + 1])], y[1:-1]
        order = jnp.argsort(x)
        x, y = x[order], y[order]
        if not bool(jnp.all(jnp.diff(x) > 0)):
            raise ValueError("spline sites must be distinct")
        if not bool(jnp.all(jnp.isfinite(y))) or (
                slopes is not None and not bool(jnp.all(jnp.isfinite(slopes)))):
            raise ValueError("spline values and slopes must be finite")
        if domain is None:
            domain = (float(x[0]), float(x[-1]))
        requested = Domain(tuple(float(v) for v in domain))
        breaks = tuple(sorted(set(requested.breakpoints) | set(float(v) for v in x)))
        coefficients = spline_coefficients(x, y, slopes)
        result = Chebfun.from_function(
            lambda z: spline_evaluate(x, coefficients, z), Domain(breaks), n=4)
        if requested.a > float(x[0]) or requested.b < float(x[-1]):
            result = result.restrict(requested.a, requested.b)
        return result

    @staticmethod
    def pchip(
        x: jax.Array,
        y: jax.Array,
        domain: tuple[float, float] | None = None,
    ) -> Chebfun:
        """Shape-preserving cubic Hermite interpolation with JAX arithmetic.

        Real and complex samples may be vectors or matrices with one site
        axis. Complex components are interpolated separately. Requested
        domain endpoints and internal breaks are retained; each interval
        contains exactly four Chebyshev coefficients. This adapter is eager.

        Provenance
        ----------
        MATLAB source: @chebfun/pchip.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils._pchip import pchip_coefficients
        from chebfunjax.utils._spline import spline_evaluate

        x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(y)
        if x.size < 2 or not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError("pchip requires at least two finite sites")
        if y.ndim == 0 or y.ndim > 2:
            raise ValueError("pchip samples must be a vector or matrix")
        if y.ndim == 2:
            if y.shape[0] != x.size:
                y = y.T
            if y.shape[1] == 1:
                y = y[:, 0]
        if y.shape[0] != x.size or not bool(jnp.all(jnp.isfinite(y))):
            raise ValueError("pchip samples must be finite and match the sites")
        if domain is None:
            domain = (float(x[0]), float(x[-1]))
        requested = Domain(tuple(float(v) for v in domain))
        order = jnp.argsort(x)
        x, y = x[order], y[order]
        if not bool(jnp.all(jnp.diff(x) > 0)):
            raise ValueError("pchip sites must be distinct")
        breaks = tuple(sorted(set(requested.breakpoints) | set(float(v) for v in x)))
        coefficients = pchip_coefficients(x, y)
        result = Chebfun.from_function(
            lambda z: spline_evaluate(x, coefficients, z), Domain(breaks), n=4)
        if requested.a > float(x[0]) or requested.b < float(x[-1]):
            result = result.restrict(requested.a, requested.b)
        return result

    # ------------------------------------------------------------------
    # V10 — Convolution, flip
    # ------------------------------------------------------------------

    def arc_length(self, a: "float | None" = None,
                   b: "float | None" = None) -> float:
        """Arc length of the curve defined by the chebfun.

        For a real chebfun, the length of the graph
        int sqrt(1 + f'(x)^2) dx; for a complex chebfun (a path in the
        plane), int |f'(t)| dt.

        Provenance
        ----------
        MATLAB source : @chebfun/arcLength.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        fp = self.diff()
        if a is None:
            a = float(self.domain.a)
        if b is None:
            b = float(self.domain.b)
        # integrate piecewise (avoids delta functions at jumps)
        total = 0.0
        for pc in fp.funs:
            lo = max(float(pc.interval[0]), a)
            hi = min(float(pc.interval[1]), b)
            if hi <= lo:
                continue

            def integrand(t, _pc=pc):
                v = _pc(t)
                if self.isreal():
                    return jnp.sqrt(1.0 + v**2)
                return jnp.abs(v)

            g = chebfun(integrand, domain=(lo, hi))
            total += float(_np.real(_np.asarray(g.sum())))
        return total

    def new_domain(self, new_dom) -> "Chebfun":
        """Linearly remap the chebfun onto a new domain.

        With two endpoints, all breakpoints are scaled linearly; with
        one endpoint per existing breakpoint, they are replaced
        directly.

        Provenance
        ----------
        MATLAB source : @chebfun/newDomain.m
        Chebfun commit: 7574c77
        """
        from copy import copy

        old = jnp.asarray(self.domain.breakpoints, dtype=jnp.float64)
        nd = jnp.asarray(new_dom, dtype=jnp.float64).reshape(-1)
        if nd.size == old.size:
            newb = nd
        elif nd.size == 2:
            c, d = old[0], old[-1]
            a, b = nd[0], nd[1]
            newb = (b - a) * (old - c) / (d - c) + a
        else:
            raise ValueError("CHEBFUN:CHEBFUN:newDomain:numints: Inconsistent domains.")
        domain = Domain(tuple(float(v) for v in newb))
        funs = [
            _Piece(tech=pc.tech,
                   interval=(float(newb[k]), float(newb[k + 1])))
            for k, pc in enumerate(self.funs)
        ]
        # Source changes each map and the domain on a value-copy of g.
        # Coefficients, pointValues and row/column orientation remain intact.
        result = copy(self)
        object.__setattr__(result, "funs", funs)
        object.__setattr__(result, "domain", domain)
        mapped_deltas = []
        for row in self.deltas:
            location, magnitude, order = _delta_row(row)
            k = int(jnp.clip(jnp.searchsorted(old, location, side="right") - 1,
                             0, old.size - 2))
            # @deltafun/changeMap changes deltaLoc, retaining deltaMag,
            # including its derivative-order rows.
            fraction = (jnp.asarray(location) - old[k]) / (old[k + 1] - old[k])
            moved = float(newb[k] + fraction * (newb[k + 1] - newb[k]))
            mapped_deltas.append((moved, magnitude, order) if len(row) == 3
                                 else (moved, magnitude))
        object.__setattr__(result, "deltas", tuple(mapped_deltas))
        return result

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

        # Dirac deltas convolve exactly (@deltafun semantics):
        #   (f + sum a_i d_{u_i}) * (g + sum b_j d_{v_j})
        #     = f*g + sum a_i g(x-u_i) + sum b_j f(x-v_j)
        #       + sum a_i b_j d_{u_i+v_j}.
        f_deltas = tuple(getattr(self, "deltas", ()))
        g_deltas = tuple(getattr(g, "deltas", ()))
        if f_deltas or g_deltas:
            extra_bps = []
            for row in f_deltas:
                loc = _delta_row(row)[0]
                extra_bps.extend([loc + c, loc + d])
                extra_bps.extend((loc + _np.asarray(g_bps)).tolist())
            for row in g_deltas:
                loc = _delta_row(row)[0]
                extra_bps.extend([loc + a, loc + b])
                extra_bps.extend((loc + _np.asarray(f_bps)).tolist())
            extra = _np.asarray(extra_bps, dtype=_np.float64)
            extra = extra[(extra > dom_pts[0] + tol)
                          & (extra < dom_pts[-1] - tol)]
            dom_pts = _np.unique(_np.concatenate([dom_pts, extra]))
            keep = _np.concatenate([[True], _np.diff(dom_pts) > tol])
            dom_pts = dom_pts[keep]
        out_deltas: dict = {}
        for fr in f_deltas:
            lu, au, ku = _delta_row(fr)
            for gr in g_deltas:
                lv, bv, kv = _delta_row(gr)
                key = (float(lu + lv), ku + kv)
                out_deltas[key] = out_deltas.get(key, 0.0) + au * bv

        def _delta_terms(x_val: float) -> float:
            sacc = 0.0
            for row in f_deltas:
                loc, mag, kk = _delta_row(row)
                t = x_val - loc
                if c <= t <= d:
                    gk = g if kk == 0 else g.diff(kk)
                    sacc += mag * float(_np.asarray(
                        gk(jnp.float64(t))))
            for row in g_deltas:
                loc, mag, kk = _delta_row(row)
                t = x_val - loc
                if a <= t <= b:
                    fk = f if kk == 0 else f.diff(kk)
                    sacc += mag * float(_np.asarray(
                        fk(jnp.float64(t))))
            return sacc

        # Smooth part via the fast Hale-Townsend algorithm: convolve
        # each pair of smooth pieces exactly in Legendre coefficient
        # space (bndfun/conv.m), then add the delta-shift terms
        # (a_i * g(x - u_i) etc.) as shifted polynomial pieces.
        _skip_quad = (float(self.vscale) == 0.0 or float(g.vscale) == 0.0)
        contribs = []
        if not _skip_quad:
            for pf in f.funs:
                cf = _np.asarray(pf.tech.coeffs)
                for pg in g.funs:
                    cg = _np.asarray(pg.tech.coeffs)
                    contribs.extend(_fun_conv_ht(
                        cf, pf.interval, cg, pg.interval))
        # delta^(k)_u * g = mag * g^(k)(x - u): shift a copy of the
        # k-th derivative (@deltafun/conv.m).
        for row, hh in ((r, g) for r in f_deltas):
            loc, mag, kk = _delta_row(row)
            hk = hh if kk == 0 else hh.diff(kk)
            for pg in hk.funs:
                ia, ib = pg.interval
                contribs.append(((ia + loc, ib + loc),
                                 mag * _np.asarray(pg.tech.coeffs)))
        for row in g_deltas:
            loc, mag, kk = _delta_row(row)
            hk = f if kk == 0 else f.diff(kk)
            for pf in hk.funs:
                ia, ib = pf.interval
                contribs.append(((ia + loc, ib + loc),
                                 mag * _np.asarray(pf.tech.coeffs)))
        out_ab = (float(dom_pts[0]), float(dom_pts[-1]))
        if contribs:
            h = _assemble_pieces(contribs, out_dom=out_ab)
        else:
            h = Chebfun.from_function(
                lambda x: jnp.zeros_like(jnp.asarray(x)),
                Domain(out_ab))
        if out_deltas:
            h = Chebfun(funs=h.funs, domain=h.domain,
                        deltas=tuple(sorted(
                            ((loc, v) if order == 0 else (loc, v, order))
                            for (loc, order), v in out_deltas.items()
                            if v != 0.0)))
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
        from chebfunjax.fun.singfun import Singfun

        def _flip_tech(t):
            """@chebtech/flipud: T_k(-x) = (-1)^k T_k(x), exactly."""
            if type(t) is Chebtech2:
                c = t.coeffs
                sgn = (-1.0) ** jnp.arange(c.shape[0], dtype=jnp.float64)
                sgn = sgn.reshape((-1,) + (1,) * (c.ndim - 1))
                return Chebtech2.from_coeffs(c * sgn)
            if isinstance(t, Singfun):
                sp = _flip_tech(t.smoothPart)
                if sp is None:
                    return None
                return Singfun(sp, (t.exponents[1], t.exponents[0]))
            return None

        for piece in reversed(self.funs):
            pa, pb = piece.interval
            new_a = mid - pb
            new_b = mid - pa
            ft = (_flip_tech(piece.tech)
                  if type(piece) is _Piece else None)
            if ft is not None:
                new_funs.append(_Piece(tech=ft, interval=(new_a, new_b)))
                continue
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

    def get(self, prop: str, simplevel: int = 2):
        """MATLAB ``get()`` property interface.

        Cells are represented as Python lists: ``simplevel=0`` returns a
        list over columns of lists over pieces; ``simplevel=1`` a 2-D
        list indexed ``[piece][column]`` (when every column has the same
        number of pieces); ``simplevel=2`` simplifies further to a bare
        array/value where MATLAB would return a numeric matrix.  Row
        chebfuns transpose the result exactly as MATLAB does.

        Provenance
        ----------
        MATLAB source : @chebfun/get.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if prop in ("domain", "ends"):
            return tuple(float(v) for v in self.domain.breakpoints)
        if prop == "vscale":
            return float(self.vscale)
        if prop == "hscale":
            return float(max(abs(float(self.domain.breakpoints[0])),
                             abs(float(self.domain.breakpoints[-1])),
                             1.0))
        if prop == "funs":
            return list(self.funs)
        if prop == "lval":
            return self(jnp.asarray(
                float(self.domain.breakpoints[0])))
        if prop == "rval":
            return self(jnp.asarray(
                float(self.domain.breakpoints[-1])))
        if prop == "deltas":
            ds = sorted(getattr(self, "deltas", ()),
                        key=lambda d: float(d[0]))
            if not ds:
                return jnp.zeros((2, 0), dtype=jnp.float64)
            locs = jnp.asarray([float(d[0]) for d in ds])
            mags = jnp.asarray([float(d[1]) for d in ds])
            return jnp.stack([locs, mags])
        if prop in ("coeffs", "values", "points", "exps", "exponents",
                    "vscale-local", "lval-local", "rval-local"):
            return self._get_local(prop, simplevel)
        raise ValueError(
            f"CHEBFUN:CHEBFUN:get:badProp -- '{prop}' is not a "
            "recognized CHEBFUN property name.")

    def _get_local(self, prop: str, simplevel: int):
        """Per-piece properties with MATLAB get.m's cell simplification
        rules (see :meth:`get`)."""
        ncols = self.n_columns
        exps_like = prop in ("exps", "exponents")

        def leaf(piece, col):
            tech = piece.tech
            if exps_like:
                e = getattr(tech, "exponents", None)
                if e is None:
                    e = getattr(piece, "exponents", (0.0, 0.0))
                return jnp.asarray(
                    [[float(e[0]), float(e[1])]], dtype=jnp.float64)
            if prop == "vscale-local":
                v = jnp.max(jnp.abs(jnp.asarray(tech.values)))
                return jnp.asarray([[float(v)]])
            if prop in ("lval-local", "rval-local"):
                vals = jnp.asarray(tech.values)
                if vals.ndim == 1:
                    vals = vals[:, None]
                row = vals[0] if prop == "lval-local" else vals[-1]
                return jnp.asarray([[float(row[col])]])
            arr = jnp.asarray(getattr(tech, prop) if prop != "points"
                              else tech.points)
            if arr.ndim == 1:
                arr = arr[:, None]
            if prop == "points":
                return arr[:, :1]
            return arr[:, col:col + 1]

        # Level 0: list over columns of lists over pieces.
        out0 = [[leaf(p, j) for p in self.funs] for j in range(ncols)]
        if simplevel == 0:
            return out0
        # Level 1: 2-D list [piece][column] (uniform piece counts by
        # construction for a single array-valued chebfun).
        out1 = [[out0[j][k] for j in range(ncols)]
                for k in range(len(self.funs))]
        if simplevel == 1:
            return (self._transpose_cell(out1)
                    if self.is_transposed else out1)
        # Level 2: try to collapse to a numeric matrix.
        if exps_like and ncols == 1:
            out = jnp.concatenate([c[0] for c in out1], axis=0)
            return out.T if self.is_transposed else out
        if len(self.funs) == 1:
            rows = {int(c.shape[0]) for c in out1[0]}
            if len(rows) == 1:
                out = jnp.concatenate(out1[0], axis=1)
                return out.T if self.is_transposed else out
        return (self._transpose_cell(out1)
                if self.is_transposed else out1)

    @staticmethod
    def _transpose_cell(cell):
        return [[cell[k][j] for k in range(len(cell))]
                for j in range(len(cell[0]))]

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
        out = Chebfun(funs=new_funs, domain=self.domain)
        _pv = getattr(self, "_point_values", None)
        if _pv is not None and jnp.ndim(_pv) == 2:
            _sl = _pv[:, jnp.asarray(idx)]
            object.__setattr__(out, "_point_values",
                               _sl[:, 0] if single else _sl)
        return out

    def assign_columns(self, cols, g) -> "Chebfun":
        """Overwrite the 0-based columns ``cols`` with the columns of
        ``g`` (MATLAB ``assignColumns`` / ``f(:, cols) = g``).

        Parameters
        ----------
        cols : int, sequence of int, or ``':'``
            Target columns (rows, for a row Chebfun).  ``':'`` means all
            of them.  Indices beyond the current column count grow the
            Chebfun, the intervening columns being filled with zeros.
        g : Chebfun, array_like, or None
            Replacement columns.  A numeric operand is treated as an
            array-valued constant Chebfun on this Chebfun's domain.
            ``None`` deletes the selected columns.

        Returns
        -------
        Chebfun

        Raises
        ------
        ValueError
            If the number of columns of ``g`` does not match the number
            of selected columns (MATLAB
            ``CHEBFUN:CHEBFUN:assignColumns:numCols``), or if the two
            domains differ (``CHEBFUN:CHEBFUN:assignColumns:domain``).

        Notes
        -----
        Differing interior breakpoints are unified with
        :meth:`_overlap`, as MATLAB's ``overlap`` does.

        Provenance
        ----------
        MATLAB source : @chebfun/assignColumns.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.extract_columns, Chebfun.mat2cell
        """
        import numpy as _np
        n_cols = self.n_columns
        if isinstance(cols, str):
            if cols.strip() != ":":
                raise ValueError(
                    f"assign_columns: unknown column selector {cols!r}; "
                    "use ':' or explicit indices.")
            cols = list(range(n_cols))
        idx = ([int(cols)] if isinstance(cols, (int, _np.integer))
               else [int(c) for c in cols])

        if g is None:
            # Column deletion keeps the existing per-tech path.
            new_funs = [piece.with_tech(piece.tech.assign_columns(cols, None))
                        for piece in self.funs]
            out = Chebfun(funs=new_funs, domain=self.domain)
            values = self._breakpoint_values().reshape((-1, n_cols))
            keep = [k for k in range(n_cols) if k not in idx]
            values = values[:, jnp.asarray(keep, dtype=jnp.int32)]
            if len(keep) == 1:
                values = values[:, 0]
            out = out.set_point_values(values)
            return Chebfun._as_transposed(out, self.is_transposed)

        if not isinstance(g, Chebfun):
            # Numeric operand: an array-valued constant on f's domain.
            vals = _np.atleast_1d(_np.asarray(g, dtype=float)).ravel()
            if len(vals) != len(idx):
                raise ValueError(
                    "assign_columns: subscripted assignment dimension "
                    f"mismatch ({len(idx)} columns selected, {len(vals)} "
                    "values given).")
            cvals = jnp.asarray(vals, dtype=jnp.float64)

            def _const_op(x, _v=cvals):
                xa = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
                if _v.shape[0] == 1:
                    return jnp.full(xa.shape, _v[0], dtype=jnp.float64)
                return jnp.broadcast_to(_v, xa.shape + (_v.shape[0],))

            # MATLAB transposes the numeric operand to match f's orientation.
            g = Chebfun._as_transposed(
                Chebfun.from_function(_const_op, self.domain, n=1),
                self.is_transposed)

        if self.is_transposed != g.is_transposed:
            raise ValueError(
                "assign_columns: subscripted assignment dimension mismatch "
                "(operands have different orientations).")
        if len(idx) != g.n_columns:
            raise ValueError(
                "assign_columns: subscripted assignment dimension mismatch "
                f"({len(idx)} columns selected, {g.n_columns} supplied).")
        htol = 1e-14 * max(abs(float(self.domain.a)),
                           abs(float(self.domain.b)), 1.0)
        if (abs(float(self.domain.a) - float(g.domain.a)) > htol
                or abs(float(self.domain.b) - float(g.domain.b)) > htol):
            raise ValueError(
                "assign_columns: inconsistent domains; "
                "domain(f) != domain(g).")

        # Source's full ordered assignment returns g, including its breaks
        # and stored pointValues, before overlap or coefficient arithmetic.
        if idx == list(range(n_cols)):
            return g
        base = self
        target = max(idx) + 1
        if target > n_cols:
            # MATLAB pads f with zero columns before assigning.
            grown = []
            for piece in base.funs:
                t = piece.tech
                c = t.coeffs if t.coeffs.ndim == 2 else t.coeffs[:, None]
                pad = jnp.zeros((c.shape[0], target - n_cols), dtype=c.dtype)
                grown.append(piece.with_tech(self._tech_with_coeffs(
                    t, jnp.concatenate([c, pad], axis=1))))
            values = base._breakpoint_values().reshape((-1, n_cols))
            values = jnp.concatenate(
                [values, jnp.zeros((values.shape[0], target - n_cols),
                                   dtype=values.dtype)], axis=1)
            base = Chebfun(funs=grown, domain=base.domain).set_point_values(values)
            base = Chebfun._as_transposed(base, self.is_transposed)

        f2, g2 = Chebfun._overlap(base, g)
        new_funs = [piece.with_tech(
            piece.tech.assign_columns(idx, g2.funs[k].tech))
            for k, piece in enumerate(f2.funs)]
        values = f2._breakpoint_values().reshape((-1, f2.n_columns))
        replacements = g2._breakpoint_values().reshape((-1, g2.n_columns))
        values = values.astype(jnp.result_type(values, replacements))
        # Repeated destinations have MATLAB's last-column-wins semantics.
        for source, destination in enumerate(idx):
            values = values.at[:, destination].set(replacements[:, source])
        if f2.n_columns == 1:
            values = values[:, 0]
        out = Chebfun(funs=new_funs, domain=f2.domain).set_point_values(values)
        return Chebfun._as_transposed(out, self.is_transposed)

    def mat2cell(self, sizes=None, n=None) -> list:
        """Split an array-valued Chebfun by column counts (MATLAB
        ``mat2cell(f, sizes)`` / ``mat2cell(f, 1, sizes)``).  An empty
        Chebfun returns a single-cell list holding the empty Chebfun.

        Parameters
        ----------
        sizes : sequence of int or None, optional
            Component counts, which must sum to the number of columns
            (rows, for a row Chebfun).  ``None`` splits into single
            components.  In the three-argument MATLAB form this is ``M``.
        n : sequence of int or None, optional
            Present only for MATLAB's three-argument
            ``mat2cell(F, M, N)`` form.  For a column Chebfun ``sizes``
            must then be the scalar 1 and ``n`` carries the counts; for a
            row Chebfun the roles are reversed.

        Returns
        -------
        list of Chebfun

        Raises
        ------
        ValueError
            If the requested sizes do not sum to the number of components
            (MATLAB ``CHEBFUN:CHEBFUN:mat2cell:size``).

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
        n_cols = base.n_columns

        def _as_sizes(v, what):
            try:
                seq = [int(t) for t in
                       (v if hasattr(v, "__len__") else (v,))]
            except (TypeError, ValueError):
                raise ValueError(
                    f"mat2cell: {what} must be numeric; got {v!r}."
                ) from None
            if any(t < 0 for t in seq):
                raise ValueError(
                    f"mat2cell: {what} must be non-negative.")
            return seq

        if n is not None:
            # MATLAB mat2cell(F, M, N): the dimension that is not split
            # must be the scalar 1.
            m_seq = _as_sizes(sizes, "M")
            n_seq = _as_sizes(n, "N")
            free, fixed = (m_seq, n_seq) if row else (n_seq, m_seq)
            if fixed != [1]:
                raise ValueError(
                    "mat2cell: input arguments, M and N, must sum to each "
                    f"dimension of the input size, [1,{n_cols}].")
            sizes = free
        elif sizes is not None:
            sizes = _as_sizes(sizes, "sizes")

        if sizes is None:
            # MATLAB mat2cell(F): split into single components (ones vector).
            sizes = [1] * n_cols
        elif sum(sizes) != n_cols:
            raise ValueError(
                "mat2cell: input arguments, M and N, must sum to each "
                f"dimension of the input size, [1,{n_cols}].")
        out = []
        j = 0
        for s in sizes:
            cols = j if s == 1 else list(range(j, j + s))
            cell = base.extract_columns(cols)
            out.append(Chebfun._as_transposed(cell, row))
            j += s
        return out

    def repmat(self, m, n=None):
        """Tile finite columns/rows with source ``(m,n)`` or ``[m,n]`` syntax.

        Columns require ``m=1`` and rows require ``n=1``. The established
        one-factor ``repmat(k)`` adapter repeats the finite dimension.
        Coefficients, breakpoint values and orientation are retained.

        Provenance
        ----------
        MATLAB source: @chebfun/repmat.m, horzcat.m, vertcat.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if n is None:
            shape = jnp.asarray(m)
            if shape.ndim == 0:
                m, n = (m, 1) if self.is_transposed else (1, m)
            elif shape.size == 2 and shape.ndim <= 2:
                m, n = shape.reshape(-1)
            else:
                raise ValueError("repmat requires (m,n), [m,n], or one repeat count")
        factors = []
        for factor in (m, n):
            value = jnp.asarray(factor)
            if (value.ndim != 0 or jnp.iscomplexobj(value)
                    or not bool(jnp.isfinite(value)) or float(value) != int(value)):
                raise ValueError("repmat factors must be integer scalars")
            factors.append(int(value))
        m, n = factors
        if self.is_transposed:
            if n != 1:
                raise ValueError("Use repmat(f,m,1) to tile row Chebfuns")
            repeats = m
        else:
            if m != 1:
                raise ValueError("Use repmat(f,1,n) to tile column Chebfuns")
            repeats = n
        if repeats <= 0:
            return jnp.empty((0, 0))
        if self.isempty():
            return self
        # Source vertcat promotes multirow inputs through num2cell into
        # a ChebMatrix, even when only one copy was requested.
        if self.is_transposed and self.n_columns > 1:
            from chebfunjax.operators.chebmatrix import ChebMatrix
            rows = [self.extract_columns(i).transpose()
                    for i in range(self.n_columns)]
            return ChebMatrix([[row] for _ in range(repeats) for row in rows],
                              domain=self.domain)
        if repeats == 1:
            return self
        if self.deltas or any(hasattr(piece.tech, "exponents") for piece in self.funs):
            # Source horzcat keeps singular and delta functions as distinct
            # quasimatrix columns instead of discarding their representations.
            if self.is_transposed:
                raise NotImplementedError("repmat of singular/delta rows needs a row quasimatrix")
            from chebfunjax.chebfun1d.linalg import Quasimatrix
            return Quasimatrix([self] * repeats, self.domain)
        new_funs = []
        for piece in self.funs:
            tech = piece.tech
            coeffs = tech.coeffs if tech.coeffs.ndim == 2 else tech.coeffs[:, None]
            new_funs.append(piece.with_tech(
                self._tech_with_coeffs(tech, jnp.tile(coeffs, (1, repeats)))))
        out = Chebfun(funs=new_funs, domain=self.domain)
        values = self.point_values
        if values.ndim == 1:
            values = values[:, None]
        out = out.set_point_values(jnp.tile(values, (1, repeats)))
        return Chebfun._as_transposed(out, self.is_transposed)

    # ------------------------------------------------------------------
    # V11 — Special functions: Bessel, Airy, elliptic, erf family
    # ------------------------------------------------------------------

    def besselj(self, nu: float, scale: int = 0) -> Chebfun:
        r"""Bessel function of the first kind :math:`J_\nu(f(x))`.

        Parameters
        ----------
        nu : float
            Order (real).
        scale : int, default 0
            Scaling flag.  ``scale=0`` returns :math:`J_\nu(f)`;
            ``scale=1`` returns :math:`J_\nu(f)\,e^{-|\mathrm{Im}\,f|}`,
            which is bounded for large imaginary parts of ``f``.  For a
            real-valued ``f`` the two agree exactly.

        Returns
        -------
        Chebfun
            Approximation to :math:`J_\nu(f(x))` on the same domain.

        Raises
        ------
        ValueError
            If ``nu`` is not real, or ``scale`` is not 0 or 1.

        Notes
        -----
        Uses JAX series, recurrence and ODE continuation. Numerical
        qualification is bounded in order and argument (see the primitive).
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/besselj.m
        Chebfun commit: 7574c77
        """
        if isinstance(nu, complex) and nu.imag != 0.0:
            raise ValueError(
                "besselj: the first argument must be real-valued.")
        if scale not in (0, 1):
            raise ValueError("besselj: scale must be 0 or 1.")
        if scale == 1:
            unscaled = self.besselj(nu, 0)
            if self.isreal():
                # exp(-|Im f|) == 1 identically.
                return unscaled
            scl = (-self.imag().abs()).exp()
            return unscaled * scl
        from chebfunjax.utils.besselj import besselj
        return self._apply_fun(lambda x: besselj(nu, x))

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

    def airy(self, k: int = 0, scale: int = 0) -> Chebfun:
        """Compose with JAX Airy Ai, Ai prime, Bi or Bi prime (k=0..3).

        Optional scale=1 follows the source exponential scaling formulas.
        Real and complex function values retain their numeric type.

        Provenance
        ----------
        MATLAB source: @chebfun/airy.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.airy_general import airy_all
        if k not in (0, 1, 2, 3) or scale not in (0, 1):
            raise ValueError("CHEBFUN:CHEBFUN:airy:params")
        result = self._apply_fun(lambda x: airy_all(x)[k])
        if scale:
            if k in (0, 1):
                result = ((2/3)*self**1.5).exp()*result
            else:
                result = (-(2/3)*(self**1.5).real().abs()).exp()*result
        return result

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
        r = self.roots(nojump=True)
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

        r = self.roots(nojump=True)
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

    def dirac(self, order: int = 0) -> "Chebfun":
        r"""Dirac delta distribution centred at the roots of the Chebfun.

        With ``order > 0`` returns the order-th distributional
        derivative, MATLAB ``dirac(f, n) = diff(dirac(f), n)``.

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
        if order:
            if int(order) != order or order < 0:
                raise ValueError(
                    "dirac: order of the derivative must be a "
                    "non-negative integer.")
            return self.dirac().diff(int(order))
        # uses-numpy: rootfinding and evaluation use NumPy arrays
        import numpy as _np

        a = float(self.domain.a)
        b = float(self.domain.b)

        # Find all interior roots
        r_jax = self.roots(nojump=True)
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
        weights = jnp.reciprocal(jnp.abs(jnp.asarray(fpvals)))
        # MATLAB @chebfun/dirac.m assigns half-strength to endpoint
        # roots. Integration sums stored masses without further halving.
        root_values = jnp.asarray(r)
        weights = jnp.where((root_values == a) | (root_values == b),
                            0.5 * weights, weights)

        # Carry the deltas on the UNIFIED ``deltas`` field (read by sum,
        # cumsum, arithmetic, and conv).  The legacy ``_delta_locs``/
        # ``_delta_weights`` attributes are kept for back-compat readers.
        result = Chebfun(funs=result.funs, domain=result.domain,
                         deltas=tuple((float(loc), float(w))
                                      for loc, w in zip(r, weights)))
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
        MATLAB source : @chebfun/innerProduct.m; @adchebfun/adchebfun.m (innerProduct)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.inner, Chebfun.norm
        """
        # MATLAB dispatch selects @adchebfun when either operand is AD.
        from chebfunjax.autodiff.adchebfun import ADChebfun
        if isinstance(other, ADChebfun):
            return other.__rmul__(self).sum()
        return self.inner(other)

    def ellipj(self, m: float, tol=None) -> tuple[Chebfun, Chebfun, Chebfun]:
        """Jacobi sn, cn, dn with a numeric tolerance or ChebfunPref.

        A numeric tolerance controls AGM stopping only. A preference object
        also controls the approximation tolerance of the output pieces.

        Provenance
        ----------
        MATLAB source : @chebfun/ellipj.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.ellipj import _compose_ellipj

        return _compose_ellipj(self, m, tol)

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

    def erfcx(self) -> Chebfun:
        r"""Scaled complementary error function
        :math:`\mathrm{erfcx}(f) = e^{f^2}\,\mathrm{erfc}(f)`.

        Returns
        -------
        Chebfun

        Notes
        -----
        Uses ``jax.scipy.special.erfcx``, which is accurate for large
        arguments where ``exp(x**2)*erfc(x)`` would overflow.  NOT
        JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/erfcx.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.erf, Chebfun.erfc, Chebfun.erfcinv
        """
        return self._apply_fun(jax.scipy.special.erfcx)

    def erfcinv(self) -> Chebfun:
        r"""Inverse complementary error function
        :math:`\mathrm{erfc}^{-1}(f)`.

        Satisfies ``f = erfc(erfcinv(f))`` for ``0 <= f <= 2``; computed
        as ``erfinv(1 - f)``.

        Returns
        -------
        Chebfun

        Raises
        ------
        ValueError
            If the Chebfun is not real-valued (MATLAB raises
            ``CHEBFUN:CHEBFUN:erfcinv:notreal``).

        Notes
        -----
        NOT JIT-safe (adaptive construction).

        Provenance
        ----------
        MATLAB source : @chebfun/erfcinv.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        Chebfun.erfinv, Chebfun.erfcx
        """
        if not self.isreal():
            raise ValueError("erfcinv: input must be real.")
        return self._apply_fun(
            lambda x: jax.scipy.special.erfinv(1.0 - x))

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
        """True if any coefficient of any piece, or any explicit
        ``pointValues`` entry, is NaN.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isnan.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import Singfun
        override = getattr(self, "_point_values", None)
        if override is not None and bool(jnp.any(jnp.isnan(override))):
            return True
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
        # Source columnIsfinite checks stored breakpoint values first,
        # including NaN and isolated infinite values.
        if bool(jnp.any(~jnp.isfinite(self.point_values))):
            return False
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
        """True if stored breakpoint values, pieces and impulses are real.

        Stored array dtypes determine realness, except for Fourier pieces,
        whose realness flag describes the represented function.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebfun/isreal.m, @deltafun/isreal.m
        Chebfun commit: 7574c77
        """
        if self._point_values is not None and jnp.iscomplexobj(self._point_values):
            return False
        if any(jnp.iscomplexobj(row[1]) for row in self.deltas):
            return False
        for piece in self.funs:
            # A real trigfun stores complex FOURIER coefficients; MATLAB
            # isreal checks the fun's realness, recorded in is_real.
            if getattr(piece.tech, "is_real", False):
                continue
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
        from chebfunjax.fun.singfun import Singfun

        new_funs = [
            p.with_tech(
                p.tech.imag() if isinstance(p.tech, Singfun)
                else Chebtech2.from_coeffs(jnp.imag(p.tech.coeffs))
                if isinstance(p.tech, Chebtech2)
                # Fourier coefficients of a real function are
                # conjugate-symmetric, not real — go through values.
                else type(p.tech).from_values(jnp.imag(p.tech.values))
            )
            for p in self.funs
        ]
        result = Chebfun(funs=new_funs, domain=self.domain)
        result = Chebfun._as_transposed(result, self.is_transposed)
        return result.set_point_values(jnp.imag(self._breakpoint_values()))

    def conj(self) -> Chebfun:
        """Complex conjugate of the Chebfun.

        Provenance
        ----------
        MATLAB source : @chebfun/conj.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.fun.singfun import Singfun

        new_funs = [
            p.with_tech(
                p.tech.conj() if isinstance(p.tech, Singfun) else
                Chebtech2.from_coeffs(jnp.conj(p.tech.coeffs))
                if isinstance(p.tech, Chebtech2)
                # Fourier coefficients of a real function are
                # conjugate-symmetric, not real — go through values.
                else type(p.tech).from_values(jnp.conj(p.tech.values))
            )
            for p in self.funs
        ]
        out = Chebfun._as_transposed(
            Chebfun(funs=new_funs, domain=self.domain), self.is_transposed)
        return self._propagate_point_values(out, jnp.conj)

    def angle(self) -> Chebfun:
        """Phase angle atan2(imag f, real f), constructed adaptively.

        Provenance
        ----------
        MATLAB source : @chebfun/angle.m
        Chebfun commit: 7574c77
        """
        # MATLAB: angle(f) = atan2(imag(f), real(f)), which introduces
        # breakpoints where the curve crosses the negative real axis (the
        # branch cut of atan2) so that unwrap() can remove the 2*pi jumps
        # (NonsmoothFOV example: a = 2*pi + unwrap(angle(c))).
        if not jnp.iscomplexobj(self.funs[0].tech.coeffs):
            return atan2(self * 0.0, self)
        return atan2(self.imag(), self.real())

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
        roots = self.roots(nojump=True)
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

    def any(self, dim: int = 1):
        """MATLAB ``any(f, dim)``.

        ``dim=1`` (default): True if the Chebfun is non-zero anywhere on
        its domain; a per-column boolean row for array-valued input
        (MATLAB returns a 1 x m logical).  ``dim=2``: reduce ACROSS the
        columns, returning a piecewise-constant 0/1 CHEBFUN that is 1
        wherever any column is nonzero, with breakpoints at the columns'
        roots and pointValues reduced by any() (isolated common zeros
        become 0 pointValues).

        Returns
        -------
        bool or jax.Array of bool, shape (m,), or Chebfun (``dim=2``)

        Provenance
        ----------
        MATLAB source : @chebfun/any.m
        Chebfun commit: 7574c77
        """
        if dim not in (1, 2):
            raise ValueError("any: DIM input must be 1 or 2.")
        tr = self.is_transposed
        if tr:
            # MATLAB: work on the column form, mapping 1 <-> 2.
            return Chebfun._as_transposed(
                self.transpose().any(2 if dim == 1 else 1), True) \
                if dim == 1 else self.transpose().any(1)
        if dim == 2:
            return self._any_dim2()
        if self.isempty():
            return False
        # Source anyDim1 checks pointValues and each FUN's coefficients.
        # Exact nonzero tests preserve tiny values; NaNs do not count.
        def nonzero_columns(values):
            values = jnp.asarray(values)
            if values.ndim == 1:
                values = values[:, None]
            return jnp.any((values != 0) & ~jnp.isnan(values), axis=0)

        result = nonzero_columns(self.point_values)
        for piece in self.funs:
            result = result | nonzero_columns(piece.tech.coeffs)
        return result if self.n_columns > 1 else bool(result[0])

    def _any_dim2(self) -> "Chebfun":
        """any() across the columns (MATLAB @chebfun/any.m anyDim2)."""
        import numpy as _np
        if self.isempty():
            return self
        # Breakpoints at every column's roots isolate common zeros.
        try:
            # nozerofun: an identically-zero column has every point as a
            # root — the subdivision rootfinder otherwise recurses into
            # an XLA compile stall (MATLAB addBreaksAtRoots uses the
            # same flag).
            r = _np.sort(_np.unique(_np.real(_np.asarray(
                self.roots(all_roots=False, nojump=True, nozerofun=True),
                dtype=complex)).ravel()))
        except Exception:
            r = _np.zeros(0)
        a, b = float(self.domain.a), float(self.domain.b)
        r = r[(r > a + 1e-14) & (r < b - 1e-14)]
        if r.size:
            # Cluster near-identical roots (different columns report the
            # same zero to within roundoff, e.g. 0 vs 1e-17).
            htol = 1e-16 * max(abs(a), abs(b), 1.0) + 1e-16
            keep = _np.concatenate([[True], _np.diff(r) > 1e3 * htol])
            r = r[keep]
        g = self
        if r.size:
            vals = _np.asarray(self(jnp.asarray(r)))
            g = self.define_point(r, _np.zeros(r.size)) \
                if vals.ndim == 1 else self.define_point(
                    r, _np.zeros(r.size))
        # Each smooth piece maps to the constant 1 if ANY column is not
        # identically zero on it.
        new_funs = []
        for p_ in g.funs:
            c = _np.asarray(p_.tech.coeffs)
            nz = _np.max(_np.abs(c)) > 1e2 * _EPS * max(
                float(self.vscale), 1e-300)
            new_funs.append(_Piece(
                tech=Chebtech2.from_coeffs(jnp.asarray(
                    [1.0 if nz else 0.0])),
                interval=(float(p_.interval[0]), float(p_.interval[1]))))
        out = Chebfun(funs=new_funs, domain=g.domain)
        # pointValues: any() across the columns of the ORIGINAL values.
        bps = _np.asarray([float(t) for t in g.domain.breakpoints])
        pv_src = _np.atleast_2d(_np.asarray(self(jnp.asarray(bps))))
        if pv_src.shape[0] == 1 and pv_src.size == bps.size:
            pv_src = pv_src.reshape(-1, 1)
        tol = 1e2 * _EPS * max(float(self.vscale), 1e-300)
        pv = (_np.abs(pv_src) > tol).any(axis=-1).astype(float)
        out = out.set_point_values(jnp.asarray(pv))
        # merge(): drop interior breakpoints where neighbours agree and
        # the pointValue matches the shared constant.
        keep_funs = [out.funs[0]]
        keep_bps = [bps[0]]
        keep_pv = [pv[0]]
        for i in range(1, len(out.funs)):
            prev, cur = keep_funs[-1], out.funs[i]
            c_prev = float(_np.asarray(prev.tech.coeffs)[0])
            c_cur = float(_np.asarray(cur.tech.coeffs)[0])
            if c_prev == c_cur and pv[i] == c_prev:
                keep_funs[-1] = _Piece(
                    tech=prev.tech,
                    interval=(float(prev.interval[0]),
                              float(cur.interval[1])))
            else:
                keep_funs.append(cur)
                keep_bps.append(bps[i])
                keep_pv.append(pv[i])
        keep_bps.append(bps[-1])
        keep_pv.append(pv[-1])
        out = Chebfun(funs=keep_funs, domain=Domain(tuple(keep_bps)))
        return out.set_point_values(jnp.asarray(_np.asarray(keep_pv)))

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

        # Spectral coefficient-space algorithm (@chebtech/fracInt.m):
        # Legendre (or Jacobi P^(0,b)) coefficients scaled by
        # beta(k+b+1, mu)/Gamma(mu), mapped back as Jacobi P^(-mu, b+mu)
        # coefficients; the result carries the analytic endpoint
        # exponent update exps + [mu, 0] as a Singfun
        # (@singfun/fracInt.m), scaled by (diff(domain)/2)^mu
        # (@bndfun/fracInt.m).  The previous pointwise Gauss-Jacobi
        # quadrature built a SMOOTH result whose endpoint branch
        # plateaued at ~1e-8.
        import scipy.special as _sps

        from chebfunjax.fun.singfun import Singfun
        from chebfunjax.utils.transforms import cheb2jac, cheb2leg, jac2cheb

        a = float(f.domain.a)
        b = float(f.domain.b)
        piece = f.funs[0]
        tech = piece.tech
        if isinstance(tech, Singfun):
            e_l, e_r = (float(v) for v in tech.exponents)
            if e_r != 0.0:
                raise ValueError(
                    "fracInt: only functions smooth at the right "
                    "boundary are supported (right exponent must be 0).")
            sp_coeffs = _np.asarray(tech.smoothPart.coeffs, dtype=float)
        else:
            e_l = 0.0
            sp_coeffs = _np.asarray(tech.coeffs, dtype=float)

        n = sp_coeffs.shape[0]
        k = _np.arange(n, dtype=float)
        if e_l == 0.0:
            c_leg = _np.asarray(cheb2leg(jnp.asarray(sp_coeffs)))
            if mu_frac != 0.5:
                c_jac = (c_leg * _sps.beta(k + 1.0, mu_frac)
                         / _gamma(mu_frac))
                c_new = _np.asarray(jac2cheb(jnp.asarray(c_jac),
                                             -mu_frac, mu_frac))
            else:
                # Half-integral special case ([1, (18.17.45)]); divide
                # out (1+x) via the tridiagonal averaging operator.
                scl = k + 0.5
                c_scl = c_leg / (scl * _gamma(0.5))
                c_ext = _np.concatenate([c_scl, [0.0]])
                c_shift = _np.concatenate([[0.0], c_scl])
                c_sum = c_ext + c_shift
                import scipy.sparse as _spa
                e = _np.ones(n)
                D = _spa.spdiags([0.5 * e, e, 0.5 * e], [0, 1, 2],
                                 n, n).tolil()
                D[0, 0] = 1.0
                sol = _spa.linalg.spsolve(D.tocsr(), c_sum[1:])
                c_new = _np.concatenate([sol, [0.0]])
        else:
            c_jac1 = _np.asarray(cheb2jac(jnp.asarray(sp_coeffs),
                                          0.0, e_l))
            c_jac2 = (c_jac1 * _sps.beta(k + e_l + 1.0, mu_frac)
                      / _gamma(mu_frac))
            c_new = _np.asarray(jac2cheb(jnp.asarray(c_jac2),
                                         -mu_frac, e_l + mu_frac))

        scl = ((b - a) / 2.0) ** mu_frac
        new_smooth = Chebtech2.from_coeffs(
            jnp.asarray(scl * c_new, dtype=jnp.float64))
        sf = Singfun(new_smooth, (e_l + mu_frac, 0.0))
        out_piece = _Piece(tech=sf, interval=(a, b))
        return Chebfun(funs=[out_piece], domain=Domain((a, b)))

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
        # uses-numpy/scipy: robust weighted-LP formulation of the
        # continuous L1 problem.  The previous hand-rolled Watson-Newton
        # loop silently diverged beyond small degrees (BestL1 deg-100:
        # sup err 14 on a function of scale 2; Inpainting1D failed to
        # recover).  min_c sum_i w_i |f(x_i) - (Vc)_i| on a dense
        # Clenshaw-Curtis grid is the discretized L1 best approximation
        # and is solved exactly by HiGHS.
        import numpy as _np
        from numpy.polynomial import chebyshev as _C
        from scipy.optimize import linprog

        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        from chebfunjax.utils.quadrature import chebpts, chebweights

        a = float(self.domain.a)
        b = float(self.domain.b)
        if len(self.funs) == 1 and self.funs[0].n <= n + 1:
            return self

        N = max(8 * (n + 1), 400)
        s = _np.asarray(chebpts(N), dtype=_np.float64)
        w = _np.asarray(chebweights(N), dtype=_np.float64)
        x = 0.5 * (b - a) * s + 0.5 * (a + b)
        F = _np.asarray(self(jnp.asarray(x)), dtype=_np.float64)
        V = _np.cos(_np.outer(_np.arccos(_np.clip(s, -1.0, 1.0)),
                              _np.arange(n + 1)))
        A_ub = _np.vstack([_np.hstack([V, -_np.eye(N)]),
                           _np.hstack([-V, -_np.eye(N)])])
        b_ub = _np.concatenate([F, -F])
        cvec = _np.concatenate([_np.zeros(n + 1), w])
        res = linprog(cvec, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * (n + 1)
                      + [(0, None)] * N, method="highs")
        if not res.success:
            raise RuntimeError(f"polyfitL1: LP failed ({res.message})")
        coef = res.x[:n + 1]

        # Watson-Newton polish (continuous optimality): minimize
        # Phi(c) = int |f - p_c| whose gradient is -G_k = -int sign(e) T_k
        # and whose Hessian J_kj = sum_i (2/|e'(tau_i)|) T_k(tau_i)
        # T_j(tau_i) over the sign crossings tau_i is symmetric PSD.
        # Crossings are bisected to machine precision and G is computed
        # exactly from antiderivatives, so the LP grid solution is
        # polished to continuous optimality in a few Newton steps.
        def _antider_coeffs(k):
            ck = _np.zeros(k + 1)
            ck[k] = 1.0
            return _C.chebint(ck)

        _A = [_antider_coeffs(k) for k in range(n + 1)]

        sf = _np.linspace(-1.0, 1.0, max(4001, 40 * (n + 1)))
        xf = 0.5 * (b - a) * sf + 0.5 * (a + b)
        Ff = _np.asarray(self(jnp.asarray(xf)), dtype=_np.float64)

        def _crossings(cc):
            ef = Ff - _C.chebval(sf, cc)
            sgn0 = _np.sign(ef[0]) if ef[0] != 0 else 1.0
            idx = _np.nonzero(_np.diff(_np.sign(ef)) != 0)[0]
            roots = []
            for i in idx:
                lo, hi = sf[i], sf[i + 1]
                flo = ef[i]
                for _bi in range(60):
                    mid = 0.5 * (lo + hi)
                    xm = 0.5 * (b - a) * mid + 0.5 * (a + b)
                    fm = (float(self(jnp.asarray(xm)))
                          - _C.chebval(mid, cc))
                    if fm == 0.0 or hi - lo < 4e-16:
                        break
                    if _np.sign(fm) == _np.sign(flo):
                        lo, flo = mid, fm
                    else:
                        hi = mid
                roots.append(0.5 * (lo + hi))
            return sgn0, _np.asarray(roots)

        for _it in range(30):
            sgn0, tau = _crossings(coef)
            if len(tau) == 0:
                break
            # G_k = int_{-1}^{1} sign(e) T_k: alternating sum of
            # antiderivative increments over the crossing partition
            nodes = _np.concatenate([[-1.0], tau, [1.0]])
            segsign = sgn0 * (-1.0) ** _np.arange(len(nodes) - 1)
            G = _np.zeros(n + 1)
            for k in range(n + 1):
                Avals = _C.chebval(nodes, _A[k])
                G[k] = _np.sum(segsign * _np.diff(Avals))
            if _np.max(_np.abs(G)) < 1e-12:
                break
            # e'(tau) by centered finite difference
            h = 1e-7
            taum = _np.clip(tau - h, -1.0, 1.0)
            taup = _np.clip(tau + h, -1.0, 1.0)
            xm_ = 0.5 * (b - a) * taum + 0.5 * (a + b)
            xp_ = 0.5 * (b - a) * taup + 0.5 * (a + b)
            em = (_np.asarray(self(jnp.asarray(xm_)), dtype=_np.float64)
                  - _C.chebval(taum, coef))
            ep = (_np.asarray(self(jnp.asarray(xp_)), dtype=_np.float64)
                  - _C.chebval(taup, coef))
            de = (ep - em) / (taup - taum)
            de = _np.where(_np.abs(de) < 1e-300, 1e-300, de)
            Tt = _np.cos(_np.outer(
                _np.arccos(_np.clip(tau, -1.0, 1.0)),
                _np.arange(n + 1)))
            J = (Tt * (2.0 / _np.abs(de))[:, None]).T @ Tt
            try:
                dc = _np.linalg.solve(
                    J + 1e-14 * _np.eye(n + 1) * _np.trace(J), G)
            except _np.linalg.LinAlgError:
                break
            # damped step with objective check
            phi0 = _np.trapezoid(_np.abs(Ff - _C.chebval(sf, coef)), sf)
            step = 1.0
            improved = False
            for _d in range(20):
                cnew = coef + step * dc
                phi = _np.trapezoid(_np.abs(Ff - _C.chebval(sf, cnew)),
                                    sf)
                if phi <= phi0 + 1e-15:
                    coef = cnew
                    improved = True
                    break
                step /= 2
            if not improved:
                break

        def p_eval(xx):
            ss = (2.0 * (_np.asarray(xx) - a) / (b - a)) - 1.0
            return jnp.asarray(_C.chebval(ss, coef))

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

    Oppositely oriented inputs follow MATLAB: ``kron(f.T, g)`` is
    ``sum_j f_j(x) g_j(y)``, and ``kron(f, g.T)`` reverses the variables.
    The Python column/column shorthand retains ``f(x) * g(y)`` for scalar
    inputs. Factors are assembled directly, without adaptive resampling.

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

    from .mtimes import _outer
    if f.is_transposed != g.is_transposed:
        if len(f.domain.breakpoints) > 2 or len(g.domain.breakpoints) > 2:
            raise ValueError("CHEBFUN:CHEBFUN:kron:breakpts: "
                             "The two CHEBFUNs must be smooth.")
        return _outer(g, f) if f.is_transposed else _outer(f, g)
    # Preserve the established Python column/column shorthand kron(f, g).
    return _outer(g, f.T)


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

    - column Chebfun/Quasimatrix ``A`` and column ``B``: continuous QR,
      followed by ``solve(R, innerProduct(Q, B))``;
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
    from chebfunjax.chebfun1d.linalg import Quasimatrix, chebfun_qr
    from chebfunjax.chebfun1d.mtimes import _columns

    if isinstance(A, (Chebfun, Quasimatrix)) and not A.is_transposed:
        acols = A.cols if isinstance(A, Quasimatrix) else A.mat2cell()
        if not isinstance(B, (Chebfun, Quasimatrix)) or B.is_transposed:
            raise ValueError("CHEBFUN:CHEBFUN:mldivide:agree: "
                             "Matrix dimensions must agree.")
        bcols = B.cols if isinstance(B, Quasimatrix) else B.mat2cell()
        # Domain.union supplies the existing source endpoint tolerance.
        A.domain.union(B.domain)
        Q, R = chebfun_qr(list(acols))
        products = jnp.stack([jnp.stack([q.inner(b) for b in bcols])
                              for q in _columns(Q)])
        result = jnp.linalg.solve(R, products)
        return result[:, 0] if len(bcols) == 1 else result

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


_DEG2RAD = jnp.pi / 180.0

# The remaining MATLAB elementary functions (@chebfun/<name>.m, commit
# 7574c77): reciprocal/inverse trig and hyperbolic families plus the
# degree-argument variants, all thin compositions like the explicit
# sin/cos/... methods above.
def _reciprocal_inverse_trig(value, cosine=False):
    """MATLAB real ACSC/ASEC branch, retaining complex input dispatch.

    Provenance: @chebfun/acsc.m, @chebfun/asec.m, commit 7574c77,
    composing the MATLAB elementary functions. Real asin(z), z > 1,
    has negative imaginary part; JAX complex asin(z+0j) chooses the other lip.
    """
    inverse = 1.0 / value
    op = jnp.arccos if cosine else jnp.arcsin
    if jnp.iscomplexobj(value) or not bool(jnp.any(jnp.abs(inverse) > 1)):
        return op(inverse)
    angle = jnp.where(
        jnp.abs(inverse) > 1,
        jnp.sign(inverse) * (jnp.pi / 2 - 1j * jnp.arccosh(jnp.abs(inverse))),
        jnp.arcsin(inverse).astype(jnp.complex128),
    )
    return jnp.pi / 2 - angle if cosine else angle


_EXTRA_ELEMENTWISE = {
    # @chebfun/sinc is unnormalized sin(x)/x, with value1 at zero.
    "sinc": lambda x: jnp.where(x == 0, jnp.ones_like(x), jnp.sin(x)/x),
    "tan": jnp.tan,
    "sec": lambda x: 1.0 / jnp.cos(x),
    "csc": lambda x: 1.0 / jnp.sin(x),
    "cot": lambda x: 1.0 / jnp.tan(x),
    "sech": lambda x: 1.0 / jnp.cosh(x),
    "csch": lambda x: 1.0 / jnp.sinh(x),
    "coth": lambda x: 1.0 / jnp.tanh(x),
    "asinh": jnp.arcsinh,
    "acosh": jnp.arccosh,
    "atanh": _atanh_log1p_real,
    "asec": lambda x: _reciprocal_inverse_trig(x, cosine=True),
    "acsc": _reciprocal_inverse_trig,
    "acot": lambda x: jnp.arctan(1.0 / x),
    "asech": lambda x: jnp.arccosh(1.0 / x),
    "acsch": lambda x: jnp.arcsinh(1.0 / x),
    "acoth": lambda x: _atanh_log1p_real(1.0 / x),
    "sind": lambda x: jnp.sin(_DEG2RAD * x),
    "cosd": lambda x: jnp.cos(_DEG2RAD * x),
    "tand": lambda x: jnp.tan(_DEG2RAD * x),
    "secd": lambda x: 1.0 / jnp.cos(_DEG2RAD * x),
    "cscd": lambda x: 1.0 / jnp.sin(_DEG2RAD * x),
    "cotd": lambda x: 1.0 / jnp.tan(_DEG2RAD * x),
    "asind": lambda x: jnp.arcsin(x) / _DEG2RAD,
    "acosd": lambda x: jnp.arccos(x) / _DEG2RAD,
    "atand": lambda x: jnp.arctan(x) / _DEG2RAD,
    "asecd": lambda x: _reciprocal_inverse_trig(x, cosine=True) / _DEG2RAD,
    "acscd": lambda x: _reciprocal_inverse_trig(x) / _DEG2RAD,
    "acotd": lambda x: jnp.arctan(1.0 / x) / _DEG2RAD,
    "pow2": lambda x: jnp.exp2(x),
}


def _install_extra_elementwise():
    def make(name, fn):
        def method(self):
            # MATLAB's acosh/asec/acoth/... of a real argument outside
            # the real domain return COMPLEX values; NumPy-style NaNs
            # would instead poison the construction.
            def op(x):
                y = fn(x)
                if (not jnp.iscomplexobj(x)
                        and bool(jnp.any(jnp.isnan(y)))):
                    y = fn(jnp.asarray(x, dtype=jnp.complex128))
                return y
            return self._apply_fun(op)
        method.__name__ = name
        method.__doc__ = (
            f"Elementwise ``{name}`` of the Chebfun.\n\n"
            "NOT JIT-safe (adaptive construction).\n\n"
            "Provenance\n----------\n"
            f"MATLAB source : @chebfun/{name}.m\n"
            "Chebfun commit: 7574c77\n"
            "Original authors: Copyright 2017 by The University of "
            "Oxford\n    and The Chebfun Developers.\n")
        return method
    for name, fn in _EXTRA_ELEMENTWISE.items():
        if not hasattr(Chebfun, name):
            setattr(Chebfun, name, make(name, fn))


_install_extra_elementwise()


def get(f, prop: str, simplevel: int = 2):
    """MATLAB ``get()`` for a Chebfun or a quasimatrix.

    A quasimatrix is passed as a list of Chebfun columns (columns may
    have different breakpoints).  Follows @chebfun/get.m's cell
    simplification rules with Python lists as cells; see
    :meth:`Chebfun.get`.

    Provenance
    ----------
    MATLAB source : @chebfun/get.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if isinstance(f, Chebfun):
        return f.get(prop, simplevel)
    cols = list(f)
    transposed = bool(getattr(cols[0], "is_transposed", False))
    out0 = []
    for c in cols:
        base = c.transpose() if getattr(c, "is_transposed", False) \
            else c
        col_cells = base._get_local(prop, 0)
        out0.append(col_cells[0])
    if simplevel == 0:
        return out0
    counts = {len(entry) for entry in out0}
    if len(counts) > 1:
        return out0
    nfuns = counts.pop()
    out1 = [[out0[j][k] for j in range(len(cols))]
            for k in range(nfuns)]
    if simplevel == 1:
        return Chebfun._transpose_cell(out1) if transposed else out1
    exps_like = prop in ("exps", "exponents")
    if exps_like and len(cols) == 1:
        out = jnp.concatenate([row[0] for row in out1], axis=0)
        return out.T if transposed else out
    if nfuns == 1:
        rows = {int(c.shape[0]) for c in out1[0]}
        if len(rows) == 1:
            out = jnp.concatenate(out1[0], axis=1)
            return out.T if transposed else out
    return Chebfun._transpose_cell(out1) if transposed else out1


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
    endpoints with interior breaks ``'none'``.  With no ``singType``,
    ``blowup=1``/``True`` defaults to ``'pole'`` (integer pole orders,
    MATLAB "blowup poles only") and ``blowup=2`` to ``'sing'``
    (fractional detection).

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m (parseInputs, blowup flag 1 ->
        singType 'pole', flag 2 -> 'sing')
    Chebfun commit: 7574c77
    """
    if singType is None:
        if not blowup:
            default = None
        elif blowup == 2:
            default = "sing"
        else:
            default = "pole"
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


def _source_find_blowup_bounded(op, a, b, vscale):
    """Literal scalar bounded findBlowup/zoomIn called after derivative growth.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m (findBlowup, zoomIn)
    Chebfun commit: 7574c77
    Keeps endpoint caches, third-point endpoint brackets and strict reject
    test. Sampled math is JAX; scalar eps uses existing IEEE _edge_eps.
    """
    from chebfunjax.utils._binary64_grid import locator_source_grid

    def sample(points):
        rows = _edge_sample_rows(op, points)
        if rows.shape[1] != 1:
            raise ValueError("bounded singular blowup locator requires scalar samples")
        return jnp.abs(rows[:, 0])

    def first_max(values):
        index = int(jnp.argmax(jnp.where(jnp.isnan(values), -jnp.inf, values)))
        return jnp.nanmax(values), index

    def zoom(left, right, left_value, right_value, size):
        grid = locator_source_grid(left, right, size)
        vals = jnp.concatenate((jnp.asarray([left_value]), sample(grid[1:-1]),
                                jnp.asarray([right_value])))
        _, index = first_max(vals)
        if index == 0:
            return left, float(grid[2]), left_value, vals[2]
        if index == size-1:
            return float(grid[-3]), right, vals[-3], right_value
        return float(grid[index-1]), float(grid[index+1]), vals[index-1], vals[index+1]

    grid = jnp.asarray([a, b], dtype=jnp.float64)
    values = sample(grid)
    ya, yb = values[0], values[1]
    while b-a > 1e7*_edge_eps(a):
        a, b, ya, yb = zoom(a, b, ya, yb, 50)
    while b-a > 50*_edge_eps(a):
        a, b, ya, yb = zoom(a, b, ya, yb, 15)
    while b-a >= 4*_edge_eps(a):
        grid = locator_source_grid(a, b, 4)
        values = jnp.concatenate((jnp.asarray([ya]), sample(grid[1:-1]), jnp.asarray([yb])))
        if bool(values[1] > values[2]):
            b, yb = float(grid[2]), values[2]
        else:
            a, ya = float(grid[1]), values[1]
    maximum, index = first_max(values)
    if bool(maximum < 1e5*vscale):
        return None
    return float(grid[index])


def _find_blowup(op, a: float, b: float, vscale: float):
    """Locate a blow-up point of ``op`` in ``(a, b)`` by function values.

    Zooms in on the maximum of ``|op|`` on successively finer grids until
    the bracketing interval reaches a few ulps, then returns the sample
    with the largest value.  Returns ``None`` if the largest value is not
    large relative to ``vscale`` (a "fake" blow-up).

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m (findBlowup, zoomIn;
        gridSize1 = 50, gridSize234 = 15)
    Chebfun commit: 7574c77
    """
    import numpy as _np

    def _absop(x):
        with _np.errstate(all="ignore"):
            y = _np.abs(_np.asarray(op(jnp.asarray(_np.atleast_1d(x))),
                                    dtype=float))
        # MATLAB's max ignores NaN; make NaN lose the argmax.
        return _np.where(_np.isnan(y), -_np.inf, y)

    ya = float(_absop(a)[0])
    yb = float(_absop(b)[0])

    def _zoom(a, b, ya, yb, grid):
        x = _np.linspace(a, b, grid)
        y = _np.concatenate([[ya], _absop(x[1:-1]), [yb]])
        ind = int(_np.argmax(y))
        if ind == 0:
            return a, x[1], ya, y[1]
        if ind == grid - 1:
            return x[-2], b, y[-2], yb
        return x[ind - 1], x[ind + 1], y[ind - 1], y[ind + 1]

    def _eps_at(v):
        return _np.spacing(max(abs(v), _np.finfo(float).tiny))

    while (b - a) > 1e7 * _eps_at(a):
        a, b, ya, yb = _zoom(a, b, ya, yb, 50)
    while (b - a) > 50 * _eps_at(a):
        a, b, ya, yb = _zoom(a, b, ya, yb, 15)
    x = _np.linspace(a, b, 4)
    y = _np.concatenate([[ya], _absop(x[1:3]), [yb]])
    while (b - a) >= 4 * _eps_at(a):
        x = _np.linspace(a, b, 4)
        y = _np.concatenate([[ya], _absop(x[1:3]), [yb]])
        if y[1] > y[2]:
            b, yb = x[2], y[2]
        else:
            a, ya = x[1], y[1]
    maxy = float(_np.max(y))
    blow_up = float(x[int(_np.argmax(y))])
    if not (maxy > 1e5 * vscale or _np.isinf(maxy)):
        return None
    return blow_up


def _build_exps_piece(op, a: float, b: float, el, er, stl, str_,
                      turbo: bool = False, maxpow2: int = 16,
                      tech_cls=None, tol: float | None = None,
                      vscale: float = 0.0, hscale: float = 1.0,
                      extrapolate: bool = False) -> _Piece:
    """Build one Chebfun piece on ``[a, b]`` honouring endpoint exponents.

    ``op`` is the physical function on ``[a, b]``.  Each exponent is either
    given (``el``/``er``), autodetected when ``NaN``/``None``, or forced to 0
    by a ``'none'`` singType.  A ``'pole'`` hint uses the integer pole-order
    finder; otherwise the fractional singularity finder (which also recovers
    integer poles) is used.  When both exponents vanish a smooth piece is
    returned; otherwise a :class:`Singfun` piece.

    Provenance
    ----------
    MATLAB source : @classicfun/classicfun.m (constructor),
        @bndfun/bndfun.m, @mapping/mapping.m, @singfun/singfun.m
    Chebfun commit: 7574c77
    """
    import math as _m

    from chebfunjax.fun.singfun import Singfun, _find_pole_order, _find_sing_order

    def _full(t):
        t = jnp.asarray(t)
        # @bndfun/bndfun.m leaves a canonical-domain operator unmapped.
        x = t if (a, b) == (-1.0, 1.0) else b*(t+1.0)/2.0+a*(1.0-t)/2.0
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
    # @bndfun/bndfun.m rescales the whole-domain hscale on the reference
    # interval before calling the underlying smooth-tech constructor.
    hscale_ref = float(hscale) / (b - a)
    if abs(el) < 1e-14 and abs(er) < 1e-14:
        return _Piece.from_function(
            op, a, b, turbo=turbo, maxpow2=maxpow2, tol=tol,
            vscale=vscale, hscale=hscale_ref, extrapolate=extrapolate)
    sf = Singfun.from_function(
        _full, exponents=(el, er), turbo=turbo, tech_cls=tech_cls,
        maxpow2=maxpow2, tol=tol, vscale=vscale, hscale=hscale_ref,
        extrapolate=extrapolate)
    return _Piece(tech=sf, interval=(float(a), float(b)))


def _chebfun_build(
    f=None,
    *,
    domain=(-1.0, 1.0),
    n: int | None = None,
    trig: bool = False,
    eps: float | None = None,
    max_length: int | None = None,
    splitting: "bool | None" = None,
    split_length: int | None = None,
    exps: tuple[float, float] | None = None,
    blowup: "bool | int | None" = None,
    singType: "list | tuple | None" = None,
    turbo: bool = False,
    equi: bool = False,
    coeffs: bool = False,
    min_samples: int | None = None,
    trunc: int | None = None,
    periodic: bool = False,
    chebkind: "int | str | None" = None,
    doubleLength: bool = False,
    vectorize: bool = False,
    tech=None,
    extrapolate: bool = False,
    split_max_length: int | None = None,
    resampling: bool = False,
    sample_test: bool | None = None,
    refinement_function: str | Callable | None = None,
) -> Chebfun:
    """Create a Chebfun from a callable, array of coefficients, or constant.

    MATLAB flag equivalents: ``periodic=True`` is ``'periodic'`` (an
    alias of ``trig``); ``chebkind=1`` builds on Chebyshev points of the
    first kind (``'chebkind', 1``); ``doubleLength=True`` samples on
    ``2N - 1`` points where ``N`` is the length the adaptive constructor
    would choose (``'doubleLength'``); ``vectorize=True`` wraps a
    scalar-only callable so it is evaluated pointwise (``'vectorize'``);
    ``tech`` names the representation (``'chebtech1'``, ``'chebtech2'``,
    ``'trigtech'``).  A STRING ``f`` such as ``'sin(x)'`` or
    ``'exp(sin(t))'`` is parsed as a MATLAB expression of its single
    variable (``chebfun('x')``).

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
    trunc : int or None, optional
        MATLAB ``chebfun(f, 'trig', 'trunc', N)``: truncate the Fourier
        series to exactly ``N`` symmetric coefficients after (adaptive)
        construction.  Requires ``trig=True``.

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

    # Session defaults (MATLAB chebfunpref / the splitting() and blowup()
    # toggles) apply when the flags are not given explicitly.
    if splitting is None or blowup is None:
        from chebfunjax.chebpref import ChebfunPref as _CP
        _pref = _CP()
        if splitting is None:
            splitting = bool(_pref.splitting)
        if blowup is None:
            if not _pref.blowup:
                blowup = False
            else:
                blowup = 1 if str(
                    _pref.blowupPrefs.defaultSingType).lower() == "pole" \
                    else 2
    from chebfunjax.chebpref import ChebfunPref as _CP
    _sample_test = (bool(_CP().sampleTest) if sample_test is None
                    else bool(sample_test))
    if resampling:
        if (refinement_function is not None
                and refinement_function != "resampling"):
            raise ValueError(
                "resampling=True conflicts with refinement_function="
                f"{refinement_function!r}")
        refinement_function = "resampling"
    _adaptive_override = (sample_test is not None
                         or refinement_function is not None)
    # --- MATLAB flag aliases ('periodic', 'tech', 'chebkind', strings,
    #     'vectorize', 'doubleLength'); see @chebfun/chebfun.m parseInputs.
    if tech is not None:
        _tkey = (tech.__name__ if isinstance(tech, type)
                 else str(tech)).lower().lstrip("@")
        if _tkey in ("trigtech", "trig", "periodic"):
            periodic = True
        elif _tkey in ("chebtech1",):
            chebkind = 1
        elif _tkey in ("chebtech", "chebtech2"):
            chebkind = chebkind or 2
        else:
            raise ValueError(f"chebfun: unknown tech {tech!r}")
    if periodic:
        if hasattr(domain, "__len__") and len(domain) > 2:
            raise ValueError(
                "CHEBFUN:parseInputs:periodic: 'periodic' construction "
                "does not support domains with breakpoints.")
        trig = True
    if chebkind is not None:
        _ck = str(chebkind).lower()
        chebkind = 1 if _ck in ("1", "1st", "first") else 2
        if coeffs:
            raise ValueError(
                " 'coeffs' and 'chebkind' should not be specified "
                "simultaneously.")
    if isinstance(f, str):
        f = _string_op(f)
    elif isinstance(f, (list, tuple)) and f and all(
            isinstance(t, str) for t in f):
        f = [_string_op(t) for t in f]
    elif isinstance(f, (list, tuple)) and f and all(
            hasattr(t, "tech") and hasattr(t, "interval") for t in f):
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides do not apply "
                "when assembling existing pieces")
        # MATLAB chebfun(f.funs): assemble a cell array of FUNs.
        _bps = [float(f[0].interval[0])] + [float(t.interval[1]) for t in f]
        return Chebfun(funs=list(f), domain=Domain(tuple(_bps)))
    if isinstance(f, Chebfun):
        # MATLAB chebfun(g): rebuild g by sampling (its pieces merge); a
        # given domain restricts, else g's own endpoints are kept.
        _fb = [float(v) for v in f.domain.breakpoints]
        if tuple(float(v) for v in domain) == (-1.0, 1.0) and \
                (_fb[0], _fb[-1]) != (-1.0, 1.0):
            domain = (_fb[0], _fb[-1])
        from chebfunjax.tech.trigtech import Trigtech
        _dom_pts = [float(v) for v in (domain if hasattr(domain, "__len__")
                                       else (domain,))]
        if isinstance(f.funs[0].tech, Trigtech) and len(_dom_pts) > 2:
            # MATLAB chebfun(f, [a b c]) with a periodic f: the result is
            # a piecewise (chebtech) chebfun on the given breakpoints.
            _g = f

            def f(x, _g=_g):
                return _g(x)
            trig = False
        elif isinstance(f.funs[0].tech, Trigtech) and not trig:
            trig = True
    if vectorize and callable(f):
        f = _vectorize_op(f)
    elif callable(f) and not isinstance(f, Chebfun):
        f = _vector_check(f)
    if doubleLength:
        if splitting:
            raise ValueError(
                "CHEBFUN:CHEBFUN:parseInputs:doubleLengthSplitting: "
                "doubleLength not supported with splitting on.")
        if hasattr(domain, "__len__") and len(domain) > 2:
            raise ValueError(
                "CHEBFUN:CHEBFUN:parseInputs:doubleLengthBreakpoints: "
                "doubleLength not supported on domains with breakpoints.")
        _kw = dict(domain=domain, trig=trig, eps=eps,
                   max_length=max_length, exps=exps, blowup=blowup,
                   singType=singType, turbo=turbo, equi=equi,
                   min_samples=min_samples, chebkind=chebkind,
                   sample_test=sample_test,
                   refinement_function=refinement_function)
        if coeffs:
            if _adaptive_override:
                raise ValueError(
                    "sample_test/refinement_function overrides do not apply "
                    "to coefficient input")
            _c = jnp.asarray(f)
            _pad = jnp.zeros((2 * int(_c.shape[0]) - 1,) + tuple(
                _c.shape[1:]), dtype=_c.dtype)
            return chebfun(_pad.at[:_c.shape[0]].set(_c), coeffs=True,
                           **_kw)
        _g = chebfun(f, n=n, **_kw)
        _N = 2 * len(_g) - 1
        _op = f if callable(f) else (lambda x, _g=_g: _g(x))
        return chebfun(_op, n=_N, **_kw)
    if chebkind == 1 and exps is None and not blowup and (callable(f) or (
            not coeffs and hasattr(f, "__len__"))):
        from chebfunjax.tech.chebtech import Chebtech1
        _dk = [float(v) for v in (domain if hasattr(domain, "__len__")
                                  else (domain,))]
        _pieces = []
        for _a, _b in zip(_dk[:-1], _dk[1:]):
            if callable(f):
                _t = Chebtech1.from_function(
                    lambda y, _f=f, _a=_a, _b=_b:
                        _f(_a + (_b - _a) * (y + 1.0) / 2.0), n=n,
                    sample_test=_sample_test,
                    min_samples=min_samples,
                    # constructorSplit overrides the ordinary tech cap.
                    max_length=((160 if split_length is None else int(split_length))
                                if splitting else
                                (None if max_length is None else int(max_length))),
                    **({} if refinement_function is None else {
                        "refinement_function": refinement_function}))
            else:
                if _adaptive_override:
                    raise ValueError(
                        "sample_test/refinement_function overrides do not "
                        "apply to sampled Chebtech1 values")
                _t = Chebtech1.from_values(
                    jnp.asarray(f, dtype=jnp.float64))
            _pieces.append(_Piece(tech=_t, interval=(_a, _b)))
        return Chebfun(funs=_pieces, domain=Domain(tuple(_dk)))
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
    if trunc is not None and not trig:
        # MATLAB chebfun(f, 'trunc', N): build (possibly with splitting)
        # and keep the first N Chebyshev coefficients.
        _g = chebfun(f, domain=domain, n=n, eps=eps, max_length=max_length,
                     splitting=splitting, split_length=split_length,
                     exps=exps, blowup=blowup, singType=singType,
                     turbo=turbo, equi=equi, coeffs=coeffs,
                     min_samples=min_samples, sample_test=sample_test,
                     refinement_function=refinement_function)
        return _g.truncate(int(trunc))
    if len(_dv) < 2 or len(set(_dv)) < len(_dv):
        raise ValueError(
            "chebfun: domain intervals must be of positive length")

    # MATLAB chebfun({op1, op2, ...}, [d0 d1 ... dn]): a CELL ARRAY of
    # per-interval operators (@chebfun/chebfun.m parseInputs).  Each entry
    # is built on its own interval and the pieces are concatenated; any
    # 'exps' vector is split per interval the same way.
    if isinstance(f, (list, tuple)):
        _elems = list(f)
        _cellish = (
            any(callable(t) for t in _elems)
            or (len(_dv) > 2 and len(_elems) == len(_dv) - 1
                and all(isinstance(t, (int, float))
                        or (hasattr(t, "__len__") and not isinstance(t, str))
                        for t in _elems)))
        if _cellish:
            def _cell_entry(t):
                if callable(t):
                    return _vector_check(t)
                if isinstance(t, (int, float)):
                    return t
                _v = jnp.asarray(t, dtype=jnp.float64).ravel()
                if _v.shape[0] == 1:
                    return float(_v[0])
                return lambda x, _v=_v: jnp.broadcast_to(
                    _v[None, :], (jnp.shape(x)[0], _v.shape[0]))
            _elems = [_cell_entry(t) for t in _elems]
        if _cellish:
            if len(_elems) != len(_dv) - 1:
                raise ValueError(
                    "chebfun: a cell array of operators needs one entry per "
                    f"interval; got {len(_elems)} for {len(_dv) - 1} "
                    "intervals.")
            _pairs = (_parse_exps(exps, len(_elems))
                      if exps is not None else [None] * len(_elems))
            _funs = []
            for _k, _op in enumerate(_elems):
                _sub = chebfun(
                    _op, domain=(_dv[_k], _dv[_k + 1]), n=n, trig=trig,
                    eps=eps, max_length=max_length, splitting=splitting,
                    split_length=split_length,
                    exps=(None if _pairs[_k] is None else _pairs[_k]),
                    blowup=blowup, singType=singType, turbo=turbo,
                    equi=equi, coeffs=coeffs, min_samples=min_samples,
                    sample_test=sample_test,
                    refinement_function=refinement_function)
                _funs.extend(_sub.funs)
            return Chebfun(funs=_funs, domain=Domain(tuple(_dv)))

    # Endpoint singularities (MATLAB 'exps'/'blowup' flags): each interval
    # becomes a Singfun piece s(x)*(1+x)^a*(1-x)^b (Fable 5, wiring the
    # SingFun class into the chebfun factory).  Exponents are either given
    # explicitly (``exps``), autodetected (``blowup``/NaN entries), or a
    # mix.  ``singType`` hints the detector (``'pole'``/``'sing'``/``'none'``).
    # MATLAB chebfun(c, dom, 'coeffs') / chebfun(c, dom, 'trig',
    # 'coeffs'): construct from a COEFFICIENT vector (@chebfun/chebfun.m
    # parseInputs 'coeffs' flag).
    if coeffs:
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides do not apply "
                "to coefficient input")
        if callable(f):
            raise ValueError("chebfun(..., coeffs=True) requires a "
                             "coefficient array, not a callable.")
        _dc = [float(v) for v in (domain if hasattr(domain, "__len__")
                                  else (domain,))]
        if len(_dc) < 2:
            _dc = [-1.0, 1.0]
        a_c, b_c = _dc[0], _dc[-1]
        arr = jnp.asarray(f)
        if trig:
            from chebfunjax.tech.trigtech import Trigtech
            tech = Trigtech.from_coeffs(arr.astype(jnp.complex128))
        else:
            tech = Chebtech2.from_coeffs(arr)
        piece = _Piece(tech=tech, interval=(a_c, b_c))
        return Chebfun(funs=[piece], domain=Domain((a_c, b_c)))

    from chebfunjax.tech.chebtech import Chebtech1 as _CT1
    _tech_cls = _CT1 if chebkind == 1 else None
    if (exps is not None or blowup) and all(
            math.isfinite(v) for v in _dv):
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides are not yet "
                "supported for exps/blowup construction")
        # (unbounded domains carry exps through the Unbndfun branch below)
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
        stypes = _parse_singtype(singType, n_int, blowup)
        exps_given = exps is not None
        _source_edge_blowup = (bool(blowup)
                               or (singType is not None and len(singType) > 0)
                               or any(value is None or float(value) != 0.0
                                      for pair in pairs for value in pair))

        def _happy(p):
            sp = getattr(p.tech, "smoothPart", p.tech)
            return bool(getattr(sp, "ishappy", True))

        # Under splitting, cap per-piece construction length (MATLAB
        # pref.splitPrefs.splitLength) so unresolved pieces fail fast and
        # the total-length budget below is meaningful.
        # MATLAB nested refinement doubles 17, 33, 65, 129, ... and gives
        # up once the next grid would exceed maxLength (= splitLength 160
        # under splitting), so the largest grid tried is 129 = 2^7 + 1.
        _mp2 = (16 if not splitting else
                int(math.floor(math.log2(max((split_length or 160) - 1, 2)))))
        _hscale = max(abs(v) for v in dom_vals)
        _tol = None if eps is None else float(eps)
        _vscale = 0.0
        funs = []
        for j in range(n_int):
            piece = _build_exps_piece(
                f, dom_vals[j], dom_vals[j + 1],
                pairs[j][0], pairs[j][1], stypes[j][0], stypes[j][1],
                turbo=turbo, maxpow2=_mp2, tech_cls=_tech_cls,
                tol=_tol, vscale=_vscale, hscale=_hscale,
                extrapolate=bool(splitting))
            funs.append(piece)
            if _happy(piece):
                _piece_vscale = float(piece.tech.vscale)
                if math.isfinite(_piece_vscale):
                    _vscale = max(_vscale, _piece_vscale)
        # With 'splitting' on, unhappy pieces are split at detected
        # interior blow-up points (poles located by function values),
        # mirroring the sad-interval loop of @chebfun/constructor.m with
        # singDetect: each new endpoint autodetects its exponent.
        if splitting:
            SPLIT_MAX_LENGTH = (6000 if split_max_length is None
                                else int(split_max_length))
            unsplittable: set = set()
            while (any(not _happy(p) and id(p) not in unsplittable
                       for p in funs)
                   and sum(p.n for p in funs) < SPLIT_MAX_LENGTH):
                # Largest sad interval:
                widths = [(p.interval[1] - p.interval[0]
                           if (not _happy(p) and id(p) not in unsplittable)
                           else 0.0) for p in funs]
                k = max(range(len(widths)), key=widths.__getitem__)
                a_, b_ = funs[k].interval
                # Compensate the operator for the piece's KNOWN endpoint
                # exponents so boundary poles (already isolated at the
                # breakpoints) do not attract the interior edge search
                # (@fun/detectEdge.m: "Compensating for exponents").
                exps_k = tuple(float(e) for e in
                               getattr(funs[k].tech, "exponents",
                                       (0.0, 0.0)))
                if any(exps_k):
                    def comp(x, _a=a_, _b=b_, _e=exps_k):
                        xx = jnp.asarray(x)
                        return (f(xx)
                                / ((xx - _a) ** _e[0]
                                   * (_b - xx) ** _e[1]))
                else:
                    comp = f
                def _edge_ok(e, _a=a_, _b=b_):
                    return (e is not None and _a < e < _b
                            and e - _a >= 4 * math.ulp(max(abs(_a), 1e-300))
                            and _b - e >= 4 * math.ulp(max(abs(_b), 1e-300)))

                # Supplied nonzero exponents enable source pref.blowup.
                # detectEdge checks blowup inside derivative refinement using
                # the constructor's global vscale, not a median prepass.
                _htol = 1e-14 * _hscale

                def _snap(e, _a=a_, _b=b_, _h=_htol):
                    if e is None:
                        return e
                    if abs(e - _a) <= _h:
                        return _a + (_b - _a) / 100
                    if abs(_b - e) <= _h:
                        return _b - (_b - _a) / 100
                    return e
                edge = _snap(_detect_edge_matlab(
                    comp, a_, b_, vscale=_vscale, hscale=_hscale,
                    blowup=_source_edge_blowup))
                if not _edge_ok(edge):
                    edge = 0.5 * (a_ + b_)
                if not _edge_ok(edge):
                    unsplittable.add(id(funs[k]))
                    continue  # try remaining sad pieces
                el, er = pairs[k]
                nan = float("nan")
                mid_l, mid_r = (0.0, 0.0) if exps_given else (nan, nan)
                stl, str_k = stypes[k]
                left = _build_exps_piece(
                    f, a_, edge, el, mid_l, stl, str_k, turbo=turbo,
                    maxpow2=_mp2, tech_cls=_tech_cls, tol=_tol,
                    vscale=_vscale, hscale=_hscale, extrapolate=True)
                if _happy(left) and math.isfinite(float(left.tech.vscale)):
                    _vscale = max(_vscale, float(left.tech.vscale))
                right = _build_exps_piece(
                    f, edge, b_, mid_r, er, stl, str_k, turbo=turbo,
                    maxpow2=_mp2, tech_cls=_tech_cls, tol=_tol,
                    vscale=_vscale, hscale=_hscale, extrapolate=True)
                if _happy(right) and math.isfinite(float(right.tech.vscale)):
                    _vscale = max(_vscale, float(right.tech.vscale))
                funs[k:k + 1] = [left, right]
                pairs[k:k + 1] = [(el, mid_l), (mid_r, er)]
                stypes[k:k + 1] = [(stl, str_k), (stl, str_k)]
        return _finalize_bounded_singular(
            funs, dom_vals, f,
            split_length=160 if split_length is None else int(split_length),
            tol=_tol, turbo=turbo, check=str(_CP().happinessCheck),
            sample_test=_sample_test, min_samples=min_samples,
            refinement_function=refinement_function)

    # --- Preferences (task #11): eps -> chop tolerance, max_length ->
    #     maximum adaptive length (2**maxpow2 + 1). ---
    _tol = None if eps is None else float(eps)
    if max_length is None:
        _maxpow2 = 16
    else:
        _maxpow2 = max(4, int(math.floor(math.log2(max(int(max_length) - 1, 2)))))

    # --- 'equi' flag: data sampled on an equispaced grid ---
    # The values are interpreted as coming from linspace(a, b, N); a
    # Floater-Hormann rational interpolant (FUNQUI) is built and then
    # resolved adaptively as a Chebfun (MATLAB @chebfun/chebfun.m
    # 'equi' -> chebfunpref.enableFunqui -> @smoothfun funqui).
    if equi:
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides do not apply "
                "to equispaced data")
        if callable(f):
            raise ValueError(
                "chebfun(..., equi=True): the 'equi' flag requires numeric "
                "data sampled on an equispaced grid; adaptive construction "
                "from a function handle is not supported "
                "(MATLAB CHEBFUN:CHEBFUN:parseInputs:equi).")
        from chebfunjax.utils.interpolation import funqui

        _fe = jnp.asarray(f, dtype=jnp.float64)
        _de = [float(v) for v in (domain if hasattr(domain, "__len__")
                                  else (domain,))]
        if len(_de) < 2:
            _de = [-1.0, 1.0]
        a_e, b_e = _de[0], _de[-1]
        if trig:
            # Periodic equispaced data IS a trigonometric interpolant.
            from chebfunjax.tech.trigtech import Trigtech
            _tt = Trigtech.from_values(_fe)
            return Chebfun(funs=[_Piece(tech=_tt, interval=(a_e, b_e))],
                           domain=Domain((a_e, b_e)))
        if _fe.ndim == 2:
            _hs = [funqui(_fe[:, j]) for j in range(_fe.shape[1])]

            def handle(t, _hs=_hs):
                return jnp.stack([h(t) for h in _hs], axis=-1)
        else:
            handle = funqui(_fe)
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
        or (hasattr(f, "__len__") and not callable(f)
            and getattr(f, "ndim", 1) != 0 and len(f) == 0)
        or (n is not None and n == 0)
    )
    if is_empty_arg:
        return Chebfun(funs=[], domain=dom)

    if not trig and (
        isinstance(f, (int, float))
        or (hasattr(f, "__float__") and not callable(f)
            and getattr(f, "ndim", 0) == 0 and not jnp.iscomplexobj(f))
    ):
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides do not apply "
                "to scalar constant input")
        # Scalar constant
        c = float(f)
        return Chebfun.from_function(lambda x: jnp.full_like(x, c), dom, n=n)

    # Try JAX scalar (0-d array)
    try:
        arr = jnp.asarray(f)
    except Exception:
        arr = None
    if arr is not None:
        # Scalar trig inputs continue to the source-specific Trigtech branch.
        if arr.ndim == 0 and not trig:
            if _adaptive_override:
                raise ValueError(
                    "sample_test/refinement_function overrides do not apply "
                    "to scalar constant input")
            if jnp.iscomplexobj(arr):
                c = arr.astype(jnp.complex128)

                def _complex_constant(x):
                    return jnp.full_like(x, c, dtype=jnp.complex128)

                return Chebfun.from_function(_complex_constant, dom, n=n)
            c = float(arr)
            return Chebfun.from_function(lambda x: jnp.full_like(x, c), dom, n=n)
        if arr.ndim in (1, 2) and not callable(f) and not coeffs \
                and not trig:
            if _adaptive_override:
                raise ValueError(
                    "sample_test/refinement_function overrides do not apply "
                    "to sampled values")
            # MATLAB chebfun(a) with a data VECTOR (or matrix: one column
            # per function): the polynomial interpolant through the values
            # at 2nd-kind Chebyshev points (approx2/Gibbs2D builds its
            # square wave this way).
            _dt = jnp.complex128 if jnp.iscomplexobj(arr) else jnp.float64
            return Chebfun.from_values(jnp.asarray(arr, dtype=_dt), dom)

    _dom_arr = [float(v) for v in (domain if hasattr(domain, "__len__")
                                    else (domain,))]
    if any(not math.isfinite(v) for v in _dom_arr):
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides are not yet "
                "supported for unbounded construction")
        from chebfunjax.fun.unbndfun import Unbndfun

        if trig:
            raise ValueError("trig=True is not supported on unbounded domains.")
        if not callable(f):
            raise ValueError("Unbounded domains require a callable.")
        if len(_dom_arr) != 2:
            # Piecewise-unbounded (MATLAB chebfun(op, [a b ... inf])):
            # build the finite intervals through the ordinary factory and
            # each infinite end piece as an Unbndfun, then assemble.
            _n_int = len(_dom_arr) - 1
            _pairs = (_parse_exps(exps, _n_int)
                      if exps is not None else [None] * _n_int)
            _funs: list = []
            for _k in range(_n_int):
                _a, _b = _dom_arr[_k], _dom_arr[_k + 1]
                if math.isfinite(_a) and math.isfinite(_b):
                    _sub = chebfun(f, domain=(_a, _b), n=n,
                                   exps=_pairs[_k])
                    _funs.extend(_sub.funs)
                else:
                    _funs.append(Unbndfun.from_function(
                        f, domain=Domain((_a, _b)), n=n,
                        exps=_pairs[_k]))
            return Chebfun(funs=_funs,
                           domain=Domain(tuple(_dom_arr)))
        dom_u = Domain((_dom_arr[0], _dom_arr[1]))
        _exps_u = (None if exps is None
                   else (float(exps[0]), float(exps[1])))
        if splitting and n is None:
            # MATLAB @chebfun/constructor.m with splitting on an
            # unbounded domain: edges are detected in the MAPPED variable
            # y in [-1, 1] (detectEdge composes the op with the map and
            # compensates the growth exponents); the interior pieces
            # become bounded funs and only the two tails stay unbounded,
            # each then resolved to full relative precision.
            _probe = Unbndfun.from_function(f, domain=dom_u, n=None,
                                            exps=_exps_u)
            _a_u, _b_u = _dom_arr[0], _dom_arr[1]
            _el = 0.0 if _exps_u is None else _exps_u[0]
            _er = 0.0 if _exps_u is None else _exps_u[1]

            def _op_y(y, _p=_probe):
                yy = jnp.asarray(y)
                val = f(_p.forward_map(yy))
                if _el or _er:
                    val = val * ((yy + 1.0) ** _el * (1.0 - yy) ** _er)
                return val
            _eps_y = 1e-3
            _ybrk = _split_breakpoints(_op_y, -1.0 + _eps_y, 1.0 - _eps_y,
                                       _maxpow2, split_pow2=8, tol=_tol)
            _ybrk = sorted(float(t) for t in _ybrk
                           if -1.0 + _eps_y < float(t) < 1.0 - _eps_y)
            if _ybrk:
                _xbrk = [float(_probe.forward_map(jnp.asarray(t)))
                         for t in _ybrk]
                _funs: list = []
                _bps: list = []
                if math.isfinite(_a_u):
                    _bps.append(_a_u)
                    _left = _a_u
                else:
                    _funs.append(Unbndfun.from_function(
                        f, domain=Domain((_a_u, _xbrk[0])), n=None,
                        exps=(None if _exps_u is None else (_el, 0.0))))
                    _bps.extend([_a_u, _xbrk[0]])
                    _left = _xbrk[0]
                    _xbrk = _xbrk[1:]
                if math.isfinite(_b_u):
                    _inner = _xbrk
                    _right_tail = None
                else:
                    _inner = _xbrk[:-1] if _xbrk else []
                    _right_tail = _xbrk[-1] if _xbrk else _left
                _pts = [_left] + _inner + ([_right_tail]
                                          if _right_tail is not None
                                          else [_b_u])
                for _xa, _xb in zip(_pts[:-1], _pts[1:]):
                    if _xb - _xa <= 0:
                        continue
                    _sub = _construct_with_splitting(
                        f, float(_xa), float(_xb), _maxpow2, tol=_tol,
                        turbo=turbo, min_samples=min_samples,
                        split_length=split_length,
                        split_max_length=split_max_length,
                        sample_test=_sample_test,
                        refinement_function=refinement_function)
                    _funs.extend(_sub.funs)
                    _bps.extend(float(v)
                                for v in _sub.domain.breakpoints[1:])
                if _right_tail is not None:
                    _funs.append(Unbndfun.from_function(
                        f, domain=Domain((_right_tail, _b_u)), n=None,
                        exps=(None if _exps_u is None else (0.0, _er))))
                    _bps.append(_b_u)
                _bps = sorted(set(_bps))
                return Chebfun(funs=_funs, domain=Domain(tuple(_bps)))
            return Chebfun(funs=[_probe], domain=dom_u)
        fun_u = Unbndfun.from_function(f, domain=dom_u, n=n, exps=_exps_u)
        return Chebfun(funs=[fun_u], domain=dom_u)

    if trig:
        if _adaptive_override:
            raise ValueError(
                "sample_test/refinement_function overrides are not yet "
                "supported for trigonometric construction")
        from chebfunjax.tech.trigtech import Trigtech

        dom_arr = tuple(float(v) for v in domain)
        if len(dom_arr) != 2:
            raise ValueError(
                "chebfun(..., trig=True) supports a single interval only."
            )
        a, b = dom_arr

        if not callable(f) and jnp.ndim(f) >= 1 and jnp.size(f) > 1:
            # MATLAB chebfun(v, 'trig'): values on the equispaced trig
            # grid (one column per function).
            _tv = jnp.asarray(f)
            tech = Trigtech.from_values(_tv)
            piece = _Piece(tech=tech, interval=(a, b))
            return Chebfun(funs=[piece], domain=Domain((a, b)))
        if not callable(f):
            # MATLAB chebfun(c, 'trig'): a (possibly complex) constant
            # is the single zero-wavenumber Fourier coefficient.
            c0 = complex(f)
            tech = Trigtech(
                coeffs=jnp.asarray([c0], dtype=jnp.complex128),
                is_real=(c0.imag == 0.0), ishappy=True)
            piece = _Piece(tech=tech, interval=(a, b))
            return Chebfun(funs=[piece], domain=Domain((a, b)))

        def f_ref(x):
            return f(a + (b - a) * (x + 1.0) / 2.0)

        tech = Trigtech.from_function(f_ref, n=n)
        if trunc is not None:
            # MATLAB chebfun(f, 'trig', 'trunc', N): truncate the
            # Fourier series to exactly N symmetric coefficients.
            tech = Trigtech(
                coeffs=tech.trigcoeffs(int(trunc)),
                is_real=tech.is_real, ishappy=tech.ishappy)
        piece = _Piece(tech=tech, interval=(a, b))
        return Chebfun(funs=[piece], domain=Domain((a, b)))

    if callable(f):
        if splitting and n is None:
            # Source constructorSplit owns one queue and shared scale over
            # every supplied smooth interval, retaining user breakpoints.
            return _construct_with_splitting(
                f, float(dom_seq[0]), float(dom_seq[-1]), _maxpow2,
                tol=_tol, turbo=turbo, min_samples=min_samples,
                split_length=split_length, split_max_length=split_max_length,
                sample_test=_sample_test,
                refinement_function=refinement_function,
                breakpoints=tuple(float(x) for x in dom_seq))
        # MATLAB 'minSamples': the first adaptive grid has at least that
        # many points (a narrow spike missed by the 17-point grid is
        # caught by the 33-point one).
        _sp2 = (max(4, int(math.ceil(math.log2(max(int(min_samples) - 1,
                                                    2)))))
                if min_samples else 4)
        return Chebfun.from_function(f, dom, n=n, maxpow2=_maxpow2,
                                     tol=_tol, turbo=turbo,
                                     extrapolate=extrapolate,
                                     start_pow2=_sp2,
                                     sample_test=_sample_test,
                                     refinement_function=refinement_function,
                                     min_samples=min_samples,
                                     max_length=(None if max_length is None
                                                 else int(max_length)))

    raise TypeError(
        f"Cannot construct a Chebfun from f of type {type(f).__name__}. "
        f"Pass a callable (e.g. jnp.sin), a scalar, or use "
        f"Chebfun.from_coeffs / Chebfun.from_values."
    )


def _bvp_solve(method: str, odefun, bcfun, solinit, params=None,
               options=None):
    """Shared driver for :func:`bvp4c` / :func:`bvp5c` (MATLAB
    @chebfun/bvp4c.m, bvp5c.m): the initial guess is an array-valued
    Chebfun sampled on a mesh, the two-point BVP is solved by
    ``scipy.integrate.solve_bvp`` and the dense collocation solution is
    resampled into an array-valued Chebfun.  ``options`` is a dict with
    MATLAB ``odeset`` keys (``RelTol``/``AbsTol``)."""
    import numpy as _np
    from scipy.integrate import solve_bvp  # type: ignore[import]

    bp = [float(v) for v in solinit.domain.breakpoints]
    a_, b_ = bp[0], bp[-1]
    ncomp = solinit.n_columns
    n_mesh = max(64, 4 * len(solinit))
    x_mesh = _np.linspace(a_, b_, n_mesh)
    y_init = _np.atleast_2d(_np.asarray(
        solinit(jnp.asarray(x_mesh)), dtype=float))
    if y_init.shape[0] != ncomp:
        y_init = y_init.T
    p0 = None if params is None else _np.atleast_1d(
        _np.asarray(params, dtype=float))

    def _fun(x, y, *p):
        cols = []
        for i in range(x.shape[0]):
            args = (x[i], jnp.asarray(y[:, i])) + tuple(
                (float(v) for v in p[0]) if p else ())
            cols.append(_np.asarray(odefun(*args), dtype=float).ravel())
        return _np.stack(cols, axis=1)

    def _bc(ya, yb, *p):
        args = (jnp.asarray(ya), jnp.asarray(yb)) + tuple(
            (float(v) for v in p[0]) if p else ())
        return _np.asarray(bcfun(*args), dtype=float).ravel()

    opts = dict(options or {})
    tol = float(opts.get("RelTol", opts.get("rtol",
                                            1e-3 if method == "bvp4c"
                                            else 1e-6)))
    sol = solve_bvp(_fun, _bc, x_mesh, y_init, p=p0, tol=tol,
                    max_nodes=100000)
    if not sol.success:
        raise RuntimeError(f"{method}: {sol.message}")

    def _ev(x):
        xx = _np.atleast_1d(_np.asarray(x, dtype=float))
        vals = sol.sol(xx).T
        if ncomp == 1:
            vals = vals[:, 0]
        return jnp.asarray(vals.reshape(
            (_np.shape(x) + ((ncomp,) if ncomp > 1 else ()))))
    y = chebfun(_ev, domain=(a_, b_))
    if params is None:
        return y
    return y, jnp.asarray(sol.p)


def bvp4c(odefun, bcfun, solinit, params=None, options=None):
    """Solve a two-point BVP from a Chebfun initial guess (MATLAB
    ``bvp4c(odefun, bcfun, y0, [params], [opts])``); returns the
    solution as an array-valued Chebfun (and the parameters when unknown
    parameters are supplied).

    Provenance
    ----------
    MATLAB source : @chebfun/bvp4c.m
    Chebfun commit: 7574c77
    """
    return _bvp_solve("bvp4c", odefun, bcfun, solinit, params, options)


def bvp5c(odefun, bcfun, solinit, params=None, options=None):
    """Solve a two-point BVP from a Chebfun initial guess with the
    higher-accuracy defaults of MATLAB's ``bvp5c``.

    Provenance
    ----------
    MATLAB source : @chebfun/bvp5c.m
    Chebfun commit: 7574c77
    """
    return _bvp_solve("bvp5c", odefun, bcfun, solinit, params, options)


def _string_op(expr: str):
    """MATLAB ``chebfun('sin(x)')``: compile an expression string of a
    single variable (any identifier that is not a known function name;
    ``x`` when the expression has none) into a vectorized callable."""
    import re

    from chebfunjax.utils.matlab_expr import _FUNS, matlab_expression
    # A scientific literal exponent (1e-100) is not a variable name.
    names = [t for t in re.findall(r"(?<![\w.])[A-Za-z_]\w*", expr)
             if t not in _FUNS]
    names = [t for t in names if not re.fullmatch(r"\d.*", t)]
    var = names[0] if names else "x"
    op = matlab_expression(expr, (var,))

    def _f(x, _op=op):
        return jnp.asarray(_op(x)) + jnp.zeros_like(x)
    return _f


def _vector_check(f):
    """MATLAB @chebfun/vectorCheck: make an operator's output conform to
    the sample grid -- a scalar (``@(x) 1``) or a constant row
    (``@(x) [1 2 3]``) is broadcast over the points, a transposed
    array-valued output is transposed back, and an operator that cannot
    take a vector at all is evaluated pointwise."""
    # The layout is decided ONCE, from the first call with at least three
    # points (a 2 x 2 output cannot be told from its transpose), and then
    # applied consistently -- the adaptive constructor's first grid has
    # 17 points, the endpoint check only two.
    mode: dict = {}

    def _f(x):
        try:
            out = jnp.asarray(f(x))
        except Exception:
            return _vectorize_op(f)(x)
        if jnp.ndim(x) == 0:
            return out
        m = int(jnp.shape(x)[0])
        if "kind" not in mode and (m >= 3 or out.ndim == 0
                                   or (out.ndim == 1 and out.shape[0] != m)):
            if out.ndim == 0:
                mode["kind"] = "scalar"
            elif out.ndim == 1 and out.shape[0] != m:
                mode["kind"] = "row"
            elif out.ndim == 2 and out.shape[0] != m and out.shape[1] == m:
                mode["kind"] = "transposed"
            else:
                mode["kind"] = "plain"
        kind = mode.get("kind", "plain")
        if kind == "scalar" or out.ndim == 0:
            return jnp.full((m,), out)
        if kind == "row":
            return jnp.broadcast_to(out[None, :], (m, out.shape[0]))
        if kind == "transposed" and out.ndim == 2:
            return out.T
        return out
    return _f


def _vectorize_op(f):
    """MATLAB 'vectorize' flag: evaluate a scalar-only callable pointwise."""
    import numpy as _np

    def _f(x):
        xa = _np.asarray(x)
        vals = [f(jnp.asarray(float(v))) for v in xa.ravel()]
        arr = jnp.asarray(_np.asarray([_np.asarray(v) for v in vals]))
        return arr.reshape(xa.shape + arr.shape[1:])
    return _f


def chebpoly(n, domain=(-1.0, 1.0), kind: int = 1) -> Chebfun:
    """Chebyshev polynomial(s) ``T_n`` (``kind=1``) or ``U_n`` (``kind=2``)
    as a Chebfun on ``domain`` (MATLAB ``chebpoly(N, D, KIND)``); a
    vector ``n`` gives an array-valued Chebfun with one column per
    degree.  (The coefficient-vector version lives in
    :mod:`chebfunjax.utils.polynomials`.)

    Provenance
    ----------
    MATLAB source : chebpoly.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    from chebfunjax.utils.polynomials import chebpoly as _cheb_coeffs
    ns = _np.atleast_1d(_np.asarray(n, dtype=int)).ravel()
    if _np.any(ns < 0):
        raise ValueError("CHEBFUN:chebpoly:integern: The first argument "
                         "must be a vector of nonnegative integers.")
    if kind not in (1, 2):
        raise ValueError("CHEBFUN:chebpoly:kind: CHEBPOLY(N, KIND) only "
                         "supports KIND = 1 or KIND = 2.")
    dv = [float(v) for v in domain]
    if any(not math.isfinite(v) for v in dv):
        raise ValueError("CHEBFUN:chebpoly:infdomain: Chebyshev "
                         "polynomials are not defined over an unbounded "
                         "domain.")
    N = int(ns.max()) + 1
    C = _np.zeros((N, ns.size))
    for j, k in enumerate(ns):
        c = _np.asarray(_cheb_coeffs(int(k), kind))
        C[:c.shape[0], j] = c
    coeffs = jnp.asarray(C[:, 0] if ns.size == 1 else C)
    f = chebfun(coeffs, domain=(dv[0], dv[-1]), coeffs=True)
    if len(dv) > 2:
        f = f._restrict_breaks(dv)
    return f


def legpoly(n, domain=(-1.0, 1.0), normalize=False) -> Chebfun:
    """Legendre polynomial(s) ``P_n`` as a Chebfun on ``domain`` (MATLAB
    ``legpoly(N, D, 'norm')``); a vector ``n`` gives an array-valued
    Chebfun.  With ``normalize`` the columns are orthonormal on the
    domain.

    Provenance
    ----------
    MATLAB source : legpoly.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    from chebfunjax.utils.polynomials import legpoly as _leg_coeffs
    if isinstance(domain, str):
        normalize, domain = domain, (-1.0, 1.0)
    if isinstance(normalize, str):
        normalize = normalize.lower().startswith("norm")
    ns = _np.atleast_1d(_np.asarray(n, dtype=int)).ravel()
    dv = [float(v) for v in domain]
    if any(not math.isfinite(v) for v in dv):
        raise ValueError("CHEBFUN:legpoly:infdomain: Legendre polynomials "
                         "are not defined over an unbounded domain.")
    N = int(ns.max()) + 1
    nmax = int(ns.max())
    # MATLAB legpoly.m method selection: method 3 (leg2cheb) for
    # nMax > 1000, falling back to the recurrence (method 1) when many
    # degrees are requested; method 2 (weighted QR on a Chebyshev grid of
    # twice the size) otherwise.
    if nmax <= 1000:
        # MATLAB legpoly.m method 2: weighted QR of the Chebyshev
        # Vandermonde matrix on a Chebyshev grid of twice the size.
        from chebfunjax.utils.quadrature import chebweights
        pts = 2 * N
        w = _np.asarray(chebweights(pts, kind=2), dtype=float)
        theta = _np.pi * _np.arange(pts - 1, -1, -1) / (pts - 1)
        A = _np.cos(_np.outer(theta, _np.arange(nmax + 1)))
        Q, _R = _np.linalg.qr(_np.sqrt(w)[:, None] * A)
        P = Q / _np.sqrt(w)[:, None]
        if normalize:
            PP = P[:, ns] * (_np.sqrt(2.0 / (dv[-1] - dv[0]))
                             * _np.sign(P[-1, ns]))
        else:
            PP = P[:, ns] * (1.0 / P[-1, ns])
        from chebfunjax.tech.chebtech import Chebtech2
        C = _np.asarray(Chebtech2.vals2coeffs(jnp.asarray(PP)))[:N, :]
        coeffs = jnp.asarray(C[:, 0] if ns.size == 1 else C)
        f = chebfun(coeffs, domain=(dv[0], dv[-1]), coeffs=True)
    elif ns.size > nmax / 5:
        # MATLAB legpoly.m method 1: the three-term recurrence evaluated
        # at nmax+1 Chebyshev points (accurate to ~1e-13 at degree 1000
        # where the leg2cheb transform is only ~1e-12).
        from chebfunjax.utils.quadrature import chebpts
        x = _np.asarray(chebpts(N, kind=2), dtype=float)
        want = {int(k): j for j, k in enumerate(ns)}
        V = _np.zeros((N, ns.size))
        L0 = _np.ones_like(x)
        L1 = x.copy()
        for k in range(0, nmax + 1):
            if k in want:
                scl = (_np.sqrt((2 * k + 1) / (dv[-1] - dv[0]))
                       if normalize else 1.0)
                V[:, want[k]] = L0 * scl
            L0, L1 = L1, ((2 * k + 3) * x * L1 - (k + 1) * L0) / (k + 2)
        vals = jnp.asarray(V[:, 0] if ns.size == 1 else V)
        f = chebfun(vals, domain=(dv[0], dv[-1]))
    else:
        C = _np.zeros((N, ns.size))
        for j, k in enumerate(ns):
            c = _np.asarray(_leg_coeffs(int(k)))
            if normalize:
                c = c * _np.sqrt((2 * k + 1) / (dv[-1] - dv[0]))
            C[:c.shape[0], j] = c
        coeffs = jnp.asarray(C[:, 0] if ns.size == 1 else C)
        f = chebfun(coeffs, domain=(dv[0], dv[-1]), coeffs=True)
    if len(dv) > 2:
        f = f._restrict_breaks(dv)
    return f


def poly(v, domain=(-1.0, 1.0)):
    """Monic polynomial with the given roots as a Chebfun (MATLAB
    ``poly(v, domain)``): the roots are Leja-ordered and the product
    formed on ``chebpts(N+1)``.  A matrix of roots gives one column per
    root vector (a list of Chebfuns).

    Provenance
    ----------
    MATLAB source : @domain/poly.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    from chebfunjax.utils.quadrature import chebpts
    V = _np.asarray(v, dtype=complex if _np.iscomplexobj(v) else float)
    dv = [float(domain[0]), float(domain[-1])]
    if V.ndim == 2 and min(V.shape) > 1:
        return [poly(V[:, k], domain) for k in range(V.shape[1])]
    V = V.reshape(-1)
    N = V.size
    V = V[~_np.isinf(V)]
    if _np.any(_np.isnan(V)):
        return chebfun(lambda x: jnp.full_like(x, jnp.nan), domain=tuple(dv))
    if N == 0:
        return Chebfun.empty() if hasattr(Chebfun, "empty") else \
            chebfun(lambda x: 0.0 * x, domain=tuple(dv))
    # Leja ordering of the roots (MATLAB poly.m)
    vv = list(V)
    j = int(_np.argmax(_np.abs(vv)))
    z = [vv.pop(j)]
    for _k in range(1, N):
        if not vv:
            break
        P = [_np.prod([zz - vl for zz in z]) for vl in vv]
        j = int(_np.argmax(_np.abs(P)))
        z.append(vv.pop(j))
    x = _np.asarray(chebpts(N + 1, dv[0], dv[1]) if False else
                    dv[0] + (dv[1] - dv[0]) * (_np.asarray(chebpts(N + 1)) + 1) / 2)
    pvals = _np.ones(N + 1, dtype=complex if _np.iscomplexobj(V) else float)
    for zk in z:
        pvals = pvals * (x - zk)
    return chebfun(jnp.asarray(pvals), domain=tuple(dv))


def polyfit(x, y, n: int, domain=(-1.0, 1.0)):
    """Least-squares polynomial of degree ``n`` through the data
    ``(x, y)`` as a Chebfun on ``domain`` (MATLAB ``polyfit(x, y, n,
    domain)``): the Chebyshev-basis Vandermonde system is solved in the
    least-squares sense.

    Provenance
    ----------
    MATLAB source : @domain/polyfit.m
    Chebfun commit: 7574c77
    """
    import numpy as _np
    x = _np.asarray(x, dtype=float).reshape(-1)
    y = _np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[0] != x.shape[0]:
        if y.shape[1] == x.shape[0]:
            y = y.T
        else:
            raise ValueError("CHEBFUN:DOMAIN:polyfit:xIn: X and Y vectors "
                             "must be the same size.")
    a, b = float(domain[0]), float(domain[-1])
    xm = 2 * (x - a) / (b - a) - 1
    T = _np.zeros((x.size, n + 1))
    T[:, 0] = 1.0
    if n >= 1:
        T[:, 1] = xm
    for k in range(2, n + 1):
        T[:, k] = 2 * xm * T[:, k - 1] - T[:, k - 2]
    if T.shape[0] >= T.shape[1]:
        c = _np.linalg.lstsq(T, y, rcond=None)[0]
    else:
        # MATLAB's backslash on an underdetermined system: the basic
        # solution from a column-pivoted QR (at most m nonzero
        # coefficients), not the minimum-norm one.
        import scipy.linalg as _sla
        Q, R, piv = _sla.qr(T, mode="economic", pivoting=True)
        m = T.shape[0]
        c = _np.zeros((T.shape[1], y.shape[1]))
        c[piv[:m], :] = _sla.solve_triangular(R[:, :m], Q.T @ y)
    coeffs = jnp.asarray(c[:, 0] if c.shape[1] == 1 else c)
    return chebfun(coeffs, domain=(a, b), coeffs=True)


def chebvar(*names, domain=(-1.0, 1.0)):
    """Identity chebfuns on ``domain`` (MATLAB ``chebvar x y [a b]``):
    one Chebfun per requested name, returned as a tuple (a single
    Chebfun for one name).

    Provenance
    ----------
    MATLAB source : chebvar.m
    Chebfun commit: 7574c77
    """
    names = [t for t in names if isinstance(t, str)] or ["x"]
    xs = tuple(chebfun(lambda t: t, domain=tuple(float(v) for v in domain))
               for _ in names)
    return xs[0] if len(xs) == 1 else xs


def polyval(p, x):
    """Evaluate a polynomial with coefficients ``p`` (highest degree
    first, MATLAB order) at a Chebfun ``x`` by Horner's rule (MATLAB
    ``polyval(p, x)``).  A coefficient MATRIX evaluates one polynomial
    per column and returns a list of Chebfuns; a list/quasimatrix ``x``
    evaluates the polynomial at each column.  Both at once is an error.

    Provenance
    ----------
    MATLAB source : @chebfun/polyval.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    P = _np.asarray(p, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    xs = [x] if isinstance(x, Chebfun) else list(x)
    if P.shape[1] > 1 and len(xs) > 1:
        raise ValueError(
            "CHEBFUN:CHEBFUN:polyval:dimMismatch: Input P must be a column "
            "vector or X must be a scalar-valued CHEBFUN.")
    outs = []
    for col in range(P.shape[1]):
        for xk in xs:
            y = 0.0 * xk + float(P[0, col])
            for j in range(1, P.shape[0]):
                y = y * xk + float(P[j, col])
            outs.append(y)
    return outs[0] if len(outs) == 1 else outs


def _hscale(f: Chebfun) -> float:
    bp = [float(v) for v in f.domain.breakpoints]
    h = max(abs(bp[0]), abs(bp[-1]))
    return h if math.isfinite(h) and h > 0 else 1.0


def tweak_domain(f: Chebfun, g=None, tol: float | None = None,
                 side: int = 0):
    """Nudge nearly-coincident breakpoints of ``f`` and ``g`` onto each
    other (MATLAB ``tweakDomain``): breakpoints that differ by less than
    ``tol`` (default ``2e-15 * hscale``) are replaced by their average
    (``side = 0``), by ``f``'s value (``side < 0``) or ``g``'s
    (``side > 0``), rounded to an integer when within ``tol`` of one.
    Numerical breakpoint operations use JAX arrays; constructing static maps
    and moved-index lists is eager. Stored point values retain their data.
    Breakpoints adjacent to intervals shorter than ``2*tol`` are left
    alone.  ``g`` may also be a domain (sequence of breakpoints).

    Returns ``(f, g, loc_f, loc_g)`` with the 0-based indices of the
    breakpoints that moved.

    Provenance
    ----------
    MATLAB source : @chebfun/tweakDomain.m
    Chebfun commit: 7574c77
    """

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.fun.unbndfun import Unbndfun

    if g is None:
        return f, g, [], []
    if side is None:
        side = 0
    if f.isempty() or (isinstance(g, Chebfun) and g.isempty()):
        return f, g, [], []

    # Source treats numeric and DOMAIN inputs as the single-input/column
    # compatibility case, with the supplied domain receiving side=+1.
    domain_given = not isinstance(g, Chebfun)
    if domain_given:
        raw_g = jnp.asarray(g.breakpoints if isinstance(g, Domain) else g,
                            dtype=jnp.float64).reshape(-1)
        unique_g = jnp.unique(raw_g)
        if unique_g.shape[0] < 2:
            return f, g, [], []
        # MATLAB sets dom=unique(g) to test emptiness, then uses the original
        # g as dom when it has at least two unique entries.
        g_dom = raw_g
        # Source calls the recursive form with SIDE=+1 for numeric/DOMAIN
        # inputs, irrespective of a caller-supplied side argument.
        side = 1
        if tol is None:
            ends = jnp.max(jnp.abs(g_dom[jnp.asarray([0, -1])]))
            hsg = float(jax.device_get(ends))
            if not math.isfinite(hsg):
                hsg = 1.0
            tol = 1e-15 * max(_hscale(f), hsg)
    else:
        g_dom = jnp.asarray(g.domain.breakpoints, dtype=jnp.float64)
        if tol is None:
            tol = 2e-15 * max(_hscale(f), _hscale(g))

    tol = float(tol)
    f_dom = jnp.asarray(f.domain.breakpoints, dtype=jnp.float64)
    if f_dom.shape[0] < 2 or g_dom.shape[0] < 2:
        return f, g, [], []

    # Source masks both ends of every interval strictly shorter than 2*tol,
    # so neither breakpoint may participate in a match.
    tiny_f = jnp.diff(f_dom) < 2.0 * tol
    mask_f = jnp.concatenate((tiny_f, jnp.asarray([False]))) | jnp.concatenate(
        (jnp.asarray([False]), tiny_f))
    tiny_g = jnp.diff(g_dom) < 2.0 * tol
    mask_g = jnp.concatenate((tiny_g, jnp.asarray([False]))) | jnp.concatenate(
        (jnp.asarray([False]), tiny_g))
    f_work = jnp.where(mask_f, jnp.nan, f_dom)
    g_work = jnp.where(mask_g, jnp.nan, g_dom)
    distances = jnp.abs(f_work[:, None] - g_work[None, :])
    matched = (distances > 0.0) & (distances < tol)
    loc_f = jnp.any(matched, axis=1)
    loc_g = jnp.any(matched, axis=0)
    if not bool(jax.device_get(jnp.any(loc_f))):
        return f, g, [], []

    if side == 0:
        new_breaks = (f_dom[loc_f] + g_dom[loc_g]) / 2.0
    elif side < 0:
        new_breaks = f_dom[loc_f]
    else:
        new_breaks = g_dom[loc_g]

    # MATLAB round() is half-away-from-zero (jnp.round is ties-to-even).
    magnitude = jnp.abs(new_breaks)
    whole = jnp.floor(magnitude)
    rounded = jnp.sign(new_breaks) * (
        whole + ((magnitude - whole) >= 0.5).astype(magnitude.dtype))
    new_breaks = jnp.where(jnp.abs(rounded - new_breaks) < tol,
                           rounded, new_breaks)
    f_new = f_dom.at[loc_f].set(new_breaks)
    g_new = g_dom.at[loc_g].set(new_breaks)

    def rebuild(h, breaks):
        breaks_host = [float(jax.device_get(x)) for x in breaks]
        funs = []
        for k, piece in enumerate(h.funs):
            interval = (breaks_host[k], breaks_host[k + 1])
            if isinstance(piece, _Piece):
                # MATLAB bndfun/changeMap changes only the affine map/domain;
                # the onefun coefficients remain on reference [-1,1].
                funs.append(_Piece(tech=piece.tech, interval=interval))
            elif isinstance(piece, Unbndfun):
                # MATLAB unbndfun/changeMap rebuilds its nonlinear map for the
                # new still-unbounded interval; from_chebtech mirrors that
                # mapping update while retaining the same mapped onefun.
                funs.append(Unbndfun.from_chebtech(
                    piece.onefun, Domain(interval)))
            else:
                raise NotImplementedError(
                    f"tweak_domain cannot remap {type(piece).__name__}")
        out = Chebfun(funs=funs,
                      domain=Domain(tuple(breaks_host)),
                      deltas=h.deltas)
        if h.is_transposed:
            object.__setattr__(out, "_is_transposed", True)
        # MATLAB keeps explicit pointValues separate while remapping smooth
        # FUNs. Preserve the exact JAX leaf and dtype, including complex data.
        if h._point_values is not None:
            object.__setattr__(out, "_point_values", h._point_values)
        return out

    f_out = rebuild(f, f_new)
    if domain_given:
        g_out = (Domain(tuple(float(jax.device_get(x)) for x in g_new))
                 if isinstance(g, Domain) else g_new)
    else:
        g_out = rebuild(g, g_new) if bool(jax.device_get(jnp.any(loc_g))) else g
    return (
        f_out,
        g_out,
        [int(i) for i in jax.device_get(jnp.flatnonzero(loc_f)).tolist()],
        [int(i) for i in jax.device_get(jnp.flatnonzero(loc_g)).tolist()],
    )



def chebfun(f=None, *, domain=(-1.0, 1.0), **kwargs) -> Chebfun:
    """Create a Chebfun (MATLAB ``chebfun(...)``); see
    :func:`_chebfun_build` for every flag.

    As in MATLAB's constructor, a Chebfun built from an operator records
    ``op(breakpoints)`` as its ``pointValues`` (@chebfun/constructor.m,
    ``getValuesAtBreakpoints``), so evaluation exactly AT a breakpoint
    returns the operator's own value there (``sign(0) = 0``,
    ``sin(pi*1)`` at ``x = 1``).

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m, @chebfun/constructor.m
    Chebfun commit: 7574c77
    """
    out = _chebfun_build(f, domain=domain, **kwargs)
    if (callable(f) or isinstance(f, str)) and not isinstance(f, Chebfun) \
            and not out.isempty() \
            and getattr(out, "_point_values", None) is None:
        try:
            import numpy as _np
            op = _string_op(f) if isinstance(f, str) else _vector_check(f)
            bps = _np.asarray(list(out.domain.breakpoints), dtype=float)
            finite = _np.isfinite(bps)
            pv = _np.array(out.point_values)
            if _np.any(finite):
                vals = _np.asarray(op(jnp.asarray(bps[finite])))
                if _np.iscomplexobj(vals) and not _np.iscomplexobj(pv):
                    pv = pv.astype(_np.complex128)
                if vals.shape == pv[finite].shape and \
                        bool(_np.all(_np.isfinite(vals))):
                    pv[finite] = vals
                    out = out.set_point_values(jnp.asarray(pv))
        except Exception:
            pass
    return out


chebfun.__doc__ = (chebfun.__doc__ or "") + "\n\n" + (
    _chebfun_build.__doc__ or "")


# Attach factory classmethods to `chebfun` callable so users can write
# ``chebfun.from_coeffs(...)`` and ``chebfun.from_values(...)`` as shown
# in the API design doc.
chebfun.from_coeffs = Chebfun.from_coeffs  # type: ignore[attr-defined]
chebfun.from_values = Chebfun.from_values  # type: ignore[attr-defined]
chebfun.constructODEsol = Chebfun.constructODEsol  # type: ignore[attr-defined]
chebfun.odesol = Chebfun.odesol             # type: ignore[attr-defined]
chebfun.identity = Chebfun.identity        # type: ignore[attr-defined]


# ============================================================================
# ODE integrators: ode45 / ode113  (V04)
# ============================================================================
# uses-numpy: scipy.integrate.solve_ivp uses NumPy arrays internally


def ode45(odefun, tspan, y0, options=None, *, rtol=None, atol=None,
          dense_n=None, return_time=False, backend=None, **kwargs):
    """Solve an IVP and fit its dense output through source ODESOL.

    ``tspan`` may contain restart breakpoints. Systems return one array-valued
    Chebfun; columns are accessible as ``y[k]`` or ``y[:, k]``. Set return_time
    for ``(t, y)``. ``options`` accepts RelTol, AbsTol, restartSolver,
    happinessCheck and Events, plus the supported solver options documented
    in ``_ode_solve``. Keyword rtol/atol override their option fields.

    The inherited integration boundary uses SciPy RK45, with omitted solver
    tolerances 1e-3/1e-6. Chebfun fitting uses the separate source ODESOL
    options/fallbacks. Native MATLAB integration and output sampling remain
    unported. The dense_n argument is retained but unused: fitting is adaptive.

    Provenance
    ----------
    MATLAB source : @chebfun/ode45.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    return _ode_solve('RK45', odefun, tspan, y0, options,
                      rtol=rtol, atol=atol, dense_n=dense_n,
                      return_time=return_time, backend=backend, **kwargs)



def ode113(odefun, tspan, y0, options=None, *, rtol=None, atol=None,
           dense_n=None, return_time=False, backend=None, **kwargs):
    """Solve and construct through ODESOL; see :func:`ode45` for options.

    The default backend uses the native JAX method for finite float64/complex128
    problems and the supported solver options. Unsupported native options raise;
    backend="scipy" explicitly selects the inherited DOP853 compatibility path.
    Events, mass matrices and broader native options remain unfinished.

    Provenance
    ----------
    MATLAB source : @chebfun/ode113.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    return _ode_solve('ode113', odefun, tspan, y0, options,
                      rtol=rtol, atol=atol, dense_n=dense_n,
                      return_time=return_time, backend=backend, **kwargs)



def ode15s(odefun, tspan, y0, options=None, *, rtol=None, atol=None,
           dense_n=None, return_time=False, backend=None, **kwargs):
    """Solve and construct through ODESOL; see :func:`ode45` for options.

    The inherited SciPy BDF integration boundary does not implement native
    MATLAB's NDF method. Native algorithm parity remains open.

    Provenance
    ----------
    MATLAB source : @chebfun/ode15s.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    return _ode_solve('BDF', odefun, tspan, y0, options,
                      rtol=rtol, atol=atol, dense_n=dense_n,
                      return_time=return_time, backend=backend, **kwargs)



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

    # MATLAB max.m: on each subinterval the winner is decided by a
    # midpoint comparison and that chebfun is RESTRICTED to the piece.
    # No re-construction: re-approximating pick(f, g) near a crossing
    # produces a piece whose vscale is O(eps), on which the residual
    # crossing offset is an O(1) relative step that never converges.
    is_max = pick is jnp.maximum
    funs = []
    for lo, hi in zip(domain[:-1], domain[1:]):
        midv = jnp.asarray(0.5 * (lo + hi))
        fv = float(f(midv))
        gv = float(g_eval(midv))
        take_f = (fv >= gv) if is_max else (fv <= gv)
        if take_f:
            piece = f.restrict(lo, hi)
        elif isinstance(other, Chebfun):
            piece = other.restrict(lo, hi)
        else:
            piece = chebfun(
                lambda x: jnp.full_like(jnp.asarray(x, dtype=jnp.float64), c),
                domain=(lo, hi))
        funs.extend(piece.funs)
    return Chebfun(funs=funs, domain=Domain(tuple(domain)))


def _merge_limit_row(piece, right):
    """Source Chebtech lval/rval, as a single row of function columns."""
    from chebfunjax.fun.singfun import Singfun
    if isinstance(piece.tech, Singfun):
        # classicfun/get delegates lval/rval to the full singular onefun.
        return jnp.atleast_1d(piece.tech(jnp.asarray(1.0 if right else -1.0)))
    coeffs = jnp.asarray(piece.tech.coeffs)
    if not right:
        signs = jnp.where(jnp.arange(coeffs.shape[0]) % 2, -1, 1)
        coeffs = coeffs * (signs if coeffs.ndim == 1 else signs[:, None])
    return jnp.atleast_1d(jnp.sum(coeffs, axis=0))


def _finalize_bounded_singular(funs, given, op, *, split_length=160,
                               tol=None, turbo=False, check="standard",
                               sample_test=True, min_samples=None,
                               refinement_function=None):
    """Capture callback point values and merge only introduced boundaries.

    Provenance
    ----------
    MATLAB source : @chebfun/chebfun.m (outer construction finalization),
        @chebfun/getValuesAtBreakpoints.m, @chebfun/merge.m
    Chebfun commit: 7574c77
    Uses existing bounded Singfun merge and limit-row adapters.
    """
    ends = [funs[0].interval[0]] + [piece.interval[1] for piece in funs]
    out = Chebfun(funs=list(funs), domain=Domain(tuple(ends)))
    object.__setattr__(out, "_point_values",
                       _source_breakpoint_values(funs, ends, op))
    introduced = [x for x in ends[1:-1] if x not in given]
    if introduced:
        out = out.merge(index=introduced, max_length=split_length,
                        splitting=True, tol=tol, turbo=turbo, check=check,
                        sample_test=sample_test, min_samples=min_samples,
                        refinement_function=refinement_function)
    return out


def _source_breakpoint_values(funs, ends, op=None):
    """Bounded smooth getValuesAtBreakpoints; evaluate OP once when supplied.

    Provenance
    ----------
    MATLAB source : @chebfun/getValuesAtBreakpoints.m
    Chebfun commit: 7574c77
    """
    rows = [_merge_limit_row(funs[0], False)]
    rows.extend((_merge_limit_row(p, True) + _merge_limit_row(q, False)) / 2
                for p, q in zip(funs[:-1], funs[1:]))
    rows.append(_merge_limit_row(funs[-1], True))
    fallback = jnp.stack(rows)
    if op is None:
        values = fallback
    else:
        values = jnp.asarray(op(jnp.asarray(ends, dtype=jnp.float64)))
        if values.ndim == 0:
            values = jnp.broadcast_to(values, fallback.shape)
        elif values.ndim == 1:
            values = values[:, None]
        if values.shape != fallback.shape:
            raise ValueError("Callback must return one row per breakpoint.")
        values = jnp.where(jnp.isnan(values), fallback, values)
    return values[:, 0] if funs[0].tech.coeffs.ndim == 1 else values


def _merge_pair_values(x, left, right):
    """@fun/merge.m myFun: current neighbors, right wins at shared end."""
    x = jnp.asarray(x)
    flat = x.reshape((-1,))
    multi = left.tech.coeffs.ndim == 2
    columns = left.tech.coeffs.shape[1] if multi else 1
    dtype = jnp.result_type(x, left.tech.coeffs, right.tech.coeffs)
    values = jnp.zeros((flat.size, columns), dtype=dtype)
    # MATLAB relational comparisons on complex arguments use real parts.
    mask_left = jnp.real(flat) <= left.interval[1]
    mask_right = jnp.real(flat) >= right.interval[0]
    if bool(jnp.any(mask_left)):
        sampled = jnp.asarray(left(flat[mask_left])).reshape((-1, columns))
        values = values.at[mask_left].set(sampled)
    if bool(jnp.any(mask_right)):
        sampled = jnp.asarray(right(flat[mask_right])).reshape((-1, columns))
        values = values.at[mask_right].set(sampled)
    return values.reshape(x.shape + ((columns,) if multi else ()))



def _split_breakpoints(f, a: float, b: float, maxpow2: int,
                       depth: int = 0, max_depth: int = 45,
                       min_w: "float | None" = None,
                       split_pow2: int = 8,
                       tol=None, vscale: float = 0.0,
                       budget: "dict | None" = None,
                       hscale: "float | None" = None,
                       check: str = "standard",
                       sample_test: bool = True,
                       min_samples: int | None = None,
                       refinement_function: str | Callable | None = None,
                       max_length: int | None = None,
                       ) -> list:
    """Recursively find breakpoints using the effective raw Tech grid cap.

    Grid clipping and endpoint extrapolation follow the source. Breakpoint
    discovery still uses Python's recursive order; MATLAB's global widest-sad
    interval scheduling requires a separate port.

    Provenance
    ----------
    MATLAB source : @chebfun/constructor.m, @chebtech2/refine.m
    Chebfun commit: 7574c77
    """
    import warnings as _warnings
    # The whole-domain scale is threaded unchanged through discovery.
    if hscale is None:
        hscale = max(abs(a), abs(b), 1.0)
    if min_w is None:
        # MATLAB keeps subdividing a sad piece down to the scale of the
        # domain's floating-point resolution (1e-14 * hscale); a fixed
        # 1e-10 floor left sqrt(1-x) slivers 1e-6 in error near x = 1.
        min_w = 1e-14 * hscale
    det = min(maxpow2, split_pow2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # thread the caller's eps into detection: with noisy data at the
        # eps level, machine-precision happiness would never be reached
        # and the recursion would grind to min_w on every subinterval
        # MATLAB constructorSplit sets pref.techPrefs.extrapolate = true:
        # the piece is built on [a, b] itself, never sampling the ends.
        p = _Piece.from_function(f, a, b, maxpow2=det, tol=tol,
                                 vscale=vscale, extrapolate=True,
                                 check=check,
                                 hscale=hscale / (b - a),
                                 sample_test=sample_test,
                                 min_samples=min_samples,
                                 max_length=max_length,
                                 refinement_function=refinement_function)
    if budget is not None:
        # This recursive adapter bounds discovery with pending-piece
        # estimates. Each completed fit replaces its estimate by actual length.
        # MATLAB checks actual total length after both children in its global
        # scheduling loop; that broader scheduling contract remains open.
        _sl = 2 ** split_pow2 + 1
        # Replace the pending estimate by the actual completed fit. A raw
        # clipped grid can be160 while the pending estimate is129.
        budget["used"] += int(p.n) - _sl
        if not p.ishappy:
            if budget["used"] >= budget["max"]:
                return []
            # Replace this parent by two pending child estimates. Global
            # source scheduling and its after-pair budget check remain open.
            budget["used"] += 2 * _sl - int(p.n)
    if p.ishappy or (b - a) < min_w or depth > max_depth:
        return []
    # Locate the singularity with the MATLAB detectEdge derivative-growth
    # test (orders 1..4 with bracket refinement; findJump bisection for
    # first-derivative blowups).  The order-adaptive localisation places
    # e.g. a spline-knot (D3-jump) edge ~1e-5 off, which is exactly tight
    # enough for the neighbouring cubic pieces to be happy at 1e-15 --
    # the previous first/second-difference scan put such edges ~3e-4 off
    # and the still-unhappy pieces cascaded into hundreds of splits.
    e = _detect_edge_matlab(f, a, b, vscale=max(float(vscale), float(p.vscale)),
                            hscale=hscale)
    if e is None:
        # MATLAB @chebfun/constructor.m: no edge detected -> bisect.
        # (A finite-difference edge heuristic used here before found
        # spurious edges everywhere on |x|^5 -- 247 pieces.)
        e = 0.5 * (a + b)
    w = b - a
    htol = 1e-14 * hscale
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
    return (_split_breakpoints(f, a, e, maxpow2, depth + 1, max_depth,
                               min_w, split_pow2, tol, vscale, budget, hscale,
                               check, sample_test, min_samples,
                               refinement_function, max_length)
            + [e]
            + _split_breakpoints(f, e, b, maxpow2, depth + 1, max_depth,
                                 min_w, split_pow2, tol, vscale, budget, hscale,
                                 check, sample_test, min_samples,
                                 refinement_function, max_length))


def _edge_sample_rows(f, x):
    """Normalize Python callback shapes without combining function columns."""
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(f(x))
    if not jnp.issubdtype(y.dtype, jnp.complexfloating):
        y = y.astype(jnp.float64)
    count = x.size
    if y.ndim == 0:
        # Scalar-returning Python callbacks represent a constant column.
        return jnp.broadcast_to(y, (count, 1))
    if y.ndim == 1:
        if count == 1:
            return y.reshape((1, -1))
        if y.size == count:
            return y.reshape((count, 1))
    elif y.ndim == 2 and y.shape[0] == count:
        return y
    raise ValueError("Edge detection requires one sample row per input point.")


def _edge_eps(x):
    """Positive binary64 MATLAB eps(x), including negative and subnormal x.

    Python's scalar IEEE ulp avoids device flush-to-zero during subtraction
    of adjacent subnormals; all sampled arrays and reductions remain JAX.
    """
    x = float(x)
    return math.ulp(x) if math.isfinite(x) else math.nan


def _edge_find_max_der(f, a, b, num_ders, grid_size):
    """Bounded identity-map findMaxDer from @fun/detectEdge.m:271-312."""
    na = jnp.full((num_ders,), a, dtype=jnp.float64)
    nb = jnp.full((num_ders,), b, dtype=jnp.float64)
    max_der = jnp.zeros((num_ders,), dtype=jnp.float64)
    dx = (b - a) / (grid_size - 1)
    x = jnp.concatenate((a + jnp.arange(grid_size - 1, dtype=jnp.float64) * dx,
                         jnp.asarray([b], dtype=jnp.float64)))
    dy = _edge_sample_rows(f, x)
    for order in range(num_ders):
        dy = jnp.diff(dy, axis=0)
        x = (x[:-1] + x[1:]) / 2
        # MATLAB max omits NaNs but preserves an all-NaN result. Preserve
        # infinities and reduce columns only after differencing sample rows.
        dydh = jnp.nanmax(jnp.abs(dy), axis=1)
        maximum = jnp.nanmax(dydh)
        # abs() makes all non-NaN entries nonnegative. Replacing NaNs by
        # -inf for the index yields the first finite/infinite tied maximum.
        ind = int(jnp.argmax(jnp.where(jnp.isnan(dydh), -jnp.inf, dydh)))
        max_der = max_der.at[order].set(maximum)
        if ind > 0:
            na = na.at[order].set(x[ind - 1])
        if ind < x.size - 2:
            nb = nb.at[order].set(x[ind + 1])
    if dx ** num_ders <= _edge_eps(0.0):
        max_der = jnp.inf + max_der
    else:
        max_der = max_der / jnp.asarray(
            [dx ** order for order in range(1, num_ders + 1)],
            dtype=jnp.float64)
    return na, nb, max_der


def _find_jump(f, a, b, vscale, hscale):
    """Literal bounded findJump, with MATLAB all/any column conditions.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m (findJump)
    Chebfun commit: 7574c77
    """
    eps = float(jnp.finfo(jnp.float64).eps)
    y = _edge_sample_rows(f, jnp.asarray([a, b], dtype=jnp.float64))
    ya, yb = y[0], y[1]
    # The source divides the complete denominator by two, even for the
    # identity derivative. Preserve that expression, not an inferred slope.
    max_der = jnp.abs(ya - yb) / ((b - a) / 2)
    if bool(jnp.all(max_der < 1e-5 * vscale / hscale)):
        return None
    cont = 0
    e1 = (b + a) / 2
    e0 = e1 + 1
    while ((cont < 2 or bool(jnp.any(max_der == jnp.inf))) and e0 != e1):
        c = (a + b) / 2
        yc = _edge_sample_rows(f, jnp.asarray(c, dtype=jnp.float64))[0]
        dyl = jnp.nanmax(jnp.abs(yc - ya))
        dyr = jnp.nanmax(jnp.abs(yb - yc))
        previous = max_der
        if bool(dyl > dyr):
            b, yb = c, yc
            max_der = dyl / (b - a)
        else:
            a, ya = c, yc
            max_der = dyr / (b - a)
        e0 = e1
        e1 = (a + b) / 2
        if bool(jnp.all(max_der < previous * 1.5)):
            cont += 1
    if (e0 - e1) <= 2 * _edge_eps(e0):
        yright = _edge_sample_rows(
            f, jnp.asarray(b + _edge_eps(b), dtype=jnp.float64))[0]
        if bool(jnp.all(jnp.abs(yright - yb) > eps * 100 * vscale)):
            return b
        return a
    return None


def _detect_edge_matlab(f, a: float, b: float,
                        vscale: "float | None" = None,
                        hscale: "float | None" = None,
                        blowup: bool = False) -> "float | None":
    """Bounded smooth detectedgeMain with JAX sample arrays and reductions.

    Return a raw edge or None, retaining the existing helper API. The
    constructor supplies global scales and handles source endpoint move-in
    and empty-edge midpoint fallback. Optional omitted scales are a Python
    compatibility adapter; they are not part of source constructorSplit.
    The caller compensates exponents before entering this identity-map
    branch. Ordinary callers retain blowup=False; singular construction
    explicitly enables the source nested function-value check.

    Provenance
    ----------
    MATLAB source : @fun/detectEdge.m (detectedgeMain, findMaxDer, findJump)
    Chebfun commit: 7574c77
    """
    a, b = float(a), float(b)
    if hscale is None:
        hscale = max(abs(a), abs(b), 1.0)
    hscale = float(hscale)
    if vscale is None:
        values = _edge_sample_rows(f, jnp.linspace(a, b, 50))
        vscale = float(jnp.nanmax(jnp.abs(values)))
    else:
        vscale = float(jnp.nanmax(jnp.asarray(vscale)))
    eps = float(jnp.finfo(jnp.float64).eps)
    num = 4
    check_blowup = bool(blowup)
    na, nb, max_der = _edge_find_max_der(f, a, b, num, 50)
    left, right = float(na[num - 1]), float(nb[num - 1])
    while (bool(max_der[num - 1] != jnp.inf)
           and not bool(jnp.isnan(max_der[num - 1]))
           and right - left > eps * hscale):
        previous = max_der[:num]
        na, nb, max_der = _edge_find_max_der(f, left, right, num, 15)
        orders = jnp.arange(1, num + 1, dtype=jnp.float64)
        grows = ((max_der > (5.5 - orders) * previous)
                 & (max_der > 10 * vscale / hscale ** orders))
        if not bool(jnp.any(grows)):
            return None
        num = int(jnp.argmax(grows)) + 1
        if num == 1 and right - left < 1e-3 * hscale:
            return _find_jump(f, left, right, vscale, hscale)
        left, right = float(na[num - 1]), float(nb[num - 1])
        if check_blowup and bool(jnp.all(jnp.abs(jnp.asarray(
                f(jnp.asarray((left+right)/2)))) > 1e2*vscale)):
            blowup_point = _source_find_blowup_bounded(f, left, right, vscale)
            if blowup_point is None:
                check_blowup = False
            else:
                return blowup_point
    return (left + right) / 2

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
        ys = _np.asarray(f(jnp.asarray(xs)))
        if not _np.iscomplexobj(ys):
            ys = ys.astype(_np.float64)
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
                              tol=None, turbo: bool = False,
                              min_samples: "int | None" = None,
                              split_length: "int | None" = None,
                              split_max_length: "int | None" = None,
                              sample_test: bool = True,
                              check: str = "standard",
                              refinement_function: str | Callable | None = None,
                              *, vscale: float = 0.0,
                              hscale: float | None = None,
                              breakpoints: tuple[float, ...] | None = None):
    """Build bounded smooth pieces with the source constructorSplit loop.

    Fit supplied intervals in order, then split the first widest unhappy
    interval. Shared scale changes only after happy fits. The actual total
    length is checked after inserting both fitted children. Source merge
    subsequently considers only breaks introduced by this constructor.
    Singfun, unbounded and first-kind construction have separate adapters.

    Provenance
    ----------
    MATLAB source : @chebfun/constructor.m (constructorSplit and getFun),
        @fun/detectEdge.m, @bndfun/bndfun.m, @chebfun/chebfun.m
    Chebfun commit: 7574c77
    """
    import warnings

    given = ((float(a), float(b)) if breakpoints is None
             else tuple(float(x) for x in breakpoints))
    hscale_g = max(abs(x) for x in given) if hscale is None else float(hscale)
    vscale_g = float(jnp.max(jnp.asarray(vscale)))
    raw_split_length = 160 if split_length is None else int(split_length)
    if raw_split_length < 1:
        raise ValueError("split_length must be a positive integer")
    total_limit = 6000 if split_max_length is None else int(split_max_length)

    def get_fun(left, right):
        nonlocal vscale_g
        # Source getFun treats a sufficiently small physical interval as a
        # constant sampled once at its midpoint, instead of adapting it.
        if right - left < 4e-14 * hscale_g:
            value = jnp.asarray(f((left + right) / 2))
            if value.ndim == 0:
                coeffs = value.reshape((1,))
            elif value.ndim == 1:
                coeffs = value.reshape((1, -1))
            elif value.ndim == 2 and value.shape[0] == 1:
                coeffs = value
            else:
                raise ValueError(
                    "A scalar callback input must return a scalar or one row.")
            piece = _Piece(
                tech=Chebtech2.from_coeffs(coeffs), interval=(left, right))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                piece = _Piece.from_function(
                    f, left, right, maxpow2=maxpow2, tol=tol, turbo=turbo,
                    extrapolate=True, vscale=vscale_g,
                    hscale=hscale_g / (right - left),
                    sample_test=sample_test, check=check,
                    min_samples=min_samples, max_length=raw_split_length,
                    refinement_function=refinement_function)
        if piece.ishappy:
            vscale_g = float(jnp.maximum(vscale_g, jnp.max(piece.vscale)))
        return piece

    funs = []
    for left, right in zip(given[:-1], given[1:]):
        funs.append(get_fun(left, right))
        # constructorSplit resets an infinite initial scale after each fit.
        if math.isinf(vscale_g):
            vscale_g = 0.0

    while any(not piece.ishappy for piece in funs):
        # Python max retains the first tied maximum, as MATLAB max does.
        widths = [(piece.interval[1] - piece.interval[0])
                  if not piece.ishappy else 0.0 for piece in funs]
        k = max(range(len(funs)), key=widths.__getitem__)
        left, right = funs[k].interval
        edge = _detect_edge_matlab(f, left, right,
                                   vscale=vscale_g, hscale=hscale_g)
        if edge is None:
            edge = (left + right) / 2
        else:
            htol = 1e-14 * hscale_g
            if abs(left - edge) <= htol:
                edge = left + (right - left) / 100
            elif abs(right - edge) <= htol:
                edge = right - (right - left) / 100
        child_left = get_fun(left, edge)
        child_right = get_fun(edge, right)
        funs[k:k + 1] = [child_left, child_right]
        length = sum(piece.n for piece in funs)
        if length > total_limit:
            warnings.warn(f"Function not resolved using {length} pts.",
                          UserWarning, stacklevel=2)
            break

    ends = [funs[0].interval[0]] + [piece.interval[1] for piece in funs]
    out = Chebfun(funs=funs, domain=Domain(tuple(ends)))
    # Outer chebfun constructor captures the original callback at all final
    # breaks before source merge; only NaNs fall back to one-sided limits.
    object.__setattr__(out, "_point_values",
                       _source_breakpoint_values(funs, ends, f))
    introduced = [x for x in ends[1:-1] if x not in given]
    if introduced:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = out.merge(index=introduced, max_length=raw_split_length,
                            splitting=True, tol=tol, turbo=turbo, check=check,
                            sample_test=sample_test, min_samples=min_samples,
                            refinement_function=refinement_function)
    return out

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


def _ode_solve(method, odefun, tspan, y0, options=None, *, rtol=None,
               atol=None, dense_n=None, return_time=False, backend=None, **kwargs):
    """Select a solver backend and preserve source ODE construction routing.

    ODESET mappings: InitialStep/MaxStep/Jacobian/JPattern/Vectorized map to
    first_step/max_step/jac/jac_sparsity/vectorized. Nonempty unsupported fields
    raise NotImplementedError. Events accepts MATLAB's triple-valued callback;
    the existing Python events keyword accepts scalar SciPy event callbacks.
    Both routes return source-shaped ie/xe for constructODEsol. Dynamic event
    terminal/direction metadata is unsupported and rejected explicitly.
    dense_n is an unused legacy argument, not a fixed fitting-grid contract.

    Native integration, step/output selection, MATLAB exception IDs and
    remaining native ODESET options require further ports. Array conversion
    here belongs to the pre-existing SciPy boundary; representation numerics
    are provided by JAX ODESOL/constructODEsol.

    Provenance
    ----------
    MATLAB source : @chebfun/constructODEsol.m, @chebfun/odesol.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.construct_ode_solution import constructODEsol

    del dense_n
    opts = {} if options is None else dict(options)

    def empty(value):
        if value is None:
            return True
        if isinstance(value, (list, tuple, dict, str, bytes)):
            return len(value) == 0
        return getattr(value, 'size', None) == 0

    for name in ('restartSolver', 'happinessCheck', 'Events'):
        if name in kwargs:
            if name in opts:
                raise TypeError(f'{name} supplied twice')
            opts[name] = kwargs.pop(name)
    if rtol is not None:
        opts['RelTol'] = rtol
    if atol is not None:
        opts['AbsTol'] = atol
    if 'events' in kwargs and not empty(opts.get('Events')):
        raise TypeError('specify Events or events, not both')
    python_events = kwargs.get('events')
    if python_events is not None:
        opts['Events'] = python_events
    native_methods = {'ode113', 'ode78', 'ode89'}
    native_selected = method in native_methods and backend != 'scipy'
    if backend not in (None, 'native', 'scipy'):
        raise ValueError('backend must be native or scipy')
    if backend == 'native' and method not in native_methods:
        raise NotImplementedError(f'native backend for {method} is not implemented')
    if native_selected:
        supported = {'RelTol', 'AbsTol', 'InitialStep', 'MaxStep', 'MinStep', 'NormControl'}
        for name, value in opts.items():
            if name not in supported | {'restartSolver', 'happinessCheck'} and not empty(value):
                raise NotImplementedError(f'native {method} option {name} is not yet implemented')
        # Legacy Python aliases map to the same native source options; never
        # forward unrelated SciPy keywords or silently select a different method.
        for alias, name in {'first_step': 'InitialStep', 'max_step': 'MaxStep'}.items():
            if alias in kwargs:
                if not empty(opts.get(name)):
                    raise TypeError(f'{name} and {alias} supplied together')
                opts[name] = kwargs.pop(alias)
        if kwargs:
            raise NotImplementedError(f'native {method} keywords {sorted(kwargs)} are not implemented')
        from chebfunjax.utils.native_ode113 import native_ode113
        from chebfunjax.utils.native_rk import native_rk

        def native_solver(fun, span, initial, fitting_options):
            native_options = {name: value for name, value in fitting_options.items()
                              if name in supported}
            if method == 'ode113':
                solved = native_ode113(fun, span, initial, native_options)
            else:
                solved = native_rk(method, fun, span, initial, native_options)
            # ODESOL sees the full fitting options, separately from native ODESET.
            solved['extdata']['options'] = dict(fitting_options)
            return solved

        return constructODEsol(native_solver, odefun, tspan, y0, opts,
                               return_time=return_time)

    # uses-numpy: host conversion at the inherited, explicitly selected SciPy boundary.
    import numpy as _np
    from scipy.integrate import solve_ivp

    if method in native_methods:
        method = 'DOP853'
    mappings = {'InitialStep':'first_step', 'MaxStep':'max_step',
                'Jacobian':'jac', 'JPattern':'jac_sparsity', 'Vectorized':'vectorized'}
    known = {'RelTol', 'AbsTol', 'restartSolver', 'happinessCheck', 'Events', *mappings}
    for name, value in opts.items():
        if name not in known and not empty(value):
            raise NotImplementedError(f'MATLAB ODE option {name} is not yet implemented')
    for name, target in mappings.items():
        value = opts.get(name)
        if not empty(value):
            if target in kwargs:
                raise TypeError(f'{name} and {target} supplied together')
            if name == 'Vectorized':
                if isinstance(value, str):
                    if value.lower() not in ('on', 'off'):
                        raise ValueError('Vectorized must be on, off or boolean')
                    value = value.lower() == 'on'
                else:
                    value = bool(value)
            kwargs[target] = value
    solver_rtol = 1e-3 if empty(opts.get('RelTol')) else opts['RelTol']
    solver_atol = 1e-6 if empty(opts.get('AbsTol')) else opts['AbsTol']

    def solver(fun, span, initial, fitting_options):
        initial = _np.atleast_1d(_np.asarray(initial))
        # Native MATLAB infers complex dynamics even from real initial data.
        # SciPy requires complex y0 before constructing its solver storage.
        probe = _np.asarray(fun(float(span[0]), jnp.asarray(initial)))
        complex_state = _np.iscomplexobj(initial) or _np.iscomplexobj(probe)
        initial = initial.astype(_np.complex128 if complex_state else _np.float64)

        def rhs(t, y):
            result = fun(float(t), jnp.asarray(y))
            return _np.atleast_1d(_np.asarray(result))

        solver_kwargs = dict(kwargs)
        # Native ODESET default applies separately to each restarted span.
        # https://www.mathworks.com/help/matlab/ref/odeset.html#namevaluepairarguments
        solver_kwargs.setdefault('max_step', 0.1*abs(span[-1]-span[0]))
        matlab_events = fitting_options.get('Events')
        if python_events is None and not empty(matlab_events):
            if not callable(matlab_events):
                raise TypeError('Events must be a MATLAB triple-valued callback')

            def event_data(t, y):
                data = matlab_events(float(t), jnp.asarray(y))
                if not isinstance(data, (tuple, list)) or len(data) != 3:
                    raise ValueError('Events must return (value, isterminal, direction)')
                values = jnp.atleast_1d(jnp.asarray(data[0]))
                terminal = jnp.broadcast_to(jnp.asarray(data[1]), values.shape)
                direction = jnp.broadcast_to(jnp.asarray(data[2]), values.shape)
                return values, terminal, direction

            values, terminal, direction = event_data(span[0], initial)
            event_functions = []
            for index in range(values.size):
                stop, sign = bool(terminal[index]), float(direction[index])

                def event(t, y, index=index, stop=stop, sign=sign):
                    values, terminal, direction = event_data(t, y)
                    if bool(terminal[index]) != stop or float(direction[index]) != sign:
                        raise NotImplementedError('dynamic Events metadata is not implemented')
                    return float(values[index])

                event.terminal = stop
                event.direction = sign
                event_functions.append(event)
            solver_kwargs['events'] = event_functions
        solved = solve_ivp(rhs, [span[0], span[-1]], initial, method=method,
                           dense_output=True, rtol=solver_rtol, atol=solver_atol,
                           **solver_kwargs)
        if not solved.success:
            raise RuntimeError(f'ODE solver ({method}) failed: {solved.message}')

        def dense(x):
            return jnp.asarray(solved.sol(_np.atleast_1d(_np.asarray(x, dtype=float))))

        event_records = [(float(t), index+1)
                         for index, times in enumerate(solved.t_events or [])
                         for t in times]
        event_records.sort(reverse=span[0] > span[-1])
        return {'y': jnp.asarray(solved.y), 'sol':dense,
                'extdata':{'options':dict(fitting_options)},
                'ie':jnp.asarray([i for _,i in event_records]),
                'xe':jnp.asarray([t for t,_ in event_records])}

    return constructODEsol(solver, odefun, tspan, y0, opts, return_time=return_time)


# ============================================================================
# Higher-order ODE integrators: ode78 / ode89
# ============================================================================


def ode78(odefun, tspan, y0, options=None, *, rtol=None, atol=None,
          dense_n=None, return_time=False, backend=None, **kwargs):
    """Solve and construct through ODESOL; see :func:`ode45` for options.

    The default backend uses the native JAX method for finite float64/complex128
    problems and the supported solver options. Unsupported native options raise;
    backend="scipy" explicitly selects the inherited DOP853 compatibility path.
    Events, mass matrices and broader native options remain unfinished.

    Provenance
    ----------
    MATLAB source : @chebfun/ode78.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    return _ode_solve('ode78', odefun, tspan, y0, options,
                      rtol=rtol, atol=atol, dense_n=dense_n,
                      return_time=return_time, backend=backend, **kwargs)



def ode89(odefun, tspan, y0, options=None, *, rtol=None, atol=None,
          dense_n=None, return_time=False, backend=None, **kwargs):
    """Solve and construct through ODESOL; see :func:`ode45` for options.

    The default backend uses the native JAX method for finite float64/complex128
    problems and the supported solver options. Unsupported native options raise;
    backend="scipy" explicitly selects the inherited DOP853 compatibility path.
    Events, mass matrices and broader native options remain unfinished.

    Provenance
    ----------
    MATLAB source : @chebfun/ode89.m, @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    return _ode_solve('ode89', odefun, tspan, y0, options,
                      rtol=rtol, atol=atol, dense_n=dense_n,
                      return_time=return_time, backend=backend, **kwargs)



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
    MATLAB source : @chebfun/innerProduct.m; @adchebfun/adchebfun.m (innerProduct)
    Chebfun commit: 7574c77
    """
    from chebfunjax.autodiff.adchebfun import ADChebfun
    if isinstance(f, ADChebfun):
        return f.innerProduct(g)
    if isinstance(g, ADChebfun):
        return g.__rmul__(f).sum()
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
    from chebfunjax.chebfun1d.mtimes import _columns

    if not A or not B:
        raise ValueError("subspace: A and B must be non-empty lists of Chebfun.")

    # Row chebfuns (MATLAB A.'): the principal angles between row spans
    # equal those between the transposed column spans, so un-transpose.
    A = [f.transpose() if getattr(f, "is_transposed", False) else f
         for f in A]
    B = [f.transpose() if getattr(f, "is_transposed", False) else f
         for f in B]

    # Orthonormalise via continuous QR
    # Quasimatrix requires a Domain argument — extract from the first column
    domA = A[0].domain
    domB = B[0].domain
    qA = Quasimatrix(A, domA)
    qB = Quasimatrix(B, domB)
    QA, _ = qr_quasimatrix(qA)
    QB, _ = qr_quasimatrix(qB)

    acols, bcols = _columns(QA), _columns(QB)
    pA = len(acols)
    pB = len(bcols)

    # Build Gram matrix C_ij = <QA_i, QB_j>
    C = _np.zeros((pA, pB), dtype=_np.float64)
    for i, colA in enumerate(acols):
        for j, colB in enumerate(bcols):
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
            for i, colA in enumerate(acols):
                proj = None
                for j, colB in enumerate(bcols):
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
            for j, colB in enumerate(bcols):
                proj = None
                for i, colA in enumerate(acols):
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
    import numpy as _np

    from chebfunjax.operators.chebop import Chebop

    a, b = float(V.domain.a), float(V.domain.b)
    n_req = int(n)

    from chebfunjax.tech.trigtech import Trigtech
    _periodic = isinstance(V.funs[0].tech, Trigtech)

    L = Chebop(lambda x_, u: -h**2 * u.diff(2) + V * u, domain=(a, b))
    if _periodic:
        # MATLAB: a periodic (trig) potential gives periodic eigenstates.
        L.bc = "periodic"
    else:
        L.lbc = 0.0
        L.rbc = 0.0
    # MATLAB quantumstates.m: [U, D] = eigs(L, n, 'sr') on the adaptive
    # linop discretisation (a fixed-grid solve leaves O(1e-9) derivative
    # jumps at breakpoints of V, which diff() then turns into deltas).
    _out = L.eigs(k=n_req, sigma="SR", return_eigenfunctions=True)
    # (the periodic path returns MATLAB's [V, D] order)
    lam, funs = (_out[1], _out[0]) if isinstance(_out[0], list) else _out
    lam = _np.real(_np.asarray(lam))
    order = _np.argsort(lam)
    lam, funs = lam[order], [funs[i] for i in order]

    out_funs = []
    for f in funs:
        nrm = float(f.norm())
        if nrm > 1e-15:
            f = f * (1.0 / nrm)
        xs = jnp.linspace(a, b, 7)
        v = _np.asarray(f(xs))
        if v[int(_np.argmax(_np.abs(v)))] < 0:
            f = -f
        out_funs.append(f)
    return jnp.asarray(lam, dtype=jnp.float64), out_funs


# MATLAB static methods and Python factory aliases share public wrapper routing.
Chebfun.ode45 = staticmethod(ode45)  # type: ignore[attr-defined]
chebfun.ode45 = ode45  # type: ignore[attr-defined]
Chebfun.ode113 = staticmethod(ode113)  # type: ignore[attr-defined]
chebfun.ode113 = ode113  # type: ignore[attr-defined]
Chebfun.ode15s = staticmethod(ode15s)  # type: ignore[attr-defined]
chebfun.ode15s = ode15s  # type: ignore[attr-defined]
Chebfun.ode78 = staticmethod(ode78)  # type: ignore[attr-defined]
chebfun.ode78 = ode78  # type: ignore[attr-defined]
Chebfun.ode89 = staticmethod(ode89)  # type: ignore[attr-defined]
chebfun.ode89 = ode89  # type: ignore[attr-defined]


def _merge_bounded_fun_source(left, right, *, maxpow2, max_length, tol,
                              splitting, vscale, hscale, sample_test,
                              min_samples, refinement_function, turbo, check):
    """Bounded FUN.merge with source outer exponents and construction prefs.

    Provenance
    ----------
    MATLAB source : @fun/merge.m, @bndfun/bndfun.m, @onefun/onefun.m,
        @singfun/singfun.m, @singfun/constructSmoothPart.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.
    Proactive blowup detection is disabled; only stored outer exponents are
    retained. Factory smooth tech is Chebtech2, as in existing merge API.
    """
    from chebfunjax.fun.singfun import Singfun

    a, b = left.interval[0], right.interval[1]
    ea = left.tech.exponents[0] if isinstance(left.tech, Singfun) else 0.0
    eb = right.tech.exponents[1] if isinstance(right.tech, Singfun) else 0.0
    if ea == 0.0 and eb == 0.0:
        return _Piece.from_function(
            lambda x: _merge_pair_values(x, left, right), a, b,
            maxpow2=maxpow2, max_length=max_length, tol=tol,
            extrapolate=splitting, vscale=vscale, hscale=hscale/(b-a),
            sample_test=sample_test, min_samples=min_samples,
            refinement_function=refinement_function, turbo=turbo, check=check)

    def mapped(t):
        x = t if (a, b) == (-1.0, 1.0) else b*(t+1)/2+a*(1-t)/2
        return _merge_pair_values(x, left, right)

    # Source SINGFUN rejects array-valued operators before sampling refinement.
    if jnp.asarray(mapped(jnp.asarray(0.0))).size > 1:
        raise ValueError('SINGFUN does not support array-valued construction.')

    def smooth(t):
        values = mapped(t)
        if ea and eb:
            return values / ((1+t)**ea * (1-t)**eb)
        if ea:
            return values / (1+t)**ea
        return values / (1-t)**eb

    options = dict(maxpow2=maxpow2, max_length=max_length,
                   tol=jnp.maximum(jnp.asarray(tol), 1e-14),
                   extrapolate=splitting or ea < 0 or eb < 0,
                   vscale=vscale, hscale=hscale/(b-a), sample_test=sample_test,
                   min_samples=min_samples, turbo=turbo, check=check)
    if refinement_function is not None:
        options['refinement_function'] = refinement_function
    smooth_part = Chebtech2.from_function(smooth, **options)
    return _Piece(Singfun(smooth_part, (ea, eb)), (a, b))


def _merge_fun_source(left, right, **options):
    """Dispatch source FUN.merge by the new union domain's representation.

    Provenance
    ----------
    MATLAB source : @fun/merge.m, @unbndfun/unbndfun.m,
        @unbndfun/feval.m, @onefun/onefun.m, @singfun/singfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.
    Unbounded trials retain global hscale unchanged and construct through
    a fresh union-owned map. No unhappy-smooth-to-Singfun retry is applied.
    """
    from chebfunjax.fun.singfun import Singfun, _find_sing_exponents
    from chebfunjax.fun.unbndfun import Unbndfun

    a, b = left.interval[0], right.interval[1]
    if math.isfinite(a) and math.isfinite(b):
        return _merge_bounded_fun_source(left, right, **options)
    frame = Unbndfun.from_chebtech(
        Chebtech2.from_coeffs(jnp.asarray([0.])), Domain((a, b)))

    def mapped(t):
        return _merge_pair_values(frame.forward_map(t), left, right)

    left_sing = isinstance(left.tech, Singfun)
    right_sing = isinstance(right.tech, Singfun)
    exponents = None
    if left_sing or right_sing:
        exponents = (left.tech.exponents[0] if left_sing else 0.,
                     right.tech.exponents[1] if right_sing else 0.)
        # Source classicfun/get returns stored exponents; unbndfun negates
        # supplied entries at infinite ends, even when they came from onefun.
        exponents = tuple(-e if math.isinf(endpoint) else e
                          for e, endpoint in zip(exponents, (a, b)))
    else:
        # Literal source detection: Inf enables singular detection; NaN alone
        # does not. Preserve both endpoint callback evaluations and their order.
        lval = mapped(jnp.asarray(-1.))
        rval = mapped(jnp.asarray(1.))
        if bool(jnp.any(jnp.isinf(lval))) or bool(jnp.any(jnp.isinf(rval))):
            exponents = _find_sing_exponents(mapped)

    ea, eb = (0., 0.) if exponents is None else exponents
    fit_op = mapped
    fit_tol = options['tol']
    extrapolate = options['splitting']
    if exponents is not None:
        if jnp.asarray(mapped(jnp.asarray(0.))).size > 1:
            raise ValueError('SINGFUN does not support array-valued construction.')
    if ea or eb:
        fit_tol = jnp.maximum(jnp.asarray(fit_tol), 1e-14)
        extrapolate = extrapolate or ea < 0 or eb < 0
        def smooth(t):
            values = mapped(t)
            if ea and eb:
                return values / ((1+t)**ea * (1-t)**eb)
            if ea:
                return values / (1+t)**ea
            return values / (1-t)**eb
        fit_op = smooth
    fit_options = dict(maxpow2=options['maxpow2'], max_length=options['max_length'],
        tol=fit_tol, extrapolate=extrapolate, vscale=options['vscale'],
        hscale=options['hscale'], sample_test=options['sample_test'],
        min_samples=options['min_samples'], turbo=options['turbo'], check=options['check'])
    if options['refinement_function'] is not None:
        fit_options['refinement_function'] = options['refinement_function']
    onefun = Chebtech2.from_function(fit_op, **fit_options)
    if ea or eb:
        onefun = Singfun(onefun, (ea, eb))
    return frame.with_tech(onefun)
