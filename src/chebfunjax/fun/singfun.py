"""Singular function on [-1, 1] — algebraic endpoint singularities.

Translated from MATLAB Chebfun class @singfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)

# Tolerance for considering an exponent "zero" (i.e. smooth at that end)
_EXP_TOL = 1e-11


def _any_empty(*operands) -> bool:
    """True if any operand is an empty Singfun or empty tech.

    MATLAB @singfun arithmetic returns an empty SINGFUN whenever either
    argument is empty; this mirrors that early return.
    """
    for s in operands:
        if isinstance(s, Singfun):
            if s.isempty():
                return True
        elif isinstance(s, (Chebtech1, Chebtech2)):
            if s.isempty():
                return True
    return False


def _demote_if_smooth(s):
    """Demote a smooth Singfun to its bare smooth part.

    Mirrors the closing block that every MATLAB ``@singfun`` arithmetic
    method (``plus``, ``times``, ``rdivide``, ``diff``) shares::

        if ( isa(f, 'singfun') && issmooth(f) )
            f = f.smoothPart;
        end

    A Singfun whose exponents are both (numerically) zero carries no
    singular structure, so the result of the operation is a plain
    ``smoothfun`` (here a :class:`Chebtech2`) rather than a Singfun.
    Non-Singfun inputs (already-demoted results, scalars) pass through.

    Provenance
    ----------
    MATLAB source : @singfun/issmooth.m (all(exponents == 0)) and the
    trailing demotion block in plus.m/times.m/rdivide.m/diff.m
    Chebfun commit: 7574c77
    """
    if isinstance(s, Singfun) and s.issmooth:
        return s.smoothPart
    return s


class Singfun(eqx.Module):
    """Function with algebraic endpoint singularities on [-1, 1].

    Represents a function of the form

        f(x) = s(x) * (1 + x)^a * (1 - x)^b

    on the interval [-1, 1], where ``s(x)`` is a smooth function
    approximated by a :class:`~chebfunjax.tech.chebtech.Chebtech2` and
    ``(a, b)`` are real exponents encoding the algebraic behaviour at the
    left and right endpoints respectively.

    When ``a = b = 0`` the object degenerates to a pure smooth function
    (Chebtech2 wrapper).  Negative exponents represent poles or
    non-integrable singularities; positive non-integer exponents represent
    algebraic blow-down (roots) or integrable singularities.

    Attributes
    ----------
    smoothPart : Chebtech2
        Chebyshev representation of the smooth factor ``s(x)`` on [-1, 1].
    exponents : tuple[float, float]
        ``(a, b)`` — exponents at the left and right endpoints.
        Stored as a static (non-traced) tuple so JAX shape inference is
        unambiguous.

    Notes
    -----
    **JAX contract:**

    * ``f(x)`` — JIT-safe, vmap-safe, grad-safe.
    * ``f.diff(k)`` — construction NOT JIT-safe (returns a new Singfun or
      Chebtech2 via Python-level product rule); result evaluation is JIT-safe.
    * ``f.sum()`` — JIT-safe when exponents are static (they always are in
      the eqx.field sense).

    Provenance
    ----------
    MATLAB source : @singfun/singfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2, Bndfun, Deltafun
    """

    smoothPart: Chebtech2
    exponents: tuple = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, smoothPart, exponents: tuple = (0.0, 0.0)):
        """Low-level constructor.  Prefer :meth:`from_function`.

        Parameters
        ----------
        smoothPart : Chebtech2, Chebtech1, or scalar
            Smooth factor on [-1, 1].  A bare smoothfun (Chebtech) is stored
            directly; a numeric scalar is promoted to the corresponding
            constant smooth part (mirroring MATLAB's ``singfun(smoothfun)`` and
            ``singfun(double)`` upgrades).
        exponents : tuple of two floats, default (0, 0)
            (a, b) — left and right algebraic exponents.
        """
        if not isinstance(smoothPart, (Chebtech1, Chebtech2)):
            const = smoothPart

            def _const(x, _c=const):
                return jnp.full(
                    jnp.shape(x), _c, dtype=jnp.result_type(_c, jnp.float64)
                )

            smoothPart = Chebtech2.from_function(_const)
        self.smoothPart = smoothPart
        self.exponents = (float(exponents[0]), float(exponents[1]))

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        exponents: tuple[float, float] | None = None,
        *,
        n: int | None = None,
        turbo: bool = False,
        maxpow2: int = 16,
        tech_cls=None,
        tol: float | None = None,
        vscale: float = 0.0,
        hscale: float = 1.0,
        extrapolate: bool = False,
    ) -> "Singfun":
        """Construct a Singfun from a callable and (optionally) known exponents.

        Given a function handle ``f`` that evaluates
        ``s(x)*(1+x)^a*(1-x)^b``, this constructor forms the smooth factor

            s(x) = f(x) / ((1+x)^a * (1-x)^b)

        and approximates it with the requested Chebyshev technology
        (Chebtech2 by default).

        Parameters
        ----------
        f : callable
            Vectorised function accepting and returning ``jax.Array``.
            May return ``inf`` or ``nan`` at the endpoints; these are handled
            by the singular factoring.
        exponents : tuple of two floats, optional
            ``(a, b)`` — algebraic exponents at -1 and +1.  When ``None`` the
            exponents are detected automatically by sampling ``f`` near the
            endpoints (see :func:`_find_sing_exponents`).
        n : int or None, optional
            Fixed number of Chebyshev points.  ``None`` triggers adaptive
            construction.
        tol : float or None, optional
            Smooth-part construction tolerance. Nonzero endpoint exponents
            impose the MATLAB constructor's minimum of ``1e-14``.
        vscale : float, optional
            Running global vertical scale for smooth-part construction.
        hscale : float, optional
            Horizontal scale on the reference interval. Bounded physical
            constructors divide the global scale by the piece width.
        extrapolate : bool, optional
            Extrapolate endpoint values on a second-kind grid. Negative
            exponents enable this automatically, as in MATLAB.

        Returns
        -------
        Singfun
            A new Singfun instance.

        Notes
        -----
        The Chebtech2 grid includes the exact endpoints x = ±1. Factoring
        a singularity can give nonfinite values there. The technology
        constructor extrapolates these rows from finite interior samples;
        negative exponents additionally exclude endpoint evaluations.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from chebfunjax.fun.singfun import Singfun
        >>> # sqrt(1 - x^2) = (1+x)^0.5 * (1-x)^0.5 * 1
        >>> sf = Singfun.from_function(
        ...     lambda x: jnp.sqrt(1 - x**2), (0.5, 0.5)
        ... )
        >>> float(sf.sum())   # integral = pi/2
        1.5707963...

        Provenance
        ----------
        MATLAB source : @singfun/singfun.m (constructor)
        Chebfun commit: 7574c77
        """
        if exponents is None:
            exponents = _find_sing_exponents(f)
        a, b = float(exponents[0]), float(exponents[1])
        # MATLAB @singfun/singfun.m loosens the smooth-part tolerance only
        # for a genuinely nonzero endpoint exponent.
        if a != 0.0 or b != 0.0:
            tol = max(1e-14, _EPS if tol is None else float(tol))

        def smooth_f(x: jax.Array) -> jax.Array:
            """Extract the smooth factor s(x) = f(x) / weight(x)."""
            val = f(x)
            if a != 0.0 and b != 0.0:
                val = val / ((1.0 + x) ** a * (1.0 - x) ** b)
            elif a != 0.0:
                val = val / (1.0 + x) ** a
            elif b != 0.0:
                val = val / (1.0 - x) ** b
            return val

        # @singfun/singfun.m chooses pref.tech through constructSmoothPart
        # and enables extrapolation whenever an exponent is negative.
        smooth_cls = Chebtech1 if tech_cls is Chebtech1 else Chebtech2
        tech_options = {}
        if smooth_cls is Chebtech2:
            tech_options["extrapolate"] = extrapolate or a < 0.0 or b < 0.0
        tech = smooth_cls.from_function(
            smooth_f, n=n, turbo=turbo, maxpow2=maxpow2, tol=tol,
            vscale=vscale, hscale=hscale, **tech_options)
        return cls(tech, (a, b))

    @classmethod
    def from_chebtech(cls, tech: Chebtech2, exponents: tuple[float, float]) -> "Singfun":
        """Wrap an existing Chebtech2 in a Singfun with given exponents.

        Parameters
        ----------
        tech : Chebtech2
            Already-constructed smooth factor.
        exponents : tuple of two floats
            ``(a, b)``.

        Returns
        -------
        Singfun

        Provenance
        ----------
        MATLAB source : @singfun/singfun.m
        Chebfun commit: 7574c77
        """
        return cls(tech, exponents)

    @classmethod
    def empty(cls) -> "Singfun":
        """The empty Singfun (MATLAB ``singfun()``): empty smooth part,
        no singularities.  ``isempty()`` is True and arithmetic with it
        propagates emptiness.

        Provenance
        ----------
        MATLAB source : @singfun/singfun.m (nargin == 0), @singfun/isempty.m
        Chebfun commit: 7574c77
        """
        return cls(Chebtech2.empty(), (0.0, 0.0))

    @classmethod
    def zeroSingFun(cls) -> "Singfun":
        """The zero Singfun: zero smooth part, no singularities.

        Provenance
        ----------
        MATLAB source : @singfun/zeroSingFun.m
        Chebfun commit: 7574c77
        """
        return cls(
            Chebtech2.from_coeffs(jnp.zeros(1, dtype=jnp.float64)),
            (0.0, 0.0),
        )

    def isempty(self) -> bool:
        """True when the smooth part is empty (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @singfun/isempty.m
        Chebfun commit: 7574c77
        """
        return self.smoothPart.isempty()

    def iszero(self) -> bool:
        """True when the smooth part is identically zero (MATLAB ``iszero``).

        Provenance
        ----------
        MATLAB source : @singfun/iszero.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return False
        return bool(jnp.all(self.smoothPart.coeffs == 0))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate f(x) = s(x) * (1+x)^a * (1-x)^b at point(s) x in [-1, 1].

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s) in [-1, 1].

        Returns
        -------
        jax.Array, same shape as x
            Function values.

        Notes
        -----
        JIT-safe, vmap-safe, grad-safe.

        Provenance
        ----------
        MATLAB source : @singfun/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x)
        x = x.astype(jnp.result_type(x, jnp.float64))
        val = self.smoothPart(x)
        a, b = self.exponents
        if a != 0.0:
            val = val * (1.0 + x) ** a
        if b != 0.0:
            val = val * (1.0 - x) ** b
        return val

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients in the smooth part."""
        return self.smoothPart.n

    @property
    def coeffs(self) -> jax.Array:
        """Chebyshev coefficients of the smooth part."""
        return self.smoothPart.coeffs

    @property
    def vscale(self) -> float:
        """Vertical scale of the smooth part.

        The full Singfun blows up where an exponent is negative, so a finite
        magnitude scale is taken from the smooth factor ``s(x)`` (mirroring
        MATLAB, which reads the SINGFUN vscale off its ``smoothPart``).

        Provenance
        ----------
        MATLAB source : @singfun/singfun.m (vscale via smoothPart)
        Chebfun commit: 7574c77
        """
        return self.smoothPart.vscale

    @property
    def ishappy(self) -> bool:
        """Happiness of the smooth part (fun-level delegation)."""
        return self.smoothPart.ishappy

    @property
    def values(self) -> jax.Array:
        """Values of the SMOOTH PART at its Chebyshev grid.

        The full singular function's endpoint values may be infinite; fun
        wrappers that need f-values should evaluate ``self(x)`` instead.
        """
        return self.smoothPart.values

    @property
    def issmooth(self) -> bool:
        """True if both exponents are (numerically) zero."""
        a, b = self.exponents
        return abs(a) < _EXP_TOL and abs(b) < _EXP_TOL

    def __len__(self) -> int:
        return self.n

    def normest(self):
        """Estimate the norm from the smooth factor, as MATLAB does.

        Provenance
        ----------
        MATLAB source: @singfun/normest.m, Chebfun7574c77.
        """
        return self.smoothPart.normest()

    def isequal(self, other) -> bool:
        """Compare source exponents and exact smooth-part coefficients.

        Provenance
        ----------
        MATLAB source: @singfun/isequal.m, @chebfunpref/chebfunpref.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        The default exponent tolerance is strictly 1.1e-11; coefficients
        and lengths must match exactly through the smooth-part predicate.
        """
        if isinstance(other, (Chebtech1, Chebtech2)):
            other = Singfun(other)
        if not isinstance(other, Singfun):
            return False
        tol = 1.1e-11
        if not all(abs(a - b) < tol for a, b in zip(self.exponents, other.exponents)):
            return False
        return self.smoothPart.isequal(other.smoothPart)

    def __eq__(self, other) -> bool:
        """Use the MATLAB isequal predicate for Python equality."""
        if not isinstance(other, (Singfun, Chebtech1, Chebtech2)):
            return NotImplemented
        return self.isequal(other)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        """Compact display.

        Examples
        --------
        >>> sf = Singfun.from_function(lambda x: jnp.sqrt(1-x**2), (0.5, 0.5))
        >>> repr(sf)
        'Singfun([-1, 1], n=1, exps=(0.5, 0.5))'
        """
        a, b = self.exponents
        return f"Singfun([-1, 1], n={self.n}, exps=({a}, {b}))"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __mul__(self, other) -> "Singfun":
        """Pointwise multiplication: f * g or f * scalar.

        When multiplying two Singfuns, the exponents are added and the smooth
        parts are multiplied.

        Provenance
        ----------
        MATLAB source : @singfun/times.m
        Chebfun commit: 7574c77
        """
        if _any_empty(self, other):
            return Singfun.empty()
        if isinstance(other, Singfun):
            new_smoothPart = self.smoothPart * other.smoothPart
            new_exps = (
                self.exponents[0] + other.exponents[0],
                self.exponents[1] + other.exponents[1],
            )
            h = Singfun(new_smoothPart, new_exps)
        elif isinstance(other, Chebtech2):
            h = Singfun(self.smoothPart * other, self.exponents)
        else:
            h = Singfun(self.smoothPart * other, self.exponents)
        # Source cancellation precedes simplification and smooth demotion.
        # A cancelled pole must not survive as a singular representation:
        # downstream unbounded integration selects its algorithm by this type.
        return _demote_if_smooth(h.cancelExponents().simplify())

    def __rmul__(self, other) -> "Singfun":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Singfun":
        """Division: f / g or f / scalar.

        Provenance
        ----------
        MATLAB source : @singfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if _any_empty(self, other):
            return Singfun.empty()
        if isinstance(other, Singfun):
            # If the divisor's smooth part vanishes at an endpoint, dividing
            # the smooth parts directly would create a new endpoint pole that
            # the naive quotient cannot represent.  Follow MATLAB's rdivide and
            # rebuild the quotient through the constructor, detecting the new
            # exponents automatically.
            gl = float(jnp.abs(other.smoothPart(jnp.float64(-1.0))))
            gr = float(jnp.abs(other.smoothPart(jnp.float64(1.0))))
            tol = 1e2 * other.smoothPart.vscale * _EPS
            if gl > tol and gr > tol:
                new_smoothPart = self.smoothPart / other.smoothPart
                new_exps = (
                    self.exponents[0] - other.exponents[0],
                    self.exponents[1] - other.exponents[1],
                )
                s = Singfun(new_smoothPart, new_exps)
            else:
                f_self, f_other = self, other

                def _quot(x, _f=f_self, _g=f_other):
                    return _f(x) / _g(x)

                s = Singfun.from_function(_quot)

            # MATLAB @singfun/rdivide (lines 87-88): cancel boundary roots
            # against negative exponents, then absorb integer exponents.
            # Then demote a smooth quotient to its bare smoothfun.
            return _demote_if_smooth(s.cancelExponents().simplify())
        else:
            return _demote_if_smooth(
                Singfun(self.smoothPart / other, self.exponents)
            )

    def __rtruediv__(self, other) -> "Singfun":
        """scalar / Singfun."""
        if _any_empty(self, other):
            return Singfun.empty()
        new_smoothPart = other / self.smoothPart
        new_exps = (-self.exponents[0], -self.exponents[1])
        # MATLAB @singfun/rdivide canonicalises the reciprocal's exponents:
        # absorb integer parts (>= 1) into the smooth part and cancel any
        # boundary roots, so e.g. 1/((1+x)^-3 (1-x)^-4) has exponents (0, 0).
        return _demote_if_smooth(
            Singfun(new_smoothPart, new_exps).cancelExponents().simplify()
        )

    def __add__(self, other) -> "Singfun":
        """Addition: f + g where g is a Singfun, Chebtech2, or scalar.

        When the two operands share the same exponents the smooth parts are
        added directly.  When the exponents differ by integers the result can
        still be expressed as a Singfun by factoring out the more-singular
        exponent (Case 2 in MATLAB Chebfun).  Otherwise a warning accompanies the
        approximation is used by evaluating the sum pointwise and
        re-constructing (Case 3).  A result whose exponents are both zero is
        demoted to its bare smooth part, mirroring the ``issmooth`` block that
        closes MATLAB ``@singfun/plus.m``.

        Provenance
        ----------
        MATLAB source : @singfun/plus.m
        Chebfun commit: 7574c77
        """
        # MATLAB @singfun/plus (lines 29-33, 41-45): adding an exact zero
        # returns the other operand unchanged (an early return that bypasses
        # the trailing issmooth demotion), so keep it here rather than in the
        # demoting ``_plus`` helper.  Without this short-circuit a ``g - 0``
        # (e.g. cumsum's F(-1)=0 shift when lval == 0 at a positive fractional
        # root) falls through to the Case-3 pointwise reconstruction,
        # which takes ``new_a = min(exp, 0) = 0`` and silently collapses the
        # exponent — turning a one-coefficient singular function into a
        # 362-coefficient polynomial with ~5e-11 evaluation error.
        if _any_empty(self, other):
            return Singfun.empty()
        if not isinstance(other, (Singfun, Chebtech2)) and jnp.ndim(other) == 0:
            if other == 0:
                return self

        # Source Case 3 zero-SINGFUN returns precede the final smooth
        # demotion too. Keep those public return types/objects intact; the
        # private _plus helper still returns raw Singfuns for differentiation.
        if not isinstance(other, Singfun):
            other = Singfun(other, (0.0, 0.0))
        exponent_differences = tuple(
            f - g for f, g in zip(self.exponents, other.exponents)
        )
        if not all(abs(round(d) - d) < _EXP_TOL for d in exponent_differences):
            if self.iszero():
                return other
            if other.iszero():
                return self
        return _demote_if_smooth(self._plus(other))

    def _plus(self, other) -> "Singfun":
        """Raw Singfun addition without the trailing smooth-part demotion.

        Splits out the Case 1/2/3 arithmetic of :meth:`__add__` so callers
        that must keep a Singfun result mid-computation (e.g. the product
        rule in :meth:`diff`, where an intermediate all-zero-exponent sum
        must NOT be demoted before the next differentiation step) can reuse
        it.  Public addition (:meth:`__add__`) wraps this in
        :func:`_demote_if_smooth`.

        Provenance
        ----------
        MATLAB source : @singfun/plus.m (Cases 1-3, pre-issmooth)
        Chebfun commit: 7574c77
        """
        # Upgrade scalar (real or complex) / Chebtech2 to a Singfun with zero
        # exponents.  Singfun.__init__ promotes a bare scalar to a constant
        # smooth part.
        if not isinstance(other, Singfun):
            other = Singfun(other, (0.0, 0.0))

        fExps = self.exponents
        gExps = other.exponents

        # Case 1: identical exponents — just add smooth parts
        if abs(fExps[0] - gExps[0]) < _EXP_TOL and abs(fExps[1] - gExps[1]) < _EXP_TOL:
            return Singfun(self.smoothPart + other.smoothPart, fExps)

        # Case 2: exponents differ by integers — factor out the smaller
        # exponent and add the resulting smooth parts
        diff0 = fExps[0] - gExps[0]
        diff1 = fExps[1] - gExps[1]
        if abs(round(diff0) - diff0) < _EXP_TOL and abs(round(diff1) - diff1) < _EXP_TOL:
            # New exponents: take the algebraically smaller at each end
            new_a = min(fExps[0], gExps[0])
            new_b = min(fExps[1], gExps[1])

            # Extra polynomial factors to compensate
            d0_f = fExps[0] - new_a  # >= 0
            d0_g = gExps[0] - new_a  # >= 0
            d1_f = fExps[1] - new_b  # >= 0
            d1_g = gExps[1] - new_b  # >= 0

            def _make_weight(da, db):
                def w(x, _da=da, _db=db):
                    v = jnp.ones_like(x, dtype=jnp.float64)
                    if _da != 0.0:
                        v = v * (1.0 + x) ** _da
                    if _db != 0.0:
                        v = v * (1.0 - x) ** _db
                    return v
                return w

            # Compensate BOTH operands for their excess exponent over the
            # common (new_a, new_b).  Compensating only one operand (an earlier
            # bug) is wrong whenever the exponents "cross" — i.e. f is the more
            # singular at one end and g at the other — which is exactly what
            # diff() of a both-endpoint-singular function produces.
            if d0_f != 0.0 or d1_f != 0.0:
                sp_f = self.smoothPart * Chebtech2.from_function(
                    _make_weight(d0_f, d1_f)
                )
            else:
                sp_f = self.smoothPart
            if d0_g != 0.0 or d1_g != 0.0:
                sp_g = other.smoothPart * Chebtech2.from_function(
                    _make_weight(d0_g, d1_g)
                )
            else:
                sp_g = other.smoothPart

            return Singfun(sp_f + sp_g, (new_a, new_b))

        # Case 3: source plus.m reconstructs using the original operand
        # scales. This scale matters when cancellation leaves a small result.
        # Delegate endpoint factoring/extrapolation to the constructor rather
        # than shifting near-endpoint samples by sqrt(eps).
        if self.iszero():
            return other
        if other.iszero():
            return self
        warnings.warn(
            "CHEBFUN:SINGFUN:plus:exponentDiff: "
            "Non-integer difference in the exponents of the two SINGFUN "
            "objects: The result may not be accurate.",
            UserWarning, stacklevel=3,
        )
        new_exponents = (
            min(fExps[0], gExps[0]), min(fExps[1], gExps[1])
        )
        return Singfun.from_function(
            lambda x: self(x) + other(x), new_exponents,
            vscale=self.vscale + other.vscale,
        )

    def __radd__(self, other) -> "Singfun":
        return self.__add__(other)

    def __sub__(self, other) -> "Singfun":
        """Subtraction.

        Provenance
        ----------
        MATLAB source : @singfun/minus.m
        Chebfun commit: 7574c77
        """
        return self.__add__(-other)

    def __rsub__(self, other) -> "Singfun":
        return (-self).__add__(other)

    def __neg__(self) -> "Singfun":
        """Unary negation.

        Provenance
        ----------
        MATLAB source : @singfun/uminus.m
        Chebfun commit: 7574c77
        """
        return Singfun(-self.smoothPart, self.exponents)

    def __pos__(self) -> "Singfun":
        return Singfun(self.smoothPart, self.exponents)

    def __pow__(self, p) -> "Singfun":
        """Raise to a real power p.

        f^p = s^p * (1+x)^(a*p) * (1-x)^(b*p)

        Provenance
        ----------
        MATLAB source : @singfun/power.m
        Chebfun commit: 7574c77
        """
        normalized = self.extractBoundaryRoots()
        a, b = normalized.exponents
        # Empty-object propagation is a Python adapter; the pinned MATLAB
        # power/compose path has no separately qualified empty-power fixture.
        smooth = normalized.smoothPart
        if smooth.isempty():
            return Singfun(smooth, (a * p, b * p))
        if isinstance(smooth, Chebtech2) and jnp.iscomplexobj(smooth.coeffs):
            # @chebtech/power.m avoids complex branch-cut ambiguity at
            # endpoints by composing with pref.extrapolate=true.
            powered = smooth.compose(lambda value: jnp.power(value, p),
                                     extrapolate=True)
        else:
            powered = smooth ** p
        return Singfun(powered, (a * p, b * p)).simplifyExponents()

    # ------------------------------------------------------------------
    # Exponent canonicalisation
    # ------------------------------------------------------------------

    def extractBoundaryRoots(self, num_roots=None) -> "Singfun":
        """Absorb boundary roots of the smooth part into the exponents.

        ``num_roots`` is an optional ``(left, right)`` pair of target
        multiplicities; ``None`` extracts every boundary root automatically.

        Provenance
        ----------
        MATLAB source : @singfun/extractBoundaryRoots.m
        Chebfun commit: 7574c77
        """
        if num_roots is None:
            nl = nr = None
        else:
            nl, nr = float(num_roots[0]), float(num_roots[1])
        new_sp, rL, rR = _extract_boundary_roots_coeffs(self.smoothPart, nl, nr)
        return Singfun(
            new_sp, (self.exponents[0] + rL, self.exponents[1] + rR)
        )

    def isnan(self) -> bool:
        """Check both the smooth part and singular endpoint evaluations.

        Provenance
        ----------
        MATLAB source : @singfun/isnan.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return False
        return self.smoothPart.isnan() or bool(
            jnp.any(jnp.isnan(self(jnp.asarray([-1.0, 1.0]))))
        )

    def cancelExponents(self) -> "Singfun":
        """Cancel negative exponents against vanishing boundary values.

        Where an exponent is negative and the smooth part vanishes at that
        endpoint, the offending boundary root is peeled off and the exponent
        incremented toward zero.

        Provenance
        ----------
        MATLAB source : @singfun/cancelExponents.m
        Chebfun commit: 7574c77
        """
        a, b = self.exponents
        tol = 100.0 * _EPS * float(self.smoothPart.vscale)
        bl = float(jnp.abs(self.smoothPart(jnp.float64(-1.0))))
        br = float(jnp.abs(self.smoothPart(jnp.float64(1.0))))
        nl = -a if (a < 0.0 and abs(bl) < tol) else 0.0
        nr = -b if (b < 0.0 and abs(br) < tol) else 0.0
        if nl > 0.0 or nr > 0.0:
            return self.extractBoundaryRoots((nl, nr))
        return self

    def simplifyExponents(self) -> "Singfun":
        """Reduce exponents to ``< 1`` by absorbing integer parts into the smooth part.

        Provenance
        ----------
        MATLAB source : @singfun/simplifyExponents.m
        Chebfun commit: 7574c77
        """
        tol = 100.0 * _EPS * float(self.smoothPart.vscale)
        exps = [float(e) for e in self.exponents]
        # Snap near-zero and near-integer exponents.
        exps = [0.0 if abs(e) < tol else e for e in exps]
        exps = [round(e) if abs(round(e) - e) < tol else e for e in exps]
        ind = [e >= 1.0 - tol for e in exps]
        if not any(ind):
            return Singfun(self.smoothPart, (exps[0], exps[1]))
        new_exps = [
            e - math.floor(e) if ind[i] else e for i, e in enumerate(exps)
        ]
        pow0 = exps[0] - new_exps[0]
        pow1 = exps[1] - new_exps[1]

        def _mult(x, _p0=pow0, _p1=pow1):
            v = jnp.ones_like(jnp.asarray(x, dtype=jnp.float64))
            if _p0 != 0.0:
                v = v * (1.0 + x) ** _p0
            if _p1 != 0.0:
                v = v * (1.0 - x) ** _p1
            return v

        mult = Chebtech2.from_function(_mult)
        return Singfun(self.smoothPart * mult, (new_exps[0], new_exps[1]))

    def simplify(self, tol: float | None = None) -> "Singfun":
        """Simplify the smooth part and canonicalise the exponents to ``< 1``.

        Provenance
        ----------
        MATLAB source : @singfun/simplify.m
        Chebfun commit: 7574c77
        """
        sp = self.smoothPart.simplify() if tol is None else self.smoothPart.simplify(tol)
        return Singfun(sp, self.exponents).simplifyExponents()

    # ------------------------------------------------------------------
    # Reflection and complex parts
    # ------------------------------------------------------------------

    def flipud(self) -> "Singfun":
        """Reflect the function: ``g(x) = f(-x)`` for x in [-1, 1].

        The smooth part is reflected and the two endpoint exponents are
        swapped.

        Provenance
        ----------
        MATLAB source : @singfun/flipud.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        a, b = self.exponents
        return Singfun(self.smoothPart.flipud(), (b, a))

    def real(self):
        """Real part of ``f``.

        Returns a :class:`~chebfunjax.tech.chebtech.Chebtech2` (the bare
        smooth part) when the result is smooth, otherwise a Singfun, matching
        MATLAB's demotion of a smooth SINGFUN to a SMOOTHFUN.

        Provenance
        ----------
        MATLAB source : @singfun/real.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        if self.issmooth:
            if jnp.iscomplexobj(self.smoothPart.coeffs):
                return self.smoothPart.real()
            return self.smoothPart
        return Singfun(self.smoothPart.real(), self.exponents)

    def imag(self):
        """Imaginary part of ``f``.

        Provenance
        ----------
        MATLAB source : @singfun/imag.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        if self.issmooth:
            return self.smoothPart.imag()
        return Singfun(self.smoothPart.imag(), self.exponents)

    def fliplr(self):
        """Reverse the columns of the smooth part (identity for a scalar).

        ``fliplr`` acts on the array (column) dimension, not the x-axis, so the
        endpoint exponents are unchanged.

        Provenance
        ----------
        MATLAB source : @singfun/fliplr.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        return Singfun(self.smoothPart.fliplr(), self.exponents)

    def conj(self):
        """Complex conjugate of ``f``.

        Provenance
        ----------
        MATLAB source : @singfun/conj.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        if self.issmooth:
            if jnp.iscomplexobj(self.smoothPart.coeffs):
                return self.smoothPart.conj()
            return self.smoothPart
        return Singfun(self.smoothPart.conj(), self.exponents)

    # ------------------------------------------------------------------
    # Factory and composition
    # ------------------------------------------------------------------

    def make(self, op, exponents=None, singType=None, pref=None) -> "Singfun":
        """Factory shortcut: build a Singfun from ``op``.

        Mirrors MATLAB's ``@singfun/make`` (a factory method used so that
        ONEFUN-level code can construct a SINGFUN without naming the class
        directly).  ``singType`` and ``pref`` are accepted for signature
        compatibility but ignored — they are redundant hints in chebfunjax.

        Provenance
        ----------
        MATLAB source : @singfun/make.m
        Chebfun commit: 7574c77
        """
        return Singfun.from_function(op, exponents)

    def compose(self, op, g=None) -> "Singfun":
        """Compose: ``op(f)`` or ``op(f, g)``.

        ``op`` is a callable.  With ``g`` supplied, the two-argument operator
        ``op(f(x), g(x))`` is formed; otherwise the single-argument
        ``op(f(x))`` is formed (if ``op`` is itself a Singfun or Chebtech2 it
        is treated as the outer function ``op(f(x))``).  The exponents of the
        result are detected automatically.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @singfun/compose.m
        Chebfun commit: 7574c77
        """
        if g is None:
            if isinstance(op, (Singfun, Chebtech2)):
                outer = op

                def new_op(x):
                    return outer(self(x))
            else:

                def new_op(x):
                    return op(self(x))
        else:

            def new_op(x):
                return op(self(x), g(x))

        return Singfun.from_function(new_op)

    # ------------------------------------------------------------------
    # Rootfinding and extrema
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """Real roots of ``f`` in [-1, 1].

        The interior roots are those of the smooth part.  A positive exponent
        at an endpoint contributes an extra root there (the singular factor
        ``(1 +/- x)^e`` vanishes when ``e > 0``).

        NOT JIT-safe (variable output size).

        Provenance
        ----------
        MATLAB source : @singfun/roots.m
        Chebfun commit: 7574c77
        """
        tol = _EPS
        out = [float(r) for r in self.smoothPart.roots()]
        any_roots = len(out) > 0
        a, b = self.exponents

        if a > 0.0:
            if any_roots:
                if abs(1.0 + out[0]) < tol:
                    out[0] = -1.0
                else:
                    out = [-1.0] + out
            else:
                out = [-1.0]

        if b > 0.0:
            if any_roots:
                if abs(1.0 - out[-1]) < tol:
                    out[-1] = 1.0
                else:
                    out = out + [1.0]
            else:
                out = out + [1.0]

        return jnp.asarray(out, dtype=jnp.float64)

    def minandmax(self):
        """Global minimum and maximum of ``f`` on [-1, 1].

        Returns
        -------
        vals : jax.Array, shape (2,)
            ``[min, max]``.
        pos : jax.Array, shape (2,)
            Positions where the min and max are attained.

        At a negative endpoint exponent the function blows up; the
        corresponding extreme value is ``+/- inf`` at that endpoint.
        Otherwise the extrema are found among the endpoints and the roots of
        ``f'``.

        NOT JIT-safe (depends on rootfinding).

        Provenance
        ----------
        MATLAB source : @singfun/minandmax.m
        Chebfun commit: 7574c77
        """
        tol = _EXP_TOL
        a, b = self.exponents

        if abs(a) < tol and abs(b) < tol:
            (min_val, min_pos), (max_val, max_pos) = self.smoothPart.minandmax()
            vals = jnp.asarray([min_val, max_val], dtype=jnp.float64)
            pos = jnp.asarray([min_pos, max_pos], dtype=jnp.float64)
            return vals, pos

        minF = maxF = minLoc = maxLoc = None

        if a < -tol:  # singularity at the left end
            fval = float(self(jnp.float64(-1.0)))
            if fval == jnp.inf:
                maxF, maxLoc = jnp.inf, -1.0
            elif fval == -jnp.inf:
                minF, minLoc = -jnp.inf, -1.0
            else:
                raise ValueError(
                    "Function has a singularity but isn't infinite at the left "
                    "endpoint."
                )

        if b < -tol:  # singularity at the right end
            fval = float(self(jnp.float64(1.0)))
            if fval == jnp.inf:
                maxF, maxLoc = jnp.inf, 1.0
            elif fval == -jnp.inf:
                minF, minLoc = -jnp.inf, 1.0
            else:
                raise ValueError(
                    "Function has a singularity but isn't infinite at the right "
                    "endpoint."
                )

        if minF is None or maxF is None:
            fp = self.diff()
            r = [float(x) for x in fp.roots()]
            r = sorted(set([-1.0] + r + [1.0]))
            rr = jnp.asarray(r, dtype=jnp.float64)
            fr = self(rr)
            if maxF is None:
                idx = int(jnp.argmax(fr))
                maxF, maxLoc = float(fr[idx]), r[idx]
            if minF is None:
                idx = int(jnp.argmin(fr))
                minF, minLoc = float(fr[idx]), r[idx]

        vals = jnp.asarray([minF, maxF], dtype=jnp.float64)
        pos = jnp.asarray([minLoc, maxLoc], dtype=jnp.float64)
        return vals, pos

    def restrict(self, s):
        """Restrict ``f`` to subinterval(s) of [-1, 1].

        ``s`` is an increasing sequence in [-1, 1].  With two entries the
        result is a single restricted piece; with more, a list of pieces (one
        per subinterval).  A piece that abuts an endpoint carrying a nonzero
        exponent is returned as a :class:`Singfun`; interior pieces are
        returned as bare :class:`~chebfunjax.tech.chebtech.Chebtech2` smooth
        functions.

        NOT JIT-safe (construction-level operation).

        Provenance
        ----------
        MATLAB source : @singfun/restrict.m
        Chebfun commit: 7574c77
        """
        s = [float(v) for v in s]
        if (
            s[0] < -1.0 - _EPS
            or s[-1] > 1.0 + _EPS
            or any(s[i + 1] - s[i] <= 0 for i in range(len(s) - 1))
        ):
            raise ValueError("Not a valid interval.")
        if len(s) == 2 and s[0] == -1.0 and s[1] == 1.0:
            return self

        num_ints = len(s) - 1
        a, b = self.exponents
        g = [None] * num_ints

        for j in range(num_ints):
            # Piece abutting the left endpoint with a nonzero left exponent.
            if s[j] == -1.0 and a != 0.0:
                base = ((1.0 + s[1]) / 2.0) ** a * self.smoothPart.restrict(s[0], s[1])
                if b == 0.0:
                    sp = base
                else:
                    s2 = s[1]
                    extra = Chebtech2.from_function(
                        lambda x, _s2=s2, _b=b: (
                            1.0 - _s2 * (x + 1.0) / 2.0 - (x - 1.0) / 2.0
                        )
                        ** _b
                    )
                    sp = base * extra
                g[j] = Singfun(sp, (a, 0.0))
                continue

            # Piece abutting the right endpoint with a nonzero right exponent.
            if s[j + 1] == 1.0 and b != 0.0:
                base = ((1.0 - s[-2]) / 2.0) ** b * self.smoothPart.restrict(
                    s[-2], s[-1]
                )
                if a == 0.0:
                    sp = base
                else:
                    sm2 = s[-2]
                    extra = Chebtech2.from_function(
                        lambda x, _sm2=sm2, _a=a: (
                            _sm2 * (1.0 - x) / 2.0 + (x + 1.0) / 2.0 + 1.0
                        )
                        ** _a
                    )
                    sp = base * extra
                g[j] = Singfun(sp, (0.0, b))
                continue

            # Interior piece — a plain smooth function.
            sj, sj1 = s[j], s[j + 1]

            def op(x, _sj=sj, _sj1=sj1):
                return self(((1.0 - x) * _sj + (1.0 + x) * _sj1) / 2.0)

            g[j] = Chebtech2.from_function(op)

        if num_ints == 1:
            return g[0]
        return g

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "Singfun":
        """Differentiate *k* times.

        Uses the product rule iteratively:

            d/dx [s(x) (1+x)^a (1-x)^b]
            = s'(x) (1+x)^a (1-x)^b
            + a s(x) (1+x)^(a-1) (1-x)^b
            - b s(x) (1+x)^a (1-x)^(b-1)

        Each iteration may increase the number of terms, but all terms share
        the same singular structure so they can be collected back into a
        single Singfun.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Singfun or Chebtech2
            The *k*-th derivative.  If all exponents of the result are zero
            the derivative is smooth and is demoted to a bare
            :class:`Chebtech2` (mirroring the ``issmooth`` block that closes
            MATLAB ``@singfun/diff.m``); otherwise a Singfun is returned.

        Notes
        -----
        NOT JIT-safe at the Python level (creates new objects via Python
        control flow).  The returned object's ``__call__`` method IS JIT-safe.

        Provenance
        ----------
        MATLAB source : @singfun/diff.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Singfun.empty()
        if k == 0:
            return Singfun(self.smoothPart, self.exponents)

        f = Singfun(self.smoothPart, self.exponents)
        for _ in range(k):
            a, b = f.exponents

            # First term: s'(x) * (1+x)^a * (1-x)^b
            s_term = Singfun(f.smoothPart.diff(), (a, b))

            # Second term: a * s(x) * (1+x)^(a-1) * (1-x)^b
            # Use the raw (non-demoting) ``_plus`` so an intermediate all-zero
            # exponent sum stays a Singfun for the next iteration; e.g. an
            # integer exponent a == 1 makes s_term + a_term smooth, and
            # demoting it to a Chebtech2 mid-loop would break ``f.exponents``
            # on the next differentiation step (k > 1).
            if abs(a) > _EXP_TOL:
                a_term = Singfun(f.smoothPart * a, (a - 1.0, b))
                s_term = s_term._plus(a_term)

            # Third term: -b * s(x) * (1+x)^a * (1-x)^(b-1)
            if abs(b) > _EXP_TOL:
                b_term = Singfun(f.smoothPart * (-b), (a, b - 1.0))
                s_term = s_term._plus(b_term)

            f = s_term

        # MATLAB @singfun/diff.m (lines 73-74): demote a smooth derivative
        # (all-zero exponents) to its bare smoothfun/Chebtech2.
        return _demote_if_smooth(f)

    def chebcoeffs(self, N: int, kind: int = 1) -> jax.Array:
        r"""First ``N`` Chebyshev coefficients of the singular function ``f``.

        Returns the coefficients :math:`a_0, \ldots, a_{N-1}` such that
        :math:`f \approx \sum_k a_k T_k` (``kind=1``) or in terms of
        second-kind polynomials :math:`U_k` (``kind=2``).  Because ``f`` is
        singular the series is infinite; the leading ``N`` coefficients are
        obtained from the Chebyshev-weighted Jacobi moments of the smooth
        part, evaluated in coefficient space via fast Toeplitz/Hankel
        products.

        Parameters
        ----------
        N : int
            Number of coefficients to return.
        kind : int, default 1
            Chebyshev polynomial kind (1 or 2).

        Returns
        -------
        jax.Array, shape (N,)

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @singfun/chebcoeffs.m
        Chebfun commit: 7574c77
        """
        a, b = self.exponents
        if a <= -0.5 or b <= -0.5:
            raise ValueError(
                "F does not have a well-defined Chebyshev expansion "
                "(exponents must exceed -1/2)."
            )
        if kind == 2:
            cT = self.chebcoeffs(N + 2, kind=1)
            return _chebT2U(cT)[:N]
        if kind != 1:
            raise ValueError("'kind' must be 1 or 2.")

        n = self.n
        coeffs = jnp.asarray(self.smoothPart.coeffs, dtype=jnp.float64)
        L = n + N - 1
        w = _jacobi_moments(a - 0.5, b - 0.5, L)
        bvec = 0.5 * jnp.concatenate(
            [coeffs, jnp.zeros(N - 1, dtype=jnp.float64)]
        )
        out = (
            _fast_toeplitz_mult(bvec, w)
            + _fast_hankel_mult(bvec, w)
            - w[0] * bvec
        )
        out = jnp.real(out[:N])
        out = out.at[0].divide(2.0)
        return (2.0 / jnp.pi) * out

    def sum(self) -> jax.Array:
        r"""Definite integral :math:`\int_{-1}^{1} f(x)\,dx`.

        Uses the Chebyshev–Jacobi moment formula:

        .. math::

            \int_{-1}^{1} s(x)(1+x)^a(1-x)^b\,dx
            = \sum_{r=0}^{n-1} c_r M_r

        where :math:`M_r = \int_{-1}^{1} (1+x)^a(1-x)^b T_r(x)\,dx` are the
        *modified moments* of the Jacobi weight.

        **Gegenbauer case** (``a == b``):

        .. math::

            M_0 = \sqrt{\pi}\,\Gamma(a+1)/\Gamma(a+3/2),\quad
            M_{2k} = M_0 \prod_{j=1}^k \frac{j - a - 1}{j + a},\quad
            M_{2k+1} = 0

        **General case** (``a ≠ b``, Sister Celine recurrence):

        .. math::

            M_0 = 2^{a+b+1} B(a+1, b+1),\quad
            M_1 = \frac{a-b}{a+b+2} M_0,\quad
            M_j = \frac{2(a-b)M_{j-1} + (j-2 - a - b - 1)M_{j-2}}{a+b+j}

        Returns
        -------
        jax.Array, scalar float64
            The definite integral.

        Notes
        -----
        When both exponents are ``<= -1`` the integral is divergent and
        ``+inf``, ``-inf``, or ``nan`` is returned as appropriate.

        JIT-safe: YES (exponents are static).

        Provenance
        ----------
        MATLAB source : @singfun/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            [1] K. Xu and M. Javed, Singfun Working Note, August 2013
            [2] Hunter & Nikolov, "Gaussian Quadrature of Chebyshev
                Polynomials", J. Comput. Appl. Math. 94, 1998.
            [3] Piessens & Branders, "The Evaluation and Application of
                Some Modified Moments", BIT 13, 1973.
            [4] Sommariva, "Fast construction of Fejer and Clenshaw-Curtis
                rules for general weight functions", Comp. Math. Appl. 65,
                2012.

        See Also
        --------
        Singfun.diff, Chebtech2.sum
        """
        a, b = self.exponents

        # Trivial case: no singularity
        if abs(a) < _EXP_TOL and abs(b) < _EXP_TOL:
            return self.smoothPart.sum()

        # Divergent cases
        if a <= -1.0 and b <= -1.0:
            # Both endpoints diverge
            lval = float(self.smoothPart(jnp.float64(-1.0)))
            rval = float(self.smoothPart(jnp.float64(1.0)))
            sl = math.copysign(1.0, lval)
            sr = math.copysign(1.0, rval)
            if sl == sr:
                return jnp.array(sl * jnp.inf, dtype=jnp.float64)
            else:
                return jnp.array(jnp.nan, dtype=jnp.float64)

        if a <= -1.0:
            lval = float(self.smoothPart(jnp.float64(-1.0)))
            sl = math.copysign(1.0, lval)
            return jnp.array(sl * jnp.inf, dtype=jnp.float64)

        if b <= -1.0:
            rval = float(self.smoothPart(jnp.float64(1.0)))
            sr = math.copysign(1.0, rval)
            return jnp.array(sr * jnp.inf, dtype=jnp.float64)

        # Non-trivial integrable case: compute Jacobi moments
        n = self.n
        coeffs = self.smoothPart.coeffs  # shape (n,)

        M = _jacobi_moments(a, b, n)  # shape (n,)
        return jnp.dot(M, coeffs)

    def inner(self, other) -> jax.Array:
        r"""L2 inner product :math:`\langle f, g\rangle = \int_{-1}^{1}
        \overline{f(x)}\,g(x)\,dx` (conjugate-linear in ``f``).

        Mirrors MATLAB ``@singfun/innerProduct.m``, which is literally
        ``sum(conj(f).*g)``.  The conjugated left operand is built directly
        as ``Singfun(self.smoothPart.conj(), self.exponents)`` rather than via
        :meth:`conj` so that a trivial-exponent operand is *not* demoted to a
        bare Chebtech2 before the product.  A smooth (Chebtech2 or scalar)
        partner is promoted to a zero-exponent Singfun.  Scalar (single
        column) operands only, which matches every chebfunjax Singfun.

        Parameters
        ----------
        other : Singfun, Chebtech2, or scalar
            Right operand ``g``.

        Returns
        -------
        jax.Array, scalar
            The (possibly complex) inner product.

        Provenance
        ----------
        MATLAB source : @singfun/innerProduct.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        cf = Singfun(self.smoothPart.conj(), self.exponents)
        if not isinstance(other, Singfun):
            other = Singfun.from_chebtech(other, (0.0, 0.0))
        return (cf * other).sum()

    innerProduct = inner

    def cumsum(self):
        """Native scalar antiderivative, or two locally mapped halves.

        The two-item list represents [-1, 0] and [0, 1], with each item
        evaluated in its own [-1, 1] coordinate. The right item includes
        the left integral's endpoint constant. Logarithmic primitives
        retain the native unsupported error.

        Construction is not JIT-safe.

        Provenance: @singfun/cumsum.m, Chebfun 7574c77.
        """
        a, b = self.exponents
        # Native issmooth also recognizes a zero smoothPart.
        if (a == 0 and b == 0) or bool(jnp.all(self.smoothPart.coeffs == 0)):
            return self.smoothPart.cumsum()
        if a != 0 and b != 0:
            left, right = self.restrict([-1.0, 0.0, 1.0])
            g_left = _sing_cumsum(left) / 2.0
            g_right = _sing_cumsum(right) / 2.0
            return [g_left, g_right + g_left(jnp.float64(1.0))]
        return _sing_cumsum(self)


# ======================================================================
# Private helpers
# ======================================================================


def _jacobi_moments(a: float, b: float, n: int) -> jax.Array:
    r"""Compute the first *n* modified moments of the Jacobi weight (1+x)^a (1-x)^b.

    Returns the vector :math:`M_0, M_1, \ldots, M_{n-1}` where

    .. math::

        M_r = \int_{-1}^{1} (1+x)^a (1-x)^b T_r(x)\,dx.

    Two algorithms are used:

    * **Gegenbauer** (``a == b``): closed-form via the recurrence
        :math:`M_0 = \sqrt\pi \Gamma(a+1)/\Gamma(a+3/2)`,
        :math:`M_{2k} = M_0 \prod_{j=1}^k (j-a-1)/(j+a)`, odd moments zero.

    * **General** (``a ≠ b``): Sister Celine three-term recurrence.

    Parameters
    ----------
    a, b : float
        Jacobi exponents.
    n : int
        Number of moments to compute.

    Returns
    -------
    jax.Array, shape (n,)

    Provenance
    ----------
    MATLAB source : @singfun/sum.m (inner algorithm)
    Chebfun commit: 7574c77
    """
    if n == 0:
        return jnp.zeros(0, dtype=jnp.float64)

    M = jnp.zeros(n, dtype=jnp.float64)

    if abs(a - b) < _EXP_TOL:
        # Gegenbauer case: a == b
        r = a + 0.5
        # M0 = Gamma(r + 0.5) * sqrt(pi) / Gamma(r + 1)
        m0 = math.gamma(r + 0.5) * math.sqrt(math.pi) / math.gamma(r + 1.0)
        M = M.at[0].set(m0)
        # Even moments: M_{2k} = m0 * prod_{j=1}^{k} (j - r - 1) / (j + r)
        # even moments live at 0-based indices 2, 4, ..., i.e. (n-1)//2 of
        # them beyond M_0 (MATLAB: k = 1:floor((n-1)/2)); n//2 overflows
        # the slice for even n.
        n_even = (n - 1) // 2
        if n_even >= 1:
            ks = jnp.arange(1, n_even + 1, dtype=jnp.float64)
            ratios = (ks - r - 1.0) / (ks + r)
            even_vals = m0 * jnp.cumprod(ratios)
            M = M.at[2::2].set(even_vals)
        # Odd moments remain zero

    else:
        # General case: Sister Celine recurrence
        c1 = a + 1.0
        c2 = b + 1.0
        c3 = a + b + 1.0
        c4 = c1 + c2   # = a + b + 2
        c5 = a - b
        c0 = (2.0 ** c3) * _beta(c1, c2)

        # Normalised moments: Mbar_r such that M = c0 * Mbar
        Mbar = jnp.zeros(n, dtype=jnp.float64)
        Mbar = Mbar.at[0].set(1.0)
        if n > 1:
            Mbar = Mbar.at[1].set(c5 / c4)
        # MATLAB (@singfun/sum.m) iterates j = 3..n ONE-indexed:
        #   M(j) = (2*c5*M(j-1) + (j-2-c4)*M(j-2)) / (c3 + j - 1)
        # so with Python's 0-based index the factors are (j-1-c4) and
        # (c3 + j). The previous transcription kept MATLAB's literals with
        # the 0-based loop — both off by one — which made sum() wrong for
        # every ASYMMETRIC exponent pair (the symmetric Gegenbauer branch
        # masked it): integral of sqrt(1+x)e^x came out 2.3137 vs the true
        # 2.6141 while evaluation stayed exact.
        for j in range(2, n):
            val = (2.0 * c5 * Mbar[j - 1] + (j - 1.0 - c4) * Mbar[j - 2]) / (c3 + j)
            Mbar = Mbar.at[j].set(val)

        M = c0 * Mbar

    return M


def _beta(a: float, b: float) -> float:
    """Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def _extract_boundary_roots_coeffs(
    tech: Chebtech1 | Chebtech2, num_left, num_right
) -> tuple[Chebtech1 | Chebtech2, int, int]:
    """Delegate scalar boundary-root extraction to the smooth part's tech.

    Retains its coefficient dtype, tech kind, and source root tolerance.
    ``num_left``/``num_right`` are
    target multiplicities; pass ``None`` for both to extract every boundary root
    automatically (MATLAB ``nargin == 1`` mode).

    Provenance
    ----------
    MATLAB source : @chebtech/extractBoundaryRoots.m
    Chebfun commit: 7574c77
    """
    requested = None
    if num_left is not None or num_right is not None:
        requested = [[0.0 if num_left is None else num_left],
                     [0.0 if num_right is None else num_right]]
    new_tech, left, right = tech.extractBoundaryRoots(requested)
    # An empty smooth-part sentinel has empty count vectors. Singfun stores
    # one scalar exponent per endpoint, so its empty multiplicities are zero.
    left_count = 0 if left.size == 0 else int(left.reshape(-1)[0])
    right_count = 0 if right.size == 0 else int(right.reshape(-1)[0])
    return new_tech, left_count, right_count


# Blowup-detection preferences (MATLAB chebfunpref factory defaults).
_EXPONENT_TOL = 1.1e-11
_MAX_POLE_ORDER = 20


def _find_sing_exponents(op: Callable) -> tuple[float, float]:
    """Detect the endpoint exponents of ``op`` by sampling near +/-1.

    Uses the default ``singType = 'sing'`` at both ends, i.e. the fractional
    singularity-order finder (which also covers pole and root cases).

    Provenance
    ----------
    MATLAB source : @singfun/findSingExponents.m
    Chebfun commit: 7574c77
    """
    return (_find_sing_order(op, "left"), _find_sing_order(op, "right"))


def _pole_order_finder(fvals: jax.Array, x: jax.Array) -> int:
    """Integer pole order from sampled |values| (MATLAB poleOrderFinder)."""
    sv = jnp.abs(fvals)
    # Samples landing exactly ON the pole (Inf/NaN) carry no order
    # information; drop them.  This happens when a detected blow-up
    # breakpoint sits within an ulp of the pole and a scaled sample
    # rounds onto it.
    keep = ~(jnp.isinf(sv) | jnp.isnan(sv))
    sv = sv[keep]
    x = x[keep]
    if int(sv.shape[0]) < 4:
        raise ValueError(
            "Too few finite samples near the endpoint to determine a "
            "pole order.")

    test_ratio = 1.01
    pole_order = 0
    while pole_order <= _MAX_POLE_ORDER:
        ratios = sv[1:] / sv[:-1]
        if not bool(jnp.all(ratios > test_ratio)):
            break
        pole_order += 1
        sv = sv * x
    if pole_order > _MAX_POLE_ORDER:
        raise ValueError("Pole order exceeds limit for maximum pole order.")
    return pole_order


def _find_pole_order(op: Callable, sing_end: str) -> float:
    """Order of an integer pole of ``op`` at an endpoint (negated exponent).

    Provenance
    ----------
    MATLAB source : @singfun/findPoleOrder.m
    Chebfun commit: 7574c77
    """
    x = jnp.asarray([10.0 ** (-k) for k in range(1, 16)], dtype=jnp.float64)
    if sing_end == "right":
        fvals = op(1.0 - x)
    elif sing_end == "left":
        fvals = op(-1.0 + x)
    else:
        raise ValueError(f'Blowup preference "{sing_end}" unknown.')
    return -_pole_order_finder(fvals, x)


def _sing_order_finder(fvals: jax.Array, x: jax.Array, pole_bound: float) -> float:
    """Fractional singularity order from sampled |values| (singOrderFinder)."""
    abs_fvals = jnp.abs(fvals)
    sing_order = pole_bound - 1.0
    n = 11
    grid = [
        (pole_bound - 1.0) + (pole_bound - (pole_bound - 1.0)) * i / (n - 1)
        for i in range(n)
    ]
    n_iter = 0
    max_iter = 100

    def convex(exp_val):
        sv = abs_fvals * x ** exp_val
        return bool(jnp.all(jnp.diff(jnp.diff(sv)) > 0))

    while (abs(grid[-1] - grid[0]) > _EXPONENT_TOL) and (n_iter <= max_iter):
        k = 0
        while convex(grid[k]) and (k < n - 1):
            k += 1
        if (k == n - 1) and convex(grid[k]):
            return grid[n - 1]
        sing_order = grid[k]
        if k == 0:
            return sing_order
        lo, hi = grid[k - 1], grid[k]
        grid = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        n_iter += 1
    return sing_order


def _find_sing_order(op: Callable, sing_end: str) -> float:
    """Order of a fractional (or integer) singularity of ``op`` at an endpoint.

    Provenance
    ----------
    MATLAB source : @singfun/findSingOrder.m
    Chebfun commit: 7574c77
    """
    pole_bound = -_find_pole_order(op, sing_end)
    x = jnp.asarray([_EPS * k for k in range(11, 1, -1)], dtype=jnp.float64)
    if sing_end == "right":
        fvals = op(1.0 - x)
    elif sing_end == "left":
        fvals = op(-1.0 + x)
    else:
        raise ValueError(f'Blowup preference "{sing_end}" unknown.')
    sing_order = -_sing_order_finder(fvals, x, pole_bound)
    # Positive branch orders >= 1 are discarded (unsupported, and the
    # Chebyshev series still converges without factoring them out).
    if sing_order >= 1.0:
        sing_order = 0.0
    if sing_order == 0.0 and pole_bound > 0:
        # The eps-scale samples can quantize onto a plateau when ``op``
        # composes a nonlinear map whose endpoint offset cancels
        # catastrophically (unbounded-domain maps: b - m(y) is a few
        # ulps of b near y = 1), which makes the fractional fit read a
        # constant.  The decade-scale integer-pole finder is immune —
        # trust its verdict.
        sing_order = -float(pole_bound)
    return sing_order


def _fast_toeplitz_mult(a: jax.Array, x: jax.Array) -> jax.Array:
    """Symmetric ``toeplitz(a) @ x`` in O(n log n) via the FFT.

    Provenance
    ----------
    MATLAB source : @singfun/chebcoeffs.m (fastToeplitzMult, 2-arg form)
    Chebfun commit: 7574c77
    """
    a = a.at[0].multiply(2.0)
    n = a.shape[0]
    top = jnp.concatenate([a, jnp.zeros(1, dtype=a.dtype), jnp.flip(a[1:])])
    xpad = jnp.concatenate([x, jnp.zeros(n, dtype=x.dtype)])
    p = jnp.fft.ifft(jnp.fft.fft(top) * jnp.fft.fft(xpad))
    return p[:n]


def _fast_hankel_mult(a: jax.Array, x: jax.Array) -> jax.Array:
    """``hankel(a) @ x`` in O(n log n) via the FFT (2-arg form).

    Provenance
    ----------
    MATLAB source : @singfun/chebcoeffs.m (fastHankelMult, 2-arg form)
    Chebfun commit: 7574c77
    """
    n = x.shape[0]
    b = jnp.concatenate([a[-1:], jnp.zeros(n - 1, dtype=a.dtype)])
    top = jnp.concatenate([b, jnp.zeros(1, dtype=a.dtype), a[:-1]])
    xr = jnp.concatenate([jnp.flip(x), jnp.zeros(n, dtype=x.dtype)])
    p = jnp.fft.ifft(jnp.fft.fft(top) * jnp.fft.fft(xr))
    return p[:n]


def _chebT2U(cT: jax.Array) -> jax.Array:
    """Convert first-kind Chebyshev coefficients to second-kind (``U``) ones.

    Uses ``T_n = (U_n - U_{n-2})/2``.

    Provenance
    ----------
    MATLAB source : @chebtech/chebTcoeffs2chebUcoeffs.m
    Chebfun commit: 7574c77
    """
    m = cT.shape[0]
    cU = jnp.concatenate([cT, jnp.zeros(2, dtype=cT.dtype)])
    cU = cU.at[0].set(2.0 * cT[0])
    return 0.5 * (cU[:m] - cU[2 : m + 2])


def _sing_cumsum(f: Singfun):
    """Literal scalar singIntegral recurrence with complex coefficient support.

    Provenance: @singfun/cumsum.m, Chebfun 7574c77 (Hale--Olver).
    Host branches select the source representation; coefficient arithmetic
    remains in JAX. This construction routine is not JIT/AD qualified.
    """
    if not isinstance(f.smoothPart, (Chebtech1, Chebtech2)):
        raise ValueError("CHEBFUN:SINGFUN:cumsum:noSupport: unsupported smoothPart")
    flip = f.exponents[1] != 0
    work = f.flipud() if flip else f
    smooth = work.smoothPart
    tech_type = type(smooth)
    alpha = jnp.float64(-work.exponents[0])
    # MATLAB round uses nearest integer, ties away from zero. Negative
    # rounded values are all replaced by 1 in this particular expression.
    ra = max(int(jnp.floor(alpha + 0.5)), 1)
    xs = tech_type.from_function(lambda x: x + 1.0) * smooth
    old_n = len(xs) - 1
    n = max(old_n, ra + 2)
    if n != old_n:
        xs = xs.prolong(n + 1)
    aa = xs.coeffs
    c = jnp.zeros(n + 1, dtype=aa.dtype)  # one-based indices, c[0] unused
    c = c.at[n].set(2 * aa[n] / (1 - alpha / n))
    c = c.at[n-1].set(2 * (aa[n-1] - c[n]) / (1 - alpha / (n-1)))
    for k in range(n-2, ra, -1):
        c = c.at[k].set(2 * (aa[k] - c[k+1] - c[k+2]*.5*(1 + alpha/k))
                            / (1 - alpha/k))
    cm = (2.0**(ra-1)) * (aa[ra] - c[ra+1] - c[ra+2]*(1 + alpha/ra)/2)
    xa = tech_type.from_function(lambda x: (x + 1.0)**ra)
    aa = aa.at[:ra+1].add(-cm * xa.coeffs[::-1])
    for k in range(ra-1, 0, -1):
        c = c.at[k].set(2 * (aa[k] - c[k+1] - c[k+2]*.5*(1 + alpha/k))
                            / (1 - alpha/k))
    kk = jnp.arange(1, n+1, dtype=jnp.float64)
    c = .5*c[1:]
    dd1 = c/kk
    dd2 = -c[2:]/kk[:-2]
    cc = jnp.concatenate([jnp.zeros(1, dtype=c.dtype),
                          dd1 + jnp.concatenate([dd2, jnp.zeros(2, dtype=c.dtype)])])
    cc = cc.at[0].set(jnp.sum(cc[1::2]) - jnp.sum(cc[2::2]))
    if n > old_n + 2:
        cc = cc[:old_n+2]
    nonzero = jnp.nonzero(cc != 0)[0]
    cc = cc[:int(nonzero[-1])+1] if nonzero.size else jnp.zeros(1, dtype=cc.dtype)
    u = tech_type.from_coeffs(cc)
    # @chebtech/iszero: exact zero, including explicit rejection of NaNs.
    u_zero = bool(jnp.all(u.coeffs == 0))
    tol = _EPS * work.vscale
    if abs(ra-alpha) > tol:
        cm_scaled = cm/(ra-alpha)
        if u_zero and abs(cm_scaled) > tol*smooth.vscale:
            g = Singfun(tech_type.from_function(lambda x: cm_scaled + 0*x),
                        (ra-alpha, 0.0))
        elif not u_zero and abs(cm_scaled) < tol:
            g = Singfun(u, work.exponents).extractBoundaryRoots((1.0, 0.0))
        else:
            g = Singfun(u + cm_scaled*xa, work.exponents).extractBoundaryRoots((1.0, 0.0))
    elif abs(cm) < tol:
        g = Singfun(u, work.exponents).extractBoundaryRoots((1.0, 0.0))
    else:
        raise ValueError("CHEBFUN:SINGFUN:cumsum:noLog: indefinite integral has a logarithmic term")
    if flip:
        g = -g.flipud()
    # Native conditional subtraction, including arbitrarily small offsets.
    if g.exponents[0] >= 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Non-integer difference.*")
            g = g - g(jnp.float64(-1.0))
    return g
