"""Singular function on [-1, 1] — algebraic endpoint singularities.

Translated from MATLAB Chebfun class @singfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)

# Tolerance for considering an exponent "zero" (i.e. smooth at that end)
_EXP_TOL = 1e-11


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

    def __init__(self, smoothPart: Chebtech2, exponents: tuple):
        """Low-level constructor.  Prefer :meth:`from_function`.

        Parameters
        ----------
        smoothPart : Chebtech2
            Smooth factor on [-1, 1].
        exponents : tuple of two floats
            (a, b) — left and right algebraic exponents.
        """
        self.smoothPart = smoothPart
        self.exponents = (float(exponents[0]), float(exponents[1]))

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        exponents: tuple[float, float],
        *,
        n: int | None = None,
    ) -> "Singfun":
        """Construct a Singfun from a callable and known exponents.

        Given a function handle ``f`` that evaluates
        ``s(x)*(1+x)^a*(1-x)^b``, this constructor forms the smooth factor

            s(x) = f(x) / ((1+x)^a * (1-x)^b)

        and approximates it with a Chebtech2.

        Parameters
        ----------
        f : callable
            Vectorised function accepting and returning ``jax.Array``.
            May return ``inf`` or ``nan`` at the endpoints; these are handled
            by the singular factoring.
        exponents : tuple of two floats
            ``(a, b)`` — algebraic exponents at -1 and +1.
        n : int or None, optional
            Fixed number of Chebyshev points.  ``None`` triggers adaptive
            construction.

        Returns
        -------
        Singfun
            A new Singfun instance.

        Notes
        -----
        The Chebtech2 grid includes the exact endpoints x = ±1.  At these
        points both ``f(x)`` and the weight ``(1±x)^exponent`` may vanish
        simultaneously (e.g. ``f(x) = sqrt(1-x^2)`` at x = ±1), producing
        a 0/0 indeterminate form.  This is resolved by perturbing the
        evaluation slightly away from the endpoints when both the numerator
        and denominator are near zero, capturing the limiting value of the
        smooth factor accurately.

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
        a, b = float(exponents[0]), float(exponents[1])

        # Threshold below which we perturb away from the endpoint:
        # ~ sqrt(eps) so perturbation is much smaller than grid spacing of
        # a degree-17 Chebtech2 (~0.38), yet large enough to avoid 0/0.
        _eps12 = float(jnp.finfo(jnp.float64).eps) ** 0.5  # ~ 1.5e-8

        def smooth_f(x: jax.Array) -> jax.Array:
            """Extract the smooth factor s(x) = f(x) / weight(x)."""
            # Perturb x away from the endpoints when the weight would be < eps^0.5
            x_safe = jnp.where(
                (1.0 + x < _eps12) & (a != 0.0),
                x + _eps12,
                jnp.where(
                    (1.0 - x < _eps12) & (b != 0.0),
                    x - _eps12,
                    x,
                ),
            )
            val = f(x_safe)
            if a != 0.0:
                lf = jnp.maximum(1.0 + x_safe, float(jnp.finfo(jnp.float64).tiny))
                val = val / lf**a
            if b != 0.0:
                rf = jnp.maximum(1.0 - x_safe, float(jnp.finfo(jnp.float64).tiny))
                val = val / rf**b
            return val

        tech = Chebtech2.from_function(smooth_f, n=n)
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
        x = jnp.asarray(x, dtype=jnp.float64)
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
    def issmooth(self) -> bool:
        """True if both exponents are (numerically) zero."""
        a, b = self.exponents
        return abs(a) < _EXP_TOL and abs(b) < _EXP_TOL

    def __len__(self) -> int:
        return self.n

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
        if isinstance(other, Singfun):
            new_smoothPart = self.smoothPart * other.smoothPart
            new_exps = (
                self.exponents[0] + other.exponents[0],
                self.exponents[1] + other.exponents[1],
            )
            return Singfun(new_smoothPart, new_exps)
        elif isinstance(other, Chebtech2):
            return Singfun(self.smoothPart * other, self.exponents)
        else:
            return Singfun(self.smoothPart * other, self.exponents)

    def __rmul__(self, other) -> "Singfun":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Singfun":
        """Division: f / g or f / scalar.

        Provenance
        ----------
        MATLAB source : @singfun/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Singfun):
            new_smoothPart = self.smoothPart / other.smoothPart
            new_exps = (
                self.exponents[0] - other.exponents[0],
                self.exponents[1] - other.exponents[1],
            )
            return Singfun(new_smoothPart, new_exps)
        else:
            return Singfun(self.smoothPart / other, self.exponents)

    def __rtruediv__(self, other) -> "Singfun":
        """scalar / Singfun."""
        new_smoothPart = other / self.smoothPart
        new_exps = (-self.exponents[0], -self.exponents[1])
        return Singfun(new_smoothPart, new_exps)

    def __add__(self, other) -> "Singfun":
        """Addition: f + g where g is a Singfun, Chebtech2, or scalar.

        When the two operands share the same exponents the smooth parts are
        added directly.  When the exponents differ by integers the result can
        still be expressed as a Singfun by factoring out the more-singular
        exponent (Case 2 in MATLAB Chebfun).  Otherwise a warning-free
        approximation is used by evaluating the sum pointwise and
        re-constructing (Case 3).

        Provenance
        ----------
        MATLAB source : @singfun/plus.m
        Chebfun commit: 7574c77
        """
        # Upgrade scalar / Chebtech2 to Singfun with zero exponents
        if not isinstance(other, Singfun):
            if isinstance(other, Chebtech2):
                other = Singfun(other, (0.0, 0.0))
            else:
                const = float(other)
                other = Singfun(
                    Chebtech2.from_function(
                        lambda x, _c=const: jnp.full_like(x, _c, dtype=jnp.float64)
                    ),
                    (0.0, 0.0),
                )

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

            if d0_f != 0.0 or d1_f != 0.0:
                wf = Chebtech2.from_function(_make_weight(d0_f, d1_f))
                new_s = self.smoothPart * wf + other.smoothPart
            else:
                wg = Chebtech2.from_function(_make_weight(d0_g, d1_g))
                new_s = self.smoothPart + other.smoothPart * wg

            return Singfun(new_s, (new_a, new_b))

        # Case 3: non-integer difference — reconstruct from pointwise sum
        new_a = min(fExps[0], gExps[0])
        new_b = min(fExps[1], gExps[1])
        self_f = self
        other_f = other

        def sum_smooth(x: jax.Array) -> jax.Array:
            """Smooth factor of the sum: (f+g) / weight."""
            fv = self_f(x)
            gv = other_f(x)
            sumv = fv + gv
            _eps12 = float(jnp.finfo(jnp.float64).eps) ** 0.5
            x_safe = jnp.where(
                (1.0 + x < _eps12) & (new_a != 0.0),
                x + _eps12,
                jnp.where(
                    (1.0 - x < _eps12) & (new_b != 0.0),
                    x - _eps12,
                    x,
                ),
            )
            fv2 = self_f(x_safe)
            gv2 = other_f(x_safe)
            sumv2 = fv2 + gv2
            sumv_use = jnp.where((1.0 + x < _eps12) | (1.0 - x < _eps12), sumv2, sumv)
            if new_a != 0.0:
                lf = jnp.maximum(1.0 + x_safe, float(jnp.finfo(jnp.float64).tiny))
                sumv_use = sumv_use / lf ** new_a
            if new_b != 0.0:
                rf = jnp.maximum(1.0 - x_safe, float(jnp.finfo(jnp.float64).tiny))
                sumv_use = sumv_use / rf ** new_b
            return sumv_use

        new_tech = Chebtech2.from_function(sum_smooth)
        return Singfun(new_tech, (new_a, new_b))

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
        a, b = self.exponents
        return Singfun(self.smoothPart ** p, (a * p, b * p))

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
        Singfun
            The *k*-th derivative.  If all exponents of the result are zero,
            a Singfun with ``exponents=(0, 0)`` is returned (not a bare
            Chebtech2, to keep the type consistent).

        Notes
        -----
        NOT JIT-safe at the Python level (creates new objects via Python
        control flow).  The returned object's ``__call__`` method IS JIT-safe.

        Provenance
        ----------
        MATLAB source : @singfun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return Singfun(self.smoothPart, self.exponents)

        f = Singfun(self.smoothPart, self.exponents)
        for _ in range(k):
            a, b = f.exponents

            # First term: s'(x) * (1+x)^a * (1-x)^b
            s_term = Singfun(f.smoothPart.diff(), (a, b))

            # Second term: a * s(x) * (1+x)^(a-1) * (1-x)^b
            if abs(a) > _EXP_TOL:
                a_term = Singfun(f.smoothPart * a, (a - 1.0, b))
                s_term = s_term + a_term

            # Third term: -b * s(x) * (1+x)^a * (1-x)^(b-1)
            if abs(b) > _EXP_TOL:
                b_term = Singfun(f.smoothPart * (-b), (a, b - 1.0))
                s_term = s_term + b_term

            f = s_term

        return f

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

    def cumsum(self) -> "Singfun":
        """Antiderivative with F(-1) = 0.

        For functions with singularity at one endpoint only (the simpler
        integrable case), this uses the algorithm of Hale & Olver.  For
        smooth functions it delegates to the smoothPart's cumsum.

        Notes
        -----
        This is a simplified implementation: only the one-sided singularity
        case is fully supported.  Functions with singularities at both
        endpoints raise ``NotImplementedError``.

        NOT JIT-safe (construction-level operation).

        Provenance
        ----------
        MATLAB source : @singfun/cumsum.m
        Chebfun commit: 7574c77
        """
        a, b = self.exponents

        if abs(a) < _EXP_TOL and abs(b) < _EXP_TOL:
            return Singfun(self.smoothPart.cumsum(), (0.0, 0.0))

        if abs(a) > _EXP_TOL and abs(b) > _EXP_TOL:
            raise NotImplementedError(
                "Singfun.cumsum: antiderivatives of functions singular at both "
                "endpoints are not yet supported.  Use Singfun.sum() for definite "
                "integrals instead."
            )

        # One-sided singularity — use singIntegral algorithm (Hale & Olver)
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
        n_even = n // 2  # number of even moments at indices 2, 4, ..., 2*(n//2)
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
        for j in range(2, n):
            val = (2.0 * c5 * Mbar[j - 1] + (j - 2 - c4) * Mbar[j - 2]) / (c3 + j - 1)
            Mbar = Mbar.at[j].set(val)

        M = c0 * Mbar

    return M


def _beta(a: float, b: float) -> float:
    """Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def _sing_cumsum(f: Singfun) -> Singfun:
    """Antiderivative for a Singfun with a singularity at exactly one endpoint.

    Uses the Hale–Olver algorithm (see MATLAB @singfun/cumsum.m).

    Provenance
    ----------
    MATLAB source : @singfun/cumsum.m (singIntegral sub-function)
    Chebfun commit: 7574c77
    Algorithm: Hale, N. and Olver, S., "Numerical Computation of Indefinite
        Integrals for Functions with Poles or Algebraic Singularities",
        Unpublished Note.
    """
    a, b = f.exponents

    # Work with singularity at the LEFT end (flip if needed)
    flip = abs(b) > _EXP_TOL and abs(a) < _EXP_TOL
    if flip:
        # Flip: replace x -> -x so singularity moves to left end
        s_ref = f.smoothPart
        flipped_smooth = Chebtech2.from_function(lambda x: s_ref(-x))
        f_work = Singfun(flipped_smooth, (b, a))
    else:
        f_work = f

    a_w = f_work.exponents[0]  # singularity exponent at the left end
    aa = -a_w  # aa > 0 for integrable singularity

    # Get smooth part: (x+1)*s as a Chebtech2
    s = f_work.smoothPart
    xs = Chebtech2.from_function(lambda x: (x + 1.0) * s(x))

    N = len(xs) - 1
    oldN = N
    ra = max(round(aa), 1)
    if N < ra + 2:
        N = ra + 2
        # Prolong xs to N+1 coefficients
        c_old = xs.coeffs
        c_new = jnp.zeros(N + 1, dtype=jnp.float64).at[: c_old.shape[0]].set(c_old)
        xs = Chebtech2(c_new)

    xsc = xs.coeffs  # shape (N+1,) array
    aa_list = [float(xsc[i]) for i in range(min(len(xsc), N + 1))]
    while len(aa_list) < N + 1:
        aa_list.append(0.0)

    # Solve the recurrence for c_k (coefficients of u')
    c = [0.0] * (N + 1)
    c[N] = 2.0 * aa_list[N] / (1.0 - aa / N)
    c[N - 1] = 2.0 * (aa_list[N - 1] - c[N]) / (1.0 - aa / (N - 1))
    for k in range(N - 2, ra, -1):
        c[k] = (
            2.0 * (aa_list[k] - c[k + 1] - c[k + 2] * 0.5 * (1.0 + aa / k))
            / (1.0 - aa / k)
        )

    # Compute Cm
    Cm = (2.0 ** (ra - 1)) * (
        aa_list[ra] - c[ra + 1] - c[ra + 2] * (1.0 + aa / ra) / 2.0
    )

    # Compute (x+1)^ra as a Chebtech2
    xa_tech = Chebtech2.from_function(lambda x: (1.0 + x) ** ra)
    xa_c = [float(xa_tech.coeffs[i]) if i < len(xa_tech.coeffs) else 0.0
            for i in range(ra + 2)]

    # Modify aa_list
    aa_mod = list(aa_list)
    for i in range(ra + 1):
        aa_mod[i] -= Cm * xa_c[ra - i]  # flipud equivalent

    # Compute remaining c_k
    for k in range(ra - 1, 0, -1):
        c[k] = (
            2.0 * (aa_mod[k] - c[k + 1] - c[k + 2] * 0.5 * (1.0 + aa / k))
            / (1.0 - aa / k)
        )

    # Integrate u' to get u coefficients
    kk = list(range(1, N + 1))
    c_half = [cv * 0.5 for cv in c[1:]]  # c[1..N] / 2

    dd1 = [c_half[k - 1] / k for k in kk]
    dd2 = [-c_half[k + 1] / kk[k - 1] for k in range(len(kk) - 2)]

    cc = [0.0] * (N + 1)
    for i, v in enumerate(dd1):
        cc[i + 1] += v
    for i, v in enumerate(dd2):
        cc[i + 1] += v

    # Choose cc[0] so u(-1) = 0
    pos = sum(cc[i] for i in range(2, N + 1, 2))
    neg = sum(cc[i] for i in range(1, N + 1, 2))
    cc[0] = neg - pos  # from T_k(-1) = (-1)^k

    # Trim
    if N > oldN + 2:
        cc = cc[: oldN + 2]

    # Remove trailing zeros
    last_nz = 0
    for i in range(len(cc) - 1, -1, -1):
        if abs(cc[i]) > 0.0:
            last_nz = i
            break
    cc = cc[: last_nz + 1] if last_nz > 0 else [0.0]

    u_coeffs = jnp.array(cc, dtype=jnp.float64)
    u_tech = Chebtech2(u_coeffs)

    # Construct the antiderivative Singfun
    exps_new = list(f_work.exponents)
    tol = _EPS * float(jnp.max(jnp.abs(f_work.smoothPart.coeffs)))

    if abs(ra - aa) > tol:
        CM = Cm / (ra - aa)
        u_smooth = Singfun(u_tech + xa_tech * CM, tuple(exps_new))
    else:
        u_smooth = Singfun(u_tech, tuple(exps_new))

    # Adjust so antiderivative is zero at -1
    if exps_new[0] >= 0.0:
        lval = float(u_smooth(jnp.float64(-1.0)))
        u_smooth = u_smooth - lval

    if flip:
        # Flip back and negate
        inner_smooth = u_smooth.smoothPart
        flipped_back = Chebtech2.from_function(lambda x: inner_smooth(-x))
        u_smooth = Singfun(
            -flipped_back, (u_smooth.exponents[1], u_smooth.exponents[0])
        )

    return u_smooth
