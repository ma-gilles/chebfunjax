"""Bounded-interval function (Bndfun) — smooth functions on [a, b].

Translated from MATLAB Chebfun class @bndfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.classicfun import Classicfun
from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)


class Bndfun(Classicfun):
    """Smooth function on a bounded interval [a, b].

    ``Bndfun`` wraps a :class:`~chebfunjax.tech.chebtech.Chebtech2` (which
    lives on the standard interval [-1, 1]) with an affine linear map to an
    arbitrary bounded interval [a, b].  All function-approximation logic
    (coefficient representation, evaluation, arithmetic, calculus, roots)
    is handled by the underlying ``Chebtech2`` (``self.onefun``); the
    ``Bndfun`` layer is responsible solely for the domain mapping.

    Attributes
    ----------
    onefun : Chebtech2
        Chebyshev representation on [-1, 1].
    domain : Domain
        The interval [a, b] (a single-interval Domain).

    Examples
    --------
    Construct from a callable on [0, π]:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.fun.bndfun import Bndfun
    >>> from chebfunjax.domain import Domain
    >>> d = Domain((0.0, float(jnp.pi)))
    >>> f = Bndfun.from_function(jnp.sin, d)
    >>> float(f.sum())      # ∫₀^π sin(x) dx = 2
    2.0
    >>> float(f(jnp.float64(jnp.pi / 2)))   # sin(π/2) = 1
    1.0

    Provenance
    ----------
    MATLAB source : @bndfun/bndfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Classicfun, Chebtech2, Domain
    """

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
    ) -> "Bndfun":
        """Construct a Bndfun from a callable on [a, b].

        The callable ``f`` is evaluated at Chebyshev-2 points mapped to
        [a, b].  If ``n`` is ``None`` (default), an adaptive algorithm
        doubles the grid size until the Chebyshev coefficients decay to
        machine precision.

        Parameters
        ----------
        f : callable
            Vectorised function accepting and returning ``jax.Array``.
        domain : Domain
            A single-interval domain [a, b] (``domain.n_intervals == 1``).
        n : int or None, optional
            Fixed number of Chebyshev points.  ``None`` triggers adaptive
            construction.

        Returns
        -------
        Bndfun
            A new Bndfun instance.

        Raises
        ------
        ValueError
            If ``domain`` is not a single-interval domain.

        Examples
        --------
        >>> d = Domain((0.0, float(jnp.pi)))
        >>> f = Bndfun.from_function(jnp.sin, d)
        >>> f.n   # typically 14 (sin needs 14 coefficients on [0,π] too)
        14

        Provenance
        ----------
        MATLAB source : @bndfun/bndfun.m
        Chebfun commit: 7574c77
        """
        _validate_single_domain(domain)
        # Remap f from [a, b] to [-1, 1]: x = forward_map(y)
        mapped_f = lambda y: f(domain.forward_map(y))  # noqa: E731
        onefun = Chebtech2.from_function(mapped_f, n=n)
        return cls(onefun=onefun, domain=domain)

    @classmethod
    def from_chebtech(cls, tech: Chebtech2, domain: Domain) -> "Bndfun":
        """Wrap an existing Chebtech2 in a domain mapping.

        Parameters
        ----------
        tech : Chebtech2
            An already-constructed Chebtech2 on [-1, 1].
        domain : Domain
            A single-interval domain [a, b].

        Returns
        -------
        Bndfun
            A new Bndfun instance.

        Raises
        ------
        ValueError
            If ``domain`` is not a single-interval domain.

        Examples
        --------
        >>> from chebfunjax.tech.chebtech import Chebtech2
        >>> t = Chebtech2.from_function(jnp.sin)
        >>> d = Domain((-1.0, 1.0))
        >>> f = Bndfun.from_chebtech(t, d)

        Provenance
        ----------
        MATLAB source : @bndfun/bndfun.m
        Chebfun commit: 7574c77
        """
        _validate_single_domain(domain)
        return cls(onefun=tech, domain=domain)

    # ------------------------------------------------------------------
    # Restriction to a sub-interval
    # ------------------------------------------------------------------

    def restrict(self, a: float, b: float) -> "Bndfun":
        """Restrict this Bndfun to the sub-interval [a, b].

        The function is re-represented on the sub-interval by evaluating the
        current Chebtech2 at Chebyshev points mapped from [a, b] into [-1, 1]
        and forming a new Chebtech2 on [-1, 1] that represents the restriction.

        Parameters
        ----------
        a : float
            Left endpoint of the sub-interval.
        b : float
            Right endpoint of the sub-interval.

        Returns
        -------
        Bndfun
            A new Bndfun on [a, b].

        Raises
        ------
        ValueError
            If [a, b] is not a sub-interval of ``self.domain``.

        Examples
        --------
        >>> d = Domain((0.0, float(jnp.pi)))
        >>> f = Bndfun.from_function(jnp.sin, d)
        >>> g = f.restrict(0.0, float(jnp.pi / 2))
        >>> float(g.sum())  # ∫₀^(π/2) sin(x) dx = 1
        1.0

        Provenance
        ----------
        MATLAB source : @bndfun/restrict.m
        Chebfun commit: 7574c77
        """
        a = float(a)
        b = float(b)
        self_a, self_b = self.domain.a, self.domain.b
        hs = (self_b - self_a) * _EPS
        if a < self_a - hs or b > self_b + hs:
            raise ValueError(
                f"Cannot restrict Bndfun on [{self_a}, {self_b}] to "
                f"[{a}, {b}]: not a sub-interval. "
                f"Use extend or construct a new Bndfun instead."
            )
        a = max(a, self_a)
        b = min(b, self_b)
        if abs(a - self_a) < hs and abs(b - self_b) < hs:
            return self

        new_domain = Domain((a, b))

        # Map [a, b] into the reference interval [-1, 1] of self.domain:
        #   new physical point x in [a, b]
        #   -> self.domain reference: t = (2x - (self_a+self_b)) / (self_b - self_a)
        # Then restrict self.onefun (which lives on [-1, 1]) to [t_a, t_b]:
        t_a = self.domain.inverse_map(jnp.float64(a))
        t_b = self.domain.inverse_map(jnp.float64(b))
        new_onefun = self.onefun.restrict(float(t_a), float(t_b))

        return Bndfun(onefun=new_onefun, domain=new_domain)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display like Chebfun.

        Examples
        --------
        >>> f = Bndfun.from_function(jnp.sin, Domain((0.0, float(jnp.pi))))
        >>> repr(f)
        'Bndfun([0, 3.142], n=14, lval=0, rval=-2.449e-15)'
        """
        a, b = self.domain.a, self.domain.b
        lval = float(self.onefun(jnp.float64(-1.0)))
        rval = float(self.onefun(jnp.float64(1.0)))
        return (
            f"Bndfun([{a:.4g}, {b:.4g}], n={self.n}, "
            f"lval={lval:.4g}, rval={rval:.4g})"
        )


# ======================================================================
# Module-level helpers
# ======================================================================

def _validate_single_domain(domain: Domain) -> None:
    """Raise ValueError if domain is not a single-interval domain."""
    if domain.n_intervals != 1:
        raise ValueError(
            f"Bndfun requires a single-interval domain, but got a domain "
            f"with {domain.n_intervals} intervals: {domain}. "
            f"Use a Domain with exactly 2 breakpoints, e.g. "
            f"Domain((a, b))."
        )
