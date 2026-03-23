"""Domain representation for piecewise Chebyshev approximation.

A Domain represents an ordered set of breakpoints [a, b1, b2, ..., b] that
partition an interval into sub-intervals. The simplest domain is a single
interval [a, b].

Translated from MATLAB Chebfun class @domain (commit 7574c77) and informed
by chebpy's Interval/Domain classes.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import equinox as eqx
import jax.numpy as jnp


class Domain(eqx.Module):
    """Ordered breakpoints defining a piecewise domain.

    A Domain is an immutable collection of breakpoints ``(a, b1, ..., b)``
    where ``a < b1 < ... < b``. The simplest domain is a single interval
    ``(a, b)`` with two breakpoints. A domain with *n* breakpoints has
    *n - 1* sub-intervals.

    Attributes
    ----------
    breakpoints : tuple[float, ...]
        Strictly increasing sequence of breakpoints (at least two).

    Examples
    --------
    >>> d = Domain((-1.0, 1.0))
    >>> d
    Domain([-1.0, 1.0])
    >>> d.n_intervals
    1

    >>> d = Domain((-1.0, 0.0, 1.0))
    >>> d.n_intervals
    2

    Provenance
    ----------
    MATLAB source : @domain/domain.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun, Bndfun
    """

    breakpoints: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, breakpoints: tuple[float, ...] | list[float]) -> None:
        """Create a Domain from a sequence of breakpoints.

        Parameters
        ----------
        breakpoints : tuple or list of float
            Strictly increasing sequence of at least two breakpoints.

        Raises
        ------
        ValueError
            If fewer than two breakpoints, or if breakpoints are not strictly
            increasing.
        """
        bp = tuple(float(b) for b in breakpoints)
        if len(bp) < 2:
            raise ValueError(
                f"Domain requires at least 2 breakpoints, got {len(bp)}. "
                f"Example: Domain((-1.0, 1.0))."
            )
        for i in range(len(bp) - 1):
            if bp[i] >= bp[i + 1]:
                raise ValueError(
                    f"Breakpoints must be strictly increasing, but "
                    f"breakpoints[{i}]={bp[i]} >= breakpoints[{i + 1}]={bp[i + 1]}."
                )
        self.breakpoints = bp

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_endpoints(cls, a: float, b: float) -> Domain:
        """Create a simple single-interval domain [a, b].

        Parameters
        ----------
        a : float
            Left endpoint.
        b : float
            Right endpoint.

        Returns
        -------
        Domain
            Domain with breakpoints ``(a, b)``.

        Raises
        ------
        ValueError
            If ``a >= b``.
        """
        return cls((a, b))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def a(self) -> float:
        """Left endpoint of the domain."""
        return self.breakpoints[0]

    @property
    def b(self) -> float:
        """Right endpoint of the domain."""
        return self.breakpoints[-1]

    @property
    def n_intervals(self) -> int:
        """Number of sub-intervals."""
        return len(self.breakpoints) - 1

    @property
    def intervals(self) -> Iterator[Domain]:
        """Iterate over sub-intervals as single-interval Domain objects.

        Yields
        ------
        Domain
            Each consecutive pair of breakpoints as a Domain.

        Examples
        --------
        >>> d = Domain((-1.0, 0.0, 1.0))
        >>> [sub for sub in d.intervals]
        [Domain([-1.0, 0.0]), Domain([0.0, 1.0])]
        """
        for left, right in itertools.pairwise(self.breakpoints):
            yield Domain((left, right))

    @property
    def support(self) -> tuple[float, float]:
        """Overall support: (left endpoint, right endpoint)."""
        return (self.a, self.b)

    # ------------------------------------------------------------------
    # Affine mapping: [-1, 1] <-> [a, b]
    # ------------------------------------------------------------------

    def forward_map(self, y: jnp.ndarray) -> jnp.ndarray:
        """Map from the reference interval [-1, 1] to [a, b].

        Computes ``0.5 * ((b - a) * y + (b + a))``.

        Only valid for single-interval domains. For piecewise domains,
        iterate over ``self.intervals`` and map each sub-interval.

        Parameters
        ----------
        y : jnp.ndarray
            Points in [-1, 1].

        Returns
        -------
        jnp.ndarray
            Corresponding points in [a, b].

        Raises
        ------
        ValueError
            If the domain has more than one interval.

        Provenance
        ----------
        MATLAB source : @domain/domain.m (mapping utilities)
        Chebfun commit: 7574c77
        """
        if self.n_intervals != 1:
            raise ValueError(
                f"forward_map is only defined for single-interval domains, "
                f"but this domain has {self.n_intervals} intervals. "
                f"Iterate over self.intervals instead."
            )
        a, b = self.a, self.b
        return 0.5 * ((b - a) * y + (b + a))

    def inverse_map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map from [a, b] to the reference interval [-1, 1].

        Computes ``(2 * x - (b + a)) / (b - a)``.

        Only valid for single-interval domains. For piecewise domains,
        iterate over ``self.intervals`` and map each sub-interval.

        Parameters
        ----------
        x : jnp.ndarray
            Points in [a, b].

        Returns
        -------
        jnp.ndarray
            Corresponding points in [-1, 1].

        Raises
        ------
        ValueError
            If the domain has more than one interval.

        Provenance
        ----------
        MATLAB source : @domain/domain.m (mapping utilities)
        Chebfun commit: 7574c77
        """
        if self.n_intervals != 1:
            raise ValueError(
                f"inverse_map is only defined for single-interval domains, "
                f"but this domain has {self.n_intervals} intervals. "
                f"Iterate over self.intervals instead."
            )
        a, b = self.a, self.b
        return (2.0 * x - (a + b)) / (b - a)

    def map_derivative(self) -> float:
        """Derivative of the forward map: ``(b - a) / 2``.

        This is the Jacobian of the affine map from [-1, 1] to [a, b],
        which is constant since the map is linear.

        Only valid for single-interval domains.

        Returns
        -------
        float
            The derivative ``(b - a) / 2``.

        Raises
        ------
        ValueError
            If the domain has more than one interval.

        Provenance
        ----------
        MATLAB source : @domain/domain.m (mapping utilities)
        Chebfun commit: 7574c77
        """
        if self.n_intervals != 1:
            raise ValueError(
                f"map_derivative is only defined for single-interval domains, "
                f"but this domain has {self.n_intervals} intervals. "
                f"Iterate over self.intervals instead."
            )
        return (self.b - self.a) / 2.0

    # ------------------------------------------------------------------
    # Containment
    # ------------------------------------------------------------------

    def __contains__(self, x: object) -> bool:
        """Test whether a scalar x is in the domain [a, b] (inclusive).

        Parameters
        ----------
        x : float
            Point to test.

        Returns
        -------
        bool
            True if ``a <= x <= b``.
        """
        if not isinstance(x, (int, float)):
            return NotImplemented
        return self.a <= float(x) <= self.b

    def is_interior(self, x: float) -> bool:
        """Test whether a scalar x is strictly interior to [a, b].

        Parameters
        ----------
        x : float
            Point to test.

        Returns
        -------
        bool
            True if ``a < x < b``.
        """
        return self.a < float(x) < self.b

    # ------------------------------------------------------------------
    # Domain operations
    # ------------------------------------------------------------------

    def union(self, other: Domain) -> Domain:
        """Merge breakpoints from two domains with matching support.

        Both domains must span the same overall interval (same left and
        right endpoints). The result contains all breakpoints from both,
        with duplicates (within tolerance) removed.

        Parameters
        ----------
        other : Domain
            Another domain with the same support.

        Returns
        -------
        Domain
            Domain with merged breakpoints.

        Raises
        ------
        ValueError
            If the supports do not match (within tolerance).

        Provenance
        ----------
        MATLAB source : @domain/merge.m
        Chebfun commit: 7574c77
        """
        tol = 100.0 * _EPS * max(abs(self.a), abs(self.b), 1.0)
        if abs(self.a - other.a) > tol or abs(self.b - other.b) > tol:
            raise ValueError(
                f"Cannot union domains with different support: "
                f"[{self.a}, {self.b}] vs [{other.a}, {other.b}]. "
                f"Supports must match within tolerance {tol:.2e}."
            )
        return self._merge(other)

    def _merge(self, other: Domain) -> Domain:
        """Merge breakpoints from two domains (no support check).

        Parameters
        ----------
        other : Domain
            Another domain.

        Returns
        -------
        Domain
            Domain with merged, deduplicated, sorted breakpoints.
        """
        all_bp = sorted(set(self.breakpoints) | set(other.breakpoints))
        # Deduplicate within tolerance
        merged = [all_bp[0]]
        for bp in all_bp[1:]:
            tol = max(_HTOL, _HTOL * abs(bp))
            if abs(bp - merged[-1]) > tol:
                merged.append(bp)
        return Domain(tuple(merged))

    def restrict(self, a: float, b: float) -> Domain:
        """Restrict the domain to the sub-interval [a, b].

        Returns a new Domain whose breakpoints are those of ``self``
        that lie within [a, b], with ``a`` and ``b`` added as endpoints.

        Parameters
        ----------
        a : float
            Left endpoint of the restriction.
        b : float
            Right endpoint of the restriction.

        Returns
        -------
        Domain
            Restricted domain.

        Raises
        ------
        ValueError
            If [a, b] is not a sub-interval of the domain, or if ``a >= b``.
        """
        if a >= b:
            raise ValueError(
                f"Restriction interval must satisfy a < b, got a={a}, b={b}."
            )
        tol = max(_HTOL, _HTOL * max(abs(self.a), abs(self.b)))
        if a < self.a - tol or b > self.b + tol:
            raise ValueError(
                f"Restriction [{a}, {b}] is not a sub-interval of "
                f"[{self.a}, {self.b}]. Use union() to extend instead."
            )
        # Keep interior breakpoints from self that fall within [a, b]
        interior = [bp for bp in self.breakpoints if a < bp < b]
        # Deduplicate near a and b
        tol_a = max(_HTOL, _HTOL * abs(a))
        tol_b = max(_HTOL, _HTOL * abs(b))
        interior = [
            bp for bp in interior
            if abs(bp - a) > tol_a and abs(bp - b) > tol_b
        ]
        new_bp = [a] + interior + [b]
        return Domain(tuple(new_bp))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Test equality of two Domains (within tolerance).

        Parameters
        ----------
        other : Domain
            Domain to compare with.

        Returns
        -------
        bool
            True if both domains have the same number of breakpoints and
            all breakpoints match within tolerance.
        """
        if not isinstance(other, Domain):
            return NotImplemented
        if len(self.breakpoints) != len(other.breakpoints):
            return False
        for s, o in zip(self.breakpoints, other.breakpoints):
            tol = max(_HTOL, _HTOL * abs(s))
            if abs(s - o) > tol:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        """Test inequality of two Domains."""
        if not isinstance(other, Domain):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.breakpoints)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        bp_str = ", ".join(str(b) for b in self.breakpoints)
        return f"Domain([{bp_str}])"

    def __str__(self) -> str:
        return f"[{self.a}, {self.b}]"


# ======================================================================
# Module-level constants
# ======================================================================

_EPS = 2.220446049250313e-16  # Machine epsilon for float64
_HTOL = 5.0 * _EPS  # Horizontal tolerance (matches chebpy's htol)
