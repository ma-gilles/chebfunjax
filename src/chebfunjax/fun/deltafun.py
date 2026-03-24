"""Deltafun — distributions with Dirac delta function support.

Translated from MATLAB Chebfun class @deltafun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

# Machine epsilon for float64
_EPS = float(jnp.finfo(jnp.float64).eps)


class Deltafun(eqx.Module):
    """Distribution of the form f(x) + Σ_k c_k δ(x − x_k).

    Represents a generalised function that is the sum of a smooth (or
    singular) part ``funPart`` and a finite collection of scaled Dirac delta
    functions.  The delta-function data is stored as a pair of arrays
    ``(delta_locs, delta_mags)`` where ``delta_locs[k]`` is the location and
    ``delta_mags[0, k]`` is the magnitude (coefficient) of the *k*-th delta.

    The magnitude array is kept as a 2-D array with rows corresponding to
    derivative orders: row 0 = deltas, row 1 = delta', etc.  For the common
    case of plain deltas, ``delta_mags`` is effectively a 1-D array promoted
    to shape (1, N).

    Attributes
    ----------
    funPart : Bndfun
        Smooth regular part of the distribution.
    delta_locs : jax.Array, shape (N,)
        Locations of the Dirac delta functions.
    delta_mags : jax.Array, shape (M, N)
        Magnitudes of delta functions and their derivatives.
        Row 0 = delta, row 1 = delta', etc.

    Notes
    -----
    **JAX contract:**

    * ``f(x)`` — evaluates the ``funPart`` only (JIT-safe).  Delta
      contributions are distributional and cannot be evaluated pointwise.
    * ``f.sum()`` — JIT-safe: returns ``funPart.sum() + sum(delta_mags[0])``.
    * ``f.diff(k)`` — NOT JIT-safe at the construction level; the result's
      evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @deltafun/deltafun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Bndfun, Singfun
    """

    funPart: Bndfun
    delta_locs: jax.Array
    delta_mags: jax.Array  # shape (M, N) where M = max derivative order + 1

    def __init__(
        self,
        funPart: Bndfun,
        delta_locs,
        delta_mags,
    ) -> None:
        """Low-level constructor.  Prefer :meth:`from_fun` or :meth:`from_fun_and_deltas`.

        Parameters
        ----------
        funPart : Bndfun
            Regular part.
        delta_locs : array-like, shape (N,)
            Delta function locations.
        delta_mags : array-like, shape (M, N) or (N,)
            Magnitudes.  If 1-D, it is treated as a single row (order 0).
        """
        self.funPart = funPart
        locs = jnp.asarray(delta_locs, dtype=jnp.float64).ravel()
        mags = jnp.asarray(delta_mags, dtype=jnp.float64)
        if mags.ndim == 1:
            mags = mags[jnp.newaxis, :]  # shape (1, N)
        self.delta_locs = locs
        self.delta_mags = mags

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_fun(cls, fun: Bndfun) -> "Deltafun":
        """Wrap a Bndfun in a Deltafun with no delta functions.

        Parameters
        ----------
        fun : Bndfun
            The smooth part.

        Returns
        -------
        Deltafun

        Provenance
        ----------
        MATLAB source : @deltafun/deltafun.m
        Chebfun commit: 7574c77
        """
        empty_locs = jnp.zeros(0, dtype=jnp.float64)
        empty_mags = jnp.zeros((1, 0), dtype=jnp.float64)
        return cls(fun, empty_locs, empty_mags)

    @classmethod
    def from_fun_and_deltas(
        cls,
        fun: Bndfun,
        delta_locs,
        delta_mags,
    ) -> "Deltafun":
        """Construct a Deltafun with both a smooth part and delta functions.

        Parameters
        ----------
        fun : Bndfun
            Regular part.
        delta_locs : array-like, shape (N,)
            Locations of delta functions.
        delta_mags : array-like, shape (N,) or (M, N)
            Magnitudes.

        Returns
        -------
        Deltafun

        Provenance
        ----------
        MATLAB source : @deltafun/deltafun.m
        Chebfun commit: 7574c77
        """
        return cls(fun, delta_locs, delta_mags)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        *,
        n: int | None = None,
    ) -> "Deltafun":
        """Construct a Deltafun from a callable with no delta functions.

        Parameters
        ----------
        f : callable
            Vectorised function on ``domain``.
        domain : Domain
            A single-interval domain [a, b].
        n : int or None, optional
            Fixed degree; None triggers adaptive construction.

        Returns
        -------
        Deltafun

        Provenance
        ----------
        MATLAB source : @deltafun/deltafun.m
        Chebfun commit: 7574c77
        """
        fun = Bndfun.from_function(f, domain, n=n)
        return cls.from_fun(fun)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def domain(self) -> Domain:
        """Domain of the funPart."""
        return self.funPart.domain

    @property
    def n_deltas(self) -> int:
        """Number of distinct delta function locations."""
        return int(self.delta_locs.shape[0])

    @property
    def has_deltas(self) -> bool:
        """True if there are any non-trivial delta functions."""
        if self.n_deltas == 0:
            return False
        return bool(jnp.any(self.delta_mags != 0.0))

    def __len__(self) -> int:
        """Number of Chebyshev coefficients in the smooth part."""
        return len(self.funPart)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the smooth part at x (ignoring delta contributions).

        Delta functions have no pointwise values; this method evaluates only
        ``funPart(x)``.  Callers wanting to detect delta locations should
        inspect ``self.delta_locs``.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation points in ``self.domain``.

        Returns
        -------
        jax.Array, same shape as x
            Values of the smooth part.

        Notes
        -----
        JIT-safe, vmap-safe.

        Provenance
        ----------
        MATLAB source : @deltafun/feval.m
        Chebfun commit: 7574c77
        """
        return self.funPart(x)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def sum(self) -> jax.Array:
        r"""Definite integral: :math:`\int f(x)\,dx + \sum_k c_k`.

        The integral of a Deltafun is the integral of its smooth part plus
        the sum of the (order-0) delta magnitudes.  Higher-order delta
        contributions (derivatives of delta) integrate to zero when acting
        on the constant function 1.

        Returns
        -------
        jax.Array, scalar float64
            The distributional integral.

        Notes
        -----
        JIT-safe.

        Provenance
        ----------
        MATLAB source : @deltafun/sum.m
        Chebfun commit: 7574c77
        """
        out = self.funPart.sum()
        if self.n_deltas > 0:
            # Row 0 of delta_mags contains the plain delta magnitudes
            out = out + jnp.sum(self.delta_mags[0])
        return out

    def diff(self, k: int = 1) -> "Deltafun":
        """Differentiate *k* times in the distributional sense.

        Differentiating a Deltafun:

        1. Differentiates ``funPart`` *k* times (ordinary sense).
        2. Shifts the delta magnitude matrix down by *k* rows (i.e., prepends
           *k* zero rows), turning each delta into its *k*-th derivative.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Deltafun
            The *k*-th distributional derivative.

        Notes
        -----
        NOT JIT-safe at the construction level.

        Provenance
        ----------
        MATLAB source : @deltafun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return Deltafun(self.funPart, self.delta_locs, self.delta_mags)

        new_funPart = self.funPart.diff(k)

        if self.n_deltas == 0:
            empty_locs = jnp.zeros(0, dtype=jnp.float64)
            empty_mags = jnp.zeros((1, 0), dtype=jnp.float64)
            return Deltafun(new_funPart, empty_locs, empty_mags)

        # Prepend k zero rows to the magnitude matrix
        m, n = self.delta_mags.shape
        zero_rows = jnp.zeros((k, n), dtype=jnp.float64)
        new_mags = jnp.concatenate([zero_rows, self.delta_mags], axis=0)
        return Deltafun(new_funPart, self.delta_locs, new_mags)

    def cumsum(self) -> "Deltafun":
        """Antiderivative in the distributional sense.

        Integrates the smooth part and converts each delta δ(x − x_k) into
        a Heaviside step H(x − x_k).  Higher-order delta derivatives are
        shifted up (i.e., the first zero row is removed from delta_mags).

        Returns
        -------
        Deltafun
            The antiderivative.

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @deltafun/cumsum.m
        Chebfun commit: 7574c77
        """
        new_funPart = self.funPart.cumsum()

        if self.n_deltas == 0:
            empty_locs = jnp.zeros(0, dtype=jnp.float64)
            empty_mags = jnp.zeros((1, 0), dtype=jnp.float64)
            return Deltafun(new_funPart, empty_locs, empty_mags)

        m, n_d = self.delta_mags.shape
        plain_mags = self.delta_mags[0]  # shape (n_d,)
        locs_py = [float(self.delta_locs[i]) for i in range(n_d)]
        mags_py = [float(plain_mags[i]) for i in range(n_d)]

        # Add Heaviside contributions to funPart
        if any(abs(mag) > 0.0 for mag in mags_py):
            def heaviside_correction(x: jax.Array) -> jax.Array:
                out = jnp.zeros_like(x, dtype=jnp.float64)
                for loc, mag in zip(locs_py, mags_py):
                    out = out + mag * jnp.where(x >= loc, 1.0, 0.0).astype(jnp.float64)
                return out

            hside = Bndfun.from_function(heaviside_correction, self.funPart.domain)
            combined = new_funPart + hside
        else:
            combined = new_funPart

        if m > 1:
            remaining_mags = self.delta_mags[1:]  # shape (m-1, n_d)
            return Deltafun(combined, self.delta_locs, remaining_mags)
        else:
            empty_locs = jnp.zeros(0, dtype=jnp.float64)
            empty_mags = jnp.zeros((1, 0), dtype=jnp.float64)
            return Deltafun(combined, empty_locs, empty_mags)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other) -> "Deltafun":
        """Add two Deltafuns (or a Deltafun and a Bndfun / scalar).

        Provenance
        ----------
        MATLAB source : @deltafun/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Deltafun):
            new_funPart = self.funPart + other.funPart
            new_locs, new_mags = _merge_deltas(
                self.delta_locs, self.delta_mags,
                other.delta_locs, other.delta_mags,
            )
            return Deltafun(new_funPart, new_locs, new_mags)
        elif isinstance(other, Bndfun):
            new_funPart = self.funPart + other
            return Deltafun(new_funPart, self.delta_locs, self.delta_mags)
        else:
            # scalar
            new_funPart = self.funPart + other
            return Deltafun(new_funPart, self.delta_locs, self.delta_mags)

    def __radd__(self, other) -> "Deltafun":
        return self.__add__(other)

    def __sub__(self, other) -> "Deltafun":
        """Subtraction.

        Provenance
        ----------
        MATLAB source : @deltafun/minus.m
        Chebfun commit: 7574c77
        """
        return self.__add__(-other)

    def __rsub__(self, other) -> "Deltafun":
        return (-self).__add__(other)

    def __neg__(self) -> "Deltafun":
        """Unary negation.

        Provenance
        ----------
        MATLAB source : @deltafun/uminus.m
        Chebfun commit: 7574c77
        """
        return Deltafun(-self.funPart, self.delta_locs, -self.delta_mags)

    def __pos__(self) -> "Deltafun":
        return Deltafun(self.funPart, self.delta_locs, self.delta_mags)

    def __mul__(self, other) -> "Deltafun":
        """Multiplication by a scalar.

        Multiplication of two Deltafuns is not generically supported
        (product of two distributions is ill-defined unless their singular
        supports are disjoint).

        Provenance
        ----------
        MATLAB source : @deltafun/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float)) or (
            hasattr(other, "shape") and other.shape == ()
        ):
            scalar = float(other)
            return Deltafun(
                self.funPart * scalar,
                self.delta_locs,
                self.delta_mags * scalar,
            )
        elif isinstance(other, Bndfun):
            # f_fun * g_bndfun: multiply funPart and scale delta mags by
            # the value of g at each delta location
            new_funPart = self.funPart * other
            if self.n_deltas == 0:
                return Deltafun(new_funPart, self.delta_locs, self.delta_mags)
            locs = self.delta_locs
            # Scale row 0 by g(x_k)
            g_vals = jax.vmap(other)(locs)  # shape (N,)
            new_mags = jnp.zeros_like(self.delta_mags)
            new_mags = new_mags.at[0].set(self.delta_mags[0] * g_vals)
            return Deltafun(new_funPart, locs, new_mags)
        elif isinstance(other, Deltafun):
            raise NotImplementedError(
                "Deltafun: multiplication of two Deltafuns is not supported "
                "unless their delta supports are disjoint.  "
                "Construct the product manually or restrict delta locations."
            )
        else:
            try:
                scalar = float(other)
                return self.__mul__(scalar)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Deltafun: cannot multiply Deltafun by {type(other).__name__}."
                )

    def __rmul__(self, other) -> "Deltafun":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Deltafun":
        """Division by a scalar.

        Provenance
        ----------
        MATLAB source : @deltafun/rdivide.m
        Chebfun commit: 7574c77
        """
        return self.__mul__(1.0 / float(other))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display.

        Examples
        --------
        >>> from chebfunjax.domain import Domain
        >>> d = Domain((-1.0, 1.0))
        >>> f = Deltafun.from_function(jnp.sin, d)
        >>> repr(f)
        'Deltafun([-1, 1], n=14, n_deltas=0)'
        """
        a, b = self.funPart.domain.a, self.funPart.domain.b
        return (
            f"Deltafun([{a:.4g}, {b:.4g}], "
            f"n={len(self.funPart)}, n_deltas={self.n_deltas})"
        )


# ======================================================================
# Private helpers
# ======================================================================


def _merge_deltas(
    locs1: jax.Array,
    mags1: jax.Array,
    locs2: jax.Array,
    mags2: jax.Array,
    *,
    tol: float = 100.0 * _EPS,
) -> tuple[jax.Array, jax.Array]:
    """Merge two sets of delta functions, combining coincident ones.

    Parameters
    ----------
    locs1, locs2 : jax.Array, shape (N1,) and (N2,)
    mags1 : jax.Array, shape (M1, N1)
    mags2 : jax.Array, shape (M2, N2)
    tol : float
        Proximity tolerance for merging coincident deltas.

    Returns
    -------
    new_locs : jax.Array, shape (N,)
    new_mags : jax.Array, shape (M, N)

    Provenance
    ----------
    MATLAB source : @deltafun/mergeDeltas.m
    Chebfun commit: 7574c77
    """
    # Convert to Python lists for easier manipulation
    l1 = [float(x) for x in locs1]
    l2 = [float(x) for x in locs2]
    m1 = [[float(mags1[r, c]) for c in range(mags1.shape[1])] for r in range(mags1.shape[0])]
    m2 = [[float(mags2[r, c]) for c in range(mags2.shape[1])] for r in range(mags2.shape[0])]

    M1 = len(m1)
    M2 = len(m2)
    M = max(M1, M2)

    # Pad m1 and m2 to have M rows
    while len(m1) < M:
        m1.append([0.0] * len(l1))
    while len(m2) < M:
        m2.append([0.0] * len(l2))

    # Build combined list starting with all from list 1
    out_locs = []
    out_mags = [[] for _ in range(M)]

    for i, loc in enumerate(l1):
        out_locs.append(loc)
        for r in range(M):
            out_mags[r].append(m1[r][i])

    # Merge from list 2
    for j, loc in enumerate(l2):
        merged = False
        for i, existing in enumerate(out_locs):
            if abs(existing - loc) <= tol:
                for r in range(M):
                    out_mags[r][i] += m2[r][j]
                merged = True
                break
        if not merged:
            out_locs.append(loc)
            for r in range(M):
                out_mags[r].append(m2[r][j])

    if len(out_locs) == 0:
        return (
            jnp.zeros(0, dtype=jnp.float64),
            jnp.zeros((1, 0), dtype=jnp.float64),
        )

    # Sort by location
    order = sorted(range(len(out_locs)), key=lambda i: out_locs[i])
    sorted_locs = [out_locs[i] for i in order]
    sorted_mags = [[out_mags[r][i] for i in order] for r in range(M)]

    return (
        jnp.array(sorted_locs, dtype=jnp.float64),
        jnp.array(sorted_mags, dtype=jnp.float64),
    )
