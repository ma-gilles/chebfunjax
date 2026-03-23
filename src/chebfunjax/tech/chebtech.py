"""Chebyshev technology — smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Clenshaw evaluation (JIT-safe, grad-safe, vmap-safe)
# ============================================================================

def _clenshaw(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate a Chebyshev series at point(s) x via Clenshaw's algorithm.

    Computes  f(x) = c[0]*T_0(x) + c[1]*T_1(x) + ... + c[n-1]*T_{n-1}(x)
    using the three-term recurrence for Chebyshev polynomials of the first kind.

    Parameters
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients c[0], c[1], ..., c[n-1].
    x : jax.Array, shape ()  or (m,)
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    y : jax.Array, same shape as x
        Evaluated values.

    Notes
    -----
    This function is JIT-safe, grad-safe, and vmap-safe. It uses
    ``jax.lax.fori_loop`` so the number of iterations is determined only by the
    static shape of ``coeffs``, which makes it trace-friendly.

    The algorithm is the standard Clenshaw recurrence:
        b_{n+1} = b_n = 0
        b_k = c[k] + 2*x*b_{k+1} - b_{k+2}    for k = n-1, ..., 1
        f(x) = c[0] + x*b_1 - b_2

    Provenance
    ----------
    MATLAB source : @chebtech/clenshaw.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2.__call__
    """
    n = coeffs.shape[0]

    # Edge cases
    if n == 0:
        return jnp.zeros_like(x, dtype=jnp.float64)
    if n == 1:
        return jnp.broadcast_to(coeffs[0], x.shape)

    x2 = 2.0 * x

    # Clenshaw recurrence from the top.
    # We use lax.fori_loop for JIT friendliness.
    # State: (bk1, bk2)  — one step behind and two steps behind.
    def body(i, state):
        bk1, bk2 = state
        # k = n - 1 - i  (we iterate i = 0..n-2)
        k = n - 1 - i
        bk = coeffs[k] + x2 * bk1 - bk2
        return (bk, bk1)

    init = (jnp.zeros_like(x, dtype=jnp.float64),
            jnp.zeros_like(x, dtype=jnp.float64))
    bk1, bk2 = jax.lax.fori_loop(0, n - 1, body, init)

    # Final step: f(x) = c[0] + x*bk1 - bk2
    return coeffs[0] + x * bk1 - bk2


# ============================================================================
# Chebtech2 — the core class
# ============================================================================

class Chebtech2(eqx.Module):
    """Chebyshev interpolant on 2nd-kind points.

    Represents a smooth function on [-1, 1] via coefficients of the
    corresponding 1st-kind Chebyshev series expansion.

    Attributes
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients (T_0, T_1, ..., T_{n-1}).
    ishappy : bool
        True if the representation is resolved to the requested tolerance.

    Provenance
    ----------
    MATLAB source : @chebtech2/chebtech2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech1, Trigtech, Bndfun
    """

    coeffs: jax.Array
    ishappy: bool = eqx.field(static=True, default=True)

    # ------------------------------------------------------------------
    # Construction (class methods — NOT __init__)
    # ------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: jax.Array) -> Chebtech2:
        """Construct a Chebtech2 from Chebyshev coefficients.

        Parameters
        ----------
        coeffs : array_like, shape (n,)
            Chebyshev series coefficients c[0], ..., c[n-1].

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Examples
        --------
        >>> c = jnp.array([1.0, 0.0, -0.5])
        >>> f = Chebtech2.from_coeffs(c)
        >>> f.n
        3
        """
        coeffs = jnp.atleast_1d(jnp.asarray(coeffs, dtype=jnp.float64))
        return cls(coeffs=coeffs)

    @classmethod
    def from_values(cls, values: jax.Array) -> Chebtech2:
        """Construct a Chebtech2 from values at 2nd-kind Chebyshev points.

        Parameters
        ----------
        values : array_like, shape (n,)
            Function values at n Chebyshev points of the 2nd kind on [-1, 1],
            ordered from x = -1 to x = 1 (ascending, matching ``chebpts``).

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Examples
        --------
        >>> x = chebpts(5)
        >>> f = Chebtech2.from_values(jnp.sin(x))
        """
        values = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int = 16,
    ) -> Chebtech2:
        """Construct a Chebtech2 from a callable.

        If ``n`` is given, evaluates the function on an ``n``-point 2nd-kind
        Chebyshev grid and forms the interpolant directly (non-adaptive).

        If ``n`` is ``None`` (the default), uses an adaptive algorithm that
        doubles the number of points until the Chebyshev coefficients decay
        below the tolerance set by ``standard_chop``.

        Parameters
        ----------
        f : callable
            Function mapping an array of points to an array of values.
            Must be vectorised (accept and return arrays of the same shape).
        n : int or None, optional
            Fixed number of points. If ``None``, adaptive construction is used.
        maxpow2 : int, default 16
            Maximum power of 2 for adaptive grid size (grid will be
            ``2**maxpow2 + 1`` at most). Only used when ``n is None``.

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Notes
        -----
        Adaptive construction is NOT JIT-safe (Python while loop with
        data-dependent termination). Fixed-length construction IS JIT-safe
        in principle, but typically called outside JIT.

        The adaptive algorithm mirrors MATLAB Chebfun's refine/happinessCheck
        cycle: it evaluates on grids of size 2^k + 1 for k = 4, 5, ...,
        maxpow2, converts to coefficients, and calls ``standard_chop`` to
        check for convergence.

        Examples
        --------
        >>> f = Chebtech2.from_function(jnp.sin)
        >>> f.n  # typically ~14 for sin(x) on [-1, 1]
        14
        >>> f(0.5)  # close to sin(0.5)
        Array(0.47942554, dtype=float64)

        Provenance
        ----------
        MATLAB source : @chebtech2/chebtech2.m, @chebtech/populate.m,
            @chebtech2/refine.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(f, maxpow2)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int
    ) -> Chebtech2:
        """Fixed-length construction on an n-point Chebyshev-2 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=2)
        values = jnp.asarray(f(x), dtype=jnp.float64)
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls,
        f: Callable[[jax.Array], jax.Array],
        maxpow2: int = 16,
    ) -> Chebtech2:
        """Adaptive construction — Python-level loop, NOT JIT-safe.

        Evaluates f on grids of size 2^k + 1 for k = 4, 5, ..., maxpow2
        and uses standard_chop to detect convergence. Returns a happy
        Chebtech2 if convergence is detected, or an unhappy one at the
        maximum grid size otherwise.
        """
        for k in range(4, maxpow2 + 1):
            n = 2**k + 1
            x = chebpts(n, kind=2)
            values = jnp.asarray(f(x), dtype=jnp.float64)
            c = vals2coeffs(values)
            cutoff = standard_chop(c)
            if cutoff < n:
                return cls(coeffs=c[:cutoff], ishappy=True)

        # Did not converge — return unhappy at max length
        warnings.warn(
            f"Chebtech2.from_function: function did not converge with "
            f"{2**maxpow2 + 1} points. Returning unhappy representation.",
            stacklevel=2,
        )
        return cls(coeffs=c, ishappy=False)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the Chebyshev interpolant at point(s) x in [-1, 1].

        Uses Clenshaw's algorithm.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s).

        Returns
        -------
        y : jax.Array, same shape as x
            Evaluated values.

        Notes
        -----
        This method is JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/clenshaw.m, @chebtech/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return _clenshaw(self.coeffs, x)

    # ------------------------------------------------------------------
    # Static methods: vals2coeffs / coeffs2vals
    # ------------------------------------------------------------------

    @staticmethod
    def vals2coeffs(values: jax.Array) -> jax.Array:
        """Convert values at 2nd-kind Chebyshev points to coefficients.

        Delegates to ``chebfunjax.utils.transforms.vals2coeffs``.

        Parameters
        ----------
        values : jax.Array, shape (n,)

        Returns
        -------
        coeffs : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech2/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        return vals2coeffs(values)

    @staticmethod
    def coeffs2vals(coeffs: jax.Array) -> jax.Array:
        """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points.

        Delegates to ``chebfunjax.utils.transforms.coeffs2vals``.

        Parameters
        ----------
        coeffs : jax.Array, shape (n,)

        Returns
        -------
        values : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech2/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        return coeffs2vals(coeffs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients (= polynomial degree + 1)."""
        return self.coeffs.shape[0]

    @property
    def values(self) -> jax.Array:
        """Function values at 2nd-kind Chebyshev points (ascending order).

        Computed from coefficients via coeffs2vals. Not cached — equinox
        modules are frozen pytrees, so we recompute on access.
        """
        return coeffs2vals(self.coeffs)

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute function value."""
        return float(jnp.max(jnp.abs(self.values)))

    def __len__(self) -> int:
        """Number of Chebyshev coefficients, same as ``self.n``."""
        return self.n

    def __repr__(self) -> str:
        """Compact display like Chebfun.

        Examples
        --------
        >>> f = Chebtech2.from_function(jnp.sin)
        >>> repr(f)
        'Chebtech2(n=14, vscale=8.415e-01)'
        """
        vs = self.vscale
        return f"Chebtech2(n={self.n}, vscale={vs:.4g})"

    # ------------------------------------------------------------------
    # Core operations (return new Chebtech2 objects — immutability)
    # ------------------------------------------------------------------

    def prolong(self, n: int) -> Chebtech2:
        """Return a new Chebtech2 with n coefficients.

        If ``n > self.n``, zero-pads the coefficient array.
        If ``n < self.n``, truncates (which may lose accuracy).
        If ``n == self.n``, returns a copy.

        Parameters
        ----------
        n : int
            Desired number of coefficients.

        Returns
        -------
        Chebtech2
            New instance with ``n`` coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/prolong.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        m = self.n
        if n == m:
            return self
        if n > m:
            padded = jnp.concatenate([
                self.coeffs,
                jnp.zeros(n - m, dtype=jnp.float64),
            ])
            return Chebtech2(coeffs=padded, ishappy=self.ishappy)
        # n < m: truncate
        n = max(n, 0)
        return Chebtech2(coeffs=self.coeffs[:n], ishappy=self.ishappy)

    def simplify(self, tol: float | None = None) -> Chebtech2:
        """Return a new Chebtech2 with trailing coefficients chopped.

        Uses ``standard_chop`` to determine a suitable cutoff for the
        coefficient series. If the Chebtech2 is not happy, returns ``self``
        unchanged.

        Parameters
        ----------
        tol : float or None, optional
            Tolerance for ``standard_chop``. Default is machine epsilon.

        Returns
        -------
        Chebtech2
            Simplified instance (possibly shorter).

        Notes
        -----
        Following the MATLAB Chebfun convention, the coefficient array is
        first prolonged (zero-padded) to at least ``max(17, round(n * 1.25 + 5))``
        so that ``standard_chop`` has enough room for its plateau-detection
        logic. The result is then capped at the original length so that
        simplification never increases the number of coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/simplify.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        standard_chop
        """
        if not self.ishappy:
            return self

        nold = self.n
        # Prolong to give standard_chop room for plateau detection
        N = max(17, round(nold * 1.25 + 5))
        prolonged = self.prolong(N)

        # Round-trip through vals/coeffs to create a slightly noisy plateau
        # (standard_chop uses logarithms and needs non-zero plateau values)
        c = vals2coeffs(coeffs2vals(prolonged.coeffs))

        cutoff = standard_chop(c, tol)
        cutoff = min(cutoff, nold)

        return Chebtech2(coeffs=self.coeffs[:cutoff], ishappy=self.ishappy)
