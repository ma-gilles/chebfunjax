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
        locs = jnp.asarray(delta_locs).ravel().astype(jnp.float64)
        mags = jnp.asarray(delta_mags)
        # Preserve complex magnitudes (needed by real()/imag()); otherwise
        # store as real float64.
        if not jnp.iscomplexobj(mags):
            mags = mags.astype(jnp.float64)
        if mags.ndim == 1:
            mags = mags[jnp.newaxis, :]  # shape (1, N)
        # Merge coincident deltas and drop negligible impulses at construction
        # (MATLAB calls simplifyDeltas in the constructor).
        if locs.shape[0] > 0 and mags.shape[1] > 0:
            locs, mags = _simplify_delta_data(locs, mags)
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

    # ------------------------------------------------------------------
    # Distribution predicates / accessors (MATLAB @deltafun methods;
    # added by Claude Fable 5, Deltafun buildout).
    # ------------------------------------------------------------------

    #: MATLAB deltafun cleanup tolerance (pref.deltaPrefs.deltaTol).
    DELTA_TOL: float = 1e-9

    def iszero(self, tol: float | None = None) -> bool:
        """True if both the smooth part and all deltas vanish.

        Provenance
        ----------
        MATLAB source : @deltafun/iszero.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        t = self.DELTA_TOL if tol is None else float(tol)
        mags_small = (self.n_deltas == 0
                      or bool(_np.all(_np.abs(_np.asarray(
                          self.delta_mags)) <= t)))
        fvals = _np.asarray(self.funPart.values)
        fscale = float(_np.max(_np.abs(fvals))) if fvals.size else 0.0
        return mags_small and fscale <= 1e2 * t * max(1.0, fscale)

    def isequal(self, other: "Deltafun",
                tol: float | None = None) -> bool:
        """Equality up to deltaTol and trailing zero-rows.

        Provenance
        ----------
        MATLAB source : @deltafun/isequal.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        if not isinstance(other, Deltafun):
            return False
        t = self.DELTA_TOL if tol is None else float(tol)
        d = self - other
        if d.n_deltas and bool(_np.any(_np.abs(_np.asarray(
                d.delta_mags)) > t)):
            return False
        xs = jnp.linspace(self.domain.a + 1e-12,
                          self.domain.b - 1e-12, 33)
        return bool(jnp.max(jnp.abs(d.funPart(
            (2.0 * xs - (self.domain.a + self.domain.b))
            / (self.domain.b - self.domain.a)))) < 1e3 * t)

    def simplify_deltas(self, tol: float | None = None) -> "Deltafun":
        """Drop deltas with |magnitude| <= deltaTol (MATLAB's
        cleanup at construction).

        Provenance
        ----------
        MATLAB source : @deltafun/simplifyDeltas.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        t = self.DELTA_TOL if tol is None else float(tol)
        if self.n_deltas == 0:
            return self
        mags = _np.asarray(self.delta_mags)
        keep = _np.any(_np.abs(mags) > t, axis=0)
        if bool(_np.all(keep)):
            return self
        if not bool(_np.any(keep)):
            return Deltafun(self.funPart,
                            jnp.zeros((0,), dtype=jnp.float64),
                            jnp.zeros((1, 0), dtype=jnp.float64))
        return Deltafun(self.funPart,
                        jnp.asarray(_np.asarray(self.delta_locs)[keep]),
                        jnp.asarray(mags[:, keep]))

    def innerProduct(self, other) -> jax.Array:
        """Distributional pairing <f, g>.

        <funPart_f + sum m_k delta_{x_k}, g> = <funPart_f, g_smooth>
        + sum m_k g(x_k) (+ the symmetric delta terms of g when g is a
        Deltafun; delta*delta pairings are ill-defined and rejected).

        Provenance
        ----------
        MATLAB source : @deltafun/innerProduct.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        g_fun = other.funPart if isinstance(other, Deltafun) else other
        out = self.funPart.inner(g_fun)
        a, b = float(self.domain.a), float(self.domain.b)

        def _eval(fun, locs):
            t = (2.0 * jnp.asarray(locs) - (a + b)) / (b - a)
            return fun(t)

        if self.n_deltas:
            out = out + jnp.sum(self.delta_mags[0]
                                * _eval(g_fun, self.delta_locs))
        if isinstance(other, Deltafun) and other.n_deltas:
            if self.n_deltas:
                locs_f = _np.asarray(self.delta_locs)
                locs_g = _np.asarray(other.delta_locs)
                if _np.min(_np.abs(locs_f[:, None]
                                   - locs_g[None, :])) < 1e-12:
                    raise ValueError(
                        "innerProduct of overlapping deltas is "
                        "ill-defined")
            out = out + jnp.sum(other.delta_mags[0]
                                * _eval(self.funPart, other.delta_locs))
        return out

    inner = innerProduct

    def real(self):
        """Real part of the distribution.

        Returns the real part of both the smooth part and the delta
        magnitudes.  If no delta functions survive, a bare Bndfun is returned
        (MATLAB demotes a delta-free DELTAFUN to its funPart).

        Provenance
        ----------
        MATLAB source : @deltafun/real.m
        Chebfun commit: 7574c77
        """
        fun_r = Bndfun.from_chebtech(
            self.funPart.onefun.real(), self.funPart.domain
        )
        if self.n_deltas == 0:
            return fun_r
        locs, mags = _simplify_delta_data(self.delta_locs,
                                          jnp.real(self.delta_mags))
        if locs.shape[0] == 0:
            return fun_r
        return Deltafun(fun_r, locs, mags)

    def imag(self):
        """Imaginary part of the distribution.

        Provenance
        ----------
        MATLAB source : @deltafun/imag.m
        Chebfun commit: 7574c77
        """
        fun_i = Bndfun.from_chebtech(
            self.funPart.onefun.imag(), self.funPart.domain
        )
        if self.n_deltas == 0:
            return fun_i
        locs, mags = _simplify_delta_data(self.delta_locs,
                                          jnp.imag(self.delta_mags))
        if locs.shape[0] == 0:
            return fun_i
        return Deltafun(fun_i, locs, mags)

    def minandmax(self):
        """Global minimum and maximum of the distribution.

        Returns
        -------
        vals : jax.Array, shape (2,)
            ``[min, max]``.
        pos : jax.Array, shape (2,)
            Locations of the min and max.

        A positive delta drives the maximum to ``+inf`` (attained at the first
        positive-delta location); a negative delta drives the minimum to
        ``-inf``.  Higher-order deltas do not affect extrema.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @deltafun/minandmax.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        (min_val, min_pos), (max_val, max_pos) = self.funPart.minandmax()
        vals = [float(min_val), float(max_val)]
        pos = [float(min_pos), float(max_pos)]

        if self.n_deltas > 0:
            delta0 = _np.asarray(self.delta_mags[0])
            locs = _np.asarray(self.delta_locs)
            pos_idx = _np.where(delta0 > 0)[0]
            neg_idx = _np.where(delta0 < 0)[0]
            if pos_idx.size > 0:
                vals[1] = _np.inf
                pos[1] = float(locs[pos_idx[0]])
            if neg_idx.size > 0:
                vals[0] = -_np.inf
                pos[0] = float(locs[neg_idx[0]])

        return (jnp.asarray(vals, dtype=jnp.float64),
                jnp.asarray(pos, dtype=jnp.float64))

    def restrict(self, s):
        """Restrict the distribution to subinterval(s) of its domain.

        ``s`` is an increasing sequence in ``[a, b]``.  With two entries the
        result is a single restricted piece; with more, a list of pieces (one
        per subinterval).  A piece with no delta functions is returned as a
        bare Bndfun.  A delta sitting exactly on an interior breakpoint is
        split evenly between the two adjacent pieces (each gets half its
        magnitude); deltas on the outer endpoints keep their full magnitude.

        NOT JIT-safe (construction-level operation).

        Provenance
        ----------
        MATLAB source : @deltafun/restrict.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        a, b = float(self.domain.a), float(self.domain.b)
        s = [float(v) for v in s]
        tol = _PROXIMITY_TOL
        if abs(s[0] - a) < tol:
            s[0] = a
        if abs(s[-1] - b) < tol:
            s[-1] = b
        if (
            s[0] < a
            or s[-1] > b
            or any(s[i + 1] - s[i] <= 0 for i in range(len(s) - 1))
        ):
            raise ValueError("Not a valid subinterval.")
        if len(s) == 2 and s[0] == a and s[1] == b:
            return self

        num_funs = len(s) - 1
        locs = _np.asarray(self.delta_locs)
        mags = _np.asarray(self.delta_mags)
        g = []
        for k in range(num_funs):
            fp = self.funPart.restrict(s[k], s[k + 1])
            idx = (locs >= s[k]) & (locs <= s[k + 1])
            dloc = locs[idx]
            dmag = mags[:, idx]
            if dloc.shape[0] > 0:
                dmag = dmag.copy()
                # Halve deltas sitting exactly on interior breakpoints so the
                # two adjacent pieces each carry half.
                if dloc[0] == s[k] and k != 0:
                    dmag[:, 0] = dmag[:, 0] / 2.0
                if dloc[-1] == s[k + 1] and k != num_funs - 1:
                    dmag[:, -1] = dmag[:, -1] / 2.0
                g.append(Deltafun(fp, jnp.asarray(dloc), jnp.asarray(dmag)))
            else:
                g.append(fp)

        if len(s) == 2:
            return g[0]
        return g

    @classmethod
    def zero_delta_fun(cls, domain=None) -> "Deltafun":
        """The zero distribution (zero funPart, no deltas).

        Provenance
        ----------
        MATLAB source : @deltafun/zeroDeltaFun.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.domain import Domain
        dom = Domain((-1.0, 1.0)) if domain is None else domain
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), dom)
        return cls(zero, jnp.zeros((0,), dtype=jnp.float64),
                   jnp.zeros((1, 0), dtype=jnp.float64))

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
        if isinstance(other, (int, float, complex)) or (
            hasattr(other, "shape") and getattr(other, "shape", None) == ()
        ):
            return Deltafun(
                self.funPart * other,
                self.delta_locs,
                self.delta_mags * other,
            )
        elif isinstance(other, Bndfun):
            # Upgrade the smooth function to a delta-free Deltafun and use the
            # general product (Leibniz rule for f*delta^(k)).
            return _deltafun_times(self, Deltafun.from_fun(other))
        elif isinstance(other, Deltafun):
            return _deltafun_times(self, other)
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


#: MATLAB delta preferences (chebfunpref factory defaults).
_DELTA_TOL = 1e-9        # pref.deltaPrefs.deltaTol
_PROXIMITY_TOL = 1e-11   # pref.deltaPrefs.proximityTol


def _merge_columns(A, v, tol: float = _PROXIMITY_TOL):
    """Merge columns of ``A`` whose locations ``v`` are (nearly) equal.

    Sorts by location, then sums columns whose locations coincide (both zero,
    both within 10*eps of zero, or within a relative ``tol``).

    Provenance
    ----------
    MATLAB source : @deltafun/mergeColumns.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    A = _np.asarray(A)
    v = _np.asarray(v).ravel()
    m = A.shape[1]
    if m == 0:
        return A, v
    idx = _np.argsort(v, kind="stable")
    v = v[idx]
    A = A[:, idx]

    out_cols = [A[:, 0].copy()]
    out_locs = [v[0]]
    for k in range(1, m):
        prev = out_locs[-1]
        cur = v[k]
        both_zero = (cur == 0.0 and prev == 0.0)
        both_near_zero = (abs(cur) < 10 * _EPS and abs(prev) < 10 * _EPS)
        vc_max = max(abs(cur), abs(prev))
        close = vc_max > 0 and abs(cur - prev) / vc_max < tol
        if both_zero or both_near_zero or close:
            out_cols[-1] = out_cols[-1] + A[:, k]
        else:
            out_cols.append(A[:, k].copy())
            out_locs.append(cur)
    return _np.array(out_cols).T if out_cols else A[:, :0], _np.asarray(out_locs)


def _clean_columns(A, v, tol: float = _DELTA_TOL):
    """Drop columns of ``A`` whose entries are all below ``tol`` in magnitude.

    Provenance
    ----------
    MATLAB source : @deltafun/cleanColumns.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    A = _np.asarray(A)
    v = _np.asarray(v).ravel()
    if A.shape[1] == 0:
        return A, v
    keep = _np.max(_np.abs(A), axis=0) >= tol
    return A[:, keep], v[keep]


def _clean_rows(A, tol: float = _DELTA_TOL):
    """Drop trailing rows of ``A`` whose entries are all below ``tol``.

    Provenance
    ----------
    MATLAB source : @deltafun/cleanRows.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    A = _np.asarray(A)
    if A.size == 0:
        return A
    last = A.shape[0]
    while last > 0 and _np.max(_np.abs(A[last - 1, :])) < tol:
        last -= 1
    return A[:last, :]


def _simplify_delta_data(locs, mags, deltaTol: float = _DELTA_TOL,
                         proxTol: float = _PROXIMITY_TOL):
    """Merge coincident deltas and drop negligible rows/columns.

    Returns cleaned ``(locs, mags)`` as jax arrays; if everything is removed,
    returns empty ``(shape (0,), shape (1, 0))`` arrays.

    Provenance
    ----------
    MATLAB source : @deltafun/simplifyDeltas.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    mags = _np.asarray(mags)
    locs = _np.asarray(locs).ravel()
    if mags.ndim == 1:
        mags = mags[_np.newaxis, :]
    if locs.shape[0] == 0 or mags.shape[1] == 0:
        return (jnp.zeros(0, dtype=jnp.float64),
                jnp.zeros((1, 0), dtype=jnp.float64))
    mags, locs = _merge_columns(mags, locs, proxTol)
    mags, locs = _clean_columns(mags, locs, deltaTol)
    mags = _clean_rows(mags, deltaTol)
    if locs.shape[0] == 0 or mags.shape[0] == 0 or mags.shape[1] == 0:
        return (jnp.zeros(0, dtype=jnp.float64),
                jnp.zeros((1, 0), dtype=jnp.float64))
    return jnp.asarray(locs), jnp.asarray(mags)


def _num_intersect(V, W, tol: float = _PROXIMITY_TOL) -> bool:
    """True if ``V`` and ``W`` share a (numerically) equal element.

    Provenance
    ----------
    MATLAB source : @deltafun/numIntersect.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    V = _np.asarray(V).ravel()
    W = _np.asarray(W).ravel()
    for vi in V:
        for wj in W:
            denom = max(abs(vi), abs(wj))
            if vi == wj or (denom > 0 and abs(vi - wj) / denom < tol):
                return True
    return False


def _fun_times_delta(fun, delta_mags, delta_locs):
    r"""Product of a smooth function with a stack of (derivative) deltas.

    Implements the distributional identity

        f * delta^(n) = sum_{j} (-1)^(n-j) C(n, j) f^(n-j)(x0) delta^(j),

    returning the resulting magnitude matrix (rows = delta-derivative order,
    columns = delta locations).

    Provenance
    ----------
    MATLAB source : @deltafun/times.m (funTimesDelta sub-function)
    Chebfun commit: 7574c77
    """
    import math

    import numpy as _np

    dmag = _clean_rows(_np.asarray(delta_mags))
    n = dmag.shape[0]
    m = dmag.shape[1]
    if n == 0 or m == 0:
        return None

    locs = jnp.asarray(delta_locs)
    rows = []
    fk = fun
    for k in range(n):
        vals = _np.asarray(fk(locs)).ravel()
        rows.append(((-1.0) ** k) * vals)
        if k < n - 1:
            fk = fk.diff()
    Fd = _np.array(rows)  # (n, m)

    D = _np.zeros((n, m), dtype=_np.result_type(Fd.dtype, dmag.dtype))
    for j in range(m):
        col = Fd[:, j]
        for i in range(1, n + 1):  # i = 1-based delta-derivative order
            mag = dmag[i - 1, j]
            if mag == 0:
                continue
            w = _np.array([math.comb(i - 1, k - 1) for k in range(1, i + 1)],
                          dtype=_np.float64)
            D[:i, j] += w * col[:i][::-1] * mag
    return D


def _merge_delta_blocks(mags1, locs1, mags2, locs2):
    """Stack two delta blocks (padding rows) for the constructor to clean/sort.

    Provenance
    ----------
    MATLAB source : @deltafun/mergeDeltas.m
    Chebfun commit: 7574c77
    """
    import numpy as _np

    if mags1 is None and mags2 is None:
        return (jnp.zeros(0, dtype=jnp.float64),
                jnp.zeros((1, 0), dtype=jnp.float64))
    if mags1 is None:
        return jnp.asarray(locs2), jnp.asarray(mags2)
    if mags2 is None:
        return jnp.asarray(locs1), jnp.asarray(mags1)

    m1 = _np.asarray(mags1)
    m2 = _np.asarray(mags2)
    n = max(m1.shape[0], m2.shape[0])
    dt = _np.result_type(m1.dtype, m2.dtype)
    if m1.shape[0] < n:
        m1 = _np.vstack([m1, _np.zeros((n - m1.shape[0], m1.shape[1]), dtype=dt)])
    if m2.shape[0] < n:
        m2 = _np.vstack([m2, _np.zeros((n - m2.shape[0], m2.shape[1]), dtype=dt)])
    mags = _np.hstack([m1, m2])
    locs = _np.concatenate([_np.asarray(locs1).ravel(),
                            _np.asarray(locs2).ravel()])
    return jnp.asarray(locs), jnp.asarray(mags)


def _deltafun_times(f: "Deltafun", g: "Deltafun") -> "Deltafun":
    """Product of two Deltafuns (smooth * smooth, and Leibniz smooth * delta).

    Delta functions at coincident points cannot be multiplied and raise.

    Provenance
    ----------
    MATLAB source : @deltafun/times.m
    Chebfun commit: 7574c77
    """
    funPart = f.funPart * g.funPart

    if f.n_deltas > 0 and g.n_deltas > 0:
        if _num_intersect(f.delta_locs, g.delta_locs):
            raise ValueError(
                "Delta functions at the same points cannot be multiplied."
            )

    dmag1 = (_fun_times_delta(g.funPart, f.delta_mags, f.delta_locs)
             if f.n_deltas > 0 else None)
    dmag2 = (_fun_times_delta(f.funPart, g.delta_mags, g.delta_locs)
             if g.n_deltas > 0 else None)

    locs, mags = _merge_delta_blocks(dmag1, f.delta_locs, dmag2, g.delta_locs)
    return Deltafun(funPart, locs, mags)


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
