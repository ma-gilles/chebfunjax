# uses-numpy: 2D adaptive construction uses numpy for pivot selection (not JIT-safe)
"""Chebfun2 — user-facing 2D function approximation on rectangles.

Wraps ``SeparableApprox`` with a friendly API for bivariate smooth
functions on rectangles [xa, xb] x [ya, yb].

Translated from MATLAB Chebfun class @chebfun2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Helper: 1-D integral of a Chebtech2 slice on a physical interval [a, b]
# ============================================================================


def _chebtech_integral(tech: Chebtech2, a: float, b: float) -> jax.Array:
    """Definite integral of ``tech`` (on reference [-1, 1]) over physical [a, b].

    Uses the standard scale factor (b - a) / 2 for the affine map
    ``x = (b - a) / 2 * t + (a + b) / 2``.
    """
    return tech.sum() * jnp.float64((b - a) / 2.0)


def _chebtech_cumsum_physical(tech: Chebtech2, a: float, b: float) -> Chebtech2:
    """Antiderivative on physical interval [a, b], vanishing at a."""
    t2 = tech.cumsum()
    half = (b - a) / 2.0
    return Chebtech2.from_coeffs(t2.coeffs * half, ishappy=tech.ishappy)


def _chebtech_diff_physical(tech: Chebtech2, a: float, b: float, k: int = 1) -> Chebtech2:
    """Differentiate ``tech`` with respect to the physical variable k times.

    Chain rule: d/dx = (2 / (b - a)) * d/dt, so the k-th derivative picks
    up a factor of (2 / (b - a))^k.

    Parameters
    ----------
    tech : Chebtech2
        A Chebtech2 on the reference interval [-1, 1].
    a, b : float
        Physical domain endpoints.
    k : int
        Differentiation order.

    Returns
    -------
    Chebtech2
        The k-th derivative (on reference [-1, 1]) scaled for the physical domain.
    """
    scale = (2.0 / (b - a)) ** k
    tech_der = tech.diff(k)
    scaled_coeffs = tech_der.coeffs * jnp.float64(scale)
    return Chebtech2.from_coeffs(scaled_coeffs)


# ============================================================================
# Main class
# ============================================================================


class Chebfun2(eqx.Module):
    """Chebfun2 — smooth function on a rectangle, via low-rank approximation.

    Represents a bivariate smooth function f(x, y) on a rectangle
    [xa, xb] x [ya, yb] using a ``SeparableApprox`` (Gaussian elimination
    with complete pivoting / Chebfun2 algorithm).

    The internal representation is:

        f(x, y) ≈ Σ_j  d_j * c_j(y) * r_j(x)

    where ``c_j`` are column slices (functions of y), ``r_j`` are row
    slices (functions of x), and ``d_j`` are scalar pivot weights.

    Attributes
    ----------
    approx : SeparableApprox
        The underlying low-rank approximation.

    Notes
    -----
    Construction is NOT JIT-safe (adaptive algorithm with Python loops).
    Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun2/chebfun2.m, @separableApprox/separableApprox.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend & L. N. Trefethen, "An extension of Chebfun to
        two dimensions", SISC, 35(6), C495–C518, 2013.

    See Also
    --------
    SeparableApprox, chebfun2
    """

    @classmethod
    def empty(cls) -> "Chebfun2":
        """The empty Chebfun2 (MATLAB chebfun2()): no data; isempty() is
        True and operations on it are undefined.

        Provenance
        ----------
        MATLAB source : @chebfun2/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Chebfun2 (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @chebfun2/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    approx: SeparableApprox

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array, jax.Array], jax.Array],
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        tol: Optional[float] = None,
        n: Optional[int] = None,
    ) -> "Chebfun2":
        """Construct a Chebfun2 from a callable f(x, y).

        Uses the Chebfun2 algorithm (Gaussian elimination with complete
        pivoting) to adaptively find a low-rank representation.

        Parameters
        ----------
        f : callable
            A function f(x, y) that accepts JAX arrays and returns JAX arrays.
            Must be vectorised: f(xx, yy) where xx and yy are 2D arrays
            from ``jnp.meshgrid``.
        domain : tuple of 4 floats, optional
            (xa, xb, ya, yb). Default is (-1, 1, -1, 1).
        tol : float, optional
            Target tolerance. Default is machine epsilon (~2.2e-16).
        n : int, optional
            If given, use exactly n x n sampling points (non-adaptive in
            the grid size sense; rank is still determined adaptively).
            Not yet implemented — raises ``NotImplementedError`` if given.

        Returns
        -------
        Chebfun2

        Raises
        ------
        ValueError
            If ``domain`` does not have exactly 4 elements.
        NotImplementedError
            If ``n`` is given (fixed-degree construction not yet implemented).

        Notes
        -----
        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2/chebfun2.m, @chebfun2/constructor.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Townsend & Trefethen, SISC 2013.

        See Also
        --------
        chebfun2
        """
        if len(domain) != 4:
            raise ValueError(
                f"Chebfun2.from_function: domain must have exactly 4 elements "
                f"(xa, xb, ya, yb), got {len(domain)}."
            )
        if n is not None:
            raise NotImplementedError(
                "Chebfun2.from_function: fixed-degree construction (n=...) "
                "is not yet implemented."
            )
        kwargs: dict = dict(domain=domain)
        if tol is not None:
            kwargs["tol"] = tol
        # Complex-valued functions: the GE constructor real-casts, so
        # build real and imaginary parts separately and recombine
        # exactly (f = re + 1j*im).  Found in the Fable 5 audit: complex
        # Chebfun2s previously silently dropped their imaginary part.
        xa, xb, ya, yb = (float(v) for v in domain)
        xprobe = jnp.asarray([[0.5 * (xa + xb) + 0.25 * (xb - xa)]])
        yprobe = jnp.asarray([[0.5 * (ya + yb) + 0.25 * (yb - ya)]])
        if jnp.iscomplexobj(jnp.asarray(f(xprobe, yprobe))):
            fre = cls(approx=SeparableApprox.from_function(
                lambda x, y: jnp.real(f(x, y)), **kwargs))
            fim = cls(approx=SeparableApprox.from_function(
                lambda x, y: jnp.imag(f(x, y)), **kwargs))
            return fre + fim * 1j
        approx = SeparableApprox.from_function(f, **kwargs)
        return cls(approx=approx)

    # ------------------------------------------------------------------
    # Evaluation (JIT-safe)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Evaluate f(x, y).

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb]. Must broadcast with x.

        Returns
        -------
        jax.Array, same shape as broadcast(x, y)
            Approximated function values.

        Notes
        -----
        JIT-safe, vmap-safe, and grad-safe.

        Provenance
        ----------
        MATLAB source : @separableApprox/feval.m, @chebfun2/feval.m
        Chebfun commit: 7574c77
        """
        return self.approx(x, y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def domain(self) -> tuple[float, float, float, float]:
        """Physical domain (xa, xb, ya, yb)."""
        return self.approx.domain

    @property
    def rank(self) -> int:
        """Numerical rank of the low-rank approximation."""
        return self.approx.rank

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, dim: int = 1, k: int = 1) -> "Chebfun2":
        """Partial derivative of f.

        Parameters
        ----------
        dim : int, default 1
            Dimension to differentiate along.
            - ``dim=1``: derivative with respect to y.
            - ``dim=2``: derivative with respect to x.
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Chebfun2
            The k-th partial derivative in the chosen direction.

        Raises
        ------
        ValueError
            If dim is not 1 or 2, or if k < 0.

        Notes
        -----
        Each col/row slice is differentiated independently, with the
        chain-rule scale factor (2/(b-a))^k for the affine map.

        Provenance
        ----------
        MATLAB source : @separableApprox/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, sum2
        """
        if dim not in (1, 2):
            raise ValueError(
                f"Chebfun2.diff: dim must be 1 (y-direction) or 2 (x-direction), "
                f"got dim={dim}."
            )
        if k < 0:
            raise ValueError(
                f"Chebfun2.diff: differentiation order k must be >= 0, got k={k}."
            )
        if k == 0:
            return self

        xa, xb, ya, yb = self.domain

        if dim == 1:
            # Differentiate column slices c_j(y) with respect to y
            new_cols = [_chebtech_diff_physical(c, ya, yb, k) for c in self.approx.cols]
            new_rows = list(self.approx.rows)
        else:
            # Differentiate row slices r_j(x) with respect to x
            new_cols = list(self.approx.cols)
            new_rows = [_chebtech_diff_physical(r, xa, xb, k) for r in self.approx.rows]

        new_approx = SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=self.approx.pivots,
            domain=self.domain,
        )
        return Chebfun2(approx=new_approx)

    def sum(self, dim: Optional[int] = None) -> Union["Chebfun2", jax.Array]:
        """Integrate f over one or both dimensions.

        Parameters
        ----------
        dim : int or None, optional
            - ``dim=None``: double integral (returns scalar).  Same as ``sum2()``.
            - ``dim=1``: integrate over y; returns a Chebfun2 with rank equal
              to the original rank, but where each column slice has been replaced
              by its integral (a constant), effectively returning a function of
              x only.  The result evaluates to g(x) = Σ_j d_j * int_ya^yb c_j(y) dy * r_j(x).
            - ``dim=2``: integrate over x; returns a function of y only.

        Returns
        -------
        Chebfun2 or jax.Array (scalar)
            - If ``dim=None``: a scalar (double integral).
            - If ``dim=1`` or ``dim=2``: a ``Chebfun2`` with collapsed
              col/row slices representing the 1D result.

        Raises
        ------
        ValueError
            If dim is not None, 1, or 2.

        Notes
        -----
        For ``dim=1`` or ``dim=2``, the returned Chebfun2 has flat slices in
        one direction (all column slices are constant=1, or all row slices are
        constant=1) and the accumulated integral weights are absorbed into the
        remaining pivots.  Evaluation along the collapsed dimension always
        returns the same value (the integral), as expected.

        Provenance
        ----------
        MATLAB source : @separableApprox/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum2, diff
        """
        if dim is None:
            return self.sum2()
        if dim not in (1, 2):
            raise ValueError(
                f"Chebfun2.sum: dim must be None, 1 (integrate over y), "
                f"or 2 (integrate over x), got dim={dim}."
            )

        xa, xb, ya, yb = self.domain
        r = self.approx.rank

        if dim == 1:
            # Integrate over y: g(x) = Σ_j d_j * int_ya^yb c_j(y) dy * r_j(x)
            # Compute col integrals (scalars)
            col_integrals = jnp.array(
                [float(_chebtech_integral(self.approx.cols[j], ya, yb)) for j in range(r)],
                dtype=jnp.float64,
            )
            # New pivots absorb the column integrals: d_j' = d_j * int(c_j)
            # New columns: constant = 1 (Chebtech2 with coeffs = [1])
            # New rows: same as before
            new_pivots = self.approx.pivots * col_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]
            new_rows = list(self.approx.rows)
        else:
            # Integrate over x: g(y) = Σ_j d_j * int_xa^xb r_j(x) dx * c_j(y)
            row_integrals = jnp.array(
                [float(_chebtech_integral(self.approx.rows[j], xa, xb)) for j in range(r)],
                dtype=jnp.float64,
            )
            new_pivots = self.approx.pivots * row_integrals
            one_coeffs = jnp.ones(1, dtype=jnp.float64)
            new_cols = list(self.approx.cols)
            new_rows = [Chebtech2.from_coeffs(one_coeffs) for _ in range(r)]

        new_approx = SeparableApprox(
            cols=new_cols,
            rows=new_rows,
            pivots=new_pivots,
            domain=self.domain,
        )
        return Chebfun2(approx=new_approx)

    def sum2(self) -> jax.Array:
        """Double integral of f over its domain.

        Computes  integral_xa^xb integral_ya^yb f(x, y) dy dx.

        Returns
        -------
        jax.Array (scalar)
            The double integral.

        Notes
        -----
        Uses the low-rank representation:
            I = Σ_j d_j * integral(c_j over [ya, yb]) * integral(r_j over [xa, xb])

        Provenance
        ----------
        MATLAB source : @separableApprox/sum2.m, @separableApprox/integral2.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, diff, norm
        """
        xa, xb, ya, yb = self.domain
        r = self.approx.rank
        total = jnp.float64(0.0)
        for j in range(r):
            col_int = _chebtech_integral(self.approx.cols[j], ya, yb)
            row_int = _chebtech_integral(self.approx.rows[j], xa, xb)
            total = total + self.approx.pivots[j] * col_int * row_int
        return total

    def norm(self, p: Union[int, float, str] = "fro") -> jax.Array:
        """Norm of f.

        Parameters
        ----------
        p : int, float, or str, default ``'fro'``
            The norm type:
            - ``2`` or ``'fro'``: Frobenius (L2) norm,
              ``sqrt(integral_domain |f(x,y)|^2 dx dy)``.
            - ``jnp.inf`` or ``float('inf')``: not implemented (raises
              ``NotImplementedError``).

        Returns
        -------
        jax.Array (scalar)
            The norm.

        Raises
        ------
        NotImplementedError
            If p is not 2 or 'fro'.

        Notes
        -----
        The Frobenius norm is computed as::

            ||f||_F^2 = Σ_j Σ_k d_j * d_k
                          * <c_j, c_k>_[ya,yb]  * <r_j, r_k>_[xa,xb]

        where the inner products use the L2 inner product on the physical
        domain with the affine-map scale factor.

        Provenance
        ----------
        MATLAB source : @separableApprox/norm.m (delegated from @chebfun2/norm.m)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum2
        """
        if p not in (2, "fro", 2.0):
            raise NotImplementedError(
                f"Chebfun2.norm: only the Frobenius/L2 norm (p=2 or p='fro') "
                f"is implemented, got p={p!r}."
            )
        xa, xb, ya, yb = self.domain
        r = self.approx.rank
        # Scale factors for physical inner products
        col_scale = jnp.float64((yb - ya) / 2.0)
        row_scale = jnp.float64((xb - xa) / 2.0)

        norm_sq = jnp.float64(0.0)
        for j in range(r):
            for k in range(r):
                # <c_j, c_k> on reference [-1,1] scaled for physical domain
                col_ip = self.approx.cols[j].inner(self.approx.cols[k]) * col_scale
                # <r_j, r_k> on reference [-1,1] scaled for physical domain
                row_ip = self.approx.rows[j].inner(self.approx.rows[k]) * row_scale
                norm_sq = norm_sq + self.approx.pivots[j] * self.approx.pivots[k] * col_ip * row_ip

        return jnp.sqrt(jnp.abs(norm_sq))

    # ------------------------------------------------------------------
    # Root finding
    # ------------------------------------------------------------------

    def roots(self):
        """Zero curves of f as a list of complex-valued Chebfun contours.

        Returns the zero level set of ``f`` as parametrized curves
        ``c(t) = x(t) + 1i*y(t)`` for ``t in [-1, 1]`` (MATLAB
        ``roots(f)``): a rank-1 ``f`` gives the horizontal/vertical lines
        through the roots of its 1D slices, and a higher-rank ``f`` is
        traced by marching squares and refined to near machine precision by
        a complex-Newton polish.  The list is the chebfunjax stand-in for
        MATLAB's quasimatrix of zero contours; per-curve quantities such as
        arc length (``sum(abs(diff(c)))``) and enclosed area
        (``sum(real(c).*diff(imag(c)))``) are obtained by iterating the
        list.

        Returns
        -------
        list of Chebfun
            One complex Chebfun per connected zero contour (empty if ``f``
            does not change sign in the domain).

        Notes
        -----
        NOT JIT-safe (marching squares + adaptive construction).

        Provenance
        ----------
        MATLAB source : @separableApprox/roots.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        diff, sum
        """
        from chebfunjax.chebfun2d.zerocurves import zero_curves
        return zero_curves(self)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Arithmetic (MATLAB @separableApprox plus/minus/times/rdivide/power)
    # Added by Claude Fable 5: Chebfun2 previously had NO arithmetic.
    # ------------------------------------------------------------------

    def _const_like(self, c) -> "Chebfun2":
        """Rank-1 constant Chebfun2 with this function's domain."""
        one = Chebtech2(coeffs=jnp.ones(1, dtype=jnp.float64), ishappy=True)
        approx = SeparableApprox(
            cols=[one], rows=[one], pivots=jnp.asarray([c]),
            domain=self.approx.domain)
        return Chebfun2(approx=approx)

    def _compress(self) -> "Chebfun2":
        """Recompress the low-rank representation (MATLAB compression).

        Orthonormalizes the column and row quasimatrices by a
        quadrature-weighted QR of their values on a common Chebyshev
        grid, forms the small core matrix, and truncates its SVD at
        machine precision.  This is the step MATLAB's
        @separableApprox/plus.m performs after concatenation; without it
        f - f keeps cancelling terms and norm(f - f) is sqrt(eps)-level
        instead of ~0.  Added by Claude Fable 5.
        """
        import numpy as _np

        from chebfunjax.tech.chebtech import _coeffs_to_values
        from chebfunjax.utils.quadrature import chebweights
        ap = self.approx
        r = len(ap.cols)
        if r <= 1:
            return self

        def _vals(funs):
            # common grid of 2*nmax points: CC quadrature is then exact
            # for pairwise products of the underlying polynomials
            n = 2 * max(int(f.n) for f in funs)
            cols = []
            for f in funs:
                c = _np.zeros(n, dtype=_np.asarray(f.coeffs).dtype)
                c[: int(f.n)] = _np.asarray(f.coeffs)
                cols.append(_np.asarray(_coeffs_to_values(jnp.asarray(c))))
            return n, _np.stack(cols, axis=1)

        nc, vc = _vals(ap.cols)
        nr, vr = _vals(ap.rows)
        wc = _np.sqrt(_np.asarray(chebweights(nc, kind=2), dtype=float))
        wr = _np.sqrt(_np.asarray(chebweights(nr, kind=2), dtype=float))
        qc, rc = _np.linalg.qr(wc[:, None] * vc)      # economy QR
        qr_, rr_ = _np.linalg.qr(wr[:, None] * vr)
        d = _np.asarray(ap.pivots)
        core = rc @ _np.diag(d) @ rr_.T               # plain transpose:
        # the reconstruction f = sum_j d_j c_j(y) r_j(x) has no conjugate
        u, sig, wh = _np.linalg.svd(core, full_matrices=False)
        scale = float(sig[0]) if sig.size else 0.0
        keep = sig > 10 * _np.finfo(float).eps * max(scale, 1e-300)
        if not bool(_np.any(keep)):
            zero = Chebtech2(coeffs=jnp.zeros(1, dtype=jnp.float64),
                             ishappy=True)
            approx = SeparableApprox(cols=[zero], rows=[zero],
                                     pivots=jnp.asarray([0.0]),
                                     domain=ap.domain)
            return Chebfun2(approx=approx)
        u = u[:, keep]
        w = wh.conj().T[:, keep]
        sig = sig[keep]
        new_col_vals = (qc / wc[:, None]) @ u          # back to function values
        new_row_vals = (qr_ / wr[:, None]) @ w.conj()
        from chebfunjax.tech.chebtech import _values_to_coeffs

        def _mk(vals_mat):
            out = []
            for m in range(vals_mat.shape[1]):
                cf = _values_to_coeffs(jnp.asarray(vals_mat[:, m]))
                out.append(Chebtech2(coeffs=cf, ishappy=True))
            return out
        approx = SeparableApprox(cols=_mk(new_col_vals),
                                 rows=_mk(new_row_vals),
                                 pivots=jnp.asarray(sig),
                                 domain=ap.domain)
        return Chebfun2(approx=approx)

    def _check_same_domain(self, other: "Chebfun2") -> None:
        if tuple(self.approx.domain) != tuple(other.approx.domain):
            raise ValueError(
                "Chebfun2 arithmetic requires matching domains: "
                f"{self.approx.domain} vs {other.approx.domain}")

    def __neg__(self) -> "Chebfun2":
        approx = SeparableApprox(
            cols=list(self.approx.cols), rows=list(self.approx.rows),
            pivots=-self.approx.pivots, domain=self.approx.domain)
        return Chebfun2(approx=approx)

    def __add__(self, other) -> "Chebfun2":
        """f + g by exact concatenation of the low-rank terms.

        MATLAB @separableApprox/plus.m concatenates the CDR factors (and
        then compresses); the uncompressed union used here represents the
        sum exactly, only with rank(f)+rank(g) terms.

        Provenance
        ----------
        MATLAB source : @separableApprox/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebfun2):
            self._check_same_domain(other)
            approx = SeparableApprox(
                cols=list(self.approx.cols) + list(other.approx.cols),
                rows=list(self.approx.rows) + list(other.approx.rows),
                pivots=jnp.concatenate(
                    [jnp.atleast_1d(self.approx.pivots),
                     jnp.atleast_1d(other.approx.pivots)]),
                domain=self.approx.domain)
            return Chebfun2(approx=approx)._compress()
        if isinstance(other, (int, float, complex)):
            return self + self._const_like(other)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other) -> "Chebfun2":
        if isinstance(other, Chebfun2):
            return self + (-other)
        if isinstance(other, (int, float, complex)):
            return self + self._const_like(-other)
        return NotImplemented

    def __rsub__(self, other) -> "Chebfun2":
        return (-self) + other

    def __mul__(self, other) -> "Chebfun2":
        """Scalar multiply scales the pivots (exact); f.*g re-approximates
        the pointwise product with the constructor, exactly as MATLAB
        @separableApprox/times.m does.

        Provenance
        ----------
        MATLAB source : @separableApprox/times.m, mtimes.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            approx = SeparableApprox(
                cols=list(self.approx.cols), rows=list(self.approx.rows),
                pivots=self.approx.pivots * other,
                domain=self.approx.domain)
            return Chebfun2(approx=approx)
        if isinstance(other, Chebfun2):
            self._check_same_domain(other)
            return Chebfun2.from_function(
                lambda x, y: self(x, y) * other(x, y),
                domain=self.approx.domain)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Chebfun2":
        if isinstance(other, (int, float, complex)):
            return self * (1.0 / other)
        if isinstance(other, Chebfun2):
            self._check_same_domain(other)
            return Chebfun2.from_function(
                lambda x, y: self(x, y) / other(x, y),
                domain=self.approx.domain)
        return NotImplemented

    def __rtruediv__(self, other) -> "Chebfun2":
        return Chebfun2.from_function(
            lambda x, y: other / self(x, y), domain=self.approx.domain)

    def cumsum(self, dim: int = 1) -> "Chebfun2":
        """Indefinite integral over y (dim=1, default) or x (dim=2),
        vanishing at the lower domain edge (MATLAB cumsum).

        Provenance
        ----------
        MATLAB source : @chebfun2/cumsum.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb = self.domain
        if dim == 1:
            new_cols = [_chebtech_cumsum_physical(c, ya, yb)
                        for c in self.approx.cols]
            new_rows = list(self.approx.rows)
        elif dim == 2:
            new_cols = list(self.approx.cols)
            new_rows = [_chebtech_cumsum_physical(r, xa, xb)
                        for r in self.approx.rows]
        else:
            raise ValueError("dim must be 1 or 2")
        return Chebfun2(approx=SeparableApprox(
            cols=new_cols, rows=new_rows, pivots=self.approx.pivots,
            domain=self.approx.domain))

    def cumsum2(self) -> "Chebfun2":
        """Double indefinite integral (MATLAB cumsum2).

        Provenance
        ----------
        MATLAB source : @chebfun2/cumsum2.m
        Chebfun commit: 7574c77
        """
        return self.cumsum(1).cumsum(2)

    def restrict(self, dom):
        """Restrict to a subdomain (MATLAB restrict / {}-indexing).

        ``dom`` is ``(xa, xb, ya, yb)``.  Degenerate intervals collapse
        dimensions: a point returns a float; a line returns a Chebfun in
        the surviving variable.  A Chebfun ``dom`` is treated as a
        complex path t + 1i*s(t) and returns f along that path.

        Provenance
        ----------
        MATLAB source : @chebfun2/restrict.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain

        if hasattr(dom, "funs"):  # a Chebfun path
            a, b = float(dom.domain.a), float(dom.domain.b)
            return Chebfun.from_function(
                lambda t: self(jnp.real(dom(t)), jnp.imag(dom(t))),
                Domain((a, b)))
        xa, xb, ya, yb = (float(v) for v in dom)
        x_pt = xa == xb
        y_pt = ya == yb
        if x_pt and y_pt:
            return float(self(jnp.asarray(xa), jnp.asarray(ya)))
        if x_pt:
            return Chebfun.from_function(
                lambda y: self(jnp.full_like(y, xa), y),
                Domain((ya, yb)))
        if y_pt:
            return Chebfun.from_function(
                lambda x: self(x, jnp.full_like(x, ya)),
                Domain((xa, xb)))
        return Chebfun2.from_function(
            lambda x, y: self(x, y), domain=(xa, xb, ya, yb))

    def squeeze(self):
        """Collapse dimensions along which f is constant, returning a
        Chebfun if possible (MATLAB squeeze).

        Provenance
        ----------
        MATLAB source : @chebfun2/squeeze.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
        xa, xb, ya, yb = self.domain
        xs = jnp.asarray(_np.linspace(xa, xb, 33))
        ys = jnp.asarray(_np.linspace(ya, yb, 33))
        xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
        vals = self(xx, yy)
        vs = max(float(jnp.max(jnp.abs(vals))), 1.0)
        tol = 1e4 * float(_np.finfo(float).eps) * vs
        var_x = float(jnp.max(vals.max(axis=0) - vals.min(axis=0)))
        var_y = float(jnp.max(vals.max(axis=1) - vals.min(axis=1)))
        if var_y < tol and var_x < tol:
            ymid = 0.5 * (ya + yb)
            return Chebfun.from_function(
                lambda x: self(x, jnp.full_like(x, ymid)),
                Domain((xa, xb)))
        if var_y < tol:  # constant in y -> function of x
            ymid = 0.5 * (ya + yb)
            return Chebfun.from_function(
                lambda x: self(x, jnp.full_like(x, ymid)),
                Domain((xa, xb)))
        if var_x < tol:  # constant in x -> function of y
            xmid = 0.5 * (xa + xb)
            return Chebfun.from_function(
                lambda y: self(jnp.full_like(y, xmid), y),
                Domain((ya, yb)))
        return self

    def mean2(self) -> jax.Array:
        """Mean value over the domain (MATLAB mean2; Fable 5)."""
        xa, xb, ya, yb = self.approx.domain
        return self.sum2() / ((xb - xa) * (yb - ya))

    def mean(self, dim: int = 1) -> "Chebfun2":
        """Average over one variable (MATLAB mean; constant in the
        averaged variable, like sum(dim) normalized)."""
        xa, xb, ya, yb = self.approx.domain
        L = (yb - ya) if dim == 1 else (xb - xa)
        return self.sum(dim=dim) * (1.0 / L)

    def std2(self) -> jax.Array:
        """Standard deviation over the domain (MATLAB std2)."""
        mu = float(self.mean2())
        var = (self - mu) * (self - mu)
        return jnp.sqrt(var.mean2())

    def diag_fun(self):
        """The 1-D chebfun g(x) = f(x, x) on the diagonal (MATLAB diag).

        Provenance
        ----------
        MATLAB source : @chebfun2/diag.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        xa, xb, ya, yb = self.approx.domain
        a = max(xa, ya)
        b = min(xb, yb)
        return _cf(lambda t: self(t, t), domain=(a, b))

    def trace(self) -> jax.Array:
        """int f(x, x) dx (MATLAB trace)."""
        return self.diag_fun().sum()

    def fliplr(self) -> "Chebfun2":
        """f(-x, y) about the vertical midline (MATLAB fliplr)."""
        xa, xb, ya, yb = self.approx.domain
        return Chebfun2.from_function(
            lambda x, y: self(xa + xb - x, y),
            domain=self.approx.domain)

    def flipud(self) -> "Chebfun2":
        """f(x, -y) about the horizontal midline (MATLAB flipud)."""
        xa, xb, ya, yb = self.approx.domain
        return Chebfun2.from_function(
            lambda x, y: self(x, ya + yb - y),
            domain=self.approx.domain)

    def minandmax2(self, ngrid: int | None = None, n_starts: int = 24):
        """Global minimum and maximum over the domain (MATLAB
        ``minandmax2``).

        A tensor grid seeds candidate extrema; the extrema are then found by
        polishing the ``n_starts`` best grid candidates with a
        bound-constrained quasi-Newton solve (``scipy`` L-BFGS-B) that uses
        the exact JAX gradient of the evaluation, keeping the best result.
        The multi-start is what makes this robust: a single polish from the
        best grid point converges to whatever critical point is nearest and,
        for oscillatory functions (e.g. ``cos(k*pi*x*y^2)*cos(k*pi*y*x^2)``),
        that is frequently a NON-global extremum -- the previous
        single-start Newton reached only ~1e-3 accuracy on those.  Trying the
        deepest handful of grid basins recovers the true global optimum to
        near machine precision.

        Parameters
        ----------
        ngrid : int or None
            Seed-grid size per axis.  ``None`` (default) picks a grid that
            resolves the function from its Chebyshev slice degrees.
        n_starts : int, default 24
            Number of distinct grid candidates polished per extremum.

        Returns
        -------
        (vals, locs) : vals = [min, max], locs = [[x*, y*] x 2]

        Provenance
        ----------
        MATLAB source : @separableApprox/minandmax2.m
        Chebfun commit: 7574c77
        """
        import jax
        import numpy as _np
        from scipy.optimize import minimize

        xa, xb, ya, yb = self.approx.domain

        # Grid resolution: sample a few points per Chebyshev mode so the
        # seed grid resolves the function's oscillations (MATLAB seeds at the
        # slice chebpts, i.e. full representation resolution).
        if ngrid is None:
            try:
                deg = max(
                    max(len(c.coeffs) for c in self.approx.cols),
                    max(len(r.coeffs) for r in self.approx.rows),
                )
            except Exception:
                deg = 33
            ngrid = int(min(max(2 * deg + 1, 65), 513))

        gx = _np.linspace(xa, xb, ngrid)
        gy = _np.linspace(ya, yb, ngrid)
        XX, YY = _np.meshgrid(gx, gy, indexing="ij")
        V = _np.asarray(self(jnp.asarray(XX), jnp.asarray(YY)),
                        dtype=_np.float64)

        def _scal(p):
            return self(p[0], p[1])

        vg = jax.jit(jax.value_and_grad(_scal))

        def _optimize(which):
            sign = 1.0 if which == "min" else -1.0
            order = _np.argsort((sign * V).ravel())      # best-first

            def fun(p):
                v, g = vg(jnp.asarray(p, dtype=jnp.float64))
                return sign * float(v), sign * _np.asarray(g, dtype=_np.float64)

            best_signed = _np.inf
            best_x = (float(XX.ravel()[order[0]]),
                      float(YY.ravel()[order[0]]))
            seen = set()
            starts = 0
            for idx_flat in order:
                if starts >= n_starts:
                    break
                i0 = _np.unravel_index(idx_flat, V.shape)
                p0 = (float(XX[i0]), float(YY[i0]))
                key = (round(p0[0], 2), round(p0[1], 2))
                if key in seen:
                    continue
                seen.add(key)
                starts += 1
                res = minimize(
                    fun, _np.array(p0), jac=True, method="L-BFGS-B",
                    bounds=[(xa, xb), (ya, yb)],
                    options={"ftol": 1e-15, "gtol": 1e-14, "maxiter": 400})
                if float(res.fun) < best_signed:
                    best_signed = float(res.fun)
                    best_x = (float(res.x[0]), float(res.x[1]))
            return sign * best_signed, best_x

        vmin, xmin = _optimize("min")
        vmax, xmax = _optimize("max")
        return (jnp.asarray([vmin, vmax], dtype=jnp.float64),
                jnp.asarray([[xmin[0], xmin[1]], [xmax[0], xmax[1]]],
                            dtype=jnp.float64))

    def max2(self):
        """Global maximum (value, [x, y]) -- MATLAB max2."""
        vals, locs = self.minandmax2()
        return vals[1], locs[1]

    def min2(self):
        """Global minimum (value, [x, y]) -- MATLAB min2."""
        vals, locs = self.minandmax2()
        return vals[0], locs[0]

    def compose(self, op) -> "Chebfun2":
        """Re-approximate op(f(x, y)) (MATLAB compose; Fable 5)."""
        return Chebfun2.from_function(
            lambda x, y: op(self(x, y)), domain=self.approx.domain)

    def exp(self):
        return self.compose(jnp.exp)

    def sin(self):
        return self.compose(jnp.sin)

    def cos(self):
        return self.compose(jnp.cos)

    def sqrt(self):
        return self.compose(jnp.sqrt)

    def log(self):
        return self.compose(jnp.log)

    def tanh(self):
        return self.compose(jnp.tanh)

    def abs(self):
        return self.compose(jnp.abs)

    def grad(self) -> "object":
        """Gradient of the scalar field: the Chebfun2v [f_x; f_y].

        Provenance
        ----------
        MATLAB source : @chebfun2/grad.m, @chebfun2/gradient.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
        fx = self.diff(2, 1)   # d/dx
        fy = self.diff(1, 1)   # d/dy
        return Chebfun2v([fx.approx, fy.approx])

    def gradient(self) -> "object":
        """Alias of :meth:`grad` (MATLAB gradient.m)."""
        return self.grad()

    def laplacian(self) -> "Chebfun2":
        """Laplacian f_xx + f_yy.

        Provenance
        ----------
        MATLAB source : @chebfun2/laplacian.m, @chebfun2/lap.m
        Chebfun commit: 7574c77
        """
        return self.diff(2, 2) + self.diff(1, 2)

    def lap(self) -> "Chebfun2":
        """Alias of :meth:`laplacian` (MATLAB lap.m)."""
        return self.laplacian()

    def real(self) -> "Chebfun2":
        """Real part, re-approximated adaptively (a complex Chebfun2's
        real part is not directly available from its low-rank slices).

        Provenance
        ----------
        MATLAB source : @chebfun2/real.m
        Chebfun commit: 7574c77
        """
        return Chebfun2.from_function(
            lambda x, y: jnp.real(self(x, y)),
            domain=self.approx.domain)

    def imag(self) -> "Chebfun2":
        """Imaginary part.

        Provenance
        ----------
        MATLAB source : @chebfun2/imag.m
        Chebfun commit: 7574c77
        """
        return Chebfun2.from_function(
            lambda x, y: jnp.imag(self(x, y)),
            domain=self.approx.domain)

    def conj(self) -> "Chebfun2":
        """Complex conjugate.

        Provenance
        ----------
        MATLAB source : @chebfun2/conj.m
        Chebfun commit: 7574c77
        """
        return Chebfun2.from_function(
            lambda x, y: jnp.conj(self(x, y)),
            domain=self.approx.domain)

    def __pow__(self, p) -> "Chebfun2":
        return Chebfun2.from_function(
            lambda x, y: self(x, y) ** p, domain=self.approx.domain)

    def plot(self, **kwargs):
        """Surface plot of this Chebfun2 (calls :func:`chebfunjax.plotting.surf`)."""
        from chebfunjax.plotting import surf
        return surf(self, **kwargs)

    def surf(self, **kwargs):
        """Surface plot of this Chebfun2 (calls :func:`chebfunjax.plotting.surf`)."""
        from chebfunjax.plotting import surf
        return surf(self, **kwargs)

    def contour(self, **kwargs):
        """Contour plot of this Chebfun2 (calls :func:`chebfunjax.plotting.contour`)."""
        from chebfunjax.plotting import contour
        return contour(self, **kwargs)

    def quiver(self, g2=None, **kwargs):
        """Quiver plot of gradient field (calls :func:`chebfunjax.plotting.quiver_2d`)."""
        from chebfunjax.plotting import quiver_2d
        return quiver_2d(self, g2, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display like MATLAB Chebfun2.

        Examples
        --------
        >>> f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        >>> repr(f)
        'Chebfun2(rank=2, domain=(-1.0, 1.0, -1.0, 1.0))'
        """
        xa, xb, ya, yb = self.domain
        return (
            f"Chebfun2(rank={self.rank}, "
            f"domain=({xa}, {xb}, {ya}, {yb}))"
        )


# ============================================================================
# Factory function
# ============================================================================


def chebfun2(
    f: Callable[[jax.Array, jax.Array], jax.Array],
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    tol: Optional[float] = None,
    n: Optional[int] = None,
) -> Chebfun2:
    """Construct a Chebfun2 representing a bivariate smooth function.

    This is the primary factory function for creating Chebfun2 objects.

    Parameters
    ----------
    f : callable
        A function f(x, y) that accepts 2D JAX arrays (from meshgrid) and
        returns an array of the same shape.  Must be vectorised.
    domain : tuple of 4 floats, optional
        (xa, xb, ya, yb). Default is (-1, 1, -1, 1).
    tol : float, optional
        Target relative tolerance. Default is machine epsilon (~2.2e-16).
    n : int, optional
        Fixed-degree construction (not yet implemented).

    Returns
    -------
    Chebfun2

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import chebfunjax as cj
    >>> f = cj.chebfun2(lambda x, y: jnp.cos(x + y))
    >>> f(0.5, -0.3)  # evaluate at a point
    Array(0.20..., dtype=float64)

    Provenance
    ----------
    MATLAB source : @chebfun2/chebfun2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun2
    """
    return Chebfun2.from_function(f, domain=domain, tol=tol, n=n)


# ============================================================================
# Empty propagation (MATLAB emptyObjects semantics, Fable 5 audit)
# ============================================================================

from chebfunjax.utils.misc import make_empty_aware  # noqa: E402

make_empty_aware(Chebfun2, [
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__",
    "__rmul__", "__truediv__", "__pow__", "__neg__",
    "sqrt", "sum", "norm", "squeeze", "diff", "cos", "sin", "exp",
    "log", "tanh", "abs", "diag_fun", "trace", "mean", "mean2",
    "std2", "minandmax2", "max2", "min2", "fliplr", "flipud",
    "cumsum", "cumsum2", "sum2", "integral2", "restrict", "compose",
])
