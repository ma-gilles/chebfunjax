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

    @classmethod
    def from_padua(
        cls,
        vals,
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    ) -> "Chebfun2":
        """Construct a Chebfun2 from data sampled at the Padua points.

        MATLAB equivalent: ``chebfun2(f, dom, 'padua')`` where ``f`` are the
        values of a function at ``paduapts(n, dom)`` (Padua2DM ordering).  The
        degree ``n`` is inferred from ``len(vals) == (n+1)(n+2)/2``.

        Parameters
        ----------
        vals : array_like, shape ((n+1)(n+2)/2,)
            Function values at ``paduapts(n, domain)``.
        domain : tuple of 4 floats, optional
            ``(xa, xb, ya, yb)``.  Default is ``(-1, 1, -1, 1)``.

        Returns
        -------
        Chebfun2

        Notes
        -----
        The Padua interpolant is a total-degree ``n`` bivariate polynomial;
        it is stored exactly in the low-rank representation via an SVD of its
        values on the ``(n+1) x (n+1)`` Chebyshev tensor grid.  NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebfun2/constructor.m ('padua' branch),
            @chebfun2/paduaVals2coeffs.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        from_function, chebfunjax.chebfun2d.padua.paduapts
        """
        import numpy as _np

        from chebfunjax.chebfun2d.padua import paduavals2coeffs
        from chebfunjax.tech.chebtech import _values_to_coeffs

        if len(domain) != 4:
            raise ValueError(
                "Chebfun2.from_padua: domain must have exactly 4 elements "
                f"(xa, xb, ya, yb), got {len(domain)}."
            )
        _, V = paduavals2coeffs(vals, domain)
        # V rows/cols are in descending 2nd-kind node order; flip both axes
        # to chebfunjax's ascending grid.  V[i, j] = g(x_j, y_i).
        Vasc = _np.asarray(V)[::-1, ::-1]

        u, sig, vh = _np.linalg.svd(Vasc, full_matrices=False)
        scale = float(sig[0]) if sig.size else 0.0
        keep = sig > 1e3 * _np.finfo(float).eps * max(scale, 1e-300)
        if not bool(_np.any(keep)):
            zero = Chebtech2(coeffs=jnp.zeros(1, dtype=jnp.float64),
                             ishappy=True)
            approx = SeparableApprox(cols=[zero], rows=[zero],
                                     pivots=jnp.asarray([0.0]),
                                     domain=tuple(float(v) for v in domain))
            return cls(approx=approx)
        u = u[:, keep]          # column slices (functions of y)
        w = vh[keep, :].T       # row slices (functions of x)
        sig = sig[keep]

        def _mk(vals_mat):
            out = []
            for k in range(vals_mat.shape[1]):
                cf = _values_to_coeffs(jnp.asarray(vals_mat[:, k]))
                out.append(Chebtech2(coeffs=cf, ishappy=True))
            return out

        approx = SeparableApprox(
            cols=_mk(_np.asarray(u)),
            rows=_mk(_np.asarray(w)),
            pivots=jnp.asarray(sig, dtype=jnp.float64),
            domain=tuple(float(v) for v in domain),
        )
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

    def on_curve(self, c) -> "object":
        """Restrict f to a complex Chebfun curve (MATLAB f(c)).

        ``c`` is a complex-valued 1D Chebfun parametrizing the curve
        ``t -> (real c(t), imag c(t))``; the result is the 1D Chebfun
        ``t -> f(real c(t), imag c(t))`` on c's domain.

        Provenance
        ----------
        MATLAB source : @separableApprox/feval.m (chebfun-curve branch)
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import chebfun

        a, b = float(c.domain.a), float(c.domain.b)

        def ev(t):
            z = c(t)
            return self(jnp.real(z), jnp.imag(z))

        return chebfun(ev, domain=(a, b))

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

    def svd(self) -> jax.Array:
        r"""Singular values of the Chebfun2 (as a Hilbert-Schmidt kernel).

        Returns the singular values of ``f`` in decreasing order.  The number
        returned equals the rank of the low-rank representation.

        Algorithm (identical to MATLAB @separableApprox/svd.m)::

            f = C D R'                 (low-rank / cdr representation)
            C = Q_C R_C                (quasimatrix QR in the y inner product)
            R = Q_R R_R                (quasimatrix QR in the x inner product)
            f = Q_C (R_C D R_R') Q_R'
            singular values of f = singular values of  R_C D R_R'

        The quasimatrix QRs are realised as ordinary QRs of the column/row
        slice *values* on a common Chebyshev grid, weighted by the square
        root of the Clenshaw-Curtis quadrature weights and the physical
        affine-map scale, so that ``Q^H Q = I`` in the physical L^2 inner
        product.  The core ``R_C D R_R'`` is then an ``r x r`` matrix whose
        singular values are those of ``f``.

        Returns
        -------
        jax.Array, shape (rank,)
            Singular values in decreasing order (non-negative reals).

        Notes
        -----
        NOT JIT-safe (uses numpy QR/SVD on the small core).

        Provenance
        ----------
        MATLAB source : @separableApprox/svd.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        norm, rank
        """
        import numpy as _np

        from chebfunjax.tech.chebtech import _coeffs_to_values
        from chebfunjax.utils.quadrature import chebweights
        ap = self.approx
        xa, xb, ya, yb = self.domain
        col_scale = float((yb - ya) / 2.0)
        row_scale = float((xb - xa) / 2.0)

        d = _np.asarray(ap.pivots)
        if _np.linalg.norm(d) == 0.0:
            return jnp.zeros((1,), dtype=jnp.float64)

        def _weighted_vals(funs, scale):
            # common grid of 2*nmax points: Clenshaw-Curtis quadrature is
            # then exact for pairwise products of the underlying polynomials.
            n = 2 * max(int(f.n) for f in funs)
            mat = []
            for f in funs:
                c = _np.zeros(n, dtype=_np.asarray(f.coeffs).dtype)
                c[: int(f.n)] = _np.asarray(f.coeffs)
                mat.append(_np.asarray(_coeffs_to_values(jnp.asarray(c))))
            vals = _np.stack(mat, axis=1)
            w = _np.sqrt(_np.asarray(chebweights(n, kind=2), dtype=float)
                         * scale)
            return w[:, None] * vals

        wc_vc = _weighted_vals(ap.cols, col_scale)
        wr_vr = _weighted_vals(ap.rows, row_scale)
        _, rc = _np.linalg.qr(wc_vc)              # economy QR
        _, rr = _np.linalg.qr(wr_vr)
        # Plain transpose (not conjugate): the reconstruction
        # f = sum_j d_j c_j(y) r_j(x) carries no conjugate on the rows.
        core = rc @ _np.diag(d) @ rr.T
        sig = _np.linalg.svd(core, compute_uv=False)
        return jnp.asarray(sig, dtype=jnp.float64)

    def norm(self, p: Union[int, float, str] = "fro") -> jax.Array:
        """Norm of f.

        Parameters
        ----------
        p : int, float, or str, default ``'fro'``
            The norm type:

            - ``'fro'`` (or omitted): Frobenius (L2) norm,
              ``sqrt(integral_domain |f(x,y)|^2 dx dy)`` = ``sqrt(sum(svd**2))``.
            - ``2``, ``'op'``, ``'operator'``: spectral (operator) norm, the
              largest singular value.
            - ``'nuc'``, ``'nuclear'``: nuclear norm, the sum of singular
              values.
            - ``jnp.inf``, ``'inf'``, ``'max'``: global maximum of ``|f|``.
            - an even integer ``p``: the ``p``-norm
              ``(sum2(f**p))**(1/p)`` (real, even ``p`` only, as in MATLAB).

        Returns
        -------
        jax.Array (scalar)
            The norm.

        Raises
        ------
        NotImplementedError
            If ``p`` is ``1``, ``'min'``, ``-inf``, or an odd/non-integer
            numeric value (MATLAB raises the same restrictions).

        Provenance
        ----------
        MATLAB source : @separableApprox/norm.m (delegated from @chebfun2/norm.m)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        svd, sum2, minandmax2
        """
        # Frobenius / L2 norm: sqrt(sum of squared singular values).  Kept as
        # the direct quadratic form because it is exact and does not need the
        # numpy QR/SVD path.
        if p in ("fro", "F"):
            return self._norm_fro()

        if p in (2, 2.0, "op", "operator"):
            return jnp.asarray(self.svd()[0], dtype=jnp.float64)

        if p in ("nuc", "nuclear"):
            return jnp.sum(self.svd())

        if p in (jnp.inf, float("inf"), "inf", "max"):
            if self._is_real():
                vals, _ = self.minandmax2()
                return jnp.max(jnp.abs(vals))
            # complex: max of |f| = sqrt(max of conj(f)*f); build |f|^2 as a
            # real-valued Chebfun2 so the optimiser sees real outputs.
            g = Chebfun2.from_function(
                lambda x, y: jnp.abs(self(x, y)) ** 2,
                domain=self.approx.domain)
            vals, _ = g.minandmax2()
            return jnp.sqrt(jnp.max(jnp.abs(vals)))

        if p in (1, "1"):
            raise NotImplementedError(
                "Chebfun2.norm: the L1 norm (p=1) is not supported "
                "(matches MATLAB)."
            )
        if p in (-jnp.inf, float("-inf"), "-inf", "min"):
            raise NotImplementedError(
                "Chebfun2.norm: the 'min' norm is not supported "
                "(matches MATLAB)."
            )

        # Even numeric p: (sum2(f**p))**(1/p).
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            pf = float(p)
            if abs(round(pf) - pf) < _EPS:
                ip = int(round(pf))
                if ip % 2 == 0:
                    val = jnp.asarray((self ** ip).sum2())
                    # An even power of a real-valued function integrates to a
                    # real number; drop negligible imaginary noise left by the
                    # complex low-rank reconstruction so the p-norm stays real.
                    if jnp.iscomplexobj(val) and \
                            abs(float(val.imag)) <= 1e-10 * abs(float(val.real)):
                        val = jnp.real(val)
                    return val ** (1.0 / ip)
                raise NotImplementedError(
                    "Chebfun2.norm: p-norm must have p even for now "
                    "(matches MATLAB)."
                )
        raise NotImplementedError(
            f"Chebfun2.norm: unknown norm p={p!r}."
        )

    def _is_real(self) -> bool:
        """True if the low-rank representation is real-valued.

        Checks the dtypes (and, for complex-typed data, the magnitude of the
        imaginary parts) of the pivots and column/row slice coefficients.
        Not JIT-safe (runs on concrete arrays).
        """
        import numpy as _np
        parts = [_np.asarray(self.approx.pivots)]
        parts += [_np.asarray(c.coeffs) for c in self.approx.cols]
        parts += [_np.asarray(r.coeffs) for r in self.approx.rows]
        for a in parts:
            if _np.iscomplexobj(a) and _np.any(_np.abs(a.imag) > 0):
                return False
        return True

    def _norm_fro(self) -> jax.Array:
        """Frobenius/L2 norm via the exact quadratic form over the pivots."""
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

    def roots(self, g=None, method="ms"):
        """Zero curves of f, or common zeros of f and g.

        With no argument returns the zero curves of ``f`` as complex-valued
        Chebfun contours.  With a second Chebfun2 ``g`` (MATLAB
        ``roots(f, g)``) returns the isolated common zeros as an ``(m, 2)``
        array of ``[x, y]`` points.

        ``method`` selects the common-zero engine (MATLAB ``roots(f, g,
        method)``): ``'ms'`` / ``'marchingsquares'`` (the default) traces the
        zero curves and polishes with 2D Newton, while ``'resultant'`` uses the
        hidden-variable Bezout resultant of
        :func:`chebfunjax.chebfun2d.resultant.resultant_common_zeros`.  Both
        return the same isolated common zeros to Newton accuracy.

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
        from chebfunjax.chebfun2d.zerocurves import (
            common_zeros,
            zero_curves,
        )
        if g is not None:
            other = g.approx if isinstance(g, Chebfun2) else g
            other = Chebfun2(approx=other)
            m = method.lower()
            if m in ("ms", "marchingsquares"):
                return common_zeros(self, other)
            if m == "resultant":
                from chebfunjax.chebfun2d.resultant import (
                    resultant_common_zeros,
                )
                return resultant_common_zeros(self, other)
            raise ValueError(
                f"Chebfun2.roots: unknown method {method!r} "
                "(expected 'ms', 'marchingsquares', or 'resultant').")
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

    def fevalm(self, x, y) -> jax.Array:
        """Evaluate on the tensor grid ``meshgrid(x, y)`` (MATLAB fevalm).

        ``Z = f.fevalm(x, y)`` returns a matrix ``Z`` of size
        ``len(y)`` by ``len(x)`` with ``Z[i, j] = f(x[j], y[i])``.  This is
        equivalent to ``feval`` on ``meshgrid(x, y)`` but exploits the
        separable representation: it evaluates the row slices at ``x`` and the
        column slices at ``y`` once and combines them, which is much cheaper
        than forming the full mesh.

        Parameters
        ----------
        x : array_like, shape (nx,)
            x-coordinates (a 1-D vector).
        y : array_like, shape (ny,)
            y-coordinates (a 1-D vector).

        Returns
        -------
        jax.Array, shape (ny, nx)
            The grid of values, ``Z[i, j] = f(x[j], y[i])``.

        Provenance
        ----------
        MATLAB source : @separableApprox/fevalm.m, @chebfun2/fevalm.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        __call__
        """
        if self.isempty():
            return jnp.zeros((0, 0), dtype=jnp.float64)
        xr = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)).reshape(-1)
        yr = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float64)).reshape(-1)
        # meshgrid('xy'): XX[i, j] = xr[j], YY[i, j] = yr[i]  -> (ny, nx).
        XX, YY = jnp.meshgrid(xr, yr)
        return self(XX, YY)

    def isequal(self, other: "Chebfun2") -> bool:
        """Equality test up to relative machine precision (MATLAB isequal).

        Returns ``True`` iff ``self`` and ``other`` represent the same
        function to relative machine precision (same domain and
        ``norm(self - other)`` negligible relative to ``norm(self)``).

        Provenance
        ----------
        MATLAB source : @chebfun2/isequal.m, @separableApprox/isequal.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if not isinstance(other, Chebfun2):
            return False
        if self.isempty() or other.isempty():
            return self.isempty() and other.isempty()
        if tuple(self.approx.domain) != tuple(other.approx.domain):
            return False
        scale = float(self.norm("fro"))
        tol = 1e4 * _EPS * max(scale, 1.0)
        return bool(float((self - other).norm("fro")) <= tol)

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

    def sample(self, m: int, n: int) -> jax.Array:
        """Values of f on an m-by-n tensor Chebyshev grid.

        Returns the (n, m) matrix ``V[j, i] = f(x_i, y_j)`` where ``x`` are
        the ``m`` 2nd-kind Chebyshev points in the x-direction and ``y`` the
        ``n`` points in the y-direction (MATLAB's row-per-y layout).

        Provenance
        ----------
        MATLAB source : @separableApprox/sample.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.quadrature import chebpts_ab

        xa, xb, ya, yb = self.domain
        x = chebpts_ab(m, xa, xb, kind=2)
        y = chebpts_ab(n, ya, yb, kind=2)
        X, Y = jnp.meshgrid(x, y)
        return self(X, Y)

    def _extremum(self, g, dim: int, reducer) -> "object":
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.domain import Domain

        if g is not None:
            raise ValueError(
                "Unable to maximize/minimize two Chebfun2 objects.")
        if dim == 0:
            raise ValueError(
                "Dimension argument must be a positive integer scalar "
                "within indexing range.")
        if dim not in (1, 2):
            # MATLAB returns f itself for out-of-range dims (like max()).
            return self
        xa, xb, ya, yb = self.domain
        n = 2049  # MATLAB's sampling resolution
        vals = self.sample(n, n)
        if dim == 1:  # extremum over y -> function of x
            v = reducer(vals, 0)
            return Chebfun.from_values(v, Domain((xa, xb)))
        v = reducer(vals, 1)  # over x -> function of y
        return Chebfun.from_values(v, Domain((ya, yb)))

    def max(self, g=None, dim: int = 1):
        """Maximum along one variable, as a 1D Chebfun (MATLAB max).

        ``max(f)`` / ``max(f, dim=1)`` maximizes over y and returns a
        function of x; ``dim=2`` maximizes over x.  (MATLAB returns a row
        chebfun for ``dim=1``; orientation is not tracked here.)  For a
        ``dim`` outside 1-2, f itself is returned, as in MATLAB.

        Provenance
        ----------
        MATLAB source : @separableApprox/max.m
        Chebfun commit: 7574c77
        """
        return self._extremum(g, dim, lambda v, ax: jnp.max(v, axis=ax))

    def min(self, g=None, dim: int = 1):
        """Minimum along one variable, as a 1D Chebfun (MATLAB min).

        Provenance
        ----------
        MATLAB source : @separableApprox/min.m
        Chebfun commit: 7574c77
        """
        return self._extremum(g, dim, lambda v, ax: jnp.min(v, axis=ax))

    def std(self, flag=None, dim: int = 1):
        """Standard deviation along one variable, as a 1D Chebfun.

        ``std(f)`` (``dim=1``) is the y-standard-deviation, a function of
        x; ``dim=2`` the x-standard-deviation, a function of y.  ``flag``
        is accepted and ignored to mirror MATLAB's ``std(f, flag, dim)``.

        Provenance
        ----------
        MATLAB source : @separableApprox/std.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.domain import Domain
        from chebfunjax.utils.quadrature import chebpts_ab

        if dim not in (1, 2):
            raise ValueError("std dim must be 1 or 2.")
        xa, xb, ya, yb = self.domain
        m = self.mean(dim=dim)
        width = (yb - ya) if dim == 1 else (xb - xa)
        var = ((self - m) ** 2).sum(dim=dim) * (1.0 / width)
        # var is a Chebfun2 flat in the averaged variable; sample it along
        # the remaining variable and take the pointwise square root.
        n = 513
        if dim == 1:
            t = chebpts_ab(n, xa, xb, kind=2)
            v = var(t, jnp.full_like(t, 0.5 * (ya + yb)))
            dom = Domain((xa, xb))
        else:
            t = chebpts_ab(n, ya, yb, kind=2)
            v = var(jnp.full_like(t, 0.5 * (xa + xb)), t)
            dom = Domain((ya, yb))
        return Chebfun.from_values(jnp.sqrt(jnp.maximum(v, 0.0)), dom)

    @classmethod
    def complex(cls, re: "Chebfun2", im: "Chebfun2" = None) -> "Chebfun2":
        """Complex Chebfun2 from real (and imaginary) parts (MATLAB complex).

        ``complex(f)`` requires a real f and returns it (f + 0i);
        ``complex(f, g)`` requires real f, g and returns f + 1i*g.

        Provenance
        ----------
        MATLAB source : @separableApprox/complex.m
        Chebfun commit: 7574c77
        """
        if im is not None:
            if not isinstance(im, Chebfun2):
                raise TypeError("Second input must be a Chebfun2.")
            return re + 1j * im
        return re

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
