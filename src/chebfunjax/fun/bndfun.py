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
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts, legpts

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
        exponents: tuple[float, float] | None = None,
    ) -> "Bndfun":
        """Construct a Bndfun from a callable on [a, b].

        The callable ``f`` is evaluated at Chebyshev-2 points mapped to
        [a, b].  If ``n`` is ``None`` (default), an adaptive algorithm
        doubles the grid size until the Chebyshev coefficients decay to
        machine precision.

        When ``exponents`` is supplied the ``onefun`` is built as a
        :class:`~chebfunjax.fun.singfun.Singfun`, giving a *singular* Bndfun
        ``s(x) * (x - a)^e0 * (b - x)^e1`` where ``s`` is smooth.  This
        mirrors MATLAB's ``bndfun(op, data, pref)`` with
        ``data.exponents = [e0 e1]`` (and ``pref.blowup``), which routes the
        remapped operator to the ``singfun`` onefun constructor.

        Parameters
        ----------
        f : callable
            Vectorised function accepting and returning ``jax.Array``.
        domain : Domain
            A single-interval domain [a, b] (``domain.n_intervals == 1``).
        n : int or None, optional
            Fixed number of Chebyshev points.  ``None`` triggers adaptive
            construction.
        exponents : tuple of two floats, optional
            ``(e0, e1)`` algebraic exponents at the left (``a``) and right
            (``b``) endpoints.  When given (and not ``(0, 0)``) the onefun
            is a Singfun.  These are the exponents in the *mapped* [-1, 1]
            frame, which coincide with the physical-endpoint exponents
            because the map is affine (``x - a ∝ 1 + y`` and
            ``b - x ∝ 1 - y``).

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
        if exponents is not None and (exponents[0] != 0.0 or exponents[1] != 0.0):
            from chebfunjax.fun.singfun import Singfun

            onefun = Singfun.from_function(mapped_f, exponents, n=n)
            return cls(onefun=onefun, domain=domain)
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
        if self.isempty():
            return type(self).empty()
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
        # Chebtech2.restrict takes two scalar endpoints; Singfun.restrict
        # takes an increasing sequence (a partition).  A singular onefun
        # restricted to a subinterval that avoids the singular endpoint
        # returns a bare smooth piece.
        from chebfunjax.fun.singfun import Singfun

        if isinstance(self.onefun, Singfun):
            new_onefun = self.onefun.restrict([float(t_a), float(t_b)])
        else:
            new_onefun = self.onefun.restrict(float(t_a), float(t_b))

        return Bndfun(onefun=new_onefun, domain=new_domain)

    # ------------------------------------------------------------------
    # QR factorisation of an array-valued (quasimatrix) Bndfun
    # ------------------------------------------------------------------

    def qr(self, mode: str = "matrix") -> tuple["Bndfun", jax.Array]:
        """Abstract QR factorisation ``f = Q R`` of a quasimatrix.

        For an array-valued Bndfun (``m`` columns), returns ``Q`` — an
        array-valued Bndfun whose columns are orthonormal in the L2 inner
        product on [a, b] — and ``R``, an ``m x m`` upper-triangular matrix,
        such that ``f = Q @ R`` column-wise and ``Q.inner(Q) == I``.

        Uses the Gauss-Legendre weighted discrete QR of MATLAB's built-in
        method: Chebyshev nodal values are mapped to Legendre nodes, scaled
        by ``sqrt(w_leg)``, factored with a dense QR, then mapped back.  The
        diagonal of ``R`` is forced non-negative.  Finally the Bndfun layer
        rescales by ``sqrt((b - a) / 2)`` (``Q /= s``, ``R *= s``) to move
        the orthonormality from [-1, 1] to [a, b].

        Parameters
        ----------
        mode : str, default "matrix"
            Accepted for MATLAB parity (permutation-output flag).  The
            built-in method performs no column pivoting, so ``mode`` has no
            effect on the two-output form.

        Returns
        -------
        Q : Bndfun
            Array-valued Bndfun with L2-orthonormal columns.
        R : jax.Array, shape (m, m)
            Upper-triangular factor.

        NOT JIT-safe (dense linear algebra with data-dependent shapes).

        Provenance
        ----------
        MATLAB source : @bndfun/qr.m and @chebtech/qr.m (built-in method)
        Chebfun commit: 7574c77
        """
        rescale = float(self.domain.map_derivative())  # (b-a)/2
        s = jnp.sqrt(jnp.float64(rescale))

        coeffs = self.onefun.coeffs
        if coeffs.ndim == 1:
            # Single-column quasimatrix: R = sqrt(<f, f>), Q = f / R.
            r_scalar = jnp.sqrt(self.onefun.inner(self.onefun))
            q_onefun = self.onefun / r_scalar
            q_bnd = Bndfun.from_chebtech(q_onefun, self.domain)
            R = jnp.reshape(r_scalar, (1, 1))
            return q_bnd / s, R * s

        # Array-valued case.
        values = self.onefun.coeffs2vals(coeffs)  # (n, m)
        n, m = values.shape
        if n < m:
            # Prolong to n = m by zero-padding the Chebyshev coefficients.
            pad = jnp.zeros((m - n, m), dtype=coeffs.dtype)
            coeffs = jnp.concatenate([coeffs, pad], axis=0)
            values = self.onefun.coeffs2vals(coeffs)
            n = m

        xc = chebpts(n, kind=2)
        vc = self.onefun.barywts(n)
        xl, wl, vl = legpts(n, bary=True)
        sqrt_wl = jnp.sqrt(wl)
        # P: Chebyshev-nodal values -> Legendre-nodal values.
        P = barymat(xl, xc, vc)
        # Pinv: Legendre-nodal values -> Chebyshev-nodal values.
        Pinv = barymat(xc, xl, vl)
        WP = sqrt_wl[:, None] * P
        invWP = Pinv * (1.0 / sqrt_wl)[None, :]

        converted = WP @ values  # (n, m)
        Q, R = jnp.linalg.qr(converted, mode="reduced")  # (n, m), (m, m)

        diag_R = jnp.diagonal(R)
        sign = jnp.where(diag_R == 0, jnp.ones_like(diag_R), jnp.sign(diag_R))
        Q = invWP @ (Q * sign[None, :])  # (n, m) Chebyshev-nodal values
        R = sign[:, None] * R

        q_coeffs = self.onefun.vals2coeffs(Q)
        q_onefun = Chebtech2.from_coeffs(q_coeffs)
        q_bnd = Bndfun.from_chebtech(q_onefun, self.domain)
        return q_bnd / s, R * s

    # ------------------------------------------------------------------
    # Linear change of variable
    # ------------------------------------------------------------------

    def change_map(self, newdom) -> "Bndfun":
        r"""Map the domain of this Bndfun via a linear change of variable.

        ``G = f.change_map(newdom)``, where ``f`` has domain ``[a, b]``,
        returns a Bndfun ``G`` defined on ``[c, d] = newdom`` such that

        .. math::
            G(x) = f\!\left(a\frac{d - x}{d - c}
                            + b\frac{x - c}{d - c}\right),
            \quad x \in [c, d].

        The underlying Chebyshev representation (``onefun`` on ``[-1, 1]``)
        is left untouched; only the affine map to the physical interval is
        replaced.  This is an *exact* reparametrisation (no resampling).

        Parameters
        ----------
        newdom : Domain or sequence of two floats
            The new interval ``[c, d]``.

        Returns
        -------
        Bndfun
            The same shape re-mapped onto ``newdom``.

        Notes
        -----
        NOT JIT-safe (construction-level operation).

        Provenance
        ----------
        MATLAB source : @bndfun/changeMap.m
        Chebfun commit: 7574c77
        """
        if isinstance(newdom, Domain):
            new_domain = newdom
        else:
            vals = [float(v) for v in newdom]
            new_domain = Domain((vals[0], vals[-1]))
        _validate_single_domain(new_domain)
        return Bndfun(onefun=self.onefun, domain=new_domain)

    # ------------------------------------------------------------------
    # Convolution
    # ------------------------------------------------------------------

    def conv(self, g: "Bndfun") -> list:
        r"""Convolution of two Bndfuns.

        Computes

        .. math::
            h(x) = \int f(t)\, g(x - t)\, dt,
            \quad x \in [a + c,\, b + d],

        where ``self`` is on ``[a, b]`` and ``g`` is on ``[c, d]``.  The
        convolution of two smooth functions on bounded intervals is
        piecewise smooth with breakpoints at the pairwise sums of the
        endpoints, so the result is returned as a *list* of Bndfun pieces
        (one per subinterval of ``[a + c, b + d]``), mirroring the cell
        array returned by MATLAB's ``@bndfun/conv``.

        Each piece is built by adaptive Chebyshev construction, evaluating
        the convolution integral by Gauss-Kronrod quadrature over the
        sub-intervals induced by the breakpoints of ``f`` and ``g``.  This
        is the general quadrature route (MATLAB's ``oldConv`` path); it is
        exact to quadrature tolerance for arbitrary smooth ``f`` and ``g``.

        Parameters
        ----------
        g : Bndfun
            The second operand, on a bounded domain.

        Returns
        -------
        list of Bndfun
            The convolution pieces, left to right on ``[a + c, b + d]``.
            An empty input on either side yields an empty list.

        Notes
        -----
        NOT JIT-safe (adaptive construction + adaptive quadrature).

        Provenance
        ----------
        MATLAB source : @bndfun/conv.m (Hale & Townsend, 2014); the
            quadrature fallback follows @chebfun/conv.m ``oldConv``.
        Chebfun commit: 7574c77
        """
        import numpy as _np

        if self.isempty() or g.isempty():
            return []

        f = self
        a, b = float(f.domain.a), float(f.domain.b)
        c, d = float(g.domain.a), float(g.domain.b)
        if not all(_np.isfinite([a, b, c, d])):
            raise ValueError("conv: only bounded domains are supported.")

        # Breakpoints of the convolution are the pairwise sums of endpoints.
        pts = _np.unique(_np.array([a + c, b + c, a + d, b + d]))
        span = max(abs(pts[0]), abs(pts[-1]))
        tol = 10.0 * float(_np.finfo(_np.float64).eps) * span
        if tol == 0.0:
            tol = 1e-14
        keep = _np.concatenate([[True], _np.diff(pts) > tol])
        pts = pts[keep]

        f_bps = _np.array([a, b])
        g_bps = _np.array([c, d])

        def _conv_at(x_val: float) -> float:
            from scipy import integrate as _scint
            a_lim = max(a, x_val - d)
            b_lim = min(b, x_val - c)
            if a_lim >= b_lim:
                return 0.0
            ends_g = x_val - g_bps
            int_bps = _np.union1d(f_bps, ends_g)
            int_bps = int_bps[(int_bps >= a_lim) & (int_bps <= b_lim)]
            sub_dom = _np.unique(_np.concatenate([[a_lim], int_bps, [b_lim]]))
            result = 0.0
            x_jax = jnp.float64(x_val)
            for j in range(len(sub_dom) - 1):
                def integrand(t):
                    t_arr = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64))
                    return float((f(t_arr) * g(x_jax - t_arr))[0])
                val, _ = _scint.quad(integrand, sub_dom[j], sub_dom[j + 1],
                                     epsabs=1e-13, epsrel=1e-13, limit=100)
                result += val
            return result

        pieces = []
        for k in range(len(pts) - 1):
            lo, hi = float(pts[k]), float(pts[k + 1])
            piece = Bndfun.from_function(
                lambda x: jnp.array(
                    [_conv_at(float(xi)) for xi in jnp.atleast_1d(x)],
                    dtype=jnp.float64,
                ),
                Domain((lo, hi)),
            )
            pieces.append(piece)
        return pieces

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
