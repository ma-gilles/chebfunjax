"""Operator and functional blocks for spectral discretization of ODEs.

Provides ``OperatorBlock`` (linear map function -> function) and
``FunctionalBlock`` (linear map function -> scalar), together with factory
functions for the most common building blocks:

- ``D(domain, order)``         -- differentiation operator
- ``I(domain)``                -- identity operator
- ``diag(f)``                  -- multiplication by a Chebfun ``f``
- ``eval_at(x, domain)``       -- point-evaluation functional
- ``sum_functional(domain)``   -- definite-integral functional

These are used as building blocks inside a :class:`ChebMatrix`.

Translated from MATLAB Chebfun classes ``@linBlock``, ``@operatorBlock``,
and ``@functionalBlock`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import jax.numpy as jnp

from chebfunjax.utils.diffmat import cumsummat, diffmat
from chebfunjax.utils.quadrature import chebpts, chebpts_ab, chebweights

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = jnp.ndarray
_DomainT = tuple[float, ...]

_DEFAULT_DOMAIN: _DomainT = (-1.0, 1.0)


# ===========================================================================
# Discretization descriptor
# ===========================================================================


class ChebColloc2Disc:
    """Chebyshev-collocation-2 discretization descriptor.

    Carries the information that ``OperatorBlock.matrix`` and
    ``FunctionalBlock.matrix`` need: the grid size(s) and the physical
    domain.  The domain may carry interior breakpoints, in which case the
    discretization is a *piecewise* collocation grid — one Chebyshev grid
    per subinterval, concatenated left to right.  This mirrors MATLAB's
    ``chebcolloc2`` discretization, whose ``dimension`` property is a
    vector with one entry per subinterval.

    Parameters
    ----------
    n : int or sequence of int
        Number of collocation points.  A scalar is broadcast to every
        subinterval; a sequence gives one size per subinterval.
    domain : tuple of float
        Physical breakpoints ``[a, ..., b]``.  Length 2 means a single
        interval.

    Attributes
    ----------
    sizes : list[int]
        Grid size on each subinterval.
    n : int
        Total dimension ``sum(sizes)``.
    domain : tuple of float
        The breakpoints.

    Provenance
    ----------
    MATLAB source : @chebcolloc2/chebcolloc2.m, @opDiscretization/opDiscretization.m
    Chebfun commit: 7574c77
    """

    def __init__(
        self,
        n: Union[int, Sequence[int]],
        domain: _DomainT = _DEFAULT_DOMAIN,
    ) -> None:
        domain = tuple(float(v) for v in domain)
        n_int = len(domain) - 1
        if isinstance(n, (int,)) or (hasattr(n, "__index__")
                                     and not isinstance(n, (list, tuple))):
            sizes = [int(n)] * n_int
        else:
            sizes = [int(v) for v in n]
            if len(sizes) != n_int:
                raise ValueError(
                    f"ChebColloc2Disc: got {len(sizes)} dimensions for "
                    f"{n_int} subintervals.")
        self.sizes = sizes
        self.n = int(sum(sizes))
        self.domain = domain

    # ------------------------------------------------------------------

    @property
    def num_intervals(self) -> int:
        """Number of subintervals."""
        return len(self.domain) - 1

    @property
    def intervals(self) -> list[tuple[float, float]]:
        """List of ``(a_k, b_k)`` subintervals."""
        return [(self.domain[k], self.domain[k + 1])
                for k in range(self.num_intervals)]

    def points(self) -> Array:
        """Concatenated physical collocation points (MATLAB ``functionPoints``)."""
        parts = [chebpts_ab(nk, a, b, kind=2)
                 for nk, (a, b) in zip(self.sizes, self.intervals)]
        return jnp.concatenate([jnp.asarray(p, dtype=jnp.float64)
                                for p in parts])

    def offsets(self) -> list[int]:
        """Starting index of each subinterval block (length ``num_intervals + 1``)."""
        out = [0]
        for s in self.sizes:
            out.append(out[-1] + s)
        return out

    def which_interval(self, location: float, direction: int = 0) -> int:
        """Index of the subinterval owning ``location`` approached from
        ``direction`` (-1 left, +1 right, 0 don't care).

        Provenance
        ----------
        MATLAB source : @opDiscretization/whichInterval.m
        Chebfun commit: 7574c77
        """
        dom = self.domain
        if len(dom) == 2:
            return 0
        loc = float(location)
        idx = 0
        for k, b in enumerate(dom):
            if loc >= b:
                idx = k
        # ``idx`` is the 0-based index of the last breakpoint <= loc.
        if idx == len(dom) - 1:
            direction = -1
        length = dom[-1] - dom[0]
        if direction < 0 and abs(loc - dom[idx]) < 10 * 2.220446049250313e-16 * length:
            idx -= 1
        return idx


# ===========================================================================
# OperatorBlock — linear operator mapping function to function
# ===========================================================================


class OperatorBlock:
    """A linear operator that maps a function to a function.

    An ``OperatorBlock`` stores a *lazy* representation: the callable
    ``op_fn`` that, when given a :class:`ChebColloc2Disc` descriptor,
    returns the ``n x n`` collocation matrix representing the operator.

    The ``order`` attribute records the differential order of the operator
    (0 = multiplication, 1 = first derivative, 2 = second derivative, etc.).

    Parameters
    ----------
    op_fn : callable(disc) -> jnp.ndarray
        A function that accepts a ``ChebColloc2Disc`` and returns the
        ``n x n`` matrix representing the operator at that discretization.
    order : int, default 0
        Differential order of the operator (e.g. 0 for identity/mult, 1 for D).
    domain : (float, float), default (-1, 1)
        Physical domain of the operator.

    Examples
    --------
    Build the first-derivative operator on [0, 1]:

    >>> op = D((0.0, 1.0))
    >>> disc = ChebColloc2Disc(8, (0.0, 1.0))
    >>> Dmat = op.matrix(disc)
    >>> Dmat.shape
    (8, 8)

    Notes
    -----
    ``OperatorBlock`` objects are **not** Equinox modules — operator
    construction is always outside JIT.  The ``matrix`` method itself
    calls ``diffmat`` / ``cumsummat`` which are JAX computations but the
    assembly logic is Python-level.

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    FunctionalBlock, ChebMatrix, D, I, diag
    """

    def __init__(
        self,
        op_fn: Callable[[ChebColloc2Disc], Array],
        order: int = 0,
        domain: _DomainT = _DEFAULT_DOMAIN,
        apply_fn: Callable | None = None,
        iszero: bool = False,
        isnotdiffint: bool = False,
        coeff_fn: Callable[[], list] | None = None,
    ) -> None:
        self._op_fn = op_fn
        self.order = order
        self.domain = tuple(float(v) for v in domain)
        # Optional coefficient realization (MATLAB blockCoeff): a thunk
        # returning [a_m, ..., a_0] with a_k the coefficient of u^(k).
        self._coeff_fn = coeff_fn
        # Optional function-space action u |-> (A u), used when the block is
        # applied to a Chebfun (D*g) rather than composed/realized as a
        # matrix.  ``None`` for composite blocks that only carry a matrix.
        self._apply_fn = apply_fn
        # MATLAB linBlock bookkeeping flags (used for linearity detection and
        # for deciding whether a block needs downsampling).
        self.iszero = bool(iszero)
        self.isnotdiffint = bool(isnotdiffint)

    @property
    def diff_order(self) -> int:
        """MATLAB ``diffOrder`` -- differential order of the block."""
        return self.order

    def to_function(self):
        """Return the function-space action as a plain callable
        (MATLAB ``toFunction``).

        Provenance
        ----------
        MATLAB source : @linBlock/toFunction.m
        Chebfun commit: 7574c77
        """
        if self._apply_fn is None:
            raise TypeError(
                "This OperatorBlock has no function-space action.")
        return self._apply_fn

    def coeff_list(self) -> list:
        """Coefficients ``[a_m, ..., a_0]`` of this differential operator.

        Provenance
        ----------
        MATLAB source : @linBlock/toCoeff.m, @blockCoeff/blockCoeff.m
        Chebfun commit: 7574c77
        """
        if self._coeff_fn is None:
            raise TypeError(
                "This OperatorBlock has no coefficient realization; "
                "conversion of integration or evaluation to coefficients "
                "is not supported.")
        return self._coeff_fn()

    def apply(self, u):
        """Apply the operator to a function ``u`` (returns a function).

        Only defined for atomic blocks that carry a function-space action
        (multiplication ``diag(f)``, derivative ``D``, identity ``I``, and
        their scalar multiples / sums / compositions)."""
        if self._apply_fn is None:
            raise TypeError(
                "This OperatorBlock has no function-space action; it can "
                "only be realized as a matrix via .matrix(n).")
        return self._apply_fn(u)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def matrix(self, disc: Union[ChebColloc2Disc, int]) -> Array:
        """Discretize the operator as an ``n x n`` matrix.

        Parameters
        ----------
        disc : ChebColloc2Disc or int
            Either a full discretization descriptor or an integer ``n``
            (in which case ``self.domain`` is used to build a default
            ``ChebColloc2Disc(n, self.domain)``).

        Returns
        -------
        M : jnp.ndarray, shape (n, n)
            The collocation matrix for this operator.
        """
        if isinstance(disc, int):
            disc = ChebColloc2Disc(disc, self.domain)
        return self._op_fn(disc)

    # ------------------------------------------------------------------
    # Operator algebra — returns new OperatorBlocks
    # ------------------------------------------------------------------

    def __add__(self, other: "OperatorBlock") -> "OperatorBlock":
        """Operator addition: ``(A + B)*u = A*u + B*u``."""
        if isinstance(other, (int, float)):
            other = _scalar_op(other, self.domain)
        if isinstance(other, FunctionalBlock):
            other = other.promote()
        _check_domains(self, other)
        new_order = max(self.order, other.order)
        domain = merge_domains(self.domain, other.domain)
        af, bf = self._apply_fn, other._apply_fn
        ac, bc = self._coeff_fn, other._coeff_fn

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) + other.matrix(disc)

        return OperatorBlock(
            _fn, order=new_order, domain=domain,
            apply_fn=(None if (af is None or bf is None)
                      else (lambda u: af(u) + bf(u))),
            iszero=(self.iszero and other.iszero),
            isnotdiffint=(self.isnotdiffint and other.isnotdiffint),
            coeff_fn=(None if (ac is None or bc is None)
                      else (lambda: _coeff_add(ac(), bc()))))

    def __radd__(self, other: "OperatorBlock") -> "OperatorBlock":
        return self.__add__(other)

    def __sub__(self, other: "OperatorBlock") -> "OperatorBlock":
        """Operator subtraction: ``(A - B)*u = A*u - B*u``."""
        if isinstance(other, (int, float)):
            other = _scalar_op(other, self.domain)
        if isinstance(other, FunctionalBlock):
            other = other.promote()
        _check_domains(self, other)
        new_order = max(self.order, other.order)
        domain = merge_domains(self.domain, other.domain)
        af, bf = self._apply_fn, other._apply_fn
        ac, bc = self._coeff_fn, other._coeff_fn

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) - other.matrix(disc)

        return OperatorBlock(
            _fn, order=new_order, domain=domain,
            apply_fn=(None if (af is None or bf is None)
                      else (lambda u: af(u) - bf(u))),
            iszero=(self.iszero and other.iszero),
            isnotdiffint=(self.isnotdiffint and other.isnotdiffint),
            coeff_fn=(None if (ac is None or bc is None)
                      else (lambda: _coeff_add(ac(),
                                               [-v for v in bc()]))))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = _scalar_op(other, self.domain)
        return other.__sub__(self)

    def __mul__(self, other) -> "OperatorBlock":
        """Operator composition, scalar scaling, or application to a function.

        - ``A * B`` composes operators (matrix product in discretization).
        - ``A * c`` scales the operator by scalar ``c``.
        - ``A * u`` (u a Chebfun) applies the operator, returning a Chebfun.
        """
        if isinstance(other, (int, float, complex)):
            c = complex(other) if isinstance(other, complex) else float(other)
            domain = self.domain
            af = self._apply_fn
            isz = self.iszero or (c == 0)

            def _fn(disc: ChebColloc2Disc) -> Array:
                return c * self.matrix(disc)

            ac = self._coeff_fn
            return OperatorBlock(
                _fn, order=(0 if isz else self.order), domain=domain,
                apply_fn=(None if af is None else (lambda u: c * af(u))),
                iszero=isz, isnotdiffint=self.isnotdiffint,
                coeff_fn=(None if ac is None
                          else (lambda: [c * v for v in ac()])))

        if isinstance(other, FunctionalBlock):
            other = other.promote()

        if isinstance(other, OperatorBlock):
            _check_domains(self, other)
            isz = self.iszero or other.iszero
            new_order = 0 if isz else self.order + other.order
            domain = merge_domains(self.domain, other.domain)
            af, bf = self._apply_fn, other._apply_fn

            def _fn(disc: ChebColloc2Disc) -> Array:
                return self.matrix(disc) @ other.matrix(disc)

            ac, bc = self._coeff_fn, other._coeff_fn
            return OperatorBlock(
                _fn, order=new_order, domain=domain,
                apply_fn=(None if (af is None or bf is None)
                          else (lambda u: af(bf(u)))),
                iszero=isz,
                isnotdiffint=((self.isnotdiffint and other.isnotdiffint)
                              or isz),
                coeff_fn=(None if (ac is None or bc is None)
                          else (lambda: _coeff_mul(ac(), bc()))))

        # Application to a function (Chebfun): A * u -> A(u).
        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(other, Chebfun):
            return self.apply(other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, c):
        """Division by a scalar (MATLAB ``mrdivide``)."""
        if isinstance(c, (int, float)):
            return self.__mul__(1.0 / float(c))
        return NotImplemented

    def __neg__(self) -> "OperatorBlock":
        """Unary minus."""
        domain = self.domain
        af = self._apply_fn

        def _fn(disc: ChebColloc2Disc) -> Array:
            return -self.matrix(disc)

        ac = self._coeff_fn
        return OperatorBlock(
            _fn, order=self.order, domain=domain,
            apply_fn=(None if af is None else (lambda u: -af(u))),
            iszero=self.iszero, isnotdiffint=self.isnotdiffint,
            coeff_fn=(None if ac is None else (lambda: [-v for v in ac()])))

    def __pow__(self, k: int) -> "OperatorBlock":
        """Repeated composition: ``A^k = A * A * ... * A`` (k times)."""
        if not (isinstance(k, int) and k >= 0):
            raise ValueError(
                f"OperatorBlock power must be a non-negative integer, got {k!r}."
            )
        result = I(self.domain)
        for _ in range(k):
            result = self * result
        result.iszero = self.iszero
        result.isnotdiffint = self.isnotdiffint
        return result

    def __repr__(self) -> str:
        return f"OperatorBlock(order={self.order}, domain={self.domain})"


# ===========================================================================
# FunctionalBlock — linear operator mapping function to scalar
# ===========================================================================


class FunctionalBlock:
    """A linear functional that maps a function to a scalar.

    A ``FunctionalBlock`` stores a *lazy* representation: the callable
    ``func_fn`` that, when given a :class:`ChebColloc2Disc`, returns the
    ``1 x n`` row vector representing the functional at that discretization.

    Parameters
    ----------
    func_fn : callable(disc) -> jnp.ndarray, shape (n,)
        A function accepting a ``ChebColloc2Disc`` and returning the row
        vector (shape ``(n,)``) representing the functional.
    domain : (float, float), default (-1, 1)
        Physical domain.

    Examples
    --------
    Build the evaluation functional at ``x = 0.5``:

    >>> ev = eval_at(0.5)
    >>> disc = ChebColloc2Disc(8)
    >>> row = ev.matrix(disc)
    >>> row.shape
    (8,)

    Notes
    -----
    ``FunctionalBlock`` rows are used as boundary-condition rows when
    assembling a :class:`ChebMatrix` into a full linear system.

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    OperatorBlock, ChebMatrix, eval_at, sum_functional
    """

    def __init__(
        self,
        func_fn: Callable[[ChebColloc2Disc], Array],
        domain: _DomainT = _DEFAULT_DOMAIN,
        apply_fn: Callable | None = None,
        order: int = 0,
        iszero: bool = False,
        isnotdiffint: bool = False,
    ) -> None:
        self._func_fn = func_fn
        self.domain = tuple(float(v) for v in domain)
        self._apply_fn = apply_fn
        self.order = order
        self.iszero = bool(iszero)
        self.isnotdiffint = bool(isnotdiffint)

    @property
    def diff_order(self) -> int:
        """MATLAB ``diffOrder`` -- differential order of the block."""
        return self.order

    def apply(self, u):
        """Apply the functional to a function ``u`` (returns a scalar)."""
        if self._apply_fn is None:
            raise TypeError(
                "This FunctionalBlock has no function-space action; it can "
                "only be realized as a row via .matrix(n).")
        return self._apply_fn(u)

    def to_function(self):
        """Return the function-space action as a plain callable
        (MATLAB ``toFunction``).

        Provenance
        ----------
        MATLAB source : @linBlock/toFunction.m
        Chebfun commit: 7574c77
        """
        if self._apply_fn is None:
            raise TypeError(
                "This FunctionalBlock has no function-space action.")
        return self._apply_fn

    def promote(self) -> "OperatorBlock":
        """Promote a functional to an operator by replicating its row.

        The promoted block maps ``u`` to the constant function whose value is
        ``F[u]``; in a collocation discretization its matrix is the row of
        ``F`` repeated ``n`` times.

        Provenance
        ----------
        MATLAB source : @functionalBlock/functionalBlock.m (``promote``)
        Chebfun commit: 7574c77
        """
        af = self._apply_fn
        dom = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            row = self.matrix(disc)
            return jnp.tile(row[None, :], (disc.n, 1))

        def _apply(u):
            from chebfunjax.chebfun1d.chebfun import Chebfun
            return Chebfun.from_function(
                lambda t: float(af(u)) * jnp.ones_like(t),
                domain=_as_domain_obj(dom))

        return OperatorBlock(_fn, order=0, domain=dom,
                             apply_fn=(None if af is None else _apply),
                             iszero=self.iszero, isnotdiffint=True)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def matrix(self, disc: Union[ChebColloc2Disc, int]) -> Array:
        """Discretize the functional as a row vector of length ``n``.

        Parameters
        ----------
        disc : ChebColloc2Disc or int
            Discretization descriptor or an integer ``n``.

        Returns
        -------
        r : jnp.ndarray, shape (n,)
            The collocation row for this functional.
        """
        if isinstance(disc, int):
            disc = ChebColloc2Disc(disc, self.domain)
        return self._func_fn(disc)

    # ------------------------------------------------------------------
    # Functional algebra
    # ------------------------------------------------------------------

    def __add__(self, other) -> "FunctionalBlock":
        if isinstance(other, OperatorBlock):
            return self.promote() + other
        if isinstance(other, (int, float)):
            raise TypeError(
                "Cannot add a scalar to a FunctionalBlock directly. "
                "Use eval_at or sum_functional to build scalar functionals."
            )
        domain = merge_domains(self.domain, other.domain)
        af, bf = self._apply_fn, other._apply_fn

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) + other.matrix(disc)

        return FunctionalBlock(
            _fn, domain=domain,
            apply_fn=(None if (af is None or bf is None)
                      else (lambda u: af(u) + bf(u))),
            order=max(self.order, other.order),
            iszero=(self.iszero and other.iszero),
            isnotdiffint=(self.isnotdiffint and other.isnotdiffint))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: "FunctionalBlock") -> "FunctionalBlock":
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other) -> "FunctionalBlock":
        """Scalar multiplication, composition, or application to a function.

        - ``F * c``  (scalar) scales the row.
        - ``F * A``  (OperatorBlock) composes: ``(F*A)[u] = F[A[u]]``,
          i.e. ``row_F @ matrix_A``.
        - ``F * u``  (Chebfun) applies the functional, returning a scalar.
        """
        if isinstance(other, (int, float, complex)):
            c = complex(other) if isinstance(other, complex) else float(other)
            domain = self.domain
            af = self._apply_fn

            def _fn(disc: ChebColloc2Disc) -> Array:
                return c * self.matrix(disc)

            return FunctionalBlock(
                _fn, domain=domain,
                apply_fn=(None if af is None else (lambda u: c * af(u))),
                order=self.order,
                iszero=(self.iszero or c == 0),
                isnotdiffint=self.isnotdiffint)

        if isinstance(other, OperatorBlock):
            domain = merge_domains(self.domain, other.domain)
            af, bf = self._apply_fn, other._apply_fn
            isz = self.iszero or other.iszero

            def _fn(disc: ChebColloc2Disc) -> Array:
                return self.matrix(disc) @ other.matrix(disc)

            return FunctionalBlock(
                _fn, domain=domain,
                apply_fn=(None if (af is None or bf is None)
                          else (lambda u: af(bf(u)))),
                order=self.order + other.order, iszero=isz,
                isnotdiffint=((self.isnotdiffint and other.isnotdiffint)
                              or isz))

        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(other, Chebfun):
            return self.apply(other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> "FunctionalBlock":
        domain = self.domain
        af = self._apply_fn

        def _fn(disc: ChebColloc2Disc) -> Array:
            return -self.matrix(disc)

        return FunctionalBlock(
            _fn, domain=domain,
            apply_fn=(None if af is None else (lambda u: -af(u))),
            order=self.order, iszero=self.iszero,
            isnotdiffint=self.isnotdiffint)

    def __repr__(self) -> str:
        return f"FunctionalBlock(domain={self.domain})"


# ===========================================================================
# Private helpers
# ===========================================================================


def _check_domains(a, b) -> None:
    """Raise if two blocks live on different intervals."""
    if (a.domain[0], a.domain[-1]) != (b.domain[0], b.domain[-1]):
        raise ValueError(
            f"Cannot combine OperatorBlocks with different domains: "
            f"{a.domain} vs {b.domain}. Restrict to a common domain first."
        )


def merge_domains(*doms: _DomainT) -> _DomainT:
    """Union of breakpoint sets, keeping the common outer interval.

    Breakpoints closer together than ``1e-14`` times the interval length are
    treated as the same point.

    Parameters
    ----------
    *doms : tuple of float
        Breakpoint tuples sharing the same endpoints.

    Returns
    -------
    tuple of float

    Examples
    --------
    >>> merge_domains((-1.0, 1.0), (-1.0, 0.0, 1.0))
    (-1.0, 0.0, 1.0)

    Provenance
    ----------
    MATLAB source : @domain/merge.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    vals = sorted({float(v) for d in doms for v in d})
    length = vals[-1] - vals[0]
    tol = 1e-14 * (length if length > 0 else 1.0)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > tol:
            out.append(v)
    return tuple(out)


def _scalar_op(c: float, domain: _DomainT) -> OperatorBlock:
    """Return ``c * I`` as an OperatorBlock."""
    return I(domain) * float(c)


def _blkdiag(mats: list[Array]) -> Array:
    """Block-diagonal assembly of a list of dense matrices."""
    if len(mats) == 1:
        return mats[0]
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = jnp.zeros((rows, cols), dtype=jnp.float64)
    r = c = 0
    for m in mats:
        out = out.at[r:r + m.shape[0], c:c + m.shape[1]].set(m)
        r += m.shape[0]
        c += m.shape[1]
    return out


def _bary_row(n: int, t: float) -> Array:
    """Barycentric interpolation row at reference point ``t`` for ``n``
    Chebyshev points of the second kind on ``[-1, 1]``.

    Provenance
    ----------
    MATLAB source : barymat.m
    Chebfun commit: 7574c77
    """
    pts_ref = chebpts(n, kind=2)
    k = jnp.arange(n, dtype=jnp.float64)
    w = jnp.where((k == 0) | (k == n - 1),
                  0.5 * jnp.ones(n, dtype=jnp.float64),
                  jnp.ones(n, dtype=jnp.float64))
    w = w * jnp.where(k % 2 == 0, 1.0, -1.0)
    dx = t - pts_ref
    close = jnp.abs(dx) < 1e-14
    if bool(jnp.any(close)):
        idx = int(jnp.argmax(close))
        return jnp.zeros(n, dtype=jnp.float64).at[idx].set(1.0)
    num = w / dx
    return num / jnp.sum(num)


def _parse_direction(direction: Union[int, str]) -> int:
    """Normalize an evaluation direction to -1, 0, or +1.

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m (``feval``)
    Chebfun commit: 7574c77
    """
    if isinstance(direction, str):
        low = direction.lower()
        if low.startswith("l") or low.startswith("-"):
            return -1
        if low.startswith("r") or low.startswith("+"):
            return 1
        raise ValueError(
            "CHEBFUN:FUNCTIONALBLOCK:feval:badDirection -- direction must "
            "be 'left', 'right', '+', or '-'.")
    return int(0 if direction == 0 else (1 if direction > 0 else -1))


def _feval_chebfun(u, x: float, side):
    """Evaluate a Chebfun at ``x`` with an optional one-sided limit."""
    val = u(jnp.asarray(float(x), dtype=jnp.float64), side) if side \
        else u(jnp.asarray(float(x), dtype=jnp.float64))
    arr = jnp.asarray(val)
    return float(jnp.ravel(arr)[0]) if arr.size == 1 else arr


def _coeff_diff(c: list) -> list:
    """One differentiation step on a coefficient list (highest order first).

    Provenance
    ----------
    MATLAB source : @blockCoeff/blockCoeff.m (``diff``)
    Chebfun commit: 7574c77
    """
    m = len(c)
    out = [c[0]] + list(c)               # c([1, 1:m])
    for k in range(1, m):                # MATLAB k = 2:m
        out[k] = out[k].diff() + out[k + 1]
    out[m] = out[m].diff()
    return out


def _coeff_add(ca: list, cb: list) -> list:
    """Add two coefficient lists, left-padding the shorter with zeros.

    Provenance
    ----------
    MATLAB source : @blockCoeff/blockCoeff.m (``plus``)
    Chebfun commit: 7574c77
    """
    if len(ca) < len(cb):
        ca = [ca[0] * 0.0] * (len(cb) - len(ca)) + list(ca)
    elif len(cb) < len(ca):
        cb = [cb[0] * 0.0] * (len(ca) - len(cb)) + list(cb)
    return [a + b for a, b in zip(ca, cb)]


def _as_constant(f):
    """Return the scalar value of a Chebfun that is globally constant,
    else ``None``.  Multiplying by such a factor is exact scalar scaling."""
    funs = getattr(f, "funs", None)
    if not funs or any(len(p) != 1 for p in funs):
        return None
    vals = [float(jnp.ravel(jnp.asarray(p.coeffs))[0]) for p in funs]
    return vals[0] if all(v == vals[0] for v in vals) else None


def _coeff_times(a, b):
    """Product of two coefficient entries.

    A constant factor is applied as a scalar scaling, which is exact; this
    mirrors the length-1 shortcut in MATLAB's ``@chebtech/times``.
    """
    ca = _as_constant(a)
    if ca is not None:
        return b * ca
    cb = _as_constant(b)
    if cb is not None:
        return a * cb
    return a * b


def _coeff_mul(ca: list, cb: list) -> list:
    """Coefficient list of the composition ``A*B`` (Leibniz convolution).

    Provenance
    ----------
    MATLAB source : @blockCoeff/blockCoeff.m (``mtimes``)
    Chebfun commit: 7574c77
    """
    c = [_coeff_times(ca[-1], b) for b in cb]
    zero = c[0] * 0.0
    cur = list(cb)
    for j in range(1, len(ca)):
        cur = _coeff_diff(cur)
        c = [zero] + c
        for k in range(len(cur)):
            c[k] = c[k] + _coeff_times(ca[-1 - j], cur[k])
    return c


def _const_chebfun(value: float, dom: _DomainT):
    """Constant Chebfun with the given value on ``dom``."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    return Chebfun.from_function(
        lambda t: float(value) * jnp.ones_like(t), domain=_as_domain_obj(dom))


def _to_values(f, disc: "ChebColloc2Disc") -> Array:
    """Values of ``f`` on the discretization grid, using one-sided limits at
    the interior breakpoints (which are duplicated in the grid).

    Provenance
    ----------
    MATLAB source : @chebcolloc2/toValues.m
    Chebfun commit: 7574c77
    """
    pts = disc.points()
    vals = jnp.ravel(jnp.asarray(f(pts), dtype=jnp.float64))
    offs = disc.offsets()
    for k in range(1, disc.num_intervals):
        idx = offs[k]
        xb = jnp.asarray(disc.domain[k], dtype=jnp.float64)
        vals = vals.at[idx - 1].set(
            jnp.ravel(jnp.asarray(f(xb, "left")))[0])
        vals = vals.at[idx].set(
            jnp.ravel(jnp.asarray(f(xb, "right")))[0])
    return vals


def _as_domain_obj(dom: _DomainT):
    """Wrap a breakpoint tuple in a :class:`~chebfunjax.domain.Domain`."""
    from chebfunjax.domain import Domain
    return Domain(tuple(float(v) for v in dom))


def _domain_of(f) -> _DomainT:
    """Breakpoints of a Chebfun-like object as a plain tuple."""
    dom_obj = getattr(f, "domain", None)
    if dom_obj is None:
        return _DEFAULT_DOMAIN
    if hasattr(dom_obj, "breakpoints"):
        return tuple(float(v) for v in dom_obj.breakpoints)
    return tuple(float(v) for v in dom_obj)


# ===========================================================================
# Factory functions — common operators
# ===========================================================================


def D(domain: _DomainT = _DEFAULT_DOMAIN, order: int = 1) -> OperatorBlock:
    """Differentiation operator of the given order.

    Returns the ``OperatorBlock`` whose ``n x n`` collocation matrix maps
    function values at ``n`` Chebyshev points of the 2nd kind to values of
    the ``order``-th derivative at the same points.

    Parameters
    ----------
    domain : (float, float), default (-1, 1)
        Physical interval ``[a, b]``.
    order : int, default 1
        Differentiation order.  ``order=0`` returns the identity.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> d = D()                        # first derivative on [-1, 1]
    >>> d2 = D(order=2)                # second derivative
    >>> d_ab = D(domain=(0.0, jnp.pi)) # first derivative on [0, pi]

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``diff``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    I, diag, FunctionalBlock, ChebMatrix
    """
    if order < 0:
        raise ValueError(
            f"Differentiation order must be a non-negative integer, got {order}."
        )

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        if disc.num_intervals == 1:
            return diffmat(disc.n, order, domain=disc.domain)
        return _blkdiag([diffmat(nk, order, domain=(a, b))
                         for nk, (a, b) in zip(disc.sizes, disc.intervals)])

    def _coeffs() -> list:
        c = [_const_chebfun(1.0, domain)]
        for _ in range(order):
            c = _coeff_diff(c)
        return c

    return OperatorBlock(_op_fn, order=order, domain=domain,
                         apply_fn=lambda u: u.diff(order),
                         isnotdiffint=(order == 0), coeff_fn=_coeffs)


def I(domain: _DomainT = _DEFAULT_DOMAIN) -> OperatorBlock:  # noqa: E743
    """Identity operator.

    Returns an ``OperatorBlock`` whose matrix is ``eye(n)`` at any
    discretization size.

    Parameters
    ----------
    domain : (float, float), default (-1, 1)
        Physical interval ``[a, b]``.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> Id = I()
    >>> Id.matrix(ChebColloc2Disc(6))
    Array([[1., 0., ...]])

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``eye``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    D, diag
    """

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        return jnp.eye(disc.n, dtype=jnp.float64)

    return OperatorBlock(_op_fn, order=0, domain=domain,
                         apply_fn=lambda u: u, isnotdiffint=True,
                         coeff_fn=lambda: [_const_chebfun(1.0, domain)])


def diag(f, domain: _DomainT | None = None) -> OperatorBlock:
    """Multiplication-by-f operator.

    Returns the ``OperatorBlock`` that maps ``u(x)`` to ``f(x)*u(x)``.
    In the collocation discretization this is ``diag(f(x_0), ..., f(x_{n-1}))``.

    Parameters
    ----------
    f : Chebfun or callable
        The multiplier function.  If a Chebfun is passed, its domain is used
        (unless ``domain`` is explicitly provided).  If a plain callable is
        passed, ``domain`` must be provided.
    domain : (float, float) or None
        Physical domain.  Inferred from ``f.domain`` when ``f`` is a Chebfun.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> import chebfunjax as cj
    >>> x_fun = cj.chebfun(lambda x: x)
    >>> M = diag(x_fun)
    >>> disc = ChebColloc2Disc(6)
    >>> # M.matrix(disc) is diag(x_0, ..., x_5) where x_i are Cheb-2 pts

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``mult``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    D, I, FunctionalBlock
    """
    # Infer domain from f if it has a .domain attribute (Chebfun / _Piece)
    if domain is None:
        domain = _domain_of(f)

    dom = tuple(float(v) for v in domain)

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        if disc.num_intervals == 1:
            # Evaluate f at the Chebyshev-2 points of the discretization.
            pts = chebpts(disc.n, kind=2)
            a, b = disc.domain[0], disc.domain[-1]
            x_phys = 0.5 * (b - a) * pts + 0.5 * (a + b)
            fvals = jnp.ravel(jnp.asarray(f(x_phys), dtype=jnp.float64))
        else:
            fvals = _to_values(f, disc)
        return jnp.diag(fvals)

    # Function-space action u |-> f*u, available when f is itself a Chebfun.
    from chebfunjax.chebfun1d.chebfun import Chebfun
    apply_fn = (lambda u: f * u) if isinstance(f, Chebfun) else None
    coeff_fn = (lambda: [f]) if isinstance(f, Chebfun) else None
    return OperatorBlock(_op_fn, order=0, domain=dom, apply_fn=apply_fn,
                         isnotdiffint=True, coeff_fn=coeff_fn)


def eval_at(x: float, domain: _DomainT = _DEFAULT_DOMAIN,
            direction: Union[int, str] = 0) -> FunctionalBlock:
    """Point-evaluation functional: ``F[u] = u(x)``.

    Returns a ``FunctionalBlock`` whose row vector ``r`` satisfies
    ``r @ u_vals ≈ u(x)`` where ``u_vals`` are the values at the
    ``n`` Chebyshev-2 points.  Uses barycentric interpolation.

    Parameters
    ----------
    x : float
        Evaluation point, must be in ``domain``.
    domain : tuple of float, default (-1, 1)
        Physical domain (may carry interior breakpoints).
    direction : {0, -1, +1, 'left', 'right', '-', '+'}, default 0
        Side from which ``x`` is approached.  Only matters when ``x`` is an
        interior breakpoint of a piecewise discretization or of the
        function being evaluated.

    Returns
    -------
    FunctionalBlock

    Raises
    ------
    ValueError
        If ``x`` is outside ``domain``.

    Examples
    --------
    >>> E = eval_at(0.5)
    >>> r = E.matrix(ChebColloc2Disc(8))
    >>> r.shape
    (8,)
    >>> # r @ sin_vals ≈ sin(0.5)

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``feval``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    sum_functional, FunctionalBlock
    """
    dom = tuple(float(v) for v in domain)
    a, b = dom[0], dom[-1]
    if x < a or x > b:
        raise ValueError(
            f"eval_at: evaluation point x={x} is outside domain [{a}, {b}]."
        )
    dirn = _parse_direction(direction)

    def _fn(disc: ChebColloc2Disc) -> Array:
        if disc.num_intervals == 1:
            a_, b_ = disc.domain[0], disc.domain[-1]
            return _bary_row(disc.n, (2.0 * x - (a_ + b_)) / (b_ - a_))
        k = disc.which_interval(x, dirn)
        a_, b_ = disc.intervals[k]
        nk = disc.sizes[k]
        sub = _bary_row(nk, (2.0 * x - (a_ + b_)) / (b_ - a_))
        off = disc.offsets()[k]
        row = jnp.zeros(disc.n, dtype=jnp.float64)
        return row.at[off:off + nk].set(sub)

    fb = FunctionalBlock(_fn, domain=dom, isnotdiffint=True)
    # Location metadata: lets piecewise discretizations (Linop.expm) place
    # this row in the sub-interval that owns the evaluation point.
    fb.loc = float(x)
    fb.direction = dirn
    if dirn == 0:
        fb._apply_fn = lambda u: _feval_chebfun(u, x, None)
    else:
        side = "left" if dirn < 0 else "right"
        fb._apply_fn = lambda u: _feval_chebfun(u, x, side)
    return fb


def sum_functional(domain: _DomainT = _DEFAULT_DOMAIN) -> FunctionalBlock:
    """Definite-integral functional: ``F[u] = integral_a^b u(x) dx``.

    Returns a ``FunctionalBlock`` whose row vector ``r`` satisfies
    ``r @ u_vals = integral_a^b u(x) dx`` where ``u_vals`` are the
    values at the ``n`` Clenshaw-Curtis (Chebyshev-2) points.

    Parameters
    ----------
    domain : (float, float), default (-1, 1)
        Physical domain ``[a, b]``.

    Returns
    -------
    FunctionalBlock

    Examples
    --------
    >>> S = sum_functional()
    >>> r = S.matrix(ChebColloc2Disc(8))
    >>> r.shape
    (8,)
    >>> # r @ ones_vals ≈ 2  (integral of 1 over [-1, 1])

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``sum``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    eval_at, FunctionalBlock
    """
    dom = tuple(float(v) for v in domain)

    def _fn(disc: ChebColloc2Disc) -> Array:
        # Clenshaw-Curtis weights scaled to each subinterval.
        parts = [chebweights(nk) * 0.5 * (b - a)
                 for nk, (a, b) in zip(disc.sizes, disc.intervals)]
        return jnp.concatenate(parts)

    return FunctionalBlock(_fn, domain=dom, order=-1,
                           apply_fn=lambda u: u.sum())


# ===========================================================================
# Further MATLAB operatorBlock / functionalBlock static methods
# ===========================================================================


def zeros_op(domain: _DomainT = _DEFAULT_DOMAIN) -> OperatorBlock:
    """Zero operator: maps every function to the zero function.

    Parameters
    ----------
    domain : tuple of float, default (-1, 1)
        Physical domain (may carry interior breakpoints).

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> Z = zeros_op((0.0, 1.0))
    >>> Z.iszero
    True

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``zeros``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    I, D, zero_functional
    """
    dom = tuple(float(v) for v in domain)

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        return jnp.zeros((disc.n, disc.n), dtype=jnp.float64)

    def _apply(u):
        return u * 0.0

    return OperatorBlock(_op_fn, order=0, domain=dom, apply_fn=_apply,
                         iszero=True, isnotdiffint=True,
                         coeff_fn=lambda: [_const_chebfun(0.0, dom)])


def cumsum_op(domain: _DomainT = _DEFAULT_DOMAIN,
              m: int = 1) -> OperatorBlock:
    """Indefinite-integration (antiderivative) operator.

    ``C*u`` is the antiderivative of ``u`` that vanishes at the left endpoint
    of ``domain``.  ``m`` gives the number of repeated antiderivatives.

    Parameters
    ----------
    domain : tuple of float, default (-1, 1)
        Physical domain (may carry interior breakpoints).
    m : int, default 1
        Number of repeated antiderivatives.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> C = cumsum_op((0.0, 1.0))
    >>> C.order
    -1

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``cumsum``),
        @chebcolloc/cumsum.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    D, sum_functional
    """
    if m < 0:
        raise ValueError(f"cumsum_op: m must be non-negative, got {m}.")
    dom = tuple(float(v) for v in domain)

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        if m == 0:
            return jnp.eye(disc.n, dtype=jnp.float64)
        C = _blkdiag([cumsummat(nk, domain=(a, b))
                      for nk, (a, b) in zip(disc.sizes, disc.intervals)])
        # Each subinterval contributes its full integral to everything to
        # its right, giving the block-triangular structure.
        offs = disc.offsets()
        total = disc.n
        for k in range(disc.num_intervals):
            row = offs[k + 1] - 1
            lo, hi = offs[k], offs[k + 1]
            last = C[row, lo:hi]
            nrem = total - offs[k + 1]
            if nrem > 0:
                C = C.at[offs[k + 1]:, lo:hi].set(
                    jnp.tile(last[None, :], (nrem, 1)))
        return jnp.linalg.matrix_power(C, m)

    return OperatorBlock(_op_fn, order=-m, domain=dom,
                         apply_fn=lambda u: u.cumsum(m) if m != 1 else u.cumsum())


def mult(f, domain: _DomainT | None = None) -> OperatorBlock:
    """Multiplication operator ``u -> f*u`` (MATLAB ``operatorBlock.mult``).

    A thin alias of :func:`diag` matching the MATLAB spelling.

    Parameters
    ----------
    f : Chebfun or callable
        Multiplier.
    domain : tuple of float or None
        Physical domain; inferred from ``f`` when omitted.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> import chebfunjax as cj
    >>> M = mult(cj.chebfun(lambda x: x))

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``mult``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diag
    """
    return diag(f, domain)


def fred_op(kernel: Callable, domain: _DomainT = _DEFAULT_DOMAIN,
            ) -> OperatorBlock:
    """Fredholm integral operator ``(F u)(x) = int_a^b K(x,y) u(y) dy``.

    Parameters
    ----------
    kernel : callable
        ``K(x, y)`` accepting broadcastable arrays.
    domain : tuple of float, default (-1, 1)
        Physical domain ``[a, b]``.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> F = fred_op(lambda x, y: jnp.exp(x - y), (0.0, 1.0))

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``fred``),
        @chebcolloc/fred.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    volt_op
    """
    dom = tuple(float(v) for v in domain)

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        pts = disc.points()
        w = jnp.concatenate([chebweights(nk) * 0.5 * (b - a)
                             for nk, (a, b) in zip(disc.sizes,
                                                   disc.intervals)])
        K = jnp.asarray(kernel(pts[:, None], pts[None, :]),
                        dtype=jnp.float64)
        return K * w[None, :]

    def _apply(u):
        from chebfunjax.operators.integral import fred
        return fred(kernel, u)

    return OperatorBlock(_op_fn, order=-100, domain=dom, apply_fn=_apply)


def volt_op(kernel: Callable, domain: _DomainT = _DEFAULT_DOMAIN,
            ) -> OperatorBlock:
    """Volterra integral operator ``(V u)(x) = int_a^x K(x,y) u(y) dy``.

    Parameters
    ----------
    kernel : callable
        ``K(x, y)`` accepting broadcastable arrays.
    domain : tuple of float, default (-1, 1)
        Physical domain ``[a, b]``.

    Returns
    -------
    OperatorBlock

    Examples
    --------
    >>> V = volt_op(lambda x, y: x * y, (0.0, 1.0))

    Provenance
    ----------
    MATLAB source : @operatorBlock/operatorBlock.m  (static method ``volt``),
        @chebcolloc/volt.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    fred_op
    """
    dom = tuple(float(v) for v in domain)

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        pts = disc.points()
        C = cumsum_op(disc.domain)._op_fn(disc)
        K = jnp.asarray(kernel(pts[:, None], pts[None, :]),
                        dtype=jnp.float64)
        return K * C

    def _apply(u):
        from chebfunjax.operators.integral import volt
        return volt(kernel, u)

    return OperatorBlock(_op_fn, order=-1, domain=dom, apply_fn=_apply)


def zero_functional(domain: _DomainT = _DEFAULT_DOMAIN) -> FunctionalBlock:
    """Zero functional: maps every function to 0.

    Parameters
    ----------
    domain : tuple of float, default (-1, 1)
        Physical domain.

    Returns
    -------
    FunctionalBlock

    Examples
    --------
    >>> z = zero_functional()
    >>> z.iszero
    True

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``zero``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    sum_functional, eval_at
    """
    dom = tuple(float(v) for v in domain)
    return FunctionalBlock(
        lambda disc: jnp.zeros(disc.n, dtype=jnp.float64),
        domain=dom, apply_fn=lambda u: 0.0, iszero=True, isnotdiffint=True)


def inner_functional(f, domain: _DomainT | None = None) -> FunctionalBlock:
    """Inner-product functional ``F[u] = int f(x) u(x) dx``.

    Parameters
    ----------
    f : Chebfun
        The function to take the inner product against.
    domain : tuple of float or None
        Physical domain; inferred from ``f`` when omitted.

    Returns
    -------
    FunctionalBlock

    Examples
    --------
    >>> import chebfunjax as cj
    >>> dt = inner_functional(cj.chebfun(lambda x: x ** 2))

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``inner``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    sum_functional
    """
    dom = tuple(float(v) for v in (domain if domain is not None
                                   else _domain_of(f)))

    def _fn(disc: ChebColloc2Disc) -> Array:
        w = jnp.concatenate([chebweights(nk) * 0.5 * (b - a)
                             for nk, (a, b) in zip(disc.sizes,
                                                   disc.intervals)])
        return w * _to_values(f, disc)

    return FunctionalBlock(_fn, domain=dom,
                           apply_fn=lambda u: float(f.inner(u)),
                           isnotdiffint=True)


def jump_functional(location: float, domain: _DomainT,
                    order: int = 0) -> FunctionalBlock:
    """Jump functional: the right limit minus the left limit of the
    ``order``-th derivative at ``location``.

    Parameters
    ----------
    location : float
        Interior point at which the jump is measured.
    domain : tuple of float
        Physical domain; ``location`` is merged in as a breakpoint.
    order : int, default 0
        Derivative order.

    Returns
    -------
    FunctionalBlock

    Examples
    --------
    >>> J = jump_functional(0.3, (0.0, 0.3, 1.0), 1)

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``jump``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    jump_at, eval_at
    """
    dom = tuple(sorted(set(tuple(float(v) for v in domain)
                           + (float(location),))))
    Er = eval_at(location, dom, direction=1)
    El = eval_at(location, dom, direction=-1)
    J = (Er - El) * D(dom, order)
    J.isnotdiffint = (order == 0)
    return J


def jump_at(domain: _DomainT) -> Callable[..., FunctionalBlock]:
    """Return a jump-functional generator ``j(loc, order)`` on ``domain``.

    Parameters
    ----------
    domain : tuple of float
        Physical domain.

    Returns
    -------
    callable
        ``j(location, order)`` -> FunctionalBlock.

    Examples
    --------
    >>> j = jump_at((0.0, 0.3, 1.0))
    >>> jump_row = j(0.3, 1)

    Provenance
    ----------
    MATLAB source : @functionalBlock/functionalBlock.m  (static method ``jumpAt``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    jump_functional
    """
    return lambda loc, order=0: jump_functional(loc, domain, order)


def primitive_operators(domain: _DomainT = _DEFAULT_DOMAIN):
    """Frequently used operator blocks ``(Z, I, D, C, M)``.

    Parameters
    ----------
    domain : tuple of float, default (-1, 1)
        Physical domain (may carry interior breakpoints).

    Returns
    -------
    Z : OperatorBlock
        Zero operator.
    Id : OperatorBlock
        Identity operator.
    Dop : OperatorBlock
        First-derivative operator.
    C : OperatorBlock
        Indefinite-integration operator.
    M : callable
        ``M(f)`` is the multiplication-by-``f`` operator.

    Examples
    --------
    >>> Z, Id, Dop, C, M = primitive_operators((-1.0, 4.0))

    Provenance
    ----------
    MATLAB source : @linop/linop.m  (static method ``primitiveOperators``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    primitive_functionals
    """
    dom = tuple(float(v) for v in domain)
    return (zeros_op(dom), I(dom), D(dom, 1), cumsum_op(dom, 1),
            lambda f, d=None: mult(f, d))


def primitive_functionals(domain: _DomainT = _DEFAULT_DOMAIN):
    """Frequently used functional blocks ``(z, E, S, dt)``.

    Parameters
    ----------
    domain : tuple of float, default (-1, 1)
        Physical domain (may carry interior breakpoints).

    Returns
    -------
    z : FunctionalBlock
        Zero functional.
    E : callable
        ``E(x)`` (or ``E(x, direction)``) is point evaluation at ``x``.
    S : FunctionalBlock
        Definite integration.
    dt : callable
        ``dt(f)`` is the inner-product-with-``f`` functional.

    Examples
    --------
    >>> z, E, S, dt = primitive_functionals((-1.0, 2.0))

    Provenance
    ----------
    MATLAB source : @linop/linop.m  (static method ``primitiveFunctionals``)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    primitive_operators
    """
    dom = tuple(float(v) for v in domain)
    return (zero_functional(dom),
            lambda x, direction=0: eval_at(x, dom, direction),
            sum_functional(dom),
            lambda f, d=None: inner_functional(f, d))


def to_function(block):
    """Return the function-space action of a block (MATLAB ``toFunction``).

    Parameters
    ----------
    block : OperatorBlock or FunctionalBlock

    Returns
    -------
    callable

    Examples
    --------
    >>> eyeop = to_function(I())

    Provenance
    ----------
    MATLAB source : @linBlock/toFunction.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    to_coeff
    """
    return block.to_function()


def to_coeff(block: OperatorBlock) -> list:
    """Variable coefficients of a differential operator block.

    Returns ``[a_m, a_{m-1}, ..., a_0]`` (highest derivative first) such that
    ``block[u] = a_m u^{(m)} + ... + a_1 u' + a_0 u``.

    The coefficients are carried symbolically through the block algebra:
    each primitive block knows its own coefficient list and ``+``, ``*`` and
    scaling combine them by the Leibniz rule, so the result is as exact as
    the underlying Chebfun arithmetic.

    Parameters
    ----------
    block : OperatorBlock
        A block of non-negative differential order built from ``I``, ``D``,
        ``mult``, and ``zeros_op``.  Integration and evaluation have no
        coefficient realization.

    Returns
    -------
    list of Chebfun
        The coefficients, highest derivative first.

    Examples
    --------
    >>> Dc = to_coeff(D())
    >>> len(Dc)
    2

    Provenance
    ----------
    MATLAB source : @linBlock/toCoeff.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    to_function
    """
    return block.coeff_list()


# ===========================================================================
# KronOp — rank-1 (or low-rank) integral operator kron(f, g', 'op')
# ===========================================================================


class KronOp:
    """Rank-k integral operator ``A = sum_j f_j (g_j' .)`` (MATLAB
    ``kron(f, g', 'op')``).

    Applied to a function ``h`` it returns
    ``A h = sum_j f_j * <g_j, h>`` where ``<g_j, h> = int g_j(x) h(x) dx``.
    In a collocation discretization at ``n`` points ``x_i`` with quadrature
    weights ``w_i`` it is the rank-k matrix
    ``M = sum_j outer(f_j(x_i), w * g_j(x))``.

    Provenance
    ----------
    MATLAB source : @chebfun/kron.m (the 'op' branch)
    Chebfun commit: 7574c77
    """

    def __init__(self, fs, gs, domain) -> None:
        self.fs = list(fs)
        self.gs = list(gs)
        if len(self.fs) != len(self.gs):
            raise ValueError(
                "CHEBFUN:CHEBFUN:kron:sizes -- f and g must have the same "
                "number of columns.")
        self.domain = domain

    def apply(self, h):
        """A h = sum_j f_j * <g_j, h>."""
        result = None
        for fj, gj in zip(self.fs, self.gs):
            ip = float((gj * h).sum())
            term = fj * ip
            result = term if result is None else result + term
        return result

    def __mul__(self, other):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(other, Chebfun):
            return self.apply(other)
        return NotImplemented

    def matrix(self, n: int, kind: str = "chebcolloc2") -> Array:
        """Rank-k collocation matrix at ``n`` points (``kind`` is one of
        ``'chebcolloc1'``, ``'chebcolloc2'``, ``'trigcolloc'``)."""
        import numpy as _np
        import numpy as _np2

        from chebfunjax.utils.quadrature import (
            chebpts_ab,
            chebweights,
            trigpts,
        )
        a, b = float(self.domain[0]), float(self.domain[1])
        if kind == "chebcolloc1":
            pts = chebpts_ab(n, a, b, kind=1)
            # Fejer's first-rule weights for the kind-1 points (integrate
            # f dx on [-1,1], then scaled to [a,b]).
            kk = _np2.arange(n)
            theta = (2 * kk + 1) * _np2.pi / (2 * n)
            xk = _np2.cos(theta)
            mm = _np2.arange(1, n // 2 + 1)
            wref = (2.0 / n) * (1.0 - 2.0 * (
                _np2.cos(2.0 * _np2.outer(theta, mm))
                / (4.0 * mm ** 2 - 1.0)).sum(axis=1))
            wref = wref[_np2.argsort(xk)]  # ascending, matches chebpts_ab
            w = jnp.asarray(wref) * (0.5 * (b - a))
        elif kind == "chebcolloc2":
            pts = chebpts_ab(n, a, b, kind=2)
            w = chebweights(n, kind=2) * (0.5 * (b - a))
        elif kind == "trigcolloc":
            pts, w = trigpts(n, (a, b))
        else:
            raise ValueError(f"Unknown discretization kind {kind!r}.")
        pts = jnp.asarray(pts)
        w = _np.asarray(w, dtype=_np.float64)
        M = _np.zeros((n, n), dtype=_np.float64)
        for fj, gj in zip(self.fs, self.gs):
            fv = _np.asarray(fj(pts), dtype=_np.float64)
            gv = _np.asarray(gj(pts), dtype=_np.float64)
            M = M + _np.outer(fv, w * gv)
        return M
