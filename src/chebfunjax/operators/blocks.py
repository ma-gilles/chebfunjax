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

from typing import Callable, Union

import jax.numpy as jnp

from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts, chebweights

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = jnp.ndarray
_DomainT = tuple[float, float]

_DEFAULT_DOMAIN: _DomainT = (-1.0, 1.0)


# ===========================================================================
# Discretization descriptor
# ===========================================================================


class ChebColloc2Disc:
    """Minimal Chebyshev-collocation-2 discretization descriptor.

    Carries only the information that ``OperatorBlock.matrix`` and
    ``FunctionalBlock.matrix`` need: the grid size ``n`` and the physical
    domain ``(a, b)``.

    Parameters
    ----------
    n : int
        Number of collocation points.
    domain : (float, float)
        Physical interval ``[a, b]``.
    """

    def __init__(self, n: int, domain: _DomainT = _DEFAULT_DOMAIN) -> None:
        self.n = n
        self.domain = domain


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
    ) -> None:
        self._op_fn = op_fn
        self.order = order
        self.domain = domain

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
        _check_domains(self, other)
        new_order = max(self.order, other.order)
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) + other.matrix(disc)

        return OperatorBlock(_fn, order=new_order, domain=domain)

    def __radd__(self, other: "OperatorBlock") -> "OperatorBlock":
        return self.__add__(other)

    def __sub__(self, other: "OperatorBlock") -> "OperatorBlock":
        """Operator subtraction: ``(A - B)*u = A*u - B*u``."""
        if isinstance(other, (int, float)):
            other = _scalar_op(other, self.domain)
        _check_domains(self, other)
        new_order = max(self.order, other.order)
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) - other.matrix(disc)

        return OperatorBlock(_fn, order=new_order, domain=domain)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = _scalar_op(other, self.domain)
        return other.__sub__(self)

    def __mul__(self, other) -> "OperatorBlock":
        """Operator composition or scalar multiplication.

        - ``A * B`` composes operators (matrix product in discretization).
        - ``A * c`` scales the operator by scalar ``c``.
        """
        if isinstance(other, (int, float)):
            c = float(other)
            domain = self.domain

            def _fn(disc: ChebColloc2Disc) -> Array:
                return c * self.matrix(disc)

            return OperatorBlock(_fn, order=self.order, domain=domain)

        if isinstance(other, OperatorBlock):
            _check_domains(self, other)
            new_order = self.order + other.order
            domain = self.domain

            def _fn(disc: ChebColloc2Disc) -> Array:
                return self.matrix(disc) @ other.matrix(disc)

            return OperatorBlock(_fn, order=new_order, domain=domain)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> "OperatorBlock":
        """Unary minus."""
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return -self.matrix(disc)

        return OperatorBlock(_fn, order=self.order, domain=domain)

    def __pow__(self, k: int) -> "OperatorBlock":
        """Repeated composition: ``A^k = A * A * ... * A`` (k times)."""
        if not (isinstance(k, int) and k >= 0):
            raise ValueError(
                f"OperatorBlock power must be a non-negative integer, got {k!r}."
            )
        result = I(self.domain)
        for _ in range(k):
            result = self * result
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
    ) -> None:
        self._func_fn = func_fn
        self.domain = domain

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

    def __add__(self, other: "FunctionalBlock") -> "FunctionalBlock":
        if isinstance(other, (int, float)):
            raise TypeError(
                "Cannot add a scalar to a FunctionalBlock directly. "
                "Use eval_at or sum_functional to build scalar functionals."
            )
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) + other.matrix(disc)

        return FunctionalBlock(_fn, domain=domain)

    def __sub__(self, other: "FunctionalBlock") -> "FunctionalBlock":
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return self.matrix(disc) - other.matrix(disc)

        return FunctionalBlock(_fn, domain=domain)

    def __mul__(self, other) -> "FunctionalBlock":
        """Scalar multiplication or composition with an OperatorBlock.

        - ``F * c``  (scalar) scales the row.
        - ``F * A``  (OperatorBlock) composes: ``(F*A)[u] = F[A[u]]``,
          i.e. ``row_F @ matrix_A``.
        """
        if isinstance(other, (int, float)):
            c = float(other)
            domain = self.domain

            def _fn(disc: ChebColloc2Disc) -> Array:
                return c * self.matrix(disc)

            return FunctionalBlock(_fn, domain=domain)

        if isinstance(other, OperatorBlock):
            domain = self.domain

            def _fn(disc: ChebColloc2Disc) -> Array:
                return self.matrix(disc) @ other.matrix(disc)

            return FunctionalBlock(_fn, domain=domain)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> "FunctionalBlock":
        domain = self.domain

        def _fn(disc: ChebColloc2Disc) -> Array:
            return -self.matrix(disc)

        return FunctionalBlock(_fn, domain=domain)

    def __repr__(self) -> str:
        return f"FunctionalBlock(domain={self.domain})"


# ===========================================================================
# Private helpers
# ===========================================================================


def _check_domains(a, b) -> None:
    """Raise if two blocks have incompatible domains."""
    if a.domain != b.domain:
        raise ValueError(
            f"Cannot combine OperatorBlocks with different domains: "
            f"{a.domain} vs {b.domain}. Restrict to a common domain first."
        )


def _scalar_op(c: float, domain: _DomainT) -> OperatorBlock:
    """Return ``c * I`` as an OperatorBlock."""
    return OperatorBlock(
        lambda disc: c * jnp.eye(disc.n, dtype=jnp.float64),
        order=0,
        domain=domain,
    )


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
        return diffmat(disc.n, order, domain=disc.domain)

    return OperatorBlock(_op_fn, order=order, domain=domain)


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

    return OperatorBlock(_op_fn, order=0, domain=domain)


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
        try:
            dom_obj = f.domain
            if hasattr(dom_obj, "breakpoints"):
                # Domain object: take first and last breakpoints
                bpts = dom_obj.breakpoints
                domain = (float(bpts[0]), float(bpts[-1]))
            else:
                domain = tuple(float(v) for v in dom_obj[:2])
        except AttributeError:
            domain = _DEFAULT_DOMAIN

    dom = domain  # capture in closure

    def _op_fn(disc: ChebColloc2Disc) -> Array:
        # Evaluate f at the Chebyshev-2 points of the discretization.
        pts = chebpts(disc.n, kind=2)
        # Scale pts from [-1,1] to [a,b]
        a, b = disc.domain
        x_phys = 0.5 * (b - a) * pts + 0.5 * (a + b)
        fvals = jnp.asarray(f(x_phys), dtype=jnp.float64)
        return jnp.diag(fvals)

    return OperatorBlock(_op_fn, order=0, domain=dom)


def eval_at(x: float, domain: _DomainT = _DEFAULT_DOMAIN) -> FunctionalBlock:
    """Point-evaluation functional: ``F[u] = u(x)``.

    Returns a ``FunctionalBlock`` whose row vector ``r`` satisfies
    ``r @ u_vals ≈ u(x)`` where ``u_vals`` are the values at the
    ``n`` Chebyshev-2 points.  Uses barycentric interpolation.

    Parameters
    ----------
    x : float
        Evaluation point, must be in ``domain``.
    domain : (float, float), default (-1, 1)
        Physical domain.

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
    a, b = domain
    if x < a or x > b:
        raise ValueError(
            f"eval_at: evaluation point x={x} is outside domain [{a}, {b}]."
        )
    dom = domain

    def _fn(disc: ChebColloc2Disc) -> Array:
        # Chebyshev-2 points on [-1, 1]
        pts_ref = chebpts(disc.n, kind=2)
        # Corresponding physical points
        a_, b_ = disc.domain
        # Map physical x to reference
        t = (2.0 * x - (a_ + b_)) / (b_ - a_)

        # Barycentric weights for Chebyshev-2 points
        n = disc.n
        # w[k] = (-1)^k, half-weight at endpoints
        k = jnp.arange(n, dtype=jnp.float64)
        w = jnp.where(
            (k == 0) | (k == n - 1),
            0.5 * jnp.ones(n, dtype=jnp.float64),
            jnp.ones(n, dtype=jnp.float64),
        )
        # Alternate signs: w[k] *= (-1)^k
        signs = jnp.where(k % 2 == 0, 1.0, -1.0)
        w = w * signs

        # Differences t - t_k
        diff = t - pts_ref  # shape (n,)

        # Check if t coincides with a node
        # (use a non-JIT-safe path for exact node hits)
        close = jnp.abs(diff) < 1e-14

        # Barycentric interpolation row
        # r[j] = (w[j] / (t - t_j)) / sum_k( w[k] / (t - t_k) )
        # If t == t_j, the j-th basis function is 1, all others 0.
        any_close = bool(jnp.any(close))
        if any_close:
            idx = int(jnp.argmax(close))
            row = jnp.zeros(n, dtype=jnp.float64).at[idx].set(1.0)
        else:
            num = w / diff
            row = num / jnp.sum(num)

        return row

    return FunctionalBlock(_fn, domain=dom)


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
    dom = domain

    def _fn(disc: ChebColloc2Disc) -> Array:
        # Clenshaw-Curtis weights scaled to [a, b]
        return chebweights(disc.n) * 0.5 * (disc.domain[1] - disc.domain[0])

    return FunctionalBlock(_fn, domain=dom)
