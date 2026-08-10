"""Abstract base class for functions on arbitrary intervals [a, b].

Translated from MATLAB Chebfun class @classicfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


def _gammaln(x: jax.Array) -> jax.Array:
    """Log-gamma, used only for stable binomial coefficients in ``poly``."""
    return jax.lax.lgamma(jnp.asarray(x, dtype=jnp.float64))


def _is_matrix_operand(other) -> bool:
    """True if ``other`` is a numeric array acting as a matrix (ndim >= 2).

    MATLAB matrices map to 2-D numpy arrays (``np.eye(2)``,
    ``np.array([[1, 1]])``), which trigger mrdivide/mtimes semantics.
    1-D arrays are this codebase's array-valued POINTWISE row convention
    (``f / [a, b]`` divides column k by ``[a, b][k]``, per the rdivide
    ports) and must stay pointwise; scalars and funs likewise.
    """
    if isinstance(other, Classicfun):
        return False
    return hasattr(other, "ndim") and not callable(other) and other.ndim >= 2


def _is_scalar_zero(other) -> bool:
    """True if ``other`` is a numeric scalar equal to zero (MATLAB F/0)."""
    if isinstance(other, Classicfun) or isinstance(other, bool):
        return False
    try:
        arr = jnp.asarray(other)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 0 and jnp.issubdtype(arr.dtype, jnp.number) and bool(arr == 0)


def _fun_isequal(f, g) -> bool:
    """Shared MATLAB ``@classicfun/isequal``: same domain and equal onefuns.

    Used by both :class:`Classicfun` and :class:`~chebfunjax.fun.unbndfun.Unbndfun`,
    which are separate class hierarchies but share the ``onefun``/``domain``
    layout.  Funs of different classes are never equal.
    """
    if type(f) is not type(g):
        return False
    f_empty = f.isempty() if hasattr(f, "isempty") else False
    g_empty = g.isempty() if hasattr(g, "isempty") else False
    if f_empty or g_empty:
        return f_empty and g_empty
    if f.domain != g.domain:
        return False
    onefun_isequal = getattr(f.onefun, "isequal", None)
    if onefun_isequal is None:
        return f.onefun is g.onefun
    return bool(onefun_isequal(g.onefun))


def _cheb_to_poly(coeffs: jax.Array) -> jax.Array:
    """Chebyshev-T coefficients to monomial coefficients on [-1, 1].

    Parameters
    ----------
    coeffs : jax.Array
        Ascending Chebyshev coefficients, shape ``(n,)`` (scalar-valued) or
        ``(n, m)`` (array-valued).

    Returns
    -------
    jax.Array, shape ``(m, n)``
        Monomial coefficients in DESCENDING power order (column 0 is the
        coefficient of ``x^(n-1)``).  Always 2-D; the caller squeezes.

    Provenance
    ----------
    MATLAB source : @chebtech/poly.m
    Chebfun commit: 7574c77
    """
    c = coeffs.reshape(coeffs.shape[0], -1)  # (n, m)
    # MATLAB flips to descending Chebyshev order before the recurrence.
    c = jnp.flipud(c)
    n, m = c.shape
    cT = c.T  # (m, n)

    if n == 1:
        return cT  # (m, 1) constant
    if n == 2:
        return cT  # (m, 2): [T1 coeff, T0 coeff] == [x coeff, const]

    out_dtype = cT.dtype
    tnold1 = jnp.zeros((m, n)).at[:, 1].set(1.0)  # T_1 = x
    tnold2 = jnp.zeros((m, n)).at[:, 0].set(1.0)  # T_0 = 1
    out = jnp.zeros((m, n), dtype=out_dtype)
    # Initial step (k = 2): out[:, 0] = T1 coeff, out[:, 1] = T0 coeff.
    out = out.at[:, 0].set(cT[:, n - 2])
    out = out.at[:, 1].set(cT[:, n - 1])

    for k in range(3, n + 1):  # 1-based k, up to n
        new_tn = jnp.zeros((m, n))
        # tn[:, 0:k] = [0, 2*tnold1[:, 0:k-1]] - [tnold2[:, 0:k-2], 0, 0]
        A = jnp.zeros((m, k)).at[:, 1:k].set(2.0 * tnold1[:, 0 : k - 1])
        B = jnp.zeros((m, k))
        if k - 2 >= 1:
            B = B.at[:, 0 : k - 2].set(tnold2[:, 0 : k - 2])
        new_tn = new_tn.at[:, 0:k].set(A - B)
        # out[:, 0:k] = coeffs[:, n-k] * reverse(tn[:, 0:k]) + [0, out[:, 0:k-1]]
        coef = cT[:, n - k]
        tn_rev = new_tn[:, 0:k][:, ::-1]
        shifted = jnp.zeros((m, k), dtype=out_dtype).at[:, 1:k].set(out[:, 0 : k - 1])
        out = out.at[:, 0:k].set(coef[:, None] * tn_rev + shifted)
        tnold2 = tnold1
        tnold1 = new_tn

    return out


class Classicfun(eqx.Module):
    """Abstract base class for smooth functions on a bounded interval [a, b].

    A Classicfun represents a smooth function on an interval [a, b] by
    wrapping a Chebtech2 (which lives on the standard interval [-1, 1]) with
    an affine domain mapping.  Concrete subclasses (``Bndfun``) handle
    different types of domain maps.

    The separation is: the ``onefun`` holds all function approximation logic
    (coefficient representation, evaluation at [-1,1], arithmetic, calculus),
    while the Classicfun layer applies the affine map so everything is
    expressed in physical coordinates [a, b].

    Attributes
    ----------
    onefun : Chebtech2
        The underlying Chebyshev representation on [-1, 1].
    domain : Domain
        The bounded interval [a, b] (exactly one sub-interval).

    Notes
    -----
    All binary operations assume the two Classicfun objects share the same
    domain.  This is checked at the Python level and raises ``ValueError``
    if the domains differ.

    Provenance
    ----------
    MATLAB source : @classicfun/classicfun.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Bndfun, Chebtech2, Domain
    """

    onefun: Chebtech2
    domain: Domain = eqx.field(static=True)

    # Let numpy defer ``ndarray <op> Classicfun`` (e.g. ``A / f`` mrdivide)
    # to our reflected operators instead of broadcasting elementwise.
    __array_ufunc__ = None

    # ------------------------------------------------------------------
    # Empty representation (MATLAB bndfun() with no arguments)
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Classicfun":
        """The empty fun (MATLAB ``bndfun()``).

        ``isempty()`` is True; arithmetic and restriction with it propagate
        empties.  Built without ``__init__`` (no onefun/domain), so guard field
        access with ``isempty()`` first.

        Provenance
        ----------
        MATLAB source : @classicfun/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty fun (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @classicfun/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    # ------------------------------------------------------------------
    # Construction (class methods — NOT __init__)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        domain: Domain,
        *,
        n: int | None = None,
    ) -> "Classicfun":
        """Construct from a callable on the given domain.

        Parameters
        ----------
        f : callable
            Function accepting and returning ``jax.Array``.  Must be
            vectorised (accept an array of points and return an array of
            values of the same shape).
        domain : Domain
            A single-interval domain [a, b].
        n : int or None, optional
            Fixed number of Chebyshev points.  If ``None`` (default), use
            the adaptive algorithm.

        Returns
        -------
        Classicfun
            A new instance.
        """
        ...

    @classmethod
    @abstractmethod
    def from_chebtech(cls, tech: Chebtech2, domain: Domain) -> "Classicfun":
        """Wrap an existing Chebtech2 in a domain mapping.

        Parameters
        ----------
        tech : Chebtech2
            An already-constructed Chebtech2 on [-1, 1].
        domain : Domain
            A single-interval domain [a, b].

        Returns
        -------
        Classicfun
            A new instance.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the function at physical point(s) x in [a, b].

        Maps x from [a, b] to [-1, 1] via the inverse affine map and
        delegates to ``self.onefun``.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s) in [a, b].

        Returns
        -------
        y : jax.Array, same shape as x
            Function values.

        Notes
        -----
        JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @bndfun/feval.m
        Chebfun commit: 7574c77
        """
        # Preserve a complex argument (the affine [a, b] -> [-1, 1] map and
        # the underlying Clenshaw recurrence are both valid for complex x);
        # everything else is promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
        # Map from [a, b] to [-1, 1]
        y = self.domain.inverse_map(x)
        return self.onefun(y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients."""
        return self.onefun.n

    @property
    def coeffs(self) -> jax.Array:
        """Chebyshev coefficients of the underlying onefun."""
        return self.onefun.coeffs

    @property
    def values(self) -> jax.Array:
        """Function values at the Chebyshev-2 points on [a, b]."""
        return self.onefun.values

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute function value."""
        return self.onefun.vscale

    @property
    def ishappy(self) -> bool:
        """True if the representation is adaptively resolved."""
        return self.onefun.ishappy

    def __len__(self) -> int:
        """Number of Chebyshev coefficients."""
        return self.n

    # ------------------------------------------------------------------
    # Arithmetic (delegate to onefun, check domains match)
    # ------------------------------------------------------------------

    def _check_domain(self, other: "Classicfun") -> None:
        """Raise ValueError if two Classicfuns have different domains."""
        if self.domain != other.domain:
            raise ValueError(
                f"Cannot perform arithmetic on Classicfun on {self.domain} "
                f"and Classicfun on {other.domain}: domains do not match. "
                f"Use f.restrict({other.domain.a}, {other.domain.b}) to "
                f"restrict the domain first."
            )

    def __add__(self, other) -> "Classicfun":
        """Add two Classicfuns or a Classicfun and a scalar.

        Provenance
        ----------
        MATLAB source : @classicfun/plus.m (delegates to @chebtech/plus.m)
        Chebfun commit: 7574c77
        """
        if self.isempty() or getattr(other, "_is_empty_object", False):
            return type(self).empty()
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun + other.onefun, self.domain)
        else:
            return self.__class__(self.onefun + other, self.domain)

    def __radd__(self, other) -> "Classicfun":
        return self.__add__(other)

    def __sub__(self, other) -> "Classicfun":
        """Subtract two Classicfuns or subtract a scalar.

        Provenance
        ----------
        MATLAB source : @classicfun/minus.m (delegates to @chebtech/minus.m)
        Chebfun commit: 7574c77
        """
        if self.isempty() or getattr(other, "_is_empty_object", False):
            return type(self).empty()
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun - other.onefun, self.domain)
        else:
            return self.__class__(self.onefun - other, self.domain)

    def __rsub__(self, other) -> "Classicfun":
        if self.isempty() or getattr(other, "_is_empty_object", False):
            return type(self).empty()
        return -(self - other)

    def __neg__(self) -> "Classicfun":
        """Unary negation.

        Provenance
        ----------
        MATLAB source : @classicfun/uminus.m
        Chebfun commit: 7574c77
        """
        return self.__class__(-self.onefun, self.domain)

    def __pos__(self) -> "Classicfun":
        """Unary plus (returns self)."""
        return self.__class__(self.onefun, self.domain)

    def __mul__(self, other) -> "Classicfun":
        """Pointwise multiplication.

        Provenance
        ----------
        MATLAB source : @classicfun/times.m (delegates to @chebtech/times.m)
        Chebfun commit: 7574c77
        """
        if self.isempty() or getattr(other, "_is_empty_object", False):
            return type(self).empty()
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun * other.onefun, self.domain)
        else:
            return self.__class__(self.onefun * other, self.domain)

    def __rmul__(self, other) -> "Classicfun":
        return self.__mul__(other)

    def __matmul__(self, other) -> "Classicfun":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued fun by a
        numeric matrix, mixing its columns (``coeffs @ A``).

        chebfunjax uses ``*`` for pointwise (``.*``) multiplication, so the
        column-mixing MATLAB ``*`` is exposed as ``@`` (matching Chebtech2).

        Provenance
        ----------
        MATLAB source : @classicfun/mtimes.m (delegates to @chebtech/mtimes.m)
        Chebfun commit: 7574c77
        """
        return self.__class__(self.onefun @ other, self.domain)

    def __truediv__(self, other) -> "Classicfun":
        """Division.

        A numeric-matrix divisor triggers MATLAB ``mrdivide`` (quasimatrix
        right division / least squares); a scalar or fun divisor is pointwise
        (``rdivide``).

        Provenance
        ----------
        MATLAB source : @classicfun/rdivide.m, @bndfun/mrdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Classicfun):
            self._check_domain(other)
            return self.__class__(self.onefun / other.onefun, self.domain)
        if _is_matrix_operand(other):
            return self._mrdivide(other)
        if _is_scalar_zero(other):
            # MATLAB CHEBTECH/double: F/0 -> NaN constant (per column).
            m = 1 if self.onefun.coeffs.ndim == 1 else self.onefun.coeffs.shape[1]
            shape = (1,) if self.onefun.coeffs.ndim == 1 else (1, m)
            nan_coeffs = jnp.full(shape, jnp.nan, dtype=jnp.float64)
            return self.__class__(Chebtech2.from_coeffs(nan_coeffs), self.domain)
        return self.__class__(self.onefun / other, self.domain)

    def __rtruediv__(self, other) -> "Classicfun":
        """Numeric divided by Classicfun.

        A numeric-matrix numerator triggers MATLAB ``mrdivide`` (double /
        quasimatrix least squares); a scalar numerator is pointwise.

        Provenance
        ----------
        MATLAB source : @classicfun/rdivide.m, @bndfun/mrdivide.m
        Chebfun commit: 7574c77
        """
        if _is_matrix_operand(other):
            return self._rmrdivide(other)
        return self.__class__(other / self.onefun, self.domain)

    def __pow__(self, exponent) -> "Classicfun":
        """Raise to a power.

        Provenance
        ----------
        MATLAB source : @classicfun/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, Classicfun):
            self._check_domain(exponent)
            return self.__class__(self.onefun ** exponent.onefun, self.domain)
        else:
            return self.__class__(self.onefun ** exponent, self.domain)

    def __abs__(self) -> "Classicfun":
        """Absolute value.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @classicfun/abs.m
        Chebfun commit: 7574c77
        """
        return self.__class__(abs(self.onefun), self.domain)

    def conj(self) -> "Classicfun":
        """Complex conjugate.

        Delegates to ``self.onefun.conj()`` (the domain map is real, so
        conjugation commutes with it).

        Provenance
        ----------
        MATLAB source : @classicfun/conj.m (delegates to @chebtech/conj.m)
        Chebfun commit: 7574c77
        """
        return self.__class__(self.onefun.conj(), self.domain)

    # ------------------------------------------------------------------
    # Calculus (with domain scaling)
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "Classicfun":
        """Differentiate *k* times with respect to x in [a, b].

        Applies the chain rule: the derivative with respect to x is the
        derivative with respect to the mapped variable (in [-1, 1])
        divided by the Jacobian ``(b - a) / 2`` raised to the *k*-th power.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Classicfun
            The k-th derivative, still on the same domain [a, b].

        Provenance
        ----------
        MATLAB source : @bndfun/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return self.__class__(self.onefun, self.domain)
        # Jacobian raised to the k-th power
        rescale = (self.domain.map_derivative()) ** k  # ((b-a)/2)^k
        new_onefun = self.onefun.diff(k) * (1.0 / rescale)
        return self.__class__(new_onefun, self.domain)

    def cumsum(self) -> "Classicfun":
        """Indefinite integral (antiderivative) with F(a) = 0.

        Scales the output of the underlying Chebtech2 cumsum (which
        satisfies F(-1) = 0 on [-1, 1]) by the Jacobian ``(b - a) / 2``
        to account for the change-of-variables.

        Returns
        -------
        Classicfun
            The antiderivative on the same domain [a, b].

        Provenance
        ----------
        MATLAB source : @bndfun/cumsum.m
        Chebfun commit: 7574c77
        """
        rescale = self.domain.map_derivative()  # (b-a)/2
        new_onefun = self.onefun.cumsum() * rescale
        return self.__class__(new_onefun, self.domain)

    def sum(self, dim: int = 1) -> "jax.Array | Classicfun":
        """Definite integral over [a, b].

        Scales the integral of the underlying Chebtech2 (over [-1, 1])
        by ``(b - a) / 2``.  ``dim=2`` sums ACROSS the columns of an
        array-valued fun and returns a scalar-column fun (MATLAB
        ``sum(f, 2)``, a no-op for scalar-valued input).

        Returns
        -------
        jax.Array (scalar or (m,)) or Classicfun
            The integral ``∫_a^b f(x) dx``, or the column-sum fun.

        Provenance
        ----------
        MATLAB source : @bndfun/sum.m
        Chebfun commit: 7574c77
        """
        if dim == 2:
            summed = self.onefun.sum(dim=2)
            if summed is self.onefun:
                return self
            return type(self)(onefun=summed, domain=self.domain)
        rescale = self.domain.map_derivative()  # (b-a)/2
        return self.onefun.sum() * jnp.float64(rescale)

    def inner(self, other: "Classicfun") -> jax.Array:
        """L2 inner product ⟨f, g⟩ = ∫_a^b f(x) g(x) dx.

        Parameters
        ----------
        other : Classicfun
            Must have the same domain.

        Returns
        -------
        jax.Array, scalar
            The inner product.

        Provenance
        ----------
        MATLAB source : @bndfun/innerProduct.m (delegates to @chebtech/innerProduct.m)
        Chebfun commit: 7574c77
        """
        self._check_domain(other)
        rescale = self.domain.map_derivative()  # (b-a)/2
        return self.onefun.inner(other.onefun) * jnp.float64(rescale)

    def norm(self, p: float = 2.0) -> jax.Array:
        """L-p norm on [a, b].

        For p=2, computes ``sqrt(∫_a^b f(x)² dx)``.

        Parameters
        ----------
        p : float, default 2.0
            The norm order.  Only p=2 uses the Chebyshev inner product.
            For p=1 or p=inf it delegates to pointwise evaluation.

        Returns
        -------
        jax.Array, scalar
            The norm value.

        Provenance
        ----------
        MATLAB source : @classicfun/normest.m
        Chebfun commit: 7574c77
        """
        if p == 2.0:
            rescale = self.domain.map_derivative()
            return (self.onefun.norm(p=2.0) ** 2 * jnp.float64(rescale)) ** 0.5
        elif p == jnp.inf or p == float("inf"):
            return jnp.array(self.onefun.vscale, dtype=jnp.float64)
        elif p == 1.0:
            return abs(self).sum()
        else:
            raise ValueError(
                f"norm(p={p}) is not supported. Use p=1, p=2, or p=inf."
            )

    def mean(self) -> jax.Array:
        """Mean value of the function over [a, b].

        Computes ``(1 / (b - a)) * ∫_a^b f(x) dx``.

        Returns
        -------
        jax.Array, scalar
        """
        a, b = self.domain.a, self.domain.b
        return self.sum() / jnp.float64(b - a)

    # ------------------------------------------------------------------
    # Matrix / least-squares division (quasimatrix)
    # ------------------------------------------------------------------

    def _column_coeffs(self) -> jax.Array:
        """Onefun coefficients as a 2-D ``(n, m)`` column array."""
        c = self.onefun.coeffs
        return c if c.ndim == 2 else c[:, None]

    def _mrdivide(self, other) -> "Classicfun":
        """``f / B`` with ``B`` a numeric matrix (MATLAB mrdivide).

        Solves ``X B = f`` in the least-squares sense column-wise, which for
        the finite column matrix ``B`` reduces to ``X.coeffs = f.coeffs B^+``.

        Provenance
        ----------
        MATLAB source : @bndfun/mrdivide.m, @chebtech/mrdivide.m
        Chebfun commit: 7574c77
        """
        B = jnp.asarray(other, dtype=jnp.result_type(other, jnp.float64))
        if B.ndim == 1:
            B = B[None, :]  # MATLAB treats a bare vector as a row.
        coeffs = self._column_coeffs()  # (n, m)
        m = coeffs.shape[1]
        if B.shape[1] != m:
            raise ValueError(
                "CHEBFUN:BNDFUN:mrdivide:size Matrix dimensions must agree."
            )
        new_coeffs = coeffs @ jnp.linalg.pinv(B)  # (n, p)
        return self._from_new_coeffs(new_coeffs)

    def _rmrdivide(self, other) -> "Classicfun":
        """``A / f`` with ``A`` a numeric matrix (MATLAB double/quasimatrix).

        Uses the weighted (L2) QR of ``self``: ``[Q, R] = qr(f)`` and
        ``X = Q (A R^{-1})^T``.  The Bndfun QR rescale cancels so no further
        domain scaling is applied.

        Provenance
        ----------
        MATLAB source : @bndfun/mrdivide.m, @chebtech/mrdivide.m
        Chebfun commit: 7574c77
        """
        A = jnp.asarray(other, dtype=jnp.result_type(other, jnp.float64))
        if A.ndim == 1:
            A = A[None, :]
        coeffs = self._column_coeffs()
        m = coeffs.shape[1]
        if A.shape[1] != m:
            raise ValueError(
                "CHEBFUN:BNDFUN:mrdivide:size Matrix dimensions must agree."
            )
        Q, R = self.qr()
        AR = jnp.linalg.solve(R.T, A.T).T  # A @ inv(R), shape (p, m)
        q_coeffs = Q._column_coeffs()  # (n, m)
        new_coeffs = q_coeffs @ AR.T  # (n, p)
        return self._from_new_coeffs(new_coeffs)

    def mrdivide(self, other) -> "Classicfun":
        """``f / B``: right matrix divide by a scalar or numeric matrix.

        A scalar divisor rescales the fun.  A matrix divisor gives the
        continuous-L2 least-squares solution of ``X B = f``, computed by
        delegating to the onefun's ``mrdivide`` exactly as MATLAB's
        ``@bndfun/mrdivide.m`` does.  ``mrdivide`` between two funs is an
        error; use ``/`` for pointwise division.

        Parameters
        ----------
        other : float or jax.Array
            Scalar or matrix divisor.  Its column count must match this
            fun's unless it is a scalar.

        Returns
        -------
        Classicfun

        Raises
        ------
        ValueError
            On a column-count mismatch (MATLAB ``mrdivide:size``), a
            fun/fun divide (``mrdivide:bndfunDivBndfun``), or a
            non-numeric divisor (``mrdivide:badArg``).

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @bndfun/mrdivide.m, @chebtech/mrdivide.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        rmrdivide, mldivide, qr
        """
        if isinstance(other, Classicfun):
            raise ValueError(
                "CHEBFUN:BNDFUN:mrdivide:bndfunDivBndfun: "
                "use ./ to divide a fun by a fun.")
        if isinstance(other, bool) or not isinstance(
                other, (int, float, complex, jnp.ndarray, list, tuple)) \
                and not hasattr(other, "shape"):
            raise ValueError(
                "CHEBFUN:BNDFUN:mrdivide:badArg: "
                f"fun/{type(other).__name__} is not well-defined.")
        return self.__class__(self.onefun.mrdivide(other), self.domain)

    @classmethod
    def rmrdivide(cls, numeric, fun) -> "Classicfun":
        """``A / f`` with a numeric ``A`` and a fun ``f`` (least squares).

        Delegates to the onefun's ``rmrdivide`` and then applies MATLAB's
        ``X = X / (0.5 * diff(domain))`` rescale, which moves the L2
        orthogonality from [-1, 1] onto the fun's domain.

        Parameters
        ----------
        numeric : float or jax.Array
            Numerator; its column count must match ``fun``'s.
        fun : Classicfun
            Denominator.

        Returns
        -------
        Classicfun

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @bndfun/mrdivide.m (``double / bndfun`` branch)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        mrdivide, mldivide, qr
        """
        onefun = type(fun.onefun).rmrdivide(numeric, fun.onefun)
        out = fun.__class__(onefun, fun.domain)
        return out * (1.0 / float(fun.domain.map_derivative()))

    def _from_new_coeffs(self, coeffs: jax.Array) -> "Classicfun":
        """Wrap ``(n, p)`` onefun coefficients as a fun on this domain.

        A single output column is squeezed to a scalar-valued fun so that
        evaluation returns a 1-D array (matching MATLAB feval shapes).
        """
        if coeffs.shape[1] == 1:
            coeffs = coeffs[:, 0]
        return self.__class__(Chebtech2.from_coeffs(coeffs), self.domain)

    def mldivide(self, other: "Classicfun") -> jax.Array:
        """``f \\ g`` (mldivide): least-squares expansion of ``g`` in ``f``.

        Returns the numeric coefficient matrix ``X`` minimising
        ``|| f X - g ||_{L2[a,b]}`` via ``[Q, R] = qr(f)`` and
        ``X = R^{-1} <Q, g>``.  The domain rescale cancels between ``Q`` and
        ``R``.

        Parameters
        ----------
        other : Classicfun
            Right-hand side (same domain).

        Returns
        -------
        jax.Array
            The coefficient matrix (scalar / vector / matrix as appropriate).

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @bndfun/mldivide.m, @chebtech/mldivide.m
        Chebfun commit: 7574c77
        """
        if not isinstance(other, Classicfun):
            raise TypeError(
                "Arguments to Bndfun mldivide must both be Bndfun objects."
            )
        self._check_domain(other)
        Q, R = self.qr()
        rhs = jnp.asarray(Q.inner(other))
        if rhs.ndim == 0:
            rhs = rhs[None]
        X = jnp.linalg.solve(R, rhs)
        return jnp.squeeze(X)

    # ------------------------------------------------------------------
    # Rootfinding and extrema
    # ------------------------------------------------------------------

    def roots(self, qz: bool = False, *, complex_roots: bool = False,
              all_roots: bool = False, prune: bool = False,
              recurse: bool = True) -> jax.Array:
        """Roots in [a, b].

        Delegates to the underlying tech rootfinder (which returns roots in
        [-1, 1]) and maps them back to [a, b] via the affine forward map.
        The option surface (``qz``, ``complex_roots``, ``all_roots``,
        ``prune``, ``recurse``) mirrors MATLAB ``@chebtech/roots.m``; the
        affine map carries complex roots through unchanged.

        NOT JIT-safe (variable output size).

        Returns
        -------
        jax.Array, shape (n_roots,)
            Roots in [a, b] (real by default; complex when ``all_roots`` /
            ``complex_roots``).

        Provenance
        ----------
        MATLAB source : @classicfun/roots.m
        Chebfun commit: 7574c77
        """
        # Roots in [-1, 1].  Preserve the exact default path for every
        # onefun type (some, e.g. Trigtech, take a different roots
        # signature); only forward the option surface when a non-default
        # option is requested (chebtech-backed funs).
        if qz or complex_roots or all_roots or prune or not recurse:
            onefun_roots = self.onefun.roots(
                qz=qz, complex_roots=complex_roots, all_roots=all_roots,
                prune=prune, recurse=recurse)
        else:
            onefun_roots = self.onefun.roots()
        # Map to [a, b] (affine map applies to complex roots too).
        return self.domain.forward_map(onefun_roots)

    def minandmax(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Global minimum and maximum on [a, b].

        Computes extrema of ``onefun`` on [-1, 1] and maps the positions
        back to [a, b] via the forward affine map.

        NOT JIT-safe.

        Returns
        -------
        (min_val, min_pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position in [a, b].
        (max_val, max_pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/minandmax.m
        Chebfun commit: 7574c77
        """
        (min_val, min_y), (max_val, max_y) = self.onefun.minandmax()
        # Map positions from [-1, 1] to [a, b]
        min_pos = self.domain.forward_map(min_y)
        max_pos = self.domain.forward_map(max_y)
        return (min_val, min_pos), (max_val, max_pos)

    def min(self) -> tuple[jax.Array, jax.Array]:
        """Global minimum on [a, b].

        NOT JIT-safe.

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/min.m
        Chebfun commit: 7574c77
        """
        (min_val, min_pos), _ = self.minandmax()
        return min_val, min_pos

    def max(self) -> tuple[jax.Array, jax.Array]:
        """Global maximum on [a, b].

        NOT JIT-safe.

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position in [a, b].

        Provenance
        ----------
        MATLAB source : @classicfun/max.m
        Chebfun commit: 7574c77
        """
        _, (max_val, max_pos) = self.minandmax()
        return max_val, max_pos

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    @abstractmethod
    def restrict(self, a: float, b: float) -> "Classicfun":
        """Restrict to a sub-interval [a, b].

        Parameters
        ----------
        a : float
            Left endpoint of the sub-interval.
        b : float
            Right endpoint of the sub-interval.

        Returns
        -------
        Classicfun
            A new instance representing the same function on [a, b].

        Raises
        ------
        ValueError
            If [a, b] is not a sub-interval of the current domain.
        """
        ...

    # ------------------------------------------------------------------
    # Splitting and comparison
    # ------------------------------------------------------------------

    def mat2cell(self, sizes=None) -> list:
        """Split an array-valued fun into a list of funs by column count.

        Delegates to the onefun's ``mat2cell`` and re-wraps each block in
        this fun's domain, mirroring MATLAB ``mat2cell(f, 1, N)``.  A block
        of size 1 becomes a scalar-valued fun.

        Parameters
        ----------
        sizes : sequence of int, optional
            Column counts of the blocks, which must sum to the number of
            columns of ``self``.  Defaults to one column per block.

        Returns
        -------
        list of Classicfun
            One fun per entry of ``sizes``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from chebfunjax.fun.bndfun import Bndfun
        >>> from chebfunjax.domain import Domain
        >>> d = Domain((-2.0, 7.0))
        >>> f = Bndfun.from_function(
        ...     lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), x], -1), d)
        >>> [g.onefun.coeffs.ndim for g in f.mat2cell([1, 2])]
        [1, 2]

        Provenance
        ----------
        MATLAB source : @classicfun/mat2cell.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        isequal, restrict
        """
        if self.isempty():
            return [self]
        if sizes is None:
            coeffs = self.onefun.coeffs
            ncols = coeffs.shape[1] if coeffs.ndim == 2 else 1
            sizes = [1] * ncols
        return [type(self).from_chebtech(t, self.domain)
                for t in self.onefun.mat2cell(sizes)]

    def isequal(self, other) -> bool:
        """True when two funs have the same domain and equal onefuns.

        Parameters
        ----------
        other : Classicfun
            Fun to compare against.

        Returns
        -------
        bool
            True if both the domains and the underlying onefuns agree.

        Provenance
        ----------
        MATLAB source : @classicfun/isequal.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        mat2cell
        """
        return _fun_isequal(self, other)

    # ------------------------------------------------------------------
    # Simplify
    # ------------------------------------------------------------------

    def simplify(self, tol: float | None = None) -> "Classicfun":
        """Return a simplified (potentially shorter) representation.

        Delegates to ``self.onefun.simplify()``.

        Parameters
        ----------
        tol : float or None, optional
            Tolerance for coefficient chopping.

        Returns
        -------
        Classicfun
            Simplified instance.

        Provenance
        ----------
        MATLAB source : @classicfun/simplify.m
        Chebfun commit: 7574c77
        """
        return self.__class__(self.onefun.simplify(tol), self.domain)

    # ------------------------------------------------------------------
    # Power-basis coefficients
    # ------------------------------------------------------------------

    def poly(self) -> jax.Array:
        """Monomial (power-basis) coefficients on [a, b].

        Returns coefficients ``C`` so that, for a scalar-valued fun,

            f(x) = C[0] x^N + C[1] x^(N-1) + ... + C[N-1] x + C[N]

        (highest power first).  For an array-valued fun the result has
        shape ``(m, N+1)`` with row ``k`` giving the coefficients of the
        ``k``-th column of ``f`` (mirroring MATLAB's row-vector-per-column
        convention).

        NOT JIT-safe (variable output size; dense recurrence).

        Provenance
        ----------
        MATLAB source : @bndfun/poly.m and @chebtech/poly.m
        Chebfun commit: 7574c77
        """
        # Monomial coefficients of the onefun on [-1, 1] (descending power).
        onefun_poly = _cheb_to_poly(self.onefun.coeffs)  # (m, n)
        m, n = onefun_poly.shape
        a = float(self.domain.a)
        b = float(self.domain.b)

        if a != -1.0 or b != 1.0:
            alpha = 2.0 / (b - a)
            beta = -(b + a) / (b - a)
            # Work in ASCENDING power order (out[:, 0] = constant term).
            out = onefun_poly[:, ::-1]
            k_all = jnp.arange(n, dtype=jnp.float64)
            new_cols = []
            for j in range(n):
                k = jnp.arange(j, n)  # powers >= j contribute to power j
                # Binomial coefficients C(k, j).
                binom = jnp.round(
                    jnp.exp(
                        _gammaln(k + 1.0)
                        - _gammaln(k - j + 1.0)
                        - _gammaln(k_all[j] + 1.0)
                    )
                )
                bba = binom * (beta ** (k - j)) * (alpha ** j)
                new_cols.append(jnp.sum(out[:, j:] * bba[None, :], axis=1))
            out = jnp.stack(new_cols, axis=1)  # ascending
            out = out[:, ::-1]  # back to descending
        else:
            out = onefun_poly

        if self.onefun.coeffs.ndim == 1:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        a, b = self.domain.a, self.domain.b
        lval = float(self.onefun(jnp.float64(-1.0)))
        rval = float(self.onefun(jnp.float64(1.0)))
        return (
            f"{self.__class__.__name__}("
            f"[{a:.4g}, {b:.4g}], n={self.n}, "
            f"lval={lval:.4g}, rval={rval:.4g})"
        )
