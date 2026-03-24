"""adchebfun — automatic Fréchet differentiation of Chebfun operators.

Implements the *symbolic linearization* approach to computing Fréchet
derivatives of nonlinear differential operators, following MATLAB Chebfun's
``@adchebfun`` class.

Overview
--------
For a nonlinear operator ``N[u]`` (e.g. ``u'' + u^2``), the Fréchet
derivative at a function ``u0`` is the linear operator ``dN[u0]`` such that::

    N[u0 + eps*v] = N[u0] + eps * dN[u0](v) + O(eps^2)

For ``N[u] = u'' + u^2``  this gives ``dN[u0](v) = v'' + 2*u0*v``.

:class:`ADChebfun` computes this via the :class:`~chebfunjax.autodiff.treevar.TreeVar`
expression-tree approach:

1. Evaluate the operator on a :class:`TreeVar` dummy variable to record the
   expression tree.
2. Use :func:`~chebfunjax.autodiff.treevar.linearize_tree` to convert the
   tree into a Fréchet-derivative :class:`~chebfunjax.operators.blocks.OperatorBlock`.

This is equivalent to MATLAB's ``chebop/linearize.m`` calling ``adchebfun``
and inspecting the resulting ``jacobian`` field.

Usage
-----
::

    from chebfunjax.autodiff.adchebfun import linearize_op

    # Build the Fréchet-derivative OperatorBlock of N[u]=u''+u^2 at u=sin
    N = lambda x, u: u.diff(2) + u ** 2
    u0 = chebfun(jnp.sin, domain=(0.0, jnp.pi))
    J = linearize_op(N, u0, domain=(0.0, float(jnp.pi)))
    # J is an OperatorBlock representing  v ↦ v'' + 2*sin(x)*v

Translated from MATLAB Chebfun ``@adchebfun`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import inspect
from typing import Callable

import jax.numpy as jnp

from chebfunjax.autodiff.treevar import TreeVar, linearize_tree
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    OperatorBlock,
    D,
    I,
    diag,
)

__all__ = [
    "ADChebfun",
    "linearize_op",
    "detect_linearity",
]


# ===========================================================================
# ADChebfun — thin wrapper  (dual-number style: value + Jacobian)
# ===========================================================================


class ADChebfun:
    """Dual-number object for Fréchet AD of Chebfun operators.

    Mirrors MATLAB's ``adchebfun``.  Each ``ADChebfun`` carries:

    ``func``
        The Chebfun value (the primal).
    ``jacobian``
        An :class:`~chebfunjax.operators.blocks.OperatorBlock` representing
        the Fréchet derivative of ``func`` with respect to the *seeding*
        variable.  Initially the identity operator (seeded at construction).
    ``is_linear``
        Bool — whether the Fréchet derivative is constant (independent of
        ``func``), i.e. the operation is linear.
    ``domain``
        Physical domain tuple.

    The arithmetic methods follow the chain rule: for each operation, they
    update both the value (``func``) and the derivative (``jacobian``).

    Typical use via :func:`linearize_op` — end users rarely instantiate
    ``ADChebfun`` directly.

    Parameters
    ----------
    u : Chebfun
        The primal value.  The Jacobian is seeded as the identity operator.

    Provenance
    ----------
    MATLAB source : @adchebfun/adchebfun.m
    Chebfun commit: 7574c77
    """

    def __init__(self, u) -> None:
        from chebfunjax.chebfun1d.chebfun import Chebfun

        if not isinstance(u, Chebfun):
            raise TypeError(
                f"ADChebfun expects a Chebfun, got {type(u).__name__}."
            )
        self.func = u
        # Extract domain as a (a, b) tuple
        bpts = u.domain.breakpoints
        self.domain: tuple[float, float] = (float(bpts[0]), float(bpts[-1]))
        # Seed Jacobian as identity operator
        self.jacobian: OperatorBlock = I(self.domain)
        self.is_linear: bool = True

    # ------------------------------------------------------------------
    # Arithmetic — primal update + Jacobian chain rule
    # ------------------------------------------------------------------

    def __add__(self, other):
        """(a + b).jacobian = a.jacobian + b.jacobian  (if both AD)."""
        if isinstance(other, ADChebfun):
            result = _copy_ad(self)
            result.func = self.func + other.func
            result.jacobian = self.jacobian + other.jacobian
            result.is_linear = self.is_linear and other.is_linear
            return result
        else:
            result = _copy_ad(self)
            result.func = self.func + other
            # jacobian unchanged
            return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ADChebfun):
            result = _copy_ad(self)
            result.func = self.func - other.func
            result.jacobian = self.jacobian - other.jacobian
            result.is_linear = self.is_linear and other.is_linear
            return result
        else:
            result = _copy_ad(self)
            result.func = self.func - other
            return result

    def __rsub__(self, other):
        result = _copy_ad(self)
        result.func = other - self.func
        result.jacobian = -self.jacobian
        return result

    def __neg__(self):
        result = _copy_ad(self)
        result.func = -self.func
        result.jacobian = -self.jacobian
        return result

    def __pos__(self):
        return self

    def __mul__(self, other):
        """Product rule: d(f*g)[v] = f*dg[v] + g*df[v]."""
        if isinstance(other, ADChebfun):
            # Product rule
            result = _copy_ad(self)
            result.func = self.func * other.func
            result.jacobian = (
                diag(self.func, self.domain) * other.jacobian
                + diag(other.func, other.domain) * self.jacobian
            )
            # Linear only if one factor is constant AND the other is linear
            result.is_linear = (
                self.is_linear and other.is_linear
                and (_jac_is_zero(self) or _jac_is_zero(other))
            )
            return result
        elif isinstance(other, (int, float)):
            result = _copy_ad(self)
            result.func = self.func * float(other)
            result.jacobian = self.jacobian * float(other)
            return result
        else:
            # other is a Chebfun — diag multiplication
            result = _copy_ad(self)
            result.func = self.func * other
            result.jacobian = diag(other, self.domain) * self.jacobian
            result.is_linear = self.is_linear
            return result

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = _copy_ad(self)
            result.func = float(other) * self.func
            result.jacobian = self.jacobian * float(other)
            return result
        else:
            # other is a Chebfun
            result = _copy_ad(self)
            result.func = other * self.func
            result.jacobian = diag(other, self.domain) * self.jacobian
            result.is_linear = self.is_linear
            return result

    def __truediv__(self, other):
        if isinstance(other, ADChebfun):
            # Quotient rule: d(f/g) = (g*df - f*dg) / g^2
            g = other.func
            g2 = g * g
            result = _copy_ad(self)
            result.func = self.func / g
            result.jacobian = (
                diag(1.0 / g, self.domain) * self.jacobian
                - diag(self.func / g2, self.domain) * other.jacobian
            )
            result.is_linear = (
                self.is_linear and other.is_linear
                and _jac_is_zero(other)
            )
            return result
        elif isinstance(other, (int, float)):
            result = _copy_ad(self)
            result.func = self.func / float(other)
            result.jacobian = self.jacobian * (1.0 / float(other))
            return result
        else:
            # other is a Chebfun (constant w.r.t. u)
            result = _copy_ad(self)
            result.func = self.func / other
            result.jacobian = diag(1.0 / other, self.domain) * self.jacobian
            result.is_linear = self.is_linear
            return result

    def __rtruediv__(self, other):
        # other / self — chain rule: d(c/f) = -c/f^2 * df
        g2 = self.func * self.func
        if isinstance(other, (int, float)):
            mult = -float(other) / g2
        else:
            mult = other / (-g2)
        result = _copy_ad(self)
        result.func = other / self.func
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.is_linear = False
        return result

    def __pow__(self, exp):
        """Power: d(u^n) = n*u^{n-1} * du."""
        if isinstance(exp, ADChebfun):
            # f^g: d(f^g) = f^g * (g/f * df + log(f) * dg)
            fg = self.func ** exp.func
            result = _copy_ad(self)
            result.func = fg
            result.jacobian = (
                diag(fg * exp.func / self.func, self.domain) * self.jacobian
                + diag(fg * self.func.log(), exp.domain) * exp.jacobian
            )
            result.is_linear = False
            return result
        elif isinstance(exp, (int, float)):
            n = float(exp)
            if n == 1.0:
                return self
            if n == 0.0:
                result = _copy_ad(self)
                result.func = self.func ** 0
                result.jacobian = I(self.domain) * 0.0
                result.is_linear = True
                return result
            mult = n * self.func ** (n - 1)
            result = _copy_ad(self)
            result.func = self.func ** n
            result.jacobian = diag(mult, self.domain) * self.jacobian
            result.is_linear = False
            return result
        else:
            # exp is a Chebfun (unusual)
            mult = exp * self.func ** (exp - 1)
            result = _copy_ad(self)
            result.func = self.func ** exp
            result.jacobian = diag(mult, self.domain) * self.jacobian
            result.is_linear = False
            return result

    def __rpow__(self, base):
        """base^self: d(a^u) = a^u * log(a) * du."""
        import math
        if isinstance(base, (int, float)):
            log_base = math.log(float(base))
            val = float(base) ** self.func
        else:
            log_base_cheb = base.log()
            val = base ** self.func
        result = _copy_ad(self)
        result.func = val
        if isinstance(base, (int, float)):
            result.jacobian = diag(val * log_base, self.domain) * self.jacobian
        else:
            result.jacobian = diag(val * log_base_cheb, self.domain) * self.jacobian
        result.is_linear = False
        return result

    # ------------------------------------------------------------------
    # Calculus operators
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "ADChebfun":
        """Differentiation: d(Du)[v] = D * du[v].

        The Fréchet derivative of the ``k``-th derivative operator is
        itself: ``D^k`` applied to the perturbation ``v``.
        """
        result = _copy_ad(self)
        result.func = self.func.diff(k)
        result.jacobian = D(self.domain, order=k) * self.jacobian
        # diff is linear, is_linear unchanged
        return result

    def cumsum(self) -> "ADChebfun":
        """Anti-differentiation (cumulative sum).

        The Fréchet derivative of cumsum is itself: an integration operator.
        """
        from chebfunjax.operators.blocks import sum_functional

        # Build the antidifferentiation OperatorBlock
        # In collocation: cumsum matrix C satisfies Cf = antiderivative values
        dom = self.domain

        def _cumsum_fn(disc: ChebColloc2Disc):
            from chebfunjax.utils.diffmat import diffmat as _dm
            # Cumsum matrix = inverse of differentiation (up to BCs)
            # Use the standard Clenshaw-Curtis approach: integrate via coefficients
            n = disc.n
            a, b = disc.domain
            scale = 0.5 * (b - a)
            # Build cumsum matrix numerically via columns
            import jax.numpy as jnp
            D1 = _dm(n, 1, domain=disc.domain)
            # We want C such that D1 @ C = I (antiderivative)
            # Chebfun uses a direct construction. For our purposes
            # use the pseudo-inverse (pinv) of the differentiation matrix.
            # This is consistent with how linop.solve works.
            return jnp.linalg.pinv(D1)

        cumsum_op = OperatorBlock(_cumsum_fn, order=-1, domain=dom)
        result = _copy_ad(self)
        result.func = self.func.cumsum()
        result.jacobian = cumsum_op * self.jacobian
        return result

    # ------------------------------------------------------------------
    # Unary functions — chain rule
    # ------------------------------------------------------------------

    def sin(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        mult = self.func.cos()
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.sin()
        return result

    def cos(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        mult = -self.func.sin()
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.cos()
        return result

    def tan(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        cos_u = self.func.cos()
        mult = 1.0 / (cos_u * cos_u)
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.tan() if hasattr(self.func, "tan") else (
            self.func.sin() / self.func.cos()
        )
        return result

    def exp(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        result.func = self.func.exp()
        result.jacobian = diag(result.func, self.domain) * self.jacobian
        return result

    def log(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        mult = 1.0 / self.func
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.log() if hasattr(self.func, "log") else (
            _chebfun_log(self.func)
        )
        return result

    def sqrt(self) -> "ADChebfun":
        return self ** 0.5

    def sinh(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        mult = self.func.cosh()
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.sinh()
        return result

    def cosh(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        mult = self.func.sinh()
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = self.func.cosh()
        return result

    def tanh(self) -> "ADChebfun":
        result = _copy_ad(self)
        result.is_linear = False
        cosh_u = self.func.cosh()
        mult = 1.0 / (cosh_u * cosh_u)
        result.jacobian = diag(mult, self.domain) * self.jacobian
        result.func = (
            self.func.tanh() if hasattr(self.func, "tanh") else
            self.func.sinh() / self.func.cosh()
        )
        return result

    # ------------------------------------------------------------------
    # Evaluation (f(x) syntax) — returns a scalar ADChebfun
    # ------------------------------------------------------------------

    def __call__(self, x):
        """Point evaluation: returns a scalar-valued ADChebfun."""
        from chebfunjax.operators.blocks import eval_at as _eval_at
        E = _eval_at(float(x), domain=self.domain)
        result = _copy_ad(self)
        result.func = float(self.func(jnp.array(x, dtype=jnp.float64)))
        result.jacobian = E * self.jacobian  # FunctionalBlock * OperatorBlock
        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ADChebfun(domain={self.domain}, "
            f"is_linear={self.is_linear})"
        )


# ===========================================================================
# Public API
# ===========================================================================


def linearize_op(
    op: Callable,
    u0,
    domain: tuple[float, float] | None = None,
) -> OperatorBlock:
    """Compute the Fréchet-derivative OperatorBlock of ``op`` at ``u0``.

    Linearizes the nonlinear operator ``op`` at the Chebfun ``u0`` using the
    ``TreeVar`` symbolic approach (option 2 from the task description).

    The operator ``op`` must accept either ``op(x, u)`` or ``op(u)`` where
    ``x`` is the identity Chebfun and ``u`` is an ``ADChebfun``.

    Parameters
    ----------
    op : callable
        The differential operator lambda.  Signature ``(x, u)`` or ``(u)``.
        Must be compatible with both :class:`~chebfunjax.chebfun1d.Chebfun`
        and :class:`ADChebfun` arguments.
    u0 : Chebfun
        The linearization point.
    domain : (float, float) or None
        Physical domain.  If ``None``, inferred from ``u0.domain``.

    Returns
    -------
    OperatorBlock
        The Fréchet derivative ``dN[u0]``.

    Examples
    --------
    Linearize ``N[u] = u'' + u^2`` at ``u0 = sin(x)`` on [0, π]:

    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> from chebfunjax.autodiff.adchebfun import linearize_op
    >>> u0 = chebfun(jnp.sin, domain=(0.0, float(jnp.pi)))
    >>> N = lambda x, u: u.diff(2) + u ** 2
    >>> J = linearize_op(N, u0, domain=(0.0, float(jnp.pi)))

    The resulting ``J`` represents ``v ↦ v'' + 2*sin(x)*v``.

    Provenance
    ----------
    MATLAB source : @chebop/linearize.m, @adchebfun (operator overloads)
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    # Infer domain
    if domain is None:
        bpts = u0.domain.breakpoints
        domain = (float(bpts[0]), float(bpts[-1]))

    # Build the identity (x) Chebfun
    a, b = domain
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
    x_fun = _chebfun(lambda x: x, domain=domain, n=2)

    # Wrap u0 in an ADChebfun to track derivatives
    ad_u = ADChebfun(u0)

    # Call the operator
    try:
        nargs = len(inspect.signature(op).parameters)
    except (TypeError, ValueError):
        nargs = 2

    if nargs == 1:
        result_ad = op(ad_u)
    else:
        result_ad = op(x_fun, ad_u)

    if isinstance(result_ad, ADChebfun):
        return result_ad.jacobian
    elif isinstance(result_ad, OperatorBlock):
        return result_ad
    else:
        # Scalar result — zero operator
        return I(domain) * 0.0


def detect_linearity(
    op: Callable,
    u0,
    domain: tuple[float, float] | None = None,
) -> bool:
    """Test whether ``op`` is linear.

    Uses ADChebfun to probe whether the Fréchet derivative is constant
    (independent of the linearization point ``u0``).

    Parameters
    ----------
    op : callable
        The operator to test.
    u0 : Chebfun
        The test point (for nonlinear operators the answer may depend on this).
    domain : (float, float) or None
        Physical domain.

    Returns
    -------
    bool
        ``True`` if the operator is linear (Jacobian is constant), ``False``
        otherwise.

    Examples
    --------
    >>> from chebfunjax.chebfun1d.chebfun import chebfun
    >>> u0 = chebfun(lambda x: x, domain=(-1.0, 1.0))
    >>> detect_linearity(lambda x, u: u.diff(2) + u, u0)
    True
    >>> detect_linearity(lambda x, u: u.diff(2) + u ** 2, u0)
    False

    Provenance
    ----------
    MATLAB source : @adchebfun/isLinear.m, @chebop/isLinear.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    if domain is None:
        bpts = u0.domain.breakpoints
        domain = (float(bpts[0]), float(bpts[-1]))

    a, b = domain
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
    x_fun = _chebfun(lambda x: x, domain=domain, n=2)

    ad_u = ADChebfun(u0)

    try:
        nargs = len(inspect.signature(op).parameters)
    except (TypeError, ValueError):
        nargs = 2

    if nargs == 1:
        result_ad = op(ad_u)
    else:
        result_ad = op(x_fun, ad_u)

    if isinstance(result_ad, ADChebfun):
        return result_ad.is_linear
    # Scalar result — trivially linear
    return True


# ===========================================================================
# Private helpers
# ===========================================================================


def _copy_ad(f: ADChebfun) -> ADChebfun:
    """Shallow-copy an ADChebfun (without calling __init__)."""
    result = object.__new__(ADChebfun)
    result.func = f.func
    result.jacobian = f.jacobian
    result.is_linear = f.is_linear
    result.domain = f.domain
    return result


def _jac_is_zero(f: ADChebfun) -> bool:
    """Heuristic check: is the Jacobian the zero operator?

    We probe by evaluating the Jacobian matrix at a small n and checking
    if it is close to zero.
    """
    try:
        disc = ChebColloc2Disc(4, f.domain)
        mat = f.jacobian.matrix(disc)
        return float(jnp.max(jnp.abs(mat))) < 1e-14
    except Exception:
        return False


def _chebfun_log(f):
    """Compute log(Chebfun f) when f doesn't have a .log() method."""
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
    bpts = f.domain.breakpoints
    a, b = float(bpts[0]), float(bpts[-1])
    return _chebfun(lambda x: jnp.log(f(x)), domain=(a, b))
