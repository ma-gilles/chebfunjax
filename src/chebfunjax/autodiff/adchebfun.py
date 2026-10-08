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

import math
from typing import Callable

import jax.numpy as jnp

from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    FunctionalBlock,
    I,
    OperatorBlock,
    cumsum_op,
    diag,
    eval_at,
    inner_functional,
    sum_functional,
    zeros_op,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

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

    @property
    def is_linear(self) -> bool:
        """Whether all seeded-variable linearity flags are true.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (isLinear).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return all(self.linearity)

    @is_linear.setter
    def is_linear(self, value):
        self.linearity = (bool(value),) * len(getattr(self, "linearity", (True,)))

    def seed(self, k, v) -> "ADChebfun":
        """Reseed variable k (one based) as function or scalar blocks.

        A scalar v=0 selects a scalar parameter; v=1 selects one function.
        Integer v>=2 selects v independent functions. A boolean vector
        chooses an operator (true) or a Chebfun parameter column (false).

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (seed).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        from chebfunjax.chebfun1d.chebfun import chebfun

        result = _copy_ad(self)
        domain = tuple(float(x) for x in self.func.domain.breakpoints)
        values = jnp.asarray(v)
        if values.size == 1:
            scalar = values.reshape(()).item()
            if scalar < 2:
                result.jacobian = I(domain) if scalar else chebfun(1., domain=domain)
                result.domain = domain
                result.linearity = (True,)
                return result
            count = int(scalar)
            if count != scalar or count < 1:
                raise ValueError("seed shorthand must be a positive integer")
            flags = (True,) * count
        else:
            flags = tuple(bool(x) for x in values.reshape(-1))
        index = int(k)-1
        if int(k) != k or index < 0 or index >= len(flags):
            raise ValueError("seed index must be one based and within the block row")
        # Scalar derivative columns are actual constant Chebfuns, not zero
        # operators with an incorrectly infinite-dimensional input space.
        blocks = [((I(domain) if j == index else zeros_op(domain))
                   if flag else chebfun(float(j == index), domain=domain))
                  for j, flag in enumerate(flags)]
        result.jacobian = ChebMatrix([blocks], domain=domain)
        result.domain = domain
        result.linearity = (True,) * len(flags)
        return result

    # ------------------------------------------------------------------
    # Arithmetic — primal update + Jacobian chain rule
    # ------------------------------------------------------------------

    def __add__(self, other):
        """(a + b).jacobian = a.jacobian + b.jacobian  (if both AD)."""
        if isinstance(other, ADChebfun):
            result = _copy_ad(self)
            result.func = self.func + other.func
            result.jacobian = self.jacobian + other.jacobian
            result.linearity = tuple(a and b for a, b in zip(self.linearity, other.linearity))
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
            result.linearity = tuple(a and b for a, b in zip(self.linearity, other.linearity))
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
                _multiply_jacobian(self.func, other.jacobian, self.domain)
                + _multiply_jacobian(other.func, self.jacobian, other.domain)
            )
            # Linear only if one factor is constant AND the other is linear
            if len(self.linearity) > 1:
                fzero = _jac_zero_flags(result.jacobian)
                gzero = _jac_zero_flags(other.jacobian)
                any_constant = all(fzero) or all(gzero)
                result.linearity = tuple(
                    a and b and (any_constant or (fz and gz))
                    for a, b, fz, gz in zip(self.linearity, other.linearity, fzero, gzero)
                )
            else:
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
            result.jacobian = _multiply_jacobian(other, self.jacobian, self.domain)
            result.linearity = self.linearity
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
            result.jacobian = _multiply_jacobian(other, self.jacobian, self.domain)
            result.linearity = self.linearity
            return result

    def __truediv__(self, other):
        if isinstance(other, ADChebfun):
            # Quotient rule: d(f/g) = (g*df - f*dg) / g^2
            g = other.func
            g2 = g * g
            result = _copy_ad(self)
            result.func = self.func / g
            result.jacobian = (
                _multiply_jacobian(1.0 / g, self.jacobian, self.domain)
                - _multiply_jacobian(self.func / g2, other.jacobian, self.domain)
            )
            if len(self.linearity) > 1:
                gzero = _jac_zero_flags(other.jacobian)
                fzero = _jac_zero_flags(result.jacobian)
                result.linearity = tuple(
                    a and gz and (all(gzero) or fz)
                    for a, gz, fz in zip(self.linearity, gzero, fzero)
                )
            else:
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
            result.jacobian = _multiply_jacobian(1.0 / other, self.jacobian, self.domain)
            result.linearity = self.linearity
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
        result.jacobian = _multiply_jacobian(mult, self.jacobian, self.domain)
        _mark_nonlinear(result, self)
        return result

    def __pow__(self, exp):
        """Power: d(u^n) = n*u^{n-1} * du."""
        if isinstance(exp, ADChebfun):
            # f^g: d(f^g) = f^g * (g/f * df + log(f) * dg)
            fg = self.func ** exp.func
            result = _copy_ad(self)
            result.func = fg
            result.jacobian = (
                _multiply_jacobian(fg * exp.func / self.func, self.jacobian, self.domain)
                + _multiply_jacobian(fg * self.func.log(), exp.jacobian, exp.domain)
            )
            if len(self.linearity) > 1:
                result.linearity = tuple(a and b for a, b in zip(
                    _jac_zero_flags(self.jacobian), _jac_zero_flags(exp.jacobian)))
            else:
                result.is_linear = False
            return result
        elif isinstance(exp, (int, float)):
            n = float(exp)
            if n == 1.0:
                return self
            if n == 0.0:
                result = _copy_ad(self)
                result.func = self.func ** 0
                result.jacobian = self.jacobian * 0.0
                result.is_linear = True
                return result
            mult = n * self.func ** (n - 1)
            result = _copy_ad(self)
            result.func = self.func ** n
            result.jacobian = _multiply_jacobian(mult, self.jacobian, self.domain)
            _mark_nonlinear(result, self)
            return result
        else:
            # exp is a Chebfun (unusual)
            mult = exp * self.func ** (exp - 1)
            result = _copy_ad(self)
            result.func = self.func ** exp
            result.jacobian = _multiply_jacobian(mult, self.jacobian, self.domain)
            _mark_nonlinear(result, self)
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
            result.jacobian = _multiply_jacobian(val * log_base, self.jacobian, self.domain)
        else:
            result.jacobian = _multiply_jacobian(val * log_base_cheb, self.jacobian, self.domain)
        _mark_nonlinear(result, self)
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

    def cumsum(self, k: int = 1) -> "ADChebfun":
        """Integrate k times with each antiderivative zero at the left endpoint.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cumsum),
        @operatorBlock/operatorBlock.m (cumsum).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        # The operatorBlock source accepts only nonnegative integer orders.
        # Order is static metadata; all integration arithmetic remains JAX.
        try:
            order = int(k)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("cumsum order must be a nonnegative integer") from error
        if order < 0 or order != k:
            raise ValueError("cumsum order must be a nonnegative integer")
        result = _copy_ad(self)
        result.func = self.func.cumsum(order)
        # Preserve interior breakpoints in the integration operator, even
        # when the original AD seed records only the endpoint interval.
        result.domain = tuple(float(x) for x in result.func.domain.breakpoints)
        result.jacobian = cumsum_op(result.domain, order) * self.jacobian
        # Integration is linear: retain the incoming linearity information.
        return result

    def sum(self, *limits) -> "ADChebfun":
        """Integrate over the full domain, or over the two supplied limits.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sum).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        domain = tuple(float(x) for x in self.func.domain.breakpoints)
        if not limits:
            result = _copy_ad(self)
            result.func = self.func.sum()
            result.domain = domain
            result.jacobian = sum_functional(domain) * self.jacobian
            return result
        if len(limits) != 2:
            raise ValueError("CHEBFUN:ADCHEBFUN:sum:nargin")
        a, b = limits
        # Retain the literal source predicate, including reversed bounds.
        if a < domain[0] or b > domain[-1]:
            raise ValueError("CHEBFUN:ADCHEBFUN:sum:domain")
        integral = self.cumsum()
        return integral(b) - integral(a)

    def innerProduct(self, other) -> "ADChebfun":
        """Source AD bilinear integral: sum of the pointwise product.

        Unlike plain Chebfun.inner, the AD source does not conjugate the
        first operand. Real inputs agree with the usual L2 inner product.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (innerProduct).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return (self * other).sum()

    def norm(self, p=2):
        """Source L2 norm AD; other norm orders return the primal norm.

        The source derivative formula is qualified here for real functions.
        No holomorphic derivative of a complex norm is asserted.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (norm).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if p != 2 and p != "fro":
            return self.func.norm(p)
        result = _copy_ad(self)
        result.domain = tuple(float(x) for x in self.func.domain.breakpoints)
        result.linearity = _jac_zero_flags(self.jacobian)
        result.func = self.func.norm(2)
        jacobian = inner_functional(self.func, result.domain) * self.jacobian
        result.jacobian = _scale_jacobian(jacobian, 1/result.func)
        return result

    def mean(self) -> "ADChebfun":
        """Compute the mean on a finite domain, retaining its Jacobian.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (mean).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        a, b = self.domain[0], self.domain[-1]
        if math.isinf(a) or math.isinf(b):
            raise ValueError("CHEBFUN:ADCHEBFUN:mean:domain")
        return self.sum() / (b-a)

    def deriv(self, x, k: int = 1) -> "ADChebfun":
        """Evaluate the kth derivative at numeric locations.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (deriv, numeric-order branch).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self(x) if k == 0 else self.diff(k)(x)

    # ------------------------------------------------------------------
    # Unary functions — chain rule
    # ------------------------------------------------------------------

    def sin(self) -> "ADChebfun":
        """Apply source sin and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sin).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sin", lambda f, g: f.cos())

    def cos(self) -> "ADChebfun":
        """Apply source cos and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cos).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cos", lambda f, g: -f.sin())

    def tan(self) -> "ADChebfun":
        """Apply source tan and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (tan).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("tan", lambda f, g: f.sec()**2)

    def exp(self) -> "ADChebfun":
        """Apply source exp and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (exp).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("exp", lambda f, g: g)

    def log(self) -> "ADChebfun":
        """Apply source log and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (log).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("log", lambda f, g: 1/f)

    def sqrt(self) -> "ADChebfun":
        return self ** 0.5

    def sinh(self) -> "ADChebfun":
        """Apply source sinh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sinh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sinh", lambda f, g: f.cosh())

    def cosh(self) -> "ADChebfun":
        """Apply source cosh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cosh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cosh", lambda f, g: f.sinh())

    def tanh(self) -> "ADChebfun":
        """Apply source tanh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (tanh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("tanh", lambda f, g: f.sech()**2)

    def _elementary(self, name, derivative):
        """Share source copy/linearity/domain plumbing, retaining formulae."""
        result = _copy_ad(self)
        result.linearity = _jac_zero_flags(self.jacobian)
        result.func = getattr(self.func, name)()
        # Source unary methods retain the incoming AD domain.
        multiplier = derivative(self.func, result.func)
        result.jacobian = _multiply_jacobian(multiplier, self.jacobian, result.domain)
        return result

    def expm1(self) -> "ADChebfun":
        """Apply source expm1 and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (expm1).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("expm1", lambda f, g: f.exp())

    def log10(self) -> "ADChebfun":
        """Apply source log10 and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (log10).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("log10", lambda f, g: 1/(jnp.log(10.)*f))

    def log1p(self) -> "ADChebfun":
        """Apply source log1p and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (log1p).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("log1p", lambda f, g: 1/(f+1))

    def log2(self) -> "ADChebfun":
        """Apply source log2 and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (log2).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("log2", lambda f, g: 1/(jnp.log(2.)*f))

    def acos(self) -> "ADChebfun":
        """Apply source acos and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acos).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acos", lambda f, g: -1/(1-f**2).sqrt())

    def acosd(self) -> "ADChebfun":
        """Apply source acosd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acosd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acosd", lambda f, g: -(180/jnp.pi)/(1-f**2).sqrt())

    def acosh(self) -> "ADChebfun":
        """Apply source acosh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acosh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acosh", lambda f, g: 1/(f**2-1).sqrt())

    def acot(self) -> "ADChebfun":
        """Apply source acot and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acot).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acot", lambda f, g: -1/(1+f**2))

    def acotd(self) -> "ADChebfun":
        """Apply source acotd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acotd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acotd", lambda f, g: -(180/jnp.pi)/(1+f**2))

    def acoth(self) -> "ADChebfun":
        """Apply source acoth and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acoth).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acoth", lambda f, g: -1/(f**2-1))

    def acsc(self) -> "ADChebfun":
        """Apply source acsc and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acsc).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acsc", lambda f, g: -1/(abs(f)*(f**2-1).sqrt()))

    def acscd(self) -> "ADChebfun":
        """Apply source acscd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acscd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acscd", lambda f, g: -(180/jnp.pi)/(abs(f)*(f**2-1).sqrt()))

    def acsch(self) -> "ADChebfun":
        """Apply source acsch and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (acsch).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("acsch", lambda f, g: -1/(f*(1+f**2).sqrt()))

    def asec(self) -> "ADChebfun":
        """Apply source asec and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asec).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asec", lambda f, g: 1/(abs(f)*(f**2-1).sqrt()))

    def asecd(self) -> "ADChebfun":
        """Apply source asecd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asecd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asecd", lambda f, g: (180/jnp.pi)/(abs(f)*(f**2-1).sqrt()))

    def asech(self) -> "ADChebfun":
        """Apply source asech and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asech).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asech", lambda f, g: -1/(f*(1-f**2).sqrt()))

    def asin(self) -> "ADChebfun":
        """Apply source asin and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asin).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asin", lambda f, g: 1/(1-f**2).sqrt())

    def asind(self) -> "ADChebfun":
        """Apply source asind and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asind).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asind", lambda f, g: (180/jnp.pi)/(1-f**2).sqrt())

    def asinh(self) -> "ADChebfun":
        """Apply source asinh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (asinh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("asinh", lambda f, g: 1/(f**2+1).sqrt())

    def atan(self) -> "ADChebfun":
        """Apply source atan and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (atan).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("atan", lambda f, g: 1/(1+f**2))

    def atand(self) -> "ADChebfun":
        """Apply source atand and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (atand).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("atand", lambda f, g: (180/jnp.pi)/(1+f**2))

    def atanh(self) -> "ADChebfun":
        """Apply source atanh and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (atanh).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("atanh", lambda f, g: 1/(1-f**2))

    def cosd(self) -> "ADChebfun":
        """Apply source cosd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cosd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cosd", lambda f, g: -jnp.pi/180*f.sind())

    def cot(self) -> "ADChebfun":
        """Apply source cot and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cot).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cot", lambda f, g: -f.csc()**2)

    def cotd(self) -> "ADChebfun":
        """Apply source cotd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cotd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cotd", lambda f, g: -(jnp.pi/180)*f.cscd()**2)

    def coth(self) -> "ADChebfun":
        """Apply source coth and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (coth).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("coth", lambda f, g: -f.csch()**2)

    def csc(self) -> "ADChebfun":
        """Apply source csc and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (csc).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("csc", lambda f, g: -f.cot()*g)

    def cscd(self) -> "ADChebfun":
        """Apply source cscd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cscd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("cscd", lambda f, g: -jnp.pi/180*f.cotd()*g)

    def csch(self) -> "ADChebfun":
        """Apply source csch and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (csch).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("csch", lambda f, g: -f.coth()*g)

    def sec(self) -> "ADChebfun":
        """Apply source sec and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sec).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sec", lambda f, g: f.tan()*g)

    def secd(self) -> "ADChebfun":
        """Apply source secd and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (secd).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("secd", lambda f, g: jnp.pi/180*f.tand()*g)

    def sech(self) -> "ADChebfun":
        """Apply source sech and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sech).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sech", lambda f, g: -f.tanh()*g)

    def sinc(self) -> "ADChebfun":
        """Apply source sinc and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sinc).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sinc", lambda f, g: f._apply_fun(lambda u: (u*jnp.cos(u)-jnp.sin(u))/(u**2)))

    def sind(self) -> "ADChebfun":
        """Apply source sind and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (sind).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("sind", lambda f, g: jnp.pi/180*f.cosd())

    def tand(self) -> "ADChebfun":
        """Apply source tand and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (tand).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("tand", lambda f, g: (jnp.pi/180)*f.secd()**2)

    # ------------------------------------------------------------------
    # Evaluation (f(x) syntax) — returns a scalar ADChebfun
    # ------------------------------------------------------------------

    def __call__(self, x):
        """Evaluate at numeric points, retaining scalar or vector Jacobians.

        Locations are static operator metadata; primal values and operator
        arithmetic remain JAX arrays, including complex values.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (feval),
        @functionalBlock/functionalBlock.m (feval).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        points = jnp.asarray(x, dtype=jnp.float64)
        domain = tuple(float(v) for v in self.func.domain.breakpoints)
        rows = [eval_at(float(v), domain=domain) for v in points.reshape(-1)]
        if points.ndim == 0:
            evaluation = rows[0]
        else:
            evaluation = FunctionalBlock(
                lambda disc: jnp.stack([row.matrix(disc) for row in rows]),
                domain=domain,
                apply_fn=lambda u: u(points),
                isnotdiffint=True,
            )
        result = _copy_ad(self)
        result.func = self.func(points)
        result.domain = domain
        result.jacobian = evaluation * self.jacobian
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
    domain: tuple[float, ...] | None = None,
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
    domain : tuple of float or None
        Physical domain, including optional interior breakpoints. If ``None``,
        inferred from ``u0.domain``.

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

    # Infer domain
    if domain is None:
        bpts = u0.domain.breakpoints
        domain = tuple(float(point) for point in bpts)

    # Build the identity (x) Chebfun
    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
    x_fun = _chebfun(lambda x: x, domain=domain, n=2)

    # Wrap u0 in an ADChebfun to track derivatives
    ad_u = ADChebfun(u0)

    # Call the operator
    from chebfunjax.utils.misc import op_arity
    nargs = op_arity(op, 2)

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
    domain: tuple[float, ...] | None = None,
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
    domain : tuple of float or None
        Physical domain, including optional interior breakpoints.

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
    MATLAB source : @adchebfun/adchebfun.m, @chebop/islinear.m, @chebop/linearize.m
    Chebfun commit: 7574c77
    """

    if domain is None:
        bpts = u0.domain.breakpoints
        domain = tuple(float(point) for point in bpts)

    from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
    x_fun = _chebfun(lambda x: x, domain=domain, n=2)

    ad_u = ADChebfun(u0)

    from chebfunjax.utils.misc import op_arity
    nargs = op_arity(op, 2)

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
    result.linearity = f.linearity
    result.domain = f.domain
    return result


def _scale_jacobian(jacobian, factor):
    """Scale derivative values using JAX, preserving their output spaces."""
    if isinstance(jacobian, ChebMatrix):
        return jacobian.cellfun(lambda block: _scale_jacobian(block, factor))
    if isinstance(jacobian, (OperatorBlock, FunctionalBlock)):
        action = jacobian._apply_fn
        kwargs = dict(
            domain=jacobian.domain,
            apply_fn=None if action is None else lambda u: factor*action(u),
            order=jacobian.order,
            iszero=jacobian.iszero,
            isnotdiffint=jacobian.isnotdiffint,
        )
        return type(jacobian)(lambda disc: factor*jacobian.matrix(disc), **kwargs)
    return factor*jacobian


def _multiply_jacobian(multiplier, jacobian, domain):
    """Source operatorBlock.mult returns a scalar for numeric scalars."""
    if callable(multiplier):
        return diag(multiplier, domain)*jacobian
    values = jnp.asarray(multiplier)
    if values.ndim != 0:
        raise ValueError("numeric vector AD multipliers require an explicit output layout")
    return _scale_jacobian(jacobian, values)


def _jac_zero_flags(jacobian):
    """Structural zero flags for a source seed row (static metadata)."""
    if isinstance(jacobian, ChebMatrix):
        if jacobian.nrows != 1:
            raise ValueError("AD Jacobian must have exactly one block row")
        return tuple(_jac_zero_flags(block)[0] for block in jacobian.blocks[0])
    flag = getattr(jacobian, "iszero", None)
    if flag is not None:
        return (bool(flag() if callable(flag) else flag),)
    return (bool(jnp.all(jnp.asarray(jacobian) == 0)),)


def _mark_nonlinear(result, argument):
    if len(argument.linearity) > 1:
        result.linearity = _jac_zero_flags(argument.jacobian)
    else:
        # Preserve the established aggregate contract for unseeded callers.
        result.is_linear = False


def _jac_is_zero(f: ADChebfun) -> bool:
    """Heuristic check: is the Jacobian the zero operator?

    We probe by evaluating the Jacobian matrix at a small n and checking
    if it is close to zero.
    """
    if isinstance(f.jacobian, ChebMatrix) or not isinstance(f.jacobian, OperatorBlock):
        return all(_jac_zero_flags(f.jacobian))
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
