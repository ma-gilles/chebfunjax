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
    fred_op,
    inner_functional,
    sum_functional,
    volt_op,
    zeros_op,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

__all__ = [
    "ADChebfun",
    "linearize_op",
    "detect_linearity",
]


class ADMatrixDimensionError(ValueError):
    """Native mtimes identifier; retains ValueError catch compatibility."""

    identifier = "CHEBFUN:ADCHEBFUN:mtimes:dims"


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
        # Keep interior breakpoints as static operator metadata
        bpts = u.domain.breakpoints
        self.domain: tuple[float, ...] = tuple(float(x) for x in bpts)
        self.jumpLocations: tuple[float, ...] = ()
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
            result = _copy_ad(self, other)
            result.func = self.func + other.func
            result.jacobian = self.jacobian + other.jacobian
            result.linearity = tuple(a and b for a, b in zip(self.linearity, other.linearity))
            return result
        else:
            result = _copy_ad(self, other)
            result.func = self.func + other
            # jacobian unchanged
            return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ADChebfun):
            result = _copy_ad(self, other)
            result.func = self.func - other.func
            result.jacobian = self.jacobian - other.jacobian
            result.linearity = tuple(a and b for a, b in zip(self.linearity, other.linearity))
            return result
        else:
            result = _copy_ad(self, other)
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

    def mtimes(self, other):
        """MATLAB matrix multiplication, exposed by Python ``@``.

        Scalar numeric operands scale the primal and Jacobian. Functional
        operands raise the native dimension error, regardless of their values.

        Provenance: @adchebfun/adchebfun.m, mtimes and scalar times,
        Chebfun 7574c77680d7e82b79626300bf255498271a72df.
        Numeric AD array expansion is not implemented by this scalar class.
        """
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            result = _copy_ad(self)
            result.func = self.func * other
            result.jacobian = self.jacobian * other
            return result
        if hasattr(other, "dtype") and jnp.issubdtype(other.dtype, jnp.number):
            values = jnp.asarray(other)
            if values.size != 1:
                raise NotImplementedError(
                    "ADChebfun numeric array mtimes expansion is not implemented.")
            scalar = values.reshape(())
            result = _copy_ad(self)
            result.func = self.func * scalar
            result.jacobian = _multiply_jacobian(scalar, self.jacobian, self.domain)
            return result
        raise ADMatrixDimensionError(
            "CHEBFUN:ADCHEBFUN:mtimes:dims: Matrix dimensions must agree. "
            "Use pointwise multiplication for two ADChebfun or Chebfun objects.")

    def __matmul__(self, other):
        return self.mtimes(other)

    def __rmatmul__(self, other):
        return self.mtimes(other)

    def __mul__(self, other):
        """Product rule: d(f*g)[v] = f*dg[v] + g*df[v]."""
        if isinstance(other, ADChebfun):
            # Product rule
            result = _copy_ad(self, other)
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
            result = _copy_ad(self, other)
            result.func = self.func * float(other)
            result.jacobian = self.jacobian * float(other)
            return result
        else:
            # other is a Chebfun — diag multiplication
            result = _copy_ad(self, other)
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
            result = _copy_ad(self, other)
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
            result = _copy_ad(self, other)
            result.func = self.func / float(other)
            result.jacobian = self.jacobian * (1.0 / float(other))
            return result
        else:
            # other is a Chebfun (constant w.r.t. u)
            result = _copy_ad(self, other)
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
        """Apply the source power cases and literal Frechet multipliers.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (power, updateDomain).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        result = _copy_ad(self)
        if isinstance(exp, ADChebfun):
            result = _copy_ad(self, exp)
            result.linearity = tuple(
                az and bz and a and b
                for az, bz, a, b in zip(
                    _jac_zero_flags(self.jacobian), _jac_zero_flags(exp.jacobian),
                    self.linearity, exp.linearity,
                )
            )
            value = self.func ** exp.func
            result.jacobian = (
                _multiply_jacobian(
                    exp.func * self.func ** (exp.func-1), self.jacobian, self.domain,
                )
                + _multiply_jacobian(
                    value * self.func.log(), exp.jacobian, self.domain,
                )
            )
            result.func = value
        else:
            if isinstance(exp, (int, float)):
                if exp == 1:
                    return self
                if exp == 0:
                    result.func = self.func ** 0
                    result.jacobian = self.jacobian * 0.0
                    result.is_linear = True
                    return result
            result.linearity = _jac_zero_flags(self.jacobian)
            result.jacobian = _multiply_jacobian(
                exp * self.func ** (exp-1), self.jacobian, self.domain,
            )
            result.func = self.func ** exp
        return _update_ad_domain(result)

    def __rpow__(self, base):
        """Apply source scalar/Chebfun-to-AD power, including complex log.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (power, updateDomain).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun

        if isinstance(base, Chebfun):
            log_base = base.log()
            value = base ** self.func
        else:
            base_array = jnp.asarray(base)
            if base_array.ndim != 0:
                raise ValueError("AD power requires a scalar or Chebfun base")
            if not jnp.iscomplexobj(base_array) and bool(base_array < 0):
                base_array = base_array.astype(jnp.complex128)
            log_base = jnp.log(base_array)
            value = (self.func.__rpow__(base_array)
                     if isinstance(self.func, Chebfun)
                     else jnp.power(base_array, self.func))
        result = _copy_ad(self)
        result.linearity = _jac_zero_flags(self.jacobian)
        result.func = value
        result.jacobian = _multiply_jacobian(
            value * log_base, self.jacobian, self.domain,
        )
        return _update_ad_domain(result)

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

    def fred(self, kernel, onevar=None) -> "ADChebfun":
        """Apply the source Fredholm action and compose its linear operator.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (fred).
        Chebfun commit: 7574c77
        """
        from chebfunjax.operators.integral import fred
        result = _copy_ad(self)
        result.func = fred(kernel, self.func)
        result.jacobian = fred_op(kernel, self.domain, onevar)*self.jacobian
        return result

    def volt(self, kernel, onevar=None) -> "ADChebfun":
        """Apply the source Volterra action and compose its linear operator.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (volt).
        Chebfun commit: 7574c77
        """
        from chebfunjax.operators.integral import volt
        result = _copy_ad(self)
        result.func = volt(kernel, self.func)
        result.jacobian = volt_op(kernel, self.domain, onevar)*self.jacobian
        return result

    def cumprod(self) -> "ADChebfun":
        """Compute the indefinite product integral through log, cumsum and exp.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (cumprod).
        Chebfun commit: 7574c77
        """
        return self.log().cumsum().exp()

    def prod(self) -> "ADChebfun":
        """Compute the product integral and scale its integral Jacobian.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (prod).
        Chebfun commit: 7574c77
        """
        result = self.log().sum()
        result.func = jnp.exp(result.func)
        result.jacobian = _scale_jacobian(result.jacobian, result.func)
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

    def deflation_fun(self, u, roots, p, alp, norm_type="L2"):
        """Deflate this residual at the AD iterate u against known roots.

        The source rank-one derivative is composed with u's incoming
        Jacobian, extending its identity-seeded assumption to AD chains.
        Real scalar-valued iterates are supported by this derivative.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (deflationFun).
        Chebfun commit: 7574c77
        """
        from chebfunjax.operators.deflation import _roots

        result = _copy_ad(self, u)
        phi = jnp.asarray(1.)
        derivative = zeros_op(result.domain)
        for root in _roots(roots):
            delta = u.func-root
            squared = delta.norm("fro")**2
            functional = inner_functional(delta, result.domain)
            if norm_type != "L2":
                differentiated = delta.diff()
                squared = squared+differentiated.norm("fro")**2
                functional = functional + inner_functional(
                    differentiated, result.domain)*D(result.domain)
            reciprocal = 1/squared
            phi = phi*reciprocal
            derivative = derivative + _multiply_jacobian(
                self.func, _scale_jacobian(functional, reciprocal), result.domain)
        factor = phi**(p/2)
        result.jacobian = _scale_jacobian(self.jacobian, factor+alp) - (
            _scale_jacobian(derivative, p*factor)*u.jacobian)
        result.func = self.func*(alp+factor)
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

    def pow2(self) -> "ADChebfun":
        """Compute 2**self through the source power dispatch.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (pow2).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return 2 ** self

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

    def besselj(self, nu: float) -> "ADChebfun":
        """Apply source Bessel J and its Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (besselj).
        Chebfun commit: 7574c77
        """
        result = _copy_ad(self)
        result.linearity = _jac_zero_flags(self.jacobian)
        result.func = self.func.besselj(nu)
        multiplier = -self.func.besselj(nu+1)+nu*result.func/self.func
        result.jacobian = _multiply_jacobian(multiplier, self.jacobian, self.domain)
        return result

    def airy(self, k: int = 0) -> "ADChebfun":
        """Apply source Airy dispatch with its literal next-index multiplier.

        The source test covers Ai (k=0) and Bi (k=2). The source formula
        uses airy(k+1) as its derivative multiplier for every supplied k.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (airy).
        Chebfun commit: 7574c77
        """
        result = _copy_ad(self)
        result.linearity = _jac_zero_flags(self.jacobian)
        result.jacobian = _multiply_jacobian(
            self.func.airy(k+1), self.jacobian, self.domain)
        result.func = self.func.airy(k)
        return result

    def ellipj(self, m):
        """Apply all three Jacobi functions and source Frechet multipliers.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (ellipj).
        Chebfun commit: 7574c77
        """
        if isinstance(m, ADChebfun):
            raise TypeError("ellipj supports AD only in the first argument")
        sn, cn, dn = self.func.ellipj(m)
        results = []
        for value, multiplier in ((sn, cn*dn), (cn, -sn*dn), (dn, -m*sn*cn)):
            result = _copy_ad(self)
            result.func = value
            result.linearity = _jac_zero_flags(self.jacobian)
            result.jacobian = _multiply_jacobian(multiplier, self.jacobian, self.domain)
            results.append(result)
        return tuple(results)

    def erf(self) -> "ADChebfun":
        """Apply source erf and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (erf).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("erf", lambda f, g: 2*(-f**2).exp()/jnp.sqrt(jnp.pi))

    def erfc(self) -> "ADChebfun":
        """Apply source erfc and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (erfc).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("erfc", lambda f, g: -2*(-f**2).exp()/jnp.sqrt(jnp.pi))

    def erfcinv(self) -> "ADChebfun":
        """Apply source erfcinv and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (erfcinv).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("erfcinv", lambda f, g: -(g**2).exp()*jnp.sqrt(jnp.pi)/2)

    def erfcx(self) -> "ADChebfun":
        """Apply source erfcx and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (erfcx).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("erfcx", lambda f, g: -2/jnp.sqrt(jnp.pi)+2*f*g)

    def erfinv(self) -> "ADChebfun":
        """Apply source erfinv and its literal Frechet multiplier.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (erfinv).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return self._elementary("erfinv", lambda f, g: (g**2).exp()*jnp.sqrt(jnp.pi)/2)

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
        domain = self.domain
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

    def jump(self, x, c=0):
        """Right minus left limit minus ``c``, with the incoming AD chain.

        The explicit condition records its breakpoint for continuity handling.

        Provenance
        ----------
        MATLAB source: @adchebfun/adchebfun.m (jump).
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import jump
        from chebfunjax.operators.blocks import jump_functional

        result = _copy_ad(self)
        location = float(x)
        result.domain = tuple(sorted(set(self.domain + (location,))))
        result.func = jump(self.func, location, c)
        result.jacobian = jump_functional(location, result.domain) * self.jacobian
        result.jumpLocations = tuple(sorted(set(self.jumpLocations + (location,))))
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


def _copy_ad(f: ADChebfun, other=None) -> ADChebfun:
    """Shallow-copy an ADChebfun (without calling __init__)."""
    result = object.__new__(ADChebfun)
    result.func = f.func
    result.jacobian = f.jacobian
    result.linearity = f.linearity
    result.domain = f.domain
    result.jumpLocations = f.jumpLocations
    if isinstance(other, ADChebfun):
        result.domain = tuple(sorted(set(f.domain + other.domain)))
        result.jumpLocations = tuple(sorted(set(f.jumpLocations + other.jumpLocations)))
    return result


def _update_ad_domain(result):
    """Union the source primal, incoming and Jacobian breakpoints.

    Provenance: @adchebfun/adchebfun.m (private updateDomain),
    Chebfun 7574c77680d7e82b79626300bf255498271a72df.
    Breakpoint topology is static metadata; numerical array work stays JAX.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    domain = set(result.domain)
    if isinstance(result.func, Chebfun):
        domain.update(float(x) for x in result.func.domain.breakpoints)
    jacobian = result.jacobian
    jac_domain = jacobian.domain
    if hasattr(jac_domain, "breakpoints"):
        jac_domain = jac_domain.breakpoints
    domain.update(float(x) for x in jac_domain)
    result.domain = tuple(sorted(domain))
    if isinstance(jacobian, Chebfun):
        result.jacobian = jacobian._with_breakpoints(result.domain)
    else:
        # Power has just composed a new Jacobian, so its domain metadata
        # can be updated without changing the input AD object's Jacobian.
        jacobian.domain = result.domain
    return result


def _scale_jacobian(jacobian, factor):
    """Scale derivative values using JAX, preserving their output spaces."""
    if isinstance(jacobian, ChebMatrix):
        return jacobian.cellfun(lambda block: _scale_jacobian(block, factor))
    if isinstance(jacobian, (OperatorBlock, FunctionalBlock)):
        action = jacobian._apply_fn
        # Static block metadata requires a concrete scalar. Traced factors
        # cannot advertise a data-dependent zero/order flag.
        iszero = jacobian.iszero or bool(jnp.asarray(factor) == 0)
        kwargs = dict(
            domain=jacobian.domain,
            apply_fn=None if action is None else lambda u: factor*action(u),
            order=0 if isinstance(jacobian, OperatorBlock) and iszero else jacobian.order,
            iszero=iszero,
            isnotdiffint=jacobian.isnotdiffint,
        )
        coordinates = jacobian._coordinate_fn
        kwargs["_coordinate_fn"] = (None if coordinates is None
                                    else lambda n: factor*coordinates(n))
        if isinstance(jacobian, OperatorBlock):
            coefficients = jacobian._coeff_fn
            kwargs["coeff_fn"] = (None if coefficients is None
                                  else lambda: [factor*c for c in coefficients()])
        capability = getattr(jacobian, "_values_capability", None)
        if capability is not None:
            kwargs["_values_capability"] = type(capability)(
                lambda disc: factor*capability.realize(disc), capability.domain)
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
