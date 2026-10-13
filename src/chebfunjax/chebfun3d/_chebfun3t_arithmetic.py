"""JAX arithmetic methods for the dense ``Chebfun3T`` representation.

This mixin deliberately depends only on the dense public contract:
``coeffs``, ``domain``, ``vscale``, ``isempty()``, ``from_coeffs()``, and the
``chebfun3t(callable, domain)`` factory.  The owning class can inherit it
without putting operator implementations into the dense storage module.

Provenance
----------
MATLAB source : ``@chebfun3t/{plus,minus,uminus,mtimes,times,rdivide,
    power,domainCheck,vertscale,sin,cos,exp,tan,tand,sinh,cosh,tanh}.m``
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and the
Chebfun Developers.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from chebfunjax.utils.quadrature import chebpts_ab


def _is_scalar(value) -> bool:
    return jnp.ndim(value) == 0


def _is_chebfun3t(value, template) -> bool:
    """Recognize this concrete Chebfun3T class, not any coeff-bearing object."""
    return isinstance(value, type(template))


def _is_empty_operand(value, template) -> bool:
    if _is_chebfun3t(value, template):
        return value.isempty()
    if value is None:
        return True
    if isinstance(value, str) and not value:
        return True
    try:
        return jnp.asarray(value).size == 0
    except (TypeError, ValueError):
        return False


class Chebfun3TArithmeticError(ValueError):
    """Operation error retaining the MATLAB identifier when one exists.

    Provenance
    ----------
    MATLAB source : ``@chebfun3t/{times,rdivide,power,mtimes}.m``
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and the
        Chebfun Developers.
    """

    def __init__(self, message: str, identifier: str | None = None):
        super().__init__(message)
        self.identifier = identifier


def _pad_coeffs(coeffs, shape):
    pads = tuple((0, target - actual)
                 for actual, target in zip(coeffs.shape, shape))
    return jnp.pad(coeffs, pads)


def _native_domain_check(f, g) -> bool:
    """MATLAB ``domainCheck`` for nonempty dense objects."""
    scale = max(max(abs(v) for v in f.domain),
                max(abs(v) for v in g.domain))
    return all(abs(a - b) < 1e-15 * scale
               for a, b in zip(f.domain, g.domain))


def _vertscale(f) -> float:
    """Reproduce ``@chebfun3t/vertscale``'s sampled maximum."""
    lengths = f.coeffs.shape
    counts = tuple(min(max(n, 9), 41) for n in lengths)
    axes = tuple(
        chebpts_ab(n, f.domain[2 * k], f.domain[2 * k + 1])
        for k, n in enumerate(counts)
    )
    xx, yy, zz = jnp.meshgrid(*axes, indexing="ij")
    values = f(xx, yy, zz)
    return jnp.max(jnp.abs(values))


def _from_coeffs(f, coeffs, domain, vscale=None):
    return type(f).from_coeffs(coeffs, domain=domain, vscale=vscale)


def _from_callable(f, op: Callable, domain):
    # Import lazily so the helper remains safe to import from chebfun3t.py.
    from chebfunjax.chebfun3d.chebfun3t import chebfun3t
    return chebfun3t(op, domain)


class Chebfun3TArithmeticMixin:
    """Source-shaped arithmetic for a dense ``Chebfun3T`` class.

    The mixin has no numerical storage of its own.  Coefficient-preserving
    methods call the concrete class's ``from_coeffs``; nonlinear operations
    follow MATLAB and reconstruct through the adaptive callable factory.

    Provenance
    ----------
    MATLAB source : ``@chebfun3t/{plus,minus,uminus,mtimes,times,rdivide,
        power,domainCheck,vertscale,sin,cos,exp,tan,tand,sinh,cosh,tanh}.m``
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and the
        Chebfun Developers.
    """

    coeffs: object
    domain: tuple
    vscale: object

    def _empty_result(self):
        return type(self).empty()

    def _binary_empty(self, other):
        return self.isempty() or _is_empty_operand(other, self)

    def __neg__(self):
        """Negate coefficients without changing vertical scale.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/uminus.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        # Native uminus changes coefficients only; vscale is intentionally
        # retained rather than recomputed.
        return _from_coeffs(self, -self.coeffs, self.domain, self.vscale)

    def __add__(self, other):
        """Add a scalar or same-domain Chebfun3T.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/plus.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self._binary_empty(other):
            return self._empty_result()
        if _is_chebfun3t(other, self):
            if tuple(self.domain) != tuple(other.domain):
                raise Chebfun3TArithmeticError(
                    "CHEBFUN3T:plus: Inputs are not defined on the same domain."
                )
            shape = tuple(max(a, b) for a, b in
                          zip(self.coeffs.shape, other.coeffs.shape))
            coeffs = _pad_coeffs(self.coeffs, shape) + _pad_coeffs(
                other.coeffs, shape
            )
            result = _from_coeffs(self, coeffs, self.domain, vscale=0.0)
            return _from_coeffs(result, result.coeffs, result.domain,
                                _vertscale(result))
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "CHEBFUN3T plus supports only scalar doubles."
            )
        dtype = jnp.result_type(self.coeffs.dtype, scalar.dtype)
        coeffs = jnp.asarray(self.coeffs, dtype=dtype)
        coeffs = coeffs.at[0, 0, 0].set(coeffs[0, 0, 0] + scalar)
        return _from_coeffs(self, coeffs, self.domain,
                            jnp.abs(self.vscale + scalar))

    def __radd__(self, other):
        """Support scalar-left addition.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/plus.m`` argument swap
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract by source ``plus(f, uminus(g))`` semantics.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/minus.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self.__add__(
            -other if _is_chebfun3t(other, self) else -jnp.asarray(other)
        )

    def __rsub__(self, other):
        """Support scalar-left subtraction through unary minus and plus.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/minus.m``, ``uminus.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return (-self).__add__(other)

    def __mul__(self, other):
        """Pointwise multiply objects or scale by a scalar.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/times.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self.isempty():
            return self._empty_result()
        if _is_chebfun3t(other, self):
            if not _native_domain_check(self, other):
                raise Chebfun3TArithmeticError(
                    "Inconsistent domains", "CHEBFUN:CHEBFUN3T:times:domain"
                )
            return _from_callable(
                self, lambda x, y, z: self(x, y, z) * other(x, y, z),
                self.domain,
            )
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            if scalar.size == 0:
                raise Chebfun3TArithmeticError(
                    "Sizes are inconsistent.", "CHEBFUN:CHEBFUN3T:mtimes:size"
                )
            raise Chebfun3TArithmeticError(
                "Undefined function 'times' for input arguments of type "
                f"{type(self).__name__} and {type(other).__name__}.",
                "CHEBFUN:CHEBFUN3T:times:unknown",
            )
        return self.__rmul__(scalar)

    def __rmul__(self, other):
        """Scalar multiplication, preserving source coefficient scaling.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/mtimes.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self.isempty():
            return self._empty_result()
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "Sizes are inconsistent.", "CHEBFUN:CHEBFUN3T:mtimes:size"
            )
        if bool(scalar == 0):
            return _from_callable(self, lambda x, y, z: jnp.zeros_like(x + y + z),
                                  self.domain)
        return _from_coeffs(self, self.coeffs * scalar, self.domain,
                            self.vscale * jnp.abs(scalar))

    def __truediv__(self, other):
        """Pointwise right divide with source domain and zero checks.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/rdivide.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self._binary_empty(other):
            return self._empty_result()
        if _is_chebfun3t(other, self):
            if not _native_domain_check(self, other):
                raise Chebfun3TArithmeticError(
                    "Domains inconsistent.",
                    "CHEBFUN:CHEBFUN3T:rdivide:domains",
                )
            return _from_callable(
                self, lambda x, y, z: self(x, y, z) / other(x, y, z),
                self.domain,
            )
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "Unrecognised operation.",
                "CHEBFUN:CHEBFUN3T:rdivide:badInputs",
            )
        if bool(scalar == 0):
            raise Chebfun3TArithmeticError(
                "Division by zero or near zero.",
                "CHEBFUN:CHEBFUN3T:rdivide:divByZero",
            )
        return self * (1 / scalar)

    def __rtruediv__(self, other):
        """Divide a scalar by this function through callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/rdivide.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self.isempty():
            return self._empty_result()
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "Unrecognised operation.",
                "CHEBFUN:CHEBFUN3T:rdivide:badInputs",
            )
        return _from_callable(self, lambda x, y, z: scalar / self(x, y, z),
                              self.domain)

    def __pow__(self, other):
        """Raise this function to a scalar or same-domain function.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/power.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self._binary_empty(other):
            return self._empty_result()
        if _is_chebfun3t(other, self):
            if not _native_domain_check(self, other):
                raise Chebfun3TArithmeticError(
                    "Domains must be the same",
                    "CHEBFUN:CHEBFUN3T:power:domain",
                )
            return _from_callable(
                self, lambda x, y, z: self(x, y, z) ** other(x, y, z),
                self.domain,
            )
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "CHEBFUN3T power supports only scalar doubles."
            )
        return _from_callable(self, lambda x, y, z: self(x, y, z) ** scalar,
                              self.domain)

    def __rpow__(self, other):
        """Raise a scalar base to this function through reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/power.m`` double-left branch
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self.isempty():
            return self._empty_result()
        scalar = jnp.asarray(other)
        if not _is_scalar(scalar):
            raise Chebfun3TArithmeticError(
                "CHEBFUN3T power supports only scalar doubles."
            )
        return _from_callable(self, lambda x, y, z: scalar ** self(x, y, z),
                              self.domain)

    def _compose_unary(self, op: Callable):
        if self.isempty():
            return self._empty_result()
        return _from_callable(self, lambda x, y, z: op(self(x, y, z)),
                              self.domain)

    def sin(self):
        """Compose sine by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/sin.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.sin)

    def cos(self):
        """Compose cosine by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/cos.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.cos)

    def exp(self):
        """Compose exponential by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/exp.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.exp)

    def tan(self):
        """Compose tangent by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/tan.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.tan)

    def tand(self):
        """Compose MATLAB's tangent in degrees over this dense tensor.

        Provenance
        ----------
        MATLAB source : @chebfun3t/tand.m
        Chebfun commit: 7574c77
        The source preserves empty input and resamples nonempty input on its
        current domain using the MATLAB degree tangent.
        """
        from chebfunjax.utils.degree_tangent import degree_tangent

        if self.isempty():
            return self
        return self._compose_unary(degree_tangent)

    def sinh(self):
        """Compose hyperbolic sine by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/sinh.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.sinh)

    def cosh(self):
        """Compose hyperbolic cosine by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/cosh.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.cosh)

    def tanh(self):
        """Compose hyperbolic tangent by adaptive callable reconstruction.

        Provenance
        ----------
        MATLAB source : ``@chebfun3t/tanh.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        return self._compose_unary(jnp.tanh)
