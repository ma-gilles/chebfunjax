"""TreeVar: syntax-tree analysis of ODEs for first-order reformulation.

The TreeVar class lets chebfunjax analyse the syntax trees of ODE
operators.  Evaluating a chebop's ``op`` function with TreeVar arguments
records the mathematical expression as a tree, which is then expanded
(distributing products over sums so the highest-order derivative stands
alone), split into derivative and non-derivative parts, and converted to
a first-order system ``u' = f(t, u)`` suitable for time-stepping IVP
solvers.  MATLAB builds an anonymous function through infix strings and
``eval``; here the tree is compiled to nested closures directly, which
is semantically identical.

Provenance
----------
MATLAB source : @treeVar/treeVar.m, bivariate.m, univariate.m,
    expandTree.m, splitTree.m, toFirstOrder.m, toRHS.m, tree2infix.m,
    sortConditions.m, printTree.m, plotTree.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np  # uses-numpy: tree bookkeeping on concrete ID vectors

from chebfunjax.utils.elementary import (
    _asinh_numeric,
    _conj_numeric,
    _sin_cos_numeric,
    _tan_numeric,
    _tanh_numeric,
    _uminus_numeric,
)


class TreeVarError(ValueError):
    """Error raised during syntax-tree analysis, carrying the MATLAB
    error identifier in ``identifier``."""

    def __init__(self, identifier: str, message: str):
        super().__init__(message)
        self.identifier = identifier


def _tree(method, numArgs, **kw):
    d = {"method": method, "numArgs": numArgs}
    d.update(kw)
    return d


def _is_tree(t) -> bool:
    return isinstance(t, dict) and "method" in t


_UNIVARIATE = [
    "abs", "acos", "acosd", "acot", "acoth", "acsc", "acscd", "acsch",
    "airy", "asec", "asecd", "asech", "asin", "asind", "asinh", "atan",
    "atand", "atanh", "conj", "cos", "cosd", "cosh", "cot", "cotd",
    "coth", "csc", "cscd", "csch", "exp", "expm1", "imag", "log",
    "log10", "log1p", "log2", "pow2", "real", "sec", "secd", "sech",
    "sin", "sind", "sinh", "sqrt", "tan", "tand", "tanh", "uminus",
    "uplus",
]



def _airy(x):
    """Ai with MATLAB's captured eager boundary and output-storage policy.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric AIRY dispatch.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Finite arithmetic reuses the existing JAX airy_all kernel. Nonfinite
    conventions and storage collapse are qualified by R2025b captures;
    opaque builtin internals are not known. In particular, real +Inf for
    imaginary-infinite inputs is a builtin convention, not a complex limit.
    The general airy_all nonfinite contract is unchanged. Eager complex
    storage collapse is not a whole-function JIT or AD guarantee.
    """
    from jax import lax

    from chebfunjax.utils.airy_general import airy_all
    from chebfunjax.utils.numeric_sign import real_sign

    values = jnp.asarray(x)
    ai = airy_all(values)[0]
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        return jnp.where(jnp.isposinf(values), 0., ai)
    real, imag = jnp.real(values), jnp.imag(values)
    on_axis = real_sign(imag) == 0
    finite = jnp.isfinite(real) & jnp.isfinite(imag)
    any_nan = jnp.isnan(real) | jnp.isnan(imag)
    positive_real_inf = jnp.isposinf(real)
    grows = (jnp.isinf(imag) | (jnp.isneginf(real) & ~on_axis))
    grows = grows & ~positive_real_inf & ~any_nan
    decays = positive_real_inf & jnp.isfinite(imag)
    boundary_real = jnp.where(grows, jnp.inf, jnp.where(decays, 0., jnp.nan))
    boundary_imag = jnp.where(positive_real_inf & jnp.isinf(imag), jnp.nan, 0.)
    result_real = jnp.where(finite, jnp.real(ai), boundary_real)
    # Ai is exactly real on the real axis, including signed imaginary zero.
    finite_imag = jnp.where(on_axis, 0., jnp.imag(ai))
    result_imag = jnp.where(finite, finite_imag, boundary_imag)
    if bool(jnp.all(real_sign(result_imag) == 0)):
        return result_real
    return lax.complex(result_real, result_imag)


def _pow2_branch(x):
    """JAX numeric ``2**x`` with MATLAB's captured complex storage edges.

    Real inputs use the JAX base-two primitive. Complex values on the real
    axis return real storage, as MATLAB's numeric POW2 builtin does. Away
    from that axis, evaluate the principal exponential in polar components;
    this preserves the component classes in the R2025b boundary capture.

    This is an eager TreeVar adapter, not a JIT contract. Native source
    ``@chebfun/pow2.m`` delegates to ``power(2,f)``; the raw numeric values
    are from built-in ``pow2`` in the retained TreeVar boundary fixture.
    Provenance: Chebfun commit 7574c77, MATLAB R2025b.
    """
    from jax import lax

    value = jnp.asarray(x)
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        return jnp.exp2(value)
    real, imag = jnp.real(value), jnp.imag(value)
    from chebfunjax.utils.numeric_sign import real_sign

    on_real_axis = real_sign(imag) == 0
    # Native numeric POW2 removes complex storage when the input is exactly
    # real, including a negative-zero imaginary component.
    if bool(jnp.all(on_real_axis)):
        return jnp.exp2(real)
    angle = imag * jnp.log(jnp.asarray(2., dtype=real.dtype))
    scale = jnp.exp2(real)
    real_part = scale * jnp.cos(angle)
    imag_part = scale * jnp.sin(angle)

    # The polar components also give the captured finite/Inf and NaN boundary
    # classes without a complex EXP2 operation's Inf*trig contamination.
    return lax.complex(real_part, imag_part)


def _reciprocal_inverse_trig(x, cosine=False):
    """Compose reciprocal and principal ASIN/ACOS branches for ACSC/ASEC.

    Exact real-axis reciprocals preserve signed zero and its infinite pole.
    The qualified inverse branch assembles real and imaginary components
    separately, avoiding complex 0*Inf contamination at zero inputs.
    Storage collapse is eager; this adapter makes no whole-function JIT claim.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m unary declarations and
        @chebfun/acsc.m, asec.m dispatch to opaque numeric builtins.
    Chebfun commit: 7574c77. The reciprocal composition is mathematical;
    real/complex branches are qualified by the retained R2025b raw36 packet.
    """
    from jax import lax

    values = jnp.asarray(x)
    result = _inverse_trig_branch(_reciprocal_axis_value(values), cosine=cosine)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        real, imag = jnp.real(values), jnp.imag(values)
        pure_imaginary = (real == 0) & (imag != 0) & ~jnp.isnan(imag)
        # asin(-i/y) = -i*asinh(1/y); acos(z) = pi/2-asin(z).
        # The exact imaginary axis has a canonical zero real ACSC component.
        height = jnp.arcsinh(1/imag)
        axis = lax.complex(jnp.full_like(real, jnp.pi/2) if cosine
                           else jnp.zeros_like(real), height if cosine else -height)
        result = jnp.where(pure_imaginary, axis, result)
    if bool(jnp.all(jnp.imag(result) == 0)):
        return jnp.real(result)
    return result


def _inverse_trig_branch(x, cosine=False):
    """Evaluate MATLAB's ASIN/ACOS real-axis branch using JAX primitives.

    Nonzero imaginary inputs use the complex kernel. Exact real-axis inputs,
    including either imaginary signed zero, use the real-domain convention:
    asin(x) = sign(x)*(pi/2 - i*acosh(abs(x))) outside [-1, 1].
    Ordinary in-domain real outputs remain real. Dtype selection is eager.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m ASIN/ACOS numeric dispatch.
    Chebfun commit: 7574c77. MATLAB R2025b primitive branch capture:
    treevar_rhs_jax_review_20261014/unary_boundary_native/capture.json.
    Complex assembly uses separate components, avoiding 0*Inf contamination.
    """
    from jax import lax

    values = jnp.asarray(x)
    real = jnp.real(values)
    on_axis = jnp.imag(values) == 0
    outside = jnp.abs(real) > 1
    op = jnp.arccos if cosine else jnp.arcsin
    if bool(jnp.all(on_axis)) and not bool(jnp.any(outside)):
        return op(real)
    ordinary = op(real)
    height = jnp.arccosh(jnp.abs(real))
    if cosine:
        axis_real = jnp.where(outside, jnp.where(real < 0, jnp.pi, 0.), ordinary)
        axis_imag = jnp.where(outside, jnp.copysign(height, real), 0.)
    else:
        axis_real = jnp.where(outside, jnp.copysign(jnp.pi/2, real), ordinary)
        axis_imag = jnp.where(outside, -jnp.copysign(height, real), 0.)
    axis_value = lax.complex(axis_real, axis_imag)
    if bool(jnp.all(on_axis)):
        return axis_value
    return jnp.where(on_axis, axis_value, op(values))


def _reciprocal_direct_branch(x, method):
    """Evaluate plain reciprocal trig/hyperbolic functions using JAX.

    Real axes use real primitives, including their poles and NaNs, without
    retrying NaNs as complex inputs. Imaginary-axis identities preserve exact
    zero components; off-axis reciprocals use scaled complex division.
    Storage selection is eager and does not extend the stable JIT subset.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m and @chebfun/{cot,csc,sec,coth,csch,sech}.m
    dispatch to opaque numeric builtins. Chebfun commit: 7574c77.
    The reciprocal and imaginary-axis formulas are mathematical identities,
    qualified against the retained R2025b primitive boundary packet.
    """
    from jax import lax

    values = jnp.asarray(x)
    real, imag = jnp.real(values), jnp.imag(values)
    functions = {"cot": jnp.tan, "csc": jnp.sin, "sec": jnp.cos,
                 "coth": jnp.tanh, "csch": jnp.sinh, "sech": jnp.cosh}
    op = functions[method]
    on_real = imag == 0
    real_value = 1/op(real)
    if bool(jnp.all(on_real)):
        return real_value
    result = _reciprocal_axis_value(op(values))
    result = jnp.where(on_real, lax.complex(real_value, jnp.zeros_like(real)), result)
    paired = {"cot": jnp.tanh, "csc": jnp.sinh, "sec": jnp.cosh,
              "coth": jnp.tan, "csch": jnp.sin, "sech": jnp.cos}
    axis_value = 1/paired[method](imag)
    if method in ("sec", "sech"):
        axis = lax.complex(axis_value, jnp.zeros_like(imag))
    else:
        axis = lax.complex(jnp.zeros_like(real), -axis_value)
    on_imag = (real == 0) & (imag != 0) & ~jnp.isnan(imag)
    result = jnp.where(on_imag, axis, result)
    if bool(jnp.all(jnp.imag(result) == 0)):
        return jnp.real(result)
    return result


def _degrees_from_radians(value, *, divide=False):
    """Scale angular components independently for captured degree outputs.

    Separate component scaling is mathematically equivalent for finite angles
    and avoids ``0*Inf`` cross terms of complex multiplication. It matches the
    captured native infinity classifications. Reciprocal inverse-degree
    methods retain their existing division by pi/180 operation order.

    Provenance
    ----------
    MATLAB source: @chebfun/acosd.m, asind.m and atand.m delegate to MATLAB
        builtins; @treeVar/univariate.m records method names, commit 7574c77.
        The componentwise calculation is a mathematical composition.
    R2025b behavior: compact raw36 fixture in the private implementation
        candidate, derived from the existing captured packet.
    """
    value = jnp.asarray(value)
    scale = (lambda y: y / (jnp.pi / 180.0)) if divide else (
        lambda y: y * (180.0 / jnp.pi))
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        from jax import lax
        return lax.complex(scale(jnp.real(value)), scale(jnp.imag(value)))
    return scale(value)


@jax.custom_jvp
def _sinh_complex(values):
    """Complex sinh with the analytic signed zero on the imaginary axis.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m records SINH; @chebfun/sinh.m delegates
    to the opaque numeric builtin. Chebfun commit:
    7574c77680d7e82b79626300bf255498271a72df. Independent R2025b scalar and
    array captures qualify the signs. The correction follows
    Re(sinh(x+iy)) = sinh(x)*cos(y), without changing nonzero arithmetic.
    """
    from jax import lax

    from chebfunjax.utils.numeric_sign import real_sign

    result = jnp.sinh(values)
    real, imag = jnp.real(values), jnp.imag(values)
    on_axis = (real_sign(real) == 0) & jnp.isfinite(imag)
    negative = jnp.signbit(real) ^ jnp.signbit(jnp.cos(imag))
    signed_zero = jnp.copysign(jnp.zeros_like(real),
                               jnp.where(negative, -jnp.ones_like(real),
                                         jnp.ones_like(real)))
    return lax.complex(jnp.where(on_axis, signed_zero, jnp.real(result)),
                       jnp.imag(result))


@_sinh_complex.defjvp
def _sinh_complex_jvp(primals, tangents):
    """Retain sinh's analytic transverse tangent at an imaginary-axis primal."""
    (value,), (tangent,) = primals, tangents
    return _sinh_complex(value), jnp.cosh(value)*tangent


def _sinh_branch(x):
    """Keep real sinh unchanged and correct the complex imaginary-axis sign."""
    values = jnp.asarray(x)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        return _sinh_complex(values)
    return jnp.sinh(values)


@jax.custom_jvp
def _atan_complex(values):
    """Complex atan with native exact-axis conventions and static output dtype.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric ATAN dispatch and @chebfun/atan.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df. Axis branches
    and nonfinite storage are independently captured R2025b builtin behavior;
    the builtin implementation is opaque. Off-axis values retain JAX's kernel.
    """
    from jax import lax

    from chebfunjax.utils.numeric_sign import real_sign

    real, imag = jnp.real(values), jnp.imag(values)
    on_real = real_sign(imag) == 0
    on_imag = (real_sign(real) == 0) & ~jnp.isnan(imag)
    magnitude = jnp.abs(imag)
    # atanh(y) for |y|<1; atanh(1/y) for |y|>1. The quotient forms
    # retain changes at the two representable neighbors of the branch point.
    inside = .5*jnp.log1p(2*magnitude/(1-magnitude))
    outside = .5*jnp.log1p(2/(magnitude-1))
    height = jnp.copysign(jnp.where(magnitude <= 1, inside, outside), imag)
    phase = jnp.where(magnitude <= 1, jnp.zeros_like(real),
                      -jnp.copysign(jnp.full_like(real, jnp.pi/2), imag))
    imaginary_axis = lax.complex(phase, height)
    real_axis = lax.complex(jnp.arctan(real), jnp.zeros_like(imag))
    result = jnp.where(on_imag, imaginary_axis, jnp.arctan(values))
    return jnp.where(on_real, real_axis, result)


@_atan_complex.defjvp
def _atan_complex_jvp(primals, tangents):
    """Retain analytic atan tangents at regular points, including interior axes.

    Exact-axis value selection must not remove transverse imaginary/real
    tangents. No derivative claim is made on branch cuts or singular poles;
    the formal analytic expression there retains JAX's existing convention.
    """
    (value,), (tangent,) = primals, tangents
    return _atan_complex(value), tangent/(1+value*value)


def _atan_branch(x):
    """Use real atan or the dtype-stable complex exact-axis adapter."""
    values = jnp.asarray(x)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        return _atan_complex(values)
    return jnp.arctan(values)


def _atand_radians(x):
    """Reuse the qualified atan branch before independent component scaling.

    Provenance
    ----------
    MATLAB source: @chebfun/atand.m and @treeVar/treeVar.m dispatch to the
    opaque numeric builtin, Chebfun commit7574c77680d7e82b79626300bf255498271a72df.
    R2025b independent ATAN/ATAND cross controls establish shared axis values.
    Traced output dtype remains static; eager exact-real output may collapse.
    """
    return _eager_unary_storage(_atan_branch)(x)


def _acot_branch(x):
    """Eager reciprocal-atan composition matching captured acot outputs.

    At pure-imaginary inputs +/-1j, the captured output canonicalizes the real
    zero to positive zero. This value-dependent adapter makes no JIT promise.

    Provenance
    ----------
    MATLAB source: @chebfun/acot.m delegates to the opaque MATLAB builtin;
        @treeVar/univariate.m records its name, commit 7574c77. The reciprocal
        atan identity is the implementation composition; tested outputs are
        from the existing R2025b raw36 packet.
    """
    from jax import lax

    values = jnp.asarray(x)
    result = jnp.arctan(1.0 / values)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        real, imag = jnp.real(values), jnp.imag(values)
        exact_pole = (real == 0) & (jnp.abs(imag) == 1)
        corrected = lax.complex(jnp.zeros_like(real), jnp.imag(result))
        result = jnp.where(exact_pole, corrected, result)
        if bool(jnp.all(jnp.imag(result) == 0)):
            return jnp.real(result)
    return result


def _atanh_branch(x):
    """Evaluate ATANH with MATLAB's exact real-axis branch using JAX.

    Outside [-1, 1], the real-axis identity is atanh(1/x) plus
    i*sign(x)*pi/2. Nonzero imaginary inputs use the complex kernel;
    the pure-imaginary identity is atanh(i*y) = i*atan(y).
    Output dtype selection remains eager and value dependent.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m ATANH numeric dispatch.
    Chebfun commit: 7574c77. MATLAB R2025b primitive branch capture:
    treevar_rhs_jax_review_20261014/unary_boundary_native/capture.json.
    """
    from jax import lax

    def real_atanh(value):
        # atanh(t) = sign(t)*log1p(2*abs(t)/(1-abs(t)))/2.
        # This avoids subtracting two log1p evaluations and retains tiny t.
        magnitude = jnp.abs(value)
        return jnp.copysign(.5*jnp.log1p(2*magnitude/(1-magnitude)), value)

    values = jnp.asarray(x)
    real, imag = jnp.real(values), jnp.imag(values)
    on_axis = imag == 0
    outside = jnp.abs(real) > 1
    if bool(jnp.all(on_axis)) and not bool(jnp.any(outside)):
        return real_atanh(real)
    axis_real = real_atanh(jnp.where(outside, 1/real, real))
    axis_imag = jnp.where(outside, jnp.copysign(jnp.pi/2, real), 0.)
    axis_value = lax.complex(axis_real, axis_imag)
    if bool(jnp.all(on_axis)):
        return axis_value
    ordinary = jnp.arctanh(values)
    imaginary_value = lax.complex(jnp.zeros_like(real), jnp.arctan(imag))
    ordinary = jnp.where(real == 0, imaginary_value, ordinary)
    return jnp.where(on_axis, axis_value, ordinary)


def _sqrt_branch(x):
    """Evaluate SQRT with MATLAB's nonnegative exact real-axis convention.

    Either imaginary signed zero selects the positive imaginary square root
    on the negative real axis. Elsewhere the principal complex kernel applies.
    Zero is nonnegative, and an entirely zero imaginary result is stored real.
    Dtype selection remains eager and value dependent.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m SQRT numeric dispatch.
    Chebfun commit: 7574c77. MATLAB R2025b primitive branch capture:
    treevar_rhs_jax_review_20261014/unary_boundary_native/capture.json.
    """
    from jax import lax

    values = jnp.asarray(x)
    real = jnp.real(values)
    on_axis = jnp.imag(values) == 0
    negative = real < 0
    nonnegative_zero = jnp.where(real == 0, jnp.zeros_like(real), real)
    if bool(jnp.all(on_axis)) and not bool(jnp.any(negative)):
        return jnp.sqrt(nonnegative_zero)
    axis_real = jnp.sqrt(jnp.where(negative, 0., nonnegative_zero))
    axis_imag = jnp.sqrt(jnp.where(negative, -real, 0.))
    axis_value = lax.complex(axis_real, axis_imag)
    if bool(jnp.all(on_axis)):
        return axis_value
    result = jnp.where(on_axis, axis_value, jnp.sqrt(values))
    if bool(jnp.all(jnp.imag(result) == 0)):
        return jnp.real(result)
    return result


def _log1p_real(value):
    """Compensated log(1+x), retaining x when the rounded sum equals one.

    The ratio x/(u-1), u=round(1+x), compensates the rounded addition.
    Endpoints are handled without a tolerance or fitted switching constant.
    """
    rounded = 1 + value
    result = jnp.log(rounded) * (value/(rounded-1))
    result = jnp.where(rounded == 1, value, result)
    return jnp.where(jnp.isposinf(value), value, result)


def _complex_log_magnitude(real, imag):
    """Stable log-hypotenuse, retaining small changes near the unit circle."""
    largest = jnp.maximum(jnp.abs(real), jnp.abs(imag))
    smallest = jnp.minimum(jnp.abs(real), jnp.abs(imag))
    correction = (largest-1)*(largest+1) + smallest*smallest
    near_unit = .5*_log1p_real(correction)
    scaled = jnp.log(largest) + .5*_log1p_real((smallest/largest)**2)
    result = jnp.where((largest >= .5) & (largest <= 1.5), near_unit, scaled)
    result = jnp.where(jnp.isinf(largest), jnp.inf, result)
    # Native's pure imaginary unit-axis logarithm stores a negative real zero.
    # Real-axis log(1) is handled separately and retains positive zero.
    return jnp.where((real == 0) & (jnp.abs(imag) == 1),
                     jnp.copysign(jnp.zeros_like(result), -1.), result)


def _log_branch(x, *, base=None, increment=False):
    """MATLAB logarithm branches with stable complex magnitude evaluation.

    The exact negative real axis uses +pi for either imaginary signed zero;
    off-axis values retain their principal phase. Real results use real
    storage. LOG1P preserves its original real input in the near-zero
    magnitude identity x*(x+2)+y*y, avoiding rounded formation of 1+x.
    Value-dependent storage selection is eager, as for the other adapters.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric LOG/LOG10/LOG1P/LOG2 dispatch.
    Chebfun commit: 7574c77. Branch and signed-zero conventions qualified by
    the retained R2025b primitive boundary capture. Magnitude identities are
    algebraic log(hypot) and log1p identities, not a port of opaque built-ins.
    """
    from jax import lax

    values = jnp.asarray(x)
    real, imag = jnp.real(values), jnp.imag(values)
    shifted = real + 1 if increment else real
    on_axis = imag == 0
    negative = shifted < 0
    if increment:
        axis_magnitude = jnp.where(negative, jnp.log(-shifted), _log1p_real(real))
    else:
        axis_magnitude = jnp.log(jnp.abs(real))
    if bool(jnp.all(on_axis)) and not bool(jnp.any(negative)):
        if base == 2 and not increment:
            return jnp.log2(jnp.abs(real))
        if base == 10 and not increment:
            return jnp.log10(jnp.abs(real))
        return axis_magnitude
    magnitude = _complex_log_magnitude(shifted, imag)
    if increment:
        near_origin = .5*_log1p_real(real*(real+2)+imag*imag)
        magnitude = jnp.where((jnp.abs(real) <= .5) & (jnp.abs(imag) <= .5),
                              near_origin, magnitude)
    magnitude = jnp.where(on_axis, axis_magnitude, magnitude)
    phase = jnp.where(on_axis, jnp.where(negative, jnp.pi, 0.),
                      jnp.arctan2(imag, shifted))
    if base is not None:
        # Polar base-log identity: round the modulus before applying its
        # real base logarithm. This differs from scaling complex natural log
        # near the unit circle; native LOG2/LOG10 exhibit that distinction.
        modulus = jnp.hypot(shifted, imag)
        magnitude = jnp.log2(modulus) if base == 2 else jnp.log10(modulus)
        denominator = jnp.log(jnp.asarray(base, dtype=real.dtype))
        phase = phase/denominator
    if bool(jnp.all(phase == 0)):
        return magnitude
    return lax.complex(magnitude, phase)


def _reciprocal_axis_value(x):
    """Scaled complex reciprocal with a separate exact-real-axis operation."""
    from jax import lax

    values = jnp.asarray(x)
    real, imag = jnp.real(values), jnp.imag(values)
    on_axis = imag == 0
    if bool(jnp.all(on_axis)):
        return 1/real
    scale = jnp.maximum(jnp.abs(real), jnp.abs(imag))
    r, i = real/scale, imag/scale
    # Infinite complex arguments have a zero reciprocal; avoid Inf/Inf.
    r = jnp.where(jnp.isinf(scale), jnp.where(jnp.isinf(real),
                  jnp.copysign(1., real), jnp.copysign(0., real)), r)
    i = jnp.where(jnp.isinf(scale), jnp.where(jnp.isinf(imag),
                  jnp.copysign(1., imag), jnp.copysign(0., imag)), i)
    denominator = r*r+i*i
    ordinary = lax.complex((r/denominator)/scale, (-i/denominator)/scale)
    axis = lax.complex(1/real, jnp.zeros_like(real))
    return jnp.where(on_axis, axis, ordinary)


def _reciprocal_hyperbolic_branch(x, method):
    """Reciprocal inverse hyperbolic functions with native axis branches.

    ACOTH composes the qualified ATANH branch with a reciprocal. ACSCH uses
    real ASINH on-axis and -i*ASIN(i/z) elsewhere. ASECH uses ACOSH(1/z),
    with explicit real-axis pieces and complex ACOSH off-axis. Native's
    ASECH(-0)=NaN is retained as a captured endpoint convention.
    Storage selection is eager; mathematical compositions do not assert
    knowledge of the opaque MATLAB built-in implementations.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m ACSCH/ASECH/ACOTH numeric dispatch.
    Chebfun commit: 7574c77. Native R2025b boundary capture in
    treevar_rhs_jax_review_20261014/unary_boundary_native/capture.json.
    """
    from jax import lax

    inverse = _reciprocal_axis_value(x)
    real, imag = jnp.real(inverse), jnp.imag(inverse)
    on_axis = imag == 0
    if method == 'acoth':
        result = _atanh_branch(inverse)
    elif method == 'acsch':
        axis = jnp.arcsinh(real)
        if bool(jnp.all(on_axis)):
            return axis
        rotated = lax.complex(-imag, real)
        asin = _inverse_trig_branch(rotated)
        ordinary = lax.complex(jnp.imag(asin), -jnp.real(asin))
        result = jnp.where(on_axis, lax.complex(axis, jnp.zeros_like(real)), ordinary)
    else:
        outside = jnp.abs(real) > 1
        height = jnp.where(outside, jnp.arccosh(jnp.abs(real)), 0.)
        phase = jnp.where(outside, jnp.where(real < 0, jnp.pi, 0.),
                          jnp.arccos(real))
        # Propagate real NaN, plus the native negative-zero endpoint failure.
        height = jnp.where(jnp.isnan(real) | jnp.isneginf(real), jnp.nan, height)
        phase = jnp.where(jnp.isnan(real) | jnp.isneginf(real), 0., phase)
        axis = lax.complex(height, phase)
        ordinary = jnp.arccosh(lax.complex(real, imag))
        result = jnp.where(on_axis, axis, ordinary)
    if bool(jnp.all(jnp.imag(result) == 0)):
        return jnp.real(result)
    return result


def _degree_sincos(real):
    """Reduce finite degree arguments before evaluating bounded radians."""
    angle = jnp.fmod(jnp.abs(real), 360.)
    quadrant = jnp.floor(angle/90. + .5)
    residual = angle - 90.*quadrant
    radians = residual * (jnp.pi/180.)
    sine, cosine = jnp.sin(radians), jnp.cos(radians)
    q = jnp.fmod(quadrant, 4.)
    s = jnp.where(q == 0, sine, jnp.where(q == 1, cosine,
                  jnp.where(q == 2, -sine, -cosine)))
    c = jnp.where(q == 0, cosine, jnp.where(q == 1, -sine,
                  jnp.where(q == 2, -cosine, sine)))
    # Exact quadrant zeros are canonical before the sine's odd symmetry.
    s = jnp.where(s == 0, 0., s)
    c = jnp.where(c == 0, 0., c)
    return jnp.copysign(1., real)*s, c


def _direct_degree_branch(x, method):
    """Degree trigonometric functions with reduced real angles.

    Periodicity and quadrant identities reduce the real part to at most
    45 degrees before conversion to radians. Complex components use the
    trigonometric/hyperbolic identities, avoiding complex angle products
    that contaminate finite components at infinite imaginary arguments.
    TAND/COTD reuse the qualified degree_tangent kernel, including captured
    exact complex-pole behavior. Storage collapse is eager and adds no
    whole-function JIT promise.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric degree-function dispatch.
    Chebfun commit: 7574c77. Reduction and complex identities are mathematical
    implementations, not asserted translations of opaque MATLAB builtins.
    Native boundary and quadrant captures qualify output conventions.
    """
    from jax import lax

    from chebfunjax.utils.numeric_sign import real_sign

    values = jnp.asarray(x)
    values = values.astype(jnp.result_type(values, 1.0))
    real, imag = jnp.real(values), jnp.imag(values)
    if method in ('tand', 'cotd'):
        from chebfunjax.utils.degree_tangent import degree_tangent

        if bool(jnp.all(real_sign(imag) == 0)):
            result = degree_tangent(real).astype(real.dtype)
            return result if method == 'tand' else 1/result
        result = degree_tangent(values).astype(values.dtype)
        if method == 'cotd':
            result = _reciprocal_axis_value(result)
        # Native's nonreal TAN imaginary-axis limit has a positive real zero.
        result = lax.complex(jnp.where((real_sign(imag) != 0) & (real_sign(jnp.real(result)) == 0),
                                      0., jnp.real(result)), jnp.imag(result))
        if bool(jnp.all(real_sign(jnp.imag(result)) == 0)):
            return jnp.real(result)
        return result
    sine, cosine = _degree_sincos(real)
    on_axis = real_sign(imag) == 0
    if method in ('sind', 'cscd'):
        axis = sine if method == 'sind' else 1/sine
    elif method in ('cosd', 'secd'):
        axis = cosine if method == 'cosd' else 1/cosine
    else:
        axis = cosine/sine
    if bool(jnp.all(on_axis)):
        return axis
    b = imag * (jnp.pi/180.)
    # At an exact quadrant, a component is identically zero for every
    # finite imaginary argument. Preserve that limiting zero at infinity
    # instead of forming 0*Inf. Nonzero factors retain their ordinary signs.
    sinh, cosh = jnp.sinh(b), jnp.cosh(b)
    if method in ('sind', 'cscd'):
        r = jnp.where(sine == 0, sine, sine*cosh)
        i = jnp.where(cosine == 0, 0., cosine*sinh)
    else:
        r = jnp.where(cosine == 0, cosine, cosine*cosh)
        i = jnp.where(sine == 0, 0., -sine*sinh)
    result = lax.complex(r, i)
    if method in ('cscd', 'secd'):
        result = _reciprocal_axis_value(result)
        result = jnp.where(real == 0, lax.complex(jnp.zeros_like(real),
                           jnp.imag(result)), result) if method == 'cscd' else result
    result = jnp.where(on_axis, lax.complex(axis, jnp.zeros_like(real)), result)
    if bool(jnp.all(real_sign(jnp.imag(result)) == 0)):
        return jnp.real(result)
    return result


_UNARY_FNS_RAW = {
    "abs": jnp.abs, "acos": lambda x: _inverse_trig_branch(x, cosine=True),
    "acosd": lambda x: _degrees_from_radians(_inverse_trig_branch(x, cosine=True)),
    "acot": _acot_branch,
    "acoth": lambda x: _reciprocal_hyperbolic_branch(x, "acoth"),
    "acsc": _reciprocal_inverse_trig,
    "acscd": lambda x: _degrees_from_radians(_reciprocal_inverse_trig(x), divide=True),
    "acsch": lambda x: _reciprocal_hyperbolic_branch(x, "acsch"),
    "airy": _airy,
    "asec": lambda x: _reciprocal_inverse_trig(x, cosine=True),
    "asecd": lambda x: _degrees_from_radians(_reciprocal_inverse_trig(x, cosine=True), divide=True),
    "asech": lambda x: _reciprocal_hyperbolic_branch(x, "asech"),
    "asin": _inverse_trig_branch,
    "asind": lambda x: _degrees_from_radians(_inverse_trig_branch(x)),
    "asinh": _asinh_numeric, "atan": _atan_branch,
    "atand": lambda x: _degrees_from_radians(_atand_radians(x)),
    "atanh": _atanh_branch, "conj": _conj_numeric,
    "cos": lambda x: _sin_cos_numeric(x, cosine=True),
    "cosd": lambda x: _direct_degree_branch(x, "cosd"), "cosh": jnp.cosh,
    "cot": lambda x: _reciprocal_direct_branch(x, "cot"),
    "cotd": lambda x: _direct_degree_branch(x, "cotd"),
    "coth": lambda x: _reciprocal_direct_branch(x, "coth"),
    "csc": lambda x: _reciprocal_direct_branch(x, "csc"),
    "cscd": lambda x: _direct_degree_branch(x, "cscd"),
    "csch": lambda x: _reciprocal_direct_branch(x, "csch"),
    "exp": jnp.exp, "expm1": jnp.expm1, "imag": jnp.imag,
    "log": _log_branch, "log10": lambda x: _log_branch(x, base=10),
    "log1p": lambda x: _log_branch(x, increment=True),
    "log2": lambda x: _log_branch(x, base=2), "pow2": _pow2_branch, "real": jnp.real,
    "sec": lambda x: _reciprocal_direct_branch(x, "sec"),
    "secd": lambda x: _direct_degree_branch(x, "secd"),
    "sech": lambda x: _reciprocal_direct_branch(x, "sech"),
    "sin": _sin_cos_numeric, "sind": lambda x: _direct_degree_branch(x, "sind"),
    "sinh": _sinh_branch, "sqrt": _sqrt_branch, "tan": _tan_numeric,
    "tand": lambda x: _direct_degree_branch(x, "tand"), "tanh": _tanh_numeric,
    "uminus": _uminus_numeric, "uplus": lambda x: +x,
}


# Branch-specific adapters retain their own storage policy. The direct
# primitives below normalize storage only for concrete eager outputs. This
# does not imply whole-function JIT or AD support for every unary method.
def _eager_unary_storage(fn):
    """Preserve native exact-real eager output storage without changing tracing.

    Provenance
    ----------
    MATLAB source: @treeVar/treeVar.m numeric unary dispatch, Chebfun commit
    7574c77680d7e82b79626300bf255498271a72df. R2025b independent scalar and
    array captures establish whole-array real storage when every output
    imaginary component is exactly zero. Builtin arithmetic is opaque.
    Traced outputs retain their static JAX dtype, including complex tangents
    at real-axis primals; eager and traced output dtypes can therefore differ.
    """
    def evaluate(x):
        result = fn(x)
        if isinstance(result, jax.core.Tracer) or not jnp.iscomplexobj(result):
            return result
        from chebfunjax.utils.numeric_sign import real_sign

        if bool(jnp.all(real_sign(jnp.imag(result)) == 0)):
            return jnp.real(result)
        return result

    return evaluate


_EAGER_STORAGE_UNARY = frozenset(
    "asinh atan conj cos cosh exp expm1 sin sinh tan tanh uminus uplus".split()
)
_UNARY_FNS = {
    name: _eager_unary_storage(fn) if name in _EAGER_STORAGE_UNARY else fn
    for name, fn in _UNARY_FNS_RAW.items()
}


_BINARY_FNS = {
    "plus": lambda a, b: a + b,
    "minus": lambda a, b: a - b,
    "times": lambda a, b: a * b,
    "rdivide": lambda a, b: a / b,
    "power": lambda a, b: a ** b,
}


class TreeVar:
    """A variable recording the syntax tree of ODE expressions.

    Parameters
    ----------
    id_vec : sequence of bool/int, default (1,)
        Which base variable this TreeVar represents.
    domain : tuple of float, default (-1, 1)
        Problem domain.

    Provenance
    ----------
    MATLAB source : @treeVar/treeVar.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """

    def __init__(self, id_vec=(1,), domain=(-1.0, 1.0)):
        self.domain = tuple(float(v) for v in domain)
        idv = np.asarray(id_vec, dtype=bool).ravel()
        self.tree = _tree("constr", 0, diffOrder=np.zeros(idv.size),
                          height=0, ID=idv, hasTerms=False)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def univariate(tree_in, method):
        """Syntax tree for a univariate operation.

        Provenance
        ----------
        MATLAB source : @treeVar/univariate.m
        Chebfun commit: 7574c77
        """
        return _tree(method, 1, center=tree_in,
                     diffOrder=tree_in["diffOrder"],
                     height=tree_in["height"] + 1,
                     ID=tree_in["ID"], hasTerms=tree_in["hasTerms"])

    @staticmethod
    def bivariate(left, right, method, type_):
        """Syntax tree for a bivariate operation.  ``type_`` is 0 when
        only the left argument is a TreeVar tree, 1 when only the right
        is, and 2 when both are.

        Provenance
        ----------
        MATLAB source : @treeVar/bivariate.m
        Chebfun commit: 7574c77
        """
        is_pm = method in ("plus", "minus")
        if type_ == 2:
            return _tree(method, 2, left=left, right=right,
                         diffOrder=np.maximum(left["diffOrder"],
                                              right["diffOrder"]),
                         ID=left["ID"] | right["ID"],
                         height=max(left["height"],
                                    right["height"]) + 1,
                         hasTerms=is_pm or left["hasTerms"]
                         or right["hasTerms"])
        if type_ == 1:
            return _tree(method, 2, left=left, right=right,
                         diffOrder=right["diffOrder"],
                         height=right["height"] + 1, ID=right["ID"],
                         hasTerms=is_pm or right["hasTerms"])
        return _tree(method, 2, left=left, right=right,
                     diffOrder=left["diffOrder"],
                     height=left["height"] + 1, ID=left["ID"],
                     hasTerms=is_pm or left["hasTerms"])

    def _wrap(self, tree):
        out = TreeVar.__new__(TreeVar)
        out.domain = self.domain
        out.tree = tree
        return out

    def _update_domain(self, other):
        if isinstance(other, TreeVar):
            merged = sorted(set(self.domain) | set(other.domain))
            return tuple(merged)
        odom = getattr(other, "domain", None)
        if odom is not None:
            bps = (odom.breakpoints if hasattr(odom, "breakpoints")
                   else odom)
            merged = sorted(set(self.domain)
                            | {float(v) for v in bps})
            return tuple(merged)
        return self.domain

    def _bin(self, other, method, reflected=False):
        if isinstance(other, TreeVar):
            t = TreeVar.bivariate(self.tree, other.tree, method, 2)
        elif reflected:
            t = TreeVar.bivariate(other, self.tree, method, 1)
        else:
            t = TreeVar.bivariate(self.tree, other, method, 0)
        out = self._wrap(t)
        out.domain = self._update_domain(other)
        return out

    # -- arithmetic ----------------------------------------------------

    def __add__(self, other):
        return self._bin(other, "plus")

    def __radd__(self, other):
        return self._bin(other, "plus", reflected=True)

    def __sub__(self, other):
        return self._bin(other, "minus")

    def __rsub__(self, other):
        return self._bin(other, "minus", reflected=True)

    def __mul__(self, other):
        return self._bin(other, "times")

    def __rmul__(self, other):
        return self._bin(other, "times", reflected=True)

    def __truediv__(self, other):
        return self._bin(other, "rdivide")

    def __rtruediv__(self, other):
        return self._bin(other, "rdivide", reflected=True)

    def __pow__(self, other):
        return self._bin(other, "power")

    def __rpow__(self, other):
        return self._bin(other, "power", reflected=True)

    def __neg__(self):
        return self._wrap(TreeVar.univariate(self.tree, "uminus"))

    def __pos__(self):
        return self._wrap(TreeVar.univariate(self.tree, "uplus"))

    def diff(self, k: int = 1):
        """Derivative node.  Only bare base variables may be
        differentiated (``diff(-u)`` etc. is unsupported, matching
        MATLAB's first-order reformulation).

        Provenance
        ----------
        MATLAB source : @treeVar/treeVar.m (diff)
        Chebfun commit: 7574c77
        """
        if not (self.tree["method"] == "constr"
                and self.tree["height"] == 0):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:diff:diffArguments",
                "For first order formulation, the diff method does "
                "currently not support arguments other than simply "
                "the unknown functions.")
        t = _tree("diff", 2, left=self.tree, right=int(k),
                  diffOrder=self.tree["diffOrder"]
                  + k * self.tree["ID"],
                  height=self.tree["height"] + 1,
                  ID=self.tree["ID"],
                  hasTerms=self.tree["hasTerms"])
        return self._wrap(t)

    def cumsum(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:cumsum:notSupported",
            "First order reformulation does not support integral "
            "equations.")

    def sum(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:cumsum:notSupported",
            "First order reformulation does not support integral "
            "equations.")

    def __call__(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:SUBSREF:notSupported",
            "t() is not supported in treeVar.")

    def print(self, *var_names):
        """Text rendering of the syntax tree (MATLAB ``print``)."""
        return print_tree(self.tree, list(var_names) or None)

    def plot(self, ax=None):
        """Plot the syntax tree (MATLAB ``plot``)."""
        return plot_tree(self.tree, ax=ax)

    def __repr__(self):
        return ("treeVar with tree:\n" + print_tree(self.tree)
                + f"and the domain:\n    {self.domain}")


def _add_univariate_methods():
    def make(name):
        def method(self):
            return self._wrap(TreeVar.univariate(self.tree, name))
        method.__name__ = name
        method.__doc__ = (f"Record ``{name}`` in the syntax tree "
                          "(MATLAB @treeVar univariate method).")
        return method
    for name in _UNIVARIATE:
        if name in ("uminus", "uplus"):
            continue
        setattr(TreeVar, name, make(name))


_add_univariate_methods()


# ----------------------------------------------------------------------
# Tree analysis
# ----------------------------------------------------------------------

def _get(tree, field, default=0):
    if _is_tree(tree):
        return tree[field]
    return default


def expand_tree(tree, max_order):
    """Distribute products over sums so the highest-order derivative
    stands alone (``5*(diff(u)+u)`` -> ``5*diff(u) + 5*u``).

    Provenance
    ----------
    MATLAB source : @treeVar/expandTree.m
    Chebfun commit: 7574c77
    """
    max_order = np.asarray(max_order, dtype=float)
    if not _is_tree(tree):
        return tree
    tdo = np.broadcast_to(
        np.atleast_1d(np.asarray(tree["diffOrder"], dtype=float)),
        max_order.shape)
    if (tree["height"] <= 1 or np.all(tdo < max_order)
            or (not tree["hasTerms"] and np.sum(tree["ID"]) <= 1)):
        return tree
    if tree["numArgs"] == 1:
        return expand_tree(tree["center"], max_order)
    if tree["method"] in ("plus", "minus"):
        out = dict(tree)
        out["left"] = expand_tree(tree["left"], max_order)
        out["right"] = expand_tree(tree["right"], max_order)
        return out

    # Must be at .*, ./ or .^.
    if (_is_tree(tree["left"]) and _is_tree(tree["right"])
            and np.any(tdo == max_order)):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:expandTree:nonlinearity",
            "Nonlinearity in highest order derivative detected. "
            "Unable to convert to first order format.")

    left, right = tree["left"], tree["right"]
    left_args = left["numArgs"] if _is_tree(left) else 0
    right_args = right["numArgs"] if _is_tree(right) else 0

    split_left = split_right = False
    if left_args == 0 or (_is_tree(left) and left["method"] == "diff"
                          and left["height"] <= 1):
        left_tree = left
    elif left_args == 1:
        left_tree = expand_tree(left, max_order)
    else:
        left_left = expand_tree(left["left"], max_order)
        left_right = expand_tree(left["right"], max_order)
        split_left = True

    if right_args == 0 or (_is_tree(right)
                           and right["method"] == "diff"
                           and right["height"] <= 1):
        right_tree = right
    elif right_args == 1:
        right_tree = expand_tree(right, max_order)
    else:
        right_left = expand_tree(right["left"], max_order)
        right_right = expand_tree(right["right"], max_order)
        split_right = True

    def times(a, b):
        return _tree(
            "times", 2, left=a, right=b,
            diffOrder=np.maximum(
                np.asarray(_get(a, "diffOrder", 0), dtype=float),
                np.asarray(_get(b, "diffOrder", 0), dtype=float)),
            height=max(_get(a, "height", 0), _get(b, "height", 0)) + 1,
            ID=(np.asarray(_get(a, "ID", False))
                | np.asarray(_get(b, "ID", False)))
            if (_is_tree(a) or _is_tree(b)) else np.asarray([False]),
            hasTerms=bool(_get(a, "hasTerms", 0))
            or bool(_get(b, "hasTerms", 0)))

    if not split_left and not split_right:
        new_left, new_right = left_tree, right_tree
    elif split_left:
        new_left = times(left_left, right_tree)
        new_right = times(left_right, right_tree)
    else:
        new_left = times(left_tree, right_left)
        new_right = times(left_tree, right_right)

    if _is_tree(new_left) and new_left.get("hasTerms"):
        new_left = expand_tree(new_left, max_order)
    if _is_tree(new_right) and new_right.get("hasTerms"):
        new_right = expand_tree(new_right, max_order)

    return _tree(
        "plus", 2, left=new_left, right=new_right,
        diffOrder=np.maximum(
            np.asarray(_get(new_left, "diffOrder", 0), dtype=float),
            np.asarray(_get(new_right, "diffOrder", 0), dtype=float)),
        height=max(_get(new_left, "height", 0),
                   _get(new_right, "height", 0)) + 1,
        ID=(np.asarray(_get(new_left, "ID", False))
            | np.asarray(_get(new_right, "ID", False))),
        hasTerms=bool(_get(new_left, "hasTerms", 0))
        or bool(_get(new_right, "hasTerms", 0)))


def split_tree(tree, max_order):
    """Split a tree into (non-derivative part, derivative part).

    Provenance
    ----------
    MATLAB source : @treeVar/splitTree.m
    Chebfun commit: 7574c77
    """
    max_order = np.asarray(max_order, dtype=float)
    diff_var = max_order > 0
    if not _is_tree(tree):
        return tree, None
    tdo = np.broadcast_to(
        np.atleast_1d(np.asarray(tree["diffOrder"], dtype=float)),
        max_order.shape)
    if np.all(tdo[diff_var] < max_order[diff_var]):
        return tree, None

    if tree["numArgs"] == 1:
        if tree["method"] == "uminus":
            temp = _tree("times", 2, left=-1.0, right=tree["center"],
                         diffOrder=tree["diffOrder"],
                         height=tree["height"], ID=tree["ID"],
                         hasTerms=tree["hasTerms"])
            return split_tree(temp, max_order)
        return split_tree(tree["center"], max_order)

    if tree["method"] in ("diff", "times", "rdivide"):
        return None, tree

    new_l, der_l = split_tree(tree["left"], max_order)
    new_r, der_r = split_tree(tree["right"], max_order)

    def from_right(t, op):
        if op != "minus":
            return t
        if _is_tree(t):
            return _tree("uminus", 1, center=t,
                         diffOrder=t["diffOrder"], ID=t["ID"],
                         height=t["height"] + 1)
        return -t

    if new_l is None:
        new_tree = from_right(new_r, tree["method"])
    elif new_r is None:
        new_tree = new_l
    elif not _is_tree(new_l) and not _is_tree(new_r):
        new_tree = _BINARY_FNS[tree["method"]](new_l, new_r)
    else:
        if not _is_tree(new_l):
            ndo, nh = new_r["diffOrder"], new_r["height"]
        elif not _is_tree(new_r):
            ndo, nh = new_l["diffOrder"], new_l["height"]
        else:
            ndo = np.maximum(new_l["diffOrder"], new_r["diffOrder"])
            nh = max(new_l["height"], new_r["height"])
        new_tree = _tree(tree["method"], 2, left=new_l, right=new_r,
                         diffOrder=ndo, height=nh)

    if der_l is None:
        der_tree = from_right(der_r, tree["method"])
    elif der_r is None:
        der_tree = der_l
    else:
        der_tree = _tree(tree["method"], 2, left=der_l, right=der_r,
                         diffOrder=np.maximum(der_l["diffOrder"],
                                              der_r["diffOrder"]),
                         height=max(der_l["height"], der_r["height"]))
    return new_tree, der_tree


def _leaf_eval(leaf, t):
    """Evaluate a scalar numeric or callable leaf without discarding its dtype.

    Numeric leaves remain constants; callable leaves retain their scalar
    JAX value, including complex values and dependence on traced time.

    Provenance
    ----------
    MATLAB source: @treeVar/tree2infix.m, @treeVar/toRHS.m.
    Chebfun commit: 7574c77.
    """
    if not callable(leaf):
        return leaf
    return jnp.asarray(leaf(jnp.asarray(t))).reshape(())


def _compile(tree, index_start, is_coeff):
    """Compile a syntax tree to a closure ``f(t, u)``.

    This replaces MATLAB's tree2infix + toAnon string round-trip.

    Provenance
    ----------
    MATLAB source : @treeVar/tree2infix.m, @treeVar/toAnon.m
    Chebfun commit: 7574c77
    """
    if tree is None:
        return lambda t, u: 0.0
    if not _is_tree(tree):
        leaf = tree
        if is_coeff:
            # Native tree2infix composes Chebfun leaves with the independent
            # Chebfun; scalar leaves stay scalar. Do not classify a function
            # as constant from a fixed set of point evaluations.
            from chebfunjax.chebfun1d.chebfun import Chebfun
            if isinstance(leaf, Chebfun):
                return lambda t, u: t.compose(leaf)
            return lambda t, u: leaf
        return lambda t, u: _leaf_eval(leaf, t)
    n_args = tree["numArgs"]
    if n_args == 0:
        idx = int(index_start[int(np.argmax(tree["ID"]))]) - 1
        return lambda t, u: u[idx]
    if n_args == 1:
        inner = _compile(tree["center"], index_start, is_coeff)
        if is_coeff and tree["method"] == "uminus":
            return lambda t, u: -inner(t, u)
        if is_coeff and tree["method"] == "uplus":
            return lambda t, u: +inner(t, u)
        fn = _UNARY_FNS[tree["method"]]
        return lambda t, u: fn(inner(t, u))
    if tree["method"] == "diff":
        idx = (int(tree["right"])
               + int(index_start[int(np.argmax(tree["ID"]))])) - 1
        return lambda t, u: u[idx]
    left = _compile(tree["left"], index_start, is_coeff)
    right = _compile(tree["right"], index_start, is_coeff)
    fn = _BINARY_FNS[tree["method"]]
    return lambda t, u: fn(left(t, u), right(t, u))


def to_first_order(fun_in, rhs, domain, num_args=None, cell_arg=False):
    """Convert a higher-order ODE operator to a first-order system.

    Returns ``(fun_out, index_start, problem_dom, coeffs,
    total_diff_orders)`` exactly as MATLAB does: ``fun_out(t, u)``
    evaluates the first-order right-hand side, ``index_start`` gives the
    first index of each variable in the state vector (1-based, matching
    the MATLAB test suite), ``problem_dom`` includes breakpoints from
    piecewise coefficients, ``coeffs`` are the leading-coefficient
    scalars/Chebfuns, and ``total_diff_orders`` the per-variable orders.

    The returned RHS assembles a JAX column array without casting scalar
    leaves to real Python floats. Dtype-stable expressions with traceable
    leaves support JIT and differentiation with respect to time and state.
    Tree construction remains eager. Existing unary fallback evaluators are
    unchanged and are not generally traceable; real-to-complex promotion is
    not forced merely to make an expression compile.

    Provenance
    ----------
    MATLAB source : @treeVar/toFirstOrder.m, @treeVar/toRHS.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    domain = tuple(float(v) for v in domain)
    t_cheb = chebfun(lambda t: t, domain=domain)

    if num_args is None:
        try:
            n_in = len(inspect.signature(fun_in).parameters)
        except (TypeError, ValueError):
            n_in = 1
        num_args = max(1, n_in - 1) if n_in > 1 else 1
        takes_t = n_in > 1
    else:
        takes_t = True

    args = []
    for j in range(num_args):
        idv = np.zeros(num_args)
        idv[j] = 1
        args.append(TreeVar(idv, domain))

    if cell_arg:
        result = fun_in(t_cheb, args)
    elif not takes_t:
        result = fun_in(*args)
    else:
        result = fun_in(t_cheb, *args)
    if isinstance(result, TreeVar):
        result = [result]
    else:
        result = list(result)

    if isinstance(rhs, (int, float, complex)):
        rhs = [rhs] * len(result)
    elif not isinstance(rhs, (list, tuple)):
        rhs = [rhs]
    rhs = list(rhs)

    # Union of breakpoints from coefficients and the RHS, collapsing
    # floating-point near-duplicates as MATLAB's domain union does.
    problem_dom = set(domain)
    for res in result:
        problem_dom.update(res.domain)
    for entry in rhs:
        edom = getattr(entry, "domain", None)
        if edom is not None:
            bps = (edom.breakpoints if hasattr(edom, "breakpoints")
                   else edom)
            problem_dom.update(float(v) for v in bps)
    pts = sorted(problem_dom)
    htol = 1e-10 * max(1.0, abs(domain[0]), abs(domain[-1]))
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] > htol:
            merged.append(p)
        elif abs(p) < abs(merged[-1]):
            # Prefer the rounder representative (e.g. 0.0 over 1e-16).
            merged[-1] = p
    problem_dom = tuple(merged)

    total_orders = np.zeros(num_args)
    for res in result:
        total_orders = np.maximum(total_orders,
                                  res.tree["diffOrder"])
    total_orders_int = total_orders.astype(int)

    index_start = np.concatenate(
        [[1], np.cumsum(total_orders_int[:-1]) + 1]).astype(int)
    index_start_der = index_start + np.arange(num_args)

    coeffs = [None] * num_args
    eq_funs = [None] * num_args
    n_state_der = int(index_start_der[-1] + total_orders_int[-1])

    for res, rhs_k in zip(result, rhs):
        diff_orders = np.asarray(res.tree["diffOrder"], dtype=float)
        if np.sum(total_orders == diff_orders) > 1:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:toFirstOrder:diffOrders",
                "The highest order derivative of more than one "
                "variable appears to be present in the same equation. "
                "Unable to convert to first order format.")
        exp_tree = expand_tree(res.tree, total_orders)
        new_tree, der_tree = split_tree(exp_tree, total_orders)
        exp_do = np.broadcast_to(
            np.atleast_1d(np.asarray(exp_tree["diffOrder"],
                                     dtype=float)),
            total_orders.shape)
        max_der_loc = int(np.argmax(exp_do == total_orders))

        # Coefficient of the highest-order derivative: evaluate the
        # derivative part with the highest-derivative slot set to 1.
        coeff_closure = _compile(der_tree, index_start_der, True)
        if max_der_loc == num_args - 1:
            one_idx = n_state_der - 1
        else:
            one_idx = int(index_start_der[max_der_loc + 1]) - 2
        coeff_arg = np.zeros(n_state_der)
        coeff_arg[one_idx] = 1.0

        # Source evaluates coeffFun(t, coeffArg) with a Chebfun t, retaining
        # its actual representation and avoiding a sampled constancy oracle.
        coeffs[max_der_loc] = coeff_closure(t_cheb, coeff_arg)

        rhs_minus = _tree("minus", 2, left=rhs_k,
                          right=0.0 if new_tree is None else new_tree,
                          diffOrder=total_orders, height=1,
                          ID=np.ones(num_args, dtype=bool),
                          hasTerms=True)
        eq_funs[max_der_loc] = _compile(rhs_minus, index_start, False)

    coeff_vals = list(coeffs)

    def fun_out(t, u):
        u = jnp.asarray(u).ravel()
        out = []
        for j in range(num_args):
            for m in range(total_orders_int[j] - 1):
                out.append(u[int(index_start[j]) + m])
            cv = jnp.asarray(_leaf_eval(coeff_vals[j], t))
            out.append(eq_funs[j](t, u) / cv)
        return jnp.stack(out).reshape(-1, 1)

    return (fun_out, index_start, problem_dom, coeffs,
            total_orders_int)


def sort_conditions(fun_in, domain, max_diff_orders):
    """How to sort the results of evaluating N.lbc/rbc so they match
    the state ordering ``u, u', v, v', ...`` (1-based indices).

    Provenance
    ----------
    MATLAB source : @treeVar/sortConditions.m
    Chebfun commit: 7574c77
    """
    max_diff_orders = np.atleast_1d(
        np.asarray(max_diff_orders, dtype=int))
    try:
        n_in = len(inspect.signature(fun_in).parameters)
    except (TypeError, ValueError):
        n_in = -1
    num_args = max(n_in, max_diff_orders.size)
    args = []
    for j in range(num_args):
        idv = np.zeros(num_args)
        idv[j] = 1
        args.append(TreeVar(idv, domain))
    if n_in == 1 and num_args > 1:
        results = fun_in(args)
    else:
        results = fun_in(*args)
    if isinstance(results, TreeVar):
        results = [results]
    else:
        results = list(results)

    var_list = [[] for _ in range(num_args)]
    order_list = [[] for _ in range(num_args)]
    for count, res in enumerate(results):
        tree = res.tree
        if int(np.sum(tree["ID"])) > 1:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:nonSeparated",
                "For initial value problems, only separated "
                "conditions are supported.")
        if not _accepted_condition(tree):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:unsupportedCondition",
                "Initial/final condition not supported.")
        active = int(np.argmax(tree["ID"]))
        order = float(np.asarray(tree["diffOrder"]).ravel()[active])
        var_list[active].append(count + 1)
        order_list[active].append(order)

    idx = []
    for j in range(num_args):
        orders = np.asarray(order_list[j])
        perm = np.argsort(orders, kind="stable")
        sorted_orders = orders[perm]
        if sorted_orders.size and sorted_orders[-1] >= \
                max_diff_orders[j]:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:tooHighOrderCondition",
                "The value of a derivative of a too high order was "
                "specified.")
        if sorted_orders.size > 1 and np.any(
                np.diff(sorted_orders) == 0):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:"
                "multipleConditionsSameVariable",
                "Multiple initial conditions on the same "
                "variable/derivative specified.")
        if not (sorted_orders.size == max_diff_orders[j]
                and (sorted_orders.size == 0
                     or (sorted_orders[0] == 0
                         and np.all(np.diff(sorted_orders) == 1)))):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:missingConditions",
                "Solving an nth order IVP/FVP requires specifying "
                "values of the solution and all the (n-1)st "
                "derivatives at the endpoint.")
        idx.extend(int(var_list[j][p]) for p in perm)
    return np.asarray(idx, dtype=int)


def _accepted_condition(tree) -> bool:
    """Reject conditions like ``5*u - 1`` or ``u + diff(u)``.

    Provenance
    ----------
    MATLAB source : @treeVar/sortConditions.m (acceptedCondition)
    Chebfun commit: 7574c77
    """
    if not _is_tree(tree) or tree["height"] == 0:
        return True
    if tree["method"] in ("plus", "minus"):
        if tree["height"] == 1:
            return True
        left, right = tree["left"], tree["right"]
        if not _is_tree(left):
            return _accepted_condition(right)
        if not _is_tree(right):
            return _accepted_condition(left)
        if np.any(left["ID"]) and np.any(right["ID"]):
            return False
        return _accepted_condition(right)
    if tree["method"] == "diff" and tree["height"] == 1:
        return True
    return False


def print_tree(tree, var_names=None, indent="") -> str:
    """Text rendering of a syntax tree.

    Provenance
    ----------
    MATLAB source : @treeVar/printTree.m
    Chebfun commit: 7574c77
    """
    if tree is None:
        return "(empty tree)\n"
    num_vars = int(np.asarray(tree["ID"]).size)
    if var_names is None:
        var_names = (["t", "u"] if num_vars == 1 else
                     ["t"] + [f"u{j + 1}" for j in range(num_vars)])

    label = tree["method"]
    if label == "constr":
        ids = np.flatnonzero(np.asarray(tree["ID"]))
        label = var_names[0] if ids.size == 0 else \
            var_names[int(ids[0]) + 1]

    do_str = " ".join(str(int(v))
                      for v in np.atleast_1d(tree["diffOrder"]))
    if indent == "":
        s = f"{label}\tdiffOrder: [{do_str}]\n"
    else:
        ind = indent if indent.endswith("|") else indent[:-1] + "|"
        s = f"{ind}--{label}\tdiffOrder: [{do_str}]\n"

    pad = indent + "  " if indent else "  "
    if tree["numArgs"] == 1:
        s += print_tree(tree["center"], var_names, pad + " ")
    elif tree["numArgs"] == 2:
        for child, mark in ((tree["left"], "|"),
                            (tree["right"], " ")):
            if _is_tree(child):
                s += print_tree(child, var_names, pad + mark)
            elif isinstance(child, (int, float, complex)):
                s += f"{pad}|--numerical \tValue: {child:2.2f}\n"
            else:
                s += f"{pad}|--chebfun\n"
    return s


def plot_tree(tree, ax=None):
    """Plot a syntax tree with matplotlib.

    Provenance
    ----------
    MATLAB source : @treeVar/plotTree.m
    Chebfun commit: 7574c77
    """
    import matplotlib
    if ax is None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    if ax is None:
        _fig, ax = plt.subplots()

    def draw(node, x, y, dx):
        label = node["method"] if _is_tree(node) else (
            f"{node:2.2f}" if isinstance(node, (int, float, complex))
            else "chebfun")
        ax.text(x, y, str(label), ha="center", va="center",
                bbox={"boxstyle": "round", "fc": "w"})
        if not _is_tree(node):
            return
        children = ([node["center"]] if node["numArgs"] == 1 else
                    [node["left"], node["right"]]
                    if node["numArgs"] == 2 else [])
        if node.get("method") == "diff":
            children = [node["left"]]
        offs = ([0.0] if len(children) == 1 else [-dx, dx])
        for child, off in zip(children, offs):
            ax.plot([x, x + off], [y - 0.12, y - 1 + 0.12], "k-",
                    lw=0.8)
            draw(child, x + off, y - 1, dx / 2)

    draw(tree, 0.0, 0.0, 1.0)
    ax.set_axis_off()
    return ax
