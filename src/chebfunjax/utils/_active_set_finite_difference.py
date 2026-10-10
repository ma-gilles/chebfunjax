"""Private finite-box forward differences for the native active-set caller.

Nominal steps follow R2017a finDiffFseminf.m90-97/nonzerosign.m; the ordinary
protected provider transaction follows nlconst.m307-311. Bound adjustment,
denominator, callback order and invalid-value behavior are backed by 37
captures of the original R2017a protected files under the R2025b runtime.
Source attribution: nonzerosign.m, Copyright 2008 The MathWorks, Inc.;
finDiffFseminf.m, Copyright 2008-2011 The MathWorks, Inc.; nlconst.m,
Copyright 1990-2015 The MathWorks, Inc. Protected provider internals were
observed, not available as readable source. The separate Chebfun caller is
@separableApprox/minandmax2.m at 7574c77680d7e82b79626300bf255498271a72df,
Copyright 2017 The University of Oxford and The Chebfun Developers.
Capture provenance is bound in tests/fixtures/active_set_fd_native.json.
Fixed caller registration only; no arbitrary option, full JIT or AD contract.
"""
import jax.numpy as jnp

from chebfunjax.utils._active_set_qp import _real64


def _fixed_options(options):
    """Reject options outside the existing two-variable native caller."""
    fixed = {'FinDiffType': 'forward', 'DiffMinChange': 0.,
             'DiffMaxChange': float('inf'), 'fwdFinDiff': True,
             'scaleObjConstr': False, 'chkFunEval': False,
             'chkComplexObj': False, 'isGrad': True}
    if set(options) != set(fixed) | {'FinDiffRelStep', 'TypicalX'}:
        raise ValueError('Unsupported native finite-difference options')
    if any(options[key] != value for key, value in fixed.items()):
        raise ValueError('Unsupported native finite-difference options')
    relative = _real64(options['FinDiffRelStep'], 'FinDiffRelStep')
    typical = _real64(options['TypicalX'], 'TypicalX')
    if (relative.shape != (2,) or typical.shape != (2,)
            or not bool(jnp.all(relative == jnp.sqrt(jnp.finfo(jnp.float64).eps)))
            or not bool(jnp.all(typical == 1))):
        raise ValueError('Unsupported native finite-difference step options')
    return relative, typical


def _bounded_step(x, lower, upper, step):
    """R2017a protected bound helper, captured finite positive-width scope.

    Feasibility is tested on rounded endpoints. Comparing step to slack
    instead changes the direction in the two near-rounding native controls.
    If neither nominal direction fits, use the larger slack; ties go lower.
    """
    # The protected helper leaves nominal steps unchanged at already
    # infeasible coordinates, including an actual accepted SQP iterate.
    if bool((x < lower) | (x > upper)):
        return step
    trial = x + step
    if bool((trial < lower) | (trial > upper)):
        opposite = x - step
        if bool((opposite >= lower) & (opposite <= upper)):
            step = -step
        elif bool(upper-x > x-lower):
            step = upper-x
        else:
            step = lower-x
    return step


def finite_difference(x, fun, lower, upper, base_value, options):
    """Return the binary64 gradient and number of extra objective calls.

    Python's two-entry vector adapts the native caller's row-shaped callback.
    Coordinates are evaluated in order and restored between calls. With the
    fixed native check flags, NaN/Inf values propagate without retries and
    objective exceptions propagate immediately. NaN payload bits are not a
    backend contract. The requested adjusted step is the denominator, even
    when the rounded displacement differs.
    """
    relative, typical = _fixed_options(options)
    x = _real64(x, 'x')
    lower, upper = _real64(lower, 'lower'), _real64(upper, 'upper')
    base = _real64(base_value, 'base_value')
    if any(v.shape != (2,) for v in (x, lower, upper)) or base.shape != ():
        raise ValueError('Native finite differences require two variables and a scalar objective')
    if not bool(jnp.all(jnp.isfinite(x) & jnp.isfinite(lower) & jnp.isfinite(upper)
                        & (lower < upper))):
        raise ValueError('Native finite differences require finite points and finite positive-width bounds')
    gradients = []
    for index in range(2):
        sign = jnp.where(x[index] < 0, -1., 1.)
        nominal = relative[index] * sign * jnp.maximum(jnp.abs(x[index]), typical[index])
        step = _bounded_step(x[index], lower[index], upper[index], nominal)
        trial = x.at[index].set(x[index] + step)
        value = _real64(fun(trial), 'objective')
        if value.shape != ():
            raise ValueError('Objective must be scalar')
        gradients.append((value-base)/step)
    return jnp.stack(gradients), 2
