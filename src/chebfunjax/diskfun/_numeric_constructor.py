"""Non-scalar numeric Diskfun construction and BMC projection on JAX arrays.

Chebfun7574c77680d7e82b79626300bf255498271a72df, @diskfun/constructor.m
constructFromDouble/PhaseOne and @diskfun/projectOntoBMCII.m. Source power,
GE, transforms and nonfinite extrapolation retain their qualified providers.
Python transfers branch and object/location metadata only. This does not
replace the native scalar-input adaptive route or promise whole-JIT/AD.
"""

import jax
import jax.numpy as jnp

from chebfunjax.operators import _qz_ieee as ieee
from chebfunjax.tech.chebtech import Chebtech2, _extrapolate_values
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils._signed_gradual import flip_sign_bits
from chebfunjax.utils.quadrature import chebpts, trigpts

from ._numeric_phase_one import _exact_tiny_sum, phase_one_numeric
from ._numeric_tolerance import numeric_tolerance


def _sum_columns(values):
    """Source sum(:,1), with proved exact gradual tiny sums when eligible.

    The normal fallback uses the JAX reduction provider; generic native sum
    order and NaN payload equality remain unqualified. Tiny eligibility and
    exact-zero signs use the independently qualified PhaseOne sum helper.
    """
    if values.shape[0] == 0:
        return jnp.zeros(values.shape[1], dtype=jnp.float64)
    eligible, exact = jax.vmap(_exact_tiny_sum, in_axes=1)(values)
    return jnp.where(eligible, exact, jnp.sum(values, axis=0))


def even_columns(coeffs):
    """Literal native even projection, retaining the first plus column.

    The source factor multiplies by a separately rounded reciprocal. Direct
    division gives different finite words. The source length-one outer
    product is evaluated by IEEE scalar products to retain tiny values;
    inactive intermediate zero signs can differ from native GEMM, while
    the final additions have separate strict native coefficient controls.
    """
    coeffs = jnp.asarray(coeffs, jnp.float64).at[1::2, :].set(0.)
    if coeffs.shape[1] > 1:
        xe = coeffs[::2, 1:]
        difference = ieee.add(_sum_columns(xe[::2, :]),
                              flip_sign_bits(_sum_columns(xe[1::2, :])))
        factor = ieee.multiply(ieee.divide(jnp.float64(1),
                                          jnp.float64(xe.shape[0])), difference)
        signs = jnp.where(jnp.arange(xe.shape[0]) % 2 == 0, 1., -1.)
        correction = ieee.multiply(signs[:, None], factor[None, :])
        coeffs = coeffs.at[::2, 1:].set(ieee.add(xe, correction))
    return coeffs


def project_numeric(function):
    """Project numeric full-length slices without NumPy array arithmetic."""
    cols, rows = list(function.cols), list(function.rows)
    for indices, even in ((function.idx_plus, True),
                          (function.idx_minus, False)):
        if not indices:
            continue
        # Numeric transforms preserve one shared radial length. Native
        # quasimatrix projection operates on this complete coefficient matrix.
        lengths = {cols[i].coeffs.shape[0] for i in indices}
        if len(lengths) != 1:
            raise ValueError("numeric source columns must share radial storage")
        coefficients = jnp.stack([cols[i].coeffs for i in indices], axis=1)
        coefficients = (even_columns(coefficients) if even else
                        coefficients.at[::2, :].set(0.))
        for position, i in enumerate(indices):
            cols[i] = Chebtech2.from_coeffs(coefficients[:, position])
            coeffs = rows[i].coeffs
            modes = jnp.arange(coeffs.shape[0]) - coeffs.shape[0]//2
            remove = modes % 2 == (1 if even else 0)
            coeffs = jnp.where(remove, 0., coeffs)
            rows[i] = Trigtech.from_coeffs(
                coeffs, real_columns=rows[i].real_columns).real()
    return type(function)(cols=cols, rows=rows, pivots=function.pivots,
                          idx_plus=function.idx_plus, idx_minus=function.idx_minus,
                          pivot_locations=function.pivot_locations,
                          nonzero_poles=function.nonzero_poles)


def construct_numeric(cls, values, tol, alpha):
    """Source non-scalar numeric route; values already have real dtype."""
    values = jnp.asarray(values, jnp.float64)
    if values.size <= 1:
        raise ValueError("non-scalar numeric constructor requires more than one entry")
    if values.ndim < 2:
        values = values.reshape(-1, 1)
    if values.ndim == 2 and values.shape[0] == 1 and values.shape[1] % 2 == 0:
        return project_numeric(cls(
            cols=[Chebtech2.from_values(values[:, 0])],
            rows=[Trigtech.from_values(jnp.ones(values.shape[1], jnp.float64))],
            pivots=jnp.ones(1, jnp.float64), idx_plus=(0,), idx_minus=(),
            pivot_locations=((-float(jnp.pi), 0.),), nonzero_poles=True))
    if values.ndim == 2:
        bits = jax.lax.bitcast_convert_type(values, jnp.uint64)
        exact_zero = bool(jnp.all((bits & jnp.uint64(0x7fffffffffffffff)) == 0))
        if exact_zero and values.shape[1] % 2 == 0:
            radial, angular = values.shape
            return project_numeric(cls(
                cols=[Chebtech2.from_coeffs(jnp.zeros(radial, jnp.float64))],
                rows=[Trigtech.from_coeffs(jnp.zeros(angular, jnp.complex128),
                                          is_real=True)],
                pivots=jnp.asarray([jnp.inf], jnp.float64),
                idx_plus=(0,), idx_minus=(),
                pivot_locations=((-float(jnp.pi), 0.),), nonzero_poles=False))
    n, m = values.shape
    if m % 2:
        raise ValueError("DISKFUN:CONSTRUCTOR:VALUES: When constructing "
                         "from values the number of columns must be even.")
    if tol is None:
        hx = ieee.divide(ieee.multiply(jnp.float64(2), jnp.float64(jnp.pi)),
                         jnp.float64(m))
        hy = ieee.divide(jnp.float64(jnp.pi), jnp.float64(n-1))
        tol = numeric_tolerance(values, hx, hy, jnp.finfo(jnp.float64).eps)[0]
    phase = phase_one_numeric(values, tol, alpha)
    if phase.cols.shape[1] == 0:
        raise IndexError("MATLAB:badsubscript: Index in position 2 exceeds array bounds.")
    col_data = phase.cols
    if (not bool(jnp.all(jnp.isnan(col_data)))
            and not bool(jnp.all(jnp.isfinite(col_data)))):
        col_data = _extrapolate_values(
            col_data, chebpts(col_data.shape[0]),
            Chebtech2.barywts(col_data.shape[0]))[0]
    cols = [Chebtech2.from_values(col_data[:, j])
            for j in range(phase.cols.shape[1])]
    rows = [Trigtech.from_values(phase.rows[:, j])
            for j in range(phase.rows.shape[1])]
    pivots = phase.pivots
    if not bool(jnp.any(ieee.nonzero(pivots))):
        pivots = jnp.full_like(pivots, jnp.inf)
    locations = []
    if len(phase.indices):
        theta = trigpts(2*m, (-jnp.pi, jnp.pi))[0]
        radius = chebpts(2*n+1)[n:]
        for index, (j, k) in enumerate(phase.indices):
            location = (float(theta[k]), float(radius[j]))
            locations.append(location)
            if (bool(ieee.nonzero(phase.pivot_array[index, 0]))
                    and bool(ieee.nonzero(phase.pivot_array[index, 1]))):
                locations.append(location)
    return project_numeric(cls(
        cols=cols, rows=rows, pivots=pivots,
        idx_plus=tuple(phase.plus), idx_minus=tuple(phase.minus),
        pivot_locations=tuple(locations[:len(pivots)]),
        nonzero_poles=phase.remove_pole))
