"""Exact active-set preferences for Chebfun's real two-variable extrema caller.

Caller: @separableApprox/minandmax2.m124-126, Chebfun7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
Private SQP/finite-difference provenance is documented in the respective helpers.
Native backend qualification covers R2017a optimizer sources under R2025b.
This adapter preserves source exception/partial assignment handling in refine_seed.
"""
from typing import NamedTuple

from chebfunjax.utils._active_set_box_sqp import active_set_box
from chebfunjax.utils._active_set_finite_difference import finite_difference


class ActiveResult(NamedTuple):
    x: object
    fun: object


def native_active_set(fun, initial, lower, upper, options):
    """Use fixed caller options; return value independently of exit status."""
    if (options.display != 'none' or options.algorithm != 'active-set'
            or options.tol_fun != 2.**-52 or options.tol_x != 2.**-52):
        raise ValueError('Unsupported native extrema active-set options')
    result = active_set_box(fun, initial, lower, upper,
                            finite_difference=finite_difference)
    return ActiveResult(result.x, result.value)
