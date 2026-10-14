"""Literal factor-based Chebfun3 squeezing, Chebfun7574c77.

Provenance
----------
MATLAB source : @chebfun3/squeeze.m, @chebfun2/constructor.m,
    @chebfun2/outerProduct.m
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun3d._hosvd import _mode_product
from chebfunjax.chebfun3d._mtimes import _panel
from chebfunjax.chebfun3d._simplify import _factor_panel
from chebfunjax.chebpref import ChebfunPref


def source_squeeze(f):
    if f.isempty():
        return Chebfun.empty()
    ranks = f.rank
    input_core = f.core.reshape(f.core.shape + (1,)*max(0, 3-f.core.ndim))
    factors = [_factor_panel(group, ChebfunPref().techPrefs.chebfuneps)
               for group in (f.cols, f.rows, f.tubes)]
    lengths = [max(t.n for t in group) for group in factors]

    def panel(axis):
        return _panel(factors[axis], tuple(f.domain[2*axis:2*axis+2]))

    def constant(axis):
        return jnp.column_stack([t.coeffs for t in factors[axis]])

    if ranks == (1, 1, 1):
        # Literal source priority: constant functions retain the z interval.
        if lengths[0] == lengths[1] == 1:
            core = _mode_product(_mode_product(input_core, constant(0), 0), constant(1), 0)
            return panel(2) @ core.reshape(())
        if lengths[0] == lengths[2] == 1:
            core = _mode_product(_mode_product(input_core, constant(0), 0), constant(2), 2)
            return panel(1) @ core.reshape(())
        if lengths[1] == lengths[2] == 1:
            core = _mode_product(_mode_product(input_core, constant(1), 1), constant(2), 2)
            return panel(0) @ core.reshape(())
        # Pinned source has no assignment in this rank-one branch.
        raise ValueError("MATLAB:unassignedOutputs: squeeze output is unassigned.")

    for axis in range(3):
        if ranks[axis] == 1 and lengths[axis] == 1:
            core = _mode_product(input_core, constant(axis), axis)
            core = jnp.squeeze(core, axis=axis)
            remaining = [i for i in range(3) if i != axis]
            # The outer product's left panel is the y factor. Native
            # chebfun2(existing, newdomain) returns the existing object,
            # retaining its reversed domain rather than applying newdomain.
            return (panel(remaining[0]) @ core) @ panel(remaining[1]).T
    return f
