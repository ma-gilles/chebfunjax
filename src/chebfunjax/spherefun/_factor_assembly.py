"""Common-length Fourier factor matrices with bounded stack compilation.

Provenance
----------
MATLAB source: @spherefun/projectOntoBMCI.m, @spherefun/sample.m,
@chebfun/trigcoeffs.m and @trigtech/prolong.m; Chebfun7574c77680d7e82b79626300bf255498271a72df.
The source uses a matrix of coefficient columns. Python stores separate factors;
assembly retains each source prolongation and original column order. Grouping
stack inputs is a compiler adapter: JAX0.11 CPU compilation of a714-input stack
exceeded300s in actual Atmospheric reconstruction. No arithmetic, precision,
source tolerance or factor count is changed by copying columns in groups.
"""
import jax
import jax.numpy as jnp

from chebfunjax.tech.trigtech import _trig_prolong_coeffs


def stack_factor_coefficients(techs):
    """Prolong each factor, then assemble columns in original order."""
    size = max(t.coeffs.shape[0] for t in techs)
    groups = [
        jnp.stack([_trig_prolong_coeffs(t.coeffs, size)
                   for t in techs[begin:begin+32]], axis=1)
        for begin in range(0, len(techs), 32)
    ]
    return groups[0] if len(groups) == 1 else jnp.concatenate(groups, axis=1)


@jax.jit
def select_factor_columns(coeffs, index):
    """Copy native factor columns in order; indices are dynamic.

    Source: @spherefun/projectOntoBMCI.m, Chebfun7574c77.
    The zero-based indexing convention is the existing Python adapter.
    """
    return coeffs[:, index]


@jax.jit
def select_factor_entries(values, index):
    """Copy selected pivot entries without changing cdr arithmetic.

    Source: @spherefun/sum2.m and @separableApprox/cdr.m, Chebfun7574c77.
    The zero-based indexing convention is the existing Python adapter.
    """
    return values[index]
