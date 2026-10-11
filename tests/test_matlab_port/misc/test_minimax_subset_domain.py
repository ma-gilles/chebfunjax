"""Python subset-domain regression for stored Chebfun breakpoint handling.

The documented minimax domain is the approximation interval. This control
uses the exact best linear approximation of x**2 on [-1, .5], rather than
introducing a relaxed native test predicate.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.minimax import minimax


def test_quadratic_subset_domain_preserves_approximation_interval():
    f = chebfun(lambda x: x**2, domain=(-2.0, 0.0, 2.0))
    result = minimax(f, 1, domain=(-1.0, .5))
    # m=-.25, r=.75. The exact best line has Chebyshev coefficients
    # [m**2+r**2/2, 2*m*r] and error r**2/2 on this requested interval.
    eps = np.finfo(float).eps
    assert np.max(np.abs(np.asarray(result.coeffs) - [.34375, -.375])) < 64 * eps
    assert abs(result.err - .28125) < 64 * eps
    assert bool(jnp.all((result.xk >= -1) & (result.xk <= .5)))
