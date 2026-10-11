"""All14 literal tests/misc/test_hermpoly.m predicates, including stale err14.

Provenance
----------
MATLAB source : tests/misc/test_hermpoly.m
Chebfun commit: 7574c77
"""

# uses-numpy: original numerical test epsilon and independent closed-form oracles.
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebfun

EPS = np.finfo(float).eps


def hermite_exact(n, x, prob):
    if prob:
        return [
            lambda x: 1 + 0 * x,
            lambda x: x,
            lambda x: x * x - 1,
            lambda x: x**3 - 3 * x,
            lambda x: x**4 - 6 * x * x + 3,
            lambda x: x**5 - 10 * x**3 + 15 * x,
            lambda x: x**6 - 15 * x**4 + 45 * x * x - 15,
        ][n](x)
    return [
        lambda x: 1 + 0 * x,
        lambda x: 2 * x,
        lambda x: 4 * x * x - 2,
        lambda x: 8 * x**3 - 12 * x,
        lambda x: 16 * x**4 - 48 * x * x + 12,
        lambda x: 32 * x**5 - 160 * x**3 + 120 * x,
        lambda x: 64 * x**6 - 480 * x**4 + 720 * x * x - 120,
    ][n](x)


@pytest.mark.parametrize("clause", range(1, 15))
def test_native_hermite(clause):
    prob = clause >= 8
    n = clause - 8 if prob else clause - 1
    f = cj.hermpoly(n, "prob" if prob else "phys")
    if n < 3:
        g = chebfun(lambda x: hermite_exact(n, x, prob), domain=(-jnp.inf, jnp.inf))
        error = float((f - g).norm())
    else:
        x = jnp.linspace(-1, 1, 100)
        error = float(jnp.linalg.norm(f(x) - hermite_exact(n, x, prob)))
    if clause == 14:
        # Native computes degree6 norm without assigning err, then reuses
        # degree5 err from clause13. Preserve this original predicate.
        previous = cj.hermpoly(5, "prob")
        error = float(jnp.linalg.norm(previous(x) - hermite_exact(5, x, True)))
    factor = {6: 10, 7: 100, 14: 10}.get(clause, 1)
    assert error < factor * 1e6 * EPS
