"""All9 literal tests/misc/test_lagpoly.m predicates on [0,inf].

Provenance
----------
MATLAB source : tests/misc/test_lagpoly.m
Chebfun commit: 7574c77
"""

import math

# uses-numpy: original numerical test epsilon and independent closed-form oracles.
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebfun

EPS = np.finfo(float).eps


@pytest.mark.parametrize("clause", range(1, 10))
def test_native_laguerre(clause):
    x = jnp.linspace(0, 1, 100)
    if clause <= 7:
        n = clause - 1
        a = 0.0
    else:
        n = 3 if clause == 8 else 7
        a = 0.5 if clause == 8 else 1 / math.sqrt(2)
    f = cj.lagpoly(n, a)
    if n < 2:
        g = chebfun(lambda x: 1 + 0 * x if n == 0 else 1 - x, domain=(0, jnp.inf))
        error = float((f - g).norm())
    else:
        forms = [
            lambda x: 1 + 0 * x,
            lambda x: 1 - x,
            lambda x: (x * x - 4 * x + 2) / 2,
            lambda x: (-(x**3) + 9 * x * x - 18 * x + 6) / 6,
            lambda x: (x**4 - 16 * x**3 + 72 * x * x - 96 * x + 24) / 24,
            lambda x: (-(x**5) + 25 * x**4 - 200 * x**3 + 600 * x * x - 600 * x + 120) / 120,
            lambda x: (
                (x**6 - 36 * x**5 + 450 * x**4 - 2400 * x**3 + 5400 * x * x - 4320 * x + 720) / 720
            ),
        ]
        if clause <= 7:
            expected = forms[n](x)
        elif clause == 8:
            expected = (
                -(x**3) / 6
                + (a + 3) * x * x / 2
                - (a + 2) * (a + 3) * x / 2
                + (a + 1) * (a + 2) * (a + 3) / 6
            )
        else:
            expected = jnp.zeros_like(x)
            for i in range(n + 1):
                binom = math.gamma(n + a + 1) / (math.gamma(a + i + 1) * math.gamma(n - i + 1))
                expected = expected + (-1) ** i * binom * x**i / math.factorial(i)
        error = float(jnp.linalg.norm(f(x) - expected))
    factor = {6: 10, 7: 100, 9: 10}.get(clause, 1)
    assert error < factor * 1e5 * EPS
