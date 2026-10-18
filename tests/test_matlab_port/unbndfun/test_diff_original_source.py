"""Literal14 default test_diff.m predicates, Chebfun7574c77.

The fixture contains the three native seeded100-point query vectors, not
NumPy RNG replacements. Existing endpoint-inclusive authored controls in
``test_diff_matlab.py`` remain separate and unchanged.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np  # Retained native fixture loading only.
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

DATA = Path(__file__).with_name("test_diff_native_rng_20261017.npz")
EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("clause", range(1, 15))
def test_original_default_diff_predicate(clause):
    with np.load(DATA) as fixture:
        key = "x_both" if clause <= 4 else "x_right" if clause <= 9 else "x_left"
        x = jnp.asarray(fixture[key]).reshape((-1,))
    assert x.shape == (100,)
    if clause <= 4:
        domain = Domain((-jnp.inf, jnp.inf))
        if clause == 1:
            op = lambda t: jnp.exp(-(t**2))
            f = Unbndfun.from_function(op, domain)
            g = f.diff(2)
            exact = 4 * x**2 * jnp.exp(-(x**2)) - 2 * jnp.exp(-(x**2))
            bound = 1e3 * EPS * g.vscale
        elif clause == 2:
            op = lambda t: t**2 * jnp.exp(-(t**2))
            f = Unbndfun.from_function(op, domain)
            g = f.diff()
            exact = 2 * x * jnp.exp(-(x**2)) - 2 * x**3 * jnp.exp(-(x**2))
            bound = 2e1 * EPS * g.vscale
        elif clause == 3:
            op = lambda t: (1 - jnp.exp(-(t**2))) / t
            f = Unbndfun.from_function(op, domain)
            g = f.diff()
            exact = 2 * jnp.exp(-(x**2)) + (jnp.exp(-(x**2)) - 1) / x**2
            bound = 1e2 * EPS * g.vscale
        else:
            op = lambda t: t**2 * (1 - jnp.exp(-(t**2)))
            # Native singPref.blowup=true with explicit exponents[2,2].
            f = Unbndfun.from_function(op, domain, exps=(2.0, 2.0))
            g = f.diff()
            exact = 2 * x**3 * jnp.exp(-(x**2)) - 2 * x * (jnp.exp(-(x**2)) - 1)
            bound = 1e7 * EPS * f.vscale
    elif clause <= 9:
        domain = Domain((1.0, jnp.inf))
        exps = None
        if clause == 5:
            op = lambda t: jnp.exp(-t)
            factor = 1e1
        elif clause == 6:
            op = lambda t: t * jnp.exp(-t)
            factor = 10
        elif clause == 7:
            op = lambda t: (1 - jnp.exp(-t)) / t
            factor = 1e1
        elif clause == 8:
            op = lambda t: 1 / t
            factor = 1e1
        else:
            op = lambda t: t * (5 + jnp.exp(-(t**3)))
            exps = (0.0, 1.0)
            factor = 1e2
        f = Unbndfun.from_function(op, domain, exps=exps)
        g, exact, bound = f, op(x), factor * EPS * f.vscale
    else:
        b = -3 * jnp.pi
        domain = Domain((-jnp.inf, b))
        exps = None
        if clause == 10:
            op = lambda t: jnp.exp(t)
        elif clause == 11:
            op = lambda t: t * jnp.exp(t)
        elif clause == 12:
            op = lambda t: (1 - jnp.exp(t)) / t
        elif clause == 13:
            op = lambda t: 1 / t
        else:
            op = lambda t: t * (5 + jnp.exp(t**3)) / (b - t)
            exps = (0.0, -1.0)
        f = Unbndfun.from_function(op, domain, exps=exps)
        g, exact = f, op(x)
        bound = (1e1 if clause == 14 else 1) * EPS * f.vscale
    error = float(jnp.max(jnp.abs(g(x) - exact)))
    print({"clause": clause, "error": error, "bound": float(bound)}, flush=True)
    assert error < bound
