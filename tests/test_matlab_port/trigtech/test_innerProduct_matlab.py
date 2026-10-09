"""All 11 predicates of tests/trigtech/test_innerProduct.m, Chebfun7574c77.

Source expressions and strict 10eps*vscale bounds are retained. Python 1D
scalar columns and eager real scalar storage are representation adapters.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = jnp.finfo(jnp.float64).eps


def make(op):
    return Trigtech.from_function(op)


def tol(f):
    return 10 * EPS * f.vscale


@pytest.mark.parametrize("slot", range(1, 12))
def test_native_inner_product(slot):
    if slot in (1, 2):
        f = make(lambda x: jnp.sin(2 * jnp.pi * x))
        g = make(lambda x: jnp.cos((2 if slot == 1 else 4) * jnp.pi * x))
        assert abs(f.innerProduct(g)) < max(tol(f), tol(g))
    elif slot in (3, 4):
        old_f = make(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        if slot == 3:
            f, g = old_f, make(lambda x: jnp.exp(-jnp.cos(jnp.pi * x)))
            exact = 2
        else:
            f = make(lambda x: 1 + 0 * x)
            g = make(lambda x: jnp.sin(jnp.pi * x) ** 4)
            exact = 3 / 4
        # Native slot4 intentionally inherits tol_f from slot3.
        assert abs(f.innerProduct(g) - exact) < max(tol(old_f), tol(g))
    elif slot <= 9:
        f = make(lambda x: jnp.sin(jnp.pi * x) ** 4)
        g = make(lambda x: jnp.exp(jnp.cos(2 * jnp.pi * x)))
        h = make(lambda x: jnp.exp(1j * 4 * jnp.pi * x))
        if slot == 5:
            alpha = -0.194758928283640 + 0.075474485412665j
            beta = -0.526634844879922 - 0.685484380523668j
            assert abs((alpha * f).innerProduct(beta * g) -
                       jnp.conj(alpha) * beta * f.innerProduct(g)) < max(tol(f), tol(g))
        elif slot == 6:
            assert abs(g.innerProduct(h) - jnp.conj(h.innerProduct(g))) < max(tol(g), tol(h))
        elif slot == 7:
            assert abs((f + g).innerProduct(h) - (f.innerProduct(h) +
                       g.innerProduct(h))) < max(tol(f), tol(g), tol(h))
        elif slot == 8:
            assert abs(f.innerProduct(g + h) - (f.innerProduct(g) +
                       f.innerProduct(h))) < max(tol(f), tol(g), tol(h))
        else:
            vals = jnp.array([u.innerProduct(u) for u in (f, g, h)])
            assert not jnp.iscomplexobj(vals)
            assert jnp.all(vals >= 0)
    else:
        f = make(lambda x: jnp.stack((jnp.cos(jnp.pi * jnp.sin(2 * jnp.pi * x)),
                                     jnp.exp(jnp.sin(2 * jnp.pi * x)),
                                     jnp.sin(jnp.pi * jnp.sin(jnp.pi * x))), axis=-1))
        if slot == 11:
            with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:innerProduct:input"):
                f.innerProduct(2)
        else:
            g = make(lambda x: jnp.stack((jnp.exp(jnp.cos(jnp.pi * x)),
                                         jnp.exp(-jnp.sin(2 * jnp.pi * x)),
                                         jnp.cos(jnp.pi * x)), axis=-1))
            exact = jnp.array([[-0.765066454912991, -1.032310819204998, 0],
                               [3.204359383938947, 2, 0], [0, 0, 0]])
            assert jnp.max(jnp.abs(f.innerProduct(g) - exact)) < max(tol(f), tol(g))
