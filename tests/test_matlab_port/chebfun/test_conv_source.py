"""Literal convolution predicates, including the retained source tolerance.

Provenance
----------
MATLAB source: tests/chebfun/test_conv.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun

EPS = float(jnp.finfo(jnp.float64).eps)


def _cf(f, domain=(-1.0, 1.0), **kwargs):
    return cj.chebfun(f, domain=domain, **kwargs)


def _retained_tol():
    # Source assigns tol in clause 8 and retains it through clauses 9--13.
    f = _cf(lambda x: x, (-1.5, 0.0))
    g = _cf(jnp.sin)
    return 10 * max(float(f.vscale) * EPS, EPS * float(g.vscale))


def _double_endpoints(f):
    values = jnp.asarray(f.point_values)
    return f.set_point_values(values.at[0].multiply(2).at[-1].multiply(2))


@pytest.mark.parametrize("clause", range(1, 22))
def test_native_conv(clause):
    if clause in (1, 2, 3):
        f = _cf(lambda x: x)
        g = _cf(lambda x: jnp.sin(5 * x), (2.0, 4.0))
        h = _cf(lambda x: jnp.cos(2 * x), (-3.0, 1.0))
        p = _cf(lambda x: jnp.sin(15 * x))
        q = _cf(lambda x: jnp.exp(jnp.cos(3 * x)))
        if clause == 1:
            error = (f.conv(g) - g.conv(f)).norm()
        elif clause == 2:
            error = (f.conv(g).conv(h) - f.conv(g.conv(h))).norm()
        else:
            error = (f.conv(p + q) - (f.conv(p) + f.conv(q))).norm()
        assert float(error) < 100 * EPS
    elif clause == 4:
        b = (1 / 6) * _cf(
            [
                lambda x: (2 + x) ** 3,
                lambda x: 1 + 3 * (1 + x) + 3 * (1 + x) ** 2 - 3 * (1 + x) ** 3,
                lambda x: 1 + 3 * (1 - x) + 3 * (1 - x) ** 2 - 3 * (1 - x) ** 3,
                lambda x: (2 - x) ** 3,
            ],
            (-2, -1, 0, 1, 2),
        )
        s = _cf(1, (-0.5, 0.5))
        f = s
        for _ in range(3):
            f = f.conv(s)
        assert float((f - b).norm()) < 100 * EPS * float(f.vscale)
    elif clause in (5, 6):
        f = _cf(0.5)
        g = f
        for _ in range(2, 5):
            g = f.conv(g)
        if clause == 5:
            assert float(jnp.abs(g(1) - 23 / 96)) < 100 * EPS
        else:
            g1 = g(0.1)
            for _ in range(4):
                g = g.diff()
            assert float(jnp.abs(g1 - 0.332114583333333) + jnp.abs(g(0.1))) < 1e-14
    elif clause == 7:
        g = _cf(
            lambda x: jnp.sin(1 / x) * jnp.sin(1 / jnp.sin(1 / x)),
            (0.2885554757, 0.3549060246),
            n=80,
            chebkind=2,
        )
        t = 1e-7
        f = _cf(lambda x: jnp.exp(-(x**2) / (4 * t)) / jnp.sqrt(4 * jnp.pi * t), (-0.003, 0.003))
        error = jnp.abs(f.conv(g).norm(2) - 0.029781437647379)
        assert float(error) < 10 * max(float(f.vscale) * EPS, EPS * float(g.vscale))
    elif clause == 8:
        f = _cf(lambda x: x, (-1.5, 0))
        g = _cf(jnp.sin)
        assert float((f.conv(g) - f.conv(g, "old")).norm(jnp.inf)) < _retained_tol()
    elif clause in (9, 10, 11):
        if clause == 9:
            f = _cf([jnp.sin, jnp.cos, lambda x: jnp.sin(4 * x**2)], (-2, -1, 0, 1))
            x = _cf("x", (-2, 1))
            g = _double_endpoints(x.dirac().conv(f).restrict((-2, 1)))
            assert float((f - g).norm(jnp.inf)) < _retained_tol()
        else:
            x = _cf("x")
            f = x.sin()
            d = x.dirac()
            rhs = d.diff() if clause == 10 else d.diff(2) + d
            g = _double_endpoints(f.conv(rhs).restrict((-1, 1)))
            error = (g - x.cos()).norm(jnp.inf) if clause == 10 else g.norm(jnp.inf)
            assert float(error) < (10 if clause == 10 else 1e3) * _retained_tol()
    elif clause in (12, 13):
        f1 = _cf(jnp.exp, (0, float(jnp.log(2.0))))
        f2 = _cf(jnp.exp, (float(jnp.log(2.0)), float(jnp.log(3.0))))
        f3 = _cf(jnp.exp, (float(jnp.log(3.0)), float(jnp.log(4.0))))
        g2 = f1.conv(f3).conv(f2)
        other = f1.conv(f2).conv(f3) if clause == 12 else f2.conv(f3).conv(f1)
        assert float((g2 - other).normest()) < 10 * _retained_tol()
    elif clause in (14, 15):
        f = _cf(lambda x: jnp.exp(-(x**2)), (-10, 10))
        g = _cf(lambda x: jnp.exp(-(x**2)), (-20, 20))
        h = f.conv(f if clause == 14 else g, "same")
        outer = jnp.asarray([h.domain.a, h.domain.b])
        assert float(jnp.linalg.norm(outer - jnp.array([-10, 10]), ord=jnp.inf)) < 10 * EPS
    elif clause in (16, 17):
        f, g = _cf(jnp.sin), _cf(jnp.cos)
        if clause == 16:
            h1 = Chebfun.horzcat(f, g).conv(Chebfun.horzcat(g, g))
            h2 = Chebfun.horzcat(f.conv(g), g.conv(g))
        else:
            h1 = Chebfun.horzcat(f, f).conv(g)
            h2 = f.conv(Chebfun.horzcat(g, g))
        assert float((h1 - h2).norm()) < 10 * EPS
    elif clause == 18:
        x = _cf("x")
        f = 1 - 0.2 * x - abs(x - 0.2)
        t = 1e-4
        phi = _cf(lambda x: jnp.exp(-(x**2) / (4 * t)) / jnp.sqrt(4 * jnp.pi * t))
        assert float((f - f.conv(phi, "same")).norm()) < 0.01
    elif clause in (19, 20):
        scale = 1e3 if clause == 19 else 1e6
        f = _cf(jnp.sin)
        g = _cf(jnp.cos, (-1 / scale, 1 / scale))
        assert float((f.conv(g) - f.conv(g, "Old")).norm()) < 10 * EPS
    else:
        z = _cf(lambda z: z, (-0.1, 0, 0.1))
        g = z.sign()
        for degree in (16, 17):
            f = _cf([0, cj.chebpoly(degree), 0], (-1.1, -1, 1, 1.1))
            f.conv(g)
