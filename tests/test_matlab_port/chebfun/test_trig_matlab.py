"""Port of MATLAB Chebfun tests/chebfun/test_trig.m (Fable 5).

Every trigonometric / hyperbolic function and its inverse (including
the degree variants) is applied to a piecewise chebfun and compared with
the exact composition at 1000 random points.  Where MATLAB returns a
complex result for a real argument outside the real domain (``acosh``,
``asec``, ``acsc``, ``acoth`` of values in (0, 1)), the reference is
evaluated in complex arithmetic.

Provenance
----------
MATLAB source : tests/chebfun/test_trig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps
D2R = np.pi / 180.0


def _cplx(fn):
    def _f(v):
        v = np.asarray(v)
        with np.errstate(all="ignore"):
            out = fn(v)
        if np.any(np.isnan(out)):
            out = fn(v.astype(complex))
        return out
    return _f


# (method name, exact NumPy reference)
TRIG = [
    ("acos", _cplx(np.arccos)), ("acosd", _cplx(lambda v: np.arccos(v) / D2R)),
    ("acosh", _cplx(np.arccosh)), ("acot", lambda v: np.arctan(1 / v)),
    ("acotd", lambda v: np.arctan(1 / v) / D2R),
    ("acoth", _cplx(lambda v: np.arctanh(1 / v))),
    ("acsc", _cplx(lambda v: np.arcsin(1 / v))),
    ("acscd", _cplx(lambda v: np.arcsin(1 / v) / D2R)),
    ("acsch", lambda v: np.arcsinh(1 / v)),
    ("asec", _cplx(lambda v: np.arccos(1 / v))),
    ("asecd", _cplx(lambda v: np.arccos(1 / v) / D2R)),
    ("asech", _cplx(lambda v: np.arccosh(1 / v))),
    ("asin", _cplx(np.arcsin)), ("asind", _cplx(lambda v: np.arcsin(v) / D2R)),
    ("asinh", np.arcsinh), ("atan", np.arctan),
    ("atand", lambda v: np.arctan(v) / D2R), ("atanh", _cplx(np.arctanh)),
    ("cos", np.cos), ("cosd", lambda v: np.cos(D2R * v)), ("cosh", np.cosh),
    ("cot", lambda v: 1 / np.tan(v)), ("cotd", lambda v: 1 / np.tan(D2R * v)),
    ("coth", lambda v: 1 / np.tanh(v)), ("csc", lambda v: 1 / np.sin(v)),
    ("cscd", lambda v: 1 / np.sin(D2R * v)), ("csch", lambda v: 1 / np.sinh(v)),
    ("sec", lambda v: 1 / np.cos(v)), ("secd", lambda v: 1 / np.cos(D2R * v)),
    ("sech", lambda v: 1 / np.cosh(v)), ("sin", np.sin),
    ("sind", lambda v: np.sin(D2R * v)), ("sinh", np.sinh), ("tan", np.tan),
    ("tand", lambda v: np.tan(D2R * v)), ("tanh", np.tanh),
    ("mysinc", lambda v: np.sin(v) / v),
]


def _base_op(x):
    return (jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x) * (np.pi / 16)
            + np.pi / 8)


class TestChebfunTrig:
    def test_all_matlab_assertions(self):
        xr = jnp.asarray(2 * np.random.RandomState(7681).rand(1000) - 1)
        # MATLAB: pref.splitting = 1 (sign(0) = 0 sits ON the breakpoint 0.1;
        # only splitting's extrapolated endpoints resolve the middle piece).
        f = chebfun(_base_op, domain=(-1.0, -0.2, 0.1, 1.0), splitting=True)
        base = np.asarray(_base_op(xr))
        failures = []
        for name, ref in TRIG:
            g = f.sin() / f if name == "mysinc" else getattr(f, name)()
            err = np.asarray(g(xr)) - ref(base)
            tol = 50 * float(np.max(np.abs(np.atleast_1d(g.vscale)))) * EPS
            if not np.max(np.abs(err)) < tol:
                failures.append((name, float(np.max(np.abs(err))), tol))
        assert not failures, failures
