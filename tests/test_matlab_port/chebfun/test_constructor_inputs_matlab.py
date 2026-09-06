"""Port of MATLAB Chebfun tests/chebfun/test_constructor_inputs.m (Fable 5).

MATLAB's preference/flag spellings map to constructor keywords:
``'vectorize'`` -> ``vectorize=True``, ``'splitting', 'on'`` ->
``splitting=True``, ``'extrapolate'`` -> ``extrapolate=True``, ``'equi'``
-> ``equi=True``, ``'coeffs'`` -> ``coeffs=True``, ``'chebkind'`` ->
``chebkind=``, ``'trunc'`` -> ``trunc=``, ``'minSamples'`` ->
``min_samples=``, ``'resampling'`` -> ``resampling=True``,
``'maxdegree'`` -> ``max_length=``, ``'splitdegree'``/``'splitLength'``
-> ``split_length=``, ``'splitMaxLength'`` -> ``split_max_length=``,
``pref.tech = @chebtech1`` -> ``tech='chebtech1'``, ``'eps'`` -> ``eps=``.

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_inputs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.tech.chebtech import Chebtech1

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _happy(f):
    return all(bool(getattr(p.tech, "ishappy", True)) for p in f.funs)


class TestChebfunConstructorInputs:
    def test_all_matlab_assertions(self):
        xx = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)

        chebfun(lambda x: x ** 2, vectorize=True)                   # pass(1)

        f = chebfun(lambda x: jnp.sin(x), n=5)
        assert len(f.funs[0]) == 5                                  # pass(2)

        f = chebfun(lambda x: jnp.abs(x), splitting=True)
        assert len(f.funs) == 2                                     # pass(3)

        f = chebfun(lambda x: jnp.sign(x), domain=(0.0, 1.0), extrapolate=True)
        assert _happy(f)                                            # pass(4)

        f = chebfun([0, 1, 2, 3, 4, 5], domain=(0, 1, 2, 3, 4, 5, 6))
        g = chebfun(f.funs)
        assert len(g.funs) == 6 and list(g.domain.breakpoints) == list(range(7))  # pass(5)

        f = chebfun([jnp.array([0.0, 1.0]), jnp.array([2.0, 3.0]),
                     jnp.array([4.0, 5.0])], domain=(0, 1, 2, 3))
        g = chebfun(f.funs)
        assert len(g.funs) == 3 and list(g.domain.breakpoints) == [0, 1, 2, 3]  # pass(6)

        f = chebfun(jnp.sin, n=10)
        assert len(f.funs) == 1 and len(f.funs[0]) == 10            # pass(7)

        f = chebfun(jnp.asarray(np.random.RandomState(1).rand(10, 3)), equi=True)
        assert len(f.funs) == 1 and len(f.funs[0]) > 10             # pass(8)

        f = chebfun(jnp.asarray([1.0, 2.0, 3.0]), coeffs=True)
        assert np.array_equal(np.asarray(f.funs[0].tech.coeffs), [1, 2, 3])  # pass(9)

        f1 = chebfun(lambda x: x, chebkind="1st")
        f3 = chebfun(lambda x: x, chebkind=1)
        assert isinstance(f1.funs[0].tech, Chebtech1) and \
            isinstance(f3.funs[0].tech, Chebtech1)                  # pass(10)

        f = chebfun("1")
        assert np.all(np.asarray(f(jnp.linspace(-1, 1, 10))) == 1)  # pass(11)

        f = chebfun(jnp.abs, trunc=10, splitting=True)
        c = np.asarray(f.chebcoeffs())
        assert abs(-4 / 63 / np.pi - c[8]) < 10 * EPS               # pass(12)

        f = chebfun(["x", "x-1"], domain=(0, 1, 2))
        assert np.linalg.norm(np.asarray(f(jnp.asarray([0.5, 1.5]))) - 0.5) < EPS  # pass(13)
        x = chebfun("x", domain=(0.0, 5.0))
        f = chebfun(x)
        assert list(f.domain.breakpoints) == [0, 5]                 # pass(13)
        f = chebfun(x, domain=(0.0, 2.0))
        assert list(f.domain.breakpoints) == [0, 2]                 # pass(14)

        f = chebfun(chebfun(["x", "x"], domain=(-1, 0, 1)))
        xs = jnp.asarray([-0.5, 0.5])
        assert len(f.funs) == 1 and \
            np.linalg.norm(np.asarray(f(xs)) - np.asarray(xs)) < EPS  # pass(15)

        f_op = lambda x: -x - x ** 2 + jnp.exp(-(50 * (x - 0.5)) ** 4)  # noqa: E731
        f1 = chebfun(f_op, min_samples=17)
        err1 = float(np.max(np.abs(np.asarray(f1(xx)) - np.asarray(f_op(xx)))))
        f2 = chebfun(f_op, min_samples=33)
        err2 = float(np.max(np.abs(np.asarray(f2(xx)) - np.asarray(f_op(xx)))))
        assert err1 > 1e-3 and err2 < 1e2 * f2.vscale * EPS        # pass(16)

        f_op = lambda x: jnp.sin(200 * x)  # noqa: E731
        f = chebfun(f_op, resampling=True)
        assert _happy(f)                                            # pass(17)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = chebfun(f_op, max_length=129, tech="chebtech2")
        assert not _happy(f) and len(f) == 129                      # pass(18)

        f = chebfun(f_op, splitting=True, split_length=65)
        assert _happy(f) and all(len(p) <= 65 for p in f.funs)      # pass(19)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = chebfun(f_op, splitting=True, split_length=65,
                        split_max_length=200)
        assert not _happy(f) and len(f) <= 65 * 4                   # pass(20)

        for kw in ({"tech": "chebtech1"}, {"chebkind": 1}):
            f = chebfun(lambda x: 1.0 / x, domain=(0.0, 1.0), exps=(-1, 0), **kw)
            assert _happy(f) and isinstance(
                f.funs[0].tech.smoothPart, Chebtech1)               # pass(21)-(24)

        chebfun(lambda x: jnp.array([x ** 2, jnp.sin(x) * jnp.cos(x)]),
                vectorize=True)                                     # pass(25)

        f = chebfun(jnp.sin)
        g = chebfun(jnp.sin, eps=1e-6)
        assert len(g) < len(f)                                      # pass(26)

        c = jnp.arange(1.0, 6.0)
        msg = "'coeffs' and 'chebkind' should not be specified simultaneously."
        with pytest.raises(ValueError, match=msg):
            chebfun(c, coeffs=True, chebkind=1)                     # pass(27)
        with pytest.raises(ValueError, match=msg):
            chebfun(c, chebkind=2, coeffs=True)                     # pass(28)

        f = chebfun(lambda x: jnp.cos(np.pi * x)).T
        g = chebfun(f)
        assert g.is_transposed is False                             # pass(29)
