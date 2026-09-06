"""Port of MATLAB Chebfun tests/chebfun/test_defineInterval.m (Fable 5).

MATLAB's ``f{a, b} = g`` subsasgn is ``f.define_interval((a, b), g)``.

Provenance
----------
MATLAB source : tests/chebfun/test_defineInterval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _dom(f):
    return list(f.domain.breakpoints)


def _ev(f, x):
    return np.asarray(f(jnp.asarray(x)))


class TestChebfunDefineInterval:
    def test_all_matlab_assertions(self):
        f = chebfun(lambda x: x, domain=(-1.0, 0.0, 1.0))
        f = f.define_interval((0, .5), 1.0)
        assert _dom(f) == [-1, 0, .5, 1] and _ev(f, .25) == 1       # pass(1)
        f = f.define_interval((-.5, 0), 2.0)
        assert _dom(f) == [-1, -.5, 0, .5, 1] and _ev(f, -.25) == 2  # pass(2)

        two = lambda x: jnp.stack([x, x], axis=-1)  # noqa: E731
        f = chebfun(two, domain=(-1.0, 0.0, 1.0))
        f = f.define_interval((0, .5), [1.0, 2.0])
        assert _dom(f) == [-1, 0, .5, 1] and np.array_equal(_ev(f, .25), [1, 2])  # pass(3)
        f = f.define_interval((-.5, 0), [2.0, 3.0])
        assert _dom(f) == [-1, -.5, 0, .5, 1] and np.array_equal(_ev(f, -.25), [2, 3])  # pass(4)

        f = chebfun(two, domain=(-1.0, 1.0))
        f = f.define_interval((0, .5), 1.0)
        assert _dom(f) == [-1, 0, .5, 1] and np.all(_ev(f, .25) == 1)  # pass(5)
        f = f.define_interval((-.5, 0), -1.0)
        assert _dom(f) == [-1, -.5, 0, .5, 1] and np.all(_ev(f, -.25) == -1)  # pass(6)

        f = chebfun(two, domain=(-1.0, 1.0))
        f = f.define_interval((0, .25, .5), 1.0)
        assert _dom(f) == [-1, 0, .25, .5, 1] and np.all(_ev(f, .125) == 1)  # pass(7)
        f = f.define_interval((-.5, -.1, 0), -1.0)
        assert _dom(f) == [-1, -.5, -.1, 0, .25, .5, 1] and \
            np.all(_ev(f, -.125) == -1)                              # pass(8)

        f = chebfun(lambda x: x, domain=(-1.0, 1.0))
        g = chebfun(jnp.sin, domain=(-.5, 0.0, .75))
        f = f.define_interval((-.5, .25), g)
        assert _dom(f) == [-1, -.5, 0, .25, 1]                       # pass(9)
        assert abs(_ev(f, .125) - np.sin(.125)) < 10 * EPS
        assert abs(_ev(f, .625) - .625) < 10 * EPS
        f = f.define_interval((-.5, .25), g)
        assert _dom(f) == [-1, -.5, 0, .25, 1]                       # pass(10)
        assert abs(_ev(f, .125) - np.sin(.125)) < 10 * EPS
        assert abs(_ev(f, .625) - .625) < 10 * EPS

        f = chebfun(lambda x: jnp.stack([x, -x], axis=-1), domain=(-1.0, 0.0, 1.0))
        g = chebfun(lambda x: jnp.stack([-2 * x, 2 * x], axis=-1), domain=(-.5, .5))
        f = f.define_interval((-.5, .5), g)
        assert np.linalg.norm(_ev(f, .25) - [-.5, .5]) < 10 * EPS   # pass(11)
        f = f.define_interval((-.5, .5), g)
        assert np.linalg.norm(_ev(f, .25) - [-.5, .5]) < 10 * EPS   # pass(12)

        f = chebfun(lambda x: x)
        f = f.define_interval((-.5, .5), None)
        assert _dom(f) == [-1, -.5, 0]                                # pass(13)
        f = chebfun(lambda x: x)
        f = f.define_interval((.5, 1), None)
        assert _dom(f) == [-1, .5]                                    # pass(14)
