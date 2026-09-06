"""Port of MATLAB Chebfun tests/misc/test_conformal.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_conformal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.conformal import conformal

jax.config.update("jax_enable_x64", True)


class TestMiscConformal:
    def test_all_matlab_assertions(self):
        circle = chebfun(lambda x: jnp.exp(np.pi * 1j * x), trig=True)
        C = circle.real() + .8j * circle.imag()
        ctr = .1 + .1j
        f, finv, *_ = conformal(C, ctr)
        f2, f2inv, *_ = conformal(C, ctr, method="poly")
        z = .2 - .1j
        err = abs(complex(np.asarray(f(z))) - complex(np.asarray(f2(z))))
        errinv = abs(complex(np.asarray(finv(z))) - complex(np.asarray(f2inv(z))))
        assert err < 1e-4 and errinv < 1e-4                          # pass(1)

        C = chebfun(lambda t: jnp.exp(np.pi * 1j * t) * (1 + .2 * jnp.cos(6 * np.pi * t)),
                    trig=True)
        f, finv, *_ = conformal(C)
        Z = .5 * np.exp(1j * np.pi * np.arange(1, 101) / 100)
        err = np.linalg.norm(Z - np.asarray(finv(np.asarray(f(Z)))))
        assert err < 1e-4                                            # pass(2)

        ff = lambda z: np.arctanh(2 * z) / 1.2  # noqa: E731
        ffinv = lambda w: np.tanh(1.2 * w) / 2  # noqa: E731
        C = chebfun(lambda x: jnp.tanh(1.2 * jnp.exp(np.pi * 1j * x)) / 2, trig=True)
        f, finv, *_ = conformal(C)
        z = -.1j
        assert abs(complex(np.asarray(f(z))) - ff(z)) < 1e-4          # pass(3)
        w = .2
        assert abs(complex(np.asarray(finv(w))) - ffinv(w)) < 1e-4    # pass(4)
