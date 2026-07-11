"""chebfun(f, exps=...) endpoint-singularity construction (Fable 5).

Wires Singfun into the chebfun factory (MATLAB's 'exps' flag), closing
the chebfun-level blowup gap for single-interval domains.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import chebfunjax as cj


class TestChebfunExps:
    def test_inverse_sqrt_weight(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1 - x * x),
                           exps=(-0.5, -0.5))
        xs = np.linspace(-0.999, 0.999, 60)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))),
                            1 / np.sqrt(1 - xs ** 2), rtol=1e-11)
        npt.assert_allclose(float(f.sum()), np.pi, atol=1e-15)

    def test_shifted_domain_right_singularity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = cj.chebfun(lambda x: (7.0 - x) ** (-0.3) * jnp.cos(x),
                           domain=(-2.0, 7.0), exps=(0.0, -0.3))
        xs = np.linspace(-1.9, 6.9, 40)
        npt.assert_allclose(np.asarray(g(jnp.asarray(xs))),
                            (7 - xs) ** (-0.3) * np.cos(xs), atol=1e-12)

    def test_fractional_root_left(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: (1 + x) ** 0.5 * jnp.exp(x),
                           exps=(0.5, 0.0))
        xs = np.linspace(-0.99, 0.99, 40)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))),
                            (1 + xs) ** 0.5 * np.exp(xs), atol=1e-12)

    def test_exps_conflicts_raise(self):
        with pytest.raises(ValueError, match="exps"):
            cj.chebfun(lambda x: x, exps=(0.0, -0.5), trig=True)
