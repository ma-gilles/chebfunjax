"""Scalar and array source power assertions with the local MATLAB test oracle.

Source assertions: 1--5, 7, 9, 11, 13--15, 17. The test file's local normest
reseeds per call and uses the induced matrix infinity norm. NumPy MT19937 is
a deterministic Python stream adapter; MATLAB sample equality is unverified.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m (assertions and local normest)
MATLAB APIs   : @chebfun/power.m, @chebfun/dimCheck.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original: Copyright 2017 by The University of Oxford and Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)

def _local_normest(fun, domain=None):
    """Translate the test file's local normest helper, including its reseed."""
    rng = np.random.RandomState(6178)
    if domain is None:
        points = 2.0 * rng.rand(100) - 1.0
    else:
        points = float(sum(domain)) * rng.rand(10) - float(domain[0])
    values = np.asarray(fun(jnp.asarray(points)))
    if values.ndim <= 1:
        return float(np.max(np.abs(values), initial=0.0))
    return float(np.linalg.norm(values, ord=np.inf))

def _array_fun(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1)

def _array_chebfun():
    return cj.chebfun(_array_fun)

def test_source_pass01_scalar_zero_power():
    f = cj.chebfun(jnp.sin)
    g = f ** 0
    assert _local_normest(g - 1) < 10 * EPS

def test_source_pass02_scalar_one_power():
    f = cj.chebfun(jnp.sin)
    g = f ** 1
    assert _local_normest(g - f) < 10 * EPS

def test_source_pass03_scalar_square():
    f = cj.chebfun(jnp.sin)
    g = f ** 2
    h = cj.chebfun(lambda x: jnp.sin(x) ** 2)
    assert _local_normest(g - h) < 10 * EPS

def test_source_pass04_scalar_cube():
    f = cj.chebfun(jnp.sin)
    g = f ** 3
    h = cj.chebfun(lambda x: jnp.sin(x) ** 3)
    assert _local_normest(g - h) < 10 * EPS

def test_source_pass05_array_zero_power():
    f = _array_chebfun()
    g = f ** 0
    assert min(g.size()) == 3
    assert _local_normest(g - 1) < EPS

def test_source_pass07_array_one_power():
    f = _array_chebfun()
    g = f ** 1
    assert min(g.size()) == 3
    assert _local_normest(g - f) < EPS

def test_source_pass09_array_square():
    f = _array_chebfun()
    g = f ** 2
    h = cj.chebfun(lambda x: jnp.stack([jnp.sin(x) ** 2, jnp.cos(x) ** 2, -jnp.exp(2 * x)], axis=-1))
    assert min(g.size()) == 3
    assert _local_normest(g - h) < 10 * float(h.vscale) * EPS

def test_source_pass11_array_cube():
    f = _array_chebfun()
    g = f ** 3
    h = cj.chebfun(lambda x: jnp.stack([jnp.sin(x) ** 3, jnp.cos(x) ** 3, -1j * jnp.exp(3 * x)], axis=-1))
    assert min(g.size()) == 3
    assert _local_normest(g - h) < 10 * float(h.vscale) * EPS

def test_source_pass13_scalar_one_to_chebfun():
    f = cj.chebfun(jnp.sin)
    g = 1.0 ** f
    assert _local_normest(g - 1) < 10 * EPS

def test_source_pass14_complex_scalar_to_chebfun():
    f = cj.chebfun(jnp.sin)
    g = 2j ** f
    h = cj.chebfun(lambda x: 2j ** jnp.sin(x))
    assert _local_normest(g - h) < 10 * EPS

def test_source_pass15_scalar_one_to_array_chebfun():
    f = _array_chebfun()
    g = 1.0 ** f
    assert min(g.size()) == 3
    assert _local_normest(g - 1) < 10 * EPS

def test_source_pass17_complex_scalar_to_array_chebfun():
    f = _array_chebfun()
    g = 2j ** f
    h = cj.chebfun(lambda x: jnp.stack([2j ** jnp.sin(x), 2j ** jnp.cos(x), 2j ** (1j * jnp.exp(x))], axis=-1))
    assert min(g.size()) == 3
    assert _local_normest(g - h) < 100 * EPS
