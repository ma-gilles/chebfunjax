"""All native tests/misc/test_jac2jac.m7574c77 assertions.

The 625-parameter sweep and three N513 cases retain native shapes, direct
reference operators, matrix infinity norms and bounds. NumPy default_rng(0)
is a deterministic fixture adapter, not MATLAB rng(0)/randn parity.
"""
import hashlib
import itertools
import json
import os
import time

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.transforms import jac2jac

from ._jac2jac_native_reference import cheb2jac_direct, jac2cheb_direct

RNG = np.random.default_rng(0)
V = RNG.standard_normal((10, 2))
LARGE_VALUES = [RNG.standard_normal((513, 2)) for _ in range(3)]
PARAMS = np.linspace(-.99, 1.1, 5)
TUPLES = list(itertools.product(PARAMS, repeat=4))
LARGE = [(.46, -.7, .56, 1.54, 2e-10),
         (-.6, -.4, -.65, -.45, 4e-10),
         (.1, -.4, .10000001, -.4, 2e-10)]


def _check(v, parameters, bound, index):
    start = time.monotonic()
    a, b, g, d = parameters
    exact = cheb2jac_direct(jac2cheb_direct(v, a, b), g, d)
    w = np.asarray(jac2jac(jnp.asarray(v), a, b, g, d))
    error = float(np.linalg.norm(exact-w, ord=np.inf))
    record = {'index': index, 'parameters': [float(p) for p in parameters],
              'shape': list(v.shape), 'input_sha256': hashlib.sha256(v.tobytes()).hexdigest(),
              'error': error, 'bound': bound, 'passed': error < bound,
              'elapsed_seconds': time.monotonic()-start}
    if path := os.environ.get('JAC2JAC_CASE_RECORDS'):
        with open(path, 'a') as stream:
            stream.write(json.dumps(record)+'\n')
    assert error < bound, record


@pytest.mark.parametrize('index', range(625), ids=lambda n: f'{n:03d}')
def test_native_sweep(index):
    _check(V, TUPLES[index], 2e-10, index)


@pytest.mark.parametrize('index', range(3))
def test_native_large(index):
    *parameters, bound = LARGE[index]
    _check(LARGE_VALUES[index], parameters, bound, 625+index)
