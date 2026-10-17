"""Eager ownership of the explicit JAX MATLAB uniform stream.

The shared stream currently serves the native dense condition estimator.
Other library random generators have separate, still unqualified stream APIs.

Provenance
----------
MATLAB source: rng/rand state progression; R2025b normest1.m random signs.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

from threading import RLock

import jax

from ._matlab_twister import init_matlab_twister, matlab_rand_uniform

_draw = jax.jit(matlab_rand_uniform, static_argnames=("shape",))


class MatlabUniformStream:
    """Own a persistent uniform state, initialized lazily to MATLAB's seed zero.

    Explicit states permit matched native replay. Each draw commits the next
    state immediately, including when a later numerical operation raises.

    Provenance: MATLAB rng/rand; pinned Chebfun7574c77 consumers.
    """

    def __init__(self, state=None):
        self._state = state
        self._lock = RLock()

    @property
    def state(self):
        with self._lock:
            if self._state is None:
                self._state = init_matlab_twister(0)
            return self._state

    def uniform(self, shape):
        with self._lock:
            values, self._state = _draw(self.state, shape)
            return values


default_uniform_stream = MatlabUniformStream()
