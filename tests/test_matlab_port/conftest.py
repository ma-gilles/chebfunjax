"""Port-tree pytest configuration.

JAX's in-process jit-compile cache grows monotonically across the
thousands of port tests and eventually OOMs the 7 GB GitHub CI
runners (jobs died with SIGTERM / 'operation was canceled' and zero
failing tests).  Clearing the caches every few dozen tests bounds the
footprint at a negligible recompilation cost.
"""

from __future__ import annotations

import jax
import pytest

_COUNTER = {"n": 0}
_CLEAR_EVERY = 25


@pytest.fixture(autouse=True)
def _clear_jax_caches_periodically(request):
    yield
    _COUNTER["n"] += 1
    # The chebfun3 constructor jit-compiles many distinct shapes per
    # test (adaptive n growth + restarts) -- its shard OOM'd CI even
    # with every-25 clearing, so clear after every chebfun3* test.
    heavy = "chebfun3" in str(
        getattr(request.node, "fspath", ""))
    if heavy or _COUNTER["n"] % _CLEAR_EVERY == 0:
        jax.clear_caches()
