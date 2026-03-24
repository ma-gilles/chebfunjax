"""Integration test configuration.

Sets XLA_PYTHON_CLIENT_PREALLOCATE=false to prevent GPU OOM when running
multiple GPU-bound tests in sequence on a shared device.
"""

import os

# Prevent JAX from pre-allocating all GPU memory, so multiple tests can
# share the GPU without exhausting memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
