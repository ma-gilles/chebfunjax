"""Shared test fixtures and MATLAB reference loading utilities."""

from pathlib import Path

import pytest

REFS_DIR = Path(__file__).parent / "references"


def load_matlab_ref(filename: str) -> dict:
    """Load a MATLAB .mat reference file as a dict of numpy arrays.

    Golden refs are committed to tests/references/. If a ref is missing,
    the test FAILS (not skips) — correctness must not silently degrade.

    To regenerate (requires MATLAB):
        source project.conf
        module load $MATLAB_MODULE
        matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/generate_refs.m')"
    """
    import scipy.io

    path = REFS_DIR / filename
    if not path.exists():
        pytest.fail(
            f"MATLAB golden ref {filename} not found at {path}.\n"
            f"Golden refs should be committed. If this is a new module,\n"
            f"generate refs and commit them. See docs/testing.md."
        )
    data = scipy.io.loadmat(str(path), squeeze_me=True)
    return {k: v for k, v in data.items() if not k.startswith("__")}


@pytest.fixture
def matlab_ref(request):
    """Generic MATLAB reference fixture, keyed by module name.

    Usage in tests:
        @pytest.mark.matlab
        @pytest.mark.parametrize("matlab_ref", ["quadrature"], indirect=True)
        def test_vs_matlab(self, matlab_ref):
            ref = matlab_ref  # dict of numpy arrays from quadrature.mat

    Or request a specific ref file directly:
        @pytest.mark.matlab
        def test_something(self, matlab_ref):
            ...
        # parametrize at class or module level
    """
    module_name = request.param
    return load_matlab_ref(f"{module_name}.mat")


# --- Legacy named fixtures (for existing tests; new tests should use matlab_ref) ---

@pytest.fixture
def matlab_quadrature():
    return load_matlab_ref("quadrature.mat")


@pytest.fixture
def matlab_interpolation():
    return load_matlab_ref("interpolation.mat")


@pytest.fixture(scope="session", autouse=True)
def enable_jax_float64():
    """Ensure JAX uses float64 for all tests."""
    import jax
    jax.config.update("jax_enable_x64", True)
