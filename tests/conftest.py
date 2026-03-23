"""Shared test fixtures and MATLAB reference loading utilities."""

from pathlib import Path

import numpy as np
import pytest

REFS_DIR = Path(__file__).parent / "references"


def load_matlab_ref(filename: str) -> dict:
    """Load a MATLAB .mat reference file as a dict of numpy arrays.

    FAILS (not skips) if the file is missing — silent skips hide broken validation.
    Run `matlab_harness/generate_refs.m` to regenerate.
    """
    import scipy.io

    path = REFS_DIR / filename
    if not path.exists():
        pytest.fail(
            f"MATLAB reference {filename} not found at {path}.\n"
            f"Run: module load matlab/R2025b && matlab -batch "
            f"\"addpath('$CHEBFUN_REF'); run('matlab_harness/generate_refs.m')\""
        )
    data = scipy.io.loadmat(str(path), squeeze_me=True)
    return {k: v for k, v in data.items() if not k.startswith("__")}


# --- Generic fixture: loads tests/references/<module>.mat automatically ---
# Tests request `matlab_<module>` fixtures, e.g. `matlab_quadrature`.
# Each module has its OWN .mat file — no shared bottleneck.

@pytest.fixture
def matlab_quadrature():
    return load_matlab_ref("quadrature.mat")


@pytest.fixture
def matlab_chebfun_basic():
    return load_matlab_ref("chebfun_basic.mat")


@pytest.fixture
def matlab_transforms():
    return load_matlab_ref("transforms.mat")


@pytest.fixture
def matlab_diffmat():
    return load_matlab_ref("diffmat.mat")


@pytest.fixture
def matlab_aaa():
    return load_matlab_ref("aaa.mat")


@pytest.fixture
def matlab_interpolation():
    return load_matlab_ref("interpolation.mat")


@pytest.fixture
def matlab_polynomials():
    return load_matlab_ref("polynomials.mat")


@pytest.fixture
def matlab_misc():
    return load_matlab_ref("misc.mat")


@pytest.fixture(scope="session", autouse=True)
def enable_jax_float64():
    """Ensure JAX uses float64 for all tests."""
    import jax
    jax.config.update("jax_enable_x64", True)
