"""Shared test fixtures and MATLAB reference loading utilities."""

import os
from pathlib import Path

import numpy as np
import pytest

REFS_DIR = Path(__file__).parent / "references"


def load_matlab_ref(filename: str) -> dict:
    """Load a MATLAB .mat reference file as a dict of numpy arrays.

    Uses scipy.io.loadmat, squeezing out MATLAB's extra dimensions.
    """
    import scipy.io

    path = REFS_DIR / filename
    if not path.exists():
        pytest.skip(f"MATLAB reference {filename} not found — run generate_refs.m first")
    data = scipy.io.loadmat(str(path), squeeze_me=True)
    # Remove MATLAB metadata keys
    return {k: v for k, v in data.items() if not k.startswith("__")}


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


@pytest.fixture(scope="session", autouse=True)
def enable_jax_float64():
    """Ensure JAX uses float64 for all tests."""
    import jax
    jax.config.update("jax_enable_x64", True)
