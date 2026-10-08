"""Preserve a source Newton failure, independently reproduced from lagpts.m.

At n3000, alpha20.3, zero-based node2975 starts11186.597863098374.
The pinned source expressions with SciPy Airy evaluation reach
11185.577531520816 after9 steps, invoking lagpts.m's count==9 error.
This diagnostic is not an actual MATLAB capture. No guard is relaxed.
"""
import jax
import pytest

from chebfunjax.utils.laguerre_rh_general import _laguerre_rh_general


def test_source_nine_iteration_guard_remains_an_error():
    with pytest.raises(jax.errors.JaxRuntimeError,
                       match="MATLAB lagpts RH Newton convergence guard"):
        _laguerre_rh_general(3000, 20.3)[0].block_until_ready()
