"""Port of MATLAB Chebfun tests/chebop/test_carrier_US.m (Fable 5).

Carrier equation under the ultraS discretization (Newton steps solved
via operators/chebop_altdisc).

Provenance
----------
MATLAB source : tests/chebop/test_carrier_US.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_matlab_port.chebop.test_carrier_C1_matlab import (
    HIQUALITY,
    TOL,
    _solve_carrier,
)

jax.config.update("jax_enable_x64", True)


class TestChebopCarrierUS:
    @pytest.mark.timeout(880)
    def test_all_matlab_assertions(self):
        u = _solve_carrier("ultraS", 384)
        xx = jnp.asarray(np.arange(-1, 1.01, 0.25))
        assert float(np.linalg.norm(np.asarray(u(xx))
                                    - HIQUALITY)) < TOL  # pass(1)
        ends = jnp.asarray([-1.0, 1.0])
        assert float(np.linalg.norm(np.asarray(u(ends)))) < TOL
