"""Port of MATLAB Chebfun tests/deltafun/test_cumsum.m (Opus 4.8).

In MATLAB, integrating a Deltafun turns each delta into a Heaviside jump, so
``cumsum`` returns a (piecewise) chebfun — wrapped in a cell — rather than a
Deltafun.  chebfunjax's ``Deltafun.cumsum`` instead returns another Deltafun
whose funPart absorbs the Heaviside contributions using the ``x >= loc``
convention, so both the return type and the endpoint values diverge from
MATLAB.  These divergences are captured as xfails.

Provenance
----------
MATLAB source : tests/deltafun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DELTA_TOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))


class TestDeltafunCumsum:
    def test_cumsum_empty(self):
        # pass(1): F = cumsum(deltafun()); isempty(F)
        assert Deltafun.empty().cumsum().isempty()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.xfail(
        reason="chebfunjax Deltafun.cumsum returns a Deltafun, not a cell array "
        "of chebfuns as MATLAB does"
    )
    def test_cumsum_returns_cell(self):
        # pass(2): F = cumsum(d); iscell(F)
        f = Bndfun.from_function(jnp.exp, DOM)
        mag = np.random.rand(5, 5)
        loc = np.sort(np.random.rand(5))
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        F = d.cumsum()
        assert isinstance(F, list)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.xfail(
        reason="chebfunjax Deltafun.cumsum returns a Deltafun (not ~isa deltafun) "
        "and evaluates the right-endpoint Heaviside with x >= loc, so "
        "feval(F, 1) is ~0 instead of the MATLAB left-limit value -1"
    )
    def test_cumsum_heaviside_endpoints(self):
        # pass(3): ~isa(F,'deltafun') && feval(F,-1)~=-1 && feval(F,1)~=-1
        f = Bndfun.from_function(lambda x: jnp.sin(jnp.pi * x), DOM)
        d = Deltafun(f, jnp.array([-1.0, 1.0]), jnp.array([-1.0, 1.0]))
        F = d.cumsum()
        assert not isinstance(F, Deltafun)
        assert abs(float(F(jnp.float64(-1.0))) - (-1.0)) < DELTA_TOL
        assert abs(float(F(jnp.float64(1.0))) - (-1.0)) < DELTA_TOL
