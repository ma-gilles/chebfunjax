"""JAX fitting backend contracts for native rational interpolation grids.

Provenance
----------
MATLAB source: ratinterp.m assembleMatrices/computeDenominatorCoeffs.
Chebfun commit: 7574c77
"""

import numpy as np
import pytest

import chebfunjax.utils.ratapprox as ra


@pytest.mark.parametrize("grid", ["type0", "type1", "type2", "equi"])
def test_no_host_fit_kernels(monkeypatch, grid):
    def forbidden(*args, **kwargs):
        raise AssertionError("ratinterp invoked a host FFT or SVD")
    monkeypatch.setattr(np.fft,"fft",forbidden)
    monkeypatch.setattr(np.fft,"ifft",forbidden)
    monkeypatch.setattr(np.linalg,"svd",forbidden)
    _,a,b,mu,nu,poles,_=ra.ratinterp(lambda x:(x**4-3)/((x+.2)*(x-2.2)),10,10,xi=grid,domain=(0.,2.))
    assert (mu,nu)==(4,2)
    np.testing.assert_allclose(np.sort_complex(np.asarray(poles)),[-.2,2.2],atol=1e-10,rtol=0)
