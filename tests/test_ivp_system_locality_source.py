"""Routing diagnostics supplement the literal DDE source capture."""

import pytest

from chebfunjax.operators._coupled_ivp import classify_local_operator
from chebfunjax.operators.chebop import Chebop


def test_local_derivative_and_nonlinearity():
    N = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + v**2 * (1 - v).exp()], domain=(1.0, 5.0))
    assert classify_local_operator(N) is True


def test_nonlocal_source_domain_error():
    N = Chebop(
        lambda t, u, v: [u.diff() - v, v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()],
        domain=(1.0, 5.0),
    )
    N.lbc = [0.0, 1.0]
    assert classify_local_operator(N) is False
    with pytest.raises(ValueError, match="locality is not established"):
        N._solve_ivp_system(0.0)


def test_unknown_syntax_is_not_local():
    N = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + v.sqrt()], domain=(1.0, 5.0))
    N.lbc = [0.0, 1.0]
    assert classify_local_operator(N) is None
    with pytest.raises(ValueError, match="locality is not established"):
        N._solve_ivp_system(0.0)


def test_local_legacy_march():
    import numpy as np

    N = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + v], domain=(0.0, 1.0))
    N.lbc = [0.0, 1.0]
    u, v = N._solve_ivp_system(0.0)
    t = np.array([0.0, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(u(t), 1 - np.exp(-t), atol=1e-10, rtol=0)
    np.testing.assert_allclose(v(t), np.exp(-t), atol=1e-10, rtol=0)


def test_old_delay_probe_domain_error():
    """The old generic numeric probe cannot establish locality here."""
    from chebfunjax import chebfun

    N = Chebop(
        lambda t, u, v: [u.diff() - v, v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()],
        domain=(1.0, 5.0),
    )
    t = chebfun(lambda t: t, domain=(1.0, 5.0))
    base = [
        chebfun(lambda t: 0 * t + 0.3, domain=(1.0, 5.0)),
        chebfun(lambda t: 0 * t + 1.1, domain=(1.0, 5.0)),
    ]
    for fs in (base, [f + 1e-3 * (t - 3.0) for f in base]):
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:compose:domain"):
            N._call_op(t, fs)
