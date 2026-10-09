import json

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._banded_lu import solve_banded

EPS = np.finfo(float).eps


def pack(a, kl, ku):
    n = len(a)
    out = np.zeros((n, 2 * kl + ku + 1), dtype=a.dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            out[i, j - i + kl] = a[i, j]
    return out


def qualify(a, kl, ku, matrix_rhs=False):
    n = len(a)
    exact = np.arange(1, n + 1, dtype=float) / (n + 1)
    if matrix_rhs:
        exact = np.column_stack((exact, exact[::-1]))
    b = a @ exact
    x, bad = solve_banded(jnp.asarray(pack(a, kl, ku)), jnp.asarray(b), lower=kl, upper=ku)
    x = np.asarray(x)
    assert not bool(bad)
    residual = np.linalg.norm(a @ x - b, np.inf)
    scale = np.linalg.norm(a, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf)
    assert residual <= 64 * n * EPS * scale
    assert np.linalg.norm(x - exact, np.inf) <= 128 * n * EPS * np.linalg.cond(a, np.inf) * max(
        1, np.linalg.norm(exact, np.inf)
    )
    return float(residual / scale)


@pytest.mark.parametrize("kl,ku", [(0, 0), (0, 2), (2, 0), (1, 1), (2, 2)])
@pytest.mark.parametrize("complex_data", [False, True])
def test_independent_bands(kl, ku, complex_data):
    rng = np.random.default_rng(431)
    n = 9
    a = np.zeros((n, n), dtype=complex if complex_data else float)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            a[i, j] = rng.normal() + (1j * rng.normal() if complex_data else 0)
    a += np.diag(2 + np.sum(abs(a), axis=1))
    qualify(a, kl, ku, matrix_rhs=True)


@pytest.mark.parametrize("complex_data", [False, True])
def test_required_row_pivots(complex_data):
    n = 10
    a = (
        np.diag(np.arange(1, n + 1, dtype=float))
        + np.diag(np.ones(n - 1), 1)
        + np.diag(3 * np.ones(n - 1), -1)
    )
    a[0, 0] = 0
    a[4, 4] = 0
    if complex_data:
        a = a.astype(complex) + 1j * np.diag(np.arange(n))
    qualify(a, 1, 1)


def test_singular_flag():
    _, bad = solve_banded(jnp.zeros((4, 4)), jnp.ones(4), lower=1, upper=1)
    assert bool(bad)


def test_actual_sphere_bands_preserved(tmp_path):
    from chebfunjax.spherefun.spherefun import _sphere_fourier_operators

    rows = []
    for m, n in [(8, 8), (16, 12)]:
        d1, d2, dn, cs, s2, _, _ = _sphere_fourier_operators(m, n)
        lap = s2 @ d2 + cs @ d1
        for mode in [1, 3]:
            a = lap - mode**2 * np.eye(m)
            off = [a[i, j] for i in range(m) for j in range(m) if abs(i - j) > 2]
            assert not np.any(off)
            assert np.any(np.diag(a, 1) != 0)
            error = qualify(a, 2, 2)
            rows.append({"m": m, "mode": mode, "backward_error": error})
    (tmp_path / "banded_results.json").write_text(json.dumps(rows, indent=2) + "\n")
