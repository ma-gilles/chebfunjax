"""Small original NUFFT/INUFFT source predicates; no tolerance changes.
NumPyMT RNG adapter is not a nativeMATLAB matching-draw claim.

Provenance
----------
Adapted from tests/misc/test_nufft.m and test_inufft.m at MATLAB Chebfun
commit 7574c77680d7e82b79626300bf255498271a72df.
Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
of Oxford, and the Chebfun Developers. All rights reserved.
See LICENSE.txt for the Chebfun license.
"""

import numpy as np

from chebfunjax.utils.nufft import inufft, nufft

EPS = np.finfo(float).eps


def run_source_cases():
    rows = []
    rng = np.random.RandomState(0)

    def compare(name, fn, expected, c, mult, N, norm="inf"):
        try:
            got = np.asarray(fn())
            assert got.shape == expected.shape
            error = float(np.linalg.norm(got - expected, ord=np.inf if norm == "inf" else 2))
            bound = float(mult * N * EPS * np.linalg.norm(c, ord=1))
            assert error < bound, (error, bound)
            row = {
                "name": name,
                "status": "pass",
                "error": error,
                "bound": bound,
                "norm": norm,
                "size": N,
            }
        except Exception as e:
            row = {"name": name, "status": "fail", "exception": repr(e), "size": N}
        rows.append(row)
        print(row, flush=True)

    for N in [1, 10, 100, 1000, 10000]:
        omega = np.arange(N) + 1.2 * rng.rand(N) / N
        c = rng.rand(N) + 1j * rng.rand(N)
        if N > 100:
            continue
        exact = np.array([np.exp(-2j * np.pi * j / N * omega) @ c for j in range(N)])
        compare("forwardI_N" + str(N), lambda: nufft(c, omega / N, nufft_type=1), exact, c, 300, N)
    N = 50
    omega = N * rng.randn(N)
    c = rng.rand(N) + 1j * rng.rand(N)
    F = np.exp(-2j * np.pi * (np.arange(N) / N)[:, None] * omega[None, :])
    compare("forwardI_random50", lambda: nufft(c, omega / N, nufft_type=1), F @ c, c, 300, N, "2")
    for N in [1, 10, 100, 1000, 10000]:
        x = np.linspace(0, 1, N + 1)[:-1] + 1.2 / N
        c = rng.rand(N) + 1j * rng.rand(N)
        if N > 100:
            continue
        exact = np.array([np.exp(-2j * np.pi * t * np.arange(N)) @ c for t in x])
        compare("forwardII_N" + str(N), lambda: nufft(c, x), exact, c, 300, N)
    N = 50
    x = 100 * rng.rand(N)
    c = rng.rand(N) + 1j * rng.rand(N)
    F = np.exp(-2j * np.pi * x[:, None] * np.arange(N)[None, :])
    compare("forwardII_random50", lambda: nufft(c, x), F @ c, c, 300, N, "2")
    # Advance native typeIII draw sequence without exercising missing API.
    for N in [1, 10, 100, 1000, 10000]:
        rng.rand(N)
        rng.rand(N)
        rng.rand(N)
    n, m = 101, 3
    x = np.linspace(0, 1, m + 1)[:-1] + 1.01 / m
    c = rng.rand(n)
    exact = np.array([np.exp(-2j * np.pi * t * np.arange(n)) @ c for t in x])
    compare("forwardII_rect101x3", lambda: nufft(c, x), exact, c, 100, 1)
    rng = np.random.RandomState(0)
    for N in [1, 10, 100, 1000, 4000]:
        omega = np.arange(N) + 0.4 * rng.rand(N)
        c = rng.rand(N) + 1j * rng.rand(N)
        if N > 100:
            continue
        F = np.exp(-2j * np.pi * (np.arange(N) / N)[:, None] * omega[None, :])
        exact = np.linalg.solve(F, c)
        compare("inverseI_N" + str(N), lambda: inufft(c, omega / N, nufft_type=1), exact, c, 300, N)
    for N in [1, 10, 100, 1000, 4000]:
        x = np.linspace(0, 1, N + 1)[:-1] + 0.2 * rng.rand(N) / N
        c = rng.rand(N) + 1j * rng.rand(N)
        if N > 100:
            continue
        F = np.exp(-2j * np.pi * x[:, None] * np.arange(N)[None, :])
        exact = np.linalg.solve(F, c)
        compare("inverseII_N" + str(N), lambda: inufft(c, x, nufft_type=2), exact, c, 300, N)
    N = 100
    x = np.linspace(0, 1, N + 1)[:-1] + 0.2 * rng.rand(N) / N
    c = rng.rand(N) + 1j * rng.rand(N)
    F = np.exp(-2j * np.pi * x[:, None] * np.arange(N)[None, :])
    compare("inverseII_default100", lambda: inufft(c, x), np.linalg.solve(F, c), c, 300, N)
    return rows


def test_small_native_nufft_inufft_predicates():
    rows = run_source_cases()
    assert len(rows) == 16
    assert all(row["status"] == "pass" for row in rows), rows
