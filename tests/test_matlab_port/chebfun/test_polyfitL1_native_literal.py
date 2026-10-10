"""Literal eight native tests/chebfun/test_polyfitL1.m predicates.

The native second predicate intentionally retains T0..5 from its first case.
All residual sign functions, products, and integrals use the public Chebfun API.
Cases7/8 remain separately scheduled due to their long/high-degree source inputs.

Provenance
----------
MATLAB source : tests/chebfun/test_polyfitL1.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np  # uses-numpy: serialization of already-computed evidence only.
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebpoly

CASES = [
    (1, "exp_sin", (-1.0, 1.0), 5, 5, False),
    (2, "exp_sin", (-1.0, 1.0), 10, 5, False),
    (3, "exp_sin", (-2.0, 2.0), 5, 5, False),
    (4, "absolute", (-1.0, 1.0), 7, 7, True),
    (5, "absolute", (-2.0, 2.0), 7, 7, True),
    (6, "sin_cos", (-3.0, 2.0), 11, 11, False),
    (7, "sin_plus_cos", (0.0, 100.0), 11, 11, False),
    (8, "sin_plus_cos", (0.0, 100.0), 101, 101, False),
]


@pytest.mark.parametrize(
    "case,function,domain,degree,basis_degree,splitting",
    CASES,
    ids=[f"native_case_{c[0]:02}" for c in CASES],
)
def test_native_optimality(case, function, domain, degree, basis_degree, splitting):
    callbacks = {
        "exp_sin": lambda x: jnp.exp(x) * jnp.sin(10 * x),
        "absolute": jnp.abs,
        "sin_cos": lambda x: jnp.sin(x - 0.1) * jnp.cos(3 * x),
        "sin_plus_cos": lambda x: jnp.sin(x - 0.1) + jnp.cos(3 * x),
    }
    kw = {"domain": domain}
    if splitting:
        kw["splitting"] = True
    f = cj.chebfun(callbacks[function], **kw)
    start = time.monotonic()
    p = f.polyfitL1(degree)
    fit_seconds = time.monotonic() - start
    T = chebpoly(jnp.arange(basis_degree + 1), domain=domain)
    err = f - p
    moments = (T * err.sign()).sum()
    residual = float(jnp.linalg.norm(jnp.asarray(moments)))
    output = os.environ.get("POLYFIT_L1_CAPTURE_OUTPUT")
    if output:
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {"moments": np.asarray(moments)}
        for label, obj in [("input", f), ("fit", p), ("error", err)]:
            arrays[label + "_breaks"] = np.asarray(
                [obj.funs[0].interval[0]] + [piece.interval[1] for piece in obj.funs]
            )
            for i, piece in enumerate(obj.funs):
                arrays[f"{label}_piece_{i}_coeffs"] = np.asarray(piece.tech.coeffs)
        np.savez(target / f"native_case_{case:02}.npz", **arrays)
        (target / f"native_case_{case:02}.json").write_text(
            json.dumps(
                {
                    "native_case": case,
                    "function": function,
                    "domain": domain,
                    "degree": degree,
                    "basis_degree": basis_degree,
                    "splitting": splitting,
                    "fit_seconds": fit_seconds,
                    "residual_norm_2": residual,
                    "native_tolerance": 1e-8,
                    "passed": residual < 1e-8,
                },
                indent=2,
            )
            + "\n"
        )
    assert residual < 1e-8, (case, residual)
