"""Native shorthand assembly and independent polynomial forward controls."""

import hashlib
import inspect
import json
from pathlib import Path

import jax.numpy as jnp

# uses-numpy: native fixture decoding and independent assertions only.
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d._pde_ndf import pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s

FIXTURE = Path(__file__).parent / "references" / "pde_shorthand_native"


def unwrap(v):
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.reshape(-1, order="F")[0]
    return v


def field(v, name):
    return unwrap(getattr(unwrap(v), name))


def vec(v):
    return jnp.asarray(unwrap(v)).reshape(-1, order="F")


def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) <= 100 * np.finfo(float).eps * max(1.0, np.max(np.abs(b)))


def assembly(fun):
    full = inspect.getclosurevars(fun).nonlocals["fun"]
    scope = inspect.getclosurevars(full).nonlocals
    raw = scope["raw"]
    # Scalar raw retains an unused system-only closure cell; read only bound cells.
    cells = dict(zip(raw.__code__.co_freevars, raw.__closure__))
    selected = {
        name: cells[name].cell_contents
        for name in ("projection", "boundary_matrix", "linear_rhs", "coordinate")
    }
    return full, raw, selected, scope["offset"]


def initial(kind, domain):
    a, b = domain
    m = (a + b) / 2
    h = (b - a) / 2
    coeffs = []
    for s, c in zip((1, 2, 3), (1, -2, 3)):
        if kind == "neumann":
            coeffs.append(
                [s + c * m + s * (m * m + h * h / 2), c * h + 2 * s * m * h, s * h * h / 2]
            )
        else:
            coeffs.append([-s * h * h / 2, 0.0, s * h * h / 2])
    return Chebfun.from_coeffs(jnp.asarray(coeffs).T, domain=domain)


def rhs(kind):
    if kind == "neumann":
        return lambda u, v, w: (0.1 * u.diff(2), 0.2 * v.diff(2), 0.3 * w.diff(2))
    return lambda t, x, u, v, w: (
        0.1 * u.diff(2) - u - 0.2 * jnp.exp(-t),
        0.2 * v.diff(2) - 2 * v - 0.8 * jnp.exp(-2 * t),
        0.3 * w.diff(2) - 3 * w - 1.8 * jnp.exp(-3 * t),
    )


@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
@pytest.mark.parametrize("domain_index,domain", [(1, (-1.0, 1.0)), (2, (2.0, 6.0))])
def test_native_arrays_and_all_public_samples(monkeypatch, kind, domain_index, domain):
    hashes = json.loads((FIXTURE / "FIXTURE_HASHES.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    meta = json.loads((FIXTURE / "metadata.json").read_text())
    assert meta["complete"] and meta["case_count"] == 4
    root = FIXTURE / f"{kind}_domain{domain_index}"
    assert json.loads((root / "complete.json").read_text())["complete"]
    kw = dict(mat_dtype=True, squeeze_me=False, struct_as_record=False)
    record = unwrap(loadmat(root / "solver_capture.mat", **kw)["record"])
    public = loadmat(root / "public_output.mat", **kw)
    for name in ("samples", "analytic"):
        assert public[name].shape == (3, 3)
        for cell in public[name].flat:
            assert np.asarray(unwrap(cell)).shape == (3, 1)
            assert np.all(np.isfinite(unwrap(cell)))
    context = field(record, "context")
    probe = field(record, "probe")
    original = pde_driver.initialize_full_mass
    seen = []

    def observed(fun, mass, y, **kwargs):
        full, raw, scope, offset = assembly(fun)
        seen.append(True)
        assert mass.shape == (27, 27) and scope["boundary_matrix"].shape == (6, 27)
        close(mass, field(context, "M"))
        close(scope["projection"], field(context, "P"))
        close(scope["boundary_matrix"], field(context, "B"))
        close(scope["linear_rhs"], vec(field(context, "q")))
        close(offset, vec(field(context, "BCVALOFFSET")))
        close(y, vec(field(record, "initial")))
        # Match both state and offset context before comparing the native operator.
        # Same-input diagnostic: Neumann[2,6] raw RHS is exact; initial sampling
        # differs by1.42e-14 and D2 amplifies it. Bound remains100eps unchanged.
        native_y = vec(field(record, "initial"))
        native_offset = vec(field(context, "BCVALOFFSET"))
        initial_same_context = (
            raw(float(vec(field(record, "span"))[0]), native_y).at[:6].add(-native_offset)
        )
        close(initial_same_context, vec(field(record, "initial_rhs")))
        probe_same_context = (
            raw(float(vec(field(probe, "time"))[0]), vec(field(probe, "state")))
            .at[:6]
            .add(-native_offset)
        )
        close(probe_same_context, vec(field(probe, "rhs")))
        return original(fun, mass, y, **kwargs)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", observed)
    times = vec(public["requested_times"])
    outputs = pde15s(
        rhs(kind),
        times,
        initial(kind, domain),
        kind,
        n=9,
        rtol=float(vec(field(record, "RelTol"))[0]),
        atol=float(vec(field(record, "AbsTol"))[0]),
    )
    assert seen == [True] and len(outputs) == 3
    close(times, vec(field(record, "primary_t")))
    x = vec(public["sample_x"])
    for j, out in enumerate(outputs):
        native = jnp.column_stack([vec(public["samples"][i, j]) for i in range(3)])
        expected = jnp.column_stack([vec(public["analytic"][i, j]) for i in range(3)])
        actual = out(x)
        assert actual.shape == native.shape == expected.shape == (3, 3)
        assert float(jnp.max(jnp.abs(actual - native))) < 1e-7
        # Default-tolerance native Dirichlet error is2.92e-6/1.17e-5 against
        # the exact solution. Compare native parity here; analytic accuracy is
        # qualified separately at the established manufactured tolerances below.


@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
@pytest.mark.parametrize("domain", [(-1.0, 1.0), (2.0, 6.0)])
def test_shorthand_manufactured_accuracy_at_established_tolerances(kind, domain):
    times = [0.0, 0.005, 0.01]
    outputs = pde15s(rhs(kind), times, initial(kind, domain), kind, n=9, rtol=1e-9, atol=1e-11)
    a, b = domain
    x = a + (b - a) * jnp.asarray([0.125, 0.5, 0.875])
    for t, output in zip(times, outputs):
        if kind == "neumann":
            slopes = [1, -2, 3]
            diffusion = [0.1, 0.2, 0.3]
            expected = jnp.column_stack(
                [
                    k + slopes[k - 1] * x + k * x * x + 2 * k * diffusion[k - 1] * t
                    for k in (1, 2, 3)
                ]
            )
        else:
            expected = jnp.column_stack(
                [k * (x - a) * (x - b) * jnp.exp(-k * t) for k in (1, 2, 3)]
            )
        assert float(jnp.max(jnp.abs(output(x) - expected))) < 1e-7
