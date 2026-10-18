"""CPU source ASY public values/storage from pinned native hermpts.m.

Chebfun 7574c77680d7e82b79626300bf255498271a72df, MATLAB R2025b.
Original complex-modulus bounds and source defects are preserved. Each test
owns a fresh ordinary worker so all-order CI does not accumulate executables.
The worker timeout follows the existing 900-second full/Matlab CI test limit.
"""

import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "hermite_small_asy_matlab_storage.json").read_text()
)
BOUNDS = dict(x=8.881784197001252e-15, w=2.220446049250313e-14, v=2.220446049250313e-13)


def _cache_snapshot(cache):
    return {
        str(path.relative_to(cache)): dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in cache.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="session")
def hermite_executable_cache(tmp_path_factory):
    """Own ordinary executable artifacts for fresh workers, never result arrays."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    sources = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*.py")
    }
    bindings = dict(
        source_hashes=sources,
        versions={name: metadata.version(name) for name in ("jax", "jaxlib", "ml_dtypes")},
        configuration=dict(
            backend="cpu",
            x64=True,
            machine=platform.machine(),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            compilation_cache_enabled=os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "true"),
            minimum_compile_seconds=1,
            minimum_entry_bytes=1,
            environment={
                key: value
                for key, value in os.environ.items()
                if key.startswith(("JAX_", "XLA_", "OMP_", "OPENBLAS_"))
                and key != "JAX_COMPILATION_CACHE_DIR"
            },
        ),
    )
    namespace = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    owned = tmp_path_factory.mktemp("hermite-executables") / namespace
    cache = owned / "executables"
    cache.mkdir(parents=True)
    (owned / "SOURCE_CONFIG.json").write_text(json.dumps(bindings, indent=2))
    initial = _cache_snapshot(cache)
    assert not initial
    print(
        json.dumps(
            dict(
                hermite_executable_cache_owner=str(cache),
                source_config_sha256=namespace,
                initial_executable_entries=0,
                source_files=len(sources),
                policy="ordinary JAX executable artifacts only; no scientific output memoization",
            )
        ),
        flush=True,
    )
    return cache, namespace


def _decode(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _word(value):
    value = complex(value)
    return [struct.pack(">d", value.real).hex(), struct.pack(">d", value.imag).hex()]


def _check(ref, records):
    import jax.numpy as jnp

    from chebfunjax.utils.quadrature import hermpts

    if not ref["success"]:
        assert ref["n"] == 3 and ref["error"] == "MATLAB:catenate:dimensionMismatch"
        with pytest.raises(ValueError, match="concatenation"):
            hermpts(ref["n"], ref["kind"], method=ref["method"].lower(), bary=ref["bary"])
        records.append(dict(n=3, source_error="concatenation"))
        return
    actual = hermpts(ref["n"], ref["kind"], method=ref["method"].lower(), bary=ref["bary"])
    for name, value in zip(("x", "w", "v"), actual):
        record = ref[name]
        expected = jnp.asarray(
            [
                complex(_decode(a), _decode(b))
                for a, b in zip(record["real_hex"], record["imag_hex"])
            ],
            jnp.complex128,
        )
        assert value.shape == expected.shape == (ref["n"],)
        assert (not jnp.iscomplexobj(value)) == record["isreal"]
        assert record["class"] == "double"
        assert value.dtype == (jnp.float64 if record["isreal"] else jnp.complex128)
        assert value.nbytes == record["bytes"]
        assert bool(jnp.all(jnp.isfinite(value)))
        error = float(jnp.max(jnp.abs(value - expected))) if value.size else 0.0
        assert math.isfinite(error) and error <= BOUNDS[name]
        records.append(
            dict(
                n=ref["n"],
                method=ref["method"],
                kind=ref["kind"],
                bary=ref["bary"],
                name=name,
                dtype=str(value.dtype),
                native_isreal=record["isreal"],
                complex_modulus_error=error,
                bound=BOUNDS[name],
                words=[_word(q) for q in value],
            )
        )


def _worker(mode, n):
    import time

    started = time.monotonic()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    records = []
    if mode == "base":
        _check(next(r for r in REFERENCE["base_cases"] if r["n"] == n), records)
    elif mode == "variants":
        for ref in REFERENCE["variants"]:
            if ref["n"] == n and ref["method"] == "ASY":
                _check(ref, records)
    elif mode == "default":
        for ref in REFERENCE["variants"]:
            if ref["method"] == "default" or ref["n"] <= 1:
                _check(ref, records)
    elif mode == "storage":
        from chebfunjax.utils._hermite_asy_small import _computed_result_storage
        from chebfunjax.utils._source_complex import eager_binary

        # Constructors remain complex until a computed operation boundary.
        value = jnp.asarray([complex(1, 0), complex(2, -0.0)], jnp.complex128)
        assert jnp.iscomplexobj(value)
        assert not jnp.iscomplexobj(_computed_result_storage(value))
        for raw in (
            complex(1, 1e-300),
            complex(float("nan"), 0),
            complex(0, float("nan")),
            complex(float("inf"), 0),
        ):
            assert jnp.iscomplexobj(_computed_result_storage(jnp.asarray(raw)))
        for operation in range(4):
            arithmetic = jax.jit(lambda a, b: eager_binary(a, b, operation))
            operand = jnp.asarray([complex(2, 0), complex(3, 0)])
            eager = eager_binary(value, operand, operation)
            compiled = arithmetic(value, operand)
            assert bool(jnp.all(compiled == eager))
            assert not jnp.iscomplexobj(_computed_result_storage(compiled))
        records.append(dict(storage_components_pass=True, nonfinite_native_storage="unqualified"))
    else:
        raise AssertionError(mode)
    print(
        json.dumps(
            dict(
                mode=mode,
                n=n,
                backend=jax.default_backend(),
                pid=os.getpid(),
                ppid=os.getppid(),
                affinity=sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                compilation_cache_directory=str(jax.config.jax_compilation_cache_dir),
                cache_source_config=os.environ.get("HERMITE_WORKER_CACHE_SOURCE_CONFIG"),
                compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
                elapsed_seconds=time.monotonic() - started,
                records=records,
            )
        ),
        flush=True,
    )


def _fresh_worker(mode, n, executable_cache):
    cache, namespace = executable_cache
    before = _cache_snapshot(cache)
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])['_worker'](sys.argv[2],int(sys.argv[3]))",
        str(Path(__file__).resolve()),
        mode,
        str(n),
    ]
    # CPU fixture qualification; inherit affinity and ordinary JAX cache settings.
    completed = subprocess.run(
        command,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
            JAX_COMPILATION_CACHE_DIR=str(cache),
            HERMITE_WORKER_CACHE_SOURCE_CONFIG=namespace,
            JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
            JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="1",
        ),
        capture_output=True,
        text=True,
        timeout=900,
    )
    print(
        json.dumps(
            dict(
                worker_mode=mode,
                n=n,
                source_config_sha256=namespace,
                executable_cache_before=before,
                executable_cache_after=_cache_snapshot(cache),
            )
        ),
        flush=True,
    )
    print(completed.stdout, end="")
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("n", range(21))
def test_small_asy_complete_public(n, hermite_executable_cache):
    _fresh_worker("base", n, hermite_executable_cache)


@pytest.mark.parametrize("n", [0, 1, 2, 4, 6, 10, 20])
def test_small_asy_public_variants(n, hermite_executable_cache):
    _fresh_worker("variants", n, hermite_executable_cache)


def test_default_and_trivial_predispatch(hermite_executable_cache):
    _fresh_worker("default", 0, hermite_executable_cache)


def test_computed_storage_component_boundaries(hermite_executable_cache):
    _fresh_worker("storage", 0, hermite_executable_cache)
