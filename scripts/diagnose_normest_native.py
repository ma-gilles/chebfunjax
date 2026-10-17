"""Diagnose native normest decisions without changing production or assertions.

Run both captured-LU actions and exact native-output tape; archive decision
locals before original assertions. This is a CPU diagnostic, not qualification
of hardware equivalence. Native fixture: pinned R2025b positive_inverse6.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path


def run(D):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    P = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(P / "src"))
    sys.path.insert(0, str(P / "tests/test_utils"))
    import jax
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.__version__ == jaxlib.__version__ == "0.11.0" and jax.default_backend() == "cpu"
    import jax.numpy as jnp
    import numpy as np

    import chebfunjax

    assert Path(chebfunjax.__file__).resolve().is_relative_to(P / "src")
    import test_normest1_native_source as T

    from chebfunjax.utils._normest1 import normest1_real

    case = T._load("positive_inverse6")
    entries = T._trace_entries(T._field(case, "trace"))
    arrays = {}
    report = {
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "origin": chebfunjax.__file__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpu": Path("/proc/cpuinfo").read_text(),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_env": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "jaxlib_origin": jaxlib.__file__,
            "numpy_version": np.__version__,
            "numpy_origin": np.__file__,
            "head": subprocess.check_output(
                ["git", "-C", str(P), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "runs": {},
        "native_sums": [],
    }
    import contextlib
    import io

    import scipy

    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
        scipy.show_config()
    report["environment"]["numpy_scipy_config"] = config.getvalue()
    report["environment"]["scipy_version"] = scipy.__version__
    report["environment"]["scipy_origin"] = scipy.__file__
    report["environment"]["loaded_libraries"] = Path("/proc/self/maps").read_text()
    arrays["native_v"] = np.asarray(T._field(case, "v"))
    arrays["native_w"] = np.asarray(T._field(case, "w"))
    for i, e in enumerate(entries):
        y = T._trace_array(e, "output")
        arrays[f"native_{i}_input"] = T._trace_array(e, "input")
        arrays[f"native_{i}_output"] = y
        if T._trace_flag(e) == "notransp":
            seq = np.zeros(y.shape[1])
            for row in np.abs(y):
                seq = seq + row
            js = np.asarray(jnp.sum(jnp.abs(jnp.asarray(y)), axis=0))
            report["native_sums"].append(
                {
                    "call": i,
                    "sequential_hex": [float(v).hex() for v in seq],
                    "jax_hex": [float(v).hex() for v in js],
                }
            )
    for mode in ("captured_lu", "native_tape"):
        trace = []
        events = []
        counter = [0]
        keys = (
            "iteration",
            "products",
            "x",
            "y",
            "norms",
            "ranked",
            "best",
            "estimate",
            "estimate_old",
            "estimate_column",
            "indices",
            "index_history",
            "w",
            "signs",
            "z",
            "scores",
            "reason",
        )

        def observe(frame, event, arg):
            if frame.f_code is normest1_real.__code__ and event in ("line", "return"):
                snap = {"line": frame.f_lineno, "event": event}
                idx = len(events)
                for k in keys:
                    if k not in frame.f_locals:
                        continue
                    value = frame.f_locals[k]
                    if isinstance(value, (int, float, str, list, tuple)):
                        snap[k] = value
                    elif hasattr(value, "shape"):
                        name = f"{mode}_event{idx}_{k}"
                        arrays[name] = np.asarray(value).copy()
                        snap[k] = name
                events.append(snap)
            return observe

        if mode == "captured_lu":
            forward, transpose = T._lu_actions(T._field(case, "L"), T._field(case, "U"), trace)
        else:

            def tape(flag, x):
                i = counter[0]
                assert i < len(entries), "extra operator call"
                e = entries[i]
                actual = np.asarray(x)
                expected = T._trace_array(e, "input")
                assert flag == T._trace_flag(e), f"flag mismatch call{i}"
                assert (
                    actual.shape == expected.shape
                    and actual.dtype == expected.dtype
                    and actual.tobytes() == expected.tobytes()
                ), f"input bytes mismatch call{i}"
                y = T._trace_array(e, "output")
                counter[0] += 1
                trace.append((flag, actual.copy(), y.copy()))
                return jnp.asarray(y)

            def forward(x):
                return tape("notransp", x)

            def transpose(x):
                return tape("transp", x)

        result = None
        owner = None
        error = None
        try:
            sys.settrace(observe)
            result, owner = T._run_normest(case, forward, transpose)
        except Exception as exc:
            error = repr(exc)
        finally:
            sys.settrace(None)
        report["environment"]["loaded_libraries_after_" + mode] = Path(
            "/proc/self/maps"
        ).read_text()
        for i, (flag, x, y) in enumerate(trace):
            arrays[f"{mode}_call{i}_x"] = x
            arrays[f"{mode}_call{i}_y"] = y
        out = {"events": events, "call_flags": [t[0] for t in trace], "execution_error": error}
        if result is not None:
            out["result"] = {
                k: (np.asarray(v).tolist() if hasattr(v, "shape") else v) for k, v in result.items()
            }
            arrays[mode + "_rng_words"] = np.asarray(owner.state.words)
            arrays[mode + "_rng_index"] = np.asarray(owner.state.index)
        # Persist every observation BEFORE applying original assertion gates.
        report["runs"][mode] = out
        np.savez(D / "arrays.npz", **arrays)
        (D / "RESULT.json").write_text(json.dumps(report, indent=2))
        failures = []
        if result is not None:
            checks = [
                (
                    "estimate",
                    lambda: T._assert_source_array(result["estimate"], T._field(case, "est")),
                ),
                (
                    "v",
                    lambda: np.testing.assert_array_equal(
                        np.asarray(result["v"]).reshape(-1, 1), np.asarray(T._field(case, "v"))
                    ),
                ),
                (
                    "w",
                    lambda: T._assert_source_array(
                        np.asarray(result["w"]).reshape(-1, 1), T._field(case, "w")
                    ),
                ),
                (
                    "iterations",
                    lambda: np.testing.assert_array_equal(
                        np.asarray(result["iterations"]).reshape(-1),
                        np.asarray(T._field(case, "iter")).reshape(-1),
                    ),
                ),
                ("trace", lambda: T._assert_native_trace(trace, T._field(case, "trace"))),
                ("rng", lambda: T._assert_state(owner.state, T._field(case, "after"))),
            ]
            for name, check in checks:
                try:
                    check()
                except Exception as exc:
                    failures.append({"check": name, "error": str(exc)})
        out["assertion_failures"] = failures
    (D / "RESULT.json").write_text(json.dumps(report, indent=2))
    np.savez(D / "arrays.npz", **arrays)
    print(
        json.dumps(
            {
                k: {
                    "error": v["execution_error"],
                    "failures": v["assertion_failures"],
                    "result": v.get("result"),
                }
                for k, v in report["runs"].items()
            }
        ),
        flush=True,
    )
    assert all(
        v["execution_error"] is None and not v["assertion_failures"]
        for v in report["runs"].values()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = list((root / "src").rglob("*.py")) + [
        Path(__file__),
        root / "pixi.lock",
        root / "pyproject.toml",
        root / "tests/test_utils/test_normest1_native_source.py",
        root / "tests/references/pde_condest_native/positive_inverse6.mat",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    selected_env = {
        k: os.environ.get(k)
        for k in (
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "CUDA_VISIBLE_DEVICES",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "RUNNER_ARCH",
            "RUNNER_OS",
        )
    }
    origin = {
        "input_sha256": hashes,
        "environment": selected_env,
        "environment_sha256": hashlib.sha256(
            json.dumps(selected_env, sort_keys=True).encode()
        ).hexdigest(),
    }
    (output / "INPUTS.json").write_text(json.dumps(origin, indent=2))
    summary = {"success": False}
    try:
        run(output)
        summary["success"] = True
    except BaseException:
        summary["error"] = traceback.format_exc()
        raise
    finally:
        summary["inputs_stable"] = all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in hashes.items()
        )
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
        artifacts = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file() and p.name != "ARTIFACTS.json"
        }
        (output / "ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2))
        print(json.dumps(summary), flush=True)
    if not summary["inputs_stable"]:
        raise RuntimeError("Diagnostic inputs changed")


if __name__ == "__main__":
    main()
