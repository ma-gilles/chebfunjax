# chebfunjax — Master Plan

> Translating MATLAB Chebfun to Python/JAX.
> This document is the single source of truth for architecture, conventions, and progress.
> Every agent MUST read this before starting work.

## Table of Contents

1. [Locked Design Decisions](#1-locked-design-decisions)
2. [Provenance & Credit](#2-provenance--credit)
3. [Documentation Standard](#3-documentation-standard)
4. [Code Patterns & Templates](#4-code-patterns--templates)
5. [Testing & Validation Standard](#5-testing--validation-standard)
6. [Module Dependency Graph](#6-module-dependency-graph)
7. [Translation Units](#7-translation-units)
8. [Quality Gates](#8-quality-gates)
9. [Progress Tracking](#9-progress-tracking)
10. [What NOT To Translate](#10-what-not-to-translate)

---

## 1. Locked Design Decisions

These decisions are **final**. Agents MUST NOT deviate without explicit user approval.

### 1.1 Array Backend: JAX-only

```python
import jax.numpy as jnp
```

- **No dual backend.** No numpy fallback layer. No `_backend.py` dispatch.
- JAX runs on CPU transparently — users switch via `JAX_PLATFORMS=cpu` or
  `jax.default_device(jax.devices("cpu")[0])`.
- The library NEVER manages device placement internally. Users control it.
- For functions JAX lacks (e.g., some scipy.special), use `scipy` outside
  JIT boundaries or `jax.pure_callback` inside them.

### 1.2 Precision: float64 always

```python
# In src/chebfunjax/__init__.py — already done
import jax
jax.config.update("jax_enable_x64", True)
```

- All array creation MUST specify `dtype=jnp.float64` explicitly.
- Never rely on JAX's default dtype.
- Spectral methods require double precision — this is non-negotiable.

### 1.3 Object Model: Equinox Modules (frozen pytrees)

```python
import equinox as eqx

class Chebtech2(eqx.Module):
    coeffs: jax.Array                          # Chebyshev coefficients
    interval: tuple[float, float] = eqx.field(static=True)  # domain endpoints
```

**Why equinox:** Automatic pytree registration, immutability enforcement,
`filter_jit`/`filter_vmap` for mixed static/dynamic fields, inheritance works.

**Rules:**
- All chebfunjax objects are **immutable**. Operations return new objects.
- Array fields are pytree children (traced by JIT/vmap).
- Non-array metadata (domain bounds, flags, preference strings) use `eqx.field(static=True)`.
- Never store Python lists of variable-length arrays — pad + mask instead if JIT-needed,
  or keep as a Python list outside JIT.

### 1.4 JIT Boundaries

| Code pattern | JIT? | How |
|-------------|------|-----|
| Adaptive construction (while loop checking coefficient decay) | **NO** | Python loop; JIT inner FFT/eval |
| Evaluation at fixed points | YES | `@jax.jit` or `eqx.filter_jit` |
| Differentiation (coefficient manipulation) | YES | `@jax.jit` |
| Integration, inner products | YES | `@jax.jit` |
| Rootfinding (colleague matrix eigenvalues) | YES | `@jax.jit` |
| Operator construction (assembling matrices) | YES | `@jax.jit` |
| Adaptive refinement loops | **NO** | Python loop; JIT the step |

The adaptive construction loop is the most important architectural boundary.
It lives in Python because:
- It creates arrays of varying length (dynamic shapes).
- It has data-dependent control flow (convergence check).
- JAX's `lax.while_loop` requires fixed-shape state.

**Pattern:** Python outer loop calls JIT-compiled inner kernels:

```python
def _adaptive_construct(f, interval, maxpow2=16):
    """Python-level adaptive loop. NOT JIT-able."""
    for k in range(4, maxpow2 + 1):
        n = 2**k + 1
        values = _evaluate_on_grid(f, n, interval)        # JIT-able
        coeffs = _values_to_coeffs(values)                  # JIT-able (FFT)
        cutoff = standard_chop(coeffs)                      # JIT-able
        if cutoff < n:
            return coeffs[:cutoff]
    warnings.warn(f"Did not converge with {2**maxpow2 + 1} points")
    return coeffs
```

### 1.5 Class Hierarchy

Mirror Chebfun's conceptual hierarchy, adapted for Python/JAX:

```
Layer 0: Preferences
  Preferences (singleton, like chebpy's UserPreferences)

Layer 1: Utilities (pure functions, no classes)
  quadrature, transforms, interpolation, diffmat, polynomials, approximation, misc

Layer 2: Tech (representation on [-1, 1])
  Chebtech (abstract eqx.Module)
  ├── Chebtech1 (1st kind grid)
  └── Chebtech2 (2nd kind grid, default)
  Trigtech (periodic functions)

Layer 3: Fun (representation on [a, b])
  Classicfun
  ├── Bndfun (bounded intervals)
  └── Unbndfun (unbounded intervals via mapping)
  Singfun (algebraic/log singularities)
  Deltafun (delta function support)

Layer 4: Chebfun (piecewise, user-facing)
  Chebfun (the main class — piecewise collection of Funs)

Layer 5: Operators
  Chebmatrix, Linop, Chebop, Chebop2

Layer 6: Higher dimensions
  SeparableApprox → Chebfun2, Diskfun, Spherefun
  Chebfun3, Ballfun

Layer 7: Time integration
  Spinop, Spinop2, Spinop3, Spinopsphere
```

### 1.6 Public API Style

The library should feel **slick like Chebfun**:

```python
import chebfunjax as cj

# Construction from function handle
f = cj.chebfun(jnp.sin)                    # adaptive on [-1, 1]
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])  # custom domain
f = cj.chebfun(jnp.sin, n=20)              # fixed degree

# Natural arithmetic
g = f + 1
h = f * f
k = f ** 2 - cj.chebfun(jnp.cos)

# Calculus
fp = f.diff()
F = f.cumsum()
integral = f.sum()       # definite integral over domain

# Rootfinding
r = f.roots()

# Evaluation
y = f(0.5)               # scalar
y = f(jnp.linspace(0, 1, 100))  # vectorized

# Info
len(f)                   # number of coefficients
f.domain                 # (a, b)
f.coeffs                 # Chebyshev coefficients
print(f)                 # pretty summary like Chebfun
```

**Factory function** `cj.chebfun(...)` is the primary entry point, not `Chebfun(...)`.
Internal classes (Chebtech2, Bndfun, etc.) are accessible but not the main API.

### 1.7 Package Name

The package is `chebfunjax` (matching the GitHub repo `ma-gilles/chebfunjax`).
Import as `import chebfunjax as cj`.

---

## 2. Provenance & Credit

### 2.1 Source Tracking

Every translated function MUST include a provenance block in its docstring:

```python
def chebpts(n: int, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points of the first or second kind on [-1, 1].

    ... (main documentation) ...

    Provenance
    ----------
    MATLAB source : chebfun/chebpts.m
    Chebfun commit: 7574c77 (v5.x, 2025-09-26)
    Original authors: Nick Trefethen
    Algorithm: Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
        quadrature rules", BIT Numerical Mathematics, 46, 2006, pp 195-202.

    See Also
    --------
    trigpts, legpts, jacpts, lagpts, hermpts, lobpts, radaupts
    """
```

**Rules:**
- `MATLAB source` = path relative to chebfun repo root (e.g., `@chebfun/sum.m`).
- `Chebfun commit` = the commit hash of the MATLAB source we're translating from.
  Our reference is **7574c77** (the shallow clone at `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/`).
- `Original authors` = names from the MATLAB file's header/comments. If the MATLAB file
  only has the generic "Copyright 2017 by The University of Oxford and The Chebfun Developers",
  write that. If it names specific people (e.g., "by Nick Hale, July 2011"), include them.
- `Algorithm` = paper references from the MATLAB file's DEVELOPER NOTES section.
- `See Also` = translated to Python function/class names.

### 2.2 License

The repo uses BSD-3-Clause. The original Chebfun is BSD-2-Clause.
We must include Chebfun's license notice. Add to `LICENSE`:

```
This project is a derivative work of Chebfun (https://github.com/chebfun/chebfun),
which is licensed under the BSD 2-Clause License:

  Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
  of Oxford, and the Chebfun Developers. All rights reserved.
  [... full BSD-2 text ...]
```

### 2.3 Module-Level Attribution

Each Python source file starts with a module docstring that credits the source:

```python
"""Chebyshev technology for smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""
```

---

## 3. Documentation Standard

### 3.1 Docstring Format

NumPy-style docstrings. Every public function/method MUST have:

```python
def roots(self, *, recurse: bool = True) -> jnp.ndarray:
    """Roots of a Chebfun in its domain of definition.

    Finds all real roots by computing eigenvalues of the colleague matrix
    formed from the Chebyshev coefficients.

    Parameters
    ----------
    recurse : bool, default True
        If True, recursively subdivide to improve accuracy for
        high-degree polynomials.

    Returns
    -------
    r : jnp.ndarray, shape (n_roots,)
        Sorted roots in the domain.

    Examples
    --------
    >>> f = cj.chebfun(lambda x: x**2 - 0.25)
    >>> f.roots()
    Array([-0.5,  0.5], dtype=float64)

    Provenance
    ----------
    MATLAB source : @chebfun/roots.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun.max, Chebfun.min, Chebfun.minandmax
    """
```

### 3.2 Preserve MATLAB Documentation

When translating, preserve the mathematical descriptions and algorithm explanations
from the MATLAB file. Translate MATLAB syntax to Python syntax in examples.
Keep references to papers verbatim.

### 3.3 Developer Notes

Substantial algorithms should include a `Notes` section with the developer notes
from the MATLAB source:

```python
    Notes
    -----
    Developer notes from MATLAB Chebfun:

    'GW' by Nick Trefethen, March 2009 — algorithm adapted from [1].
    'REC' by Nick Hale, July 2011.
    'ASY' algorithm by Bogaert [2]. Matlab code by Nick Hale, July 2014.

    References
    ----------
    .. [1] G. H. Golub and J. A. Welsch, "Calculation of Gauss quadrature
       rules", Math. Comp. 23, 221-230, 1969.
    .. [2] I. Bogaert, "Iteration-free computation of Gauss-Legendre
       quadrature nodes and weights", SIAM J. Sci. Comput., 36(3),
       A1008-A1026, 2014.
```

---

## 4. Code Patterns & Templates

### 4.1 Utility Function Template

```python
"""Quadrature points and weights.

Translated from MATLAB Chebfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def legpts(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Legendre points and Gauss-Legendre quadrature weights.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Legendre points in (-1, 1), ascending order.
    w : jnp.ndarray, shape (n,)
        Corresponding quadrature weights.

    Examples
    --------
    >>> x, w = legpts(5)
    >>> float(jnp.dot(w, x**4))  # integral of x^4 on [-1,1] = 2/5
    0.4

    Provenance
    ----------
    MATLAB source : chebfun/legpts.m
    Chebfun commit: 7574c77
    Original authors: Nick Trefethen (GW, 2009), Nick Hale (REC, 2011;
        ASY, 2014), Ignace Bogaert (ASY algorithm).
    Algorithm:
        [1] Golub & Welsch, "Calculation of Gauss quadrature rules", 1969.
        [2] Bogaert, "Iteration-free computation of Gauss-Legendre quadrature
            nodes and weights", SIAM J. Sci. Comput., 2014.

    See Also
    --------
    chebpts, jacpts, hermpts, lagpts, ultrapts
    """
    ...
```

### 4.2 Class (eqx.Module) Template

```python
"""Chebyshev technology — smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class Chebtech2(eqx.Module):
    """Chebyshev interpolant on 2nd-kind points.

    Represents a smooth function on [-1, 1] via its values at Chebyshev
    points of the 2nd kind (Clenshaw-Curtis / Chebyshev-Lobatto points)
    and the coefficients of the corresponding 1st-kind Chebyshev series.

    Attributes
    ----------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev series coefficients (1st kind: T_0, T_1, ..., T_{n-1}).
    values : jnp.ndarray, shape (n,)
        Function values at the n Chebyshev-2 points on [-1, 1].
    ishappy : bool
        True if the representation is resolved to the requested tolerance.

    Provenance
    ----------
    MATLAB source : @chebtech2/chebtech2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech1, Trigtech, Bndfun
    """

    coeffs: jax.Array
    values: jax.Array
    ishappy: bool = eqx.field(static=True, default=True)
    epslevel: float = eqx.field(static=True, default=2.220446049250313e-16)

    # --- Construction (classmethods, NOT __init__) ---

    @classmethod
    def from_function(cls, f: Callable, *, n: int | None = None) -> Chebtech2:
        """Construct from a callable. Adaptive if n is None."""
        if n is None:
            return _adaptive_construct(cls, f)
        else:
            return _fixed_construct(cls, f, n)

    @classmethod
    def from_coeffs(cls, coeffs: jnp.ndarray) -> Chebtech2:
        """Construct from Chebyshev coefficients."""
        values = _coeffs_to_values(coeffs)
        return cls(coeffs=coeffs, values=values)

    @classmethod
    def from_values(cls, values: jnp.ndarray) -> Chebtech2:
        """Construct from function values at Chebyshev-2 points."""
        coeffs = _values_to_coeffs(values)
        return cls(coeffs=coeffs, values=values)

    # --- Evaluation ---

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate at point(s) x in [-1, 1] via Clenshaw's algorithm."""
        return _clenshaw(self.coeffs, x)

    # --- Properties ---

    @property
    def n(self) -> int:
        """Number of Chebyshev points / coefficients."""
        return self.coeffs.shape[0]

    # --- Arithmetic (return new objects) ---

    def __add__(self, other):
        ...

    def __neg__(self):
        return Chebtech2(coeffs=-self.coeffs, values=-self.values,
                         ishappy=self.ishappy, epslevel=self.epslevel)

    # --- Calculus ---

    def diff(self, k: int = 1) -> Chebtech2:
        """Differentiate k times."""
        new_coeffs = _diff_coeffs(self.coeffs, k)
        return Chebtech2.from_coeffs(new_coeffs)

    def cumsum(self) -> Chebtech2:
        """Indefinite integral (antiderivative with F(-1) = 0)."""
        new_coeffs = _cumsum_coeffs(self.coeffs)
        return Chebtech2.from_coeffs(new_coeffs)

    def sum(self) -> float:
        """Definite integral over [-1, 1]."""
        return float(_definite_integral(self.coeffs))

    # --- Roots ---

    def roots(self) -> jnp.ndarray:
        """Real roots in [-1, 1] via colleague matrix eigenvalues."""
        return _roots_colleague(self.coeffs)

    # --- Display ---

    def __repr__(self) -> str:
        return (f"Chebtech2(n={self.n}, "
                f"vscale={float(jnp.max(jnp.abs(self.values))):.4g})")
```

### 4.3 Operator Overloading Pattern

Python dunder methods replace MATLAB's `@folder/plus.m`, `@folder/times.m`, etc.

| MATLAB file | Python method | Notes |
|------------|--------------|-------|
| `plus.m` | `__add__`, `__radd__` | |
| `minus.m` | `__sub__`, `__rsub__` | |
| `times.m` | `__mul__`, `__rmul__` | Element-wise |
| `mtimes.m` | `__matmul__` | Only for operator application |
| `rdivide.m` | `__truediv__`, `__rtruediv__` | |
| `power.m` | `__pow__`, `__rpow__` | |
| `uminus.m` | `__neg__` | |
| `uplus.m` | `__pos__` | |
| `abs.m` | `__abs__` | |
| `eq.m` | `__eq__` | |
| `lt.m`, `le.m` | `__lt__`, `__le__` | |
| `subsref.m` | `__call__`, `__getitem__` | f(x) for evaluation |
| `length.m` | `__len__` | Number of coefficients |
| `display.m` | `__repr__`, `__str__` | |

### 4.4 GPU Transparency

The library does **nothing** special for GPU. JAX handles it:

```python
# CPU (default if no GPU, or explicit)
import jax
jax.config.update("jax_default_device", jax.devices("cpu")[0])
f = cj.chebfun(jnp.sin)  # runs on CPU

# GPU (automatic if available)
# Just don't set default_device — JAX picks GPU automatically
f = cj.chebfun(jnp.sin)  # runs on GPU if available

# Explicit GPU
with jax.default_device(jax.devices("gpu")[0]):
    f = cj.chebfun(jnp.sin)
```

No `device=` arguments, no `.to_gpu()` methods, no backend selection.
JAX arrays live where they're created; operations follow the data.

---

## 5. Testing & Validation Standard

### 5.1 Three Tiers of Tests

**Tier 1: Unit tests (pure Python, no MATLAB)**
- Mathematical properties: `diff(cumsum(f)) ≈ f`, `sum(1) = 2`, symmetry.
- Edge cases: empty inputs, n=0, n=1, constant functions.
- Marker: none (always run).

**Tier 2: MATLAB cross-validation**
- Compare output against MATLAB Chebfun reference data.
- Reference data stored as `.mat` files in `tests/references/`.
- Generated by `matlab_harness/generate_refs.m`.
- Marker: `@pytest.mark.matlab`.

**Tier 3: Performance benchmarks**
- Timing comparison vs MATLAB (informational, not pass/fail).
- Memory usage.
- GPU vs CPU speedup.
- Stored in `benchmarks/`.
- Marker: `@pytest.mark.benchmark`.

### 5.2 Tolerance Rules

```python
# Default for all MATLAB comparisons:
np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-14)

# For functions with known precision limits (e.g., rootfinding):
np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)
# MUST document why tolerance is relaxed in a comment.
```

**NEVER widen tolerances without documenting the reason.**
If precision is worse than expected, investigate the root cause first.

### 5.3 Test File Structure

One test file per source module:

```
tests/
├── conftest.py                    # fixtures, MATLAB ref loaders
├── test_utils/
│   ├── test_quadrature.py         # tests src/chebfunjax/utils/quadrature.py
│   ├── test_transforms.py
│   ├── test_interpolation.py
│   └── ...
├── test_tech/
│   ├── test_chebtech.py
│   └── test_trigtech.py
├── test_chebfun/
│   └── test_chebfun.py
├── test_operators/
│   └── ...
└── references/                    # MATLAB-generated .mat files
    ├── quadrature.mat
    ├── chebfun_basic.mat
    └── ...
```

### 5.4 MATLAB Reference Generation Workflow

For each module, add test cases to `matlab_harness/generate_refs.m`:

```matlab
%% --- Module: quadrature ---
ref = struct();
for n = [5, 10, 17, 32, 64, 128]
    ref.(sprintf('chebpts2_n%d', n)) = chebpts(n);
    ref.(sprintf('chebpts1_n%d', n)) = chebpts(n, 1);
    [x, w] = legpts(n);
    ref.(sprintf('legpts_x_n%d', n)) = x;
    ref.(sprintf('legpts_w_n%d', n)) = w;
end
save(fullfile(outdir, 'quadrature.mat'), '-struct', 'ref');
```

Run with:
```bash
module load matlab/R2025b
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"
```

### 5.5 Performance Benchmarking Template

```python
# benchmarks/bench_quadrature.py
import time
import jax
import jax.numpy as jnp
from chebfunjax.utils.quadrature import chebpts, chebweights

def bench_chebpts():
    """Benchmark chebpts for various n, CPU vs GPU."""
    results = []
    for n in [64, 256, 1024, 4096, 16384]:
        # Warmup (JIT compile)
        _ = chebpts(n)

        # CPU timing
        with jax.default_device(jax.devices("cpu")[0]):
            t0 = time.perf_counter()
            for _ in range(100):
                x = chebpts(n)
                x.block_until_ready()
            cpu_time = (time.perf_counter() - t0) / 100

        # GPU timing (if available)
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            with jax.default_device(gpu_devices[0]):
                _ = chebpts(n)  # warmup on GPU
                t0 = time.perf_counter()
                for _ in range(100):
                    x = chebpts(n)
                    x.block_until_ready()
                gpu_time = (time.perf_counter() - t0) / 100
        else:
            gpu_time = None

        results.append({"n": n, "cpu_ms": cpu_time*1000,
                        "gpu_ms": gpu_time*1000 if gpu_time else "N/A"})

    # Print table
    print(f"{'n':>8} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>8}")
    for r in results:
        speedup = (r['cpu_ms'] / r['gpu_ms']) if isinstance(r['gpu_ms'], float) else "N/A"
        print(f"{r['n']:>8} {r['cpu_ms']:>10.3f} {str(r['gpu_ms']):>10} {str(speedup):>8}")
```

---

## 6. Module Dependency Graph

```
                    ┌─────────────────────────────────────────────┐
                    │            LAYER 1: UTILITIES               │
                    │  quadrature  transforms  interpolation      │
                    │  diffmat  polynomials  approximation  misc  │
                    │  domain  pref                               │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │            LAYER 2: TECH                    │
                    │      Chebtech1  Chebtech2  Trigtech         │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │            LAYER 3: FUN                     │
                    │   Bndfun  Unbndfun  Singfun  Deltafun       │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │          LAYER 4: CHEBFUN                   │
                    │     Chebfun (piecewise, user-facing)        │
                    └───────┬──────────┬──────────┬───────────────┘
                            │          │          │
              ┌─────────────▼─┐  ┌─────▼────┐  ┌─▼──────────────┐
              │  LAYER 5:     │  │ LAYER 6: │  │ LAYER 7:       │
              │  DISCRETIZE   │  │ 2D/3D    │  │ SPIN           │
              │  chebcolloc   │  │ chebfun2 │  │ spinop/2/3     │
              │  ultraS       │  │ diskfun  │  │ spinopsphere   │
              │  trigcolloc   │  │ spherfun │  │                │
              └───────┬───────┘  │ chebfun3 │  └────────────────┘
                      │          │ ballfun  │
              ┌───────▼───────┐  └──────────┘
              │  LAYER 5B:    │
              │  OPERATORS    │
              │  linop        │
              │  chebop       │
              │  chebmatrix   │
              └───────────────┘
```

**Critical path for a working "vertical slice":**
`quadrature` → `transforms` → `misc` (standardChop) → `Chebtech2` → `Bndfun` → `Chebfun`

This gives: construction, evaluation, arithmetic, differentiation, integration, roots.
**Target this first.** Everything else fans out from here.

---

## 7. Translation Units

Each unit = one branch = one PR = one agent assignment.
Every PR requires CI to pass before merge. See `.github/workflows/ci.yml`.
Units within the same phase have no mutual dependencies and can run in parallel.

### Phase 0: Infrastructure [DONE]
- [x] U00: Repo skeleton, pyproject.toml, pixi.toml
- [x] U01: MATLAB reference harness
- [x] U02: Test infrastructure (conftest.py)
- [x] U03: quadrature.py (template module)
- [x] U04: GitHub Actions CI (`ci.yml`)

### Phase 1: Utilities (all independent — max parallelism)

| Unit | Module | MATLAB sources | Key functions | Est. size |
|------|--------|---------------|--------------|-----------|
| U10 | `utils/quadrature.py` | `chebpts.m`, `legpts.m`, `jacpts.m`, `hermpts.m`, `lagpts.m`, `ultrapts.m`, `radaupts.m`, `lobpts.m`, `trigpts.m`, `paduapts.m` | `chebpts`, `legpts`, `jacpts`, `hermpts`, `lagpts`, `ultrapts`, `radaupts`, `lobpts`, `trigpts`, `paduapts` | ~800 LOC |
| U11 | `utils/transforms.py` | `cheb2leg.m`, `leg2cheb.m`, `jac2cheb.m`, `cheb2jac.m`, `chebvals2legcoeffs.m`, `chebcoeffs2legvals.m` + variants | `cheb2leg`, `leg2cheb`, `jac2cheb`, `cheb2jac`, + 8 others | ~600 LOC |
| U12 | `utils/interpolation.py` | `bary.m`, `trigBary.m`, `baryWeights.m`, `trigBaryWeights.m`, `barymat.m` | `bary`, `trig_bary`, `bary_weights`, `trig_bary_weights`, `barymat` | ~300 LOC |
| U13 | `utils/diffmat.py` | `diffmat.m`, `intmat.m`, `cumsummat.m`, `introw.m`, `diffrow.m` | `diffmat`, `intmat`, `cumsummat`, `introw`, `diffrow` | ~400 LOC |
| U14 | `utils/polynomials.py` | `chebpoly.m`, `legpoly.m`, `jacpoly.m`, `hermpoly.m`, `lagpoly.m`, `ultrapoly.m` | `chebpoly`, `legpoly`, `jacpoly`, `hermpoly`, `lagpoly`, `ultrapoly` | ~300 LOC |
| U15a | `utils/aaa.py` | `aaa.m`, `aaatrig.m` | `aaa`, `aaatrig` | ~500 LOC |
| U15b | `utils/minimax.py` | `minimax.m` | `minimax` (Remez exchange) | ~500 LOC |
| U15c | `utils/ratapprox.py` | `ratinterp.m`, `padeapprox.m`, `trigratinterp.m`, `cf.m` | `ratinterp`, `padeapprox`, `trigratinterp`, `cf` | ~500 LOC |
| U16 | `utils/misc.py` | `standardChop.m`, `gridsample.m`, `abstractQR.m`, `outerProd.m` | `standard_chop`, `gridsample`, `abstract_qr`, `outer_prod` | ~300 LOC |
| U17 | `domain.py` | `@domain/` (17 methods) | `Domain` class | ~200 LOC |
| U18 | `pref.py` | `@chebpref/`, `@chebfunpref/`, `@cheboppref/` | `Preferences` singleton | ~150 LOC |

### Phase 2: Tech (depends on Phase 1)

| Unit | Module | MATLAB sources | Key classes/functions | Est. size |
|------|--------|---------------|----------------------|-----------|
| U20a | `tech/chebtech_core.py` | `@chebtech/` subset | `Chebtech2` class, `coeffs2vals`, `vals2coeffs`, Clenshaw evaluation, `prolong` | ~500 LOC |
| U20b | `tech/chebtech_construct.py` | `@chebtech/` subset | Adaptive constructor, `happinessCheck`, `standard_chop`, `refine` | ~500 LOC |
| U20c | `tech/chebtech_ops.py` | `@chebtech/` subset | Arithmetic, `simplify`, `diff`, `cumsum`, `sum`, `roots`, `innerProduct` | ~500 LOC |
| U20d | `tech/chebtech_misc.py` | `@chebtech/` subset + `@chebtech1/` | Remaining methods, `Chebtech1` variant | ~400 LOC |
| U21a | `tech/trigtech_core.py` | `@trigtech/` subset | `Trigtech` class, trig coefficients, evaluation | ~500 LOC |
| U21b | `tech/trigtech_ops.py` | `@trigtech/` subset | Arithmetic, calculus, roots for periodic functions | ~500 LOC |

### Phase 3: Fun (depends on Phase 2)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U30 | `fun/classicfun.py` + `fun/bndfun.py` | `@classicfun/` (60), `@bndfun/` (20) | `Classicfun`, `Bndfun` — maps [a,b] to [-1,1] | ~800 LOC |
| U31 | `fun/unbndfun.py` | `@unbndfun/` (19) | `Unbndfun` — maps (-∞,b], [a,∞), (-∞,∞) | ~400 LOC |
| U32 | `fun/singfun.py` | `@singfun/` (81) | `Singfun` — algebraic/log singularities | ~800 LOC |
| U33 | `fun/deltafun.py` | `@deltafun/` (68) | `Deltafun` — Dirac delta support | ~600 LOC |

### Phase 4: Chebfun 1D (depends on Phase 3)

| Unit | Module | MATLAB sources | Scope | Est. size |
|------|--------|---------------|-------|-----------|
| U40 | `chebfun1d/chebfun.py` | `@chebfun/chebfun.m` + construction methods | Constructor, `from_function`, `from_values`, `from_coeffs`, `restrict`, `simplify` | ~800 LOC |
| U41 | `chebfun1d/arithmetic.py` | `@chebfun/plus.m`, `minus.m`, `times.m`, `rdivide.m`, `power.m`, `compose.m`, etc. | All arithmetic + composition ops, registered on Chebfun class | ~600 LOC |
| U42 | `chebfun1d/calculus.py` | `@chebfun/diff.m`, `sum.m`, `cumsum.m`, `norm.m`, `innerProduct.m`, `mean.m` | Differentiation, integration, norms | ~500 LOC |
| U43 | `chebfun1d/rootfinding.py` | `@chebfun/roots.m`, `max.m`, `min.m`, `minandmax.m` | Roots, extrema | ~500 LOC |
| U44 | `chebfun1d/specfun.py` | `@chebfun/sin.m`, `cos.m`, `exp.m`, `log.m`, `abs.m`, `sign.m`, `besselj.m`, etc. (~80 methods) | Apply standard functions to chebfuns via composition | ~400 LOC |
| U45 | `chebfun1d/linalg.py` | `@chebfun/qr.m`, `svd.m`, `eig.m` | Linear algebra on quasimatrices | ~600 LOC |
| U46 | `chebfun1d/ode.py` | `@chebfun/ode45.m`, `ode113.m`, `bvp4c.m`, `pde15s.m` | ODE/PDE solvers (may defer to diffrax) | ~800 LOC |

### Phase 5: Discretization (depends on Phase 2)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U50 | `discretization/chebcolloc.py` | `@chebcolloc/`, `@chebcolloc1/`, `@chebcolloc2/` | Chebyshev collocation discretization | ~400 LOC |
| U51 | `discretization/ultras.py` | `@ultraS/` (16) | Ultraspherical spectral method | ~500 LOC |
| U52 | `discretization/trigcolloc.py` + `discretization/trigspec.py` | `@trigcolloc/` (15), `@trigspec/` (13) | Trig collocation + spectral | ~400 LOC |

### Phase 6: Operators (depends on Phase 4 + Phase 5)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U60 | `operators/chebmatrix.py` | `@chebmatrix/` (36) | Block matrix of chebfuns/operators | ~500 LOC |
| U61 | `operators/blocks.py` | `@linBlock/`, `@operatorBlock/`, `@functionalBlock/` | Operator building blocks | ~500 LOC |
| U62 | `operators/linop.py` | `@linop/` (26), `@linopConstraint/` | Linear operator + BCs | ~600 LOC |
| U63 | `operators/chebop.py` | `@chebop/` (57) | Nonlinear operator, Newton iteration | ~1000 LOC |
| U64 | `operators/chebop2.py` | `@chebop2/` (28) | 2D operator | ~600 LOC |

### Phase 7: 2D functions (depends on Phase 4)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U70a | `chebfun2d/separable_approx.py` (core) | `@separableApprox/` subset | Construction, evaluation, low-rank SVD | ~500 LOC |
| U70b | `chebfun2d/separable_approx.py` (ops) | `@separableApprox/` subset | Arithmetic, calculus, roots, plotting data | ~500 LOC |
| U70c | `chebfun2d/separable_approx.py` (misc) | `@separableApprox/` subset | Remaining methods (108 total) | ~500 LOC |
| U71a | `chebfun2d/chebfun2.py` | `@chebfun2/` (139) | 2D scalar on rectangles | ~800 LOC |
| U71b | `chebfun2d/chebfun2v.py` | `@chebfun2v/` (54) | 2D vector fields | ~400 LOC |
| U72a | `diskfun/diskfun.py` | `@diskfun/` (130) | 2D scalar on disk | ~800 LOC |
| U72b | `diskfun/diskfunv.py` | `@diskfunv/` (46) | Disk vector fields | ~400 LOC |
| U73a | `spherefun/spherefun.py` | `@spherefun/` (132) | 2D scalar on sphere | ~800 LOC |
| U73b | `spherefun/spherefunv.py` | `@spherefunv/` (48) | Sphere vector fields | ~400 LOC |

### Phase 8: 3D functions (depends on Phase 7)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U80a | `chebfun3d/chebfun3.py` | `@chebfun3/` (113) | 3D scalar on cuboids | ~700 LOC |
| U80b | `chebfun3d/chebfun3v.py` | `@chebfun3v/` (50) | 3D vector fields | ~400 LOC |
| U80c | `chebfun3d/chebfun3t.py` | `@chebfun3t/` (37) | Tucker tensor representation | ~300 LOC |
| U81a | `ballfun/ballfun.py` | `@ballfun/` (66) | 3D scalar on ball | ~500 LOC |
| U81b | `ballfun/ballfunv.py` | `@ballfunv/` (34) | Ball vector fields | ~300 LOC |

### Phase 9: Time integration (depends on Phase 4 + Phase 5)

| Unit | Module | MATLAB sources | Key classes | Est. size |
|------|--------|---------------|------------|-----------|
| U90a | `spin/spinop.py` | `@spinoperator/` (5), `@spinop/` (14), `@spinop2/` (14) | Base operator + 1D/2D PDE stepping | ~500 LOC |
| U90b | `spin/spinop3_sphere.py` | `@spinop3/` (14), `@spinopsphere/` (14) | 3D + sphere PDE stepping | ~400 LOC |
| U90c | `spin/schemes.py` | `@expinteg/` (9), `@imex/` (4), `@spinscheme/` | Time-stepping schemes (may leverage diffrax) | ~300 LOC |

### Phase 10: Autodiff + integration tests

| Unit | Module | Scope | Est. size |
|------|--------|-------|-----------|
| U100 | `autodiff/adchebfun.py` | Automatic differentiation of chebfun operations | ~300 LOC |
| U101 | Integration tests | End-to-end tests across all layers | ~500 LOC |
| U102 | Benchmarks | Comprehensive CPU/GPU benchmark suite | ~400 LOC |

---

## 8. Quality Gates

Every PR MUST pass these gates before merge:

### Gate 1: Code Quality
- [ ] Follows code pattern templates (Section 4)
- [ ] Every public function has full docstring with Provenance section
- [ ] No `import numpy` — only `jax.numpy` (except in tests/benchmarks)
- [ ] Type annotations on all public functions
- [ ] `ruff check` passes with zero warnings

### Gate 2: Tests
- [ ] Unit tests for every public function (Tier 1)
- [ ] MATLAB cross-validation tests (Tier 2) — `@pytest.mark.matlab`
- [ ] All tests pass: `pixi run test-fast` + `pixi run test-matlab`
- [ ] Coverage ≥ 90% for new code (measured per-file)

### Gate 3: Accuracy
- [ ] Agreement with MATLAB Chebfun at `rtol=1e-12, atol=1e-14`
- [ ] Any relaxed tolerances documented with reason
- [ ] No accuracy regression vs previous version

### Gate 4: Performance
- [ ] Benchmark results recorded in PR description with this table:

| Function | n | JIT compile (ms) | Steady-state CPU (ms) | Steady-state GPU (ms) | MATLAB (ms) | Peak mem (MB) |
|----------|---|------------------|----------------------|----------------------|-------------|---------------|

- [ ] Steady-state (post-JIT) should be within 3x of MATLAB for n ≥ 64
- [ ] If > 3x slower, document why (e.g., JAX dispatch overhead for small n)
- [ ] Report both first-call (JIT compile) and steady-state (warm) timings
- [ ] For core functions (chebpts, eval, diff, sum, roots): include scaling curves (n vs time)

### Gate 5: JAX Semantics
- [ ] JIT: functions work under `jax.jit` (with appropriate static args)
- [ ] vmap: batch-friendly functions work under `jax.vmap` where applicable
- [ ] grad: differentiable functions return correct gradients via `jax.grad`
- [ ] CPU/GPU parity: if GPU is available, results match CPU at `rtol=1e-12`
- [ ] Not every function needs all four — mark which apply in the PR description

### Gate 6: Integration
- [ ] Existing tests still pass (no regressions)
- [ ] Module imports work: `python -c "from chebfunjax.utils.quadrature import chebpts"`
- [ ] If this is a class: operator overloads tested with scalar and chebfun inputs

---

## 9. Progress Tracking

### Status File

The file `STATUS.md` in the repo root tracks completion. Machine-readable format:

```markdown
| Unit | Module | Status | PR | Agent | Notes |
|------|--------|--------|-----|-------|-------|
| U10 | utils/quadrature | done | #3 | claude-1 | 13 tests pass |
| U11 | utils/transforms | in_progress | #5 | claude-2 | |
| U12 | utils/interpolation | todo | | | |
```

Status values: `todo`, `in_progress`, `in_review`, `done`.

### Branch Naming

`translate/U{XX}-{short-name}`

Examples:
- `translate/U10-quadrature`
- `translate/U20-chebtech`
- `translate/U40-chebfun-core`

### PR Title Format

`[U{XX}] Translate {module}: {short description}`

Examples:
- `[U10] Translate utils/quadrature: chebpts, legpts, jacpts, and 7 more`
- `[U20] Translate tech/chebtech: Chebtech2 with adaptive construction`

---

## 10. What NOT To Translate

| MATLAB class/file | Reason |
|------------------|--------|
| `@chebgui/`, `@chebguiController/`, `@chebguiExporter*/` | GUI — use matplotlib for plotting |
| `chebguiDemos/` | Write Jupyter notebooks instead |
| `@stringParser/` | MATLAB string parsing — irrelevant in Python |
| `chebguiWindow.m`, `chebguiEdit.m` | GUI |
| `Contents.m` | MATLAB help system |
| `ATAPformats.m`, `ODEformats.m` | MATLAB formatting |
| `cheblogo.m`, `cheblogo2.m` | Fun but not needed |
| `chebsnake.m`, `chebsnake2.m` | Games |
| `scribble.m`, `scribble2.m` | Visualization toy |
| `phaseplot.m` | Use existing Python phase plot libraries |
| `conformal.m`, `conformal2.m` | Niche — defer |
| `chebguiDemos/*` | Replace with Jupyter notebooks |
| `@chebdouble/` | MATLAB double-precision wrapper — unnecessary in Python |

**Total saved: ~200 methods, ~15 files.** Focus effort on the mathematical core.

---

## Appendix A: Key References

- MATLAB Chebfun source: `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/` (commit 7574c77)
- Python chebpy reference: `/scratch/gpfs/GILLES/mg6942/chebpy_ref/`
- Chebfun Guide: https://www.chebfun.org/docs/guide/
- Trefethen, "Approximation Theory and Approximation Practice" (ATAP)
- Driscoll, Hale, Trefethen, "Chebfun Guide" (2014)
- Aurentz & Trefethen, "Chopping a Chebyshev series" (2015)

## Appendix B: MATLAB Environment

```bash
module load matlab/R2025b
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); ..."
```

MATLAB R2025b confirmed working with Chebfun on Princeton Della cluster.
