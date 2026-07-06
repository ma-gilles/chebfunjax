# MATLAB Chebfun test-suite port — guide (Opus 4.8)

Goal: replicate **every** MATLAB `tests/<module>/test_*.m` assertion
(except `chebgui`) as a passing pytest, at the **same tolerance**.

## Where MATLAB tests live
`/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/tests/<module>/test_*.m`

## Where ported tests go
`tests/test_matlab_port/<module>/test_<name>_matlab.py`
One Python file per MATLAB file; one test method per `pass(k)` assertion
(group logically; name methods for what they check).

## Core principle: self-validating
Most MATLAB tests check an operation against an **analytic exact** at a
tolerance like `1e3*get(f,'vscale')*eps`. Reproduce the SAME math and the
SAME tolerance in Python. Do NOT invent new tolerances, and NEVER widen a
tolerance to force a pass. Test points can be your own (np.linspace over
the domain) — the assertion is `error < tol`, which holds at any point,
so MATLAB's exact RNG stream is not needed.

## API mapping (MATLAB -> chebfunjax)
| MATLAB | chebfunjax |
|---|---|
| `bndfun(@(x)f, struct('domain',dom), pref)` | `Bndfun.from_function(f, Domain(dom))` |
| `chebtech2(@(x)f)` / `chebtech1(...)` | `Chebtech2.from_function(f)` / `Chebtech1` |
| `trigtech(@(x)f)` | `Trigtech.from_function(f)` (see tech/trigtech.py) |
| `chebfun(@(x)f, dom)` | `chebfun(f, domain=dom)` |
| `diff(f)`, `diff(f,k)` | `f.diff()`, `f.diff(k)` |
| `cumsum(f)`, `sum(f)` | `f.cumsum()`, `f.sum()` |
| `feval(f,x)` / `f(x)` | `f(x)` |
| `get(f,'vscale')` | `f.vscale` |
| `norm(err, inf)` | `float(jnp.max(jnp.abs(err)))` |
| `norm(f)` (2-norm) | `f.norm()` |
| `eps` | `float(np.finfo(np.float64).eps)` |
| `roots(f)` | `f.roots()` |
| `f.*g`, `f+g` | `f*g`, `f+g` |

Grep `src/chebfunjax/<module>/` for the real method names when unsure.
Import from the same paths existing tests use (see the exemplar).

## Exemplar
`tests/test_matlab_port/bndfun/test_diff_matlab.py` — a clean port of
`test_diff.m`. Follow its structure.

## Features chebfunjax may lack
If a MATLAB assertion needs a feature chebfunjax does not implement
(e.g. array-valued/quasimatrix funs, a `dim` option, a specific pref),
do NOT silently drop it. Either:
- implement a faithful equivalent if trivial, or
- mark the method `@pytest.mark.xfail(reason="chebfunjax lacks <X>")`
  or `pytest.skip("<X> not implemented")` with a precise reason.
Report every such gap in your final summary so it can be triaged.

## Non-negotiable
- Attribution: `Opus 4.8` in the module docstring.
- Every ported test must actually PASS (or be an honest xfail/skip).
- Never fake a pass, never widen a tolerance without a MATLAB-justified
  reason written in a comment.
- Run your files: `JAX_PLATFORMS=cpu .pixi/envs/default/bin/python -m
  pytest tests/test_matlab_port/<module>/ -q` before reporting.
