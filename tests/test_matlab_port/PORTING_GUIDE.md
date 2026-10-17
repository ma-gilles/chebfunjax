# MATLAB Chebfun assertion porting guide

Goal: reproduce every original MATLAB assertion required by the active parity
objective, at its original bound. File-name coverage alone does not establish
assertion completeness. Record any unresolved scope explicitly.

## Where MATLAB tests live

`/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/tests/<module>/test_*.m`

## Where ported tests go

`tests/test_matlab_port/<module>/test_<name>_matlab.py`
One Python file per MATLAB file; one test method per `pass(k)` assertion
(group logically; name methods for what they check).

## Preserve each source predicate

Read the pinned MATLAB source before writing or changing its port. Preserve
its operands, domains, preferences, array orientation, operation dispatch,
probes, norms, tolerance factors and strict versus non-strict comparison.
Record the original `pass(k)` or assertion location, including overwritten
flags and loop/policy cases; grouping predicates must not drop cases.

A source function norm must remain a function norm. Sampling on a convenient
finite grid does not establish the same predicate. Likewise, scalar
multiplication is not a substitute for a missing division dispatch, even
when the mathematical result is equivalent.

RNG-dependent source operands require a qualified compatible stream or
explicitly captured native operands with provenance. A different seed or
sampling grid may be useful as a supplemental control; it does not restore
an omitted original assertion or establish example-output parity.

Keep original tolerance expressions, including their scale and operation
order. Distinguish original source assertions from supplemental native
representation, bit-pattern and derivative controls. Do not weaken any of
them to make a candidate pass. If pinned MATLAB itself fails an original
predicate, preserve the predicate and document the native failure separately.

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
| `norm(f, inf)` for a Chebfun | `f.norm(jnp.inf)`; retain source function semantics |
| `norm(A, inf)` for a numeric matrix | Matrix infinity norm (maximum absolute row sum), not maximum entry |
| `norm(f)` (2-norm) | `f.norm()` |
| `eps` | `float(np.finfo(np.float64).eps)` |
| `roots(f)` | `f.roots()` |
| `f.*g`, `f+g` | `f*g`, `f+g` |

Grep `src/chebfunjax/<module>/` for the real method names when unsure.
Import from the same paths existing tests use (see the exemplar).

## Exemplar

`tests/test_matlab_port/bndfun/test_diff_matlab.py` — a clean port of
`test_diff.m`. Follow its structure.

## Missing behavior and failures

Implement missing behavior using the pinned source and the repository's
JAX-only numerical rules. Keep failing candidates isolated and report the
original failing predicate and measured discrepancy. Do not omit assertions,
replace unsupported operations with convenient equivalents, or add xfails
solely to obtain green CI. Existing documented skips/xfails are unresolved
inventory items unless their disposition is established against the reference.

## Provenance and verification

- Give the exact MATLAB source path and commit in the module docstring;
  retain original copyright and credit actual contributors accurately.
- Explain Python API adaptations separately from the native assertion.
- Use the pinned CPU environment described in `HANDOFF_CODEX.md`; preserve
  its dependency paths and assert JAX/JAXlib versions, origins and CPU/x64.
  The base interpreter alone may load different dependency versions.
- Run each affected test group with unchanged bounds. Record terminal results,
  source/input identity, and resource censoring separately. Collection counts
  are not test passes; successful local groups do not establish green CI.
- Keep the completion checklist and evidence references current. Full parity
  requires the original assertions and any stricter existing controls to be
  resolved, alongside the goal's example, figure, performance and CI gates.
