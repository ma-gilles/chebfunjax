# Parity status — 2026-10-09

**Full parity is incomplete.** The target is MATLAB Chebfun commit
`7574c77680d7e82b79626300bf255498271a72df`, all native functions/tests, and
all 322 example pages, including computations, prose, outputs and figures.
Work is CPU-only. [HANDOFF_CODEX.md](HANDOFF_CODEX.md) takes precedence over
the July handoff. No defensible overall completion percentage is available.

## Latest verified changes

| Local commit | Change | Qualification |
| --- | --- | --- |
| `17fb15c6` | AtmosphericTemperature captured cameras and contour framing | Full source replay plus isolated corrected figure 3; all 10 images; historical graphics/scientific-reference differences remain |
| `cf3cde14` | Ballfun default source slices and coefficient emptiness | 27 distinct checks across staged runs plus a normal-pytest artifact check; native lighting remains open |
| `f5d7d450` | SolidHarmonics full source page | All 13 constructions, including degree 150; 11 public plots/55 surfaces; two 600×253 figures; lighting/layout differ |
| `510b3f8a` | WaveDecay adaptive eigensolves and continuous normalization | Full source run, two 600×480 figures and eight matching formatted eigenvalue labels; solver/rendering parity remains open |

These are scoped qualifications, not a complete passing suite or exact-head CI.

## Remaining gates

- **Tests and functions:** complete semantic coverage and full CPU suite.
  Exact-commit inventory at `1bacf226` maps all 1,102 native test files:
  1,091 have no literal skip/xfail marker; 11 contain markers. File presence
  is not complete predicate coverage or a passing execution result.
- **Figures:** fresh audit at `510b3f8a` checks 1,305 mapped image pairs with
  no mapping holes. **71 dimension mismatches remain across 14 pages.**
  Correct dimensions do not establish matching content, colors or rendering.
- **Correctness:** trigonometric constructors and mixed-column realness,
  rational poles, Hermite edge cases, eigensolver semantics, difficult
  examples and the remaining source-linked handoff backlog.
- **Performance:** inverse-function and other slow paths need qualification;
  no full MATLAB performance-parity claim is supported.
- **Implementation:** remaining host numerical paths must become JAX-only.
- **Acceptance:** native MATLAB comparisons, publication and final green CI.
  Latest observed session failures are GitHub network/SSH access and MATLAB
  IPC startup. Last confirmed remote main was `8a6d9c80`; newer listed commits
  are local. Useful local implementation and verification continue.

## Current work

The C2 eigensolver package restores all four original basic predicates and
passes 14 controls plus four affected consumer/dispatch checks. Its inherited
host eigendecomposition and other backend/adapter semantics remain open.
Trigonometric column metadata is committed with 59 passing checks. The next
constructor package has passed 38 analytical and nine native tests; composition
regressions and public callback/preference routing are still being qualified.
Solid harmonics now pass all 14 original predicates and 25 controls, with a
fresh full-page run including degree 150. New recurrence and radial contraction
code is JAX; inherited Ballfun host numerics and rendering differences remain.
The constructor package is still uncommitted. These scoped qualifications are
not full-suite or exact-head CI evidence.

Resource-heavy examples run serially; independent bounded work uses up to
three worker slots. Work remains CPU-only.

## Evidence

Immutable run receipts, runtime hashes and reviews are under:
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`

- `atmospheric_composite_root_integration_20261009.json`
- `ball_plot_composite_root_runtime_review_20261009.json`
- `ball_plot_ci_fallback_root_runtime_review_20261009.json`
- `solid_harmonics_root_integration_20261009.json`
- `wavedecay_root_integration_20261009.json`
- `matlab_test_static_inventory_1bacf226_20261009.json`
- `figure_size_current_510b3f8a_20261009.json`

[PARITY_MATRIX.md](PARITY_MATRIX.md) and the handoff retain dated historical
qualification records. The old initial-port phase labels and percentage in this
file did not establish MATLAB parity; their historical inventory remains in git.
