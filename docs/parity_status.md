# MATLAB Chebfun parity checklist

Updated 2026-10-17. **Active and incomplete; CPU only.** Reference:
`7574c77680d7e82b79626300bf255498271a72df`. Checked entries certify only the
scope stated. There is no verified overall completion percentage or current
remaining-page count.

## Completed, scoped work

- [x] Pinned JAX/JAXlib 0.11.0 CPU/x64 environment and direct-main publication.
- [x] C1/C2 source coefficient-grid composition; all 44 original Tech compose
  predicates and six one-column controls now pass. The former 22 adapted
  controls remain supplemental. See [composition evidence](chebtech_compose_original44_cpu_20261017.json).
- [x] Minimax QR SVD and native exchange-error reporting; eight combined
  minimax/power/endpoint regression checks pass with the composition changes.
  See [integration evidence](chebtech_compose_original44_integration_cpu_20261017.json).
- [x] Scoped mixed linear-system, source AD, parameter boundary-condition,
  derivative-order, roots, Airy, public Trigtech and Spherefun corrections.
  These are not whole-module completion claims; see HANDOFF_CODEX.md.
- [x] Original mapped PDE technology-session predicate passes unchanged;
  this differs from the still-failing authored fixed-48 advection control.
- [x] Cached structure for 322 pages, 2,408 code blocks and 1,305 figure slots
  aligned. This does not qualify computations, printed output or image pixels.
- [x] All 13 original handoff issues and eight smaller subitems reconciled
  against source/evidence: [issue-by-issue list](handoff_issue_reconciliation_cpu_20261017.md).
- [x] CI timeout repartition preserves test membership and existing gates;
  actual latest-head runtime and success remain unverified.

- [x] Public C1/C2 nonperiodic composition preference propagation: 52 scoped
  controls across preserved runs; source first-piece fallback, raw-operator
  fixed/Turbo behavior and custom refiners retained. See
  [preference evidence](nonperiodic_compose_preferences_cpu_20261017.json).
- [x] Exactly-real complex negative-axis Airy correction: 39 first-candidate
  controls, final affected-control repeats and current public integration.
  See [joint integration](compose_airy_joint_cpu_20261017.json).

- [x] Public smooth Bndfun.compose and constructor DATA scales, with nine
  restored original predicates and preserved supplemental legacy controls.
- [x] Cross-kind C1/C2 outer composition and native minimum-length/column
  checks. Combined qualification: 81 controls, including the nine originals.
  See [combined evidence](bndfun_crosskind_integration_cpu_20261017.json).

- [x] Source inverse callbacks return raw algorithm results; optional final
  rangeCheck remains. 52 test nodes pass across separate frozen processes;
  this is not 52 literal MATLAB predicates. All actual callback samples match
  native for the retained input; transform length/speed parity remains open.
- [x] Real-valued Fourier CDR outputs retain the dtype of evaluated products;
  20 distinct controls include all three prior CI failures and complex/JIT/AD
  controls. See [joint integration](inverse_cdr_joint_cpu_20261017.json).

- [x] Large-rank CDR evaluation used by Poisson: all nine original Poisson
  assertions pass unchanged in 193.16 seconds, within the existing 900-second
  timeout. Equal genuine C2 slices use ordered matrix Clenshaw; 32-column
  stacks and a final block concatenation avoid the large assembly cost.
  All 26 authored value/AD/fallback controls pass. See
  [Poisson evidence](poisson_grid_chunk_cpu_20261017.json).

- [x] All 32 original Chebfun Subsref predicates covered through frozen
  component gates and an exact source bridge. Corrected JAX-only record
  helper passes 54 current controls; total 88 distinct component controls.
  Root 12 latest-source integration controls pass in 13.58 seconds.
  [Evidence and remaining record API gaps](subsref_jax_source_cpu_20261017.json).
- [x] Current source bindings for the original 12 interrupted scripts
  reconciled with their retained acceptance evidence. This is a source-only
  audit, not new numerical or figure qualification. See
  [script dispositions](handoff_twelve_current_bindings_cpu_20261017.json).

- [x] Native happinessCheck local DATA scale derives from the coefficient
  representation, preserving independent supplied-value scales and running
  constructor metadata. 153 component controls pass; 27 latest-source
  integration controls pass in 30.48 seconds. Native FFT and inverse length
  parity remain open. See [scale evidence](happiness_representation_vscale_joint_cpu_20261017.json).

- [x] Classify source AD linearity before optional mixed-linear preparation;
  the original natural C2 nonlinear parameter setup now passes adaptively at
  the unchanged bound. Final component13 nodes include11 routing controls,
  fixed48 supplemental and one literal adaptive original. Root latest-source
  routing11 and original-adaptive1 pass separately; C1/US remain open. See
  [nonlinear dispatch evidence](nonlinear_parameter_dispatch_joint_cpu_20261017.json).

- [x] Native programmatic record column selection now preserves max/if
  ordering, logical/complex/NaN quirks and computed outCols validation.
  Component116 controls pass; latest-source68 integration controls pass in
  18.89 seconds. Native48 comparisons qualify selectors/errors/shapes, not
  constructor/output-bit parity. Remaining query-type/recursive/property
  gaps are explicit in [index evidence](subsref_record_indices_cpu_20261017.json).

## Correctness and original test parity

- [ ] Every original function and test predicate; remaining omissions, adapted
  assertions, skips, strict xfails and host NumPy/SciPy numerical routes.
- [ ] Singfun/Unbndfun/heterogeneous composition and native warning/API
  gaps; remaining invalid-input and future JAX list-DATA behavior.
- [ ] Authored fixed-48 advection control and chemical PDE accuracy.
- [ ] Small explicit Hermite ASY, nonlinear parameter/backend cases,
  interpolation norms/grids, broader Subsref records and Chebfun3T predicate.
- [ ] Strict composition/division representations, general rational poles,
  conformal inverse counts, IVP representation, QR/LU/QZ and other API gaps.
- [ ] Native normal RNG values and production integration/draw ordering.

## Pages, figures and speed

- [ ] All 322 actual computations and printed outputs; all 12 handoff scripts
  on the final affected source. Historical 295/322 and 27-page counts are stale.
- [ ] Current complete Tricky/Krylov qualification; DDE meshes/output/clipping.
- [ ] Every figure's data, RNG, cameras, axes, markers, fonts and raster at
  reference sizes; final regeneration and full audits.
- [ ] Matched native CPU inverse, Hermite, QR/LU, BVP, Bartels-Stewart and
  roots performance. Historical speed improvements are scoped observations.
- [ ] Green latest-main CI with coverage at least 79%, final JAX-only,
  provenance/lint checks and requirement-by-requirement acceptance audit.

Do not mark the goal complete while any required test, page, figure,
correctness, performance or CI requirement is unresolved.
