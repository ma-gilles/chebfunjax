## JAX spin comparison and complex C2 construction — 2026-10-17

Public `spincomp` now compares 1D/2D/3D schemes with source sampling,
normalization, fresh preferences and internal stepping times. Public 1D spin
uses the JAX integration engine. Source checkpoint behavior is preserved:
one-output short/nonaligned solves retain the initial output; timed calls
reproduce the missing-third-output error when no endpoint was saved. The
loop stops immediately on the native checkpoint tolerance. Final gates pass
39 1D/fixture checks, four affected 2D/3D checks and six root adapter/render
checks. Six native N=8 cases confirm the output distinction, including a
nonzero initial tspan entry. NaN plot reductions now follow native omission.
Full spin API, long examples, graphics and matched performance remain open.

The default nonperiodic C2 Chebfun3 constructor now carries complex callback
values through native ACA/QR/DEIM/Tucker construction rather than reconstructing
real/imaginary parts separately. All five existing std3-file controls and two
new real/complex constructor controls pass (59.09s, sampled peak 3,008,872 KiB).
These include one literal original MATLAB constant predicate; they are not
seven original MATLAB predicates. Native complex polynomial/conjugate/product
controls have matching core sizes and analytic errors below 1.6e-14. C1 and
periodic complex construction retain their existing paths.

The Chebfun3 wrapper exits 4 because its final runtime auditor found one
unpreguarded dynamic JAX dialect source, despite seven passing tests and
stable source inputs. Post-run inspection proves it is the same file as the
pinned wheel alias; this does not establish a prelaunch freeze. No full runtime
closure, complete Chebfun3 parity, earlier CI shutdown cause, whole-suite green,
CPU performance parity or coverage acceptance is claimed.

Evidence: [parent review](spin_complex_integration_cpu_20261017.json).
The receipt retains failed/superseded observers and the root mock correction;
external dependency preimages remain referenced rather than all embedded.

## PDE option consumers and source color cycle — 2026-10-17

Basic pdeset options now reach polynomial pde15s: Eps, N, scalar/vector AbsTol,
RelTol and AdjustBCs. The sixth positional grid size overrides options.N;
the wrapper overrides options.ODESolver as native pde15s does. Explicit Python
keywords retain precedence. Unsupported active options raise explicitly;
Plot='on', other ODE controls, periodic/complex/mass and native output grammar
remain open. No fixed-48 advection workaround or existing-bound change is made.
Four authored numerical controls, 17 factory controls and three unchanged
heat/Neumann/Robin regressions pass. Four native fixed-grid controls also pass;
heat errors are 1.10e-9 / 1.21e-8 and nominal boundary error with AdjustBCs=False
is 3.12e-17. Spatial Eps forwarding is source-reviewed; full adaptive option
behavior is not qualified by these fixed-grid controls.

Plotting now distinguishes a named Color property from a color in LineSpec.
Only the former consumes a color entry when LineSpec has no color; source
marker/jump artists do not add entries. Native Chebfun and numeric probes
confirm the distinction. The 21-check render/API gate passes; one subsequent
comment-only edit has an independently verified executable-AST bridge. The
exact prior comment preimage was recovered and matches its original guard SHA.
Uppercase Python Color alias and full native graphics grammar remain open.

ThreeBody's two numeric blue-root plots now use the public adapter. Its whole
CPU run passes in 94.311 seconds with stable inputs, all 1,290 actual imports
preguarded and no survivors. All retained rational/pole/residue/grouping
artifacts equal the prior Python run byte-for-byte. Figures 2 and 4 now use
the source orange ellipse; all four remain 600 × 270 and differ in decoded
pixels. MATLAB accuracy, fonts/axes/subplot layout, performance and latest-main
CI/coverage acceptance remain open. The inventory preserves prior figure
observations and adds the new hashes without promoting full-page statuses.

Evidence: [PDE consumer](pdeset_consumer_cpu_20261017.json),
[color cycle](colororder_cpu_20261017.json),
[whole example](pde_color_whole_cpu_20261017.json).
Earlier entries retain their historical source/evidence scope.

## Latest CPU integration and complete figure references — 2026-10-17

The ThreeBody example now calls the public six/seven-output rational API,
public plotregion, and numeric point-quiver adapter. The residue polynomial
recurrence uses a fixed-shape JAX scan; 168 retained cases and 24,205 recurrence
states match the prior implementation exactly, with 105 component regressions.
Optimized CPU HLO removes the explicit barrier: this is evidence for the pinned
CPU cases, not a universal compiler-rounding guarantee.

The combined-source gate passes 135 tests with no skips or failures. All 1,421
loaded runtime files were frozen before execution. The whole example exits
successfully in 94.622 seconds, with all 1,290 runtime imports covered, stable
inputs and no surviving owned processes. Its saved rational inputs, p/q,
poles/residues and pole grouping are byte-identical to the prior Python
observation. This is not MATLAB accuracy or matched performance parity. Four
regenerated figures have the required 600 × 270 dimensions; all differ in
pixels. The cached pole discrepancy, automatic axes, color cycle, subplot
layout, fonts and raster parity remain open.

GET now covers all 38 original predicates plus 45 existing value controls and
one singleton control. The production correction preserves native per-column
delta cells and collapses a singleton to its matrix. Broader GET API and type
coverage remain open. Plotregion now draws source artists by default; optional
`decorate=True` retains the earlier presentation. Complex Chebtech curves use
the paired real/imaginary source plotting route. Numeric point quiver uses
native scaling with readable compatibility heads; eight native metadata cases
and 54 component tests qualify that scope. Native automatic limits and opaque
renderer equivalence remain open.

All 322 official HTML pages and 1,305 official reference PNGs are now archived;
current HTML is byte-identical to the cached target. At source 43e5a5b7 all
1,305 committed PNG dimensions match the reference, and none is decoded-pixel
identical. The new four ThreeBody images have their own source-bound audit.
Reference acquisition and matching dimensions do not verify figure content or
computations. Latest-main CI and coverage acceptance remain required.

Evidence: [combined CPU review](residue_render_joint_cpu_20261017.json),
[GET](get_remaining_cpu_20261017.json),
[residue scan](residue_poly_scan_cpu_20261017.json),
[plotregion](plotregion_native_cpu_20261017.json),
[point quiver](quiver_points_cpu_20261017.json),
[complex plotting](complex_plot_cpu_20261017.json),
[reference archive](official_example_figures_cpu_20261017.json),
[dimensions](reference_figure_dimensions_cpu_20261017.json),
[pixels](reference_figure_pixels_cpu_20261017.json).

Earlier entries below retain their historical source and evidence scopes.

## Running parity checklist and complete source inventory — 2026-10-17

Tracking now has a repository [completed/to-complete checklist](parity_checklist.md),
a [scope/method summary](parity_inventory.md), and machine-readable
parity_inventory.json with2333native library .m sources,1102original test files,
and322cached example pages.3707library/1256test regex declaration counts are
not public API counts. Every full-item parity status begins unverified; candidate
path mappings never confer qualification. Historical scoped evidence is linked
for selected original tests and the ThreeBody observation; other evidence needs
reconciliation into rows. The source-only generator records source pins/hashes,
requires clean mapping inputs, separates MATLAB/Python script paths and refuses
to overwrite the live ledger. Existing checked scopes remain in the checklist.

ThreeBody's optimized whole run now completes131.896s/3.75GiB withboth6/7output
calls and4figures600x270, stable inputs/no survivors. This is an observation,
not page/native parity: cached first8poles differ1.26814e-5 and one actual
stdlib runpy.py import was not preguarded. Native rendering must replace its
local plotregion; highdegree correctness, printed output and pixel parity
remain open. Evidence:three_body_optimized_observation_cpu_20261017.json.

## Spinop2 GL authored test repair

Identical-input MATLAB and JAX normalized norm reevaluations both reject the
old authored32eps expectation. The check now uses512eps with both measured
values adjacent; the2e-14 pointwise comparison is unchanged. Fresh instrumented
focused test passes on isolatedf328 source, all1500actualimports preguarded;
fullstatic checks pass. Native diagnostic has one unavailable incidental prior
guard preimage; full norm/RNG equivalence and latest-main CI remain open.
[Scoped evidence](spinop2_gl_norm_cpu_20261017.json).

## Latest scoped package: polynomial conversion and CI process lifetime

CPU-only qualification:43 polynomial/rational integration checks and6 runner
protocol checks pass. Physical-domain conversion retains intermediate/final
bytes on retained cases; observed shared-host polynomial-pair time79.343s to
28.655s is not final MATLAB performance parity. Five existing3D shards use a
fresh process per file with identical selection/timeouts. Root archive and
runtime-import coverage verified. See [evidence](poly_port_joint_cpu_20261017.json).
Whole ThreeBody, all322 actual pages/figures, remaining original assertions and
API/correctness issues, matched performance and green latest-main CI remain open.

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

- [x] Natural adaptive first-kind numeric parameter backend: original C1
  passes at unchanged5e-10; original C2 regression unchanged. Root C1
  repeat44.028s, exact143runtime-source bridge into combined checkout.
- [x] Source numeric query classification:122component controls and fresh
  combined-source68integration controls pass19.075s. Logical/foreign
  nonnumeric queries reject without conversion hooks; direct Python calls
  retain their separate convention. [Combined evidence](c1_query_joint_cpu_20261017.json).
- [ ] Broader parameter/C1/US APIs, C2 factor eligibility policy, GET values
  and cell/foreign/property dispatch beyond qualified numeric recursion.

- [x] Skip redundant scalar AD extraction for known-local one-sided solves;
  two-sided source routing and nonlocal behavior retained. Fourteen unchanged
  metadata/source-route controls pass, plus the existing numeric system BC
  vector/length-error regression with consistent Python error text. Initial
  memory censor preserved, no benchmark/native-message parity claim.
  [CI routing evidence](ci_routing_controls_cpu_20261017.json).

- [x] Natural typed ultraspherical scalar-parameter backend: literal adaptive
  originalUS and unchanged C1/C2 regressions pass, with native tolerances and
  schedules. Latest-source originalUS repeat41.351s; classification6repeat
  passes19.167s, preserving601 CI fixes.
- [x] Source builtin numeric-result recursion:194distinct component controls,
  native65API capture and fresh latest-source72controls pass15.955s. Native
  public patterns are scoped to errors/shapes/selection; constructor bits,
  cell/foreign recursion and GET values remain open.
  [Combined qualification](ultras_recursion_joint_cpu_20261017.json).

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

## Interpolation / GET / rational output integration (2026-10-17)

- [x] All35 literal interp1 original predicates plus13authored/11supplemental controls; row/Y-Chebfun/trig JAX source paths.
- [x] Scoped GET120controls,33literal originals; root78repeat.
- [x] Rational explicit output-count/symmetry57controls, covering23original clauses; native dispatch/copy diagnosis and root57repeat.
- [x] Combined-source root48/78/57 allpass, stable8228inputs/allimports pregated/no survivors; fullstatic and archives reviewed.
- [ ] GET originals31/35–38 and broader cells/types/empty/family APIs.
- [ ] Actual high-degree ThreeBody/conformal computations, figures and remaining fitting/provider differences.
- [ ] Complete original function/test inventory, all322scientific/output/figure gates, CPU performance and green latest-main CI/coverage.

See [joint evidence](interp_get_rat_joint_cpu_20261017.json). Older open interpolation/GET/rational entries are superseded only within the scoped qualifications above.

## PDE option factory / recursive3D adapter (2026-10-17)

- [x] Public32field pdeset/display factory,17scopedcontrols; captured native NaN/complex/Inf metadata and per-function provenance. Parent fixes whole-vector conversion and pre-parser error order.
- [x] 3D full-slice token translation at recursive delegation; exact pinned badDomain authored expectation;42controls pass after two baseline failures.
- [x] Latest combined20nodes pass23.004s,8236stableinputs/all1440imports pregated/no survivors; fullstatic and completearchive reviews pass.
- [ ] PDE solver mapping consumption and fullfactory invalid MATLAB class/grammar behavior.
- [ ] Original3D1–5/12–18 latest qualification (whole-module attempt censored3GiB, no completedgate credit).
- [ ] Full originaltest/function/page/figure/performance/greenlatestCI requirements remain.

See [joint evidence](pdeset_core4_joint_cpu_20261017.json).
