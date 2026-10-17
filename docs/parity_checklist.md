# Full MATLAB Chebfun parity — checklist

Updated 2026-10-17. **Active, incomplete; CPU only.** Latest updates cover original roots/minandmax predicates, native breakpoint grammar, Hermite arithmetic and stable CI ownership; scoped evidence is recorded below. MATLAB reference: `7574c77`.
[Repository status and evidence](parity_status.md).
[Full source/test/page inventory](parity_inventory.md).
Historical detail is preserved in HANDOFF_CODEX.md and the linked scoped evidence.
Checked items certify only their stated scope.

## At a glance

| Workstream | Verified so far | Still required |
|---|---|---|
| Library and original tests | Scoped fixes and original predicates linked below | Complete function/assertion audit across 2,333 native library files and 1,102 original test files; remaining API/JAX gaps |
| Example pages | All 322 references cached; structural prose/block/figure alignment checked | Final-source computations, printed output and complete qualification of all 322 pages, including the 12 interrupted scripts |
| Figures | All 1,305 reference images archived; all dimensions match at the recorded global snapshot | Scientific content, axes/fonts/layout and raster parity; 0/1,305 images were pixel-identical in that snapshot |
| Numerical correctness | Published GET, rational output, AD/backend, interpolation, plotting and basic PDE option fixes | Inverse transform discrepancy, small Hermite ASY, complex poles, PDE failures, Chebfun3 and broader original-test gaps |
| CPU performance | Scoped measurements and optimizations recorded | Matched native benchmarks for inverse, Hermite, QR/LU, BVP, roots and other known slow paths |
| Final acceptance | Published packages linked below; prior documentation deployment passed | Latest-main CI and coverage ≥79%, final provenance/lint and full objective audit remain required |

### Active work, not yet credited as complete

- **Spin API follow-up:** scoped JAX comparison/checkpoint behavior is published below; full native API, scheme matrix, long examples, graphics and matched performance remain open.
- **Chebfun3 follow-up:** default complex C2 construction now passes the seven-control gate below 3 GiB; C1/periodic paths, broader original tests and complete runtime audit remain open. The earlier CI shutdown cause is unproven.
- **Evidence reconciliation:** historical passing scopes must still be attached to individual inventory rows. Candidate path mappings and passing subsets do not establish full-file parity.

There is **no defensible overall completion percentage** yet. Checked items below qualify only their stated scope, source and bounds; unchecked items remain required.

## Completed scoped work

- [x] Restore all41literal default original power predicates and verify27actual native draw events; final24rerun plus17unchanged-node source/AST bridge, with360production files identical to publication. [Evidence](power_original41_cpu_20261017.json). Extra predicate33warning/fullAPI/preferences remain open.

- [x] Restore native scalar(n,1)denominator root insertion and lossless singleton quotient layout; actual constructors08-12/literal18 retained.68combined parent checks pass; child116pass and native26finalslots true are separately bound. [Evidence](rdivide_scalar_column_cpu_20261017.json). Overwritten native24/full API/preferences remain unqualified.

- [x] Fix actualCIperiodic quantumstates column-coefficient shape failure with JAX diagonal scaling. Seven baseline layout failures become passing;12scientific checks and20CIprotocol checks pass, source bytes bridged to publication. [Evidence](quantum_periodic_column_cpu_20261017.json). Full backend JAX conversion/native API/latestCI remain open.

- [x] Restore all19default original roots predicates and native breakpoint/pointValue grammar; JAX getRootsForBreaks, NaN-padded quasimatrix roots and numeric imaginary suffixes. Combined parent 139controls pass with original14minandmax and LU controls retained. [Evidence](roots_original19_joint_cpu_20261017.json).
- [ ] Full ROOTS APIs/alternate preferences, inherited host numerical paths and latest-main CI/coverage.

- [x] Preserve measured core CI lane ownership as tests are added; all1072current selections retain their lanes, simulated new roots file moves0existing selections versus1004previously.20protocol checks pass. [Evidence](ci_stable_core_cpu_20261017.json). Latest-main execution and coverage remain open.

- [x] Hermite ASY derivative forms native scale before multiplying dval;17existing mapped, independent and normalization controls pass. [Evidence](hermite_derivative_order_cpu_20261017.json). Small-order complex ASY remains open.

- [x] Restore all14default original minandmax predicates with exact source operands/bounds, retaining authored tests separately. Fix string gamma native real-pole convention and denominator jump-root discovery. Parent37 checks pass on current FFT source. [Evidence](minmax_original14_cpu_20261017.json).
- [ ] Remaining host root algorithms and full minandmax/gamma/division APIs/alternate preferences; native breakpoint grammar and JAX filtering are qualified above.

- [x] Split Core1/2 at unchanged process boundaries after observed90minute cancellations;9to11lanes preserve all1072selectors, priorCore8 and scientific/coverage limits.17protocol checks pass. [Evidence](ci_core_deadline_cpu_20261017.json).
- [ ] Actual latest-main core/3D runtime, scientific assertions and coverage>=79%.

- [x] Full current LaplaceBall script executes in34.994s with captured input, stable source/no survivors;11actual outputs retained and page refreshed. Three600x253 PNGs match published bytes; numerical/RNG/render parity gaps remain. [Evidence](laplaceball_current_cpu_20261017.json).

- [x] Complex global extrema now preserve stored isolated point values and native left/right breakpoint repair; array columns follow scalar policy. Six baseline failures become passing; combined30 controls pass. [Evidence](complex_minmax_cpu_20261017.json).
- [ ] Remaining minandmax technology/preference/warning grammar and broader LU pivot fidelity; all14default original predicates are verified above.

- [x] Scalar33 real float64 CPU Fourier arithmetic follows observed native plan; exact-word and algebraic controls pass. Unchanged authored fixed48 PDE CI regression passes without tolerance/integrator changes; combined21 parent checks pass. [Evidence](fft64_pde_cpu_20261017.json).
- [ ] Broader batched/complex FFT plans, inverse/Hermite/rational coefficient gaps, cold/warm overhead and matched native performance.

- [x] JAX LU preserves complex values and source Gaussian-elimination order; public column adapters and length-one constant extrema correction. Native original8true, child20 and combined parent34 controls pass. [Evidence](lu_source_cpu_20261017.json).
- [ ] Full LU pivot/complex breakpoint fidelity, remaining API and matched native CPU performance.

- [x] Correct subspace default Frobenius residual norm (supersedes earlier spectral claim); 27 controls pass, including four native-backed multi-column cases. Deltafun restores seven original predicates and derivative/location/overlap/empty semantics; 20 controls pass. [Evidence](fro_delta_cpu_20261017.json).

- [x] Subspace empty/rank-zero handling and four rank-deficient orderings: final nine controls pass; earlier16-node gate includes original predicates. Native values/errors source-bound, with unrelated moving-main HEAD guard failure preserved and reconciled. [Evidence](subspace_empty_rank0_cpu_20261017.json).

- [x] Localize retained complex-fit assembly difference: native/JAX matched product and mirroring are bit-identical; same-input normalized IFFT differs1.11e-16. Independent native public call proves capture neutrality. [Diagnosis](ratinterp_preifft_cpu_20261017.json); correctness remains open.

- [x] Genuine first-kind Chebfun2 callable preference/grid/refinement/factor path; three original technology predicates pass. Restore all seven no-dots predicates, including exact inner-dimension marker and meaningful operator adapters. Joint15 focused nodes pass. [Evidence](c1_nodots_cpu_20261017.json).
- [ ] Full Chebfun2 C1 algebra/preference grammar and source sample-test order, broader raw division/power/string grammar and global JAX conversion; latest-main CI/coverage.

- [x] JAX subspace QR/SVD and complex inner products (original spectral residual claim corrected above); six original predicates and affected API/fallback controls pass. Unbounded inner product conjugates its first operand; three originals and two complex controls pass. Joint19 nodes pass, plus final domain-check selector. [Evidence](subspace_unbnd_inner_cpu_20261017.json).
- [ ] Remaining subspace orientation/noncollatable API cases and near-rank quantitative fidelity; broader unbounded input grammar.

- [x] Pair all ten original sphere optimization predicates; four Python nodes include two bounded even-polynomial controls. Same maximum antipode in both captures, no code/tolerance change. [Evidence](sphere_extrema_tie_cpu_20261017.json). Historical unspecified operand remains unqualified.
- [ ] Resolve transform-sensitive PDE, Hermite and complex-rational paths; [source-bound diagnosis](numerical_parity_gaps_20261017.md) records what is known and which evidence is missing.

- [x] Repartition 3D CI from five to eight fresh-file shards after the prior90-minute cancellation. All155files and771collected nodes remain disjoint/exhaustive; six process-protocol controls pass. Job90-minute and per-test900-second limits unchanged. [Evidence](ci_3d_repartition_cpu_20261017.json).
- [ ] Actual latest-main CI runtime, scientific assertions and coverage; collection alone does not qualify those.

- [x] Repair spin3 named-preset mock signature for explicit timing flag; eight unchanged parser scenarios pass, now asserting defaultFalse timing as well. [Evidence](spin3_parser_mock_cpu_20261017.json).
- [ ] Latest-main CI/coverage and separate fixed48 PDE FFT/coefficient fidelity; parser checks do not qualify solver trajectories.

- [x] Public Chebfun tensor-query adapter fixes six CI indexing failures while preserving raw Chebtech dimensional errors; combined30 controls pass. Original subspace distinct-breakpoint and row predicates1–6 now pass with native terminal evidence. [Evidence](ndarray_subspace_cpu_20261017.json).
- [ ] Remaining subspace edge cases and near-rank quantitative fidelity; broader ND evaluation and latest-main CI/coverage.

- [x] Simple piecewise Legendre projection and all-column padding; all six literal original legcoeffs predicates plus five supplemental controls pass. Native JSON records six true results; original native terminal receipt is unavailable, so that scope remains result-only. [Evidence](legcoeffs_original_cpu_20261017.json).
- [ ] Non-simple piecewise legcoeffs projection and full API/native invocation qualification.

- [x] Four original MATLAB exporter smoke tests pass, paired to four Python cohorts plus a fixture checksum control. All 64 demo fixtures match native bytes. [Evidence](chebgui_paired_originals_cpu_20261017.json). Generated-text equality and full API parity remain open.

- [x] Public quasimatrix/cheb2quasi and native row/empty-column metadata; rectangle integral2/quad2d and empty integration. Joint 39 Python nodes pass, 17 factory comparisons pass, original cell2quasi predicate and three original integral2 predicates paired. [Evidence](quasimatrix_integral2_cpu_20261017.json).
- [ ] Inherited empty-domain representation, full constructor grammar and broader integration/curve API qualification.

- [x] ComplexArcLength uses source layout/equal-axis/marker adapters. Full run 68.635s, actual 64-point P/Q groups captured, inverse arrays unchanged; four images retain 600×270 dimensions. Figure 4 comparison improves, with raster/marker/typography gaps still open. [Evidence](complex_arc_renderer_cpu_20261017.json).
- [x] Repair the historical handoff-audit link and four grouped exporter navigation mappings. All 64 demo fixtures match native bytes; five Python controls pass. Fresh paired original smoke scope is now recorded below; generated-text and full API parity remain open. [Evidence](chebgui_inventory_mapping_20261017.json).

- [x] Packed Clenshaw: 95 component controls, 59 retained native raw Brent states and four inverse output byte comparisons qualify unchanged arithmetic/outputs. Warm median 6.9312s to 2.2687s (3.055x); native 0.6536s remains faster. [Evidence](inverse_packed_color_cpu_20261017.json).
- [x] Unchanged whole ComplexArcLength completes in 66.112s; input bytes match retained fixture; generated page uses actual stdout and four refreshed 600x270 figures.
- [x] Correct trigremez named-Color double advancement; all 24 combined plotting controls pass.
- [ ] ComplexArcLength representation/stdout precision, inverse 9030 vs native8630 coefficients, visible axes/markers/layout/raster, broader performance and latest-main CI/coverage.

- [x] Restore strict C1/C2 large-input NDCT dispatch; fix first-import JIT tracer leak; 23 authored forward/AD/shape controls pass, eight same-input native captures agree within 2.45e-15. Real-node polynomial derivatives preserve strict 2e-12 controls. [Evidence](chebtech_ndct_cpu_20261017.json).
- [x] Pair all 18 original Chebtech feval predicates with exact MATLAB-seeded points; native fixture-word check passes; old linspace tests retained. Forward source bridge carries the passing scope to final candidate. [Evidence](chebtech_feval_seeded_cpu_20261017.json).
- [x] Whole BestApprox CPU rerun 151.93s, stable source/no survivors, stdout aligned, generated page unchanged and five reference-sized figures.
- [ ] BestApprox rational accuracy (8.86e-8 sampled curve gap), reference pixel/layout parity, matched performance and latest-main CI/coverage.

- [x] Pair five default original MATLAB invocations (141 predicates) with 133 passing Python nodes at c0a1bca9; attach source-bound scopes to individual inventory rows. GET38/interp135/ratinterp23/compose44/std3one. [Evidence](original_five_predicates_cpu_20261017.json). Full API and alternate preferences remain open.

- [x] Public JAX spincomp and 1D spin engine; native short/nonaligned checkpoint and timed-output error behavior. Final 39 + 4 component controls and six root adapter/render controls pass; six native output cases captured. [Evidence](spin_complex_integration_cpu_20261017.json).
- [x] Native complex default C2 Chebfun3 construction: seven scientific controls pass, with source-stable inputs. Final runtime auditor exits 4 for one dynamic JAX file outside preguard; no full runtime closure claim. [Evidence](spin_complex_integration_cpu_20261017.json).
- [ ] Full spin API/schemes/long examples/graphics/performance; C1/periodic complex Chebfun3 and broader original tests; latest-head CI/coverage.

- [x] Working pinned JAX/JAXlib0.11.0 CPU/x64 environment and shared writable checkouts.
- [x] Explicit-path staging, ordinary direct-main pushes, source hashes and delta-bundle verification.
- [x] C1compose41focusedchecks; C2defaultnested38focusedchecks (11legacy mapped, not44literalnative).
- [x] Tiny rational minimax JAXQR19checks; native errmin reporting17unchangedchecks; combined8subset with all1423runtimefiles preguarded.
- [x] Mixed linear numeric/function columns49checks; scalar source-AD route73checks; combined14subset.
- [x] Original parameter predicates:16linearization+2aggregate+4parameterBC, defaultadaptive solves/unchanged source bounds; natural C1/US original predicates additionally qualified/published17a/4d.
- [x] Subsref39focused checks;12original clauses restored, existing12/23 separately credited.
- [x] Nested derivative order54; sampledroots17; realnegativeAiry165passes/2existing strictxfails.
- [x] Public Trigtech128checks (14originalfeval clauses), Spherefun30, JAX2D85, galleryAiry29.
- [x] Earlier Clenshaw79/shared1D132, finite-difference100, PDEarithmetic149, coupledrectangular51 and other historical scoped packages.
- [x] Original roots stress e95 passes both native clauses within900s; latest/full native performance still open.
- [x] Original mapped PDE technology-session predicate at0b5 passes unchanged; documented adapters retained.
- [x] Full DDE23outputs/24figures and Tricky12outputs/9figures captured/published at their recorded older source heads; selected Airy cells corrected later.
- [x] Cached322page prose/equations/2408blocks and1305figure slot/order/size structural alignment.
- [x] CIdeadline repartition collection/protocol qualification:12943core+affectedportnodes same union,17protocolchecks; scientific/time/coverage gates unchanged.

- [x] Public nonperiodic preference propagation52scopedcontrols and exactlyrealcomplexAiry dispatch correction published717f057f; joint12+8existingintegrations pass.

## Recent integration packages

- [x] Qualify combined GET/plotregion/complex plotting/point-quiver package:135integration checks pass, all1421actualimports preguarded; wholeThreeBody completes with unchanged retained numerical bytes.
- [x] Review residue scan exact-case bridge and105regressions; optimizedHLO barrier limitation explicit.
- [x] Correct supported Color-property versus LineSpec cycle semantics;21render/API checks plus native Chebfun/numeric metadata, comment-only AST bridge verified. [Evidence](colororder_cpu_20261017.json).
- [ ] Complete native automatic axes, broader graphics grammar/cycles, subplot and renderer behavior; latest-main CI still required.
- [x] Qualify affine polynomial conversion optimization; identical retained coefficient bytes,105 component regressions and43 root integration checks pass.
- [x] Complete root import/evidence review and publish optimization5bbf557c; observed2.77x shared-host polynomial pair is not final performance parity.
- [x] Qualify/publish selective 3D per-file CI process lifecycle controls5bbf557c;6 protocol checks and exact workflow bridge pass.
- [x] Review/publish authored Spinop2 bound repair45c04b29; fresh actual-process1500imports preguarded, focused test passes. Native incidental prior-guard preimage gap explicitly scoped.
- [ ] Green latest-main CI and coverage acceptance. Historical runs remain evidence for their own source only.

## Next integration

- [x] Isolated interp1 candidate passes all35 literal original predicates; baseline27pass/8fail retained. This is candidate qualification, not publication.
- [x] Finish24 additional interp1 controls, archive/review, integrate and publish5105d431; root48repeatpass26.607s.
- [x] GET candidate qualification:120distinct controls/33original predicates. Rational output candidate:57distinct controls/all23original clauses; rational parent archive review passes.
- [x] Complete GET source/archive review and fresh combined interp1/GET/rational integration; publish5105d431. Root48/78/57allpass; fullstatic/remote/clean/bundleverified.
- [x] Restore and qualify GET originals31/35–38; all38original predicates plus45value controls and1singleton control pass. [Evidence](get_remaining_cpu_20261017.json).
- [ ] Broader GET cell/type behavior and empty/API cases.
- [x] Publish pdeset factory/native metadata qualification and recursive 3D selector fixes f3285a81; combined20 checks pass.
- [x] Integrate basic pdeset fixed-grid consumers: Eps/N/AbsTol scalar-vector/RelTol/AdjustBCs and native ODESolver override;24checks and4native comparisons pass. [Evidence](pdeset_consumer_cpu_20261017.json).
- [ ] Complete remaining PDE options/plots, adaptive-option qualification and native output grammar; spincomp has the scoped published qualification above; full API/graphics/performance remain open.
- [x] Run combined ThreeBody adapter alone: exit0/94.622s,1290imports preguarded, finite6/7output poles/residues,4figures600×270; saved numerical artifacts byte-identical to prior Python observation. [Evidence](residue_render_joint_cpu_20261017.json).
- [ ] Close ThreeBody MATLAB accuracy/stdout/figure-content and raster differences.


- [x] Review/archive/publish all44literal compose predicates +6scalarcolumn controls and source-supported resampling/error/shape corrections (main66bb7dea). Root8integration repeats pass; fullpublicpref/API scope remains open.
- [x] Reconcile all13 original handoff issues plus8 smaller subitems against current source/evidence. [Detailed audit](handoff_section6_reconciliation_b16_20261017/REPORT.md). Next: small Hermite ASY and matched actual-example inverse qualification.
- [x] Review public Bndfun.compose/DATA and cross-kind C1/C2 candidates; combined81 controls pass on frozen source (unpublished).
- [x] Archive/final review/publish Bndfun + cross-kind package9a22b851; combined81pass, fullstatic/cleanremote/bundleverified.
- [x] Diagnose exact-input inverse rawBrent: all59 raw states match native; source-incorrect global endpoint override identified.
- [x] Qualify/publish endpoint correction f45269ec:52testnodes plus7rootintegration repeats.
- [ ] Resolve remaining9030vs8630 transform-coefficient gap; identicalsamples/chop crossreplay now identify first divergence.
- [x] Qualify/publish real-valued2D output dtype fix f45269ec:20distinct controls including all3CI failures; root12repeat.
- [x] PublishPoisson900sfix4b5f8f98:26controls plusall9originalassertions193.16s underunchanged900s; originaltestbyteunchanged,8082archivepayloads verified. LatestCI stillrequired.
- [x] Publish corrected JAX-only Subsref candidate af5f3828: all32 original predicates covered across component gates/exact bridge; parent archives verified; latest-source12integration13.58s/fullstatic pass. Broader record API gaps remain open.
- [x] Publish representation-based happiness/vscale correction787922f8:153componentcontrols; root27integration30.48s,8185stableinputs/all1413runtimeimports pregated, fullstatic/remote/bundleverified. No FFT/inverse length or performance closure.
- [ ] Obtain green latest-head CI/runtime/coverage after published fixes. Prior3D runner shutdowns remain separate; latest run must be checked live, no manual cancellation.

- [x] Publish nonlinear AD classification-before-linear-prepare18823ffc:13componentnodes including literaladaptiveC2 original, rootrouting11+adaptive1 repeatpassed46.55s at unchanged5e-10bound; original C1/US/defaultbackend cases remainopen.

- [x] Publish native record selector/index semantics7d488479:116component controls, root68latest-source controls pass18.89s; native48 error/shape/column comparisons; full static/clean remote/archive verification.
- [x] Publish typed-parameter C1 backend17a38188: literal adaptive C1 and unchanged C2 pass; independent root C1 passes44.03s. Final146payload/35pinned source/283575guard-edge review and combined integration passed. Ultraspherical not closed.
- [x] Publish numeric-query type correction17a38188:122distinct component controls pass; fresh combined-source root68 integration passes19.075s. Numeric recursion remains open.

## Correctness and full test parity

- [ ] Audit every original function/assertion; mapped-file counts and passing subsets are insufficient.
- [x] Publish CI local-classification AD-skip and numericBC wording correction601f9da2:15unchangedtests pass,8148archivepayloads verified, fullstatic/cleanremote/bundleverified. Older3d-e runner shutdown tracked separately.
- [x] Publish naturalUS backend4d4c2259:15componentnodes including unchangedliteralUS/C1/C2 originals, rootlatestUS1pass41.351s;173payloads/33pinnednativesources/422590guardedges independentlyreviewed. BroaderbackendAPIs open.
- [x] Publish numeric-result recursion4d4c2259:194distinctcontrols, rootlatest72pass15.955s andmetadata6pass19.167s;8108payloads/129952guardedges reviewed. No cell/GET-value/nativeconstructorbit closure.
- [ ] Broader GET cells/types/empty/family behavior; all38 original predicates are now covered, without whole-API closure.
- [x] Publish rational explicit six/seven output source semantics and arbitrary-node symmetry5105d431; unchanged original23 clauses covered.
- [ ] High-degree complex fitting/poles/residue accuracy and actualThreeBody scientific parity.
- [ ] C2 factor eligibility and newly retained positive-order manufactured Newton convergence failure (unpublished diagnostic; native behavior not yet compared, not a confirmed parity bug).
- [ ] Authored fixed48CIadvection integration failure (distinct from passing mapped original PDE); FFT/provider/source representation.
- [ ] ChemicalPDE lateaccuracy/constructor/LU differences.
- [ ] Broader nonlinear parameter/default setup APIs and original predicates beyond the qualified C1/US/C2 cases; interpolation invalid/API semantics and broader Subsref record behavior.
- [ ] Composition/division strict shape/length candidates; reassess after source sampling fixes.
- [ ] Chebfun3T original predicate7; smallHermiteASY/complexAiry strictxfails and broader specialfunctions.
- [ ] Broader QR/LU/QZ/AD/rational/conformal/ODE/periodic/sphere/TreeVar/API omissions/skips/xfails.
- [ ] Exact normalRNG constants/generation/production integration and example draw ordering.
- [ ] Repository-wide JAX-only numerical implementation; remaining host NumPy/SciPy paths.

## Pages, output and figures

- [x] Rerun ThreeBody with public numeric plot adapter:94.311s,1290actualimports covered, unchanged retained numerical bytes; orange ellipse corrected in figures2/4. [Evidence](pde_color_whole_cpu_20261017.json).
- [x] Cache/archive all322officialHTML and1305referencePNGs; HTML matches cached target. [Receipt](official_example_figures_cpu_20261017.json).
- [x] Audit all1305committed figure dimensions at43e5a5b7: allmatch. All1305decoded pixels differ; no content/parity credit. New4ThreeBody images separately audited.
- [ ] Qualify all322actual computations/stdout and all12handoff example scripts on final affected source.
- [ ] Complete current Tricky/Krylov execution and native scientific/representation parity.
- [ ] DDE exactDDESDmesh/stdout and figure17clipping; no unprovedepsilon fix.
- [ ] All figuredata/RNG/cameras/axes/markers/fonts/raster at reference sizes; cached structure alone does not prove this.
- [ ] Fresh affected regeneration and complete final page/figure audits; historical27page list is not current count.

## CPU performance and final acceptance

- [ ] Matched native CPU inverse setup/warm throughput and representations; historical improvement not full parity.
- [ ] Hermite fast/small methods and scale; QR compilation/memory, LU throughput/tracing, BVP coldsolve.
- [ ] Bartels–Stewart largeheat, roots end-to-end and known slow paths; heavy examples serial.
- [ ] Green latest-main CI and actual coverage≥79%; no upstream-failure substitute.
- [ ] Final provenance/lint, clean publication and requirement-by-requirement audit of the full original objective.

Do not mark complete while required tests/pages/figures/correctness/performance/CI remain unresolved.
