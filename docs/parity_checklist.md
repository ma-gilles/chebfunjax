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
- **Evidence reconciliation:** roots (19 predicates), power (41) plus bounded zero preflight, minandmax (14), and rdivide (26 final slots / 28 assignments) now have precise original-test/function entries with source references and limitations. [Reconciliation](scoped_originals_inventory_reconciliation_20261017.json). Remaining historical scopes still require review; candidate paths and passing subsets do not establish full-file parity.

There is **no defensible overall completion percentage** yet. Checked items below qualify only their stated scope, source and bounds; unchecked items remain required.

## Completed scoped work

- [x] All6default original system2 cell/autonomous predicates pass onfc at unchanged1e-9; no library edit, stale unsupported callback entry corrected. New regression+21protocol checks preserve1088prior owners+1new. [Evidence](original_system2_polyfit8_cpu_20261017.json).
- [x] All8default original polyfitL1 predicates pass on frozen36e at unchanged1e-8; fresh unmodified native testfunction also all8true. [Per-clause table](polyfit_l1_original8_current_cpu_20261017.md).
- [ ] Broader API/backend/performance and latest-source/CI qualification remain open; supplemental external cosine warning remains separate.


- [x] Repair3recorded nonfinite ABS mask cases through shared C1/C2 transforms:24new native-stage,16exact C1/JIT/AD,44original transform,111parent tech/native/protocol,3unbounded and20complex AD nodes pass (counts overlap). [Evidence](nonfinite_transform_native_cpu_20261017.json).
- [ ] Full nonfinite/extreme/storage/API and native FFT-word coverage, matched CPU speed and final CI remain open;8captured cases are a scoped qualification.


- [x] Bounded source cosine analytic derivatives:15new+8existing checks and unchanged fixed48 PDE pass;1040independent native vector/scalar and50actual sample words exact; parent21protocol checks. [Evidence](bounded_cosine_parent_cpu_20261017.json).
- [ ] Supplemental polyfitL1(cos,4) warning parity remains open: two external cosine sample words differ1ULP; same-input transforms agree. [Diagnosis](polyfit_cos4_first_divergence_cpu_20261017.json). No constructor/provider or full solver closure.


- [x] Source existing-grid C1/C2 ABS storage dispatch and representation:87parent tech/original/protocol checks and3literal unbounded assertions pass, all1085prior core owners preserved+1new. [Evidence](abs_existing_grid_parent_cpu_20261017.json).
- [ ] Native nonfinite transform-mask differences in3cases, full ABS/API/JAX and matched speed remain open; storage dispatch success is not full output parity.


- [x] C1 native adaptive eigs and explicit system preferences:6focused checks,39existing regressions and21CIprotocol checks pass; fresh pinned native system2 has6true predicates. [Evidence](c1_eigs_adaptive_cpu_20261017.json).
- [ ] Broader cell/BC grammar, full eigs/backend/JAX qualification and matched performance remain open; these numerical clauses do not qualify the whole original test file.


- [x] Reconcile the twelve interrupted scripts:12committed replacements and12clean historical complete-run receipts found; original dirty payloads preserved and superseded. [Per-script table](handoff_twelve_disposition_review_cpu_20261017.md).
- [ ] All twelve still need final-source numerical/stdout/scientific-figure qualification. Historical source/renderer/adapter bridges and exact next steps are listed per script; no new full-page pass is claimed.


- [x] Native barycentric operation boundaries repair actual EQUI74vs72CI failure:902native sample words exact,39bary/native+12FUNQUI/EQUI+35original interp1checks pass. Combined41node gate passes,1083prior core owners preserved. [Evidence](bary_rounding_joint_cpu_20261017.json).
- [x] Four-stage sampled-root recurrence:141pytest+48direct controls,20inverse artifacts identical; local warm median1.752641→1.611100s. [Evidence](inverse_sampled_four_stage_cpu_20261017.json).
- [ ] Small-query/native coefficient low words/full APIs and matched performance remain open; sampled capacity compilation worsens about6%, dominant cold compilation remains unresolved. Latest-main scientificCI/coverage required.


- [x] Projected C2 system eigenproblem repair: both actual CI regressions, four original native predicates,38child scientific selectors and24parent integration nodes pass (counts overlap). Complex eigenfunctions and genuine repeated modes preserved. [Evidence](maxwell_system_eigs_parent_cpu_20261017.json).
- [ ] Full eigs/BC grammar, wider C1 preferences, generalized-eig JAX conversion, matched performance and latest CI remain open.
- [x] Matched-input single-CPU inverse benchmark: MATLAB warm0.642599s, JAX1.710633s; matching8630coefficient counts and nearroundoff sample agreement. [Evidence](inverse_single_cpu_20261017.json).
- [ ] Inverse speed parity remains open:2.662xwarm and32.8xfirst-after-setup in this descriptive shared-host measurement; full API/pages/figures and other known slow paths remain required.


- [x] Four-stage ordered CPU Clenshaw:90existing scientific plus64original-source checks pass; actual inverse3warm median2.3134s→1.7380s,20output artifacts identical. [Evidence](clenshaw_four_stage_cpu_20261017.json).
- [ ] Matched native CPU performance and full inverse/Clenshaw API/page/figure qualification remain open.


- [x] Native-motivated real CPU radix-two C2R DCT-I family:254scientific checks plus21parent CIprotocol checks; actual inverse8630/native8630 coefficients and nearroundoff value/composition controls. [Evidence](inverse_c2r_parent_cpu_20261017.json).
- [ ] Inverse full API/low-word/page/figure and matched CPU performance remain open; dense724 strictxfail unchanged.


- [x] Native complex baryWeights capacity/phase and analytic rotation JVP:12focused checks and4fresh native controls pass; real arithmetic/other function ASTs/old test bodies unchanged. [Evidence](bary_complex_cpu_20261017.json).
- [ ] Full baryWeights API/large-n/performance; latest-main CI/coverage and Maxwell duplicate-zero system eigenmodes remain open.


- [x] ActualCIinterp1predicate25 source-loop correction:65interpolation/bary checks pass unchanged bounds,2complex-JVP boundary checks. Five minimax scalar/shape adapters pass on finalsource. [Evidence](interpolation_minimax_ci_cpu_20261017.json).
- [ ] Native low-word/weights/full APIs, matched speed and latest-main scientificCI/coverage remain open.


- [x] Compare real-even C2R transforms on retained3inverse stages and same native363cubed tensor. Neither qualifies a production fix; reject coefficient-count matching as acceptance. [Evidence](transform_c2r_diagnostics_cpu_20261017.json).
- [ ] Native actual724/32768plans, dense3Tstrictxfail and inverse representation/performance remain open.


- [x] Coupled complex derivative towers use existing JAX native IVP engine:final44controls,56earlier bridged controls and full221.80s ThreePlanets with0SciPy integration calls. Parent21CIprotocol checks pass;1080priorowners retained plus1selector. [Evidence](coupled_ivp_parent_cpu_20261017.json).
- [ ] Native chaotic trajectory, full higherorder/complex grammar, cached warning, page/figure parity and matched performance remain open.


- [x] Fix explicit native nobreaks exchange call and split actualCore7deadline11to12lanes, preserving all1080selectors/process batches.29checks pass. [Evidence](ci_core7_cpu_20261017.json).
- [ ] Latest-main actual scientific CI, new-lane runtime and aggregate coverage remain required.


- [x] Public ChebPadé padding uses existing JAX normal provider:4baseline failures repaired,51regression and6additional matched-native-draw checks pass. [Evidence](public_chebpade_padding_cpu_20261017.json).
- [ ] Exact MATLAB RNG bitstream, full CF/ChebPadé APIs, rank warnings and matched performance remain open.


- [x] Legacy ChebPadé JAX fitting/evaluation, complex data and physical domain:8initial failures repaired,15finalchecks pass,7fresh native numerical controls. [Evidence](legacy_chebpade_cpu_20261017.json).
- [ ] Full ChebPadé API/original tests, exact rank-warning semantics, MATLAB RNG bits and matched performance remain open.


- [x] ThreePlanets actual warning visibility: globalignore removed, fullapplication219.22s/4unchangedframes/0warnings/all1582runtimefiles preguarded. Parent reconciles preserved root-only resource-observation race without changing rawreceipt. [Evidence](threeplanets_parent_cpu_20261017.json).
- [ ] ThreePlanets cached65537warning/nativechaotictrajectory/raster and existing complex highorder SciPy LSODA provider remain unqualified; original page slot retained.


- [x] Padé native empty optional defaults:None/emptylist/emptyJAXarray accepted; zero tolerance preserved. Six baseline failures fixed;23finalCPUchecks pass plus fresh pinnednative empty/default/zero controls. [Evidence](pade_optional_defaults_cpu_20261017.json). Wider API/numerical/performance/latestCI remains open.


- [x] Close Lorenz fourth blank outputslot with actual captured complex-plot warning and explicit source-slot markers. Full final33.63s/all1381runtimefiles preguarded, exact non-outputMarkdown, allfour600x270figures. [Evidence](lorenz_warning_cpu_20261017.json). Six other targeted blanks/three count differences and full numerical/raster parity remain open.
- [x] Diagnose TransientGrowth originalconstructor warning:28happyFUNs/2032coefficients,0actualwarning attempts/emissions; no library fix or synthetic warning justified. [Evidence](transient_growth_constructor_cpu_20261017.json). Fullpage and freshnative warning difference remain unqualified.


- [x] Review exact ten-page mapping of seven blank output slots and three count differences:72archive payloads/68snapshot files verified; no missing numerical print among these specific slots. Ten page ledger rows linked. [Evidence](output_slot_reconciliation_cpu_20261017.json).
- [ ] Resolve those warning/environment/version differences with actual execution evidence; do not synthesize cached warning output.
- [ ] Dense Chebfun3T original constructor7 still fails unchanged100eps after current shared transform changes; fresh error2.864e-14. [Current failure](chebfun3t_current_transform_cpu_20261017.json). Strict xfail retained.


- [x] Fix coefficient-break scalar C2 Newton dispatch and merged-domain RHS sampling; preserve scalar callable broadcast. Child7joint controls and final4focused pass; actual Bloodhound233s/7iterations succeeds with zero actual warnings. Explicit source branch bridge retained. Parent20CI protocol checks preserve1079owners and append one selector. [Evidence](bloodhound_parent_cpu_20261017.json). Native convergence/warning, full API and page/figure parity remain open.


- [x] Padé JAX FFT/Toeplitz/SVD/QR/roots/evaluation port; zero degree and tiny complex numerator repairs. Native original3true; final17checks pass, including four baseline failures and jit/complex-JVP evaluation. [Evidence](pade_jax_source_cpu_20261017.json). Full API, exact numerical and matched performance qualification remain open.


- [x] Split actual90minute MATLAB-port chebfun-a/b deadline failures:3→9lanes preserve390files/eightroots, oldlaneancestry, runner/steps,900second tests and90minute job limits. YAML/directorycoverage/exactinventory/source-byte checks pass. [Evidence](ci_port_deadline_cpu_20261017.json). Latest-main runtime/scientific assertions/coverage remain required.


- [x] Restore reference prose after moving27implementation notes from21pages into preserved evidence documents. Fresh current-tree offlineaudit:322pages,0parser-normalizedprose/equation/header mismatches;2408exact MATLABinputblocks and1305figure-reference positions/interleaving agree. [Evidence](page_structure_prose_parent_cpu_20261017.json). Runtime/stdout/pixel/content qualification remains separate;7blank outputslots and3historicalwarning-count differences remain.


- [x] Fresh full ThreeBody after FFT602 integration:104.57s,1380actual imports preguarded, stable/no survivors. Fresh415coefficients match retained input words; bothfits and allfour600x270figures complete. Five actual stdout blocks and two changed PNGs regenerated. [Evidence](threebody_fft602_current_cpu_20261017.json). Full nativeODE/second-fit/renderer/performance parity remains open.


- [x] Integrate source CPU602 Fourier/C2R arithmetic, direct inverse normalization and omitted Chebfun-domain inheritance. Final129checks pass, including all23literal default original ratinterp predicates and six native real-primal controls. Source D/product/coefficient assembly matches native words; public pole gap improves to2.076e-11. [Evidence](fft602_rational_integration_cpu_20261017.json).
- [ ] Remaining rational SVD/pole gap, FFT wider plans/extreme values/dynamic native storage, full APIs and CPU performance. Local source kernels remain52.1x/32.7x slower than generic JAX at measured shapes; matched native performance unqualified.


- [x] Remove non-native bounded fractional-power norm preflight:4tiny-complex falsezero failures fixed,18controls pass with original33bound/warningcheck, and20CIprotocol checks retain1075anchors plus1new selector. [Evidence](power_zero_preflight_cpu_20261017.json). Standalone norm/extrema/nonfinite primitive/fullAPI remain open.

- [x] Real602forward diagnostic matches275424native nonredundant FFT words; sourceD302reconstruction matches all91204words. All690captured stage words and independent constants match. [Evidence](ratinterp_real602_cpu_20261017.json). Production integration/public poles/API remain open.

- [x] Complete source-derived JAX602inverse diagnostic matches367220native outputwords across positive/actual/independent matrices; outer2/7stages match204distinct words, independent constants and division normalization. [Evidence](ratinterp_full602_cpu_20261017.json). No production patch/public pole claim; real D route remains open.

- [x] Correct minimax original11test scalar extraction exposed by actualCI; original operand/degrees/scaling/bound unchanged, focused candidate passes on identical360production files. [Evidence](minimax_scalar_ci_cpu_20261017.json). LatestCI/fullAPI remain open.

- [x] Restore all41literal default original power predicates and verify27actual native draw events; final24rerun plus17unchanged-node source/AST bridge, with360production files identical to publication. [Evidence](power_original41_cpu_20261017.json). The historical predicate 33 extra warning is superseded for supported bounded fractional powers by [the 18-control preflight follow-up](power_zero_preflight_cpu_20261017.json); standalone norm/extrema/nonfinite primitives and full API/preferences remain open.

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
- [x] Authored fixed48 CI advection regression is qualified by the published FFT64 gate above; final FFT602 integration reruns that unchanged control. Broader PDE and performance gaps remain open.
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
