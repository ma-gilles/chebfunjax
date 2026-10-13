# Final large-domain GL evolution: source and arithmetic isolation

## Decision

The final historical phase portrait is not a verified numerical oracle for the current float64 trajectory. Replaying the **exact native initial Fourier state, nonlinear state, and every ETDRK4 coefficient** with the source-shaped JAX step still produces order-one field differences at t=60. Thus neither the example's t=30 restart nor initial construction alone explains the discrepancy. No solver tolerance, grid size, time step, rank, formula or coefficient correction is justified by these captures.

A separate source-fidelity correction is justified: the example should eventually call the already-existing public multi-time spin2 API, which runs continuously and uses the native trig grid. Its current local wrapper uses a legacy grid and reconstructs/restarts at t=30. That change must not be advertised as reproducing the cached t=60 field. The frozen graphics package was not modified.

## Exact inputs and source audit

Private unmodified checkout `../gl_final_evolution_candidate_20261013`, base e73957d4626f47f0895ddd5d99f05bb4d4dc6da6. Native source is Chebfun commit7574c77680d7e82b79626300bf255498271a72df in `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref`; installed MATLAB R2024b. Public initial construction uses x/y Chebfun2s, the exact two shifted Gaussian spiral expression from the page, domain[-100,100]^2, N128,dt8/128=.0625, original tspan[0,30,60]. Preferences are ETDRK4,M32,dealias off,plot off. Native and Python source inputs, drivers and retained arrays are hashed in individual guards and `artifact_manifest.json`.

Native diagnostic tspan[0,dt,2dt,1,5,15,30,60] changes only requested snapshots. Inspection of solvepde.m establishes that with plot off, intermediate output only evaluates c2v(cNew), stores vOut/tOut and advances pos; it does not update cOld/NcOld, scheme coefficients or dt. Final tf remains60 and all960 steps are unchanged. Snapshot functions are constructed after the time loop. The native driver also saves raw c0/Nc0 and the first2 source oneStep results, avoiding output-constructor effects for the local-step comparison. No native rendering was performed.

Python solver isolation replays saved values from the original source-built initial function on the exact matching grid, rather than reconstructing that function in the same process as contour setup. This adaptation is recorded in drivers; it is not a literal public end-to-end timing result. The legacy arm separately reconstructs the actual initial function and uses its actual low-level solver grid.

## First divergence and controls

- Python/native initial function samples: max4.3613e-13,RMS2.1837e-14, field scale3.4622. Their respective errors versus the direct analytic formula are2.8677e-13 and3.3774e-13. Neither source-built object equals its analytic formula exactly.
- Public linear symbol L is bit-identical to native. Public exponential factors differ by at most2.2204e-16; A/B factors by at most4.5103e-17.
- Starting from identical native c0/Nc0 but Python-evaluated coefficients: first2 raw Fourier errors5.8593e-14/7.6528e-14 at coefficient scales219.398/199.137.
- Starting from identical native c0/Nc0 **and native coefficients**: first2 raw Fourier errors4.2633e-14/4.4939e-14 at the same scales (roughly2.3e-16 relative). This isolates the step arithmetic/FFT backend difference from initial construction and coefficient evaluation.
- Starting from identical native value array: initial JAX/native FFT max error7.3031e-14; initial nonlinear FFT error8.1981e-13.
- Legacy np.linspace grid differs from native trig grid by max1.4211e-14. Its t30 Chebfun2 reconstruction changes values by max1.5168e-13,RMS6.7109e-15.

## Trajectory differences versus native

Maximum absolute complex field error on the source128x128 grid:

| Time | Public, Python-built initial | Public, native initial values | Native initial Fourier states and factors replay |
|---|---:|---:|---:|
|1|4.2455e-13|3.5924e-14|3.5957e-14|
|5|8.5198e-12|7.8983e-14|8.4527e-14|
|15|7.7051e-8|1.5951e-10|9.8082e-11|
|30|.128876|.000277667|.000169821|
|60|1.993363|1.978332|1.950613|

At t60 the native field maximum is.999906; RMS errors for those columns are.995229,.808594,.579496. The actual legacy/restart path has max1.995758,RMS.940064 versus native; its difference from continuous public with the same Python-built initial object has max1.965077,RMS.704802.

These results demonstrate amplification of small arithmetic perturbations. The PDE linearization around the nearly zero far field includes +u, so amplification is mathematically expected; this observation does not prove that every discrepancy is harmless or establish any new scientific acceptance bound. No cached-pixel, final-field or tolerance fitting was used. Native output reconstruction causes approximately1e-12 differences even in first-step sampled function comparisons; the raw Fourier comparisons above are the appropriate early local control.

## Receipts and limitations

`receipts_summary.json` links all stages. All guards remained stable, and all owned process trees are terminal with no survivors:

- Native capture: exit0,93.96s,sampled1,636,148KiB under120s/2GiB.
- Initial Python combined attempt: RSS-capped exit137,28.91s,1,597,596KiB under1.5GiB. Initial arrays survive; no evolution success claimed.
- Python_v2: public trajectory and coefficient/first-step arms completed and persisted; then legacy input-grid assertion correctly stopped the combined driver (exit1,8.57s,753,252KiB). Its retained public trajectory column is a completed diagnostic arm, not an all-driver passing gate.
- Legacy_v3: exit0,34.61s,1,822,028KiB under90s/3GiB; actual original construction/grid/restart path captured.
- Identical_factors: exit0,10.68s,651,628KiB under90s/3GiB; both complete native-value public replay and exact native-Fourier/coefficient replay captured with partial checkpoints.

CPU16–19, CUDA visibility empty, JAX/jaxlib0.11.0 x64. Native was headless with software OpenGL launch flags; no graphics calls. Times are observational and not paired performance measurements. No new full-page run, image generation or library edits occurred. The historical cached field remains unqualified independently of native7574 source arithmetic parity. All native and Python states needed for further focused analysis are retained; no further native run is warranted by the current evidence alone.

## Proposed public-wrapper migration (not implemented)

Import `Spinop2, spin2` from `chebfunjax.operators.spinop2` (these names are not currently exported by top-level chebfunjax). The existing page operator-setup helper can create `S = Spinop2(dom, tspan)`, assign `S.lin = '@(u)lap(u)'` and `S.nonlin = '@(u)u-(1+1.5i)*u.*(abs(u).^2)'`, then return S. Replace every `S.u0 = ...` with `S.init = ...`. Call `spin2(S, npts, dt, 'plot', 'off')` directly. Remove the example-local time-loop wrapper, `_values2chebfun2`, and low-level solver/SpinOp2 imports.

For scalar two-endpoint solves, public spin2 returns the final Chebfun2 directly, matching current plot calls. For `[0,30,60]`, it returns Spin2SolutionMatrix of shape(1,3); use `u[0,1]` for t30 and `u[0,2]` for t60 (zero-based). Existing `u[1]`/`u[2]` also select those columns via column-major indexing, but explicit pairs clarify the native chebmatrix relationship. Public output construction calls Chebfun2.from_values with native row-y/column-x arrays and trig=True. `return_times=True` optionally returns actual source collection times; no transpose or local reconstruction is required.

Qualify this later as source API/operation-order fidelity with existing public spin2 controls and a bounded example final-clause check. Historical final-field/pixel parity remains unresolved; do not use its cached extrema or colors to choose numerical parameters. No migration changes or additional runs were made in this diagnosis package.
