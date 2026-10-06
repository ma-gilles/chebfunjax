# chebfunjax: handoff to a Codex agent (2026-10-02)

## Latest shared checkout status (2026-10-06; takes precedence below)

Full MATLAB7574c77 and322-page parity remains active and incomplete. Work in
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/publication_checkout`.
Read `CURRENT_GOAL_STATUS.md` in its parent for current evidence and next gaps.
The original home checkout and twelve interrupted scripts remain protected;
write only under shared scratch. Normal SSH push and licensed CPU MATLAB work.
Use gh-cli, explicit staging, GPT-6 trailer and full Ruff gates. CPU only;
heavy examples/tests serial; at most two helpers, Astra medium for numerical
review as requested and cheap routine execution. Freeze qualification inputs.

P105 535fef0e89a44c7bcdbee8f94d15e2bc354e846f is pushed. Combined48
coast/source-grid colorbar cases pass37.971s; archive verified. New API retains
corrected C and a shared norm snapshot. V1 new exact-edge assertion failed
because Matplotlib inverse normalization differs oneULP from fresh MATLAB
CLim; gamma2 operation bound documented, production unchanged.

Exact P105 CPU baseline reproduced all4historicalCI failures and3independent
metadata failures. New scoped corrections preserve existing pointValues through
simplify/common coefficient padding, adapt accepted typed0Dinteger scalars at
the public Chebfun boundary (lower-tech dtype diagnostics unchanged), and
observe actual sampleTest calls separately from detector bracket endpoints.
Original numerical bounds remain unchanged. Oscillatory singular cube remains
a genuine strict accuracy failure; source-RNG/evaluator decomposition queued.
No full latest-main green CI or atmospheric/pixel parity claim.

P104 8e0155ec666a4aa05db0196b0076664ff7d3c9c1 is pushed with the62-case
short-sum/channel and camera qualification below. New CPU exp diagnostic13
selected inputs/12paths/threeflagsettings found identical output bits; HIGHEST
requested in HLO did not repair one-ulp errors. C1 remains open.

A coastline renderer correction passed41focused CPU cases in30.062s;
new general colorbar adapter is under combined frozen qualification: actual registered
opaque sphere artists govern masking; hidden/removed/transparent artists and
user zorder are respected. Conservative JAX ray arithmetic retains ambiguous
front boundary points without moving CoastData. P104 actual-artist baseline
two true rendering failures and one harness import error in first gate;
corrected-import isolated baseline and complete candidate gates are recorded. This is analytic sphere
visibility, not exact faceted-mesh depth; near-limb segments, source framing,
full pixel parity remain open. The opt-in scalar mappable uses the exact
corrected plotted tensor C and same normalization; it is documented as a
snapshot, preserving explicit/bumpy surface colors. Baseline7API controls fail
before this adapter. Constant CLim/face interpolation/custom norms remain open.

P103 ab9b4feaf51c1db30f1128c035ab7ec9d17ad541 is pushed. It replaces dense
adaptive sphere output with literal northern-grid construction using JAX FFTs
and preserves tensor plotting arguments. Frozen29small plus11solver cases pass,
including fresh5MATLAB API outputs, Poisson, four Helmholtz grids and source
Spinsphere trajectory. The existing host from_values constructor remains open.
Bounded warm constructors improved36.55/38.94ms to31.80/33.90ms, while cold
costs increased3.64/1.24s to4.91/5.53s; no full Atmospheric timing claim.

The next source short-array sum/channel correction passed62frozen CPU cases,
including7independent short-real controls,14camera controls and retainedP100
source/regression cases. Fresh MATLAB5public CDR fixtures pass68.909s. Source
SUM without a dimension can reduce across factors; scalar-times-D then retains
multiple channels. An earlier independent zero-pivot expected2pi was wrong:
MATLAB gives4pi, now corrected with the original bound. Source quirks retained.
Camera helper converts source azimuth50 to Matplotlib-40 and preserves projection
and layout. Coast clipping/framing/full pixel parity remain unqualified.
Generic complex addition, MATLAB query reshaping and full per-channel policy
remain open. Source-body/import-set proofs cover formatting-only post-gate edits.

AtmosphericTemperature V11 failed signal9 after1093s/four of ten figures at
407GiB peakRSS; immutable failure archive is preserved. Rank185 and three
scalars computed, with final digit differences. Next rerun needs verified
rendering fixes and a fresh frozen full execution, page/stdout/figure audit.
MATLAB graphics metadata captured in96.799s, actual source/properties stable;
its software WebGL PNGs are nearly black and unusable as visual references.
Official cached1305figure slots remain available. No arbitrary95%viewport
fitting or source asset substitution is accepted.

P97 stable realatanh9passes, P98 compensated extrapolation72passes, P94Fejer
original18bounds pass; Fejer warm n100000 cost4.781x remains recorded. Focused
passes overlap; no full-suite or universal MATLAB rounding identity claim.
CI P98–P103 snapshots remain in progress, completed checks have no failures,
docs allsuccess. Latest fullgreen is not verified. C1 detector remains open:
a JAX exp one-ulp bump triggers the literal source zero-slope rejection; no
cutoff/tolerance/xfail change justified. All322orderedprose/2403inputs match,
11output-count gaps and154figure-size mismatches remain; computations/pixels/RNG
and exact-byte disposition of the12protected originals are unfinished.

Next: full Atmospheric run/figures, detector backend accuracy, source merge12/13,
repeated-residue correctness, full functions/tests/JAX/RNG/performance and CI.
Drafts and failed/unrun revisions are preserved; no external blocker.

This document takes over from the Claude session that ran the parity
campaign from 2026-07 to 2026-09. `HANDOFF.md` (2026-07) is older and partly
stale: where they disagree, this file wins.

## 1. Mission and standing rules

The goal is full parity with MATLAB Chebfun (commit `7574c77`, source at
`/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref`). That covers every library
function and every MATLAB test, and every chebfun.org example page must be
replicated exactly: same prose, same computations, the same printed outputs
and the same figures at the same sizes.

Rules the user has set (all still in force):

- **Push directly to `main`; CI is the only reviewer.** Never force-push. Use
  `gh-cli` (not `gh`) for GitHub.
- **Stage with explicit pathspecs.** Never `git add -A` on the whole tree,
  since other agents may share it.
- **Library code is JAX-only.** Any file that imports numpy needs a
  `# uses-numpy: <reason>` comment, which CI checks. Every public function
  needs a `Provenance` docstring section naming the MATLAB source and the
  `Chebfun commit: 7574c77` line.
- **Tolerances.** Never widen a test tolerance without a comment explaining
  why, with the MATLAB value and ours.
- **Lint.** `ruff check src tests scripts examples` must pass before every
  push; CI lints `examples/` too.
- **Commit trailer.** End every commit message with
  `Co-Authored-By: <your model> <noreply@...>`.
- **Storage.** Never write data under `/home/mg6942`. Put scratch in
  `~/myscratch/tmp/<dated-name>/`, or `/scratch/gpfs/GILLES/mg6942/_agent_scratch/`
  for work that spans sessions.
- **Token budget.** Be economical: at most 2 parallel sub-agents, on a cheap
  model or low reasoning effort for routine script fixes.
- **Examples use the library.** They are translations: never re-implement a
  library feature in numpy inside an example. When the library is wrong or
  missing something, fix the library (with tests) or log it in section 6.

## 2. Environment: fix this first

**The pixi environment is gone.** `.pixi` is a symlink to
`/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_pixi_env_20260702/.pixi`, which
has been deleted. Recreate it:

```bash
cd ~/chebfunjax
rm .pixi                                   # dangling symlink
mkdir -p /scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_pixi_20261002
ln -s /scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_pixi_20261002 .pixi
pixi install                               # ~/.pixi/bin/pixi
pixi run smoke
```

Always run Python as
`JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .pixi/envs/default/bin/python ...`.
Without `OMP_NUM_THREADS=1`, linear algebra on the login node is about 70x
slower.

MATLAB R2025b is available for reference runs:
`module load matlab/R2025b; matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('x.m')"`
(about 1 minute of startup).

The MATLAB example sources are at `/scratch/gpfs/GILLES/mg6942/chebfun_examples/<cat>/<Stem>.m`.

## 3. State at handoff

- `main` is at `ecfda8fc`, and CI passed on the last three pushes.
- 322 example pages (`docs/examples/<cat>/<Stem>.md`) are generated from the
  chebfun.org HTML plus our script output.
- A structural audit finds 295 of 322 pages whose printed output blocks line
  up with the original. The 27 that don't are listed in section 5.
- Example figures are saved with
  `chebfunjax.plotting.save_chebfun_figure(fig, path, size=(w, h))`. The
  default is 600x270; other pages use the size of the reference PNG in
  `/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images/<cat>/`.
- `TEST_CHECKLIST.md` (generated by `scripts/gen_test_checklist.py`) records
  MATLAB test-suite coverage: 1069 ported, 26 skipped by policy (chebgui,
  adchebfun), 7 missing (chebgui).

### Uncommitted work in the tree: verify or revert

Interrupted agents left edits that have not been verified, in 12 example
scripts and their images:

```
examples/approx/best_approx.py              examples/pde/trapezoideigs.py
examples/approx/fermi_dirac.py              examples/roots/marching_squares.py
examples/approx2/localization.py            examples/sphere/atmospherictemperature.py
examples/fourier/fourier_based_chebfuns.py  examples/sphere/laplaceball.py
examples/linalg/vandermonde_arnoldi.py      examples/stats/resampling_random_variables.py
examples/ode-nonlin/delay_differential_equations.py
examples/ode-random/randomswitching.py
```

Modified images also appear under `docs/images/` for GlobalMinimum and
WhiteCurves, whose scripts are unchanged; those were overwritten by a test
run, so revert them with `git checkout`. For each script, run it alone, run
`unmatched.py` on its page (section 4), then commit or
`git checkout -- <file>`.

Four scripts are slow or failed when run in parallel: VandermondeArnoldi
(over 30 min), DelayDifferentialEquations (over 1 h, because piecewise delay
chebop solves are slow), RandomSwitching (crashed at about 25 min) and
AtmosphericTemperature (killed, probably out of memory, while other jobs
ran). Run them one at a time, or as a CPU Slurm job on a general partition
with about 32G of memory.

## 4. Tooling

The durable copies are in `/scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_handoff_20261002/`
(called `$D` below):

| Item | What it does |
|---|---|
| `scripts/gen_example_pages.py` (in the repo) | Builds `docs/examples/<cat>/<Stem>.md` from the chebfun.org HTML (prose, headings, MATLAB cells, figure slots), our stdout and our images. `--stdout-dir $D/stdout --html-cache $D/pages_html [cat/Stem ...]`; with no pages it does all of them. It reports missing images and output blocks it couldn't match. |
| `$D/pages_html/<cat>_<Stem>.html` | Cached chebfun.org pages. This is the target for the print format; the pages were published with `format long`. |
| `$D/stdout/<cat>_<script>.txt` | Our latest stdout per script, which feeds the page generator. Regenerate it by running the script with stdout redirected there. |
| `$D/regen_all.py [substr ...]` | Runs example scripts 6 at a time with a 30-min cap, writing stdout and stderr to `$D/stdout/` and rc lines to the log (`REGEN_LOG` env). |
| `$D/run_slow.sh <script>` | Runs one script with a 4-h cap and appends to `slow_runs.log`. |
| `$D/unmatched.py <cat>/<Stem>` | Lists the page's output blocks that our stdout lacks, with the MATLAB input cell that produced each. |
| `$D/structure_audit.py` | Audits every page: blocks whose line counts differ, and extra lines we print. Writes `structure_audit.csv`. |
| `$D/refs_matlab/`, `$D/compare_vs_matlab.py` | MATLAB R2025b reruns of the examples (diary output). Use them only to compare numbers; a fresh rerun defaults to `format short`. |
| `$D/batch_prompt.md` | The prompt template used for the page-fidelity sub-agents. |

The workflow for one page:

1. Read `$D/pages_html/<cat>_<Stem>.html` and the `.m` source.
2. Edit the script so every output cell prints the same labels and layout and
   every `img/<Stem>_NN.png` is saved in order. Note that a MATLAB `hold on`
   cell produces separate before and after images.
3. Run the script, writing stdout to `$D/stdout/`.
4. Run `unmatched.py`, then `gen_example_pages.py <cat>/<Stem>`.
5. Run ruff, then commit the script, its images and its page.

Library display helpers that already match MATLAB: `repr(chebfun)`,
`Chebfun2.disp()`, `Diskfun.disp()`.

## 5. Remaining example work

Pages the audit marks as not matching (as of 2026-09-24). Many are blocked
only by random draws, which MATLAB's `randn` makes impossible to reproduce,
or by the library issues in section 6. Fix what is fixable in the script and
note the rest.

```
approx/EdgeDetection approx/NoisyNonsmooth approx3/Wagon cheb/ChebExplain
complex/ConformalL fourier/FourierBasedChebfuns geom/Ellipses
linalg/ConstrainedLeastSquares linalg/EigsViaDet linalg/FieldOfValues linalg/SOR
linalg/VandermondeArnoldi ode-eig/Drum ode-eig/Eigenstates ode-linear/DawsonIntegral
ode-linear/Krylov ode-nonlin/Blasius ode-nonlin/Carrier
ode-nonlin/DelayDifferentialEquations ode-nonlin/GulfStream ode-nonlin/IVPCapabilities
ode-random/RandomSwitching opt/Catenary quad/HermiteQuad roots/RandomPolys
sphere/AtmosphericTemperature stats/ResamplingRandomVariables
```

Some audit hits are false positives where the output is already correct
(Blasius, Catenary, Carrier apart from its iteration table).

Other known example gaps:

- **Checkmark** misses the local minimum at alpha = 0. Our E_3 piece on
  [0, 0.48] has 5 coefficients where MATLAB's has 12. Using
  `norm(p - f, inf)` instead of `r.err` would probably fix it, at about twice
  the runtime.
- **LaneEmden** now matches MATLAB (radius 3.653753736219 against
  3.653753736220) but takes about 28 minutes, because the nonlinear system
  Newton step uses a finite-difference Jacobian.

## 6. Library issues found by the examples, not yet fixed

These are ordered by value. Each item names the example that exposes it, and
should be fixed in `src/` with a MATLAB-port test.

1. **`Chebfun.inv` is about 800x slower than MATLAB.** It bisects pointwise:
   ComplexArcLength took 2018 s against MATLAB's 2.5 s. Port `@chebfun/inv.m`,
   which uses a roots-based inverse.
2. **`hermpts` uses only Golub-Welsch.** n = 10^4 takes 136 s, and 10^5 would
   need an 80 GB matrix. Port the GLR and asymptotic methods from `hermpts.m`
   (HermiteQuad).
3. **`ratinterp` returns wrong poles for complex f.** `utils/ratapprox.py`
   (`_chebyshev_roots`, about line 957) takes the real part of the
   coefficients. Its `r_fn` also casts x to float (ThreeBodyProblem).
4. **Building a periodic chebfun2 from complex values is wrong:**
   `chebfun2(complex matrix, trig=True)` gives O(1) errors (GinzburgLandau).
5. **MINRES stalls on indefinite operators.** On `-u'' - 100u` it reaches a
   relative residual of 5.8e-4, where MATLAB reaches 1.85e-15, and it
   diverges on the piecewise case (ode-linear/Krylov).
6. **Nonlinear `Chebop` solves.**
   - Newton starts at 16 points and doubles, restarting each time, while
     MATLAB starts at 64 and refines once to 128. As a result, Carrier's
     starting guesses 2 and 3 converge to different solutions.
   - There is no `'iter'` display, so Carrier's iteration table can't be
     printed.
   - `info["normDelta"]` concatenates the Newton histories of every
     refinement level.
7. **Splitting creates spurious breakpoints.** `_split_edge_fd` zooms to about
   1e-15 regardless of `eps`, which leaves clusters of breakpoints 1e-15 apart
   at eigenvalue-max kinks (EdgeDetection, NoisyNonsmooth).
8. **Backslash on an ill-conditioned basis is far less accurate than MATLAB.**
   The quasimatrix `\` gives a vertical scale of 7.7e7 where MATLAB gets 1.5
   (VandermondeArnoldi).
9. **The IVP marcher returns 16 pieces** where MATLAB returns one smooth piece
   (IVPCapabilities).
10. **The AAA conformal map keeps about twice MATLAB's spurious poles**
    (ConformalL).
11. **There are three `randnfun` implementations.** Only
    `chebfun1d/randfuns.py` follows `randnfun.m`. `cj.randnfun`
    (`utils/random.py`) returns a numpy callable and scales `'big'` wrongly.
    Unify them on the faithful one.
12. **`spin2` only returns arrays.** It doesn't accept a multi-time `tspan` or
    return a chebfun2, and `SpinOp2('gl')` uses a deterministic initial
    condition.
13. **Smaller issues:**
    - `Spherefun.max2` returns the antipode for even functions.
    - `plot_ball_slices` shading and depth order are poor.
    - `plotregion` estimates rho differently from `plotregionData`.
    - Chebfun has no `__rpow__` (`complex ** chebfun` fails).
    - `matlab_plot` gives each piece of a complex piecewise chebfun its own
      colour.
    - `roots(..., 'ms')` finds points MATLAB misses (ResultantMethod).
    - Disk Helmholtz at k = 7 has an error of 3.8e-8 from laplacian noise; the
      test is xfail.
    - There is no phase-portrait plot for complex chebfun2.

## 7. Fixed in September 2026 (for context; don't redo)

- **chebfun2 constructor.** It now uses MATLAB's sampleTest (Halton points),
  standardCheck happiness and global simplify. The Greeks vega matches MATLAB
  to 5e-12.
- **Splitting.** The length cap is now 129 points, the global hscale and
  vscale are threaded through, and detectEdge snaps to endpoints.
- **Small 1-D methods.**
  - `angle` uses atan2 with branch-cut breakpoints.
  - New `chebfun1d.fov`.
  - `restrict` snaps to piece endpoints.
  - `flipud` is exact.
  - `min(f, 'local')` finds kinks.
  - Fractional powers use the principal branch.
- **Chebop.**
  - The nonlinear Newton step uses rectangular projection with cached
    boundary rows, which fixes LaneEmden for n >= 2.
  - Systems accept scalar unknowns.
- **Disk and sphere solvers.**
  - Diskfun Helmholtz and Poisson are now MATLAB's ultraspherical port, about
    90x faster.
  - `Spherefun.rotate` uses MATLAB's ZXZ convention.
  - `randnfunsphere` is stable at high degree.
  - Spherefun `coeffs2vals` is vectorized.
- **Plotting.** New `plot_earth`, and `save_chebfun_figure` produces exact
  pixel sizes.
- **Examples.** The `_replica` suffix is gone, about 150 scripts were
  rewritten cell by cell, and every page figure is present.
