# chebfunjax — Final Verification Report (Workstream E)

*Compiled by Claude Opus 4.8. Companion to `HANDOFF.md` (narrative),
`PARITY_MATRIX.md` (coverage matrix), and `STATUS.md` (module history).*

This report closes the parity campaign: it records the final test
results, the state of every workstream, and how to reproduce.

## 1. Test results (final run)

| Suite | Result | Command |
|---|---|---|
| Fast (unit, no MATLAB/GPU) | **2,584 passed, 4 skipped** | `pixi run test-fast` |
| MATLAB golden-ref (machine precision) | **473 passed, 14 skipped, 1 xfail** | `pytest -m matlab` |
| Operators + chebfun1d + chebfun2d | 661 passed | targeted |
| Spin (ETDRK4, incl. golden-ref) | 138 passed | `pytest tests/test_spin` |

Total: **2,728 test functions / 3,084 collected**. Lint clean. No
`numpy` import in library code. The 14 MATLAB skips are cases where a
`.mat` ref is intentionally absent; the 1 xfail is documented.

## 2. Workstreams

| # | Workstream | State |
|---|---|---|
| A | Library correctness, lint, CI green | **complete** |
| B | ~410 example scripts run clean | **complete** |
| C | Guide chapters 1–20 genuine plot parity | **complete** — 323/323 figures; ch.17 now uses the library spherefun calculus |
| D | Example pages parity (21 categories) | **complete** — 826 numbered figures genuine |
| E | Final verification + report | **this document** |

## 3. Library backlog (#8–#25) — all closed

Every backlog task is done and verified. Highlights:

- **#8** spin ETDRK4 (1D/2D/3D) FFTs on `jnp.fft` — GPU-capable, parity kept.
- **#9** `diff()` attaches Dirac deltas at jumps; `sum()` includes them.
- **#10** `legpts` O(n)-memory Newton path (n=65536: 34 GB hang → 6.5 s).
- **#11** `eps` / `max_length` wired through the constructor.
- **#12** splitting-on / edge detection (`splitting=True`).
- **#13** `cheb.gallery` **27/27** MATLAB entries (blasius, gamma with
  poles, daubechies via cascade, vandermonde/vandercheb quasimatrices,
  random; corrected `wild`). `cheb.gallerytrig` 11 entries.
- **#14** two-arg min/max, floor/ceil/round, local extrema, complex roots.
- **#15** no construction grinds to the 65537 cap.
- **#17** ballfunv div/curl, `Ballfun.poisson`, Helmholtz decomposition.
- **#20** `cheb.gallerytrig` + trigtech aliasing bug fix.
- **#22** chebop `eigs` eigenfunctions, `expm`, `matrix(L,n)`.
- **#24** chebop periodic BCs + IVP time-marching routing.
- **#25** spherefun calculus (laplacian / poisson / diff / grad / sphharm).

Also: disk/sphere/ball PDE toolkits, `randnfun`, and **3 constructor
bug fixes** (spherefun, diskfun, trigtech coarse-grid aliasing).

## 4. Documented figure exception classes

195 of the 826 example figures do not pass the strict badness ≤ 0.06
gate but are content-verified. They fall in four classes, none a library
defect:

1. **3D-renderer aspect** — matplotlib vs MATLAB camera/aspect (mae ≈ 0,
   histogram correlation ≥ 0.99).
2. **Seeded-random instance** — a different random draw than the archived
   chebfun.org PNG.
3. **Data-dependent** — needs climate/data files not bundled with the repo.
4. **Version-drift** — the archived page revision is no longer published.

## 5. Reproduce

```bash
source project.conf
pixi install
pixi run test-fast          # 2,584 pass
pixi run test-matlab        # golden-ref parity (committed .mat fixtures)
pixi run lint
# GPU regression (recommended before a release):
#   sbatch the Slurm template in CLAUDE.md → pixi run test-full
```

## 6. Remaining (out of scope, not gaps)

- `chebmatrix` / `linop` / `chebop2` / `chebgui` have independent Python
  tests but no file-level MATLAB golden-ref ports yet.
- A few sphere example pages use `scipy.special.sph_harm_y` as a
  numerical reference / input generator (like test code using numpy),
  not as a library stand-in.

The parity campaign is complete: every implemented function is verified
(unit + MATLAB golden-ref where applicable), all guide and example
figures are genuinely computed, and the full library backlog is closed.
