# MATLAB port mirror execution (CPU), 2026-10-10

Frozen source: `e6748499264764a798a2c95f1e1da8c79ee8be37` (tree `78e9fc6dd975bdf0a857b7f7272b9f4775c74dc9`), from the tracked snapshot archive SHA-256 `d8415384a246717532f60ad824c8c7e07234606f216c11902733a3d606615713`. This is isolated qualification evidence for 52 selected mirror wrappers. Their preflight inventory contained 57 nested plain test callables; that inventory is not an executed-callable count. Independent scope review found no overlap with the 5,575 ordinary or 99 supplemental direct-node ledgers; mirror integration coverage is counted separately.

## Results

- **49 mirror selectors passed.** Three lanes completed their exact 13-selector plans (handles 82635, 39387, 4925); lane 1 completed the ordered 12-selector prefix before fail-fast stopped it.
- **One selector failed:** `tests.test_matlab_port.misc.test_pde15s_matlab::TestMiscPde15s.test_all_matlab_assertions`, with `err=2.834910858659294e-06` against `tol=2.220446049250313e-11`.
- **One helper-only skip:** minimax reported no plain test callables. It receives no scientific test credit.
- **One selector remains unexecuted:** quantumstates. Its follow-up supervisor (handle 14026) rejected the initial input guard before creating an output directory or launching pytest; the recorded startup failure lists 25 changed OS-library hashes. This is zero test credit. Keep the old gate immutable and wait for a stable, freshly resolved OS dependency closure before a new isolated attempt.
- The separate `_cart` selector was preclassified as helper-only and excluded from the 52 selected mirror wrappers.

All four v4 handles reached terminal receipts with no surviving owned processes. Lane 1 has `manifest_matches=true`, clean runtime provenance, and no bad origins; its `selection_matches=false` records the expected fail-fast prefix after the pde15s assertion, not a provenance defect. Preserve the failure and partial passes; no automatic reruns are implied.

## Lane evidence

Shared evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005`.

The authoritative lane artifacts are under `port_mirror_qualification_e6748499_20261010/execution_v4/` in shared scratch. Hashes below are SHA-256 of the saved files.

| Lane / original handle | JUnit outcome | Wall time; sampled peak RSS | Audit | Terminal receipt | JUnit |
|---|---:|---:|---|---|---|
| `lane0_cpu0_3` / 82635 | 13 pass | 371.31 s; 5,477,076 KiB | `8d580b26288088649a4297c66e375eccf1f1b3f70a53974c8f4dcc84cbc53111` | `301ff3649856c55fe83ff9c210475acac5480728e858bdc0a2f5e81dc4b25473` | `ccee4670809d1dc626bcfc32419b9e81488dca0d2374304818dd097385db2b42` |
| `lane1_cpu4_7` / 48681 | 10 pass, 1 fail, 1 helper skip | 328.18 s; 4,791,160 KiB | `64f3ca8c3c4b688609b37a47e496e1f24dee01c11a70dfea9085f48358fcf0dd` | `a8f1e9b1483bdc7b469edc5fa92901abea2c82386c60814cd6aeb1c72b407b13` | `181de941d93ce44988db8890332a6aa571b3d659deb41bcca533bcd0236f25f9` |
| `lane2_cpu28_31` / 39387 | 13 pass | 591.79 s; 8,019,032 KiB | `c3af7789a841d0b26965a29fde11ec6ca815567921a3c9a2d94acde563be0f24` | `8fa200fffde1b65debfa21db4f7cc62b7177829cbf834822b3e57db2b27c7f0d` | `66a9fb874e5c4fe8959b8ad842793bba0715759927a8e90b4bdc570692807d33` |
| `lane3_cpu32_35` / 4925 | 13 pass | 232.51 s; 4,351,740 KiB | `eaea40ce96c42ca7e42b0f66c2869643a9586f3f349d02455bb9b5c1ba596812` | `313dcc6ba738d7a9f712f36eb207657f3ff9ce8919cd2277124eabe63bd96527` | `db11c945ac4babe25d92e93c729c17000859d47056475cd675e28f33f7fc6bf6` |

Per-lane timeout-policy hashes are recorded in `recovery_20261010/MIRROR_EXECUTION_ACCOUNTING_v1.json`. That report also binds the quantumstates startup failure; SHA-256 `6e18f0652e14d5279eb7d374902960aa9a8a79b13ac10d8712bbfd68df7c7acf`. The source-backed failure analysis is `recovery_20261010/PDE15S_GAP_REVIEW_v1.json`, SHA-256 `fcffe1d5ba49d80c5f73387865b1949793f7bb7f11b5d7c664acfe8fa3712b2e`.
Independent root review `recovery_20261010/ROOT_REVIEW_v1.json` confirms the 49/1/1/1 accounting, exact lane ordered prefixes, historical guard stability, and clean runtime-end records; it rehashed 5,675 unique frozen scratch inputs. Review SHA-256 `9b5391d92f1d2ec09cc7233c163e3f8ab846d5fb460fa0d1cc00c0e65845aa2d`. The review explicitly does not claim current OS library bytes equal the historical run bytes after the concurrent system upgrade.

## pde15s contract gap

Native `tests/misc/test_pde15s.m` compares the same PDE under `chebtech1` and `chebtech2`, with native default solver options, and checks the cross-run norm against `1e5*pref.chebfuneps`. Native `pdeSolve.m` sets PDE tolerance to `1e-6` by default, maps ODE tolerances from it, and adaptively refines the nonperiodic `chebtech2` spatial grid. The frozen Python port test instead supplies `rtol=1e-10`, `atol=1e-12`; the solver uses a fixed inferred resolution and has no spatial adaptation. Treat this as an open solver implementation/port-contract gap. Do not loosen the assertion or claim a tolerance-only repair. A bounded follow-up should reproduce native tolerance mapping and spatial convergence, then record refinement lengths and the final comparison. SciPy dependency debt remains separately identified; this evidence does not establish a JAX-only solver.
