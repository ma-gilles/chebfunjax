# MATLAB port mirror execution (CPU), 2026-10-10

Frozen source: `e6748499264764a798a2c95f1e1da8c79ee8be37` (tree `78e9fc6dd975bdf0a857b7f7272b9f4775c74dc9`), from tracked snapshot archive SHA-256 `d8415384a246717532f60ad824c8c7e07234606f216c11902733a3d606615713`. The selected scope contains 52 wrapper selectors. The preflight inventory had 57 nested plain test callables; that inventory is not an executed-callable count. Independent scope review found no overlap with 5,575 ordinary or 99 supplemental direct-node ledgers. Wrapper integration coverage is reported separately.

## Results

- **50 selected wrappers passed**, including the previously unexecuted quantumstates selector on fresh OS closure bindings.
- **One selector failed:** `tests.test_matlab_port.misc.test_pde15s_matlab::TestMiscPde15s.test_all_matlab_assertions`, with `err=2.834910858659294e-06` against `tol=2.220446049250313e-11`.
- **One helper-only wrapper was skipped:** minimax contained no plain test callables and receives no scientific credit.
- **No selected wrapper remains unexecuted.** `_cart` is a separate helper-only selector, preclassified and excluded from the 52.

The original four v4 lane handles and quantumstates handle 36052 are terminal with no surviving owned processes. The first quantumstates attempt (v5, handle 14026) stopped at its initial guard before child/pytest launch; preserve it as a startup failure with zero scientific credit. A v7 preparation verifier exposed a missing gate-local collection probe before launch; the reconstructed prelaunch failure record is `execution_v7/PRELAUNCH_FAILURE_v1.json` (SHA-256 `a752802c1fd94df809fe97ddb2aebea73fd3a9547fd74a5843193b1c928d0a50`). V8 corrected that in a fresh gate. The v8 exact quantumstates node passed its version and collection preflights, then passed the scientific test and runtime audit.

## Quantumstates v8 evidence

Gate: `port_mirror_qualification_e6748499_20261010/execution_v8/lane4_quantumstates_cpu4_7/`, CPU 4–7, cap 2400 s / 12 GiB, selector timeout 600 s. Exact collected and passed node: `tests/test_coverage/test_port_mirrors.py::test_port_mirror[tests.test_matlab_port.misc.test_quantumstates_matlab]`.

Original handle 36052 finished in 119.739 s, exit 0, no censor/cap, stable inputs, no surviving child. Sampled peak summed RSS was 3,106,636 KiB. Runtime audit saw 1,860 observed files, no failures, matching manifest, no bad origins, exact collection/JUnit identity, and one pass. Audit SHA-256 `45c89f06b57a3af84cbcabd2799b2cc5b88b05e5122488cafbe7b374d61e1ac4`; terminal receipt `fd38cf8823fd51533221f0d28f8827be49649729c603863350a61418701eb07c`; JUnit `bbe44e67aef6cca45e08730eca895669cd4dab6cbc4bec2c1c384feda2d9a58e`; runtime-end `723845e8469537b38d3d9bfc912e10c3a46e454e81ea87dbe0f527d696965771`; collected IDs `0883a7069088d5de7be50a548f02fe422e8858fe4746304a760407bc9a65c5db`. Consolidated terminal report: `execution_v8/TERMINAL_REPORT_v1.json`, SHA-256 `576ea48ff3778c03b54774b6980a238566f621425ddd42ec1895fee37f7bfec1`.

The fresh gate retained all frozen source, Python and environment bindings and refreshed only OS bindings/aliases from the released check3 closure. The ledger records 294 closure paths, 42 system-root files, and 20 unbound CUDA provider alternatives absent from both exact prior CPU preflight maps; the runtime observer remains fail-closed if an alternative is mapped. Check3 stability summary: `os_closure_stability_20261010/check3/SUMMARY.json`; pass SHA-256 `04e5ea33a56083d99ac5bc378e0dced88d21c5acece68486de3e04cd53afa477`. The pinned CPU run emitted a warning that installed `jax_cuda12_plugin` 0.11.2 is incompatible with pinned `jaxlib` 0.11.0 and would not be used; `JAX_PLATFORMS=cpu`, empty `CUDA_VISIBLE_DEVICES`, and runtime provenance checks passed.

## Earlier lane evidence

| Lane / handle | JUnit outcome | Wall; peak sampled RSS | Audit | Receipt | JUnit |
|---|---:|---:|---|---|---|
| `lane0_cpu0_3` / 82635 | 13 pass | 371.31 s; 5,477,076 KiB | `8d580b26288088649a4297c66e375eccf1f1b3f70a53974c8f4dcc84cbc53111` | `301ff3649856c55fe83ff9c210475acac5480728e858bdc0a2f5e81dc4b25473` | `ccee4670809d1dc626bcfc32419b9e81488dca0d2374304818dd097385db2b42` |
| `lane1_cpu4_7` / 48681 | 10 pass, 1 fail, 1 helper skip | 328.18 s; 4,791,160 KiB | `64f3ca8c3c4b688609b37a47e496e1f24dee01c11a70dfea9085f48358fcf0dd` | `a8f1e9b1483bdc7b469edc5fa92901abea2c82386c60814cd6aeb1c72b407b13` | `181de941d93ce44988db8890332a6aa571b3d659deb41bcca533bcd0236f25f9` |
| `lane2_cpu28_31` / 39387 | 13 pass | 591.79 s; 8,019,032 KiB | `c3af7789a841d0b26965a29fde11ec6ca815567921a3c9a2d94acde563be0f24` | `8fa200fffde1b65debfa21db4f7cc62b7177829cbf834822b3e57db2b27c7f0d` | `66a9fb874e5c4fe8959b8ad842793bba0715759927a8e90b4bdc570692807d33` |
| `lane3_cpu32_35` / 4925 | 13 pass | 232.51 s; 4,351,740 KiB | `eaea40ce96c42ca7e42b0f66c2869643a9586f3f349d02455bb9b5c1ba596812` | `313dcc6ba738d7a9f712f36eb207657f3ff9ce8919cd2277124eabe63bd96527` | `db11c945ac4babe25d92e93c729c17000859d47056475cd675e28f33f7fc6bf6` |

The v4 lane audit details and timeout-policy hashes remain in `recovery_20261010/MIRROR_EXECUTION_ACCOUNTING_v1.json`; v2 consolidated accounting is `recovery_20261010/MIRROR_EXECUTION_ACCOUNTING_v2.json`. Root independently reviewed the complete v2 results and quantumstates v8 bindings: `execution_v8/ROOT_REVIEW.json`, SHA-256 `26bcad7e2ea3e48d498eb0162929e405b1a1231d72c7b47da3ba3ecb462e2973` (42,896 guard bindings and 1,860 runtime files rehashed; no surviving processes). The source-backed PDE contract gap remains in `PDE15S_GAP_REVIEW_v1.json`.

## pde15s contract gap

Native `tests/misc/test_pde15s.m` compares the same PDE under `chebtech1` and `chebtech2`, with native default solver options, and checks cross-run norm against `1e5*pref.chebfuneps`. Native `pdeSolve.m` sets PDE tolerance to `1e-6` by default, maps ODE tolerances from it, and adaptively refines nonperiodic `chebtech2` spatial grids. The frozen Python test instead supplies `rtol=1e-10`, `atol=1e-12`; its solver uses fixed inferred resolution with no spatial adaptation. Treat this as an open solver implementation/port-contract gap. Do not loosen the assertion or claim a tolerance-only repair. SciPy dependency debt remains separate; this evidence does not establish a JAX-only solver.

## Direct Chebfun3 battery: 72 cases qualified

Seventy-two of the 80 direct cases now pass on the frozen `e6748499` source, separately
from the wrapper results above. The plain battery indices 0–39 and chebfun3f indices 0–15 are qualified;
8 direct cases remain unexecuted. These results do not establish current-main suite success.

| Gate | Original handle | Cases | Wall time | Peak summed RSS |
|---|---:|---:|---:|---:|
| `execution_v11` | 85989 | index 0 | 141.19 s | 1,898,784 KiB |
| `execution_v12` | 9903 | indices 1–3 | 231.03 s | 2,325,424 KiB |
| `execution_v13` | 84327 | indices 4–7 | 319.04 s | 2,326,928 KiB |
| `execution_v14` | 15703 | indices 8–15 | 587.71 s | 3,350,932 KiB |
| `execution_v15` | 57135 | indices 16–23 | 706.68 s | 3,277,836 KiB |
| `execution_v16` | 79002 | indices 24–31 | 420.81 s | 2,183,724 KiB |
| `execution_v17` | 45680 | indices 32–39 | 370.55 s | 1,914,480 KiB |

Each case ran serially in a fresh spawn worker, with its original 600-second
per-case timeout and a 4 GiB RSS cap. All seven runs exited normally, with stable
inputs, no resource cap, and no surviving owned process. The observer records
worker completion before the pool terminates it; parent and worker module/native
snapshots all pass. Snapshots do not prove coverage of transient dynamic loads.
Earlier v9 provenance failure and v10 collection failure receive no test credit.

Root independently rehashed 42,899 and 42,900 guard bindings respectively,
plus 1,859 observed runtime files per run. Review:
`port_mirror_qualification_e6748499_20261010/ROOT_FIRST4_REVIEW.json`,
SHA-256 `a1cb29ea0eb92205a376bcee56b7692d4e17c6c2df43ccb8202c64ae89859a24`.
The corresponding audit hashes are
`c1fef5fcb27a739b592fe3418e08e52a04e85020cb8aec2abb60474bbcbff8dd`
and `eadd54a94d69b9658c32e65fd845e42667dcedec7ce936a492e2b40ffa5af0b5`.

Root also rehashed all 42,901 v13 guard bindings and 1,859 observed runtime
files. Its review is `port_mirror_qualification_e6748499_20261010/ROOT_NEXT4_REVIEW.json`,
SHA-256 `a8d3638ca1d758782fa366de8f6db92b54a16a23c7eea8757377ebb3f5455961`. The v13 audit SHA-256 is
`e9f74d43d914687559e280d20411ee2213fd5cee911b7121f4cbc3f512280731`.

Static triage of native warning output gaps remains in
`recovery_20261010/NATIVE_OUTPUT_GAP_TRIAGE_v1.json`.

The v14 independent audit rehashed 42,902 guarded inputs and 1,859 observed
runtime files. Root verified its 24 result/report bindings, exact JUnit scope,
clean terminal receipt and eight worker completion records. Pytest time was
534.68 s; the table reports supervision time. Evidence:
`port_mirror_qualification_e6748499_20261010/ROOT_NEXT8_REVIEW.json`;
`execution_v14/AUDIT_v1.json` SHA-256
`beb2c7d8afa210f01b68be9fe58f75b21faf92165fe2343c17bbf63dbc72473c`.

The v15 audit verifies all eight cases and worker task-end records, exact
collection, stable inputs and no resource cap or survivors. Root independently
verified all 30 result bindings and the eight-case JUnit result. Audit:
`execution_v15/AUDIT_v15.json`, SHA-256
`1242b144cc580530e60ddaae6d37c34d04f3fbc34305f13886b9039ef813e512`.
Root review: `ROOT_INDICES16_23_REVIEW.json`. These remain historical results.

The v16 audit verifies all eight exact nodes and worker completion records,
stable inputs and no resource cap or survivors. Root verified all 13 result
bindings and ordered eight-case JUnit scope. Audit:
`execution_v16/AUDIT_v16.json`, SHA-256
`1de5550b6c80d59156620bb2031a44ba3f119fd62850490a21c8ceb0157da60d`.
Root review: `ROOT_INDICES24_31_REVIEW.json`.

The v17 audit verifies all eight exact nodes and worker completion records.
Root verified 13 artifact bindings and ordered JUnit indices32–39. Inputs
remain stable, with no cap or survivors. Audit: `execution_v17/AUDIT_v17.json`,
SHA-256 `3cfbe5d496bee9cfa4ca23dacaa403e716719810add7d42d3ab3329074cdac05`.
Root review: `ROOT_INDICES32_39_REVIEW.json`. These remain historical results.

The v18 and v19b audits add 16 disjoint `chebfun3f` cases, indices 0–15
(combined historical selectors 40–55). Both gates finished with eight passes,
no errors/skips/caps/survivors and 42,902 unchanged guarded inputs. Root
verified their artifact bindings, JUnit results and all nine cumulative audit
hashes for 56 distinct nodes. Evidence under the same frozen gate root:

| Gate | Original handle | Wall time | Peak summed RSS | Audit SHA-256 |
|---|---:|---:|---:|---|
| `execution_v18` | 10445 | 307.04 s | 1,274,512 KiB | `664e4b5722bae230d792886bd425bf372628a341d19d6e31a4a4fce7e558352a` |
| `execution_v19b` | 25308 | 317.60 s | 1,435,196 KiB | `c241ed3a48e8592981b5247a7251cb42af1cf320b60e721f3cdc9df68701abb0` |

Cumulative source record: `execution_v19b/CUMULATIVE_STATUS_56_OF_80_CANDIDATE.json`;
root review: `execution_v19b/ROOT_REVIEW.json`.

### Later corrections do not rewrite the historical results

Current main has a separately qualified scalar JAX NDF PDE default and native
spatial adaptation (`5b9d8c14`); see `pde15s_ndf_source_cpu_20261010.json` for
source controls and remaining systems/Fourier/complex/method limitations.
The literal 20 minimax predicates are separately qualified in
`minimax_native_clauses_cpu_20261010.json`. Neither correction changes the
frozen wrapper failure/skip reported above.

### Further historical direct cases

The v20 and v21 gates add 16 disjoint chebfun3f cases (members16–31),
bringing the frozen e6748499 total to72/80. Both have eight passes, stable
inputs and no caps or survivors. Root verified all artifact bindings, JUnit
results and all cumulative audit hashes for72 unique nodes.

| Gate | Original handle | Wall time | Peak summed RSS |
|---|---:|---:|---:|
| execution_v20 |94553|331.82s|1,468,204KiB|
| execution_v21 |49874|300.08s|1,307,032KiB|

Audits: execution_v20/AUDIT_v20.json (`a8606e403fdb6f31cfa3b6b8ecffb2f52b2978d97dc80e2ccc9873dba0da935b`)
and execution_v21/AUDIT_v21.json (`5f45c82a4754c028191dd901b6bcdcde5615d400fca13913ef7cbfc2a6ed7fd2`).
Cumulative record: execution_v21/CUMULATIVE_STATUS_72_OF_80_CANDIDATE.json;
root review: execution_v21/ROOT_REVIEW.json. These remain historical results.
