# Python execution evidence for RandomSwitching

[Reference example page](../../examples/ode-random/RandomSwitching.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `a93c3e31f59f5bf6db8869bfd64808303dbacf3bb6b84dc3931b77fb59e588cf`.
Cached native HTML SHA-256: `319a46f853987f72d4d99a3ce23fcb037014a696639cdfb277b4d04009f272b0`.

## Recorded note 1

<!-- Exact original note SHA-256: de5cf7dcf72f5e0ad4a1d00e0bfdfa387604934ac7068ca50de154197d183662 -->

> **Qualification of the Python results.** The prose and MATLAB code below are preserved from the native example. The figures and printed outputs come from the tested Python translation, using JAX key 1 as an explicit adapter without reproducing the MATLAB random stream. The cached published webpage starts with `rng(0)`; the checked-out `.m` source instead uses `rng(1)`. The realizations differ: the first Python figure peaks near 75 (the native reference stays below 10), and the second Python figure does not show the decay described in the source illustration. The original data and axis limits are retained. The four solves use the native defaults and `maxnorm` setting; the third stops at about 18.27595224 and its continuous solution and squared norm retain a NaN tail. Execution checks passed, but numerical accuracy differences remain unresolved. Rendering also differs in apparent stroke thickness, major and minor grids, and inward top/right ticks; source line widths are retained. These results do not establish MATLAB RNG, numerical, or visual parity.

<!-- End preserved historical qualification note. -->
