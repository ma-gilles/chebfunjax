# Python execution evidence for Inpainting1D

[Reference example page](../../examples/approx/Inpainting1D.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `3bce87b8b906969bfc82ad035e26dba5eb12c5faa14f2217501b862970ef520f`.
Cached native HTML SHA-256: `ea56ba84198ad5c73d144a539fb542b3b3ded6410e75d9d78fcc2174307efea7`.

## Recorded note 1

<!-- Exact original note SHA-256: cef45acef9413dd11dda1cb284f69902cd64051a5e29084c8248d905eeeaebc1 -->

> **Qualification:** This source computation uses public nonperiodic `cj.randnfun(.1, key=PRNGKey(1))`; its normal draws do not reproduce MATLAB `rng(1)`. All three public fits receive the same captured nine-piece corrupted function. The source Watson L1 iteration reached its unchanged 100-iteration limit and emitted the warning below; its actual recovery error is `2.166633346778370e-05`, so the native prose’s near-zero recovery is not established. L2 and minimax values use this different noise realization. Native prose/code are retained for comparison. All five figures are captured at the reference 600×270 sizes; curves, limits, ticks, grid and line styling differ. A MATLAB replay using the saved piece coefficients also reaches 100 iterations, with recovery error `7.061517343020836e-06`. The initial polynomial coefficients differ by at most `1.12e-16`; damping/root-count choices first diverge at iteration 27. This does not establish numerical or visual parity.

<!-- End preserved historical qualification note. -->
