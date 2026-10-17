# Python execution evidence for Randfuneig

[Reference example page](../../examples/ode-eig/Randfuneig.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `c3aa9f88923b079974510edbb04c7fa7304b56ce6d0e26869050b5a733e54d63`.
Cached native HTML SHA-256: `726f7014b17d63c95530a9937b9e11eaf454b036695f6342f1edc1e804da4ce0`.

## Recorded note 1

<!-- Exact original note SHA-256: 1b85dd983a2a0937b72f9618654008af5b3dd27c78d795375fdb9397c4f9439b -->

> **Execution qualification.** The figures and printed counts below come from one complete CPU execution of the literal source computations with chebfunjax. Sections 1–3 retain `n=1000`, section 2 uses `m=10000`, and section 5 retains the 401×401 complex coefficient array. The Python script uses explicit NumPy and JAX RNG adapters seeded with 2017; these are not MATLAB RNG streams, so counts and individual spectrum points need not match the published MATLAB sample. Native eigenfunction normalization remains unverified. All five canvases are 600×269 pixels; Matplotlib dots are visibly larger than the reference, and framing and line style can differ. The MATLAB code and original discussion remain below; output blocks contain the actual Python stdout.

<!-- End preserved historical qualification note. -->
