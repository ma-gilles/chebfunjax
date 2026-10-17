# Python execution evidence for Pushnitski

[Reference example page](../../examples/approx/Pushnitski.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `dccb31b28c2fea3302443b8bebd356eda0b0de5377424dc39691dbcf0159d2d7`.
Cached native HTML SHA-256: `a5a6721496004e4e98f0ec1a55e8080cb5b9b17bc7504ca791f12d27cc5ad9b9`.

## Recorded note 1

<!-- Exact original note SHA-256: 38590c3b649c10a09c44e2375d301f5e605261b2d80b86255c21755a9ed718fa -->

> **Execution qualification.** All four polynomial, four rational and four CF fits completed on the literal fixed-1000-per-piece input in guarded CPU stages. A monolithic attempt hit the 4 GiB limit; the figures replay actual captured outputs. The polynomial timer is historical measured execution; continuous rational/CF cell timers were not recorded. Large CF matrices use dense JAX `eigh`; matched native-input comparisons are recorded in [execution evidence](../../pushnitski_source_cpu_20261011.json). Whole-example numerical, historical figure and pixel parity remain unverified; the native oracle has explicitly incomplete prebound Symbolic provider coverage. Timings on the shared host are not benchmarks.

## Recorded note 2

<!-- Exact original note SHA-256: 2eca5bb510343a71a8526a32da398ca28b77ebb685fc966af96e29999bd935ee -->


Plotting emitted the following actual warnings (file paths omitted):

## Recorded note 3

<!-- Exact original note SHA-256: af02b87fcc10f4fb5738805581aaff31d13bcc62bd48cd5765da3b5e48550b5c -->

*The rational type (3,3) fit completed separately after the memory-capped attempt; the two initializer messages above are retained from that original attempt. No continuous rational cell timer was captured.*

## Recorded note 4

<!-- Exact original note SHA-256: b27b0f4dfe5d5a573dac5212f932f5176607585d2f4c6b8f369909c5d918b93a -->

*The four CF calls completed in a separate guarded process; a continuous cell elapsed time was not recorded.*

<!-- End preserved historical qualification note. -->
