# Python execution evidence for Gibbs2D

[Reference example page](../../examples/approx2/Gibbs2D.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `491a0e0451226bf4319ffcb44eadc7a7204ac3de2fb5e1dd9f2efe131ebeb0ec`.
Cached native HTML SHA-256: `fa10c48fd5954c4439df3a04f4f4fb1a0ebfc8c8287f870c079fc05aa3b774e9`.

## Recorded note 1

<!-- Exact original note SHA-256: 66ff6453a3bf0184a736bc05ca920a7ee5b942ae39244f4c8368aa2226fd9082 -->

> **Qualification:** The complete CPU source replay passed after preserving native seed nodes on [-1,1]: the original 100x100 computation, all six extrema calls, twelve fixed-4000 reconstructions and four active-set solves completed without fallback or unsupported branches. The triangle minimum -0.004032268925530 agrees with a fresh R2025b native source run to about 3.3e-16 at full precision. The maximum is 1.171307334393718 versus fresh native 1.1722914503389705, leaving a gap of about 0.0009841. The maximum seed remains sensitive to rounding at near-equal sampled values; the JAX adapter follows R2017a optimization mechanics while this fresh native run uses the installed R2025b provider. The historical published native extrema 1.294875501773784 and -0.228957699300502 are a separate reference. All eight figures are actual 600x269 outputs, byte-identical to the prior publication. Lighting, face interpolation, automatic contour levels, framing and typography remain graphics differences. These results do not establish full numerical or graphics parity.

<!-- End preserved historical qualification note. -->
