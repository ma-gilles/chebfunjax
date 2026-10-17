# Python execution evidence for Tricky

[Reference example page](../../examples/quad/Tricky.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `8734c4b211af25df7f925358f7e0f46445ed71b52a6ef8cb2b14d282ee3e720c`.
Cached native HTML SHA-256: `88782bbb149011b889d1ef37e22ea4494d0f3f224d91c2ff2addd7a6b59ed48c`.

## Recorded note 1

<!-- Exact original note SHA-256: 523445d01f9a092d51b8d3946ef1b4cd3a985b0f6f05b56c58d4f4b67e832099 -->

> Execution qualification (2026-10-17): Airy output blocks 6 and 7 and figure 4 come from a separate completed CPU execution of those current source cells using the JAX Airy primitive. The unbounded integral is now finite; both displayed integrals match the historical native displays. The ten other output blocks and eight other figures retain the earlier completed full-script capture; their scientific source statements are unchanged. Pinned JAX/JAXlib 0.11.0 was used for both runs. The intentional unresolved approximation and its `Function not resolved using 6091 pts.` warning remain. Axis limits, ticks, fonts and raster differences remain; pixel parity is not claimed. See [`tricky_airy_jax_cpu_20261017.json`](../../tricky_airy_jax_cpu_20261017.json) and the [earlier full-run receipt](../../tricky_full_current_cpu_20261017.json).

<!-- End preserved historical qualification note. -->
