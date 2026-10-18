# AbsoluteValue current CPU audit

The complete source-faithful script ran on `fc674a6b` using pinned JAX/JAXlib 0.11.0, x64, CPU36–39. All five figures and eighteen Newton panels were produced. This package improves computation and presentation; numerical, raster and full-page parity remain open.

## Repairs

The script now uses the original string constructors/domains and public Chebfun plotData. Error plots construct `abs(r-abs(x))` before plotting. The previous samplewise error and artificial `maximum(error,1e-19)` are removed. Raw zero/negative interpolation values remain in the saved data; logarithmic rendering masks nonpositive values. All eighteen Newton updates and public supremum norms retain the source operands. Source line width 1.6 and axis limits are preserved.

The final page-specific font/tick/grid framing was replayed from the actual captured artists. Every plotted x/y byte and title string is unchanged. The final script's selected mathematical assignments and public plot/save calls have identical ASTs to the executed source. This bridge avoids repeating the nonlinear calculation for presentation alone. It is not a universal conversion of MATLAB point sizes.

## Actual evidence

| Execution | Result | Time | Sampled tree RSS |
|---|---|---:|---:|
| Full Python application | complete, five frames, empty stdout, no scientific warning events | 132.33 s application / 139.70 s supervisor | 4,495,924 KiB |
| Native literal five cells plus neutral eighteen-state observer | complete, final three functions word-identical between literal and observer | 93.32 s supervisor | 3,057,844 KiB |
| Final presentation replay | exact artist words, stable inputs | see receipt | 261,532 KiB |

All 1,380 actual Python runtime files were guarded before execution, with no missing/changed bindings. Import-time CUDA plugin version warning text remains in raw stderr; the actual backend was CPU. The initial observer attempt failed with SyntaxError before imports/operands and is preserved. The unrounded-current-norm observer missed Equinox wrapper frames; actual source norm calls and resulting title strings executed, but those scalar intermediates were not captured.

## Remaining differences

Current, native R2025b and cached historical figure degree counts/titles differ. Evaluating current states on finite native plotData coordinates gives maximum sampled discrepancy 2.8573e-12 at the last panel; this is not a continuous bound. Error figure03 has 66 current pieces/558 coefficients versus 71/548 native; figure05 has 927/8479 versus 945/8697. Their sampling, oscillations and tails visibly differ. No new isolated library fault was established, and no acceptance tolerance was invented for this demonstration.

All five images are 600x270. Cache pixel differences decrease after framing, but all remain unequal; source curve width, renderer stroke and scientific data differences remain visible. Native default figure geometry/font metadata differ from the historical cached publisher. The cached prose, five code cells and figure order were retained. Regeneration with actual empty stdout leaves the Markdown page byte-identical, so no page edit is proposed.

## Receipts and raw data

The immutable package contains `REPORT.json`, `NATIVE_COMPARISON.json`, `RASTER_AUDIT.json`, reference source/HTML/images and hashes, actual/raw/presentation figures, all artist arrays, native literal/observer MAT data, source snapshots, environment/runtime bindings, supervisor receipts, source AST bridge and exact publication patch. `REPORT.json` records absolute evidence paths and precise limitations. No library changes, additional test selectors, whole-API closure or final figure/page parity are claimed.
