# Exact single-to-double primal promotion with retained autodiff

## Candidate and scope

Private checkout `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/exact_promotion_candidate_20261013` is based at f6a50aaa. Publication checkout and its concurrent logical repair were not copied or modified. Only two integration paths changed, listed with SHA256 in `changed_paths_manifest.json`; `source.patch` contains the29-line source diff. Copy the new test file separately. Apply the source patch to preserve concurrent changes elsewhere in chebfun.py.

The traced Chebfun evaluation path previously used astype(float64/complex128). CPU denormal handling flushes binary32 subnormal inputs during this cast, so existing exact routing still evaluated the wrong value. The fix reuses the existing `_float32_to_float64_bits` integer widening and bitcasts its binary64 result. Complex64 real and imaginary components widen independently and combine using lax.complex, avoiding arithmetic that changes zeros/nonfinite components. Other input dtypes and the concrete evaluation path are unchanged.

The widening operation has a custom JVP: its primal uses exact integer widening; its tangent uses the existing linear float32-to-float64 cast. This preserves ordinary forward/reverse derivatives instead of silently giving zero derivatives through integer bitcasts. It is a **primal correction**, not arbitrary bit-preserving differentiation.

## Qualification

- Baseline_v2:2 value failures and3 gradient passes on the unchanged f6a source. The retained original report at `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/subnormal_routing_review_20261013/promotion/stdout` independently records the +/-minimum-float32 failure on domain[-1e-40,1e-40]. The first baseline attempt accidentally passed the unhashable Chebfun object directly to jit; that harness error is retained separately and excluded.
- Final focused gate:34 tests passed (13 new promotion/AD tests plus21 existing ordered-routing controls), terminal0, sampled peak675,800KiB, stable hashes and no owned survivors. New controls cover real/complex input, min/max signed binary32 subnormals, narrow-domain evaluation including1e-40, fused arithmetic with single/multiple pieces, exact primal words for signed zeros/normals/Inf/quietNaN, forward and reverse identity derivatives, complex holomorphic gradients/JVP conventions and second derivative at a normal point. Existing acceptance predicates were unchanged.
- New tests pass Ruff; git diff --check passes. All numerical tests use CPU16–19,JAX/jaxlib0.11.0,x64,CUDA visibility empty under90s/1536MiB supervisors. No heavy examples or performance measurements were run.
- A combined58-test attempt and a subsequent fresh broad shape-module attempt exceeded the1.5GiB sampled RSS cap. Both were stopped cleanly with stable inputs/no survivors and are not counted as passing coverage. The focused34-test gate is sufficient for this scoped dtype-promotion correction; no claim is made that the entire shape module was qualified here.

## Explicit AD limitation

`ad_v2/receipt.json` is terminal0, sampled458,644KiB; `ad_scope_v2.json` records comparisons against the original ordinary cast at normal primal. Both implementations behave identically in these controls:

|Control|Original cast|Exact-primal custom JVP|
|---|---:|---:|
|Normal reverse gradient|1|1|
|JVP with minimum subnormal float32 tangent|0|0|
|Reverse gradient with constant minimum-subnormal weight, rawfloat32 word|1|1|
|VJP with dynamic minimum-subnormal double cotangent, rawfloat32 word|0|0|

Constant folding can preserve the constant-weight result while runtime conversion flushes the dynamic input. The candidate preserves the existing AD contract and its limitations; it does not preserve arbitrary subnormal tangent/cotangent bits. `ad_scope.json` is the earlier probe without dynamic cotangents; the v2 report is authoritative. No global FP mode, native extension, callback or NumPy numerical library implementation was introduced.

All receipts/caps are retained in `receipts_summary.json`. The candidate is frozen for integration, with no commit or push performed.

Evidence package: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/exact_promotion_review_20261013`. Relative artifact names above resolve against that package. This report records the private candidate; integrated publication qualification is recorded separately.
