# Inverse CPU profile at d7c2c8d0

A newer [MATLAB/JAX comparison](inverse_matlab_cpu_20261010.json) measures public flower inversion at fa6c0901: warmed medians 0.584467s JAX and 0.630269s MATLAB; first after setup 17.779799s and 0.767396s. Adaptive representation lengths still differ. These are observations from a shared host, not a universal performance ratio.

Frozen commit d7c2c8d066df2dc355b229058f8deb7bb1d23c69. Both fresh processes terminal0/stable/uncensored/no survivors; each2964runtimefiles rehashed clean. Inverse80 and derivative14 binary captures verified. Quiet window released after final audit; no numerical process remains.

| Operation | Setup | First after setup | Warm median | Warm range |
|---|---:|---:|---:|---:|
| Default inverse | 18.488463s | 22.272436s | 0.611954s | 0.611267–0.621347s |
| Derivative roots | 18.846952s | 16.230008s | 0.110641s | 0.109976–0.117172s |

Actual forward length2090, inverse8880. Every inverse call used Brent target lengths2,4097,4096,8192,2; the initial2 now comes from public vectorCheck normalization. These were observed, not imposed. Roundtrip error1.0547118733938987e-15 satisfies unchanged1e-10. Both arms built bit-identical forward/derivative coefficients; each arm repeated bit-identical outputs. All serialized captures finite.

First inverse recorded544 backend_compile_and_load calls,14.264820s cumulative; first derivative roots411 calls,11.191997s cumulative. All five warmed calls in both arms had no captured backend compilation rows. Transform cache was initially empty and had one miss by the end of setup; no later misses,97hits added per operation. Setup and first-call compilation must not be conflated.

First inverse domain discovery17.154507s included minandmax16.286705s and recursive roots15.778117s; default eigen adapter9.760869s and subdivision5.987810s are nested portions. A representative warm inverse(call3) spent0.470971s in inverse values callbacks inside0.486324s public construction and0.1247s outside that constructor interval. Representative derivative roots(call3)0.109976s included default-eigen adapter0.073859s(0.043051s Python self) and subdivision0.012642s. These cumulative numbers overlap; they do not form an exclusive accounting or pure device-kernel measurement.

Raw stats show109 default eigen leaf calls,217 recursive roots calls and108 subdivision calls per operation. Source review identifies eager finite/spectrum adapters as a concrete next attribution target, but existing profiler rows do not assign every backend compilation to shape/kernel. No production optimization is accepted from these totals alone.

Instrumentation: cProfile, recursion/count wrappers and synchronized Brent capture are present; hashes/IO/scalar finiteness checks are outside timed intervals. First-after-setup is not process startup. Five warm observations are not confidence bounds. Host1/5/15-minute loads are retained in SUMMARY_v1.json (1-minute3.77–5.71 observed); quiet agent coordination is not host exclusivity. There is no matched revision speedup or current MATLAB ratio. Historical800x and earlier warm measurements are not current comparators.

Provider note: initial preflight inspected environment compiler source; actual pinned runtime loaded /scratch/gpfs/AMITS/mg6942/.cache/rattler/cache/uv-cache/archive-v0/kbfpF4D2exWHSnMSv9Z6g/jax/_src/compiler.py. Its observed hash and compiler signatures are explicitly bound in SUMMARY_v1.json; actual runtime provenance was clean and source symbol names matched. Interpret actual runtime rows, not preflight line offsets. No source or numerical implementation changed.

## Independent review

Root independently rehashed 2,964 observed runtime files per arm and verified all
94 saved binary captures for hash, shape, finite values and repeated output bits.
JUnit and source/payload bindings were also checked. Compact measurements and
evidence hashes are retained in [the JSON record](inverse_cpu_profile_20261009.json).
Raw evidence is under `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Full-suite, page parity, fresh MATLAB measurements and final CI remain open.
