% Exact sequential source clauses 31--35 from tests/chebfun/test_feval.m.
% Capture only; this is not a replacement port of the test source.
root = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
addpath(root);
cd(root);
outdir = '/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/feval_unbounded_capture_20261013';
sourceFile = fullfile(root, 'tests/chebfun/test_feval.m');
seedFile = fullfile(root, 'seedRNG.m');
provenance = struct('release', version('-release'), 'version', version, ...
    'sourcePath', sourceFile, 'seedPath', seedFile, 'nativeCommit', ...
    '7574c77680d7e82b79626300bf255498271a72df');

% Pass 31 starts the only reseed in the source tail. Do not reseed between
% clauses: constructor and rand consumption remains in source order.
dom31 = [-2 7];
seedRNG(6178);
rngAfterSeed31 = rng;
x31 = diff(dom31) * rand(100, 1) + dom31(1);
rngAfterDraw31 = rng;
pow31 = -0.5;
op31 = @(x) (x - dom31(1)).^pow31 .* sin(200*x);
f31 = chebfun(op31, dom31, 'exps', [pow31 0], 'splitting', 'on');
rngAfterConstruct31 = rng;
fval31 = feval(f31, x31);
exact31 = feval(op31, x31);
vscale31 = vscale(f31);
bound31 = 1e5 * eps * norm(exact31, inf);
err31 = fval31 - exact31;
pass31 = norm(err31, inf) < bound31;
rngAfterPass31 = rng;

% Pass 32, entire real line, finite removable-singularity function.
dom32 = [-Inf Inf];
domCheck32 = [-1e2 1e2];
x32 = diff(domCheck32) * rand(100, 1) + domCheck32(1);
rngAfterDraw32 = rng;
op32 = @(x) (1-exp(-x.^2))./x;
f32 = chebfun(op32, dom32);
rngAfterConstruct32 = rng;
fval32 = feval(f32, x32);
exact32 = feval(op32, x32);
vscale32 = vscale(f32);
plusInf32 = feval(f32, Inf);
minusInf32 = feval(f32, -Inf);
bound32 = 1e1 * eps * vscale32;
err32 = fval32 - exact32;
pass32 = (norm(err32, inf) < bound32) && ...
    (plusInf32 < eps*vscale32) && (minusInf32 < eps*vscale32);
rngAfterPass32 = rng;

% Pass 33, same infinite domain, prescribed endpoint exponents.
dom33 = dom32;
op33 = @(x) x.^2 .* (1-exp(-x.^2));
f33 = chebfun(op33, dom32, 'exps', [2 2]);
rngAfterConstruct33 = rng;
fval33 = feval(f33, x32);
exact33 = feval(op33, x32);
vscale33 = vscale(f33);
plusInf33 = feval(f33, Inf);
minusInf33 = feval(f33, -Inf);
bound33 = 1e5 * eps * vscale33;
err33 = fval33 - exact33;
pass33 = (norm(err33, inf) < bound33) && ...
    (plusInf33 == Inf) && (minusInf33 == Inf);
rngAfterPass33 = rng;

% Pass 34, right half-line.
dom34 = [1 Inf];
domCheck34 = [1 1e2];
x34 = diff(domCheck34) * rand(100, 1) + domCheck34(1);
rngAfterDraw34 = rng;
op34 = @(x) x .* exp(-x);
f34 = chebfun(op34, dom34);
rngAfterConstruct34 = rng;
fval34 = feval(f34, x34);
exact34 = feval(op34, x34);
vscale34 = vscale(f34);
plusInf34 = feval(f34, Inf);
bound34 = 1e1 * eps * vscale34;
err34 = fval34 - exact34;
pass34 = (norm(err34, inf) < bound34) && (plusInf34 < 1e2*eps*vscale34);
rngAfterPass34 = rng;

% Pass 35, left half-line array-valued output. MATLAB norm(matrix,inf) is
% the maximum row sum, exactly as written in the native predicate.
dom35 = [-Inf -3*pi];
domCheck35 = [-1e6 -3*pi];
x35 = diff(domCheck35) * rand(100, 1) + domCheck35(1);
rngAfterDraw35 = rng;
op35 = @(x) [exp(x) x.*exp(x) (1-exp(x))./x];
f35 = chebfun(op35, dom35);
rngAfterConstruct35 = rng;
fval35 = feval(f35, x35);
exact35 = feval(op35, x35);
vscale35 = vscale(f35);
minusInf35 = feval(f35, -Inf);
bound35 = 1e2 * max(eps * vscale35);
err35 = fval35 - exact35;
norm35 = norm(err35, inf);
endpointPred35 = all(minusInf35 < 1e2*eps*vscale35);
pass35 = (norm35 < bound35) && endpointPred35;
rngAfterPass35 = rng;

pass = [pass31 pass32 pass33 pass34 pass35];
assert(all(pass), 'Native clauses 31--35 did not all pass.');
meta = struct('clauses', [31 32 33 34 35], 'seed', 6178, ...
    'seedRNGImplementation', 'rng(''default''); rng(s)', ...
    'rngAfterSeed31', rngAfterSeed31, 'rngAfterDraw31', rngAfterDraw31, ...
    'rngAfterConstruct31', rngAfterConstruct31, ...
    'rngAfterPass31', rngAfterPass31, 'rngAfterDraw32', rngAfterDraw32, ...
    'rngAfterConstruct32', rngAfterConstruct32, ...
    'rngAfterPass32', rngAfterPass32, 'rngAfterConstruct33', rngAfterConstruct33, ...
    'rngAfterPass33', rngAfterPass33, 'rngAfterDraw34', rngAfterDraw34, ...
    'rngAfterConstruct34', rngAfterConstruct34, 'rngAfterPass34', rngAfterPass34, ...
    'rngAfterDraw35', rngAfterDraw35, 'rngAfterConstruct35', rngAfterConstruct35, ...
    'rngAfterPass35', rngAfterPass35, 'provenance', provenance);
outputFile = fullfile(outdir, 'feval_tail31_35_native.mat');
save(outputFile, 'meta', 'pass', ...
    'dom31', 'pow31', 'x31', 'fval31', 'exact31', 'vscale31', 'bound31', 'err31', ...
    'dom32', 'domCheck32', 'x32', 'fval32', 'exact32', 'vscale32', ...
    'plusInf32', 'minusInf32', 'bound32', 'err32', ...
    'dom33', 'fval33', 'exact33', 'vscale33', 'plusInf33', 'minusInf33', ...
    'bound33', 'err33', ...
    'dom34', 'domCheck34', 'x34', 'fval34', 'exact34', 'vscale34', ...
    'plusInf34', 'bound34', 'err34', ...
    'dom35', 'domCheck35', 'x35', 'fval35', 'exact35', 'vscale35', ...
    'minusInf35', 'bound35', 'err35', 'norm35', 'endpointPred35', '-v7');
fprintf('native clauses %s; capture %s\n', mat2str(pass), outputFile);
