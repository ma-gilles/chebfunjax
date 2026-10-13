restoredefaultpath;
M = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
addpath(M);
assert(strcmp(which('chebfun'), fullfile(M, '@chebfun', 'chebfun.m')) || startsWith(which('chebfun'), M));
assert(strcmp(which('seedRNG'), fullfile(M, 'seedRNG.m')));

% Replay tests/chebfun/test_diff.m clause 10, then clause 11 in source order.
dom = [-2 7];
domCheck = [dom(1)+0.1 dom(2)-0.1];
seedRNG(6178);
x10 = diff(domCheck) * rand(100, 1) + domCheck(1);
pow = -0.5;
op = @(x) (x - dom(1)).^pow .* sin(200*x);
f = chebfun(op, dom, 'exps', [pow 0], 'splitting', 'on');
df = diff(f);
vals_df = feval(df, x10);
df_exact = @(x) (x - dom(1)).^(pow-1) .* (pow*sin(200*x) + 200*(x-dom(1)).*cos(200*x));
vals_exact = feval(df_exact, x10);
err10 = vals_df - vals_exact;
pass10 = norm(err10, inf) < 1e6*eps*norm(vals_exact, inf);

% The native test does not reseed before these draws.
dom = [-Inf Inf];
domCheck = [-1e2 1e2];
x11 = diff(domCheck) * rand(100, 1) + domCheck(1);
op = @(x) (1-exp(-x.^2))./x;
f = chebfun(op, dom, 'splitting', 'on');
g = diff(f);
op_g = @(x) 2*exp(-x.^2) + (exp(-x.^2)-1)./x.^2;
gVals = feval(g, x11);
gExact = op_g(x11);
err11 = gVals - gExact;
pass11 = norm(err11, inf) < 1e3*eps*get(g,'vscale');

save('/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/diff_rng_capture_20261014/capture.mat', ...
    'x10', 'x11', 'vals_df', 'vals_exact', 'err10', 'pass10', ...
    'gVals', 'gExact', 'err11', 'pass11', '-v7');
disp(jsonencode(struct('pass10', pass10, 'pass11', pass11, ...
    'clause10_err', norm(err10, inf), ...
    'clause10_bound', 1e6*eps*norm(vals_exact, inf), ...
    'clause11_err', norm(err11, inf), ...
    'clause11_bound', 1e3*eps*get(g,'vscale'))));
