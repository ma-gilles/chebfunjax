%% Comprehensive MATLAB reference data for core Chebfun operations.
%
% Cross-validates chebfunjax's Chebfun (1D) against MATLAB Chebfun's
% @chebfun for the operations that its 166-file test suite exercises:
% construction, evaluation, calculus (sum/cumsum/diff), roots, min/max,
% norms, abs/sign, arithmetic, composition, restrict.  Added by Claude
% Opus 4.8 to broaden core-class golden-ref coverage (the pre-existing
% chebfun ref only covered basic sin/exp construction).
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun_core_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Deterministic evaluation points inside (-1, 1).
xt = linspace(-0.93, 0.93, 17).';
ref.xt = xt;

% ---- A smooth test function on [-1, 1] ----
f = chebfun(@(x) exp(x) .* sin(5*x));
ref.f_length = length(f);
ref.f_vals   = feval(f, xt);
ref.f_sum    = sum(f);
ref.f_diff   = feval(diff(f), xt);
ref.f_diff2  = feval(diff(f, 2), xt);
ref.f_cumsum = feval(cumsum(f), xt);
ref.f_norm2  = norm(f);
ref.f_norm1  = norm(f, 1);
ref.f_norminf = norm(f, inf);

% ---- Roots ----
g = chebfun(@(x) sin(6*x));
ref.g_roots = sort(roots(g));

% ---- min / max (values and locations) ----
h = chebfun(@(x) x.^3 - x);
[hmin, hminx] = min(h);
[hmax, hmaxx] = max(h);
ref.h_min = hmin;  ref.h_minx = hminx;
ref.h_max = hmax;  ref.h_maxx = hmaxx;

% ---- abs / sign (piecewise) ----
a = chebfun(@(x) x);
absf = abs(chebfun(@(x) sin(3*x)));
ref.abs_vals = feval(absf, xt);
ref.abs_sum  = sum(absf);

% ---- Arithmetic ----
p = chebfun(@(x) 1 + x.^2);
q = chebfun(@(x) cos(2*x));
ref.pq_plus  = feval(p + q, xt);
ref.pq_minus = feval(p - q, xt);
ref.pq_times = feval(p .* q, xt);
ref.pq_rdiv  = feval(p ./ (2 + q), xt);

% ---- Composition (unary math on a chebfun) ----
u = chebfun(@(x) 0.5*x);
ref.exp_u  = feval(exp(u), xt);
ref.sin_u  = feval(sin(u), xt);
ref.tanh_u = feval(tanh(chebfun(@(x) 3*x)), xt);

% ---- Restrict + subinterval integral ----
w = chebfun(@(x) exp(x));
ref.restrict_sum = sum(restrict(w, [-0.5, 0.5]));

% ---- Domain / other scalars ----
ref.f_dot_g = sum(f .* g);          % inner product
ref.poly_p_deg = length(p) - 1;      % 1 + x^2 is degree 2 (length 3)

save(fullfile(outdir, 'chebfun_core.mat'), '-struct', 'ref');
fprintf('Wrote chebfun_core.mat with %d fields.\n', numel(fieldnames(ref)));
