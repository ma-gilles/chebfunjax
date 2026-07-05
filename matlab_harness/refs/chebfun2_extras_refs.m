%% Expanded MATLAB reference data for Chebfun2 operations.
%
% The pre-existing chebfun2d ref covered eval/sum2/norm/rank on one
% function.  This adds partial derivatives, double integrals, norms and
% rank on richer functions, cross-validating chebfunjax's Chebfun2
% against MATLAB @chebfun2 (R2025b).  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun2_extras_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Deterministic evaluation grid inside [-1,1]^2.
xv = linspace(-0.8, 0.8, 5);
yv = linspace(-0.7, 0.7, 5);
[X, Y] = meshgrid(xv, yv);
ref.X = X;  ref.Y = Y;

% ---- f = exp(x) sin(2y) ----
f = chebfun2(@(x,y) exp(x).*sin(2*y));
ref.f_eval  = feval(f, X, Y);
ref.f_sum2  = sum2(f);
ref.f_norm  = norm(f);
ref.f_rank  = rank(f);
ref.f_dx    = feval(diff(f, 1, 2), X, Y);   % d/dx
ref.f_dy    = feval(diff(f, 1, 1), X, Y);   % d/dy

% ---- g = cos(3 x y) + x ----
g = chebfun2(@(x,y) cos(3*x.*y) + x);
ref.g_eval = feval(g, X, Y);
ref.g_sum2 = sum2(g);
ref.g_norm = norm(g);
ref.g_dx   = feval(diff(g, 1, 2), X, Y);
ref.g_dy   = feval(diff(g, 1, 1), X, Y);

% ---- Laplacian of a harmonic-ish function: h = exp(x) cos(y) ----
% (exact: h_xx + h_yy = 0)
h = chebfun2(@(x,y) exp(x).*cos(y));
ref.h_lap = feval(diff(h, 2, 2) + diff(h, 2, 1), X, Y);   % ~ 0

save(fullfile(outdir, 'chebfun2_extras.mat'), '-struct', 'ref');
fprintf('Wrote chebfun2_extras.mat with %d fields.\n', numel(fieldnames(ref)));
