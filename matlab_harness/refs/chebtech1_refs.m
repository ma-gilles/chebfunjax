%% MATLAB reference data for Chebtech1 (Chebyshev 1st-kind tech).
%
% Cross-validates chebfunjax's Chebtech1 against MATLAB @chebtech1
% (R2025b): construction, feval, sum, diff, cumsum, roots.  Added by
% Claude Opus 4.8 (chebtech1 previously had no MATLAB golden ref).
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebtech1_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
xt = linspace(-0.9, 0.9, 15).';
ref.xt = xt;

f = chebtech1(@(x) exp(x).*sin(3*x));
ref.f_length = length(f);
ref.f_vals = feval(f, xt);
ref.f_sum = sum(f);
ref.f_diff = feval(diff(f), xt);
ref.f_cumsum = feval(cumsum(f), xt);

g = chebtech1(@(x) cos(5*x));
ref.g_roots = sort(roots(g));

% finite polynomial: exact coeffs preserved
p = chebtech1(@(x) 1 + 2*x + 3*x.^2);
ref.p_vals = feval(p, xt);
ref.p_sum = sum(p);      % int_{-1}^1 (1 + 2x + 3x^2) = 2 + 0 + 2 = 4

save(fullfile(outdir, 'chebtech1.mat'), '-struct', 'ref');
fprintf('Wrote chebtech1.mat with %d fields.\n', numel(fieldnames(ref)));
