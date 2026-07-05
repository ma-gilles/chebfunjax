%% MATLAB reference data for Singfun (endpoint singularities) and
%  Deltafun (Dirac deltas) — standalone cross-validation.
%
% MATLAB Chebfun has 24 singfun + 19 deltafun test files; chebfunjax
% previously exercised these classes only thinly via the fun-layer ref.
% This pins the core operations at machine precision.  Added by Claude
% Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/fun_singdelta_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Interior evaluation points (avoid endpoints and delta locations).
pts = linspace(-0.85, 0.85, 15).';
ref.pts = pts;

% =====================  SINGFUN  =====================
% s1 = (1-x)^(-1/2) cos(x),  exponents [0, -1/2]
s1 = singfun(@(x) (1-x).^(-0.5).*cos(x), struct('exponents', [0, -0.5]));
ref.s1_vals = feval(s1, pts);
ref.s1_sum  = sum(s1);

% s2 = (1+x)^(1/2) exp(x),  exponents [1/2, 0]
s2 = singfun(@(x) (1+x).^(0.5).*exp(x), struct('exponents', [0.5, 0]));
ref.s2_vals = feval(s2, pts);
ref.s2_sum  = sum(s2);

% s3 = 1/sqrt(1-x^2),  exponents [-1/2, -1/2]  (arcsine weight; sum = pi)
s3 = singfun(@(x) (1+x).^(-0.5).*(1-x).^(-0.5), ...
             struct('exponents', [-0.5, -0.5]));
ref.s3_vals = feval(s3, pts);
ref.s3_sum  = sum(s3);

% derivative of s2 (smooth-ish, weakly singular at x=-1)
ref.s2_diff_vals = feval(diff(s2), pts);

% =====================  DELTAFUN  =====================
% smooth part exp(x) on [-1,1], one delta (mag 2 at x = 0.5)
f = fun.constructor(@(x) exp(x), struct('domain', [-1, 1]));
d1 = deltafun(f, struct('deltaMag', 2, 'deltaLoc', 0.5));
ref.d1_sum = sum(d1);                 % int exp + 2
% Away from the delta the deltafun evaluates to its smooth part.
ref.d1_smooth_vals = feval(f, pts);

% two deltas (mags [1.5, -0.5] at [-0.4, 0.6])
d2 = deltafun(f, struct('deltaMag', [1.5, -0.5], 'deltaLoc', [-0.4, 0.6]));
ref.d2_sum = sum(d2);                 % int exp + 1.0

% deltafun with a non-trivial smooth part: sin(3x) + deltas
g = fun.constructor(@(x) sin(3*x), struct('domain', [-1, 1]));
d3 = deltafun(g, struct('deltaMag', [1, 1, 1], ...
                        'deltaLoc', [-0.5, 0, 0.5]));
ref.d3_sum = sum(d3);                 % int sin(3x) (=0) + 3

save(fullfile(outdir, 'fun_singdelta.mat'), '-struct', 'ref');
fprintf('Wrote fun_singdelta.mat with %d fields.\n', numel(fieldnames(ref)));
