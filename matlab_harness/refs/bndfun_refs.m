%% Generate MATLAB reference data for Bndfun tests.
%
% Requires the Chebfun toolbox to be on the path:
%   addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref')
%
% Usage (from repo root):
%   module load matlab/R2025b
%   cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/bndfun_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% --- Test points ---
% 10 points in (0, pi) for evaluation / derivative tests
test_points_0_pi = linspace(0.05, pi - 0.05, 10)';
ref.test_points_0_pi = test_points_0_pi;

%% --- sin on [0, pi] ---
f_sin = bndfun(@sin, struct('domain', [0, pi]));

% Evaluation at test points
ref.sin_values_0_pi = feval(f_sin, test_points_0_pi);

% Definite integral: ∫₀^π sin = 2
ref.sum_sin_0_pi = sum(f_sin);

% Derivative values (should match cos at test points)
fp_sin = diff(f_sin);
ref.diff_sin_values = feval(fp_sin, test_points_0_pi);

% Cumsum: antiderivative evaluated at test points
F_sin = cumsum(f_sin);
ref.cumsum_sin_values = feval(F_sin, test_points_0_pi);

%% --- exp on [1, 3] ---
f_exp = bndfun(@exp, struct('domain', [1, 3]));

% Definite integral: ∫₁^3 exp = e³ - e
ref.sum_exp_1_3 = sum(f_exp);

%% --- Roots ---
% Quadratic x² - 1 on [-2, 2]: roots at ±1
f_quad = bndfun(@(x) x.^2 - 1, struct('domain', [-2, 2]));
ref.roots_xsq_minus1 = roots(f_quad);

% sin on [0, 2π]: interior root at π
f_sin_2pi = bndfun(@sin, struct('domain', [0, 2*pi]));
ref.roots_sin_0_2pi = roots(f_sin_2pi);

%% --- Extrema ---
% max of sin on [0, pi]: val=1, pos=π/2
[~, pos_max] = max(f_sin);
ref.max_sin_val = feval(f_sin, pos_max);
ref.max_sin_pos = pos_max;

%% Save
save(fullfile(outdir, 'bndfun.mat'), '-struct', 'ref', '-v7');
fprintf('Saved bndfun.mat to %s\n', outdir);
