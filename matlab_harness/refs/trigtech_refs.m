%% Generate MATLAB reference data for Trigtech tests.
%
% Cross-validates the chebfunjax Trigtech (periodic tech) against MATLAB
% Chebfun's @trigtech at machine precision.  Added by Claude Opus 4.8 to
% close the trigtech golden-ref gap (53 MATLAB test files, previously no
% Python cross-validation).
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/trigtech_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Deterministic test points strictly inside (-1, 1) so Python evaluates
% at the identical locations.
t = linspace(-0.9, 0.9, 15).';
ref.test_points = t;

% ---- Test functions (smooth, periodic on [-1, 1]) ----
f1 = trigtech(@(x) exp(cos(pi*x)));
f2 = trigtech(@(x) cos(2*pi*sin(pi*x)));
f3 = trigtech(@(x) 1 + sin(pi*x) + 0.5*cos(2*pi*x));   % finite trig poly
f4 = trigtech(@(x) exp(sin(pi*x)) .* cos(3*pi*x));

% ---- Construction + evaluation ----
ref.f1_length = length(f1);
ref.f2_length = length(f2);
ref.f3_length = length(f3);
ref.f4_length = length(f4);
ref.f1_vals = feval(f1, t);
ref.f2_vals = feval(f2, t);
ref.f3_vals = feval(f3, t);
ref.f4_vals = feval(f4, t);

% ---- Definite integral over [-1, 1] ----
ref.f1_sum = sum(f1);
ref.f2_sum = sum(f2);
ref.f3_sum = sum(f3);
ref.f4_sum = sum(f4);

% ---- Derivative ----
ref.f1_diff_vals = feval(diff(f1), t);
ref.f2_diff_vals = feval(diff(f2), t);
ref.f4_diff_vals = feval(diff(f4), t);
% second derivative
ref.f1_diff2_vals = feval(diff(f1, 2), t);

% ---- Cumsum (antiderivative; requires zero mean) ----
g = trigtech(@(x) sin(pi*x) + cos(3*pi*x));   % zero mean
ref.g_cumsum_vals = feval(cumsum(g), t);

% ---- Arithmetic ----
ref.sum_vals   = feval(f1 + f3, t);
ref.minus_vals = feval(f1 - f3, t);
ref.times_vals = feval(f1 .* f3, t);
ref.scalar_mul_vals = feval(2.5 * f1, t);

% ---- Roots ----
r = sort(roots(trigtech(@(x) sin(pi*x))));
ref.sin_roots = r;

% ---- Finite trig polynomial: exact Fourier coefficients ----
% f3 = 1 + sin(pi x) + 0.5 cos(2 pi x); store its coeffs (descending
% wavenumber order as MATLAB trigtech stores them).
ref.f3_coeffs = f3.coeffs;

% ---- vscale ----
ref.f1_vscale = vscale(f1);

save(fullfile(outdir, 'trigtech.mat'), '-struct', 'ref');
fprintf('Wrote trigtech.mat with %d fields.\n', numel(fieldnames(ref)));
