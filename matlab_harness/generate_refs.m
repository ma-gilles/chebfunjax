% generate_refs.m — Run in MATLAB with Chebfun on path.
% Generates reference data (.mat files) that Python tests compare against.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('Generating reference data into: %s\n', outdir);

%% --- Quadrature points ---
ref = struct();
for n = [5, 10, 17, 32, 64, 128]
    ref.(sprintf('chebpts2_n%d', n)) = chebpts(n);           % 2nd kind (default)
    ref.(sprintf('chebpts1_n%d', n)) = chebpts(n, 1);        % 1st kind
    [x, w] = legpts(n);
    ref.(sprintf('legpts_x_n%d', n)) = x;
    ref.(sprintf('legpts_w_n%d', n)) = w;
end
save(fullfile(outdir, 'quadrature.mat'), '-struct', 'ref');
fprintf('  quadrature.mat written (%d fields)\n', numel(fieldnames(ref)));

%% --- Chebfun construction and evaluation ---
ref = struct();
f = chebfun(@(x) sin(x));
ref.sin_coeffs = chebcoeffs(f, length(f));
ref.sin_length = length(f);
ref.sin_sum = sum(f);
ref.sin_at_half = f(0.5);

g = chebfun(@(x) exp(x));
ref.exp_coeffs = chebcoeffs(g, length(g));
ref.exp_length = length(g);
ref.exp_sum = sum(g);

h = chebfun(@(x) abs(x));
ref.abs_length = length(h);
ref.abs_sum = sum(h);
ref.abs_breakpoints = h.domain;

% Differentiation
fp = diff(f);
ref.sin_diff_coeffs = chebcoeffs(fp, length(fp));

% Integration
F = cumsum(f);
ref.sin_cumsum_at_half = F(0.5);

% Roots
r = roots(chebfun(@(x) x.^2 - 0.25));
ref.roots_x2minus025 = sort(r);

save(fullfile(outdir, 'chebfun_basic.mat'), '-struct', 'ref');
fprintf('  chebfun_basic.mat written (%d fields)\n', numel(fieldnames(ref)));

%% --- Basis transforms ---
ref = struct();
for n = [8, 16, 32, 64]
    c = randn(n, 1);
    ref.(sprintf('cheb2leg_n%d_in', n)) = c;
    ref.(sprintf('cheb2leg_n%d_out', n)) = cheb2leg(c);
    ref.(sprintf('leg2cheb_n%d_in', n)) = c;
    ref.(sprintf('leg2cheb_n%d_out', n)) = leg2cheb(c);
end
save(fullfile(outdir, 'transforms.mat'), '-struct', 'ref');
fprintf('  transforms.mat written (%d fields)\n', numel(fieldnames(ref)));

%% --- Differentiation matrices ---
ref = struct();
for n = [5, 10, 20]
    ref.(sprintf('diffmat_n%d', n)) = diffmat(n);
end
save(fullfile(outdir, 'diffmat.mat'), '-struct', 'ref');
fprintf('  diffmat.mat written (%d fields)\n', numel(fieldnames(ref)));

%% --- AAA rational approximation ---
ref = struct();
Z = linspace(-1, 1, 1000)';
F = abs(Z);
[r, pol, res, zer] = aaa(F, Z, 'tol', 1e-10);
ref.aaa_abs_poles = sort(pol);
ref.aaa_abs_zeros = sort(zer);
ref.aaa_abs_vals = r(Z);
save(fullfile(outdir, 'aaa.mat'), '-struct', 'ref');
fprintf('  aaa.mat written (%d fields)\n', numel(fieldnames(ref)));

fprintf('Done. All references saved to %s\n', outdir);
