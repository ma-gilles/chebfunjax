% aaa_refs.m — Generate MATLAB references for utils/aaa module.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/aaa_refs.m')"
%
% All test cases use explicit mmax to ensure reproducible output.
%
% Provenance: Chebfun commit 7574c77 (Y. Nakatsukasa, O. Sète, L.N. Trefethen,
% "The AAA algorithm for rational approximation", SIAM J. Sci. Comp. 2018).

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% ---- Test 1: |x| on [-1, 1], mmax=33 (type (32,32) rational approx) ----
Z1 = linspace(-1, 1, 1000).';
F1 = abs(Z1);
[r1, pol1, res1, zer1, zj1, fj1, wj1] = aaa(F1, Z1, 'mmax', 33, 'cleanup', 'off');
ref.aaa_abs_Z        = Z1;
ref.aaa_abs_mmax     = 33;
ref.aaa_abs_vals     = r1(Z1);   % rational approx values
ref.aaa_abs_poles    = pol1;
ref.aaa_abs_zeros    = zer1;
ref.aaa_abs_zj       = zj1;      % support points
ref.aaa_abs_fj       = fj1;      % support values
ref.aaa_abs_wj       = wj1;      % barycentric weights
ref.aaa_abs_err      = norm(F1 - r1(Z1), inf);
fprintf('|x|: err=%.2e, m=%d, n_poles=%d\n', ref.aaa_abs_err, length(zj1), length(pol1));

%% ---- Test 2: exp(x) on [-1, 1], default tol ----
Z2 = linspace(-1, 1, 1000).';
F2 = exp(Z2);
[r2, pol2, res2, zer2, zj2, fj2, wj2] = aaa(F2, Z2, 'cleanup', 'off');
ref.aaa_exp_Z        = Z2;
ref.aaa_exp_vals     = r2(Z2);
ref.aaa_exp_poles    = pol2;
ref.aaa_exp_zeros    = zer2;
ref.aaa_exp_zj       = zj2;
ref.aaa_exp_fj       = fj2;
ref.aaa_exp_wj       = wj2;
ref.aaa_exp_err      = norm(F2 - r2(Z2), inf);
fprintf('exp(x): err=%.2e, m=%d\n', ref.aaa_exp_err, length(zj2));

%% ---- Test 3: Runge function 1/(1+25x^2) on [-1, 1] ----
Z3 = linspace(-1, 1, 1000).';
F3 = 1./(1 + 25*Z3.^2);
[r3, pol3, res3, zer3, zj3, fj3, wj3] = aaa(F3, Z3, 'cleanup', 'off');
ref.aaa_runge_Z      = Z3;
ref.aaa_runge_vals   = r3(Z3);
ref.aaa_runge_poles  = pol3;
ref.aaa_runge_zeros  = zer3;
ref.aaa_runge_zj     = zj3;
ref.aaa_runge_fj     = fj3;
ref.aaa_runge_wj     = wj3;
ref.aaa_runge_err    = norm(F3 - r3(Z3), inf);
fprintf('Runge: err=%.2e, m=%d\n', ref.aaa_runge_err, length(zj3));

%% ---- Test 4: 1/(1+z^2) on complex unit circle (complex data) ----
Z4 = exp(1i*pi*linspace(-1, 1, 500).');
F4 = 1./(1 + Z4.^2);
[r4, pol4, res4, zer4, zj4, fj4, wj4] = aaa(F4, Z4, 'cleanup', 'off');
ref.aaa_complex_Z    = Z4;
ref.aaa_complex_F    = F4;
ref.aaa_complex_vals = r4(Z4);
ref.aaa_complex_zj   = zj4;
ref.aaa_complex_wj   = wj4;
ref.aaa_complex_err  = norm(F4 - r4(Z4), inf);
fprintf('complex: err=%.2e, m=%d\n', ref.aaa_complex_err, length(zj4));

%% ---- Test 5: Known rational function (z-2)/(z+3) on [-1,1] ----
% A rational function should be reproduced exactly (up to machine precision)
Z5 = linspace(-1, 1, 500).';
F5 = (Z5 - 2)./(Z5 + 3);
[r5, pol5, res5, zer5, zj5, fj5, wj5] = aaa(F5, Z5, 'cleanup', 'off');
ref.aaa_rational_Z   = Z5;
ref.aaa_rational_F   = F5;
ref.aaa_rational_vals = r5(Z5);
ref.aaa_rational_zj  = zj5;
ref.aaa_rational_wj  = wj5;
ref.aaa_rational_err = norm(F5 - r5(Z5), inf);
fprintf('(z-2)/(z+3): err=%.2e, m=%d\n', ref.aaa_rational_err, length(zj5));

%% ---- Test 6: tan(x) on [-1, 1] (has poles nearby in complex plane) ----
Z6 = linspace(-1, 1, 1000).';
F6 = tan(Z6);
[r6, pol6, res6, zer6, zj6, fj6, wj6] = aaa(F6, Z6, 'cleanup', 'off');
ref.aaa_tan_Z        = Z6;
ref.aaa_tan_vals     = r6(Z6);
ref.aaa_tan_poles    = pol6;
ref.aaa_tan_zeros    = zer6;
ref.aaa_tan_zj       = zj6;
ref.aaa_tan_wj       = wj6;
ref.aaa_tan_err      = norm(F6 - r6(Z6), inf);
fprintf('tan(x): err=%.2e, m=%d\n', ref.aaa_tan_err, length(zj6));

%% ---- Save ----
save(fullfile(outdir, 'aaa.mat'), '-struct', 'ref');
fprintf('aaa.mat: %d fields\n', numel(fieldnames(ref)));
