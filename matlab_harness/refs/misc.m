%% misc.m — Generate MATLAB reference data for utils/misc.py tests.
%
% Functions tested: standardChop, gridsample, abstractQR.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/misc.m')"

rng(42);  % Reproducibility
outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% --- standardChop ---

% Test 1: Simple geometric decay  10^{-k}
coeffs1 = 10.^(-(1:50));
ref.sc_geom_coeffs = coeffs1;
ref.sc_geom_cutoff = standardChop(coeffs1);

% Test 2: Geometric decay + eps noise
random2 = cos((1:50).^2);
coeffs2 = coeffs1 + 1e-16 * random2;
ref.sc_geom_eps_coeffs = coeffs2;
ref.sc_geom_eps_cutoff = standardChop(coeffs2);

% Test 3: Geometric decay + 1e-13 noise
coeffs3 = coeffs1 + 1e-13 * random2;
ref.sc_geom_13_coeffs = coeffs3;
ref.sc_geom_13_cutoff = standardChop(coeffs3);

% Test 4: Geometric decay + 1e-10 noise (not happy at default tol)
coeffs4 = coeffs1 + 1e-10 * random2;
ref.sc_geom_10_coeffs = coeffs4;
ref.sc_geom_10_cutoff = standardChop(coeffs4);

% Test 5: Same as 4 but with tol=1e-10
ref.sc_geom_10_tol10_cutoff = standardChop(coeffs4, 1e-10);

% Test 6: All-zero coefficients
coeffs6 = zeros(1, 50);
ref.sc_zero_cutoff = standardChop(coeffs6);

% Test 7: Short vector (< 17)
coeffs7 = 10.^(-(1:10));
ref.sc_short_coeffs = coeffs7;
ref.sc_short_cutoff = standardChop(coeffs7);

% Test 8: Chebfun of sin(x) — extract coefficients and chop
f8 = chebfun(@sin);
c8 = chebcoeffs(f8);
ref.sc_sin_coeffs = c8;
ref.sc_sin_cutoff = standardChop(c8);

% Test 9: Chebfun of exp(x)
f9 = chebfun(@exp);
c9 = chebcoeffs(f9);
ref.sc_exp_coeffs = c9;
ref.sc_exp_cutoff = standardChop(c9);

% Test 10: Chebfun of exp(-100*x^2) — sharp function, many coefficients
f10 = chebfun(@(x) exp(-100*x.^2));
c10 = chebcoeffs(f10);
ref.sc_sharp_coeffs = c10;
ref.sc_sharp_cutoff = standardChop(c10);

% Test 11: Various tolerances on sin coefficients
c11 = chebcoeffs(chebfun(@sin, 'eps', 1e-8));
ref.sc_sin_tol8_coeffs = c11;
ref.sc_sin_tol8_cutoff = standardChop(c11, 1e-8);

% Test 12: Constant function
f12 = chebfun(@(x) 3.0*ones(size(x)));
c12 = chebcoeffs(f12);
ref.sc_const_coeffs = c12;
ref.sc_const_cutoff = standardChop(c12);

%% --- gridsample ---

% Gridsample sin on 5 chebpts
ref.gs_sin5 = gridsample(@sin, 5);

% Gridsample exp on 10 chebpts
ref.gs_exp10 = gridsample(@exp, 10);

% Gridsample on custom domain [0, pi]
ref.gs_sin10_0pi = gridsample(@sin, 10, [0, pi]);

%% --- abstractQR ---

% Small 5x3 matrix with standard inner product
A1 = randn(5, 3);
E1 = eye(5, 3);
ref.aqr_A1 = A1;
ref.aqr_E1 = E1;
[Q1, R1] = abstractQR(A1, E1, @(u,v) u'*v);
ref.aqr_Q1 = Q1;
ref.aqr_R1 = R1;

% Verify Q'Q ≈ I and QR ≈ A
ref.aqr_QtQ1 = Q1' * Q1;
ref.aqr_QR1 = Q1 * R1;

% 4x2 matrix
A2 = randn(4, 2);
E2 = eye(4, 2);
ref.aqr_A2 = A2;
ref.aqr_E2 = E2;
[Q2, R2] = abstractQR(A2, E2, @(u,v) u'*v);
ref.aqr_Q2 = Q2;
ref.aqr_R2 = R2;

%% --- Save ---
save(fullfile(outdir, 'misc.mat'), '-struct', 'ref');
fprintf('Saved misc.mat with %d fields.\n', numel(fieldnames(ref)));
