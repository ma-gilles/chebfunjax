% chebtech.m — Generate golden reference data for chebfunjax Chebtech2 tests.
%
% Usage (from repository root):
%   source project.conf
%   module load $MATLAB_MODULE
%   matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/refs/chebtech.m')"
%
% Output: tests/references/chebtech.mat

%% Setup
refs_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(refs_dir, 'dir')
    mkdir(refs_dir);
end

%% Test points (fixed set for cross-validation)
test_pts = [-0.9; -0.7; -0.5; -0.3; -0.1; 0.0; 0.1; 0.3; 0.5; 0.7; 0.9];

%% sin(x) — the canonical test function
f_sin = chebtech2(@sin);
sin_coeffs = f_sin.coeffs;
sin_n = length(f_sin);
sin_vals = feval(f_sin, test_pts);

% Prolong sin to 30 points
f_sin_prolonged = prolong(f_sin, 30);
sin_prolonged_30 = f_sin_prolonged.coeffs;

%% exp(x)
f_exp = chebtech2(@exp);
exp_coeffs = f_exp.coeffs;
exp_n = length(f_exp);
exp_vals = feval(f_exp, test_pts);

%% cos(x)
f_cos = chebtech2(@cos);
cos_coeffs = f_cos.coeffs;
cos_n = length(f_cos);
cos_vals = feval(f_cos, test_pts);

%% Runge function 1/(1+25x^2)
f_runge = chebtech2(@(x) 1./(1+25*x.^2));
runge_coeffs = f_runge.coeffs;
runge_n = length(f_runge);
runge_vals = feval(f_runge, test_pts);

%% vals2coeffs / coeffs2vals round-trip
test_values = sin(chebtech2.chebpts(10));
test_coeffs_from_vals = chebtech2.vals2coeffs(test_values);
test_vals_from_coeffs = chebtech2.coeffs2vals(test_coeffs_from_vals);

%% Clenshaw evaluation of known coefficients
% c = [1, 0.5, -0.25, 0.1, 0]  =>  T_0 + 0.5*T_1 - 0.25*T_2 + 0.1*T_3
known_coeffs = [1; 0.5; -0.25; 0.1; 0];
f_known = chebtech2({[], known_coeffs});
known_vals = feval(f_known, test_pts);

%% Save
outfile = fullfile(refs_dir, 'chebtech.mat');
save(outfile, ...
    'test_pts', ...
    'sin_coeffs', 'sin_n', 'sin_vals', 'sin_prolonged_30', ...
    'exp_coeffs', 'exp_n', 'exp_vals', ...
    'cos_coeffs', 'cos_n', 'cos_vals', ...
    'runge_coeffs', 'runge_n', 'runge_vals', ...
    'test_values', 'test_coeffs_from_vals', 'test_vals_from_coeffs', ...
    'known_coeffs', 'known_vals', ...
    '-v7');

fprintf('Saved golden refs to %s\n', outfile);
fprintf('sin:   n=%d\n', sin_n);
fprintf('exp:   n=%d\n', exp_n);
fprintf('cos:   n=%d\n', cos_n);
fprintf('runge: n=%d\n', runge_n);
