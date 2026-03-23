% chebtech_construct_refs.m — Generate golden reference data for U20b
% (compose, restrict, happinessCheck).
%
% Usage (from repository root):
%   source project.conf
%   module load $MATLAB_MODULE
%   matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/refs/chebtech_construct_refs.m')"
%
% Output: tests/references/chebtech_construct.mat

%% Setup
refs_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(refs_dir, 'dir')
    mkdir(refs_dir);
end

%% Test points (fixed set for cross-validation)
test_pts = [-0.9; -0.7; -0.5; -0.3; -0.1; 0.0; 0.1; 0.3; 0.5; 0.7; 0.9];

%% 1. compose: exp(sin(x))
f_sin = chebtech2(@sin);
f_exp_sin = compose(f_sin, @exp);
compose_exp_sin_coeffs = f_exp_sin.coeffs;
compose_exp_sin_n = length(f_exp_sin);
compose_exp_sin_vals = feval(f_exp_sin, test_pts);
compose_exp_sin_exact = exp(sin(test_pts));

%% 2. compose: cos(exp(x))
f_exp = chebtech2(@exp);
f_cos_exp = compose(f_exp, @cos);
compose_cos_exp_coeffs = f_cos_exp.coeffs;
compose_cos_exp_n = length(f_cos_exp);
compose_cos_exp_vals = feval(f_cos_exp, test_pts);
compose_cos_exp_exact = cos(exp(test_pts));

%% 3. compose: sin^2(x) via compose(sin, @(x) x.^2)
f_sin2 = compose(f_sin, @(x) x.^2);
compose_sin2_coeffs = f_sin2.coeffs;
compose_sin2_n = length(f_sin2);
compose_sin2_vals = feval(f_sin2, test_pts);
compose_sin2_exact = sin(test_pts).^2;

%% 4. compose: two chebtech objects (function composition)
% g = chebtech2 of x^2, f = sin, so compose(g, f) = sin(g(x)) = sin(x^2)
f_x2 = chebtech2(@(x) x.^2);
f_sin_x2 = compose(f_x2, f_sin);
compose_sin_x2_coeffs = f_sin_x2.coeffs;
compose_sin_x2_n = length(f_sin_x2);
compose_sin_x2_vals = feval(f_sin_x2, test_pts);
compose_sin_x2_exact = sin(test_pts.^2);

%% 5. restrict: sin from [-1,1] to [0, 0.5]
f_sin_restricted = restrict(f_sin, [0, 0.5]);
% restrict returns a cell array when given breakpoints; for 2 endpoints it
% returns a single chebtech.
if iscell(f_sin_restricted)
    f_sin_restricted = f_sin_restricted{1};
end
restrict_sin_0_05_coeffs = f_sin_restricted.coeffs;
restrict_sin_0_05_n = length(f_sin_restricted);
% Map test_pts from [-1,1] to [0, 0.5]: y = 0.25*x + 0.25
mapped_pts = 0.25 * test_pts + 0.25;
restrict_sin_0_05_vals = feval(f_sin_restricted, test_pts);
restrict_sin_0_05_exact = sin(mapped_pts);

%% 6. restrict: exp from [-1,1] to [-0.5, 0.5]
f_exp_restricted = restrict(f_exp, [-0.5, 0.5]);
if iscell(f_exp_restricted)
    f_exp_restricted = f_exp_restricted{1};
end
restrict_exp_m05_05_coeffs = f_exp_restricted.coeffs;
restrict_exp_m05_05_n = length(f_exp_restricted);
% Map test_pts from [-1,1] to [-0.5, 0.5]: y = 0.5*x
mapped_pts2 = 0.5 * test_pts;
restrict_exp_m05_05_vals = feval(f_exp_restricted, test_pts);
restrict_exp_m05_05_exact = exp(mapped_pts2);

%% 7. restrict: identity (full interval)
f_sin_full = restrict(f_sin, [-1, 1]);
if iscell(f_sin_full)
    f_sin_full = f_sin_full{1};
end
restrict_sin_full_coeffs = f_sin_full.coeffs;

%% 8. happinessCheck on sin with 33 points
n_test = 33;
x33 = chebtech2.chebpts(n_test);
v33 = sin(x33);
c33 = chebtech2.vals2coeffs(v33);
f_test = chebtech2({v33, c33});
[happy_sin33, cutoff_sin33] = happinessCheck(f_test, @sin, v33);

%% 9. happinessCheck on exp with 33 points
v33e = exp(x33);
c33e = chebtech2.vals2coeffs(v33e);
f_test_exp = chebtech2({v33e, c33e});
[happy_exp33, cutoff_exp33] = happinessCheck(f_test_exp, @exp, v33e);

%% Save
outfile = fullfile(refs_dir, 'chebtech_construct.mat');
save(outfile, ...
    'test_pts', ...
    'compose_exp_sin_coeffs', 'compose_exp_sin_n', 'compose_exp_sin_vals', 'compose_exp_sin_exact', ...
    'compose_cos_exp_coeffs', 'compose_cos_exp_n', 'compose_cos_exp_vals', 'compose_cos_exp_exact', ...
    'compose_sin2_coeffs', 'compose_sin2_n', 'compose_sin2_vals', 'compose_sin2_exact', ...
    'compose_sin_x2_coeffs', 'compose_sin_x2_n', 'compose_sin_x2_vals', 'compose_sin_x2_exact', ...
    'restrict_sin_0_05_coeffs', 'restrict_sin_0_05_n', 'restrict_sin_0_05_vals', 'restrict_sin_0_05_exact', ...
    'restrict_exp_m05_05_coeffs', 'restrict_exp_m05_05_n', 'restrict_exp_m05_05_vals', 'restrict_exp_m05_05_exact', ...
    'restrict_sin_full_coeffs', ...
    'happy_sin33', 'cutoff_sin33', ...
    'happy_exp33', 'cutoff_exp33', ...
    '-v7');

fprintf('Saved golden refs to %s\n', outfile);
fprintf('compose exp(sin): n=%d\n', compose_exp_sin_n);
fprintf('compose cos(exp): n=%d\n', compose_cos_exp_n);
fprintf('compose sin^2:    n=%d\n', compose_sin2_n);
fprintf('compose sin(x^2): n=%d\n', compose_sin_x2_n);
fprintf('restrict sin [0,0.5]:  n=%d\n', restrict_sin_0_05_n);
fprintf('restrict exp [-0.5,0.5]: n=%d\n', restrict_exp_m05_05_n);
fprintf('happiness sin@33: happy=%d, cutoff=%d\n', happy_sin33, cutoff_sin33);
fprintf('happiness exp@33: happy=%d, cutoff=%d\n', happy_exp33, cutoff_exp33);
