% chebfun_refs.m — Generate golden reference data for chebfunjax Chebfun tests.
%
% Usage (from repository root):
%   source project.conf
%   module load $MATLAB_MODULE
%   matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/refs/chebfun_refs.m')"
%
% Output: tests/references/chebfun.mat

%% Setup
refs_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(refs_dir, 'dir')
    mkdir(refs_dir);
end

%% Evaluation test points (fixed, not on any Chebyshev grid)
eval_pts = [-0.9; -0.7; -0.5; -0.3; -0.1; 0.0; 0.1; 0.3; 0.5; 0.7; 0.9];

%% sin(x) on [-1, 1] — canonical test
f_sin = chebfun(@sin);
sin_n = length(f_sin);
sin_coeffs = chebcoeffs(f_sin, sin_n);
sin_eval_vals = feval(f_sin, eval_pts);

%% exp(x) on [-1, 1]
f_exp = chebfun(@exp);
exp_n = length(f_exp);

%% sin(x) on [0, pi]
f_sin_pi = chebfun(@sin, [0, pi]);
sin_pi_n = length(f_sin_pi);
sin_pi_eval = feval(f_sin_pi, eval_pts * pi / 2 + pi / 2);  % map to [0, pi]

%% Save
outfile = fullfile(refs_dir, 'chebfun.mat');
save(outfile, ...
    'eval_pts', ...
    'sin_n', 'sin_coeffs', 'sin_eval_vals', ...
    'exp_n', ...
    'sin_pi_n', 'sin_pi_eval', ...
    '-v7');

fprintf('Saved golden refs to %s\n', outfile);
fprintf('sin on [-1,1]:  n=%d\n', sin_n);
fprintf('exp on [-1,1]:  n=%d\n', exp_n);
fprintf('sin on [0,pi]:  n=%d\n', sin_pi_n);
