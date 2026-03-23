% run_chebfun_tests.m — Run Chebfun's own test suite and capture results.
% This tells us what the "correct" behavior is for every function.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/run_chebfun_tests.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% Run a subset of tests and capture pass/fail
test_dirs = {'chebtech', 'chebfun', 'misc'};
results = struct();

for i = 1:numel(test_dirs)
    tdir = test_dirs{i};
    fprintf('\n=== Testing %s ===\n', tdir);
    test_path = fullfile('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref', 'tests', tdir);
    files = dir(fullfile(test_path, 'test_*.m'));

    for j = 1:numel(files)
        tname = files(j).name(1:end-2);  % strip .m
        try
            olddir = cd(test_path);
            pass = feval(tname);
            cd(olddir);
            n_pass = sum(pass);
            n_total = numel(pass);
            results.(tname) = struct('pass', n_pass, 'total', n_total, 'all_pass', n_pass == n_total);
            fprintf('  %s: %d/%d\n', tname, n_pass, n_total);
        catch ME
            cd(olddir);
            results.(tname) = struct('pass', 0, 'total', 0, 'all_pass', false, 'error', ME.message);
            fprintf('  %s: ERROR - %s\n', tname, ME.message);
        end
    end
end

save(fullfile(outdir, 'matlab_test_results.mat'), 'results');
fprintf('\nResults saved to %s/matlab_test_results.mat\n', outdir);
