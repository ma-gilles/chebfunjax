% generate_refs.m — Run ALL per-module reference generators.
%
% Each module has its own script in matlab_harness/refs/<module>.m.
% Agents add new scripts there — no need to edit this shared runner.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"

refs_dir = fullfile(fileparts(mfilename('fullpath')), 'refs');
scripts = dir(fullfile(refs_dir, '*.m'));

fprintf('Generating MATLAB references from %d module scripts...\n', numel(scripts));
for i = 1:numel(scripts)
    fprintf('\n=== %s ===\n', scripts(i).name);
    run(fullfile(refs_dir, scripts(i).name));
end
fprintf('\nDone. All references generated.\n');
