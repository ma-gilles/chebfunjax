% transforms.m — Generate MATLAB references for utils/transforms module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/transforms.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

rng(42);  % Fixed seed for reproducibility
ref = struct();
for n = [8, 16, 32, 64]
    c = randn(n, 1);
    ref.(sprintf('cheb2leg_n%d_in', n)) = c;
    ref.(sprintf('cheb2leg_n%d_out', n)) = cheb2leg(c);
    ref.(sprintf('leg2cheb_n%d_in', n)) = c;
    ref.(sprintf('leg2cheb_n%d_out', n)) = leg2cheb(c);
end
save(fullfile(outdir, 'transforms.mat'), '-struct', 'ref');
fprintf('transforms.mat: %d fields\n', numel(fieldnames(ref)));
