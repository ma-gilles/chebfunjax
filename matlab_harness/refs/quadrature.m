% quadrature.m — Generate MATLAB references for utils/quadrature module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/quadrature.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
for n = [5, 10, 17, 32, 64, 128]
    ref.(sprintf('chebpts2_n%d', n)) = chebpts(n);
    ref.(sprintf('chebpts1_n%d', n)) = chebpts(n, 1);
    [x, w] = legpts(n);
    ref.(sprintf('legpts_x_n%d', n)) = x;
    ref.(sprintf('legpts_w_n%d', n)) = w;
end
save(fullfile(outdir, 'quadrature.mat'), '-struct', 'ref');
fprintf('quadrature.mat: %d fields\n', numel(fieldnames(ref)));
