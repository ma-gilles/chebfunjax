% minimax_refs.m — Generate MATLAB references for minimax module.
outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% Best polynomial approximation of |x| on [-1,1]
for n = [2, 4, 6, 10, 20]
    f = chebfun(@abs);
    p = minimax(f, n);
    ref.(sprintf('minimax_abs_n%d_err', n)) = norm(f - p, inf);
    ref.(sprintf('minimax_abs_n%d_coeffs', n)) = chebcoeffs(p, n+1);
end

% Best approximation of exp(x)
for n = [3, 5, 10]
    f = chebfun(@exp);
    p = minimax(f, n);
    ref.(sprintf('minimax_exp_n%d_err', n)) = norm(f - p, inf);
end

save(fullfile(outdir, 'minimax.mat'), '-struct', 'ref');
fprintf('minimax.mat: %d fields\n', numel(fieldnames(ref)));
