% minimax_refs.m — Generate MATLAB references for minimax module.
outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% |x| degree 10
f = chebfun(@abs);
p = minimax(f, 10);
ref.abs_deg10_err = norm(f - p, inf);
ref.abs_deg10_coeffs = chebcoeffs(p, 11);
% Equioscillation points: roots of (f - p) ± err
err_fn = f - p;
xk = roots(err_fn - ref.abs_deg10_err);
xk = [xk; roots(err_fn + ref.abs_deg10_err)];
ref.abs_deg10_xk = sort(xk);

% sin(x) degree 6
f = chebfun(@sin);
p = minimax(f, 6);
ref.sin_deg6_err = norm(f - p, inf);

save(fullfile(outdir, 'minimax.mat'), '-struct', 'ref');
fprintf('minimax.mat: %d fields\n', numel(fieldnames(ref)));
