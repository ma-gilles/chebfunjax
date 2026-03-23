% transforms.m — Generate MATLAB references for utils/transforms module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/transforms.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

rng(42);  % Fixed seed for reproducibility
ref = struct();

%% --- cheb2leg / leg2cheb ---
for n = [8, 16, 32, 64]
    c = randn(n, 1);
    ref.(sprintf('cheb2leg_n%d_in', n)) = c;
    ref.(sprintf('cheb2leg_n%d_out', n)) = cheb2leg(c);
    ref.(sprintf('leg2cheb_n%d_in', n)) = c;
    ref.(sprintf('leg2cheb_n%d_out', n)) = leg2cheb(c);
end

%% --- cheb2jac / jac2cheb ---
for n = [8, 16, 32]
    c = randn(n, 1);
    for params = {[0.5, 0.5], [1.0, 0.5], [2.0, 1.5], [-0.5, -0.5]}
        ab = params{1};
        a = ab(1); b = ab(2);
        tag = sprintf('a%s_b%s', strrep(num2str(a),'.','p'), strrep(num2str(b),'.','p'));
        ref.(sprintf('cheb2jac_n%d_%s_in', n, tag)) = c;
        ref.(sprintf('cheb2jac_n%d_%s_out', n, tag)) = cheb2jac(c, a, b);
        ref.(sprintf('jac2cheb_n%d_%s_in', n, tag)) = c;
        ref.(sprintf('jac2cheb_n%d_%s_out', n, tag)) = jac2cheb(c, a, b);
    end
end

%% --- vals2coeffs / coeffs2vals (2nd kind) ---
for n = [5, 10, 20]
    c = randn(n, 1);
    v = chebtech2.coeffs2vals(c);
    ref.(sprintf('coeffs2vals_n%d_in', n)) = c;
    ref.(sprintf('coeffs2vals_n%d_out', n)) = v;
    ref.(sprintf('vals2coeffs_n%d_in', n)) = v;
    ref.(sprintf('vals2coeffs_n%d_out', n)) = chebtech2.vals2coeffs(v);
end

save(fullfile(outdir, 'transforms.mat'), '-struct', 'ref');
fprintf('transforms.mat: %d fields\n', numel(fieldnames(ref)));
