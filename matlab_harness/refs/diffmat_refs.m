% diffmat_refs.m — Generate MATLAB references for utils/diffmat module.
%
% Functions tested: diffmat, cumsummat, intmat, introw, diffrow.
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/diffmat_refs.m')"

outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% --- diffmat: 2nd-kind Chebyshev (default) ---
for n = [5, 10, 17, 20, 32, 64]
    ref.(sprintf('diffmat_n%d', n)) = diffmat(n);
end

%% --- diffmat: higher-order derivatives ---
for n = [10, 20]
    for p = [2, 3, 4]
        ref.(sprintf('diffmat_n%d_p%d', n, p)) = diffmat(n, p);
    end
end

%% --- diffmat: 1st-kind Chebyshev ---
for n = [5, 10, 17, 20]
    ref.(sprintf('diffmat_n%d_kind1', n)) = diffmat(n, 1, [-1 1], 'chebkind1');
end

%% --- diffmat: higher-order, 1st-kind ---
for n = [10, 20]
    ref.(sprintf('diffmat_n%d_p2_kind1', n)) = diffmat(n, 2, [-1 1], 'chebkind1');
end

%% --- diffmat: domain scaling ---
ref.diffmat_n10_dom02 = diffmat(10, 1, [0 2]);
ref.diffmat_n10_p2_dom02 = diffmat(10, 2, [0 2]);

%% --- cumsummat ---
for n = [5, 10, 17, 20]
    ref.(sprintf('cumsummat_n%d', n)) = cumsummat(n);
end

%% --- cumsummat with domain ---
ref.cumsummat_n10_dom02 = cumsummat(10, [0 2]);

%% --- introw (Clenshaw-Curtis weights) ---
for n = [5, 10, 17, 20, 32]
    [~, w] = chebpts(n);
    ref.(sprintf('introw_n%d', n)) = w;
end

%% --- introw with domain ---
[~, w_dom] = chebpts(10, [0, 2]);
ref.introw_n10_dom02 = w_dom;

%% --- diffrow ---
for n = [5, 10, 20]
    ref.(sprintf('diffrow_n%d_p1_left', n)) = diffrow(n, 1, -1);
    ref.(sprintf('diffrow_n%d_p1_right', n)) = diffrow(n, 1, 1);
    ref.(sprintf('diffrow_n%d_p2_left', n)) = diffrow(n, 2, -1);
    ref.(sprintf('diffrow_n%d_p2_right', n)) = diffrow(n, 2, 1);
end

%% --- Save ---
save(fullfile(outdir, 'diffmat.mat'), '-struct', 'ref');
fprintf('Saved diffmat.mat with %d fields.\n', numel(fieldnames(ref)));
