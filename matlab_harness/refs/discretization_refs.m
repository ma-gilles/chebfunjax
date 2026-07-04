% discretization_refs.m — MATLAB references for the spectral discretization
% operators that were UNCOVERED by chebcolloc_refs.m / diffmat_refs.m:
%   * @ultraS  : diffmat, convertmat, multmat  (ultraspherical spectral method)
%   * @trigcolloc : diffmat (Fourier), trigpts, trapezoidal weights
%
% Pins the raw dense operator matrices ENTRYWISE at rtol 1e-12 against MATLAB
% Chebfun (commit 7574c77).  These are the banded coefficient-space operators
% (Olver & Townsend, SIAM Rev. 2013) and the Fourier pseudospectral matrices
% (Trefethen, Spectral Methods in MATLAB) that form the backbone of the BVP /
% ODE solvers, so any entrywise mismatch is a genuine bug, not a tolerance
% issue.
%
% NOTE (API gap, intentionally NOT pinned): @trigcolloc/cumsummat.m raises
%   'CHEBFUN:TRIGCOLLOC:cumsummat:notSupported' — MATLAB Chebfun does not
%   support indefinite integration on the trig collocation discretization, so
%   there is no golden reference for chebfunjax's trig_cumsummat / cumsummat.
%
% All ultraS.* and trigcolloc.* below are STATIC methods (verified in the
% classdefs), callable without constructing an instance.  Sparse outputs are
% densified with full() before saving.
%
% Usage (from repo root):
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/discretization_refs.m')"

outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% =====================================================================
% ultraS: differentiation matrices  (Chebyshev T coeffs -> C^{(m)} coeffs)
%   ultraS.diffmat(n, m)  is an n-by-n banded upper-triangular matrix.
% =====================================================================
for n = [8, 12]
    for m = [1, 2, 3]
        ref.(sprintf('us_diff_n%d_m%d', n, m)) = full(ultraS.diffmat(n, m));
    end
end

% =====================================================================
% ultraS: conversion matrices  C^{(K1)} -> C^{(K2+1)}
%   ultraS.convertmat(n, K1, K2)
%   (0,0): T -> C^{(1)}=U ; (0,1): T -> C^{(2)} ; (1,1): C^{(1)} -> C^{(2)}
%   (0,2): T -> C^{(3)}   ; (1,2): C^{(1)} -> C^{(3)} ; (2,2): C^{(2)} -> C^{(3)}
% =====================================================================
conv_pairs = [0 0; 0 1; 1 1; 0 2; 1 2; 2 2];
for n = [8, 12]
    for r = 1:size(conv_pairs, 1)
        k1 = conv_pairs(r, 1);
        k2 = conv_pairs(r, 2);
        ref.(sprintf('us_conv_n%d_%d_%d', n, k1, k2)) = ...
            full(ultraS.convertmat(n, k1, k2));
    end
end

% =====================================================================
% ultraS: multiplication matrices in the C^{(lambda)} basis
%   ultraS.multmat(n, f, lambda), f = Chebyshev-T coefficients of multiplier.
%   The coefficient vector has several nonzero higher-order entries so that
%   the lambda>=2 three-term recurrence branch is genuinely exercised.
% =====================================================================
mult_coeffs = [1.0; 0.5; 0.25; 0.1];    % Cheb-T coeffs (column vector)
ref.us_mult_coeffs = mult_coeffs;
for n = [8, 12]
    for lam = [0, 1, 2, 3]
        ref.(sprintf('us_mult_n%d_lam%d', n, lam)) = ...
            full(ultraS.multmat(n, mult_coeffs, lam));
    end
end

% =====================================================================
% trigcolloc: Fourier differentiation matrices (period 2, interval [-1,1))
%   trigcolloc.diffmat(N, m).  Even and odd N use different analytic
%   formulae (m=1..4) and a FFT eigenvalue branch (m>=5), so both parities
%   and all branches are pinned.  Final scaling is pi^m (built in).
% =====================================================================
for N = [7, 8, 9, 16]
    for m = [1, 2, 3, 4, 5]
        ref.(sprintf('tc_diff_n%d_m%d', N, m)) = full(trigcolloc.diffmat(N, m));
    end
end

% =====================================================================
% trigcolloc: equidistant points and trapezoidal quadrature weights.
%   trigpts(N)          -> N equispaced points on [-1,1)  (left-closed)
%   trigpts(N, [a b])   -> mapped to [a,b)
%   trigtech.quadwts(N) -> trapezoidal (periodic) weights, each = 2/N on [-1,1)
% =====================================================================
for N = [7, 8, 16]
    ref.(sprintf('tc_pts_n%d', N)) = trigpts(N);
end
ref.tc_pts_n8_dom02 = trigpts(8, [0 2]);
for N = [8, 16]
    ref.(sprintf('tc_wts_n%d', N)) = trigtech.quadwts(N);
end

% =====================================================================
% Save
% =====================================================================
save(fullfile(outdir, 'discretization.mat'), '-struct', 'ref');
fprintf('Saved discretization.mat with %d fields.\n', numel(fieldnames(ref)));
