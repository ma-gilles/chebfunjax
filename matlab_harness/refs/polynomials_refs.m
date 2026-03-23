% polynomials_refs.m — Generate MATLAB references for utils/polynomials module.
% Usage: matlab -batch "addpath('CHEBFUN_PATH'); run('matlab_harness/refs/polynomials_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

% -------------------------------------------------------------------
% Chebyshev T_n coefficients (kind=1)
% chebpoly(n) returns a chebfun; extract its Chebyshev coefficients.
% -------------------------------------------------------------------
for n = [0, 1, 2, 5, 10, 20]
    f = chebpoly(n);
    c = chebcoeffs(f);
    ref.(sprintf('chebpoly_T%d', n)) = c(:)';
end

% -------------------------------------------------------------------
% Chebyshev U_n coefficients (kind=2)
% chebpoly(n, 2) returns U_n as a chebfun.
% -------------------------------------------------------------------
for n = [0, 1, 2, 5, 10]
    f = chebpoly(n, 2);
    c = chebcoeffs(f);
    ref.(sprintf('chebpoly_U%d', n)) = c(:)';
end

% -------------------------------------------------------------------
% Legendre P_n Chebyshev coefficients
% -------------------------------------------------------------------
for n = [0, 1, 2, 5, 10, 20]
    f = legpoly(n);
    c = chebcoeffs(f);
    ref.(sprintf('legpoly_P%d', n)) = c(:)';
end

% -------------------------------------------------------------------
% Jacobi P_n^(a,b) Chebyshev coefficients
% -------------------------------------------------------------------
params = {0.5, 0.5, '0p5', '0p5'; ...
          1.0, 0.5, '1p0', '0p5'; ...
          2.0, 1.5, '2p0', '1p5'};
for n = [0, 1, 2, 5, 10]
    for p = 1:size(params, 1)
        a = params{p, 1};
        b = params{p, 2};
        a_str = params{p, 3};
        b_str = params{p, 4};
        f = jacpoly(n, a, b);
        c = chebcoeffs(f);
        ref.(sprintf('jacpoly_n%d_a%s_b%s', n, a_str, b_str)) = c(:)';
    end
end

% -------------------------------------------------------------------
% Ultraspherical C_n^(lam) Chebyshev coefficients
% -------------------------------------------------------------------
lams = {1.5, '1p5'; 2.0, '2p0'};
for n = [0, 1, 2, 5, 10]
    for l = 1:size(lams, 1)
        lam = lams{l, 1};
        lam_str = lams{l, 2};
        f = ultrapoly(n, lam);
        c = chebcoeffs(f);
        ref.(sprintf('ultrapoly_n%d_lam%s', n, lam_str)) = c(:)';
    end
end

% -------------------------------------------------------------------
% Hermite H_n(x) values (physicist's type) at sample points
% -------------------------------------------------------------------
x_herm = linspace(-3, 3, 50);
ref.hermeval_x = x_herm;
for n = [0, 1, 2, 3, 5]
    f = hermpoly(n);
    ref.(sprintf('hermeval_H%d', n)) = feval(f, x_herm);
end

% -------------------------------------------------------------------
% Laguerre L_n(x) values at sample points
% -------------------------------------------------------------------
x_lag = linspace(0, 10, 50);
ref.lageval_x = x_lag;
for n = [0, 1, 2, 3, 5]
    f = lagpoly(n);
    ref.(sprintf('lageval_L%d', n)) = feval(f, x_lag);
end

save(fullfile(outdir, 'polynomials.mat'), '-struct', 'ref');
fprintf('polynomials.mat: %d fields\n', numel(fieldnames(ref)));
