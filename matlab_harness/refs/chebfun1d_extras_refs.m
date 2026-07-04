% chebfun1d_extras_refs.m — MATLAB golden references for the higher-level,
% user-facing chebfun1d functions that were NOT already pinned by
% chebfun_refs.m (which only covers sin/exp construction + eval).
%
% Pins, at rtol 1e-12 unless noted, MATLAB Chebfun ground truth for:
%   conv, polyfit, interp1, spline, pchip,
%   besselj, bessely, airy, erf, erfc,
%   cumsum / diff chains, sum, norm,
%   quasimatrix svd (singular values) and qr (R factor),
%   and an ODE IVP reference (exact solution of u'=-u, u(0)=1 -> exp(-t)).
%
% The ODE reference is the *analytic* solution: chebfunjax ode45/ode113 wrap
% scipy solve_ivp with adaptive step control (rtol 1e-6), so its output is an
% approximation whose accuracy — not a 1e-12 identity — is what the test
% checks (Gate-3 implementation-dependent).
%
% Usage:
%   /usr/licensed/matlab-R2025b/bin/matlab -batch \
%     "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); \
%      run('/home/mg6942/chebfunjax/matlab_harness/refs/chebfun1d_extras_refs.m')"
%
% Output: tests/references/chebfun1d_extras.mat

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% Shared evaluation points inside [-1,1] (not on any Chebyshev grid).
pts = [-0.9; -0.5; -0.1; 0.2; 0.6; 0.85];
ref.pts = pts;

x = chebfun('x');   % identity on [-1,1]

%% --- Special functions (composition then eval) -----------------------------
% besselj(nu, g) with g = 0.5*x + 0.3   (arg in [-0.2, 0.8])
g_bj = chebfun(@(x) 0.5*x + 0.3);
ref.besselj0 = feval(besselj(0, g_bj), pts);
ref.besselj1 = feval(besselj(1, g_bj), pts);

% bessely(1, g) with g = 0.5*x + 1.0   (strictly positive arg in [0.5, 1.5])
g_by = chebfun(@(x) 0.5*x + 1.0);
ref.bessely1 = feval(bessely(1, g_by), pts);

% Airy Ai (K=0) and Bi (K=2) of x
ref.airy_Ai = feval(airy(0, x), pts);
ref.airy_Bi = feval(airy(2, x), pts);

% erf / erfc of 2x
g_erf = chebfun(@(x) 2*x);
ref.erf  = feval(erf(g_erf),  pts);
ref.erfc = feval(erfc(g_erf), pts);

%% --- polyfit (least-squares degree-5 fit of exp) ---------------------------
% MATLAB polyfit truncates the LEGENDRE series (true L2 least-squares).
fexp = chebfun(@exp);
ref.polyfit_exp5 = feval(polyfit(fexp, 5), pts);

%% --- interp1 (default 'poly' = barycentric polynomial interpolant) ---------
xi = [-1.0; -0.6; -0.2; 0.3; 0.7; 1.0];
yi = [ 0.0;  0.5; -0.3; 0.9; 0.2; -0.5];
ref.interp1_xi = xi;
ref.interp1_yi = yi;
ref.interp1    = feval(chebfun.interp1(xi, yi, 'poly'), pts);

%% --- spline / pchip (piecewise cubic interpolants) -------------------------
xs = [-1.0; -0.5; 0.0; 0.4; 0.8; 1.0];
ys = [ 1.0;  0.2; -0.4; 0.6; 0.1; -0.7];
ref.spline_xs = xs;
ref.spline_ys = ys;
ref.spline = feval(chebfun.spline(xs, ys), pts);
ref.pchip  = feval(chebfun.pchip(xs, ys),  pts);

%% --- conv (convolution of two chebfuns) ------------------------------------
% f = sin, g = exp on [-1,1]; h = conv(f,g) lives on [-2,2].
f_c = chebfun(@sin);
g_c = chebfun(@exp);
h_c = conv(f_c, g_c);
cpts = [-1.5; -0.5; 0.0; 0.7; 1.4];
ref.conv_dom  = [h_c.domain(1); h_c.domain(end)];
ref.conv_pts  = cpts;
ref.conv_eval = feval(h_c, cpts);

%% --- cumsum / diff chains, sum, norm ---------------------------------------
fc = chebfun(@(x) cos(3*x));
ref.cumsum = feval(cumsum(fc), pts);   % antiderivative, F(-1) = 0
ref.diff1  = feval(diff(fc),   pts);   % -3 sin(3x)
ref.diff2  = feval(diff(fc, 2), pts);  % -9 cos(3x)
ref.sum    = sum(fc);                  % scalar integral over [-1,1]
ref.norm2  = norm(fc);                 % L2 norm

%% --- quasimatrix qr / svd of A = [1, x, x^2, x^3] on [-1,1] ----------------
A = [1 + 0*x, x, x.^2, x.^3];
ref.svd_S = svd(A);                    % 4x1 singular values (non-increasing)
[~, Rqr] = qr(A);
ref.qr_R    = Rqr;                     % 4x4 upper-triangular R
ref.qr_Rdiag = diag(Rqr);

%% --- ODE IVP reference: analytic exp(-t) -----------------------------------
tt = [0.0; 0.3; 0.8; 1.4; 2.0];
ref.ode_t     = tt;
ref.ode_exact = exp(-tt);

%% --- Save ------------------------------------------------------------------
save(fullfile(outdir, 'chebfun1d_extras.mat'), '-struct', 'ref', '-v7');
fprintf('chebfun1d_extras.mat: %d fields\n', numel(fieldnames(ref)));
