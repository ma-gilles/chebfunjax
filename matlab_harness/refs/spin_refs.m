% spin_refs.m — MATLAB references for the spin package (1D ETDRK4).
%
% Pins chebfunjax.spin.spin (ETDRK4 exponential time-stepping, Fourier
% spectral in space) against MATLAB Chebfun SPIN at rtol 1e-12 where the
% problem is well-posed, and documents genuine value discrepancies as bugs.
%
% Every case fixes N, dt and the initial condition EXACTLY so the two
% implementations discretize identically:
%   * grid              : trigpts(N, dom)  == np.linspace(a,b,N,endpoint=False)
%   * vals<->coeffs      : fft / ifft (Chebfun getVals2/Coeffs2Transform)
%   * dealiasing         : OFF  (Chebfun spinpref default; chebfunjax dealias=False)
%   * contour points M   : 64   (Chebfun spinpref default; chebfunjax M=64)
%   * scheme             : etdrk4 (Chebfun spinpref default)
% dt is chosen so tf = nsteps*dt is an exact integer number of steps.
%
% Cases:
%   LINEAR (N(u)=0), ETDRK4 exact up to exp(dt*L)  -> strict 1e-12:
%     lin_diff   : u_t = 0.05*u_xx           [-pi,pi], real, band-limited IC
%     lin_schrod : u_t = 1i*u_xx             [-pi,pi], complex (free particle)
%   NONLINEAR short, well-posed (NOT chaotic) -> parity target 1e-12:
%     allen_cahn : u_t = 5e-3*u_xx + u - u^3 [0,2pi],  real
%     nls        : u_t = 1i*u_xx + 1i|u|^2 u [-pi,pi], complex
%
% Usage:
%   module load matlab/R2025b
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/spin_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

% ================= CASE 1: linear diffusion (real) =================
dom = [-pi pi]; N = 32; dt = 0.05; tf = 1.0;   % 20 steps
[v, xx] = run_case(@(u) 0.05*diff(u,2), @(u) 0*u, dom, tf, N, dt, ...
                   @(x) sin(x) + 0.5*cos(2*x));
ref.lin_diff_N = N; ref.lin_diff_dt = dt; ref.lin_diff_tf = tf;
ref.lin_diff_dom = dom; ref.lin_diff_x = xx; ref.lin_diff_u = v;
fprintf('lin_diff done: nsteps=%d, max|u|=%.4g\n', round(tf/dt), max(abs(v)));

% ============ CASE 2: linear free Schrodinger (complex) ============
dom = [-pi pi]; N = 32; dt = 0.05; tf = 1.0;   % 20 steps
[v, xx] = run_case(@(u) 1i*diff(u,2), @(u) 0*u, dom, tf, N, dt, ...
                   @(x) 1 + 0.6*cos(x) + 0.3*sin(2*x));
ref.lin_schrod_N = N; ref.lin_schrod_dt = dt; ref.lin_schrod_tf = tf;
ref.lin_schrod_dom = dom; ref.lin_schrod_x = xx; ref.lin_schrod_u = v;
fprintf('lin_schrod done: nsteps=%d, max|u|=%.4g\n', round(tf/dt), max(abs(v)));

% ============= CASE 3: Allen-Cahn short (real nonlinear) ===========
dom = [0 2*pi]; N = 64; dt = 0.05; tf = 1.0;   % 20 steps
[v, xx] = run_case(@(u) 5e-3*diff(u,2), @(u) u - u.^3, dom, tf, N, dt, ...
                   @(x) 0.5*sin(x));
ref.allen_cahn_N = N; ref.allen_cahn_dt = dt; ref.allen_cahn_tf = tf;
ref.allen_cahn_dom = dom; ref.allen_cahn_x = xx; ref.allen_cahn_u = v;
fprintf('allen_cahn done: nsteps=%d, max|u|=%.4g\n', round(tf/dt), max(abs(v)));

% ================ CASE 4: NLS short (complex nonlinear) ============
dom = [-pi pi]; N = 64; dt = 0.02; tf = 1.0;   % 50 steps
[v, xx] = run_case(@(u) 1i*diff(u,2), @(u) 1i*abs(u).^2.*u, dom, tf, N, dt, ...
                   @(x) 1 + 0.5*cos(x));
ref.nls_N = N; ref.nls_dt = dt; ref.nls_tf = tf;
ref.nls_dom = dom; ref.nls_x = xx; ref.nls_u = v;
fprintf('nls done: nsteps=%d, max|u|=%.4g\n', round(tf/dt), max(abs(v)));

save(fullfile(outdir, 'spin.mat'), '-struct', 'ref');
fprintf('spin.mat: %d fields\n', numel(fieldnames(ref)));

% ------------------------------------------------------------------
% Helper: build a 1-var spinop, run spin (no plot), feval at trigpts.
% ------------------------------------------------------------------
function [vals, xx] = run_case(Lfun, Nfun, dom, tf, N, dt, u0fun)
    S = spinop(dom, [0 tf]);
    S.lin = Lfun;
    S.nonlin = Nfun;
    S.init = chebfun(u0fun, dom, 'trig');
    u = spin(S, N, dt, 'plot', 'off');
    xx = trigpts(N, dom);
    vals = feval(u, xx);
end
