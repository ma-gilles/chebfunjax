%% Second batch of core Chebfun reference data (broadens coverage).
%
% Cross-validates more @chebfun operations vs MATLAB R2025b: mean,
% flipud, complex real/imag/abs/angle, special functions (erf, besselj),
% sign.  Added by Claude Opus 4.8.
%
% Usage:
%   module load matlab/R2025b
%   cd /home/mg6942/chebfunjax
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebfun_core2_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();
xt = linspace(-0.9, 0.9, 15).';
ref.xt = xt;

% ---- mean and flipud ----
f = chebfun(@(x) exp(x).*sin(3*x));
ref.f_mean = mean(f);
ref.f_flipud = feval(flipud(f), xt);       % flipud(f)(x) = f(-x) on [-1,1]

% ---- complex-valued chebfun c = exp(2i x) ----
c = chebfun(@(x) exp(2i*x));
ref.c_real = feval(real(c), xt);
ref.c_imag = feval(imag(c), xt);
ref.c_abs  = feval(abs(c), xt);            % == 1
ref.c_angle = feval(angle(c), xt);          % == 2x

% ---- special functions ----
ref.erf_vals = feval(erf(chebfun(@(x) 2*x)), xt);
ref.besselj0_vals = feval(besselj(0, chebfun(@(x) 5*x)), xt);

% ---- sign of a function with roots ----
s = chebfun(@(x) sin(4*x));
ref.sign_vals = feval(sign(s), xt);

% ---- cumsum then diff recovers the function (up to a constant) ----
g = chebfun(@(x) cos(2*x));
ref.cumsum_diff = feval(diff(cumsum(g)), xt);   % == cos(2x)

save(fullfile(outdir, 'chebfun_core2.mat'), '-struct', 'ref');
fprintf('Wrote chebfun_core2.mat with %d fields.\n', numel(fieldnames(ref)));
