%% Generate MATLAB reference data for chebtech arithmetic and calculus ops.
%
% Requires the Chebfun toolbox to be on the path:
%   addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref')
%
% Usage:
%   module load matlab/R2025b
%   cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/chebtech_ops_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end

ref = struct();

%% --- Arithmetic ---
n = 30;
x = chebpts(n, 2);
fs = chebtech2(@sin);
fc = chebtech2(@cos);
fe = chebtech2(@exp);

% Addition
h_add = fs + fc;
ref.add_coeffs = h_add.coeffs;

% Multiplication
h_mul = fs .* fs;
ref.mul_coeffs = h_mul.coeffs;

% Scalar add
h_sadd = fs + 1;
ref.sadd_coeffs = h_sadd.coeffs;

% Scalar mul
h_smul = 2 * fs;
ref.smul_coeffs = h_smul.coeffs;

% Negation
h_neg = -fs;
ref.neg_coeffs = h_neg.coeffs;

% Subtraction
h_sub = fs - fc;
ref.sub_coeffs = h_sub.coeffs;

%% --- Calculus ---

% diff(sin) = cos
df = diff(fs);
ref.diff_sin_coeffs = df.coeffs;

% diff(exp) = exp
dfe = diff(fe);
ref.diff_exp_coeffs = dfe.coeffs;

% diff(sin, 2) = -sin
d2f = diff(fs, 2);
ref.diff2_sin_coeffs = d2f.coeffs;

% cumsum(cos) should give sin - sin(-1)
Fc = cumsum(fc);
ref.cumsum_cos_coeffs = Fc.coeffs;

% Definite integrals
ref.sum_sin = sum(fs);       % ~0
ref.sum_cos = sum(fc);       % 2*sin(1)
ref.sum_exp = sum(fe);       % e - 1/e

% x^2 integral
fx2 = chebtech2(@(x) x.^2);
ref.sum_x2 = sum(fx2);      % 2/3

% Inner product <sin, sin>
ref.inner_sin_sin = innerProduct(fs, fs);

% Inner product <sin, cos>
ref.inner_sin_cos = innerProduct(fs, fc);

%% --- Roots ---

% Roots of sin
rts_sin = roots(fs);
ref.roots_sin = rts_sin;

% Roots of T_5
T5 = chebtech2(@(x) cos(5*acos(x)));
rts_T5 = roots(T5);
ref.roots_T5 = sort(rts_T5);

% Roots of x^2 - 0.25
fq = chebtech2(@(x) x.^2 - 0.25);
rts_q = roots(fq);
ref.roots_quadratic = sort(rts_q);

%% --- Coefficient references ---
% Store raw input coefficients for cross-validation
ref.sin_coeffs = fs.coeffs;
ref.cos_coeffs = fc.coeffs;
ref.exp_coeffs = fe.coeffs;

save(fullfile(outdir, 'chebtech_ops.mat'), '-struct', 'ref');
fprintf('Saved chebtech_ops.mat with %d fields.\n', length(fieldnames(ref)));
