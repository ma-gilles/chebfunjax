% ballfun_refs.m — MATLAB references for the ballfun module.
% Named *_refs.m to avoid colliding with Chebfun's @ballfun class.
% chebfunjax Ballfun is constructed Cartesian (op(x,y,z), default) and
% evaluated in spherical coords f(r, lam, th) with x=r sin(th)cos(lam),
% y=r sin(th)sin(lam), z=r cos(th) (r in [0,1], lam in [-pi,pi] longitude,
% th in [0,pi] colatitude).  Pins evaluation, sum3 (integral over the unit
% ball) and the L2 norm at rtol 1e-12.
%
% Usage:
%   matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/ballfun_refs.m')"

outdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'references');
if ~exist(outdir, 'dir'), mkdir(outdir); end
ref = struct();

rp   = [0.2;  0.5;  0.7;  0.9;  0.3;  0.6];   % radius in [0,1]
lamp = [-2.5; -1.0;  0.0;  1.2;  2.8; -0.5];  % longitude in [-pi,pi]
thp  = [0.3;  0.8;  1.5;  2.0;  2.7;  1.1];   % colatitude in [0,pi]
ref.rp = rp;
ref.lamp = lamp;
ref.thp = thp;

cfuns = { @(x,y,z) z, ...
          @(x,y,z) x, ...
          @(x,y,z) exp(x), ...
          @(x,y,z) 1 + x.*z };
for i = 1:numel(cfuns)
    f = ballfun(cfuns{i});
    ev = zeros(numel(rp), 1);
    for k = 1:numel(rp)
        ev(k) = feval(f, rp(k), lamp(k), thp(k), 'spherical');
    end
    ref.(sprintf('f%d_eval', i)) = ev;
    ref.(sprintf('f%d_sum', i))  = sum3(f);
    ref.(sprintf('f%d_norm', i)) = norm(f);
end

save(fullfile(outdir, 'ballfun.mat'), '-struct', 'ref');
fprintf('ballfun.mat: %d fields\n', numel(fieldnames(ref)));
