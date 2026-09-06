% Golden reference for every expinteg time-stepping scheme (-> spin_schemes.mat).
%
% KdV (spinop('KDV'), N = 128, dt = 1e-6, 20 steps) with all 31 schemes of
% @expinteg/expinteg.m, and Gray-Scott (spinop2('GS'), N = 64, dt = 1e-2,
% 10 steps) with a subset, both with MATLAB's default dealias = 'off'.
% The GS grids are stored subsampled (every 4th point in each direction).
%
% Run from the repo root with the chebfun reference on the path:
%   matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/refs/spin_schemes_refs.m')"

schemes = {'abnorsett4','abnorsett5','abnorsett6','etdrk2','etdrk4', ...
    'exprk5s8','friedli','hochbruck-ostermann','krogstad','minchev', ...
    'strehmel-weiner','ablawson4','lawson4','genlawson41','genlawson42', ...
    'genlawson43','genlawson44','genlawson45','modgenlawson41', ...
    'modgenlawson42','modgenlawson43','modgenlawson44','modgenlawson45', ...
    'pec423','pecec433','pec524','pecec534','pec625','pecec635','pec726', ...
    'pecec736'};
kdv_N = 128; kdv_dt = 1e-6; kdv_nsteps = 20;
S = spinop('KDV'); S.tspan = [0 kdv_nsteps*kdv_dt];
x = trigpts(kdv_N, S.domain);
kdv_u = zeros(kdv_N, numel(schemes));
for k = 1:numel(schemes)
    u = spin(S, kdv_N, kdv_dt, 'plot', 'off', 'scheme', schemes{k});
    kdv_u(:, k) = u(x);
end
kdv_schemes = schemes;

gs_schemes = {'etdrk4','pecec736','krogstad','genlawson43','abnorsett4', ...
    'exprk5s8','lawson4','modgenlawson44'};
gs_N = 64; gs_dt = 1e-2; gs_nsteps = 10;
S2 = spinop2('GS'); S2.tspan = [0 gs_nsteps*gs_dt];
[xx, yy] = meshgrid(trigpts(gs_N, [0 1]), trigpts(gs_N, [0 1]));
xx = xx(1:4:end, 1:4:end); yy = yy(1:4:end, 1:4:end);
gs_uv_sub4 = zeros(numel(xx), 2*numel(gs_schemes));
for k = 1:numel(gs_schemes)
    u = spin2(S2, gs_N, gs_dt, 'plot', 'off', 'scheme', gs_schemes{k});
    c = u.blocks;
    v1 = feval(c{1}, xx, yy); v2 = feval(c{2}, xx, yy);
    gs_uv_sub4(:, 2*k-1) = v1(:); gs_uv_sub4(:, 2*k) = v2(:);
end
save('tests/references/spin_schemes.mat', 'kdv_schemes', 'kdv_u', 'kdv_N', ...
    'kdv_dt', 'kdv_nsteps', 'gs_schemes', 'gs_uv_sub4', 'gs_N', 'gs_dt', ...
    'gs_nsteps');
