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
% 3-D systems (spinop3 'GS' and 'SCHNAK'), N = 16, every 2nd grid point.
% NOTE (MATLAB 7574c77): the multistep schemes (pecec736, ...) disagree
% with every one-step scheme by ~1e-2 in 3-D, while chebfunjax's 3-D
% multistep agrees with ETDRK4 to ~1e-10, so the test pins only the
% one-step schemes to MATLAB.
spin3_schemes = {'etdrk4', 'pecec736', 'krogstad'};
spin3_N = 16; spin3_nsteps = 5; spin3_gs_dt = 1e-1; spin3_schnak_dt = 1e-3;
cases = {'GS', spin3_gs_dt; 'SCHNAK', spin3_schnak_dt};
for c = 1:2
    S3 = spinop3(cases{c, 1}); dt3 = cases{c, 2};
    S3.tspan = [0 spin3_nsteps*dt3]; G = S3.domain(2);
    [xx, yy, zz] = meshgrid(trigpts(spin3_N, [0 G]), trigpts(spin3_N, [0 G]), ...
        trigpts(spin3_N, [0 G]));
    xx = xx(1:2:end, 1:2:end, 1:2:end); yy = yy(1:2:end, 1:2:end, 1:2:end);
    zz = zz(1:2:end, 1:2:end, 1:2:end);
    M = zeros(numel(xx), 2*numel(spin3_schemes));
    for k = 1:numel(spin3_schemes)
        u = spin3(S3, spin3_N, dt3, 'plot', 'off', 'scheme', spin3_schemes{k});
        b = u.blocks;
        v1 = feval(b{1}, xx, yy, zz); v2 = feval(b{2}, xx, yy, zz);
        M(:, 2*k-1) = v1(:); M(:, 2*k) = v2(:);
    end
    if c == 1, spin3_gs_uv_sub2 = M; else, spin3_schnak_uv_sub2 = M; end
end
save('tests/references/spin_schemes.mat', 'kdv_schemes', 'kdv_u', 'kdv_N', ...
    'kdv_dt', 'kdv_nsteps', 'gs_schemes', 'gs_uv_sub4', 'gs_N', 'gs_dt', ...
    'gs_nsteps', 'spin3_schemes', 'spin3_N', 'spin3_nsteps', 'spin3_gs_dt', ...
    'spin3_schnak_dt', 'spin3_gs_uv_sub2', 'spin3_schnak_uv_sub2');
