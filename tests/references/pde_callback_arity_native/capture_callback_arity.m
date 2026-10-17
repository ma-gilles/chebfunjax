function capture_callback_arity(outputDir)
% Observation harness for pdeSolve's N+1 independent-variable callback roles.
% No acceptance tolerance is measured or selected here; this writes the native
% output arrays and role observations for an unrun Python replay draft.
base = fileparts(mfilename('fullpath'));
ref = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
pin = '7574c77680d7e82b79626300bf255498271a72df';
assert(strcmp(outputDir, fullfile(base, 'output')) && ~isfolder(outputDir), ...
    'Capture:Output', 'Expected a fresh prepared output directory');
[rc, head] = system(['git -C ' ref ' rev-parse HEAD']);
assert(rc == 0 && strcmp(strtrim(head), pin), 'Capture:Pin', 'Reference pin changed');
[rc, dirty] = system(['git -C ' ref ' status --porcelain']);
assert(rc == 0 && isempty(strtrim(dirty)), 'Capture:Dirty', 'Reference must be clean');
addpath(ref); rehash;
assert(strcmp(version('-release'), '2025b'), 'Capture:Release', 'R2025b required');
assert(strcmp(which('chebfun'), fullfile(ref, '@chebfun', 'chebfun.m')), ...
    'Capture:Binding', 'Unexpected Chebfun binding');
assert(strcmp(which('ode15s'), '/usr/licensed/matlab-R2025b/toolbox/matlab/funfun/ode15s.m'), ...
    'Capture:Binding', 'Expected installed R2025b ode15s');
mkdir(outputDir);
role = struct('pde_coordinate_values', [], 'pde_coordinate_domain', [], ...
              'boundary_time_values', [], ...
              'pde_argument_classes', {{}}, 'boundary_argument_classes', {{}});

% Keep the existing three-component PDE coefficients and solver controls.
% The added x and t forcing terms make independent-variable routing observable.
rng(7, 'twister');
x = chebfun('x', [-1, 1]);
u0 = chebfun([1 + 0*x; 2 + 0*x; 4 + 0*x]);
pde = @(x,u,v,w) pde_x_u_v_w(x,u,v,w);
bc.left = @(t,u,v,w) bc_t_u_v_w(t,u,v,w);
bc.right = @(t,u,v,w) bc_t_u_v_w(t,u,v,w);
tIn = [0, 0.005, 0.01];
probe = [-0.75; 0; 0.625];
opts = pdeset('N', 9, 'Plot', 'off', 'RelTol', 1e-9, 'AbsTol', 1e-11);
optsSummary = struct('N', opts.N, 'RelTol', opts.RelTol, 'AbsTol', opts.AbsTol, ...
                     'Eps', opts.Eps, 'solver', which('ode15s'));
metadata = struct('complete', false, 'pin', pin, 'matlab_version', version, ...
    'pdeSolve_source', fullfile(ref, '@chebfun', 'pdeSolve.m'), ...
    'pdeSolve_binding', which('pdeSolve'), ...
    'pde_signature', 'pde(x,u,v,w): N+1, parser independent role is x', ...
    'boundary_signature', 'bc(t,u,v,w): N+1, left/right parser independent role is t', ...
    'fixed_N', 9, 'times', tIn, 'probe', probe, 'options', optsSummary);
writejson(fullfile(outputDir, 'metadata.json'), metadata);

[tout, solutions] = pdeSolve(pde, tIn, u0, bc, opts);
samples = cell(size(solutions));
assert(isequal(size(solutions), [3,3]), 'Capture:Shape', 'Expected species-by-time solution');
for species = 1:size(solutions,1)
    for timeIndex = 1:size(solutions,2)
        samples{species,timeIndex} = feval(solutions{species,timeIndex}, probe);
        assert(isequal(size(samples{species,timeIndex}), [3,1]) && ...
            all(isfinite(samples{species,timeIndex})), ...
            'Capture:Samples', 'Every species/time cell must have three finite samples');
    end
end
metadata.pde_coordinate_values = role.pde_coordinate_values;
metadata.pde_coordinate_domain = role.pde_coordinate_domain;
metadata.boundary_time_values = unique(role.boundary_time_values, 'stable');
metadata.pde_argument_classes = unique(role.pde_argument_classes);
metadata.boundary_argument_classes = unique(role.boundary_argument_classes);
metadata.complete = true;
save(fullfile(outputDir, 'public_output.mat'), ...
    'tout', 'samples', 'probe', 'tIn', 'optsSummary', '-v7');
writejson(fullfile(outputDir, 'metadata.json'), metadata);

    function values = pde_x_u_v_w(xarg, u, v, w)
        role.pde_argument_classes{end+1} = class(xarg);
        % Store coordinate content once without assuming the proxy class. The
        % documented chebdouble worker exposes .values/.domain and double().
        xValues = double(xarg);
        if isempty(role.pde_coordinate_values) && numel(xValues) == 9
            role.pde_coordinate_values = xValues;
            if isobject(xarg) && isprop(xarg, 'domain')
                role.pde_coordinate_domain = xarg.domain;
            else
                role.pde_coordinate_domain = [min(role.pde_coordinate_values(:)), ...
                                              max(role.pde_coordinate_values(:))];
            end
        end
        values = [0.1*diff(u,2) - u + v + xarg; ...
                  0.2*diff(v,2) + u - 2*v + w + 2*xarg; ...
                  0.05*diff(w,2) + v - w - xarg];
    end

    function values = bc_t_u_v_w(targ, u, v, w)
        role.boundary_argument_classes{end+1} = class(targ);
        role.boundary_time_values(end+1) = double(targ);
        values = [diff(u) - targ; diff(v) - 2*targ; diff(w) + targ];
    end
end

function writejson(path, value)
f = fopen(path, 'w'); assert(f >= 0, 'Capture:Write', 'Could not open metadata file');
cleanup = onCleanup(@() fclose(f)); %#ok<NASGU>
fwrite(f, jsonencode(value, PrettyPrint=true), 'char'); fwrite(f, newline, 'char');
end
