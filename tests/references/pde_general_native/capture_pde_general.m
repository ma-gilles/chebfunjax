function capture_pde_general(outputDir)
global PDE_CAPTURE DIFFORDER SYSSIZE
% Diagnostic preparation. Primary numerical route is installed ode15s.
base=fileparts(mfilename('fullpath'));
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
pin='7574c77680d7e82b79626300bf255498271a72df';
assert(strcmp(outputDir,fullfile(base,'output')) && ~isfolder(outputDir), ...
    'Capture:Output','Fresh prepared output path required');
manifest=jsondecode(fileread(fullfile(base,'SOURCE_MANIFEST.json')));
verify(manifest);
[rc,head]=system(['git -C ' ref ' rev-parse HEAD']);
assert(rc==0 && strcmp(strtrim(head),pin),'Capture:Pin','Reference pin changed');
[rc,dirty]=system(['git -C ' ref ' status --porcelain']);
assert(rc==0 && isempty(strtrim(dirty)),'Capture:Dirty','Reference must be clean');
assert(strcmp(version('-release'),'2025b'),'Capture:Release','R2025b required');
addpath(ref);addpath(fullfile(base,'licensed_runtime'));rehash;
assert(strcmp(which('ode15s'),'/usr/licensed/matlab-R2025b/toolbox/matlab/funfun/ode15s.m'), ...
    'Capture:Binding','Primary solver must be installed ode15s');
assert(strcmp(which('chebfun'),fullfile(ref,'@chebfun','chebfun.m')), ...
    'Capture:Binding','Wrong Chebfun constructor');
assert(strcmp(which('diagnostic_ode15s'),fullfile(base,'licensed_runtime','diagnostic_ode15s.m')), ...
    'Capture:Binding','Wrong diagnostic solver');
assert(strcmp(which('condest'),'/usr/licensed/matlab-R2025b/toolbox/matlab/matfun/condest.m'), ...
    'Capture:Binding','Unexpected condest binding');
assert(strcmp(which('capture_note'),fullfile(base,'capture_note.m')) && ...
    strcmp(which('capture_condest'),fullfile(base,'capture_condest.m')), ...
    'Capture:Binding','Unexpected observation hook binding');
mkdir(outputDir);
metadata=struct('complete',false,'pin',pin,'matlab_version',version, ...
    'solver',which('ode15s'),'diagnostic_solver',which('diagnostic_ode15s'), ...
    'scope','fixed n=9, three real second-order components, Neumann handles');
writejson(fullfile(outputDir,'metadata.json'),metadata);
rng(7,'twister');
x=chebfun('x',[-1,1]);
u0=chebfun([1+0*x;2+0*x;4+0*x]);
f=@(u,v,w) [.1*diff(u,2)-u+v; .2*diff(v,2)+u-2*v+w; .05*diff(w,2)+v-w];
bc=struct('left',@(u,v,w)[diff(u);diff(v);diff(w)], ...
          'right',@(u,v,w)[diff(u);diff(v);diff(w)]);
opts=pdeset('N',9,'Plot','off','ODESolver',@observeSolver);
% pdeSolve takes (PDEFUN,T,U0,BC,OPTS), NOT a solver first argument.
% Three requested times avoid the source's two-time -> 51-point expansion.
try
    [tout,solutions]=pdeSolve(f,[0,.005,.01],u0,bc,opts);
    samples=cell(size(solutions));
    for i=1:size(solutions,1)
        for j=1:size(solutions,2)
            samples{i,j}=feval(solutions{i,j},[-.75;0;.625]);
        end
    end
    save(fullfile(outputDir,'public_output.mat'),'tout','samples','-v7');
catch err
    metadata.error_id=err.identifier; metadata.error_message=err.message;
    writejson(fullfile(outputDir,'metadata.json'),metadata); rethrow(err);
end
verify(manifest);
[rc,dirty]=system(['git -C ' ref ' status --porcelain']);
assert(rc==0 && isempty(strtrim(dirty)),'Capture:Changed','Reference changed');
saved=load(fullfile(outputDir,'solver_capture.mat'),'record');
PDE_CAPTURE=saved.record;
metadata.complete=true;
writejson(fullfile(outputDir,'metadata.json'),metadata);

    function [t,y]=observeSolver(fun,span,initial,options)
        assert(~isfile(fullfile(outputDir,'solver_capture.mat')), ...
            'Capture:ExtraSegment','Expected one fixed-N solve');
        assert(SYSSIZE==3 && isequal(DIFFORDER,[2,2,2]), ...
            'Capture:Dimensions','Unexpected native component/order context');
        mass=odeget(options,'Mass');
        assert(isnumeric(mass) && isequal(size(mass),[27,27]), ...
            'Capture:Mass','Expected native 27-by-27 full mass');
        PDE_CAPTURE=struct('mass',mass,'initial',initial,'span',span, ...
            'difforder',DIFFORDER,'syssize',SYSSIZE,'options',cleanOptions(options));
        % Copy only named numeric closure fields; never serialize function handles.
        info=functions(fun); context=struct;
        names={'P','M','x','n','BCVALOFFSET','numBCs','numLinBCs','numLBCs','numRBCs','q'};
        if isfield(info,'workspace')
            for h=1:numel(info.workspace)
                ws=info.workspace{h};
                for z=1:numel(names)
                    key=names{z};
                    if isfield(ws,key) && isnumeric(ws.(key)),context.(key)=ws.(key);end
                end
            end
        end
        PDE_CAPTURE.context=context;
        PDE_CAPTURE.initial_rhs=fun(span(1),initial(:));
        originalOutput=odeget(options,'OutputFcn');
        assert(isa(originalOutput,'function_handle'),'Capture:Callback','Expected pdeSolve callback');
        primaryLog={}; replayLog={};
        primaryOptions=odeset(options,'OutputFcn',@forwardOutput);
        PDE_CAPTURE.rng_before_primary=rng;
        [t,y]=ode15s(fun,span,initial,primaryOptions);
        afterPrimary=rng;
        PDE_CAPTURE.rng_after_primary=afterPrimary;
        PDE_CAPTURE.primary_t=t; PDE_CAPTURE.primary_y=y;
        PDE_CAPTURE.primary_callback_log=primaryLog;
        PDE_CAPTURE.rhs_after_primary=fun(span(1),initial(:));
        assert(isequal(PDE_CAPTURE.initial_rhs,PDE_CAPTURE.rhs_after_primary), ...
            'Capture:ClosureChanged','Fixed-N RHS changed during primary callbacks');
        % Replay on the same reachable assembled RHS, with inert output logging.
        % Original Chebfun callback must NOT execute again or mutate its counters.
        restore=onCleanup(@()rng(afterPrimary));
        rng(PDE_CAPTURE.rng_before_primary);
        replayOptions=odeset(options,'OutputFcn',@replayOutput);
        try
            [rt,ry]=diagnostic_ode15s(fun,span,initial,replayOptions);
            PDE_CAPTURE.replay_t=rt; PDE_CAPTURE.replay_y=ry;
            PDE_CAPTURE.replay_callback_log=replayLog;
            PDE_CAPTURE.rng_after_replay=rng;
            PDE_CAPTURE.replay_mesh_equal=isequal(t,rt);
            PDE_CAPTURE.replay_state_equal=isequal(y,ry);
            PDE_CAPTURE.replay_callbacks_equal=isequal(primaryLog,replayLog);
            PDE_CAPTURE.replay_rng_equal=isequal(afterPrimary,rng);
        catch err
            PDE_CAPTURE.replay_error_id=err.identifier;
            PDE_CAPTURE.replay_error_message=err.message;
            record=PDE_CAPTURE;save(fullfile(outputDir,'solver_capture.mat'),'record','-v7');
            rethrow(err);
        end
        record=PDE_CAPTURE;save(fullfile(outputDir,'solver_capture.mat'),'record','-v7');
        summary=struct('J_size',size(record.algebraic.J), ...
            'algebraic_count',numel(record.algebraic.AlgVar), ...
            'condest',record.condest.value, ...
            'index_margin_product',eps*nnz(record.algebraic.J)*record.condest.value, ...
            'replay_mesh_equal',record.replay_mesh_equal, ...
            'replay_state_equal',record.replay_state_equal, ...
            'replay_callbacks_equal',record.replay_callbacks_equal, ...
            'replay_rng_equal',record.replay_rng_equal);
        writejson(fullfile(outputDir,'solver_summary.json'),summary);
        assert(isequal(size(record.algebraic.J),[6,6]),'Capture:Algebraic','Expected reachable 6x6 J');
        assert(record.replay_mesh_equal && record.replay_state_equal && ...
            record.replay_callbacks_equal && record.replay_rng_equal, ...
            'Capture:ReplayMismatch','Diagnostic replay differs; do not qualify internals');

        function stop=forwardOutput(tt,yy,flag)
            stop=originalOutput(tt,yy,flag);
            assert(isempty(stop) || ~any(stop(:)),'Capture:Stop','Unexpected fixed-N callback stop');
            primaryLog{end+1}=struct('t',tt,'y',yy,'flag',flag);
        end
        function stop=replayOutput(tt,yy,flag)
            stop=false;
            replayLog{end+1}=struct('t',tt,'y',yy,'flag',flag);
        end
    end
end
function result=cleanOptions(options)
result=options;
fields=fieldnames(options);
for i=1:numel(fields)
    name=fields{i};
    if isa(result.(name),'function_handle'),result.(name)=func2str(result.(name));end
end
end
function verify(manifest)
for i=1:numel(manifest.files)
    entry=manifest.files(i);
    [rc,value]=system(['sha256sum "' entry.path '"']);
    assert(rc==0 && startsWith(value,entry.sha256),'Capture:Hash','Source changed: %s',entry.path);
end
end
function writejson(path,value)
f=fopen(path,'w');assert(f>=0,'Capture:Write','Cannot write output');
cleanup=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(value));
end
