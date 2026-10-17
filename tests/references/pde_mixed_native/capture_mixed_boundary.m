function capture_mixed_boundary(outputDir)
global DIFFORDER SYSSIZE
% Primary installed solver only. No instrumented solver or replay claim.
base=fileparts(mfilename('fullpath'));
ref='/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
pin='7574c77680d7e82b79626300bf255498271a72df';
assert(strcmp(outputDir,fullfile(base,'output')) && ~isfolder(outputDir), ...
    'Capture:Output','Use fresh prepared output directory');
manifest=jsondecode(fileread(fullfile(base,'CAPTURE_MANIFEST.json')));
verifySources(manifest);
checkReference(ref,pin);
assert(strcmp(version('-release'),'2025b'),'Capture:Release','R2025b required');
addpath(ref);rehash;
assert(strcmp(which('ode15s'),'/usr/licensed/matlab-R2025b/toolbox/matlab/funfun/ode15s.m'), ...
    'Capture:Binding','Installed ode15s required');
assert(strcmp(which('chebfun'),fullfile(ref,'@chebfun','chebfun.m')) && ...
       strcmp(which('chebdouble'),fullfile(ref,'@chebdouble','chebdouble.m')), ...
    'Capture:Binding','Unexpected constructor binding');
mkdir(outputDir);
metadata=struct('complete',false,'reference_pin',pin,'matlab_version',version, ...
    'solver',which('ode15s'),'mode','primary installed ode15s; no instrumented replay', ...
    'publication_context','321fb12d','expected_orders',[2,2,0], ...
    'expected_boundary_counts',[2,2],'expected_state_size',27);
writejson(fullfile(outputDir,'metadata.json'),metadata);
rng(7,'twister');
x=chebfun('x',[-1,1]);
u0=chebfun([1+0*x;2+0*x;3+x.^2]);
pdefun=@(u,v,w)[.1*diff(u,2);.2*diff(v,2);u.*v-w];
bc=struct('left',@(u,v,w)[diff(u);diff(v)], ...
          'right',@(u,v,w)[diff(u);diff(v)]);
opts=pdeset('N',9,'Plot','off','ODESolver',@observeSolver);
requested_times=[0,.005,.01];
try
    % Hidden nonstatic method: ordinary dispatch on Chebfun initial argument.
    [tout,solutions]=pdeSolve(pdefun,requested_times,u0,bc,opts);
    samples=cell(size(solutions));
    sample_x=[-.75;0;.625];
    for i=1:size(solutions,1)
        for j=1:size(solutions,2)
            samples{i,j}=feval(solutions{i,j},sample_x);
        end
    end
    save(fullfile(outputDir,'public_output.mat'),'tout','requested_times','sample_x','samples','-v7');
catch err
    metadata.error_id=err.identifier;metadata.error_message=err.message;
    writejson(fullfile(outputDir,'metadata.json'),metadata);rethrow(err);
end
verifySources(manifest);checkReference(ref,pin);
metadata.complete=true;
writejson(fullfile(outputDir,'metadata.json'),metadata);

    function [t,y]=observeSolver(fun,span,initial,options)
        % Global names inherit the outer scope; do NOT redeclare in nested function.
        assert(~isfile(fullfile(outputDir,'solver_capture.mat')), ...
            'Capture:Segments','Expected one fixed-N segment');
        assert(SYSSIZE==3 && isequal(DIFFORDER,[2,2,0]), ...
            'Capture:Orders','Unexpected native order/component context');
        context=struct;bindings=struct;
        info=functions(fun);
        numericNames={'P','M','x','n','BCVALOFFSET','numBCs','numLinBCs','numLBCs','numRBCs','q'};
        handleNames={'opLBC','opRBC','pdeFun'};
        for h=1:numel(info.workspace)
            ws=info.workspace{h};
            for j=1:numel(numericNames)
                name=numericNames{j};
                if isfield(ws,name) && isnumeric(ws.(name)),context.(name)=ws.(name);end
            end
            for j=1:numel(handleNames)
                name=handleNames{j};
                if isfield(ws,name) && isa(ws.(name),'function_handle'),bindings.(name)=ws.(name);end
            end
        end
        assert(all(isfield(context,numericNames)), ...
            'Capture:Context','Missing actual numeric source context; never synthesize P');
        assert(all(isfield(bindings,handleNames)), ...
            'Capture:Bindings','Missing parsed native expression/boundary handles');
        assert(context.n==9 && context.numBCs==4 && context.numLinBCs==0 && ...
            context.numLBCs==2 && context.numRBCs==2, ...
            'Capture:ResidualCount','Unexpected native residual counts');
        mass=odeget(options,'Mass');
        assert(isequal(size(mass),[27,27]) && isequal(mass,context.M) && ...
            isequal(context.P,context.M),'Capture:Mass','Unexpected default mass/projection');
        assert(isequal(context.P(1:4,:),zeros(4,27)), ...
            'Capture:Rows','Boundary mass rows must be zero');
        record=struct('mass',mass,'initial',initial,'span',span,'context',context, ...
            'difforder',DIFFORDER,'syssize',SYSSIZE,'options',cleanOptions(options));
        record.initial_rhs_before_probes=fun(span(1),initial(:));
        xx=context.x(:);
        % Fixed nonconstant polynomial state: independent of any observed outputs.
        probe=[1+xx+2*xx.^2, 2-3*xx+2*xx.^2+xx.^3, 3+xx.^2-2*xx.^3];
        probe_time=.003;
        myU=chebdouble(probe,[-1,1]);
        full_pde=double(bindings.pdeFun(probe_time,xx,myU));
        full_left=double(bindings.opLBC(probe_time,xx,myU));
        full_right=double(bindings.opRBC(probe_time,xx,myU));
        probe_rhs=fun(probe_time,probe(:));
        assert(isequal(size(full_pde),[9,3]) && isequal(size(full_left),[9,2]) && ...
            isequal(size(full_right),[9,2]),'Capture:ExpressionShape','Unexpected expression columns');
        endpoint_residuals=[full_left(1,:).';full_right(end,:).'];
        % These assertions are literal source bookkeeping identities, not fitted bounds.
        assert(isequal(probe_rhs(1:4),endpoint_residuals-context.BCVALOFFSET), ...
            'Capture:RowOrder','Boundary endpoint/offset order differs');
        record.probe=struct('time',probe_time,'state',probe,'full_pde',full_pde, ...
            'full_left',full_left,'full_right',full_right, ...
            'endpoint_residuals',endpoint_residuals,'rhs',probe_rhs);
        record.initial_rhs_after_probes=fun(span(1),initial(:));
        assert(isequal(record.initial_rhs_before_probes,record.initial_rhs_after_probes), ...
            'Capture:ProbeMutation','Probe evaluations changed initial RHS');
        originalOutput=odeget(options,'OutputFcn');
        assert(isa(originalOutput,'function_handle'),'Capture:Callback','Missing native callback');
        callbackLog={};
        primaryOptions=odeset(options,'OutputFcn',@forwardOutput);
        record.rng_before_primary=rng;
        try
            [t,y]=ode15s(fun,span,initial,primaryOptions);
        catch err
            record.callback_log=callbackLog;
            record.error_id=err.identifier;record.error_message=err.message;
            save(fullfile(outputDir,'solver_capture.mat'),'record','-v7');rethrow(err);
        end
        record.rng_after_primary=rng;
        record.primary_t=t;record.primary_y=y;record.callback_log=callbackLog;
        record.initial_rhs_after_primary=fun(span(1),initial(:));
        assert(isequal(record.initial_rhs_before_probes,record.initial_rhs_after_primary), ...
            'Capture:ClosureMutation','Fixed-N initial RHS changed');
        % Save before pinned pdeSolve's clear global; no globals needed downstream.
        save(fullfile(outputDir,'solver_capture.mat'),'record','-v7');
        summary=struct('orders',DIFFORDER,'boundary_counts',[context.numLBCs,context.numRBCs], ...
            'matrix_size',size(mass),'block_row_counts',[7,7,9], ...
            'primary_output_times',t,'callback_count',numel(callbackLog), ...
            'rng_unchanged',isequal(record.rng_before_primary,record.rng_after_primary), ...
            'zero_order_block_is_identity',isequal(context.P(19:27,19:27),eye(9)), ...
            'instrumented_replay',false);
        writejson(fullfile(outputDir,'solver_summary.json'),summary);

        function stop=forwardOutput(tt,yy,flag)
            stop=originalOutput(tt,yy,flag);
            assert(isempty(stop) || ~any(stop(:)),'Capture:Stop','Unexpected fixed-N stop');
            callbackLog{end+1}=struct('t',tt,'y',yy,'flag',flag);
        end
    end
end
function result=cleanOptions(options)
result=options;fields=fieldnames(options);
for i=1:numel(fields)
    name=fields{i};
    if isa(result.(name),'function_handle'),result.(name)=func2str(result.(name));end
end
end
function verifySources(manifest)
for i=1:numel(manifest.files)
    entry=manifest.files(i);[rc,value]=system(['sha256sum "' entry.path '"']);
    assert(rc==0 && startsWith(value,entry.sha256),'Capture:Hash','Source changed: %s',entry.path);
end
end
function checkReference(ref,pin)
[rc,head]=system(['git -C ' ref ' rev-parse HEAD']);
assert(rc==0 && strcmp(strtrim(head),pin),'Capture:Pin','Reference pin changed');
[rc,dirty]=system(['git -C ' ref ' status --porcelain']);
assert(rc==0 && isempty(strtrim(dirty)),'Capture:Dirty','Reference must be clean');
end
function writejson(path,value)
f=fopen(path,'w');assert(f>=0,'Capture:Write','Cannot write output');
cleanup=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(value));
end
