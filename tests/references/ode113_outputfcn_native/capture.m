function capture(outputDir)
% Native OutputFcn observations only; no acceptance-bound changes.
ref = '/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref';
pin = '7574c77680d7e82b79626300bf255498271a72df';
base = fileparts(mfilename('fullpath'));
expected = fullfile(base,'output');
assert(strcmp(outputDir,expected),'Capture:OutputPath','Use the prepared output directory');
assert(~isfolder(outputDir),'Capture:Exists','Output already exists');
[rc,head] = system(['git -C ' ref ' rev-parse HEAD']);
assert(rc==0 && strcmp(strtrim(head),pin),'Capture:Pin','Wrong reference pin');
[rc,dirty] = system(['git -C ' ref ' status --porcelain']);
assert(rc==0 && isempty(strtrim(dirty)),'Capture:Dirty','Reference must be clean');
addpath(ref); rehash;
assert(strcmp(version('-release'),'2025b'),'Capture:Release','R2025b required');
assert(strcmp(which('ode113'),fullfile(matlabroot,'toolbox','matlab','funfun','ode113.m')), ...
    'Capture:SolverBinding','Unexpected ode113');
assert(strcmp(which('chebfun'),fullfile(ref,'@chebfun','chebfun.m')), ...
    'Capture:ConstructorBinding','Unexpected chebfun');
manifest = jsondecode(fileread(fullfile(base,'SOURCE_MANIFEST.json')));
verifySources(manifest);
mkdir(outputDir);
metadata = struct('reference',ref,'pin',pin,'matlab_version',version, ...
    'solver',which('ode113'),'constructor',which('chebfun'), ...
    'rhs','[-y(1); 2*y(2)]','initial_state',[1;2], ...
    'complete',false,'cases_complete',0,'source_manifest',manifest);
writejson(fullfile(outputDir,'metadata.json'),metadata);
% Each run uses fixed source options. No numerical acceptance criteria are fitted.
baseopts = odeset('RelTol',1e-6,'AbsTol',1e-9,'InitialStep',.01,'MaxStep',.05);
cases = { ...
 spec('forward','backend',[0,.2],1,[], 'none',false,false), ...
 spec('refined_selected','backend',[0,.2],4,[2,1,2],'none',false,false), ...
 spec('reverse','backend',[.2,0],4,[2,1,2],'none',false,false), ...
 spec('requested','backend',[0,.013,.071,.19,.2],4,[],'none',false,false), ...
 spec('stop','backend',[0,.2],4,[],'stop2',false,false), ...
 spec('init_return','backend',[0,.2],1,[],'initstop',false,false), ...
 spec('event_requested','backend',[0,.04,.1,.2],4,[],'none',true,false), ...
 spec('throw','backend',[0,.2],1,[],'throw2',false,false), ...
 spec('requested_stop','backend',[0,.001,.002,.003,.2],4,[],'stop2',false,false), ...
 spec('wrapper_normal','wrapper',[0,.2],4,[2,1,2],'none',false,true), ...
 spec('wrapper_stop','wrapper',[0,.2],1,[],'stop2',false,true), ...
 spec('wrapper_restart_stop','wrapper',[0,.1,.2],1,[],'stop2',false,true), ...
 spec('wrapper_no_restart_stop','wrapper',[0,.1,.2],4,[],'stop2',false,false), ...
 spec('wrapper_event','wrapper',[0,.1,.2],4,[],'none',true,true)};
summaries = cell(size(cases));
for k=1:numel(cases)
    c = cases{k}; log = {}; ordinary = 0;
    options = odeset(baseopts,'Refine',c.refine,'OutputSel',c.selection,'OutputFcn',@record);
    if c.event, options = odeset(options,'Events',@terminalEvent); end
    if strcmp(c.route,'wrapper'), options.restartSolver=c.restart; end
    observation = struct('case',c,'success',false,'error_id','','error_message','','phase','solve','construction_success',false);
    sol = []; control = []; wrapper_values = []; wrapper_domain = [];
    t0=tic;
    try
        if strcmp(c.route,'backend')
            sol = ode113(@rhs,c.span,[1;2],options);
        else
            f = chebfun.ode113(@rhs,c.span,[1;2],options);
            observation.construction_success=true;
            observation.phase='evaluate_constructed_solution';
            wrapper_domain = domain(f);
            query = linspace(wrapper_domain(1),wrapper_domain(end),9);
            wrapper_values = feval(f,query);
        end
        observation.success=true; observation.phase='complete';
    catch err
        observation.error_id=err.identifier; observation.error_message=err.message;
    end
    observation.seconds=toc(t0);
    % Freeze callback log before a callback-free control. No callback replays.
    callback_log=log;
    observation.callback_count=numel(callback_log);
    observation.callback_flags=cellfun(@(v)v.flag,callback_log,'UniformOutput',false);
    observation.mesh_equal_control=[]; observation.state_equal_control=[];
    if strcmp(c.route,'backend') && observation.success 
        controlopts=odeset(options,'OutputFcn',[]);
        try
            control=ode113(@rhs,c.span,[1;2],controlopts);
            observation.mesh_equal_control=isequaln(sol.x,control.x);
            observation.state_equal_control=isequaln(sol.y,control.y);
        catch err
            observation.control_error_id=err.identifier;
            observation.control_error_message=err.message;
        end
    end
    if ~isempty(sol)
        observation.mesh=sol.x; observation.state=sol.y;
        if isfield(sol,'xe'), observation.event_times=sol.xe; end
        if isfield(sol,'ie'), observation.event_indices=sol.ie; end
    end
    % Function handles in sol/extdata are excluded; preserve numerical history.
    solution_numeric=striphandles(sol); control_numeric=striphandles(control);
    save(fullfile(outputDir,[c.name '.mat']),'c','callback_log','observation', ...
        'solution_numeric','control_numeric','wrapper_domain','wrapper_values','-v7');
    summaries{k}=observation;
    writejson(fullfile(outputDir,[c.name '.json']),observation);
    writejson(fullfile(outputDir,[c.name '_callbacks.json']),callback_log);
    metadata.cases_complete=k;
    writejson(fullfile(outputDir,'metadata.json'),metadata);
end
[rc,dirty] = system(['git -C ' ref ' status --porcelain']);
assert(rc==0 && isempty(strtrim(dirty)),'Capture:Changed','Reference changed');
verifySources(manifest);
metadata.complete=true;
writejson(fullfile(outputDir,'metadata.json'),metadata);
writejson(fullfile(outputDir,'summary.json'),summaries);

    function stop=record(t,y,flag)
        stop=false;
        if isempty(flag)
            ordinary=ordinary+1;
            stop=strcmp(c.mode,'stop2') && ordinary==2;
        elseif strcmp(flag,'init')
            ordinary=0;
            stop=strcmp(c.mode,'initstop');
        end
        entry=struct('flag',flag,'t',t,'y',y,'t_size',size(t),'y_size',size(y), ...
            'return_stop',stop,'throws',false);
        if isempty(flag) && strcmp(c.mode,'throw2') && ordinary==2
            entry.throws=true; log{end+1}=entry;
            error('Capture:CallbackException','Intentional second ordinary callback exception');
        end
        log{end+1}=entry;
    end
end
function s=spec(name,route,span,refine,selection,mode,event,restart)
s=struct('name',name,'route',route,'span',span,'refine',refine, ...
    'selection',selection,'mode',mode,'event',event,'restart',restart);
end
function dy=rhs(~,y)
dy=[-y(1);2*y(2)];
end
function [v,terminal,direction]=terminalEvent(t,~)
v=t-.075;terminal=1;direction=0;
end
function out=striphandles(sol)
out=sol;
if isstruct(out) && isfield(out,'extdata'),out=rmfield(out,'extdata');end
end
function writejson(path,value)
f=fopen(path,'w');assert(f>=0,'Capture:Write','Cannot open output');
cleanup=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value));
end

function verifySources(manifest)
for i=1:numel(manifest.sources)
    item=manifest.sources(i);
    [rc,result]=system(['sha256sum "' item.path '"']);
    assert(rc==0 && startsWith(result,item.sha256), ...
        'Capture:SourceHash','Native source changed: %s',item.path);
end
end
