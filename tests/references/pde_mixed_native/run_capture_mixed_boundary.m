function run_capture_mixed_boundary()
% Call serially inside the parent's guarded CPU MATLAB session.
base=fileparts(mfilename('fullpath'));
addpath(base);
assert(strcmp(which('capture_mixed_boundary'),fullfile(base,'capture_mixed_boundary.m')), ...
    'Capture:Binding','Unexpected harness binding');
% Static messages are observable; main capture still fails on a parser error.
issues=checkcode(fullfile(base,'capture_mixed_boundary.m'),'-id');
disp(issues);
capture_mixed_boundary(fullfile(base,'output'));
end
