function value=capture_condest(J)
global PDE_CAPTURE
before=rng;
value=condest(J);
after=rng;
PDE_CAPTURE.condest=struct('J',J,'value',value,'rng_before',before,'rng_after',after);
end
