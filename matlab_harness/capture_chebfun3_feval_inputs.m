% Primitive RNG inputs only; no expected answers or coefficients.
% Pinned tests/chebfun3/test_feval.m, Chebfun commit 7574c77.
seedRNG(42);
inputs.pts = 2*rand(3,1)-1;
inputs.r = rand(10,1); inputs.s = rand(10,1); inputs.t = rand(10,1);
inputs.v16 = {rand(100,1), rand(100,1), rand(100,1)};
inputs.v17 = {rand(100,1), rand(100,1), rand(100,1)};
inputs.t28 = {rand(10,20,30), rand(10,20,30), rand(10,20,30)};
inputs.t29 = {rand(10,20,30), rand(10,20,30), rand(10,20,30)};
fid = fopen(getenv('CHEBFUN_CAPTURE_OUTPUT'), 'w');
fprintf(fid, '%s\n', jsonencode(inputs));
fclose(fid);
