addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref');
seedRNG(7681);
xr = 2 * rand(1000, 1) - 1;
save('/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/feval_finite_source_20261013/native/xr.mat', 'xr', '-v7');
