# Notes 
Should use as much shared as possible 

defaults: 
- numba thread count 
- maxiter
- tol


# Scaling experiments: 
defaults: 
- communicator: Custom only 
- numba thread count: 1 


- Processors: 1 to 100: spread nicely for log-log plots: use counts that are suitable for cubic decomposition (doesn't need to be perfect cubes: but nice ish ones) 

Solvers: 
## baselines: 
- Jacobi sequential 
- FMG sequential
## MPI 
- jacobimpi 
- fmgmpi 

## hybrid: using 4 numba threads and avoiding oversubscribtion 

Problem sizes: 
- strong scaling: 250, 500 ish 



# Strong scaling 
# Weak scaling 


# Communication experiment 
solver: jacobiMPI
separate jobscript for binding optimization and no optimization 
sweep over problem size, decomposition type and communicator (numpy vs custom)








