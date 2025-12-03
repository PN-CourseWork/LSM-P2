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



# From report: 


\subsection{Scaling analysis}
\label{sub:scaling_results}
Experiment~05 measures strong and weak scaling of the
validated configuration.
Present speedup and efficiency for fixed problem sizes,
weak-scaling efficiency for constant work per rank,
memory footprint per rank and globally as
ranks/problem size grow,
and---if measured---the impact of parallel I/O
versus gather-to-rank-0.
% \begin{figure}
%   \centering
%   \includegraphics[width=\linewidth]{figures/<scaling-figure>.pdf}
%   \caption{Strong and weak scaling efficiency for the
%            validated solver configuration.}
%   \label{fig:scaling-efficiency}
% \end{figure}

\subsection{Communicator: Rank Placement/Mappings}
%TODO: 
Metric: Mlups, speedup 


dims: 

Plots: 
- Mlups vs. problem size (col=mapping)
- Halo exchange time (wall time/pr. iter) vs. problem size
    - loglog plot 
(More or less like Communication experiment - but with inter/intra node comparison) 

legend variation: 
- communicator (numpy, custom)
sweep: problem size, mapping  
const: Ranks 
Total: 40 runs ish


Conclude which COMM and SPREAD to keep using 


\subsection{Decomposition}

Metric: Speedup, Parallel efficiency

Plots: 

Metric: Speedup/parallel efficiency    

    - Strong Scaling 
        - legend: Decomposition
        - constant: Comm, Global Problem size, mapping
        - sweep: Ranks, decomp
        
    - Weak Scaling 
        - Legend: Decomposition 
        - constant: Comm, Local Problem size, mapping
        - sweep: ranks, decomp
        
Metric: Mlups 

\subsection{Full Multigrid}


Hybrid: 
const: Comm, Decomposition, spread
sweep: Ranks, Numba threads (single, 4, 8?)

Plot: 
Legend: Numba threads 
x-axis: Number of Processes (MPI ranks * numba threads)

Strong and weak scaling plots


