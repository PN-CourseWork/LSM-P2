from mpi4py import MPI
from argparse import ArgumentParser
import math
from time import perf_counter as time
import numpy as np


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# argument-parser
parser = ArgumentParser(description="Poisson problem - Sliced Decomposition")

parser.add_argument(
    "-N",
    type=int,
    default=100,
    help="Number of divisions along each of the 3 dimensions",
)
parser.add_argument("--iter", type=int, default=20, help="Number of (max) iterations.")
parser.add_argument("-v0", "--value0", type=float, default=0., help="The initial value of the grid u")
parser.add_argument(
    "--tolerance",
    type=float,
    default=1e-8,
    help="The tolerance of the normalized Frobenius norm of the residual for the convergence.",
)
parser.add_argument(
    "--save-slice",
    nargs=3,
    metavar=["axis", "pos", "FILE"],
    default=None,
    help="Store an image of a slice (pos in [-1;1])",
)
parser.add_argument(
    "--axis",
    choices=['x', 'y', 'z'],
    default='z',
    help="Axis along which to slice the domain (x=0, y=1, z=2)",
)
parser.add_argument(
    "--omega",
    type=float,
    default=0.75,
    help="Relaxation parameter for weighted Jacobi",
)
parser.add_argument(
    "--output",
    type=str,
    default='output/u_final.npy',
    help="Output filename for assembled solution (rank 0)",
)

methods = ["view"]
parser.add_argument(
    "--method",
    choices=methods,
    default=methods[0],
    help="The chosen method to solve the Poisson problem.",
)

# Parse options
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance
omega: float = options.omega

# Map axis to dimension
axis_map = {'x': 0, 'y': 1, 'z': 2}
slice_dim = axis_map[options.axis]


def split_sizes(n, parts):
    """Split n points into parts as evenly as possible."""
    base = n // parts
    rem = n % parts
    counts = [base + (1 if i < rem else 0) for i in range(parts)]
    starts = [sum(counts[:i]) for i in range(parts)]
    return counts, starts


# Domain decomposition
counts, starts = split_sizes(N, size)
nlocal = counts[rank]
start_idx = starts[rank]

# Create local array shape based on slice dimension
# Ghost layers: +2 in the sliced dimension
if slice_dim == 2:  # Z-axis
    local_shape = (nlocal + 2, N, N)
elif slice_dim == 1:  # Y-axis
    local_shape = (N, nlocal + 2, N)
else:  # X-axis (dim==0)
    local_shape = (N, N, nlocal + 2)

# Allocate local arrays
h: float = 2.0 / (N - 1)
u_local = np.full(local_shape, options.value0, dtype=np.float64)
u_old = u_local.copy()
u_new = u_local.copy()
f_local = np.zeros_like(u_local)

# Initialize f on local interior
# Compute global coordinates for local points
if slice_dim == 2:  # Z
    global_indices = np.arange(start_idx, start_idx + nlocal)
    coords = -1.0 + global_indices * h
    Z = coords.reshape((nlocal, 1, 1))
    Y_grid, X_grid = np.ogrid[-1:1:complex(N), -1:1:complex(N)]
    Y = Y_grid.reshape((1, N, 1))
    X = X_grid.reshape((1, 1, N))
    f_local[1:-1, :, :] = 3 * (math.pi ** 2) * np.sin(math.pi * X) * np.sin(math.pi * Y) * np.sin(math.pi * Z)
elif slice_dim == 1:  # Y
    global_indices = np.arange(start_idx, start_idx + nlocal)
    coords = -1.0 + global_indices * h
    Y = coords.reshape((1, nlocal, 1))
    Z_grid, X_grid = np.ogrid[-1:1:complex(N), -1:1:complex(N)]
    Z = Z_grid.reshape((N, 1, 1))
    X = X_grid.reshape((1, 1, N))
    f_local[:, 1:-1, :] = 3 * (math.pi ** 2) * np.sin(math.pi * X) * np.sin(math.pi * Y) * np.sin(math.pi * Z)
else:  # X (dim==0)
    global_indices = np.arange(start_idx, start_idx + nlocal)
    coords = -1.0 + global_indices * h
    X = coords.reshape((1, 1, nlocal))
    Z_grid, Y_grid = np.ogrid[-1:1:complex(N), -1:1:complex(N)]
    Z = Z_grid.reshape((N, 1, 1))
    Y = Y_grid.reshape((1, N, 1))
    f_local[:, :, 1:-1] = 3 * (math.pi ** 2) * np.sin(math.pi * X) * np.sin(math.pi * Y) * np.sin(math.pi * Z)

# Set boundary conditions on ghost layers (will be updated during exchange)
# Create 1D Cartesian topology
cart = comm.Create_cart(dims=[size], periods=[False], reorder=False)
source, dest = cart.Shift(0, 1)


def exchange_ghosts(u, slice_dim, cart, source, dest):
    """Exchange ghost layers with neighbors along the sliced dimension."""
    
    if slice_dim == 2:  # Z-axis slicing
        # Send top boundary, receive bottom ghost
        if dest != MPI.PROC_NULL or source != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[-2, :, :])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=dest,
                         recvbuf=recvbuf, source=source)
            if source != MPI.PROC_NULL:
                u[0, :, :] = recvbuf
            else:
                u[0, :, :] = 0.0
        
        # Send bottom boundary, receive top ghost
        if source != MPI.PROC_NULL or dest != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[1, :, :])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=source,
                         recvbuf=recvbuf, source=dest)
            if dest != MPI.PROC_NULL:
                u[-1, :, :] = recvbuf
            else:
                u[-1, :, :] = 0.0
                
    elif slice_dim == 1:  # Y-axis slicing
        # Send top boundary, receive bottom ghost
        if dest != MPI.PROC_NULL or source != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[:, -2, :])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=dest,
                         recvbuf=recvbuf, source=source)
            if source != MPI.PROC_NULL:
                u[:, 0, :] = recvbuf
            else:
                u[:, 0, :] = 0.0
        
        # Send bottom boundary, receive top ghost
        if source != MPI.PROC_NULL or dest != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[:, 1, :])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=source,
                         recvbuf=recvbuf, source=dest)
            if dest != MPI.PROC_NULL:
                u[:, -1, :] = recvbuf
            else:
                u[:, -1, :] = 0.0
                
    else:  # X-axis slicing (dim==0)
        # Send top boundary, receive bottom ghost
        if dest != MPI.PROC_NULL or source != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[:, :, -2])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=dest,
                         recvbuf=recvbuf, source=source)
            if source != MPI.PROC_NULL:
                u[:, :, 0] = recvbuf
            else:
                u[:, :, 0] = 0.0
        
        # Send bottom boundary, receive top ghost
        if source != MPI.PROC_NULL or dest != MPI.PROC_NULL:
            sendbuf = np.ascontiguousarray(u[:, :, 1])
            recvbuf = np.empty_like(sendbuf)
            cart.Sendrecv(sendbuf=sendbuf, dest=source,
                         recvbuf=recvbuf, source=dest)
            if dest != MPI.PROC_NULL:
                u[:, :, -1] = recvbuf
            else:
                u[:, :, -1] = 0.0


def step_view_local(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float, slice_dim: int):
    """Run a single Poisson step on local domain."""
    c: float = 1.0 / 6.0
    h2: float = h * h
    
    if slice_dim == 2:  # Z-axis
        # Only update interior points (exclude ghost layers in Z and boundary in Y,X)
        u[1:-1, 1:-1, 1:-1] = omega * c * (
            uold[0:-2, 1:-1, 1:-1] + uold[2:, 1:-1, 1:-1] +
            uold[1:-1, 0:-2, 1:-1] + uold[1:-1, 2:, 1:-1] +
            uold[1:-1, 1:-1, 0:-2] + uold[1:-1, 1:-1, 2:] +
            h2 * f[1:-1, 1:-1, 1:-1]
        )
        # Keep boundary conditions (already set to 0)
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
        
    elif slice_dim == 1:  # Y-axis
        # Only update interior points (exclude ghost layers in Y and boundary in Z,X)
        u[1:-1, 1:-1, 1:-1] = omega * c * (
            uold[0:-2, 1:-1, 1:-1] + uold[2:, 1:-1, 1:-1] +
            uold[1:-1, 0:-2, 1:-1] + uold[1:-1, 2:, 1:-1] +
            uold[1:-1, 1:-1, 0:-2] + uold[1:-1, 1:-1, 2:] +
            h2 * f[1:-1, 1:-1, 1:-1]
        )
        # Keep boundary conditions
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
        
    else:  # X-axis (dim==0)
        # Only update interior points (exclude ghost layers in X and boundary in Z,Y)
        u[1:-1, 1:-1, 1:-1] = omega * c * (
            uold[0:-2, 1:-1, 1:-1] + uold[2:, 1:-1, 1:-1] +
            uold[1:-1, 0:-2, 1:-1] + uold[1:-1, 2:, 1:-1] +
            uold[1:-1, 1:-1, 0:-2] + uold[1:-1, 1:-1, 2:] +
            h2 * f[1:-1, 1:-1, 1:-1]
        )
        # Keep boundary conditions
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0


# Initial exchange
exchange_ghosts(u_old, slice_dim, cart, source, dest)

# Main iteration loop
for i in range(N_iter):
    # Swap pointers
    if i % 2 == 0:
        uold = u_old
        u = u_new
    else:
        u = u_old
        uold = u_new
    
    # Exchange ghost layers
    exchange_ghosts(uold, slice_dim, cart, source, dest)
    
    # Local update
    step_view_local(uold, u, f_local, h, omega, slice_dim)
    
    # Compute local residual (only interior points)
    local_sum_sq = np.sum((u[1:-1, 1:-1, 1:-1] - uold[1:-1, 1:-1, 1:-1]) ** 2)
    
    # Global residual
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    diff = math.sqrt(global_sum_sq) / (N ** 3)
    
    if diff < tolerance:
        if rank == 0:
            print(f"Converged at iteration {i+1}")
        break

if rank == 0:
    print(f"Final residual: {diff}")
    print(f"Iterations run: {i+1}")

# Gather result to rank 0
if slice_dim == 2:  # Z-axis
    local_interior = u[1:-1, :, :].copy()
elif slice_dim == 1:  # Y-axis
    local_interior = u[:, 1:-1, :].copy()
else:  # X-axis
    local_interior = u[:, :, 1:-1].copy()

if rank == 0:
    u_global = np.zeros((N, N, N), dtype=np.float64)
    
    # Place own data
    if slice_dim == 2:
        u_global[start_idx:start_idx+nlocal, :, :] = local_interior
    elif slice_dim == 1:
        u_global[:, start_idx:start_idx+nlocal, :] = local_interior
    else:
        u_global[:, :, start_idx:start_idx+nlocal] = local_interior
    
    # Receive from other ranks
    for r in range(1, size):
        r_count = counts[r]
        r_start = starts[r]
        
        if slice_dim == 2:
            buf = np.empty((r_count, N, N), dtype=np.float64)
        elif slice_dim == 1:
            buf = np.empty((N, r_count, N), dtype=np.float64)
        else:
            buf = np.empty((N, N, r_count), dtype=np.float64)
        
        comm.Recv(buf, source=r, tag=99)
        
        if slice_dim == 2:
            u_global[r_start:r_start+r_count, :, :] = buf
        elif slice_dim == 1:
            u_global[:, r_start:r_start+r_count, :] = buf
        else:
            u_global[:, :, r_start:r_start+r_count] = buf
    
    # Save result
    np.save(options.output, u_global)
    print(f"Saved assembled solution to {options.output}")
else:
    comm.Send(local_interior, dest=0, tag=99)

comm.Barrier()
if rank == 0:
    print("Done.")