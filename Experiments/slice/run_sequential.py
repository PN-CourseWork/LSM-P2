#!/usr/bin/env python3
"""
Quick script to run sequential solver and save output for comparison
"""
import numpy as np
import math
from time import perf_counter as time

N = 100
N_iter = 100
omega = 0.75
tolerance = 1e-8

# Allocate
h = 2.0 / (N - 1)
u1 = np.zeros([N, N, N], dtype=np.float64)
u2 = u1.copy()

# Initialize f
xs, ys, zs = np.ogrid[-1:1:complex(N), -1:1:complex(N), -1:1:complex(N)]
f = 3 * (math.pi ** 2) * np.sin(math.pi * xs) * np.sin(math.pi * ys) * np.sin(math.pi * zs)

# True solution for validation
u_true = np.sin(math.pi * xs) * np.sin(math.pi * ys) * np.sin(math.pi * zs)

print("Running sequential Jacobi...")
c = 1.0 / 6.0
h2 = h * h

for i in range(N_iter):
    if i % 2 == 0:
        uold, u = u1, u2
    else:
        u, uold = u1, u2
    
    u[1:-1, 1:-1, 1:-1] = omega * c * (
        uold[0:-2, 1:-1, 1:-1] + uold[2:, 1:-1, 1:-1] +
        uold[1:-1, 0:-2, 1:-1] + uold[1:-1, 2:, 1:-1] +
        uold[1:-1, 1:-1, 0:-2] + uold[1:-1, 1:-1, 2:] +
        h2 * f[1:-1, 1:-1, 1:-1]
    )
    
    diff = math.sqrt(np.sum((u - uold) ** 2)) / (N ** 3)
    
    if diff < tolerance:
        print(f"Converged at iteration {i+1}")
        break

print(f"Final residual: {diff:.6e}")

# Validate
diff_true = math.sqrt(np.sum((u - u_true) ** 2)) / (N ** 3)
print(f"Error vs true solution: {diff_true:.6e}")

# Save
np.save("./Experiments/slice/output/u_sequential.npy", u)
print("Saved to ./Experiments/slice/output/u_sequential.npy")