"""
Plot FMG Spatial Convergence
============================

Load FMG validation results and plot spatial convergence
against the analytical solution. Run compute_multigrid_fmg.py first.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from Poisson import get_project_root

# Setup
sns.set_style()

repo_root = get_project_root()
data_dir = repo_root / "data" / "multigrid_fmg"
fig_dir = repo_root / "figures" / "multigrid"
fig_dir.mkdir(parents=True, exist_ok=True)

h5_files = list(data_dir.glob("*.h5"))
if not h5_files:
    raise FileNotFoundError(
        f"No data found in {data_dir}. Run compute_multigrid_fmg.py first."
    )

df = pd.concat([pd.read_hdf(f, key="results") for f in h5_files], ignore_index=True)
df["Communicator"] = (
    df["communicator"]
    .str.replace("haloexchange", "", regex=False)
    .str.replace("custom", "Custom", regex=False)
    .str.replace("numpy", "NumPy", regex=False)
)

fig, ax = plt.subplots(figsize=(8, 6))

sns.lineplot(
    data=df,
    x="N",
    y="final_error",
    hue="Communicator",
    style="Communicator",
    markers=True,
    dashes=True,
    ax=ax,
)

# Reference O(N^-2)
N_min, N_max = df["N"].min(), df["N"].max()
ax.plot(
    [N_min, N_max],
    [df["final_error"].max(), df["final_error"].max() * (N_min / N_max) ** 2],
    "k:",
    alpha=0.6,
    label=r"$O(N^{-2})$",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlabel("Grid Size N")
ax.set_ylabel("L2 Error")
ax.set_title("Spatial Convergence: FMG")
ax.legend()

fig.tight_layout()
output_file = fig_dir / "fmg_convergence.pdf"
fig.savefig(output_file)
print(f"Saved: {output_file}")
