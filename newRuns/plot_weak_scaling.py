import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling.csv')

# Rename columns for clarity in plots
df = df.rename(columns={
    'solver': 'Solver',
    'strategy': 'Decomposition',
    'communicator': 'Datatype',
    'n_ranks': 'Ranks',
    'mlups': 'MLUPS'
})

# Capitalize values for better legend labels
df['Solver'] = df['Solver'].str.upper()
df['Decomposition'] = df['Decomposition'].str.capitalize()
df['Datatype'] = df['Datatype'].str.capitalize()

# Create a combined label for hue
df['Config'] = df['Solver'] + ' / ' + df['Decomposition'] + ' / ' + df['Datatype']

# Set the style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Aggregate data by taking mean for duplicate configurations
df_agg = df.groupby(['Ranks', 'Solver', 'Decomposition', 'Datatype', 'Config']).agg({
    'MLUPS': 'mean'
}).reset_index()

# Plot 1: Full faceted view - Solver x Decomposition with Datatype as hue
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, solver in enumerate(['JACOBI', 'FMG']):
    for j, decomp in enumerate(['Cubic', 'Sliced']):
        ax = axes[i, j]
        subset = df_agg[(df_agg['Solver'] == solver) & (df_agg['Decomposition'] == decomp)]

        for datatype, marker in [('Numpy', 'o'), ('Custom', 's')]:
            data = subset[subset['Datatype'] == datatype].sort_values('Ranks')
            if len(data) > 0:
                ax.plot(data['Ranks'], data['MLUPS'], marker=marker, markersize=8,
                       linewidth=2, label=datatype)

        ax.set_xlabel('Number of Ranks', fontsize=11)
        ax.set_ylabel('MLUPS', fontsize=11)
        ax.set_title(f'{solver} - {decomp}', fontsize=12, fontweight='bold')
        ax.legend(title='Datatype')
        ax.grid(True, alpha=0.3)

plt.suptitle('Weak Scaling: MLUPS vs Ranks\n(by Solver, Decomposition, and Datatype)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_faceted.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Single plot with all combinations
fig, ax = plt.subplots(figsize=(14, 8))

# Use different line styles and markers for each combination
styles = {
    ('JACOBI', 'Cubic', 'Numpy'): {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue'},
    ('JACOBI', 'Cubic', 'Custom'): {'marker': 's', 'linestyle': '-', 'color': 'tab:orange'},
    ('JACOBI', 'Sliced', 'Numpy'): {'marker': 'o', 'linestyle': '--', 'color': 'tab:blue'},
    ('JACOBI', 'Sliced', 'Custom'): {'marker': 's', 'linestyle': '--', 'color': 'tab:orange'},
    ('FMG', 'Cubic', 'Numpy'): {'marker': '^', 'linestyle': '-', 'color': 'tab:green'},
    ('FMG', 'Cubic', 'Custom'): {'marker': 'D', 'linestyle': '-', 'color': 'tab:red'},
    ('FMG', 'Sliced', 'Numpy'): {'marker': '^', 'linestyle': '--', 'color': 'tab:green'},
    ('FMG', 'Sliced', 'Custom'): {'marker': 'D', 'linestyle': '--', 'color': 'tab:red'},
}

for (solver, decomp, datatype), style in styles.items():
    subset = df_agg[(df_agg['Solver'] == solver) &
                    (df_agg['Decomposition'] == decomp) &
                    (df_agg['Datatype'] == datatype)].sort_values('Ranks')
    if len(subset) > 0:
        ax.plot(subset['Ranks'], subset['MLUPS'],
               marker=style['marker'], linestyle=style['linestyle'],
               color=style['color'], markersize=8, linewidth=2,
               label=f'{solver} / {decomp} / {datatype}')

ax.set_xlabel('Number of Ranks', fontsize=12)
ax.set_ylabel('MLUPS (Million Lattice Updates Per Second)', fontsize=12)
ax.set_title('Weak Scaling Performance\n(Solver / Decomposition / Datatype)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_combined.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Seaborn relplot for elegant faceting
g = sns.relplot(
    data=df_agg,
    x='Ranks', y='MLUPS',
    hue='Datatype', style='Solver',
    col='Decomposition',
    kind='line',
    markers=True,
    markersize=10,
    height=5, aspect=1.2
)
g.fig.suptitle('Weak Scaling: MLUPS vs Ranks', fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Number of Ranks', 'MLUPS')
plt.tight_layout()
plt.savefig('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_seaborn_relplot.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Ideal scaling comparison (normalized)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, solver in enumerate(['JACOBI', 'FMG']):
    ax = axes[idx]
    subset = df_agg[df_agg['Solver'] == solver]

    # Get baseline (1 rank) for each configuration
    for decomp in ['Cubic', 'Sliced']:
        for datatype in ['Numpy', 'Custom']:
            config_data = subset[(subset['Decomposition'] == decomp) &
                                (subset['Datatype'] == datatype)].sort_values('Ranks')
            if len(config_data) > 0:
                baseline = config_data[config_data['Ranks'] == config_data['Ranks'].min()]['MLUPS'].values[0]
                base_ranks = config_data['Ranks'].min()

                # Normalize by ideal scaling
                config_data = config_data.copy()
                config_data['Efficiency'] = config_data['MLUPS'] / (baseline * config_data['Ranks'] / base_ranks) * 100

                linestyle = '-' if decomp == 'Cubic' else '--'
                color = 'tab:blue' if datatype == 'Numpy' else 'tab:orange'
                marker = 'o' if datatype == 'Numpy' else 's'

                ax.plot(config_data['Ranks'], config_data['Efficiency'],
                       marker=marker, linestyle=linestyle, color=color,
                       markersize=8, linewidth=2,
                       label=f'{decomp} / {datatype}')

    ax.axhline(y=100, color='gray', linestyle=':', linewidth=2, label='Ideal (100%)')
    ax.set_xlabel('Number of Ranks', fontsize=12)
    ax.set_ylabel('Weak Scaling Efficiency (%)', fontsize=12)
    ax.set_title(f'{solver} Solver', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 120)

plt.suptitle('Weak Scaling Efficiency\n(Relative to Ideal Linear Scaling)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_efficiency.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plots saved:")
print("  1. weak_scaling_faceted.png - 2x2 faceted view by solver and decomposition")
print("  2. weak_scaling_combined.png - All configurations in one plot")
print("  3. weak_scaling_seaborn_relplot.png - Seaborn relplot with faceting")
print("  4. weak_scaling_efficiency.png - Weak scaling efficiency comparison")

# Summary statistics
print("\n--- Data Summary ---")
print(f"Total runs: {len(df)}")
print(f"\nSolvers: {df['Solver'].unique()}")
print(f"Decompositions: {df['Decomposition'].unique()}")
print(f"Datatypes: {df['Datatype'].unique()}")
print(f"Rank counts: {sorted(df['Ranks'].unique())}")
print(f"\nMLUPS range: {df['MLUPS'].min():.2f} - {df['MLUPS'].max():.2f}")
