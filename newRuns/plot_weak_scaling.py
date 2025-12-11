import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Load and combine both CSV files
df1 = pd.read_csv('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling.csv')
df2 = pd.read_csv('/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_large.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Calculate local grid size (N / cbrt(n_ranks) for cubic decomposition)
df['local_size_raw'] = df['N'] / np.cbrt(df['n_ranks'])

# Bin into approximate local size groups (within ~5% tolerance)
def bin_local_size(x):
    if 245 <= x <= 270:
        return 257
    elif 500 <= x <= 530:
        return 513
    else:
        return int(round(x))

df['local_size'] = df['local_size_raw'].apply(bin_local_size)

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

# Set the style
sns.set_theme()
plt.rcParams['text.usetex'] = True

# Aggregate data by taking mean for duplicate configurations
df_agg = df.groupby(['Ranks', 'Solver', 'Decomposition', 'Datatype', 'local_size']).agg({
    'MLUPS': 'mean'
}).reset_index()

# Get unique local sizes
local_sizes = sorted(df_agg['local_size'].unique())
print(f"Found local grid sizes: {local_sizes}")

# Create a separate plot for each local grid size
for local_size in local_sizes:
    df_local = df_agg[df_agg['local_size'] == local_size].copy()

    # Get rank ticks for this local size
    rank_ticks = sorted(df_local['Ranks'].unique())

    if len(rank_ticks) < 2:
        print(f"Skipping local_size={local_size} - only {len(rank_ticks)} rank value(s)")
        continue

    # Calculate efficiency for each configuration
    efficiency_data = []
    for solver in ['JACOBI', 'FMG']:
        for decomp in ['Cubic', 'Sliced']:
            for datatype in ['Numpy', 'Custom']:
                config_data = df_local[(df_local['Solver'] == solver) &
                                       (df_local['Decomposition'] == decomp) &
                                       (df_local['Datatype'] == datatype)].sort_values('Ranks')
                if len(config_data) > 1:
                    baseline = config_data[config_data['Ranks'] == config_data['Ranks'].min()]['MLUPS'].values[0]
                    base_ranks = config_data['Ranks'].min()
                    for _, row in config_data.iterrows():
                        efficiency_data.append({
                            'Ranks': row['Ranks'],
                            'Solver': solver,
                            'Decomposition': decomp,
                            'Datatype': datatype,
                            'Efficiency': row['MLUPS'] / (baseline * row['Ranks'] / base_ranks) * 100
                        })

    df_eff = pd.DataFrame(efficiency_data)

    # Use seaborn relplot
    g = sns.relplot(data=df_eff, x='Ranks', y='Efficiency', col='Solver',
                    hue='Decomposition', style='Datatype',
                    kind='line', markers=True, markersize=8, linewidth=2,
                    height=5, aspect=1.1, facet_kws={'legend_out': False})
    g.set_axis_labels(r'Number of Ranks', r'Weak Scaling Efficiency (\%)')

    # Filter ticks to avoid overlap (remove values too close together)
    filtered_ticks = []
    for t in rank_ticks:
        if not filtered_ticks or t > filtered_ticks[-1] * 1.15:  # At least 15% apart
            filtered_ticks.append(t)

    for ax in g.axes.flat:
        ax.set_xticks(filtered_ticks)
        ax.set_xticklabels([str(int(r)) for r in filtered_ticks])
        ax.set_ylim(0, 120)
        ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    filename = f'/Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-P2-PREP/newRuns/weak_scaling_efficiency_local{local_size}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: weak_scaling_efficiency_local{local_size}.pdf")

print("\nData summary:")
print(df_agg.groupby('local_size')[['Ranks']].apply(lambda x: sorted(x['Ranks'].unique())))
