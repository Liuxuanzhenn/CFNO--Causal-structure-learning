import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman and increase size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16  # Increased font size

# --- Data ---
# Nodes are the same for both datasets
nodes = [6, 8, 10, 12, 15, 20]

# Dataset 1: VAR (Linear)
data_var = {
    'title': 'Linear VAR Datasets',
    'CFNO-W': {
        'P': [0.80, 0.67, 0.58, 0.53, 0.54, 0.39],
        'R': [0.80, 0.71, 0.71, 0.75, 0.72, 0.70],
        'F': [0.80, 0.69, 0.64, 0.62, 0.62, 0.50]
    },
    'CFNO-P': {
        'P': [0.83, 0.93, 0.90, 0.67, 0.60, 0.46],
        'R': [0.91, 0.76, 0.72, 0.85, 0.82, 0.82],
        'F': [0.87, 0.84, 0.80, 0.75, 0.69, 0.59]
    }
}

# Dataset 2: Nonlinear
data_nonlinear = {
    'title': 'Highly Nonlinear Datasets',
    'CFNO-W': {
        'P': [0.67, 0.63, 0.56, 0.53, 0.46, 0.44],
        'R': [0.83, 0.77, 0.83, 0.70, 0.64, 0.64],
        'F': [0.74, 0.69, 0.67, 0.60, 0.54, 0.52]
    },
    'CFNO-P': {
        'P': [0.91, 0.86, 0.80, 0.73, 0.71, 0.72],
        'R': [0.77, 0.75, 0.74, 0.76, 0.63, 0.47],
        'F': [0.83, 0.80, 0.77, 0.74, 0.67, 0.57]
    }
}

datasets = [data_var, data_nonlinear]
metrics = ['P', 'R', 'F']
metric_names = ['Precision', 'Recall', 'F1-score']

# --- Plotting ---
# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(24, 14), sharex=True, sharey='col')

# Darker colors inspired by Node2 (cyan) and Node5 (yellow)
color_w = '#7FD17F'  # Deeper mint green - slightly darker version
color_p = '#C084FC'  # Deeper lavender purple - slightly darker version

# Loop through datasets (rows) and metrics (columns)
for row, dataset in enumerate(datasets):
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        
        # Plot CFNO-W data
        ax.plot(nodes, dataset['CFNO-W'][metric], marker='o', linestyle='-', color=color_w, label='CFNO-W', linewidth=6, markersize=14)
        
        # Plot CFNO-P data
        ax.plot(nodes, dataset['CFNO-P'][metric], marker='s', linestyle='--', color=color_p, label='CFNO-P', linewidth=6, markersize=14)

        # --- Customization ---
        # Set titles only for the top row of subplots
        if row == 0:
            ax.set_title(metric_names[col], fontsize=20, weight='bold')

        # Set common X-axis label only for the bottom row
        if row == 1:
            ax.set_xlabel('Number of Nodes', fontsize=18, weight='bold')
        
        # Set Y-axis labels on the first column to act as row titles
        if col == 0:
            # We add a dummy y-label to the plot, then set it to be the title for the row
            ax.set_ylabel(dataset['title'], fontsize=20, weight='bold', labelpad=20)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=14)
        ax.set_xticks(nodes)
        ax.tick_params(axis='both', which='major', labelsize=14)


# Adjust layout and save the figure as SVG
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('performance_comparison.svg', format='svg')

print("Plot saved as performance_comparison.svg") 