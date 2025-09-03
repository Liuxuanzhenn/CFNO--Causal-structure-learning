
import matplotlib.pyplot as plt
import numpy as np

# Use a professional plotting style and set the font
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'


# --- Data Extracted from the provided Table 1 ---
nodes = [6, 8, 10, 12, 15, 20]
methods = ['CFNO-P', 'CUTS+', 'eSRU', 'CSL-HNTS']

# Precision (P) scores from the table
precision_scores = {
    'CFNO-P': [0.83, 0.93, 0.90, 0.67, 0.60, 0.46],
    'CUTS+':  [0.82, 0.61, 0.60, 0.54, 0.54, 0.53],
    'eSRU':   [0.83, 0.81, 0.82, 0.82, 0.41, 0.32],
    'CSL-HNTS': [0.82, 0.77, 0.55, 0.46, 0.43, 0.35]
}

# Recall (R) scores from the table
recall_scores = {
    'CFNO-P': [0.91, 0.76, 0.72, 0.85, 0.82, 0.82],
    'CUTS+':  [0.86, 0.93, 0.89, 0.86, 0.66, 0.40],
    'eSRU':   [0.67, 0.62, 0.50, 0.41, 0.88, 0.82],
    'CSL-HNTS': [0.60, 0.48, 0.55, 0.46, 0.38, 0.26]
}


# --- Plotting Style (adopted from F1andSHDplotVAR.py) ---
colors = {
    'CFNO-P': '#d62728',  # Red
    'CUTS+': '#1f77b4',   # Blue
    'eSRU': '#7f7f7f',   # Grey
    'CSL-HNTS': '#aec7e8' # Light Blue
}

markers = {
    'CFNO-P': 'o',
    'CUTS+': 's',
    'eSRU': '^',
    'CSL-HNTS': 'D'
}

# --- Create the plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Performance Analysis on Linear VAR Datasets', fontsize=20, y=1.03)


# Plot 1: Precision
for method in methods:
    ax1.plot(nodes, precision_scores[method], marker=markers[method], linestyle='-', label=method, color=colors[method], linewidth=2.5, markersize=9)

ax1.set_title('Precision On Linear VAR datasets: default lag=2 default T=1000', fontsize=16)
ax1.set_xlabel('Number of Nodes', fontsize=14)
ax1.set_ylabel('Precision', fontsize=14)
ax1.set_xticks(nodes)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_ylim(0, 1.05)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(fontsize=14)


# Plot 2: Recall
for method in methods:
    ax2.plot(nodes, recall_scores[method], marker=markers[method], linestyle='-', label=method, color=colors[method], linewidth=2.5, markersize=9)

ax2.set_title('Recall On Linear VAR datasets: default lag=2 default T=1000', fontsize=16)
ax2.set_xlabel('Number of Nodes', fontsize=14)
ax2.set_ylabel('Recall', fontsize=14)
ax2.set_xticks(nodes)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_ylim(0, 1.05)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(fontsize=14)


# --- Final Touches and Saving ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('precision_recall_scores_VAR.svg', format='svg', dpi=600, bbox_inches='tight')

plt.show()

