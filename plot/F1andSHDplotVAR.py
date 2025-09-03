
import matplotlib.pyplot as plt
import numpy as np

# Use a professional plotting style and set the font
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'

# --- Data updated from the new table (vs. Lag) ---
lags = [1, 2, 3, 4, 5]
methods = ['CFNO-P', 'CSL-HNTS', 'eSRU', 'CUTS+']

f1_scores = {
    'CFNO-P':   [0.80, 0.80, 0.79, 0.78, 0.77],
    'CSL-HNTS': [0.63, 0.55, 0.46, 0.41, 0.39],
    'eSRU':     [0.77, 0.62, 0.68, 0.56, 0.56],
    'CUTS+':    [0.75, 0.71, 0.63, 0.52, 0.50]
}

shd_scores = {
    'CFNO-P':   [7, 9, 8, 10, 11],
    'CSL-HNTS': [19, 23, 34, 35, 38],
    'eSRU':     [13, 17, 20, 25, 25],
    'CUTS+':    [12, 14, 25, 31, 33]
}

# Color palette and markers
colors = {
    'CFNO-P': '#C084FC',  # Light Purple
    'CUTS+': '#1f77b4',   # Blue (Unchanged)
    'eSRU': '#E6B800',   # Light Gold
    'CSL-HNTS': '#FFA500'  # Light Orange
}
markers = {
    'CFNO-P': 'o',
    'CUTS+': 's',
    'eSRU': '^',
    'CSL-HNTS': 'D'
}

linestyles = {
    'CFNO-P': '--',
    'CUTS+': '-',
    'eSRU': '--',
    'CSL-HNTS': '-'
}

# --- Create the plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
# fig.suptitle('Performance Analysis with Varying Lag', fontsize=20, y=1.03, weight='bold')

# Plot 1: F1-scores vs. Lag
for method in methods:
    ax1.plot(lags, f1_scores[method], marker=markers[method], linestyle=linestyles[method], label=method, color=colors[method], linewidth=5, markersize=9)

# ax1.set_title('F1-score vs. Lag', fontsize=16, weight='bold')
ax1.set_xlabel('Lag', fontsize=14, weight='bold')
ax1.set_ylabel('F1-score', fontsize=14, weight='bold')
ax1.set_xticks(lags)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_ylim(0, 1.0)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
legend1 = ax1.legend(fontsize=14)
for text in legend1.get_texts():
    text.set_fontweight('bold')

# Plot 2: SHD scores vs. Lag
for method in methods:
    ax2.plot(lags, shd_scores[method], marker=markers[method], linestyle=linestyles[method], label=method, color=colors[method], linewidth=5, markersize=9)

# ax2.set_title('SHD vs. Lag', fontsize=16, weight='bold')
ax2.set_xlabel('Lag', fontsize=14, weight='bold')
ax2.set_ylabel('SHD', fontsize=14, weight='bold')
ax2.set_xticks(lags)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
legend2 = ax2.legend(fontsize=14)
for text in legend2.get_texts():
    text.set_fontweight('bold')

# --- Final Touches and Saving ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('F1_SHD_scores_vs_Lag.svg', format='svg', dpi=600, bbox_inches='tight')

plt.show() 