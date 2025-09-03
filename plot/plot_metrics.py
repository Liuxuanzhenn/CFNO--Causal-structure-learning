import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import Akima1DInterpolator

# Data from the tables
lag = [1, 2, 3, 4, 5]

f1_scores = {
    'CFNO-W': [0.69, 0.65, 0.63, 0.63, 0.61],
    'CFNO-P': [0.70, 0.69, 0.73, 0.71, 0.67],
    'eSRU':   [0.77, 0.79, 0.68, 0.56, 0.56],
    'CUTS+':  [0.64, 0.60, 0.51, 0.52, 0.50]
}

shd_scores = {
    'CFNO-W': [18, 27, 25, 26, 29],
    'CFNO-P': [22, 21, 21, 23, 27],
    'eSRU':   [13, 10, 20, 25, 25],
    'CUTS+':  [31, 34, 52, 50, 57]
}

# Create a DataFrame
df_f1 = pd.DataFrame(f1_scores, index=lag)
df_shd = pd.DataFrame(shd_scores, index=lag)

# --- Enhanced Plotting Styles ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 20,
})

colors = sns.color_palette("viridis", 4)
style_map = {
    'CFNO-W':   {'color': colors[0], 'marker': 'o', 'linestyle': '-'},
    'CFNO-P':   {'color': colors[1], 'marker': 's', 'linestyle': '--'},
    'eSRU':     {'color': colors[2], 'marker': '^', 'linestyle': ':'},
    'CUTS+':    {'color': colors[3], 'marker': 'D', 'linestyle': '-.'}
}


# --- Create a single figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot F1-scores on the first subplot (ax1) ---
for method in df_f1.columns:
    styles = style_map.get(method, {})
    x_original = df_f1.index
    y_original = df_f1[method]

    # Create a moderately smoothed line using Akima interpolator
    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
    akima_interp = Akima1DInterpolator(x_original, y_original)
    y_smooth = akima_interp(x_smooth)

    # Plot the smoothed line
    ax1.plot(x_smooth, y_smooth,
             color=styles.get('color'),
             linestyle=styles.get('linestyle'),
             label=method,
             linewidth=2.5,
             zorder=5) # Place line below markers

    # Plot original data points as markers on top
    ax1.scatter(x_original, y_original,
                color=styles.get('color'),
                marker=styles.get('marker'),
                s=90,
                zorder=10,
                edgecolors='white',
                linewidth=1.5)

ax1.set_xlabel('Time Lag', fontsize=14)
ax1.set_ylabel('F1-score', fontsize=14)
ax1.legend(loc='lower left', frameon=True, shadow=True, fancybox=True)
ax1.grid(False)
ax1.set_xticks(df_f1.index)

# --- Plot SHD on the second subplot (ax2) ---
for method in df_shd.columns:
    styles = style_map.get(method, {})
    x_original = df_shd.index
    y_original = df_shd[method]
    
    # Create a moderately smoothed line using Akima interpolator
    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
    akima_interp = Akima1DInterpolator(x_original, y_original)
    y_smooth = akima_interp(x_smooth)
    
    # Plot the smoothed line
    ax2.plot(x_smooth, y_smooth,
             color=styles.get('color'),
             linestyle=styles.get('linestyle'),
             label=method,
             linewidth=2.5,
             zorder=5)
    
    # Plot original data points as markers on top
    ax2.scatter(x_original, y_original,
                color=styles.get('color'),
                marker=styles.get('marker'),
                s=90,
                zorder=10,
                edgecolors='white',
                linewidth=1.5)

ax2.set_xlabel('Time Lag', fontsize=14)
ax2.set_ylabel('SHD', fontsize=14)
ax2.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
ax2.grid(False)
ax2.set_xticks(df_shd.index)

# --- Finalize and save the combined plot ---
plt.tight_layout(pad=2.0)
plt.savefig('lag_effect_plot_styled.png', dpi=300, bbox_inches='tight')
plt.show()

print("Styled combined plot saved as lag_effect_plot_styled.png") 