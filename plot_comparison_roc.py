import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curves_on_axis(ax, data_dir, model_name, title, show_title=True):
    """
    Plots a full ROC subplot with all labels and titles on the given axes.
    Can optionally hide the title.
    """
    print(f"  -> Plotting subplot for {model_name}...")
    
    try:
        json_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.json')],
            key=lambda f: int(re.search(r'lag[（(]?(\d+)', f).group(1))
        )
    except (FileNotFoundError, AttributeError, TypeError):
        json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])

    colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    for i, filename in enumerate(json_files):
        match = re.search(r'lag[（(]?(\d+)', filename)
        lag_value = int(match.group(1)) if match else i + 1
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            label = f"{model_name}, lag={lag_value}, AUC={data['auroc']:.3f}"
            ax.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)], lw=2.5, label=label)

    ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='_nolegend_')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12, labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=12, labelpad=8)
    
    if show_title:
        ax.set_title(title, fontsize=14, pad=8)
        
    ax.tick_params(axis='both', which='major', labelsize=10)
    # The global rcParams setting now handles bolding, so this loop is not needed.
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_weight('bold')
    ax.legend(loc="lower right", fontsize=9)
    # ax.set_aspect('equal', adjustable='box')

def create_vertically_merged_plot(output_basename='CFNO_Merged_Stacked_ROC'):
    """
    Creates a vertically stacked ROC plot where the axes are merged to save space.
    """
    print(" Creating vertically merged ROC plot...")
    plt.rcParams['font.family'] = 'Times New Roman'
    # Set all relevant font elements to bold globally
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    sns.set_theme(style="white")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Plot data on both axes, now showing title for both
    plot_roc_curves_on_axis(ax1, 'AUROC_sum', 'CFNO-P', '(a) CFNO-P (Full Model)', show_title=True)
    plot_roc_curves_on_axis(ax2, 'AUROC_add', 'CFNO-W', '(b) CFNO-W (Ablation Model)', show_title=True)

    # Manually remove the x-axis label from the top plot
    ax1.set_xlabel('')
    
    # Save the final image
    png_path = f"{output_basename}.png"
    svg_path = f"{output_basename}.svg"
    try:
        # Use a tight bbox to ensure all elements fit
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        print(f"✅ 600 DPI PNG image saved: {png_path}")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"✅ SVG image saved: {svg_path}")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

if __name__ == '__main__':
    create_vertically_merged_plot() 