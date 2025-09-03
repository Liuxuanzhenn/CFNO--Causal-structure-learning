import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_academic_plots():
    """
    Generates and saves academic-style line plots for F1-scores and SHD
    based on the provided experimental data for the NONLINEAR case.
    """
    # Data extracted from the user's second image (Nonlinear dataset)
    data = {
        'Methods': ['CFNO-W', 'CUTS+', 'eSRU', 'CSL-HNTS'] * 6,
        'Nodes': np.repeat([6, 8, 10, 12, 15, 20], 4),
        'F1-scores': [
            0.741, 0.465, 0.571, 0.609,  # 6 nodes
            0.694, 0.462, 0.587, 0.412,  # 8 nodes
            0.667, 0.337, 0.538, 0.314,  # 10 nodes
            0.603, 0.267, 0.431, 0.320,  # 12 nodes
            0.521, 0.279, 0.318, 0.252,  # 15 nodes
            0.537, 0.316, 0.340, 0.246   # 20 nodes
        ],
        'SHD': [
            7, 23, 18, 9,          # 6 nodes
            15, 35, 31, 20,        # 8 nodes
            25, 59, 43, 35,       # 10 nodes
            53, 77, 74, 51,       # 12 nodes
            70, 114, 103, 77,      # 15 nodes
            138, 243, 186, 141    # 20 nodes
        ]
    }

    df = pd.DataFrame(data)

    # Set seaborn style for academic plots
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    
    # --- Plot 1: F1-scores (Nonlinear) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Nodes',
        y='F1-scores',
        hue='Methods',
        style='Methods',
        markers=True,
        dashes=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title('F1-score Comparison (Nonlinear)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.xticks([6, 8, 10, 12, 15, 20])
    plt.legend(title='Methods', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    f1_filename = 'f1_scores_comparison_nonlinear.png'
    plt.savefig(f1_filename, dpi=300, bbox_inches='tight')
    print(f"Saved F1-score plot to {f1_filename}")
    plt.close()


    # --- Plot 2: SHD (Structural Hamming Distance) (Nonlinear) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Nodes',
        y='SHD',
        hue='Methods',
        style='Methods',
        markers=True,
        dashes=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title('SHD Comparison (Nonlinear)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Structural Hamming Distance (SHD)', fontsize=12)
    plt.xticks([6, 8, 10, 12, 15, 20])
    plt.legend(title='Methods', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    shd_filename = 'shd_comparison_nonlinear.png'
    plt.savefig(shd_filename, dpi=300, bbox_inches='tight')
    print(f"Saved SHD plot to {shd_filename}")
    plt.close()

if __name__ == '__main__':
    create_academic_plots()