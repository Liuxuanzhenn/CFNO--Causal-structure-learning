
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_lorenz96_data(
    n_vars=10,
    seq_length=1000,
    F=5,
    dt=0.01,
    spin_up_steps=50
):
    """
    Generates time series data from the Lorenz-96 model and its true causal graph.

    Args:
        n_vars (int): The number of variables (nodes) in the system.
        seq_length (int): The number of time steps to generate for the final series.
        F (float): The forcing constant in the Lorenz-96 equations.
        dt (float): The time step for numerical integration.
        spin_up_steps (int): The number of initial steps to discard to allow the
                             system to settle onto its attractor.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The generated time series data (shape: seq_length, n_vars).
            - np.ndarray: The ground-truth adjacency matrix (shape: n_vars, n_vars).
    """

    # --- 1. Define the Lorenz-96 ODE system ---
    def lorenz96(t, x, F):
        dxdt = np.zeros(n_vars)
        for i in range(n_vars):
            # Periodic boundary conditions
            x_prev2 = x[(i - 2 + n_vars) % n_vars]
            x_prev1 = x[(i - 1 + n_vars) % n_vars]
            x_next1 = x[(i + 1) % n_vars]
            
            dxdt[i] = (x_next1 - x_prev2) * x_prev1 - x[i] + F
        return dxdt

    # --- 2. Generate the ground-truth causal adjacency matrix ---
    # In Lorenz-96, x_i is caused by x_{i-2}, x_{i-1}, and x_{i+1}.
    # The matrix entry A[j, i] = 1 means j -> i.
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars):
        adj_matrix[(i - 2 + n_vars) % n_vars, i] = 1
        adj_matrix[(i - 1 + n_vars) % n_vars, i] = 1
        adj_matrix[(i + 1) % n_vars, i] = 1
    
    # --- 3. Generate the time series data ---
    # Set initial conditions: a small perturbation from a constant state
    x0 = np.full(n_vars, F)
    x0[0] += 0.01 

    # Spin-up period to let the system settle onto the attractor
    spin_up_t_span = [0, spin_up_steps * dt]
    spin_up_sol = solve_ivp(
        lorenz96, spin_up_t_span, x0, args=(F,), dense_output=True, t_eval=np.linspace(*spin_up_t_span, spin_up_steps)
    )
    x_final_spin_up = spin_up_sol.y[:, -1]

    # Main run to generate the actual time series
    t_span = [0, seq_length * dt]
    t_eval = np.linspace(*t_span, seq_length)
    sol = solve_ivp(
        lorenz96, t_span, x_final_spin_up, args=(F,), dense_output=True, t_eval=t_eval
    )
    time_series = sol.y.T  # Transpose to get (seq_length, n_vars)
    
    return time_series, adj_matrix

def visualize_results(time_series, adj_matrix, output_path):
    """
    Visualizes the generated time series and the adjacency matrix.

    Args:
        time_series (np.ndarray): The time series data.
        adj_matrix (np.ndarray): The adjacency matrix.
        output_path (str): Path to save the visualization image.
    """
    print(f"\nGenerating visualization and saving to {output_path}...")
    
    n_vars = time_series.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Plot 1: Time series data (first 5 variables) ---
    ax1 = axes[0]
    n_plot_vars = min(5, n_vars)
    for i in range(n_plot_vars):
        ax1.plot(time_series[:, i], label=f'x_{{{i+1}}}')
    ax1.set_title(f'Lorenz-96 Time Series (First {n_plot_vars} Variables)', fontsize=16)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: Causal Adjacency Matrix Heatmap ---
    ax2 = axes[1]
    sns.heatmap(adj_matrix, ax=ax2, cbar=False, cmap='Blues', linewidths=.5, linecolor='black', square=True)
    ax2.set_title('Ground-Truth Causal Adjacency Matrix', fontsize=16)
    ax2.set_xlabel('Effect (To Node)', fontsize=12)
    ax2.set_ylabel('Cause (From Node)', fontsize=12)
    ax2.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Visualization saved successfully.")


if __name__ == "__main__":
    # --- Configuration ---
    # Reflecting the parameters you chose
    N_VARS = 10
    SEQ_LENGTH = 1000
    FORCING = 5.0
    SPIN_UP = 50

    # --- File Paths (Robustly locate project root) ---
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Get the project root directory (one level up from 'scripts')
    project_root = os.path.dirname(script_dir)
    # Define the data directory path relative to the project root
    DATA_DIR = os.path.join(project_root, 'data')
    
    TIME_SERIES_PATH = os.path.join(DATA_DIR, 'generated_time_series2.csv')
    ADJ_MATRIX_PATH = os.path.join(DATA_DIR, 'causal_adjacency_matrix2.csv')
    VISUALIZATION_PATH = os.path.join(DATA_DIR, 'lorenz96_visualization.png')

    # --- Execution ---
    print("Starting data generation process...")

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate data
    generated_ts, causal_adj = generate_lorenz96_data(
        n_vars=N_VARS,
        seq_length=SEQ_LENGTH,
        F=FORCING,
        spin_up_steps=SPIN_UP
    )

    # Save data to CSV files
    np.savetxt(TIME_SERIES_PATH, generated_ts, delimiter=",", fmt='%.8f')
    print(f"Successfully saved time series data to: {TIME_SERIES_PATH}")
    
    # Save adjacency matrix with indices and headers using pandas
    adj_df = pd.DataFrame(causal_adj,
                          index=range(N_VARS),
                          columns=range(N_VARS))
    adj_df.to_csv(ADJ_MATRIX_PATH)
    print(f"Successfully saved causal adjacency matrix to: {ADJ_MATRIX_PATH}")

    # Visualize the results
    visualize_results(generated_ts, causal_adj, VISUALIZATION_PATH)

    print("\nData generation and visualization complete.")
    print(f"Generated time series shape: {generated_ts.shape}")
    print(f"Adjacency matrix shape: {causal_adj.shape}") 