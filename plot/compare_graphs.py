import pandas as pd
import graphviz
import numpy as np
import math

def load_and_prepare_matrices(true_path, pred_path, threshold):
    """
    Loads the true and predicted causal matrices, and applies a threshold to the predicted one.
    """
    # Load true weights, treating the first column as the index
    true_weights = pd.read_csv(true_path, index_col=0)
    # The column names are strings like 'Node_0', 'Node_1', we clean them to be integers.
    true_weights.columns = [int(col.split('_')[-1]) for col in true_weights.columns]
    true_weights.index = [int(idx.split('_')[-1]) for idx in true_weights.index]


    # Load predicted probabilities, treating the first column as the index
    pred_probs = pd.read_csv(pred_path, index_col=0)
    # Make sure columns are integers
    pred_probs.columns = pred_probs.columns.astype(int)
    pred_probs.index = pred_probs.index.astype(int)

    # Apply threshold to predicted probabilities to get an adjacency matrix
    pred_adj = (pred_probs >= threshold).astype(int)

    return true_weights, pred_adj

def create_manual_side_by_side_graph(true_weights, predicted_adj):
    """
    Creates a single SVG with two manually positioned, side-by-side graphs.
    """
    font = "Times New Roman"
    # Use 'neato' engine and set transparent background
    dot = graphviz.Digraph('ManualSideBySide', engine='neato')
    dot.attr(bgcolor='transparent', splines='line')

    # --- Layout Constants ---
    radius = 2.5
    h_spacing = 8.0  # Reduced spacing to bring graphs and legend closer
    left_center_x, left_center_y = 0, 0
    right_center_x, right_center_y = h_spacing, 0

    # --- Node Styling ---
    node_fillcolor = "#dbe4f0:#aec7e8"
    node_border_color = "#99b3d1"
    node_style = {
        'shape': 'circle',
        'style': 'filled',
        'fontname': font,
        'fillcolor': node_fillcolor,
        'color': node_border_color,
        'fontcolor': 'black',
        'penwidth': '2',
        'shadow': 'true'
    }

    # --- Edge Styling ---
    # Softer, dark grey for less contrast
    dark_grey = '#444444'
    # Style for correctly predicted edges (True Positives)
    tp_style = {'penwidth': '2.5', 'color': dark_grey}
    # Style for incorrectly predicted edges (False Positives)
    fp_style = {'style': 'dashed', 'color': '#cccccc', 'penwidth': '1.5'}
    # Style for missed edges (False Negatives)
    fn_style = {'style': 'dashed', 'color': '#ffaaaa', 'penwidth': '1.5'}

    # --- Data Preparation ---
    nodes = true_weights.index
    nodes_str = [str(n) for n in nodes]
    
    true_edges = set()
    for i in nodes:
        for j in nodes:
            if true_weights.loc[i, j] > 0:
                true_edges.add((str(i), str(j)))

    pred_edges = set()
    # As per previous request, manually remove the 3->7 edge
    if 3 in predicted_adj.index and 7 in predicted_adj.columns:
        predicted_adj.loc[3, 7] = 0
    for i in nodes:
        for j in nodes:
            if predicted_adj.loc[i, j] > 0:
                pred_edges.add((str(i), str(j)))

    num_nodes = len(nodes_str)

    # --- Create Left Graph (True) ---
    dot.node('true_label', 'True Causal Graph', pos=f"{left_center_x},{left_center_y - radius - 0.8}!", shape='plaintext', fontsize='20', fontname=font)
    for i, node_name in enumerate(nodes_str):
        angle = 2 * math.pi * i / num_nodes
        x = left_center_x + radius * math.cos(angle)
        y = left_center_y + radius * math.sin(angle)
        dot.node(f"true_{node_name}", node_name, pos=f"{x},{y}!", **node_style)

    for u, v in true_edges:
        # If the edge was missed by the model, show it as a red dashed line
        if (u, v) not in pred_edges:
            dot.edge(f"true_{u}", f"true_{v}", **fn_style)
        else:
            # Otherwise, use the unified style for correctly identified edges
            dot.edge(f"true_{u}", f"true_{v}", **tp_style)


    # --- Create Right Graph (Predicted) ---
    dot.node('pred_label', 'Predicted Causal Graph', pos=f"{right_center_x},{right_center_y - radius - 0.8}!", shape='plaintext', fontsize='20', fontname=font)
    for i, node_name in enumerate(nodes_str):
        angle = 2 * math.pi * i / num_nodes
        x = right_center_x + radius * math.cos(angle)
        y = right_center_y + radius * math.sin(angle)
        dot.node(f"pred_{node_name}", node_name, pos=f"{x},{y}!", **node_style)

    for u, v in pred_edges:
        style = tp_style if (u, v) in true_edges else fp_style
        dot.edge(f"pred_{u}", f"pred_{v}", **style)

    # --- Create Legend in the Middle ---
    legend_x = h_spacing / 2.0
    legend_y_start = 1.0
    y_step = -0.6
    line_len = 0.4
    label_offset = 0.5  # Reduced offset to bring text closer to lines
    key_node_style = {'shape': 'point', 'width': '0.01', 'height': '0.01'}
    label_node_style = {'shape': 'plaintext', 'fontsize': '14', 'fontname': font}

    # 1. Correct
    y_pos = legend_y_start
    dot.node('L1s', '', pos=f"{legend_x - line_len},{y_pos}!", **key_node_style)
    dot.node('L1e', '', pos=f"{legend_x + line_len},{y_pos}!", **key_node_style)
    dot.edge('L1s', 'L1e', **tp_style, arrowhead='none')
    dot.node('L1_label', 'Correct', pos=f"{legend_x + line_len + label_offset},{y_pos}!", **label_node_style)

    # 2. Incorrect
    y_pos += y_step
    dot.node('L2s', '', pos=f"{legend_x - line_len},{y_pos}!", **key_node_style)
    dot.node('L2e', '', pos=f"{legend_x + line_len},{y_pos}!", **key_node_style)
    dot.edge('L2s', 'L2e', **fp_style, arrowhead='none')
    dot.node('L2_label', 'Incorrect', pos=f"{legend_x + line_len + label_offset},{y_pos}!", **label_node_style)
    
    # 3. Missed
    y_pos += y_step
    dot.node('L3s', '', pos=f"{legend_x - line_len},{y_pos}!", **key_node_style)
    dot.node('L3e', '', pos=f"{legend_x + line_len},{y_pos}!", **key_node_style)
    dot.edge('L3s', 'L3e', **fn_style, arrowhead='none')
    dot.node('L3_label', 'Missed', pos=f"{legend_x + line_len + label_offset},{y_pos}!", **label_node_style)

    # Render the final graph
    dot.render('manual_side_by_side_graph', view=True, format='svg', cleanup=True)
    print("Manually positioned side-by-side graph saved and opened.")

def main():
    """
    Main function to compare and visualize the causal graphs.
    """
    # File paths
    true_causal_matrix_path = 'data/causal_weights_matrix2.csv'
    predicted_causal_matrix_path = 'results/probabilistic_causal_pfno_20250721_095618/causal_probabilities_final.csv'
    
    # Threshold for predicted probabilities
    PROBABILITY_THRESHOLD = 0.6

    # Load data and prepare matrices
    true_weights, predicted_adj = load_and_prepare_matrices(
        true_causal_matrix_path,
        predicted_causal_matrix_path,
        PROBABILITY_THRESHOLD
    )

    # Create and save the side-by-side visualization
    create_manual_side_by_side_graph(true_weights, predicted_adj)

    print("\nComparison graph generated successfully.")

if __name__ == '__main__':
    main() 