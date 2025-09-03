import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import matplotlib.pyplot as plt


def generate_dag_and_weights(n_nodes: int, p_edge: float = 0.3, min_weight: float = 0.1, max_weight: float = 0.99):
    """
    生成一个随机的有向无环图 (DAG) 及其对应的权重矩阵。

    Args:
        n_nodes (int): 节点数量。
        p_edge (float): 任意两个节点间存在边的概率。
        min_weight (float): 最小因果强度。
        max_weight (float): 最大因果强度。

    Returns:
        tuple: 包含邻接矩阵 (numpy array) 和权重矩阵 (numpy array) 的元组。
    """
    # 生成一个随机有向图
    G = nx.gnp_random_graph(n_nodes, p_edge, directed=True)

    # 确保图是无环的 (DAG)
    # 如果有环，则移除形成环的边，直到无环为止
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[0][0], cycle[0][1])
        except nx.NetworkXNoCycle:
            break

    # 获取邻接矩阵
    adjacency_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))

    # 创建权重矩阵
    weights_matrix = np.zeros_like(adjacency_matrix, dtype=float)
    edges = np.argwhere(adjacency_matrix > 0)
    for i, j in edges:
        weights_matrix[i, j] = np.random.uniform(min_weight, max_weight)

    print("✅ 已成功生成有向无环图 (DAG) 和权重矩阵。")
    return adjacency_matrix, weights_matrix


def get_nonlinear_functions():
    """返回一组非线性函数及其名称。"""
    return {
        'sin': np.sin,
        'tanh': np.tanh,
        'square': lambda x: x ** 2,
        'leaky_relu': lambda x: np.where(x > 0, x, 0.1 * x)
    }


def generate_nonlinear_time_series_data(adj_matrix, weights_matrix, n_time_steps: int, noise_std: float = 0.1):
    """
    根据给定的DAG和权重，生成非线性时间序列数据。

    Args:
        adj_matrix (np.ndarray): 邻接矩阵。
        weights_matrix (np.ndarray): 权重矩阵。
        n_time_steps (int): 时间序列的长度。
        noise_std (float): 噪声的标准差。

    Returns:
        pd.DataFrame: 生成的时间序列数据。
    """
    n_nodes = adj_matrix.shape[0]
    time_series_data = np.zeros((n_time_steps, n_nodes))

    # 初始化t=0时刻的值
    time_series_data[0, :] = np.random.randn(n_nodes) * noise_std

    # 获取非线性函数
    functions = get_nonlinear_functions()
    func_keys = list(functions.keys())

    # 为每条边随机分配一个非线性函数
    edge_functions = {}
    edges = np.argwhere(adj_matrix > 0)
    for i, j in edges:
        edge_functions[(i, j)] = random.choice(func_keys)

    print("🧬 因果关系与应用的非线性函数:")
    for (i, j), func_name in edge_functions.items():
        print(f"  节点 {i} -> 节点 {j}  (权重: {weights_matrix[i, j]:.2f}, 函数: {func_name})")

    # 迭代生成数据
    for t in range(1, n_time_steps):
        # 计算每个节点在t时刻的值
        for j in range(n_nodes):
            current_value = 0.0
            # 找到所有父节点
            parents = np.where(adj_matrix[:, j] > 0)[0]

            for parent_i in parents:
                # 获取父节点在t-1时刻的值
                parent_value = time_series_data[t - 1, parent_i]
                # 获取权重和函数
                weight = weights_matrix[parent_i, j]
                func_name = edge_functions[(parent_i, j)]
                non_linear_func = functions[func_name]

                # 计算贡献值
                current_value += weight * non_linear_func(parent_value)

            # 添加高斯噪声
            noise = np.random.normal(0, noise_std)
            time_series_data[t, j] = current_value + noise

    print(f"\n✅ 已成功生成 {n_time_steps} 个时间步的非线性数据。")
    # 转换为Pandas DataFrame
    columns = [f"Node_{i}" for i in range(n_nodes)]
    return pd.DataFrame(time_series_data, columns=columns)


def visualize_results(time_series_df, adj_matrix, weights_matrix, output_dir='data'):
    """
    可视化生成的时间序列和因果图。

    Args:
        time_series_df (pd.DataFrame): 时间序列数据。
        adj_matrix (np.ndarray): 邻接矩阵。
        weights_matrix (np.ndarray): 权重矩阵。
        output_dir (str): 保存图片的目录。
    """
    n_nodes = adj_matrix.shape[0]

    plt.figure(figsize=(16, 7))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 1. 绘制时间序列图
    ax1 = plt.subplot(1, 2, 1)
    nodes_to_plot = min(n_nodes, 5)
    time_steps_to_plot = min(len(time_series_df), 200)
    for i in range(nodes_to_plot):
        ax1.plot(time_series_df.index[:time_steps_to_plot], time_series_df[f'Node_{i}'][:time_steps_to_plot],
                 label=f'节点 {i}')
    ax1.set_title(f'生成的时间序列 (前 {time_steps_to_plot} 个时间步)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True)

    # 2. 绘制因果关系图
    ax2 = plt.subplot(1, 2, 2)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)  # for reproducibility

    # 绘制节点和标签
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
    node_labels = {i: f'节点 {i}' for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black')

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=2.0, arrowsize=20, alpha=0.7, node_size=800)

    # 在边上显示权重
    edge_labels = {}
    edges = np.argwhere(adj_matrix > 0)
    for i, j in edges:
        edge_labels[(i, j)] = f'{weights_matrix[i, j]:.2f}'
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=11)

    ax2.set_title('真实因果图与权重')
    ax2.axis('off')  # 关闭坐标轴

    plt.suptitle('非线性因果数据生成结果可视化', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    save_path = os.path.join(output_dir, "causal_data_visualization.png")
    plt.savefig(save_path, dpi=300)
    print(f"📈 可视化图像已保存至: {save_path}")
    plt.close()


def main():
    """主函数，执行数据生成和保存。"""
    # --- 参数配置 ---
    N_NODES = 10
    N_TIME_STEPS = 1000
    EDGE_PROBABILITY = 0.5
    NOISE_LEVEL = 0.5
    OUTPUT_DIR = "data"

    print("--- 开始生成非线性因果时间序列数据 ---")
    print(f"参数: 节点数={N_NODES}, 时间步={N_TIME_STEPS}, 边概率={EDGE_PROBABILITY}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 生成DAG和权重
    adj_matrix, weights_matrix = generate_dag_and_weights(N_NODES, EDGE_PROBABILITY)

    # 2. 生成时间序列数据
    time_series_df = generate_nonlinear_time_series_data(adj_matrix, weights_matrix, N_TIME_STEPS, NOISE_LEVEL)

    # 3. 准备要保存的数据帧
    node_indices = [f"Node_{i}" for i in range(N_NODES)]
    adj_df = pd.DataFrame(adj_matrix, index=node_indices, columns=node_indices)
    weights_df = pd.DataFrame(weights_matrix, index=node_indices, columns=node_indices)

    # 4. 定义文件路径
    ts_path = os.path.join(OUTPUT_DIR, "generated_time_series2.csv")
    adj_path = os.path.join(OUTPUT_DIR, "causal_adjacency_matrix2.csv")
    weights_path = os.path.join(OUTPUT_DIR, "causal_weights_matrix2.csv")

    # 5. 保存文件
    time_series_df.to_csv(ts_path, index=False)
    adj_df.to_csv(adj_path, index=True, header=True)
    weights_df.to_csv(weights_path, index=True, header=True)

    print("\n--- 数据保存成功 ---")
    print(f"  - 原始数据: {ts_path}")
    print(f"  - 真实因果图: {adj_path}")
    print(f"  - 因果强度: {weights_path}")

    # 6. 可视化结果
    visualize_results(time_series_df, adj_matrix, weights_matrix, OUTPUT_DIR)

    print("------------------------")


if __name__ == "__main__":
    main()
