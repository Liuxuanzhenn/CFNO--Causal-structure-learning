import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec
import argparse
import os

# 生成scale-free网络作为因果图
def generate_graph_scale_free(n, m=2):
    """
    生成scale-free有向无环图
    Args:
        n: 节点数量
        m: 每个新节点连接到现有节点的边数
    """
    # 使用Barabási-Albert模型生成无向scale-free网络
    G_undirected = nx.barabasi_albert_graph(n=n, m=m)

    # 将无向图转换为有向图，方向从低索引指向高索引
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(range(n))

    for edge in G_undirected.edges():
        # 确保方向始终从低索引指向高索引以保证无环
        i, j = min(edge), max(edge)
        G_directed.add_edge(i, j)

    # 转换为邻接矩阵
    adj_matrix = nx.to_numpy_array(G_directed)

    # 可选：随机重新排列节点以避免固定模式
    perm = np.random.permutation(n)
    adj_matrix_permuted = adj_matrix[perm, :][:, perm]

    return adj_matrix_permuted


# Erdos-Renyi图生成函数
def generate_graph_erdos(n, average_deg):
    """
    生成Erdos-Renyi有向无环图
    Args:
        n: 节点数量
        average_deg: 期望的平均度
    """
    # 确保平均度不超过n-1（完全图的度）
    average_deg = min(average_deg, n-1)

    p = average_deg / (n - 1)
    arr = np.triu(np.random.rand(n, n), k=1)
    G = (arr > 1 - p).astype(int)
    perm = np.random.permutation(n)
    G_permuted = G[perm, :][:, perm]
    return G_permuted


# 改进的ER2类型图生成函数
def generate_graph_er2(n, average_deg):
    """
    生成ER2类型的有向无环图，比普通ER图更加分块化
    Args:
        n: 节点数量
        average_deg: 期望的平均度
    """
    # 确保平均度不超过理论最大值
    average_deg = min(average_deg, n-1)

    # 分两块，每块内部连接概率更高，块间连接概率较低
    block_size = n // 2

    # 安全地计算概率，确保不超过1
    # 根据ER图理论，目标平均度为average_deg
    # 假设块内连接占总连接的70%，块间连接占30%
    total_possible_edges = (n * (n-1)) / 2  # 可能的总边数
    block1_possible_edges = (block_size * (block_size-1)) / 2  # 第一块内可能的边数
    block2_possible_edges = ((n-block_size) * (n-block_size-1)) / 2  # 第二块内可能的边数
    between_possible_edges = block_size * (n-block_size)  # 两块之间可能的边数

    # 计算每个区域的目标边数
    total_edges = average_deg * n / 2  # 总目标边数
    block1_edges = total_edges * 0.35  # 第一块内目标边数
    block2_edges = total_edges * 0.35  # 第二块内目标边数
    between_edges = total_edges * 0.3  # 两块之间目标边数

    # 计算概率，同时确保不超过1
    p_within1 = min(1.0, block1_edges / max(1, block1_possible_edges))
    p_within2 = min(1.0, block2_edges / max(1, block2_possible_edges))
    p_between = min(1.0, between_edges / max(1, between_possible_edges))

    # 记录生成的边数，用于调试
    edge_count = 0

    # 创建空图
    G = np.zeros((n, n))

    # 第一块内部连接
    arr1 = np.triu(np.random.rand(block_size, block_size), k=1)
    block1 = (arr1 > 1 - p_within1).astype(int)
    G[:block_size, :block_size] = block1
    edge_count += np.sum(block1)

    # 第二块内部连接
    arr2 = np.triu(np.random.rand(n-block_size, n-block_size), k=1)
    block2 = (arr2 > 1 - p_within2).astype(int)
    G[block_size:, block_size:] = block2
    edge_count += np.sum(block2)

    # 两块之间的连接
    arr_between = np.random.rand(block_size, n-block_size)
    between = (arr_between > 1 - p_between).astype(int)
    G[:block_size, block_size:] = between
    edge_count += np.sum(between)

    # 计算实际平均度
    actual_avg_degree = edge_count / n

    # 输出调试信息
    print(f"ER2图 - 目标平均度: {average_deg}, 实际平均度: {actual_avg_degree:.2f}")
    print(f"内部连接概率: {p_within1:.3f}, {p_within2:.3f}, 块间连接概率: {p_between:.3f}")

    # 随机排列
    perm = np.random.permutation(n)
    G_permuted = G[perm, :][:, perm]
    return G_permuted


# 改进的ER3类型图生成函数
def generate_graph_er3(n, average_deg):
    """
    生成ER3类型的有向无环图，有三个社区结构
    Args:
        n: 节点数量
        average_deg: 期望的平均度
    """
    # 确保平均度不超过理论最大值
    average_deg = min(average_deg, n-1)

    # 分三块
    block_size = n // 3
    sizes = [block_size, block_size, n - 2*block_size]

    # 计算可能的边数
    total_possible_edges = (n * (n-1)) / 2

    # 块内可能的边数
    block_possible_edges = []
    for size in sizes:
        block_possible_edges.append((size * (size-1)) / 2)

    # 块间可能的边数
    between_possible_edges = []
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            between_possible_edges.append(sizes[i] * sizes[j])

    # 计算总目标边数
    total_edges = average_deg * n / 2

    # 分配边数 - 60%块内，40%块间
    block_edges_percent = 0.6
    block_edges = [total_edges * block_edges_percent / 3] * 3  # 每块20%

    between_edges_percent = 0.4
    between_edges = [total_edges * between_edges_percent / 3] * 3  # 每对块间13.33%

    # 计算概率
    p_within = []
    for i in range(3):
        if block_possible_edges[i] > 0:
            p_within.append(min(1.0, block_edges[i] / block_possible_edges[i]))
        else:
            p_within.append(0.0)

    p_between = []
    for i in range(3):
        if between_possible_edges[i] > 0:
            p_between.append(min(1.0, between_edges[i] / between_possible_edges[i]))
        else:
            p_between.append(0.0)

    # 创建空图
    G = np.zeros((n, n))
    edge_count = 0

    # 填充块内连接
    start_idx = 0
    for i in range(3):
        size = sizes[i]
        end_idx = start_idx + size

        if size > 1:  # 确保块大小至少为2，才能有内部连接
            arr = np.triu(np.random.rand(size, size), k=1)
            block = (arr > 1 - p_within[i]).astype(int)
            G[start_idx:end_idx, start_idx:end_idx] = block
            edge_count += np.sum(block)

        start_idx = end_idx

    # 填充块间连接
    between_idx = 0
    for i in range(3):
        start_i = sum(sizes[:i])
        end_i = start_i + sizes[i]

        for j in range(i+1, 3):
            start_j = sum(sizes[:j])
            end_j = start_j + sizes[j]

            arr_between = np.random.rand(sizes[i], sizes[j])
            between = (arr_between > 1 - p_between[between_idx]).astype(int)
            G[start_i:end_i, start_j:end_j] = between
            edge_count += np.sum(between)
            between_idx += 1

    # 计算实际平均度
    actual_avg_degree = edge_count / n

    # 输出调试信息
    print(f"ER3图 - 目标平均度: {average_deg}, 实际平均度: {actual_avg_degree:.2f}")
    print(f"内部连接概率: {p_within}, 块间连接概率: {p_between}")

    # 随机排列
    perm = np.random.permutation(n)
    G_permuted = G[perm, :][:, perm]
    return G_permuted


# VAR时间序列生成函数 - 修改版本
def generate_var_time_series(G, n_samples, noise_scale=0.3, noise_type="gaussian", lag=2):
    """
    生成具有VAR(lag)结构的时间序列，针对因果发现进行优化
    Args:
        G: 邻接矩阵 (n x n)，其中G[i,j]=1表示i->j有因果关系
        n_samples: 时间步数量
        noise_scale: 噪声强度，影响信噪比
        noise_type: 噪声类型 - gaussian/exp/gumbel
        lag: VAR模型阶数
    """
    n_nodes = G.shape[0]

    # 权重矩阵 B 的值在 [-0.95, -0.2] U [0.2, 0.95] 的范围内随机生成
    # 确保权重不会太接近0（难以检测）或太接近1（可能导致不稳定）
    weights = (np.random.rand(n_nodes, n_nodes) * 1)  # 在 [0.2, 0.95] 内
    signs = np.random.choice([0, 1], size=(n_nodes, n_nodes))
    B = G * weights

    # 改进的VAR稳定性检查
    if lag == 1:
        spectral_radius = np.max(np.abs(np.linalg.eigvals(B)))
        if spectral_radius >= 0.99: # 使用更严格的阈值
            B = B * (0.99 / spectral_radius)
    else:
        # 对于高阶VAR，构建伴随矩阵并检查稳定性
        companion_matrix = np.zeros((n_nodes * lag, n_nodes * lag))

        # 填充伴随矩阵的第一行
        # 逐渐减弱的滞后效应
        companion_matrix[:n_nodes, :n_nodes] = B
        for i in range(1, lag):
            companion_matrix[:n_nodes, i*n_nodes:(i+1)*n_nodes] = B * (0.9 / (i+1))

        # 填充单位矩阵块 (标准形式)
        for i in range(1, lag):
            companion_matrix[i*n_nodes:(i+1)*n_nodes, (i-1)*n_nodes:i*n_nodes] = np.eye(n_nodes)

        # 检查稳定性
        spectral_radius = np.max(np.abs(np.linalg.eigvals(companion_matrix)))
        if spectral_radius >= 0.99:
             # 按比例缩放主系数矩阵B
            scale_factor = 0.99 / spectral_radius
            B *= scale_factor
            print(f"警告: 伴随矩阵谱半径为 {spectral_radius:.2f} >= 0.95。已将系数矩阵B缩放 {scale_factor:.2f} 倍。")


    # 初始化时间序列
    X = np.zeros((n_samples + lag, n_nodes))

    # 初始化前lag个时间步的值
    X[:lag] = np.random.randn(lag, n_nodes) * noise_scale

    # 减小噪声，提高信噪比
    noise_generators = {
        "gaussian": lambda size: np.random.normal(scale=noise_scale, size=size),
        "exp": lambda size: (np.random.exponential(scale=1.0, size=size) - 1.0) * noise_scale,
        "gumbel": lambda size: np.random.gumbel(scale=noise_scale, size=size)
    }
    noise = noise_generators[noise_type](size=(n_samples + lag, n_nodes))

    # VAR生成过程
    for t in range(lag, n_samples + lag):
        for lag_i in range(1, lag + 1):
            # 使用B直接计算，因为B[i,j]已经表示i->j的影响
            # 随滞后阶数增加，效应减弱
            X[t] += (B / lag_i) @ X[t - lag_i]
        X[t] += noise[t]

        # 适度的剪裁范围，主要用作安全网
        X[t] = np.clip(X[t], -200, 200)

    # 确保返回的邻接矩阵G真实反映了权重矩阵B的结构（非零元素）
    G_final = (B != 0).astype(int)

    return X[lag:], B, G_final


# 保存和读取文件的辅助函数
def save_matrix_to_csv(matrix, filename, add_indices=False):
    """
    保存矩阵到CSV文件
    Args:
        matrix: 要保存的矩阵
        filename: 文件名
        add_indices: 是否在第一行和第一列添加节点索引
    """
    if add_indices:
        n = matrix.shape[0]
        # 创建带索引的DataFrame
        df = pd.DataFrame(matrix)
        # 添加节点索引作为列名
        df.columns = [f"Node_{i}" for i in range(n)]
        # 添加节点索引作为行索引
        df.index = [f"Node_{i}" for i in range(n)]
        # 保存到CSV，保留索引
        df.to_csv(filename)
    else:
        pd.DataFrame(matrix).to_csv(filename, index=False, header=False)


def load_matrix_from_csv(filename, has_indices=False):
    """
    从CSV文件加载矩阵
    Args:
        filename: 文件名
        has_indices: 文件是否在第一行和第一列包含节点索引
    """
    if has_indices:
        df = pd.read_csv(filename, index_col=0)
        return df.values
    else:
        return pd.read_csv(filename, header=None).values


# 可视化函数修改版
def visualize_results(time_series, adj_matrix, weights_matrix=None, output_file='visualization_results.png'):
    """
    可视化生成的时间序列和因果图
    Args:
        time_series: 生成的时间序列数据
        adj_matrix: 邻接矩阵
        weights_matrix: 权重矩阵 (可选)
        output_file: 输出文件名
    """
    # 创建一个大图
    plt.figure(figsize=(16, 12))

    # 根据是否传入权重矩阵调整布局
    if weights_matrix is not None:
        gs = GridSpec(2, 3)
        has_weights = True
    else:
        gs = GridSpec(2, 2)
        has_weights = False

    # 跳过前50个时间步，避免初始不稳定性的影响
    skip_steps = 50
    display_steps = 200
    time_series_to_plot = time_series[skip_steps:skip_steps+display_steps]

    # 绘制时间序列（显示200个步长和最多8个节点）
    ax1 = plt.subplot(gs[0, :])
    n_nodes_to_plot = min(8, time_series.shape[1])
    for i in range(n_nodes_to_plot):
        ax1.plot(time_series_to_plot[:, i], label=f'节点 {i + 1}')
    ax1.set_title(f'生成的时间序列数据 (第{skip_steps+1}-{skip_steps+display_steps}步)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True)

    # 绘制网络图
    ax2 = plt.subplot(gs[1, 0])
    G = nx.DiGraph()
    n_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(n_nodes))

    # 添加边
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)

    # 绘制网络
    pos = nx.spring_layout(G, seed=21)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue',
                     node_size=600, arrowsize=15, ax=ax2)
    ax2.set_title('因果网络图')
    ax2.axis('off')

    # 绘制邻接矩阵热图
    ax3 = plt.subplot(gs[1, 1])
    im = ax3.imshow(adj_matrix, cmap='Blues')
    ax3.set_title('邻接矩阵热图 (Aij表示i→j的因果关系)')

    # 添加标签
    for i in range(min(10, n_nodes)):  # 限制标签数量以避免过于拥挤
        for j in range(min(10, n_nodes)):
            text = ax3.text(j, i, f"{adj_matrix[i, j]:.0f}",
                            ha="center", va="center", color="w" if adj_matrix[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel('被影响节点 (j)')
    ax3.set_ylabel('影响节点 (i)')

    # 如果提供了权重矩阵，则绘制权重矩阵热图
    if has_weights:
        ax4 = plt.subplot(gs[1, 2])
        # 动态设置颜色映射范围
        max_abs_weight = np.max(np.abs(weights_matrix)) if np.max(np.abs(weights_matrix)) > 0 else 1
        im2 = ax4.imshow(weights_matrix, cmap='coolwarm', vmin=-max_abs_weight, vmax=max_abs_weight)
        ax4.set_title('权重矩阵热图 (因果影响强度)')

        # 添加标签
        for i in range(min(10, n_nodes)):
            for j in range(min(10, n_nodes)):
                if weights_matrix[i, j] != 0:
                    text = ax4.text(j, i, f"{weights_matrix[i, j]:.2f}",
                                    ha="center", va="center", color="w" if abs(weights_matrix[i, j]) > 0.5 else "black")

        plt.colorbar(im2, ax=ax4)
        ax4.set_xlabel('被影响节点 (j)')
        ax4.set_ylabel('影响节点 (i)')

    # 添加注释说明因果关系方向
    plt.figtext(0.5, 0.01, "注意：在邻接矩阵中，元素Aij=1表示从节点i到节点j存在因果关系（i→j）",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # 为底部注释留出空间
    plt.savefig(output_file, dpi=300)
    plt.show()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成VAR时间序列数据及其因果图')

    # 添加参数
    parser.add_argument('--n_nodes', type=int, default=10, help='节点数量')
    parser.add_argument('--n_time_steps', type=int, default=1000, help='时间步数量')
    parser.add_argument('--lag', type=int, default=2, help='VAR模型的滞后阶数')
    parser.add_argument('--graph_type', type=str, default='er3',
                        choices=['scale_free', 'erdos', 'er2', 'er3'],
                        help='图结构生成方式')
    parser.add_argument('--average_degree', type=float, default=3,
                        help='ER图的平均度（仅对erdos, er2, er3类型有效）')
    parser.add_argument('--connectivity', type=int, default=4,
                        help='Scale-free图每个新节点的连接数（仅对scale_free类型有效）')
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'exp', 'gumbel'],
                        help='噪声类型')
    parser.add_argument('--noise_scale', type=float, default=0.5, help='噪声强度/标准差')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--output_prefix', type=str, default='generated',
                        help='输出文件前缀')

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()

    # 从参数获取设置
    n_nodes = args.n_nodes
    n_time_steps = args.n_time_steps
    lag = args.lag
    graph_type = args.graph_type
    output_prefix = args.output_prefix
    output_dir = args.output_dir

    print(f"\n正在生成具有{n_nodes}个节点的{graph_type}图...")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 根据选择的图类型生成网络结构
    if graph_type == "scale_free":
        adj_matrix = generate_graph_scale_free(n_nodes, args.connectivity)
        print(f"生成Scale-free网络，连接数: {args.connectivity}")
    elif graph_type == "erdos":
        adj_matrix = generate_graph_erdos(n_nodes, args.average_degree)
        print(f"生成Erdos-Renyi网络，平均度: {args.average_degree}")
    elif graph_type == "er2":
        adj_matrix = generate_graph_er2(n_nodes, args.average_degree)
        print(f"生成ER2网络，平均度: {args.average_degree}")
    elif graph_type == "er3":
        adj_matrix = generate_graph_er3(n_nodes, args.average_degree)
        print(f"生成ER3网络，平均度: {args.average_degree}")

    # 生成时间序列数据
    print(f"正在生成VAR({lag})时间序列，时间步数: {n_time_steps}...")
    time_series, weights_matrix, adj_matrix = generate_var_time_series(
        G=adj_matrix,
        n_samples=n_time_steps,
        noise_type=args.noise_type,
        noise_scale=args.noise_scale,
        lag=lag
    )

    # 保存文件
    print("保存数据...")
    df = pd.DataFrame(time_series)
    # 使用os.path.join来构建完整的文件路径
    df.to_csv(os.path.join(output_dir, 'generated_time_series2.csv'), index=False)
    save_matrix_to_csv(adj_matrix, os.path.join(output_dir, 'causal_adjacency_matrix2.csv'), add_indices=True)
    save_matrix_to_csv(weights_matrix, os.path.join(output_dir, 'causal_weights_matrix2.csv'), add_indices=True)
    print(f"数据已保存到 '{output_dir}' 目录。")

    # 数据统计
    print(f"\n生成的数据统计:")
    print(f"时间序列形状: {time_series.shape}")
    print(f"时间序列最小值: {np.min(time_series):.4f}")
    print(f"时间序列最大值: {np.max(time_series):.4f}")
    print(f"因果边数量: {np.sum(adj_matrix)}")
    print(f"平均每个节点的因果入度: {np.mean(np.sum(adj_matrix, axis=0)):.2f}")
    print(f"平均每个节点的因果出度: {np.mean(np.sum(adj_matrix, axis=1)):.2f}")


    # 可视化结果
    visualization_file = os.path.join(output_dir, f'{output_prefix}_visualization.png')
    print(f"\n可视化生成的数据，保存至 {visualization_file}...")
    visualize_results(time_series, adj_matrix, weights_matrix, visualization_file)
    print("可视化完成。")

    print(f"\n示例运行命令:\n python scripts/Var-data.py --n_nodes 15 --n_time_steps 1000 --lag 2 --graph_type er2 --output_prefix custom") 