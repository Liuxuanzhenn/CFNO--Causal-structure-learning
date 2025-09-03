"""
非线性时间序列生成工具 - 专为因果发现研究设计

这个脚本用于生成具有明确因果结构的非线性时间序列数据，特别适用于因果发现算法的测试和评估。

特点:
1. 生成具有真实因果结构的有向无环图(DAG)
2. 使用多种非线性函数模拟复杂的现实世界因果关系
3. 支持不同的时间延迟效应
4. 包含数据质量评估功能，确保生成的数据适合因果发现
5. 提供参数调整建议，以优化数据生成过程


输出文件:
- generated_time_series2.csv: 生成的时间序列数据
- causal_adjacency_matrix2.csv: 因果关系邻接矩阵
- causal_weights_matrix2.csv: 因果权重矩阵
- lag_matrix.csv: 时间延迟矩阵
- model_config.json: 模型配置文件(可用于重现结果)
- visualization_nonlinear_results.png: 可视化结果

推荐参数设置（用于因果发现）:
- 节点数量(n_nodes): 8-15
- 时间步长(n_time_steps): 3000-10000
- 平均度(average_degree): 1.5-3.0
- 噪声尺度(noise_scale): 0.05-0.2
- 最大延迟阶数(max_lag): 1-3


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec
import time
import argparse
import os
import json


# 生成DAG的函数
def generate_graph_erdos(n, average_deg, seed=None):
    """生成随机DAG"""
    # 使用随机种子
    if seed is None:
        seed = int(time.time()) % 10000
    print(f"使用随机种子: {seed}")
    np.random.seed(seed)

    # 计算连接概率
    p = 1 / (n - 1) * average_deg

    # 限制概率范围，避免图过于稀疏或过于密集
    p = min(max(p, 0.1), 0.8)

    # 生成上三角随机邻接矩阵
    arr = np.triu(np.random.rand(n, n), k=1)
    G = (arr > 1 - p).astype(int)

    # 如果生成的图太稀疏（边数太少），确保每个节点至少有一个输入或输出
    edge_count = np.sum(G)
    if edge_count < n // 2:
        # 确保每个节点至少有一个连接
        for i in range(n):
            # 检查该节点是否没有输入边和输出边
            if np.sum(G[i, :]) == 0 and np.sum(G[:, i]) == 0:
                # 随机选择一个其他节点进行连接
                j = (i + np.random.randint(1, n)) % n
                # 确保形成有向边 (i->j 或 j->i)，保持 DAG 性质
                if i < j:
                    G[i, j] = 1
                else:
                    G[j, i] = 1

    # 随机排列节点以增加多样性
    perm = np.random.permutation(n)
    G_permuted = G[perm, :][:, perm]

    # 确保是有向无环图
    if not is_dag(G_permuted):
        print("警告：生成的图不是DAG，进行修复...")
        # 确保图是 DAG，移除成环的边
        G_permuted = ensure_dag(G_permuted)

    return G_permuted


# 确保图是 DAG 的辅助函数
def ensure_dag(adj_matrix):
    """确保图是 DAG，必要时移除成环的边"""
    n = adj_matrix.shape[0]

    # 拓扑排序
    visited = [0] * n  # 0: 未访问, 1: 正在访问, 2: 已完成访问
    topo_order = []

    def dfs(node):
        visited[node] = 1

        # 访问所有邻居
        for neighbor in range(n):
            if adj_matrix[node, neighbor] > 0:
                if visited[neighbor] == 0:
                    dfs(neighbor)
                elif visited[neighbor] == 1:
                    # 发现环，移除这条边
                    adj_matrix[node, neighbor] = 0

        visited[node] = 2
        topo_order.append(node)

    # 对每个未访问的节点进行 DFS
    for i in range(n):
        if visited[i] == 0:
            dfs(i)

    # 返回修改后的邻接矩阵
    return adj_matrix


# 生成自定义DAG
def generate_custom_graph(edges):
    """
    根据指定的边列表生成自定义DAG

    参数:
    edges: 列表，每个元素是元组(i, j)表示从节点i到节点j的边

    返回:
    邻接矩阵
    """
    # 确定节点数量
    nodes = set()
    for i, j in edges:
        nodes.add(i)
        nodes.add(j)
    n = max(nodes) + 1

    # 创建邻接矩阵
    G = np.zeros((n, n), dtype=int)
    for i, j in edges:
        G[i, j] = 1

    # 检查是否为DAG (没有环)
    if not is_dag(G):
        raise ValueError("提供的边列表不能形成DAG(有向无环图)，请检查是否有环")

    return G


def is_dag(adj_matrix):
    """检查图是否为DAG"""
    n = adj_matrix.shape[0]
    visited = [0] * n  # 0: 未访问, 1: 正在访问, 2: 已完成访问

    def has_cycle(node):
        if visited[node] == 1:  # 如果正在访问，说明有环
            return True
        if visited[node] == 2:  # 如果已访问，不需要再次访问
            return False

        visited[node] = 1  # 标记为正在访问

        # 检查所有邻居
        for neighbor in range(n):
            if adj_matrix[node, neighbor] > 0:
                if has_cycle(neighbor):
                    return True

        visited[node] = 2  # 标记为已完成访问
        return False

    # 检查每个节点
    for i in range(n):
        if visited[i] == 0:  # 如果未访问
            if has_cycle(i):
                return False

    return True


# 非线性函数库
def nonlinear_functions():
    """返回一组增强的非线性函数，更适合因果发现"""
    functions = [
        (lambda x: np.sin(x), "sin", 2.0),
        (lambda x: np.tanh(x), "tanh", 2.0),
        (lambda x: x ** 2, "square", 1.5),
        (lambda x: np.maximum(0, x), "relu", 2.0),
        (lambda x: 1 / (1 + np.exp(-2 * x)), "sigmoid", 2.0),
        (lambda x: np.where(x > 0, x, 0.1 * x), "leaky_relu", 2.0),
        (lambda x: np.log(np.abs(x) + 1) * np.sign(x), "log_sign", 2.0),
        (lambda x: np.cbrt(x), "cbrt", 2.0),
        (lambda x: np.where(x > 0, np.sqrt(x), -np.sqrt(-x)), "signed_sqrt", 2.0),
    ]
    return functions


# 时间序列生成函数
def generate_nonlinear_time_series(G, n_samples, noise_scale=0.1, max_lag=1, seed=None):
    """
    使用非线性函数生成时间序列

    参数:
    G: 邻接矩阵
    n_samples: 样本数量
    noise_scale: 噪声尺度
    max_lag: 最大延迟阶数
    seed: 随机种子

    返回:
    time_series: 生成的时间序列
    weights: 权重矩阵
    G: 邻接矩阵
    functions_dict: 函数字典
    function_names: 函数名字典
    lag_matrix: 延迟阶数矩阵
    """
    # 使用随机种子
    if seed is None:
        seed = int(time.time() * 1000) % 10000
    np.random.seed(seed)

    n_nodes = G.shape[0]
    all_functions = nonlinear_functions()

    # 准备权重和映射函数
    weights = np.zeros((n_nodes, n_nodes))
    functions = {}
    function_names = {}
    lag_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # 对于每个连接，分配非线性函数
    for i in range(n_nodes):
        for j in range(n_nodes):
            if G[i, j] > 0:
                # 为每条边随机选择一个非线性函数
                func_idx = np.random.randint(0, len(all_functions))
                func, name, weight_scale = all_functions[func_idx]

                # 生成影响权重 (0.5 ~ 2.0范围) - 调整更窄的权重区间，避免过强的因果效应
                weight = (np.random.rand() * 0.8 + 0.2) * weight_scale

                # 随机选择延迟阶数 (1到max_lag)
                lag = np.random.randint(1, max_lag + 1)
                lag_matrix[i, j] = lag

                # 保存函数和权重
                functions[(i, j)] = [(func, weight)]
                function_names[(i, j)] = name

                # 记录总权重
                weights[i, j] = weight

    # 以数学公式形式打印每个节点的影响关系
    print("\n======= 变量间关系详细信息（数学公式形式）=======")
    for j in range(n_nodes):
        # 构建公式字符串
        formula = f"X{j}(t) = "
        terms = []

        # 找出所有影响节点j的节点i
        influences = False
        for i in range(n_nodes):
            if G[i, j] > 0:
                influences = True
                weight = weights[i, j]
                func_name = function_names[(i, j)]
                lag = lag_matrix[i, j]
                terms.append(f"{weight:.3f}*{func_name}(X{i}(t-{lag}))")

        # 组合完整公式
        if influences:
            formula += " + ".join(terms) + " + e(t)"
        else:
            formula += "e(t)  # 仅受噪声影响，无其他节点影响"

        # 打印公式
        print(formula)

    # 初始化时间序列
    X = np.zeros((n_samples + max_lag, n_nodes))

    # 使用较小的初始值波动
    X[:max_lag] = np.random.randn(max_lag, n_nodes)

    # 记录每个节点的上一个有效值
    last_values = X[max_lag - 1].copy()

    # 生成噪声 - 使用较小的噪声比例
    noise = np.random.normal(scale=noise_scale, size=(n_samples + max_lag, n_nodes))

    # 生成过程
    for t in range(max_lag, n_samples + max_lag):
        for j in range(n_nodes):  # 对每个目标节点j
            # 首先添加噪声项
            X[t, j] = noise[t, j]

            # 计算来自各个源节点的非线性影响
            total_influence = 0
            for i in range(n_nodes):  # 遍历所有可能的源节点i
                if G[i, j] > 0:  # 如果存在i->j的边
                    # 获取为此连接选择的非线性函数和延迟阶数
                    funcs_for_edge = functions[(i, j)]
                    lag = lag_matrix[i, j]

                    # 应用非线性函数
                    for func, weight in funcs_for_edge:
                        try:
                            # 源节点i在特定延迟时间步的值作为输入
                            input_val = X[t - lag, i]
                            # 如果输入无效，使用上一个有效值
                            if np.isnan(input_val) or np.isinf(input_val):
                                input_val = last_values[i]

                            # 应用非线性函数和权重
                            func_output = func(input_val)

                            # 处理无效输出
                            if np.isnan(func_output) or np.isinf(func_output):
                                func_output = 0

                            # 限制函数输出幅度
                            func_output = np.clip(func_output, -2, 2)

                            # 累加到目标节点
                            X[t, j] += weight * func_output
                            total_influence += weight * func_output
                        except Exception as e:
                            print(f"错误: 计算节点{i}到节点{j}的非线性影响时发生异常: {e}")

            # 确保因果影响明显大于噪声 - 提高因果贡献比例
            if total_influence != 0:
                # 增加因果贡献权重到 0.95，使信号更清晰
                influence_weight = 0.95
                noise_weight = 1 - influence_weight
                X[t, j] = noise_weight * X[t, j] + influence_weight * total_influence

            # 限制输出范围，避免极值
            X[t, j] = np.clip(X[t, j], -100, 100)

            # 更新上一次值
            last_values[j] = X[t, j]

    # 返回数据
    return X[max_lag:], weights, G, functions, function_names, lag_matrix


# 保存和读取文件的辅助函数
def save_matrix_to_csv(matrix, filename, add_indices=False):
    """保存矩阵到CSV文件"""
    if add_indices:
        n = matrix.shape[0]
        df = pd.DataFrame(matrix)
        df.columns = [f"Node_{i}" for i in range(n)]
        df.index = [f"Node_{i}" for i in range(n)]
        df.to_csv(filename)
    else:
        pd.DataFrame(matrix).to_csv(filename, index=False, header=False)


def save_model_config(n_nodes, adj_matrix, weights_matrix, function_names, lag_matrix, filename="model_config.json"):
    """保存模型配置到JSON文件"""
    config = {
        "n_nodes": int(n_nodes),
        "adjacency_matrix": adj_matrix.tolist(),
        "weights_matrix": weights_matrix.tolist(),
        "function_names": {f"{i},{j}": name for (i, j), name in function_names.items()},
        "lag_matrix": lag_matrix.tolist(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"模型配置已保存到: {filename}")


def load_model_config(filename="model_config.json"):
    """从JSON文件加载模型配置"""
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)

    adj_matrix = np.array(config["adjacency_matrix"])
    weights_matrix = np.array(config["weights_matrix"])

    # 恢复function_names字典
    function_names = {}
    for key, name in config["function_names"].items():
        i, j = map(int, key.split(','))
        function_names[(i, j)] = name

    lag_matrix = np.array(config["lag_matrix"])

    return config["n_nodes"], adj_matrix, weights_matrix, function_names, lag_matrix


# 可视化函数
def visualize_results(time_series, adj_matrix, weights_matrix, lag_matrix=None, output_dir='.'):
    """可视化生成的时间序列和因果图"""
    plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2)

    # 绘制时间序列（显示前150个步长和前6个节点）
    ax1 = plt.subplot(gs[0, :])
    for i in range(min(5, time_series.shape[1])):
        ax1.plot(time_series[:500, i], label=f'节点 {i}')
    ax1.set_title('生成的非线性时间序列数据 (前150个时间步)', fontsize=14)
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
                # 如果有延迟矩阵，则使用延迟信息作为边标签
                if lag_matrix is not None:
                    G.add_edge(i, j, weight=weights_matrix[i, j], lag=lag_matrix[i, j])
                else:
                    G.add_edge(i, j, weight=weights_matrix[i, j])

    # 绘制网络
    pos = nx.spring_layout(G, seed=42)

    # 节点绘制
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)

    # 边绘制 - 根据权重设置宽度
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:  # 确保有边存在
        min_weight, max_weight = min(edge_weights), max(edge_weights)
        width_scale = 4.0 / max(max_weight - min_weight, 1e-6)  # 避免除以零
        edge_widths = [1 + (w - min_weight) * width_scale for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=15, alpha=0.7)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=10)

    # 如果有延迟矩阵，则添加边标签显示延迟
    if lag_matrix is not None:
        edge_labels = {(i, j): f"w={G[i][j]['weight']:.2f}\nlag={G[i][j]['lag']}"
                       for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    ax2.set_title('因果网络图')
    ax2.axis('off')

    # 绘制二值邻接矩阵热图
    ax3 = plt.subplot(gs[1, 1])
    im = ax3.imshow(adj_matrix, cmap='Blues')
    ax3.set_title('二值邻接矩阵 (Aij=1表示i→j的因果关系)')

    # 添加标签
    for i in range(n_nodes):
        for j in range(n_nodes):
            text = ax3.text(j, i, f"{adj_matrix[i, j]:.0f}",
                            ha="center", va="center", color="w" if adj_matrix[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel('被影响节点 (j)')
    ax3.set_ylabel('影响节点 (i)')

    # 绘制加权邻接矩阵热图
    ax4 = plt.subplot(gs[2, :])
    im = ax4.imshow(weights_matrix, cmap='coolwarm')
    ax4.set_title('加权邻接矩阵 (显示因果强度)')

    # 添加标签
    for i in range(n_nodes):
        for j in range(n_nodes):
            if weights_matrix[i, j] > 0:
                text = ax4.text(j, i, f"{weights_matrix[i, j]:.2f}",
                                ha="center", va="center",
                                color="white" if abs(weights_matrix[i, j]) > 1.0 else "black")

    plt.colorbar(im, ax=ax4)
    ax4.set_xlabel('被影响节点 (j)')
    ax4.set_ylabel('影响节点 (i)')

    plt.tight_layout()

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, 'visualization_nonlinear_results.png'), dpi=300)
    plt.show()


# 命令行参数解析函数
def parse_arguments():
    parser = argparse.ArgumentParser(description='非线性时间序列生成工具')

    # 基本参数
    parser.add_argument('--n_nodes', type=int, default=20, help='节点数量')
    parser.add_argument('--n_time_steps', type=int, default=1000, help='时间步长')
    parser.add_argument('--average_degree', type=float, default=11, help='平均度（每个节点平均影响几个其他节点）')
    parser.add_argument('--noise_scale', type=float, default=0.55, help='噪声尺度')
    parser.add_argument('--max_lag', type=int, default=2, help='最大延迟阶数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--prefix', type=str, default='', help='输出文件前缀')

    # 特殊功能标志
    parser.add_argument('--load_config', type=str, default=None, help='从JSON文件加载模型配置')
    parser.add_argument('--no_visualization', action='store_true', help='不生成可视化结果')

    return parser.parse_args()


# 主函数
def main():
    # 解析命令行参数
    args = parse_arguments()

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 设置文件名前缀
    prefix = args.prefix + "_" if args.prefix else ""

    # 处理模型配置
    if args.load_config:
        print(f"从配置文件加载模型: {args.load_config}")
        n_nodes, adj_matrix, weights_matrix, function_names, lag_matrix = load_model_config(args.load_config)

        # 使用加载的模型生成时间序列
        time_series, _, _, _, _, _ = generate_nonlinear_time_series(
            G=adj_matrix,
            n_samples=args.n_time_steps,
            noise_scale=args.noise_scale,
            max_lag=args.max_lag,
            seed=args.seed
        )
    else:
        # 参数设置
        n_nodes = args.n_nodes
        n_time_steps = args.n_time_steps
        average_degree = args.average_degree

        # 生成DAG（随机）
        adj_matrix = generate_graph_erdos(n_nodes, average_degree, seed=args.seed)

        # 生成非线性时间序列数据
        time_series, weights_matrix, adj_matrix, functions_dict, function_names, lag_matrix = generate_nonlinear_time_series(
            G=adj_matrix,
            n_samples=n_time_steps,
            noise_scale=args.noise_scale,
            max_lag=args.max_lag,
            seed=args.seed
        )

        # 保存模型配置
        save_model_config(n_nodes, adj_matrix, weights_matrix, function_names, lag_matrix,
                          filename=os.path.join(args.output_dir, f"{prefix}model_config.json"))

    # 打印因果结构摘要
    print("\n====== 因果结构摘要 ======")
    edge_count = np.sum(adj_matrix)
    print(f"节点数量: {n_nodes}")
    print(f"因果边数量: {int(edge_count)}")
    print(f"平均每个节点影响: {edge_count / n_nodes:.2f} 个其他节点")

    # 打印各种非线性函数的使用情况
    print("\n======= 非线性函数使用统计 =======")
    func_counts = {}
    for (i, j) in function_names:
        name = function_names[(i, j)]
        if name not in func_counts:
            func_counts[name] = 0
        func_counts[name] += 1

    for func_name, count in func_counts.items():
        print(f"函数 {func_name}: 使用了 {count} 次")

    # 打印所有因果边的详细信息
    print("\n======= 所有因果边的详细信息 =======")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                lag = lag_matrix[i, j] if lag_matrix is not None else 1
                print(f"节点{i} -> 节点{j}: 权重={weights_matrix[i, j]:.3f}, 函数={function_names[(i, j)]}, 延迟={lag}")

    # 保存文件
    save_matrix_to_csv(time_series, os.path.join(args.output_dir, f"{prefix}generated_time_series2.csv"))
    save_matrix_to_csv(adj_matrix, os.path.join(args.output_dir, f"{prefix}causal_adjacency_matrix2.csv"),
                       add_indices=True)
    save_matrix_to_csv(weights_matrix, os.path.join(args.output_dir, f"{prefix}causal_weights_matrix2.csv"),
                       add_indices=True)
    if lag_matrix is not None:
        save_matrix_to_csv(lag_matrix, os.path.join(args.output_dir, f"{prefix}lag_matrix.csv"), add_indices=True)

    print("\n======= 非线性时间序列生成完成 =======")
    print(f"数据已保存到目录: {args.output_dir}")
    print(f"- {prefix}generated_time_series2.csv")
    print(f"- {prefix}causal_adjacency_matrix2.csv")
    print(f"- {prefix}causal_weights_matrix2.csv")
    if lag_matrix is not None:
        print(f"- {prefix}lag_matrix.csv")

    # 评估数据质量
    quality_metrics = evaluate_time_series(time_series, adj_matrix)

    # 可视化结果
    if not args.no_visualization:
        visualize_results(time_series, adj_matrix, weights_matrix, lag_matrix, output_dir=args.output_dir)
        print(f"可视化结果已保存为 {args.output_dir}/visualization_nonlinear_results.png")

    # 如果数据质量不佳，提供更明确的建议
    if quality_metrics['quality_percentage'] < 40:
        print("\n警告: 生成的数据可能不适合因果发现。建议使用以下命令重新生成:")
        cmd = f"python scripts/Nolinear-data.py --n_nodes {n_nodes} --n_time_steps {n_time_steps * 2} --average_degree {min(average_degree * 1.2, 3.0):.1f} --noise_scale {args.noise_scale / 2:.3f}"
        print(f"\n{cmd}\n")


# 评估时间序列数据的质量
def evaluate_time_series(time_series, adj_matrix):
    """
    评估生成的时间序列数据质量，分析其是否适合因果发现

    参数:
    time_series: 生成的时间序列数据
    adj_matrix: 真实的邻接矩阵（因果关系）

    返回:
    各种评估指标
    """
    n_nodes = adj_matrix.shape[0]
    n_samples = time_series.shape[0]

    print("\n======= 数据质量评估 =======")

    # 1. 检查数据稳定性
    means = np.mean(time_series, axis=0)
    stds = np.std(time_series, axis=0)
    print(f"数据均值范围: [{np.min(means):.4f}, {np.max(means):.4f}]")
    print(f"数据标准差范围: [{np.min(stds):.4f}, {np.max(stds):.4f}]")

    # 2. 检查信噪比 - 计算自相关系数作为信号强度估计
    acf_values = []
    for i in range(n_nodes):
        # 计算lag=1的自相关系数
        series = time_series[:, i]
        acf = np.corrcoef(series[:-1], series[1:])[0, 1]
        acf_values.append(acf)

    avg_acf = np.mean(np.abs(acf_values))
    print(f"平均自相关系数（信号强度估计）: {avg_acf:.4f}")

    # 3. 估计因果强度 - 简单检查存在因果关系的节点对之间的相关性
    causal_corrs = []
    non_causal_corrs = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            # 略过自相关
            if i == j:
                continue

            # 计算延迟相关
            corr = np.corrcoef(time_series[:-1, i], time_series[1:, j])[0, 1]

            # 区分有因果关系和无因果关系的节点对
            if adj_matrix[i, j] > 0:
                causal_corrs.append(abs(corr))
            else:
                non_causal_corrs.append(abs(corr))

    avg_causal_corr = np.mean(causal_corrs) if causal_corrs else 0
    avg_non_causal_corr = np.mean(non_causal_corrs) if non_causal_corrs else 0

    print(f"有因果关系的节点对平均相关性: {avg_causal_corr:.4f}")
    print(f"无因果关系的节点对平均相关性: {avg_non_causal_corr:.4f}")
    print(f"因果信号与噪声对比度: {avg_causal_corr / max(avg_non_causal_corr, 0.001):.4f}")

    # 4. 数据非线性度评估 - 使用简单的线性vs非线性拟合比较
    nonlinearity_scores = []

    for j in range(n_nodes):
        # 找出影响节点j的所有节点i
        causes = []
        for i in range(n_nodes):
            if adj_matrix[i, j] > 0:
                causes.append(i)

        if not causes:
            continue

        # 对每个受影响的节点计算非线性度
        y = time_series[1:, j]  # 目标变量在t时刻

        # 线性模型的预测变量 - 所有原因在t-1时刻
        X = time_series[:-1, causes]

        # 计算线性模型的R²
        if X.shape[0] > 0 and X.shape[1] > 0:
            # 添加常数列
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])

            try:
                # 计算线性系数
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

                # 预测值
                y_pred = X_with_const @ beta

                # 计算R²
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r2_linear = 1 - ss_residual / ss_total if ss_total > 0 else 0

                # 简单的非线性模型：平方项
                X_nonlinear = np.column_stack([X_with_const, X ** 2])

                # 计算非线性系数
                beta_nonlinear = np.linalg.lstsq(X_nonlinear, y, rcond=None)[0]

                # 预测值
                y_pred_nonlinear = X_nonlinear @ beta_nonlinear

                # 计算R²
                ss_residual_nonlinear = np.sum((y - y_pred_nonlinear) ** 2)
                r2_nonlinear = 1 - ss_residual_nonlinear / ss_total if ss_total > 0 else 0

                # 非线性提升
                nonlinearity_scores.append(max(0, r2_nonlinear - r2_linear))
            except:
                # 跳过计算问题
                pass

    avg_nonlinearity = np.mean(nonlinearity_scores) if nonlinearity_scores else 0
    print(f"平均非线性度评分: {avg_nonlinearity:.4f}")

    # 5. 总体评估
    signal_noise_ratio = avg_causal_corr / max(avg_non_causal_corr, 0.001)

    quality_score = 0
    if signal_noise_ratio > 5:
        quality_score += 3
    elif signal_noise_ratio > 2:
        quality_score += 2
    elif signal_noise_ratio > 1:
        quality_score += 1

    if avg_nonlinearity > 0.3:
        quality_score += 3
    elif avg_nonlinearity > 0.1:
        quality_score += 2
    elif avg_nonlinearity > 0.05:
        quality_score += 1

    if avg_acf > 0.7:
        quality_score += 3
    elif avg_acf > 0.5:
        quality_score += 2
    elif avg_acf > 0.3:
        quality_score += 1

    max_score = 9
    quality_percentage = quality_score / max_score * 100

    print(f"\n数据质量评分: {quality_score}/{max_score} ({quality_percentage:.1f}%)")

    if quality_percentage >= 80:
        print("评估结果: 优秀 - 数据非常适合因果发现")
    elif quality_percentage >= 60:
        print("评估结果: 良好 - 数据适合因果发现")
    elif quality_percentage >= 40:
        print("评估结果: 中等 - 数据可用于因果发现，但可能需要更大的样本量或更多预处理")
    else:
        print("评估结果: 不佳 - 建议调整参数重新生成数据")

    # 提供改进建议
    print("\n改进建议:")
    if signal_noise_ratio < 2:
        print("- 增大信噪比: 减小 noise_scale 参数，增大影响权重")
    if avg_nonlinearity < 0.1:
        print("- 增强非线性关系: 使用更多样的非线性函数，或增大权重")
    if avg_acf < 0.5:
        print("- 增强时序依赖: 减小噪声，增大自相关")

    return {
        'signal_noise_ratio': signal_noise_ratio,
        'nonlinearity': avg_nonlinearity,
        'autocorrelation': avg_acf,
        'quality_score': quality_score,
        'quality_percentage': quality_percentage
    }


if __name__ == "__main__":
    main() 