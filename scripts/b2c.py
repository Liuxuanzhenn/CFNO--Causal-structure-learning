import numpy as np
import pandas as pd
from scipy import stats
import os

def get_nonlinear_equations_with_confounder(n_vars=10, max_lag=2, generator=None):
    """根据生成器内部的随机参数，动态生成非线性方程的描述字符串"""
    if generator is None or generator.coeffs is None:
        return ["生成器未初始化，请先运行数据生成。"]

    equations = []
    for i in range(n_vars):
        source_node = (i - 1 + n_vars) % n_vars
        lag = generator.lags[i]
        c1, c2, c_u = generator.coeffs[i]
        
        formula = f"X{i+1}[t] = {c1:.2f}*X{source_node+1}[t-{lag}]**2 + {c2:.2f}*X{source_node+1}[t-{lag}]**3"
        if i in generator.confounder_nodes:
            formula += f" + {c_u:.2f}*U[t]**2"
        formula += f" + e{i+1}"
        equations.append(formula)
    
    equations.append("U[t] = eU")
    return equations

class MixedNonlinearConfoundedGenerator:
    def __init__(self, noise_mix_ratio=0.5, noise_params={'scale': 0.1}, random_state=42):
        self.noise_mix_ratio = noise_mix_ratio
        self.noise_params = noise_params
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        # 将参数初始化为None，以便在首次生成时创建
        self.coeffs = None
        self.lags = None
        self.confounder_nodes = None
        self.n_vars = 0

    def _initialize_params(self, n_vars, max_lag):
        """为给定的节点数和最大滞后随机初始化模型参数"""
        self.n_vars = n_vars
        # 为非线性项和混杂因素生成随机系数
        # 结构: [coeff_x^2, coeff_x^3, coeff_U^2]
        self.coeffs = (np.random.rand(n_vars, 3) - 0.5) * 2  # 范围: [-1, 1]
        self.coeffs[:, 0] *= 0.4  # x^2 项的系数
        self.coeffs[:, 1] *= -0.15 # x^3 项的系数 (通常为负)
        self.coeffs[:, 2] *= 0.5   # U^2 项的系数
        
        # 为每个依赖关系分配滞后
        self.lags = np.random.randint(1, max_lag + 1, size=n_vars)
        
        # 随机选择约1/3的节点作为被混杂的节点
        num_confounded = n_vars // 3
        self.confounder_nodes = np.random.choice(n_vars, size=num_confounded, replace=False)
        # 对非混杂节点，将其混杂系数设为0
        for i in range(n_vars):
            if i not in self.confounder_nodes:
                self.coeffs[i, 2] = 0

    def generate_mixed_noise(self, size):
        mask = np.random.random(size) < self.noise_mix_ratio
        gaussian_noise = np.random.normal(0, self.noise_params['scale'], size=size)
        laplace_noise = np.random.laplace(0, self.noise_params['scale'], size=size)
        return np.where(mask, gaussian_noise, laplace_noise)

    def generate_multivariate_ts(self, n_points=500, n_vars=10, max_lag=2):
        # 如果参数未初始化或节点数发生变化，则重新初始化
        if self.coeffs is None or self.n_vars != n_vars:
            self._initialize_params(n_vars, max_lag)
            
        X = np.zeros((n_points, n_vars))
        U = np.zeros(n_points)

        for i in range(max_lag):
            X[i] = self.generate_mixed_noise(n_vars)
            U[i] = self.generate_mixed_noise(1)[0]

        for t in range(max_lag, n_points):
            # 生成混杂因素 U 和噪声
            U[t] = self.generate_mixed_noise(1)[0]
            noise = self.generate_mixed_noise(n_vars)

            # 由于所有滞后阶数>=1，我们可以在一个循环内安全地计算所有X[t]
            for i in range(n_vars):
                # 定义循环依赖关系: i 由 (i-1) 导致
                source_node_idx = (i - 1 + n_vars) % n_vars
                lag = self.lags[i]
                c1, c2, c_u = self.coeffs[i]

                source_val = X[t - lag, source_node_idx]
                
                # 计算当前节点的值
                val = c1 * source_val**2 + c2 * source_val**3 + noise[i]
                
                # 如果是混杂节点，则添加混杂因素的影响
                if i in self.confounder_nodes:
                    val += c_u * U[t]**2
                
                X[t, i] = val
        
        columns = [f'X{i+1}' for i in range(n_vars)] + ['U']
        df = pd.DataFrame(np.column_stack([X, U]), columns=columns)
        df['time'] = np.arange(n_points)
        return df

if __name__ == "__main__":
    N_NODES = 10
    MAX_LAG = 3
    
    # 1. 初始化并生成数据
    print(f"--- 正在生成 {N_NODES} 个节点的非线性循环数据... ---")
    generator = MixedNonlinearConfoundedGenerator(random_state=42)
    # 增加数据点数量以进行更稳健的分析
    data_with_confounder_df = generator.generate_multivariate_ts(
        n_points=2000, n_vars=N_NODES, max_lag=MAX_LAG
    )
    
    print("\n--- 生成的数据样本 (包含混杂因素U): ---")
    print(data_with_confounder_df.head())

    # 2. 打印生成的方程
    print("\n--- 生成的因果方程 ---")
    equations = get_nonlinear_equations_with_confounder(N_NODES, MAX_LAG, generator)
    for eq in equations:
        print(eq)
    print(f"被混杂的节点 (0-indexed): {generator.confounder_nodes}")


    # 3. 定义正确的因果邻接矩阵 (A[i,j]=1 表示 i -> j)
    # 因果链: X1->X2, X2->X3, ..., X10->X1
    # 0-indexed: 0->1, 1->2, ..., 9->0
    # 在我们的生成逻辑中，是(i-1)导致i，所以是 (i-1)->i
    causal_adj_matrix = np.zeros((N_NODES, N_NODES), dtype=int)
    for i in range(N_NODES):
        # X[i]依赖于X[i-1]
        source_node_for_i = (i - 1 + N_NODES) % N_NODES
        causal_adj_matrix[source_node_for_i, i] = 1


    # 4. 准备保存数据
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n目录 '{output_dir}' 已创建。")

    # 提取不含混杂因素U的时间序列数据
    ts_columns = [f'X{i+1}' for i in range(N_NODES)]
    time_series_df = data_with_confounder_df[ts_columns]

    # 将因果矩阵转换为带索引的DataFrame
    adj_matrix_df = pd.DataFrame(causal_adj_matrix,
                                 columns=[f'Node_{i}' for i in range(N_NODES)],
                                 index=[f'Node_{i}' for i in range(N_NODES)])

    # 5. 保存文件，与其它脚本格式保持一致
    ts_path = os.path.join(output_dir, 'generated_time_series2.csv')
    adj_path = os.path.join(output_dir, 'causal_adjacency_matrix2.csv')

    time_series_df.to_csv(ts_path, index=False, header=False)
    adj_matrix_df.to_csv(adj_path)

    print("\n--- 数据保存成功 ---")
    print(f"时间序列数据已保存至: {ts_path}")
    print(f"因果邻接矩阵已保存至: {adj_path}")
    print("\n--- 因果关系矩阵 (A[i,j]=1 表示 i -> j) ---")
    print("行=来源 (Source), 列=目标 (Target)")
    print(adj_matrix_df)
