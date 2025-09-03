import pandas as pd
import numpy as np

class CausalData:
    """
    封装因果数据的类，包含邻接矩阵、权重矩阵和时间序列数据
    """
    
    def __init__(self, adjacency_file, weights_file, timeseries_file):
        """
        初始化数据
        
        Parameters:
        -----------
        adjacency_file : str
            因果邻接矩阵文件路径
        weights_file : str
            因果权重矩阵文件路径
        timeseries_file : str
            时间序列数据文件路径
        """
        self.adjacency_matrix = self._load_matrix(adjacency_file)
        self.weights_matrix = self._load_matrix(weights_file)
        self.timeseries_data = self._load_timeseries(timeseries_file)
        
        # 验证数据一致性
        self._validate_data()
        
        # 获取节点信息
        self.num_nodes = self.adjacency_matrix.shape[0]
        self.node_names = list(self.adjacency_matrix.columns)
        self.num_timepoints = len(self.timeseries_data)
        
    def _load_matrix(self, filepath):
        """加载矩阵数据"""
        return pd.read_csv(filepath, index_col=0)
    
    def _load_timeseries(self, filepath):
        """加载时间序列数据"""
        return pd.read_csv(filepath, header=None)
    
    def _validate_data(self):
        """验证数据一致性"""
        # 检查矩阵维度
        assert self.adjacency_matrix.shape == self.weights_matrix.shape, \
            "邻接矩阵和权重矩阵维度不一致"
        
        # 检查时间序列数据的节点数
        assert self.timeseries_data.shape[1] == self.adjacency_matrix.shape[0], \
            "时间序列数据的节点数与矩阵维度不一致"
        
        # 检查邻接矩阵和权重矩阵的一致性（权重为0的地方，邻接矩阵也应该为0）
        weights_nonzero = (self.weights_matrix != 0).astype(int)
        assert np.array_equal(weights_nonzero.values, self.adjacency_matrix.values), \
            "邻接矩阵和权重矩阵的非零位置不一致"
    
    def get_causal_relationships(self):
        """获取因果关系列表"""
        relationships = []
        for i, source in enumerate(self.node_names):
            for j, target in enumerate(self.node_names):
                if self.adjacency_matrix.iloc[i, j] == 1:
                    weight = self.weights_matrix.iloc[i, j]
                    relationships.append({
                        'source': source,
                        'target': target,
                        'weight': weight
                    })
        return relationships
    
    def get_node_timeseries(self, node_name):
        """获取特定节点的时间序列数据"""
        if isinstance(node_name, str):
            node_index = self.node_names.index(node_name)
        else:
            node_index = node_name
        return self.timeseries_data.iloc[:, node_index]
    
    def get_summary(self):
        """获取数据摘要"""
        causal_edges = np.sum(self.adjacency_matrix.values)
        summary = {
            'num_nodes': self.num_nodes,
            'node_names': self.node_names,
            'num_timepoints': self.num_timepoints,
            'num_causal_edges': causal_edges,
            'adjacency_matrix_shape': self.adjacency_matrix.shape,
            'weights_matrix_shape': self.weights_matrix.shape,
            'timeseries_shape': self.timeseries_data.shape
        }
        return summary
    
    def save_data(self, output_dir='./'):
        """保存所有数据到指定目录"""
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        self.adjacency_matrix.to_csv(os.path.join(output_dir, 'adjacency_matrix.csv'))
        self.weights_matrix.to_csv(os.path.join(output_dir, 'weights_matrix.csv'))
        self.timeseries_data.to_csv(os.path.join(output_dir, 'timeseries_data.csv'), 
                                   header=self.node_names, index=False)
        
        # 保存因果关系列表
        relationships_df = pd.DataFrame(self.get_causal_relationships())
        relationships_df.to_csv(os.path.join(output_dir, 'causal_relationships.csv'), index=False)
        
        # 保存摘要信息
        summary = self.get_summary()
        with open(os.path.join(output_dir, 'data_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("数据摘要\n")
            f.write("="*50 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")


def load_causal_data():
    """
    加载因果数据
    
    Returns:
    --------
    CausalData: 封装了所有数据的对象
    """
    # 创建数据对象
    data = CausalData(
        adjacency_file='causal_adjacency_matrix2.csv',
        weights_file='causal_weights_matrix2.csv',
        timeseries_file='generated_time_series2.csv'
    )
    
    return data


if __name__ == "__main__":
    # 加载数据
    print("正在加载数据...")
    data = load_causal_data()
    
    # 打印摘要信息
    print("\n数据摘要:")
    print("="*50)
    summary = data.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 显示因果关系
    print(f"\n因果关系 (共{summary['num_causal_edges']}条):")
    print("-"*50)
    relationships = data.get_causal_relationships()
    for rel in relationships:
        print(f"{rel['source']} -> {rel['target']} (权重: {rel['weight']:.4f})")
    
    # 显示时间序列数据的前5个时间点
    print(f"\n时间序列数据预览 (前5个时间点):")
    print("-"*50)
    print(data.timeseries_data.head())
    
    # 保存数据
    print("\n正在保存封装的数据...")
    data.save_data('./packaged_data/')
    print("数据已保存到 './packaged_data/' 目录")
    
    print("\n数据封装完成！") 