import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# --- 设置Matplotlib支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 定义参数 ---
NUM_NODES = 10  # 节点数量
TRUE_EDGES_COUNT = 8  # 真实因果图中边的数量 (减少以使图形更稀疏)
# 定义各图层的垂直偏移量
GAP_PRIOR = 3.0
GAP_PREDICTION = 6.0

# --- 2. 生成与原脚本一致的数据 ---
# 真实因果强度矩阵 (Ground Truth)
true_matrix = np.zeros((NUM_NODES, NUM_NODES))
upper_tri_edges = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i < j]
num_edges = min(TRUE_EDGES_COUNT, len(upper_tri_edges))
upper_indices = np.random.choice(len(upper_tri_edges), num_edges, replace=False)
selected_edges = [upper_tri_edges[i] for i in upper_indices]
for i, j in selected_edges:
    true_matrix[i, j] = np.random.uniform(0.4, 0.75)

# 外部先验知识矩阵
prior_matrix = np.zeros((NUM_NODES, NUM_NODES))
true_edge_locations = np.argwhere(true_matrix > 0)

# 从真实矩阵为0的位置中寻找虚假边的候选位置
# 关键约束：只从上三角部分选取虚假边 (i < j)，以保证先验图的无环性 (DAG)
false_edge_locations = np.array([loc for loc in np.argwhere(true_matrix == 0) if loc[0] < loc[1]])

# 随机选择一半的真实边
num_true_edges_for_prior = len(true_edge_locations) // 2
if num_true_edges_for_prior > 0:
    true_indices_for_prior = np.random.choice(len(true_edge_locations), num_true_edges_for_prior, replace=False)
    for idx in true_indices_for_prior:
        i, j = true_edge_locations[idx]
        prior_matrix[i, j] = 1

# 添加与所含真实边数量相等的虚假边
num_false_edges_for_prior = num_true_edges_for_prior

if len(false_edge_locations) == 0:
    print("警告：没有可用的位置来添加虚假的非循环边。")
elif num_false_edges_for_prior > 0:
    num_false_to_add = min(num_false_edges_for_prior, len(false_edge_locations))
    false_indices_for_prior = np.random.choice(len(false_edge_locations), num_false_to_add, replace=False)
    for idx in false_indices_for_prior:
        i, j = false_edge_locations[idx]
        prior_matrix[i, j] = 1

# 预测得到的加权邻接矩阵
predicted_matrix = true_matrix.copy()
noise = np.random.normal(0, 0.05, predicted_matrix.shape)
predicted_matrix += noise
missed_edge_loc = true_edge_locations[np.random.choice(len(true_edge_locations))]
predicted_matrix[missed_edge_loc[0], missed_edge_loc[1]] = np.random.uniform(0, 0.1)
false_positive_loc = false_edge_locations[np.random.choice(len(false_edge_locations))]
predicted_matrix[false_positive_loc[0], false_positive_loc[1]] = np.random.uniform(0.4, 0.6)
predicted_matrix = np.clip(predicted_matrix, 0, 1)
np.fill_diagonal(predicted_matrix, 0)

# --- 3. 绘制3D节点图 ---
fig = plt.figure(figsize=(18, 16))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('3D Causal Node-Graph Analysis', fontsize=24, y=0.92)

# 定义节点在XY平面的位置（圆形排列）
radius = 5
angles = np.linspace(0, 2 * np.pi, NUM_NODES, endpoint=False)
node_x = radius * np.cos(angles)
node_y = radius * np.sin(angles)

# 定义各层的高度和颜色（与曲面图匹配）
layer_specs = {
    'truth': {'z': 0, 'color': 'grey', 'label': 'Ground Truth Layer'},
    'prior': {'z': GAP_PRIOR, 'color': 'beige', 'label': 'Prior Layer'},
    'prediction': {'z': GAP_PREDICTION, 'color': 'cornflowerblue', 'label': 'Prediction Layer'}
}

# 绘制节点和标签
for spec in layer_specs.values():
    ax.scatter(node_x, node_y, spec['z'], c=spec['color'], s=200, depthshade=False, edgecolors='k')
    for i in range(NUM_NODES):
        ax.text(node_x[i] * 1.15, node_y[i] * 1.15, spec['z'], str(i), fontsize=10, ha='center', va='center', color='k')

# --- 3D箭头类定义 ---
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# 定义一个绘制带箭头边的辅助函数
def draw_edge_with_arrow(ax, start_idx, end_idx, z_level, color, alpha=0.8, linewidth_scale=1.0):
    x_start, y_start = node_x[start_idx], node_y[start_idx]
    x_end, y_end = node_x[end_idx], node_y[end_idx]

    # 稍微缩短箭头，使其不与节点球重叠
    vec_x = x_end - x_start
    vec_y = y_end - y_start
    length = np.sqrt(vec_x**2 + vec_y**2)
    
    node_radius = 0.35  # 视觉上的节点半径
    if length > 0:
        x_start_new = x_start + vec_x * (node_radius / length)
        y_start_new = y_start + vec_y * (node_radius / length)
        x_end_new = x_end - vec_x * (node_radius / length)
        y_end_new = y_end - vec_y * (node_radius / length)
    else:
        return # 避免绘制零长度箭头

    arrow = Arrow3D([x_start_new, x_end_new], [y_start_new, y_end_new], [z_level, z_level],
                    mutation_scale=15, lw=linewidth_scale*1.5, arrowstyle="-|>", color=color, alpha=alpha)
    ax.add_artist(arrow)

# 绘制边
# 1. 真实因果层
z_truth = layer_specs['truth']['z']
color_truth = layer_specs['truth']['color']
for i, j in np.argwhere(true_matrix > 0):
    draw_edge_with_arrow(ax, i, j, z_truth, color_truth, linewidth_scale=true_matrix[i, j])

# 识别出先验中错误存在的边
false_prior_indices = set(map(tuple, np.argwhere((prior_matrix == 1) & (true_matrix == 0))))

# 2. 先验知识层 (区分正确和虚假的先验)
z_prior = layer_specs['prior']['z']
color_prior_correct = 'darkgoldenrod'
color_prior_false = 'red'
for i, j in np.argwhere(prior_matrix > 0):
    # 如果这条边是虚假的，就用红色表示
    is_false_prior = (i, j) in false_prior_indices
    edge_color = color_prior_false if is_false_prior else color_prior_correct
    draw_edge_with_arrow(ax, i, j, z_prior, edge_color, linewidth_scale=1.0)

# 3. 预测结果层（包含纠错逻辑）
z_pred = layer_specs['prediction']['z']
color_pred = layer_specs['prediction']['color']

# 绘制预测边
for i, j in np.argwhere(predicted_matrix > 0.15): # 使用阈值过滤掉非常弱的连接
    draw_edge_with_arrow(ax, i, j, z_pred, color_pred, linewidth_scale=predicted_matrix[i, j])

# 标注被纠正的虚假先验
for i, j in false_prior_indices:
    if predicted_matrix[i, j] < 0.15: # 如果预测值很低，说明成功纠正
        x, y, z = node_x[i], node_y[i], z_pred
        ax.text(x, y, z + 0.3, 'X', color='red', fontsize=12, ha='center', va='center', weight='bold')

# --- 设置坐标轴和视角 ---
ax.set_xlabel('X coordinate', fontsize=14, labelpad=15)
ax.set_ylabel('Y coordinate', fontsize=14, labelpad=15)
ax.set_zlabel('Layer', fontsize=14, labelpad=10)
ax.set_xlim(-radius*1.2, radius*1.2)
ax.set_ylim(-radius*1.2, radius*1.2)
ax.set_zlim(-1, GAP_PREDICTION + 2)

# 设置Z轴刻度标签
ax.set_zticks([spec['z'] for spec in layer_specs.values()])
ax.set_zticklabels([spec['label'] for spec in layer_specs.values()], fontsize=10)

ax.grid(True, linestyle='--', alpha=0.3) # 调低网格透明度
ax.view_init(elev=20, azim=-75)
ax.dist = 11

# --- 创建图例 ---
truth_patch = mpatches.Patch(color=layer_specs['truth']['color'], label='Ground Truth Edge')
prior_true_line = mlines.Line2D([], [], color='darkgoldenrod', linestyle='-', linewidth=2, label='True Prior Edge')
prior_false_line = mlines.Line2D([], [], color='red', linestyle='-', linewidth=2, label='False Prior Edge')
pred_patch = mpatches.Patch(color=layer_specs['prediction']['color'], label='Predicted Edge')
corrected_marker = mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=7, label='Correction of False Prior')

fig.legend(handles=[truth_patch, prior_true_line, prior_false_line, pred_patch, corrected_marker],
           bbox_to_anchor=(0.95, 0.25),
           loc='lower right',
           fontsize=12,
           frameon=True,
           title="Legend",
           title_fontsize=13)

plt.tight_layout(rect=[0, 0.05, 0.85, 0.93])
# 保存图像
output_filename = 'fno_3d_nodegraph_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n已生成3D节点图并保存为: {output_filename}")
plt.show() 