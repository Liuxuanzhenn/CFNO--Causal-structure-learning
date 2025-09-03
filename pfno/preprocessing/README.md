# PFNO增强预处理系统

本系统为PFNO模型提供了四种不同级别的时间序列数据预处理方法，替换了原来简单的`normalize`参数，提供更加灵活和强大的数据预处理能力。

## 🎯 四种预处理方法

### 1. **minimal** - 最小预处理
- **描述**: 仅处理缺失值和异常值，不进行标准化
- **适用场景**: 数据质量较好，希望保持原始数据分布的场景
- **主要步骤**:
  - 缺失值插值处理
  - IQR方法检测和处理异常值
  - 不进行数据标准化

### 2. **standard** - 标准预处理（推荐）
- **描述**: 包含基本清理 + Z-score标准化
- **适用场景**: 适用于大多数场景，平衡了数据质量和处理效果
- **主要步骤**:
  - 缺失值插值处理
  - IQR方法检测和处理异常值
  - Z-score标准化
  - 极端值截断（1%-99%分位数）

### 3. **comprehensive** - 全面预处理
- **描述**: 包含所有预处理步骤，包括异常值处理和平稳性检查
- **适用场景**: 噪声较多、需要深度清理的数据
- **主要步骤**:
  - 缺失值插值处理
  - 孤立森林检测异常值
  - Robust标准化（对异常值更鲁棒）
  - 平稳性检查和处理
  - 线性去趋势
  - 数据平滑

### 4. **time_series_focused** - 时间序列专用
- **描述**: 专为时间序列数据设计，包含插值、去趋势等
- **适用场景**: 具有明显时序特征的数据
- **主要步骤**:
  - 三次样条插值处理缺失值
  - Z-score异常值检测
  - Z-score标准化
  - 线性去趋势
  - Butterworth低通滤波

## 🚀 使用方法

### 方法1: 修改配置文件
在`pfno_config.py`中修改预处理方法：

```python
PFNO_DATA_PARAMS = {
    # ... 其他配置 ...
    'preprocessing': {
        'method': 'standard',  # 选择预处理方法
        # ... 详细配置 ...
    }
}
```

### 方法2: 运行时指定
```bash
# 使用标准预处理运行PFNO
python neural_operators.py --mode pfno --script probabilistic_causal --model_type probabilistic

# 或者使用示例脚本测试不同预处理方法
python pfno/preprocessing/example_usage.py --method standard
```

### 方法3: 编程方式使用
```python
from pfno.preprocessing import load_pfno_data
from pfno.pfno_config import PFNO_CONFIG

# 设置预处理方法
config = PFNO_CONFIG.copy()
config['data']['preprocessing']['method'] = 'comprehensive'

# 加载和预处理数据
(train_dataloader, val_dataloader, seq_length, num_nodes, 
 preprocessor, raw_data, preprocessing_info) = load_pfno_data(config)
```

## 📋 配置参数说明

### 通用参数
- `handle_missing`: 是否处理缺失值
- `missing_method`: 缺失值处理方法 ('interpolate', 'spline', 'forward_fill', 'backward_fill', 'mean')
- `handle_outliers`: 是否处理异常值
- `outlier_method`: 异常值检测方法 ('iqr', 'zscore', 'isolation')
- `outlier_threshold`: 异常值阈值
- `normalize`: 是否进行标准化
- `normalization_method`: 标准化方法 ('zscore', 'minmax', 'robust')

### comprehensive方法特有参数
- `check_stationarity`: 是否检查平稳性
- `make_stationary`: 是否进行差分使数据平稳
- `remove_trend`: 是否去除趋势
- `smooth_data`: 是否平滑数据

### time_series_focused方法特有参数
- `detrend`: 是否去趋势
- `apply_filter`: 是否应用滤波器
- `filter_type`: 滤波器类型 ('butterworth', 'moving_average')
- `spline_order`: 样条插值阶数

## 🔧 示例脚本

### 查看所有预处理方法说明
```bash
python pfno/preprocessing/example_usage.py --demo
```

### 测试不同预处理方法
```bash
# 最小预处理
python pfno/preprocessing/example_usage.py --method minimal

# 标准预处理
python pfno/preprocessing/example_usage.py --method standard

# 全面预处理
python pfno/preprocessing/example_usage.py --method comprehensive

# 时间序列专用预处理
python pfno/preprocessing/example_usage.py --method time_series_focused
```

### 使用自定义数据文件
```bash
python pfno/preprocessing/example_usage.py --method standard --data_file path/to/your/data.csv
```

## 📊 输出文件

预处理系统会自动保存以下文件：
- `results/preprocessor.pkl`: 预处理器对象（包含标准化器等）
- `results/preprocessing_info.pkl`: 预处理步骤信息

这些文件可用于后续的数据反变换或分析。

## 🔍 选择建议

- **数据质量好**: 选择 `minimal`
- **一般使用**: 选择 `standard`（推荐）
- **数据噪声多**: 选择 `comprehensive`
- **时序特征明显**: 选择 `time_series_focused`

## 🛠️ 自定义配置

可以通过修改`pfno_config.py`中的预处理配置来自定义处理步骤：

```python
'preprocessing': {
    'method': 'custom',
    'custom': {
        'handle_missing': True,
        'missing_method': 'spline',
        'handle_outliers': True,
        'outlier_method': 'isolation',
        'normalize': True,
        'normalization_method': 'robust',
        # 添加其他自定义参数...
    }
}
```

## 📦 依赖包

确保安装以下依赖包：
```bash
pip install pandas numpy scikit-learn scipy torch
# 对于comprehensive方法的平稳性检查（可选）
pip install statsmodels
```

## 📁 模块结构

```
pfno/preprocessing/
├── __init__.py           # 模块初始化，导出主要功能
├── core.py              # 核心预处理功能
├── data_loader.py       # 增强数据加载器
├── example_usage.py     # 使用示例脚本
└── README.md           # 本文档
```

## 🔗 API 参考

### 主要函数

- `TimeSeriesPreprocessor`: 核心预处理器类
- `preprocess_pfno_data()`: 便捷预处理函数
- `load_pfno_data()`: 完整数据加载和预处理
- `load_experimental_data_enhanced()`: 实验数据加载（兼容旧接口）

### 导入方式

```python
# 导入所有主要功能
from pfno.preprocessing import (
    TimeSeriesPreprocessor,
    preprocess_pfno_data,
    load_pfno_data,
    load_experimental_data_enhanced
)

# 或分别导入
from pfno.preprocessing.core import TimeSeriesPreprocessor
from pfno.preprocessing.data_loader import load_pfno_data
``` 