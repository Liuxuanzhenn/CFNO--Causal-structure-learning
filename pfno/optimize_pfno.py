import optuna
import yaml
import subprocess
import os
import sys
import json
import datetime
from pathlib import Path

# 确保项目根目录在python路径中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_base_config(config_path='pfno/pfno_config.yaml'):
    """从YAML文件加载基础配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def update_config(base_config, params):
    """使用新参数递归更新基础配置。"""
    config = base_config.copy()
    for key, value in params.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config

def objective(trial, base_config):
    """Optuna优化的目标函数。"""
    
    # 1. 建议超参数
    params = {
        "architecture": {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [64, 128, 256]),
            "lifting_channels": trial.suggest_categorical("lifting_channels", [128, 256]),
            "projection_channels": trial.suggest_categorical("projection_channels", [128, 256]),
        },
        "data": {
            "sequence_length": trial.suggest_categorical("sequence_length", [10, 30, 50]),
        },
        "training": {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.001, 0.0005, 0.0002]),
            "batch_size": trial.suggest_categorical("batch_size", [100, 200]),
        },
    }

    # 遵循pfno_config.yaml的结构，将loss_weights放在training下
    params["training"]["loss_weights"] = {
        "l1_regularization": trial.suggest_categorical("l1_regularization", [0.001, 0.0005, 0.0001]),
        "dag_constraint": trial.suggest_categorical("dag_constraint", [0.02, 0.1, 0.05]),
    }

    # 处理dist/direct权重
    dist_direct_options_str = ["1.0,1.0", "0.4,0.6", "0.6,0.4"]
    selected_str = trial.suggest_categorical("dist_direct_weights", dist_direct_options_str)
    dist_weight, direct_weight = map(float, selected_str.split(','))
    
    params["training"]["loss_weights"]["reconstruction_weights"] = {
        "dist": dist_weight,
        "direct": direct_weight,
    }

    # 2. 创建特定于试验的配置
    trial_config = update_config(base_config, params)
    
    # 为此试验的产物创建唯一目录
    run_id = f"trial_{trial.number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trial_dir = os.path.join("results", "optuna_runs", run_id)
    os.makedirs(trial_dir, exist_ok=True)
    
    config_path = os.path.join(trial_dir, "config.yaml")
    output_json_path = os.path.join(trial_dir, "metrics.json")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(trial_config, f, default_flow_style=False)
        
    # 3. 作为子进程运行训练脚本
    # 使用sys.executable确保使用相同的Python解释器。
    # 使用-m运行模块以正确处理Python路径。
    command = [
        sys.executable,
        "-m", "pfno.run_probabilistic_causal",
        "--config", config_path,
        "--output_json", output_json_path,
        "--no-viz"
    ]
    
    print(f"\n--- 开始试验 {trial.number} ---")
    print(f"命令: {' '.join(command)}")
    
    try:
        # 不捕获输出，以便在控制台中实时查看。
        subprocess.run(command, check=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"试验 {trial.number} 失败，错误代码 {e.returncode}。")
        # 如果试验失败，则进行剪枝
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"试验 {trial.number} 期间发生意外错误: {e}")
        raise optuna.exceptions.TrialPruned()

    # 4. 读取结果
    if not os.path.exists(output_json_path):
        print(f"未找到试验 {trial.number} 的指标文件于 {output_json_path}")
        return 0.0

    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        f1_score = metrics.get('best_f1_score', 0.0)
        print(f"--- 试验 {trial.number} 完成 ---")
        print(f"F1 分数: {f1_score}")
        print(f"---------------------------------\n")
        return f1_score
    except (json.JSONDecodeError, KeyError) as e:
        print(f"读取试验 {trial.number} 的指标时出错: {e}")
        return 0.0

if __name__ == "__main__":
    # 定义研究
    study_name = "pfno-optimization"
    # 将数据库文件保存在项目根目录
    storage_name = f"sqlite:///{project_root.joinpath('pfno_study.db').as_posix()}"
    
    # 一次性加载基础配置
    base_config = load_base_config()
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # 添加剪枝器
    )
    
    # 开始优化
    try:
        study.optimize(lambda trial: objective(trial, base_config), n_trials=200) # 设置试验次数
    except KeyboardInterrupt:
        print("优化被用户停止。")

    # 打印结果
    print("\n--- 优化完成 ---")
    print(f"研究统计: ")
    print(f"  完成的试验次数: {len(study.trials)}")
    
    try:
        best_trial = study.best_trial
        print(f"最佳试验:")
        print(f"  值 (F1 分数): {best_trial.value}")
        print(f"  参数: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("未找到最佳试验。可能是因为所有试验都失败了。")
        
    # 提供仪表板说明
    print("\n--- 要查看仪表板，请在您的终端中运行以下命令: ---")
    print(f"optuna-dashboard {storage_name}") 