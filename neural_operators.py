#!/usr/bin/env python3
"""
神经算子统一入口

该脚本提供了统一的入口来运行不同类型的神经算子模型：
- PFNO (概率傅里叶神经算子)
- FNO (确定性傅里叶神经算子)

使用方式:
python neural_operators.py --mode pfno --script probabilistic_causal --model_type probabilistic --args "--num_epochs 10"
python neural_operators.py --mode pfno --script probabilistic_causal --model_type enhanced --args "--num_epochs 10"
python neural_operators.py --mode fno --script main --args "--data_file data.csv --epochs 50"
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_pfno_script(script_name, script_args, model_type=None, extra_args=None):
    """运行PFNO相关脚本"""
    available_scripts = {
        'probabilistic_causal': 'pfno/run_probabilistic_causal.py',
        'run_probabilistic_causal': 'pfno/run_probabilistic_causal.py'
    }
    
    if script_name not in available_scripts:
        print(f"错误: PFNO模块中没有找到脚本 '{script_name}'")
        print(f"可用脚本: {list(available_scripts.keys())}")
        return False
    
    script_path = available_scripts[script_name]
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 脚本文件不存在: {script_path}")
        return False
    
    # 构建运行命令 - 使用-m标志作为模块运行
    # 从脚本路径推导模块名
    module_name = script_path.replace('/', '.').replace('\\', '.').replace('.py', '')
    cmd = [sys.executable, '-m', module_name]
    
    # 添加模型类型参数（如果指定了的话）
    if model_type:
        # 将模型类型信息传递给脚本
        # 注意：run_probabilistic_causal.py会根据配置自动选择模型类型
        # 这里我们可以通过环境变量或其他方式传递信息
        os.environ['PFNO_MODEL_TYPE'] = model_type
    
    # 添加其他脚本参数
    if script_args:
        cmd.extend(script_args)
    
    # 添加额外参数
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"执行PFNO脚本: {script_name}")
    print(f"模型类型: {model_type or 'auto'}")
    print(f"脚本路径: {script_path}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    except Exception as e:
        print(f"运行失败: {e}")
        return False

def run_fno_script(script_name, script_args):
    """运行FNO相关脚本"""
    available_scripts = {
        'main': 'fno/main.py',
        'train': 'fno/train.py',
        'al_trainer': 'fno/al_trainer.py'
    }
    
    if script_name not in available_scripts:
        print(f"错误: FNO模块中没有找到脚本 '{script_name}'")
        print(f"可用脚本: {list(available_scripts.keys())}")
        return False
    
    script_path = available_scripts[script_name]
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 脚本文件不存在: {script_path}")
        return False
    
    cmd = [sys.executable, script_path] + script_args
    
    print(f"执行FNO脚本: {script_name}")
    print(f"脚本路径: {script_path}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    except Exception as e:
        print(f"运行失败: {e}")
        return False

def list_available_scripts():
    """列出所有可用的脚本"""
    print("=== 可用的神经算子脚本 ===\n")
    
    print("PFNO (概率傅里叶神经算子):")
    print("  - probabilistic_causal: 概率因果发现")
    print("    * probabilistic: 原始概率模型")
    print("    * enhanced: 改进概率模型 (集成多时间尺度)")
    print("  - run_probabilistic_causal: 概率因果发现(完整版)")
    print("    * probabilistic: 原始概率模型")
    print("    * enhanced: 改进概率模型 (集成多时间尺度)")
    print()
    
    print("FNO (确定性傅里叶神经算子):")
    print("  - main: 主要训练脚本")
    print("  - train: 标准训练脚本")
    print("  - al_trainer: 主动学习训练器")
    print()
    
    print("使用示例:")
    print("  # 原始概率模型")
    print("  python neural_operators.py --mode pfno --script probabilistic_causal --model_type probabilistic")
    print()
    print("  # 改进概率模型 (多时间尺度)")
    print("  python neural_operators.py --mode pfno --script probabilistic_causal --model_type enhanced")
    print()
    print("  # FNO确定性模型")
    print("  python neural_operators.py --mode fno --script main --args '--data_file data.csv --epochs 50'")

def main():
    parser = argparse.ArgumentParser(
        description='神经算子统一入口脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行概率因果发现 (原始模型)
  python neural_operators.py --mode pfno --script probabilistic_causal --model_type probabilistic
  
  # 运行概率因果发现 (改进模型 - 多时间尺度)
  python neural_operators.py --mode pfno --script probabilistic_causal --model_type enhanced
  
  # 运行确定性FNO训练
  python neural_operators.py --mode fno --script main --args "--data_file data.csv --epochs 50"
  
  # 列出所有可用脚本
  python neural_operators.py --list
        """
    )
    
    parser.add_argument('--mode', choices=['pfno', 'fno'], 
                       help='选择神经算子类型: pfno (概率) 或 fno (确定性)')
    parser.add_argument('--script', type=str,
                       help='要运行的脚本名称')
    parser.add_argument('--model_type', type=str, choices=['probabilistic', 'enhanced'],
                       help='PFNO模型类型: probabilistic (原始), enhanced (改进-多时间尺度)')
    parser.add_argument('--args', type=str, default='',
                       help='传递给脚本的额外参数 (用引号包围)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的脚本')
    
    args = parser.parse_args()
    
    # 如果请求列表，显示可用脚本并退出
    if args.list:
        list_available_scripts()
        return
    
    # 检查必需参数
    if not args.mode or not args.script:
        print("错误: 必须指定 --mode 和 --script 参数")
        print("使用 --help 查看帮助，或使用 --list 查看可用脚本")
        return
    
    # 解析额外脚本参数
    if args.args:
        import shlex
        extra_args = shlex.split(args.args)
    else:
        extra_args = []
    
    # 运行对应的脚本
    success = False
    if args.mode == 'pfno':
        success = run_pfno_script(args.script, [], args.model_type, extra_args)
    elif args.mode == 'fno':
        success = run_fno_script(args.script, extra_args)
    
    if success:
        print("\n✅ 脚本执行成功!")
    else:
        print("\n❌ 脚本执行失败!")
        sys.exit(1)

if __name__ == '__main__':
    main() 