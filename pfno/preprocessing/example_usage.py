#!/usr/bin/env python3
"""
PFNO增强预处理系统使用示例

演示如何使用四种不同的预处理方法：
1. minimal - 最小预处理
2. standard - 标准预处理（推荐）
3. comprehensive - 全面预处理
4. time_series_focused - 时间序列专用
"""

import sys
import os
# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pfno.pfno_config import get_pfno_config, PFNO_CONFIG
import argparse


def demonstrate_preprocessing_methods():
    """演示不同的预处理方法"""
    
    print("🎯 PFNO增强预处理系统演示")
    print("="*60)
    
    # 获取基础配置
    base_config = PFNO_CONFIG.copy()
    
    # 预处理方法说明
    preprocessing_methods = {
        'minimal': {
            'name': '最小预处理',
            'description': '仅处理缺失值和异常值，不进行标准化',
            'use_case': '适用于数据质量较好，希望保持原始数据分布的场景'
        },
        'standard': {
            'name': '标准预处理（推荐）',
            'description': '包含基本清理 + Z-score标准化',
            'use_case': '适用于大多数场景，平衡了数据质量和处理效果'
        },
        'comprehensive': {
            'name': '全面预处理',
            'description': '包含所有预处理步骤，包括异常值处理和平稳性检查',
            'use_case': '适用于噪声较多、需要深度清理的数据'
        },
        'time_series_focused': {
            'name': '时间序列专用',
            'description': '专为时间序列数据设计，包含插值、去趋势等',
            'use_case': '适用于具有明显时序特征的数据'
        }
    }
    
    for method, info in preprocessing_methods.items():
        print(f"\n📋 {info['name']} ({method})")
        print(f"   描述: {info['description']}")
        print(f"   适用场景: {info['use_case']}")
        
        # 显示该方法的详细配置
        method_config = base_config['data']['preprocessing'][method]
        print(f"   主要参数:")
        
        if method_config.get('handle_missing'):
            print(f"     • 缺失值处理: {method_config.get('missing_method', 'interpolate')}")
        
        if method_config.get('handle_outliers'):
            print(f"     • 异常值处理: {method_config.get('outlier_method', 'iqr')} (阈值: {method_config.get('outlier_threshold', 3.0)})")
        
        if method_config.get('normalize'):
            print(f"     • 标准化: {method_config.get('normalization_method', 'zscore')}")
        else:
            print(f"     • 标准化: 无")
            
        # 特殊处理步骤
        special_steps = []
        if method == 'comprehensive':
            if method_config.get('check_stationarity'):
                special_steps.append("平稳性检查")
            if method_config.get('remove_trend'):
                special_steps.append("去趋势")
            if method_config.get('smooth_data'):
                special_steps.append("数据平滑")
                
        elif method == 'time_series_focused':
            if method_config.get('detrend'):
                special_steps.append("去趋势")
            if method_config.get('apply_filter'):
                special_steps.append(f"滤波 ({method_config.get('filter_type', 'butterworth')})")
            if method_config.get('missing_method') == 'spline':
                special_steps.append("样条插值")
        
        if special_steps:
            print(f"     • 特殊步骤: {', '.join(special_steps)}")


def run_preprocessing_example(method='standard', data_file=None):
    """运行预处理示例"""
    
    print(f"\n🚀 运行 '{method}' 预处理示例")
    print("="*50)
    
    # 获取配置并设置预处理方法
    config = PFNO_CONFIG.copy()
    config['data']['preprocessing']['method'] = method
    
    # 如果指定了数据文件，更新路径
    if data_file:
        config['data']['file_path'] = data_file
        print(f"📂 使用数据文件: {data_file}")
    else:
        print(f"📂 使用默认数据文件: {config['data']['file_path']}")
    
    # 检查数据文件是否存在
    if not os.path.exists(config['data']['file_path']):
        print(f"❌ 数据文件不存在: {config['data']['file_path']}")
        print("请确保data目录下有generated_time_series2.csv文件，或使用--data_file指定其他文件")
        return
    
    try:
        # 导入并运行预处理 - 使用新的路径
        from pfno.preprocessing.data_loader import load_pfno_data
        
        # 运行数据加载和预处理
        (train_dataloader, val_dataloader, seq_length, num_nodes, 
         preprocessor, raw_data, preprocessing_info) = load_pfno_data(config)
        
        print(f"\n✅ 预处理成功完成!")
        print(f"   • 预处理方法: {preprocessing_info['method']}")
        print(f"   • 执行步骤数: {len(preprocessing_info['steps'])}")
        print(f"   • 最终数据维度: 节点数={num_nodes}, 序列长度={seq_length}")
        print(f"   • 训练批次: {len(train_dataloader)}, 验证批次: {len(val_dataloader)}")
        
        # 显示预处理步骤详情
        print(f"\n📋 预处理步骤详情:")
        for i, step in enumerate(preprocessing_info['steps'], 1):
            print(f"   {i}. {step}")
            
        # 保存预处理器信息
        if hasattr(preprocessor, 'scalers') and preprocessor.scalers:
            print(f"\n💾 标准化器信息:")
            print(f"   • 已为 {len(preprocessor.scalers)} 个节点创建标准化器")
            
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已正确安装所需依赖包")
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PFNO增强预处理系统演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 显示所有预处理方法说明
  python pfno/preprocessing/example_usage.py --demo
  
  # 运行标准预处理示例
  python pfno/preprocessing/example_usage.py --method standard
  
  # 运行时间序列专用预处理
  python pfno/preprocessing/example_usage.py --method time_series_focused --data_file data/your_data.csv
  
  # 运行全面预处理
  python pfno/preprocessing/example_usage.py --method comprehensive
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='显示所有预处理方法的说明')
    parser.add_argument('--method', type=str, 
                       choices=['minimal', 'standard', 'comprehensive', 'time_series_focused'],
                       default='standard',
                       help='要演示的预处理方法')
    parser.add_argument('--data_file', type=str,
                       help='数据文件路径（可选，默认使用配置中的路径）')
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_preprocessing_methods()
    else:
        success = run_preprocessing_example(args.method, args.data_file)
        if success:
            print(f"\n🎉 '{args.method}' 预处理演示成功完成!")
        else:
            print(f"\n💥 '{args.method}' 预处理演示失败!")
            sys.exit(1)


if __name__ == '__main__':
    main() 