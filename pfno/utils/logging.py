"""
日志记录工具模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import datetime
import json


class Logger:
    """统一的日志记录器"""
    
    def __init__(self, name: str = "PFNO", log_file: Optional[str] = None, 
                 console_output: bool = True, log_level: str = "INFO"):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            console_output: 是否输出到控制台
            log_level: 日志级别
        """
        self.name = name
        self.log_file = log_file
        self.console_output = console_output
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误"""
        self.logger.critical(message)
    
    def log_dict(self, data: Dict[str, Any], level: str = "INFO", title: str = ""):
        """记录字典数据"""
        if title:
            self.logger.log(getattr(logging, level.upper()), f"{title}:")
        
        for key, value in data.items():
            self.logger.log(getattr(logging, level.upper()), f"  {key}: {value}")
    
    def log_json(self, data: Dict[str, Any], level: str = "INFO", title: str = ""):
        """以JSON格式记录数据"""
        if title:
            self.logger.log(getattr(logging, level.upper()), title)
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        self.logger.log(getattr(logging, level.upper()), json_str)
    
    def log_separator(self, char: str = "=", length: int = 50):
        """记录分隔符"""
        self.logger.info(char * length)
    
    def log_section(self, title: str, char: str = "=", length: int = 50):
        """记录章节标题"""
        self.log_separator(char, length)
        self.logger.info(f" {title} ")
        self.log_separator(char, length)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO", 
                 console_output: bool = True) -> Logger:
    """
    设置默认日志记录
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        console_output: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pfno_{timestamp}.log"
    
    # 创建并返回日志记录器
    return Logger(
        name="PFNO",
        log_file=str(log_file),
        console_output=console_output,
        log_level=log_level
    )


def log_model_info(logger: Logger, model, model_name: str = "Model"):
    """
    记录模型信息
    
    Args:
        logger: 日志记录器
        model: PyTorch模型
        model_name: 模型名称
    """
    logger.log_section(f"{model_name} 信息")
    
    # 模型基本信息
    logger.info(f"模型类型: {type(model).__name__}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    logger.info(f"不可训练参数数: {total_params - trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 模型特定信息
    if hasattr(model, 'num_nodes'):
        logger.info(f"节点数: {model.num_nodes}")
    if hasattr(model, 'sequence_length'):
        logger.info(f"序列长度: {model.sequence_length}")
    if hasattr(model, 'hidden_channels'):
        logger.info(f"隐藏层维度: {model.hidden_channels}")
    if hasattr(model, 'n_modes'):
        logger.info(f"模态数: {model.n_modes}")
    
    # 设备信息
    device = next(model.parameters()).device
    logger.info(f"设备: {device}")


def log_experiment_info(logger: Logger, config: Dict[str, Any], 
                       experiment_name: str = "PFNO实验"):
    """
    记录实验信息
    
    Args:
        logger: 日志记录器
        config: 实验配置
        experiment_name: 实验名称
    """
    logger.log_section(f"{experiment_name} 配置")
    
    # 记录配置信息
    for section, settings in config.items():
        logger.info(f"{section.upper()}:")
        if isinstance(settings, dict):
            for key, value in settings.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {settings}")


def log_training_progress(logger: Logger, epoch: int, total_epochs: int,
                         train_loss: float, val_loss: Optional[float] = None,
                         metrics: Optional[Dict[str, float]] = None):
    """
    记录训练进度
    
    Args:
        logger: 日志记录器
        epoch: 当前轮次
        total_epochs: 总轮次
        train_loss: 训练损失
        val_loss: 验证损失
        metrics: 其他指标
    """
    progress = f"Epoch [{epoch}/{total_epochs}]"
    loss_info = f"Train Loss: {train_loss:.6f}"
    
    if val_loss is not None:
        loss_info += f", Val Loss: {val_loss:.6f}"
    
    message = f"{progress} - {loss_info}"
    
    if metrics:
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message += f", {metric_str}"
    
    logger.info(message)


def log_evaluation_results(logger: Logger, results: Dict[str, Any], 
                          title: str = "评估结果"):
    """
    记录评估结果
    
    Args:
        logger: 日志记录器
        results: 评估结果
        title: 标题
    """
    logger.log_section(title)
    
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric}: {value:.4f}")
        elif isinstance(value, dict):
            logger.info(f"{metric}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    logger.info(f"  {sub_key}: {sub_value:.4f}")
                else:
                    logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{metric}: {value}")


class ExperimentLogger:
    """实验专用日志记录器"""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        """
        初始化实验日志记录器
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        
        # 创建实验目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        log_file = self.experiment_dir / "experiment.log"
        self.logger = Logger(
            name=experiment_name,
            log_file=str(log_file),
            console_output=True
        )
        
        # 记录开始时间
        self.start_time = datetime.datetime.now()
        self.logger.info(f"实验开始: {self.start_time}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log_experiment_info(self.logger, config, self.experiment_name)
    
    def log_model(self, model, model_name: str = "Model"):
        """记录模型信息"""
        log_model_info(self.logger, model, model_name)
    
    def log_training(self, epoch: int, total_epochs: int, **kwargs):
        """记录训练进度"""
        log_training_progress(self.logger, epoch, total_epochs, **kwargs)
    
    def log_results(self, results: Dict[str, Any], title: str = "结果"):
        """记录结果"""
        # 保存到JSON文件
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 记录到日志
        log_evaluation_results(self.logger, results, title)
    
    def finish(self):
        """结束实验"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(f"实验结束: {end_time}")
        self.logger.info(f"实验耗时: {duration}")
        self.logger.log_separator()
        
        return str(self.experiment_dir)


__all__ = [
    'Logger',
    'setup_logging',
    'log_model_info',
    'log_experiment_info',
    'log_training_progress',
    'log_evaluation_results',
    'ExperimentLogger'
] 