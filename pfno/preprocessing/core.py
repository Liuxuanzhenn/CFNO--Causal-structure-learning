"""
PFNO数据预处理模块

提供四种不同级别的时间序列数据预处理方法：
1. minimal - 最小预处理：仅处理缺失值和异常值，不进行标准化
2. standard - 标准预处理（推荐）：包含基本清理 + Z-score标准化
3. comprehensive - 全面预处理：包含所有预处理步骤，包括异常值处理和平稳性检查
4. time_series_focused - 时间序列专用：专为时间序列数据设计，包含插值、去趋势等
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy import signal, interpolate, stats
from scipy.signal import butter, filtfilt, detrend as scipy_detrend
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesPreprocessor:
    """时间序列数据预处理器"""
    
    def __init__(self, config):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置字典
        """
        self.config = config
        self.scalers = {}
        self.preprocessing_stats = {}
        
    def preprocess(self, data, method='standard'):
        """
        根据指定方法预处理数据
        
        Args:
            data: pandas.DataFrame, 原始时间序列数据
            method: str, 预处理方法
            
        Returns:
            processed_data: 预处理后的数据
            preprocessing_info: 预处理信息
        """
        if method not in self.config:
            raise ValueError(f"不支持的预处理方法: {method}")
            
        method_config = self.config[method]
        processing_info = {'method': method, 'steps': []}
        
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        print(f"\n🔧 开始 '{method}' 预处理...")
        print(f"原始数据形状: {processed_data.shape}")
        
        # 1. 处理缺失值
        if method_config.get('handle_missing', False):
            processed_data, missing_info = self._handle_missing_values(
                processed_data, method_config
            )
            processing_info['steps'].append(f"处理缺失值: {missing_info}")
            
        # 2. 处理异常值
        if method_config.get('handle_outliers', False):
            processed_data, outlier_info = self._handle_outliers(
                processed_data, method_config
            )
            processing_info['steps'].append(f"处理异常值: {outlier_info}")
            
        # 3. 时间序列专用预处理
        if method == 'time_series_focused':
            processed_data = self._time_series_specific_preprocessing(
                processed_data, method_config
            )
            processing_info['steps'].append("时间序列专用预处理")
            
        # 4. 全面预处理的额外步骤
        if method == 'comprehensive':
            processed_data = self._comprehensive_preprocessing(
                processed_data, method_config
            )
            processing_info['steps'].append("全面预处理步骤")
            
        # 5. 数据标准化
        if method_config.get('normalize', False):
            processed_data, norm_info = self._normalize_data(
                processed_data, method_config
            )
            processing_info['steps'].append(f"数据标准化: {norm_info}")
            
        print(f"预处理后数据形状: {processed_data.shape}")
        print(f"预处理步骤: {len(processing_info['steps'])} 个")
        
        return processed_data, processing_info
    
    def _handle_missing_values(self, data, config):
        """处理缺失值"""
        missing_count = data.isnull().sum().sum()
        if missing_count == 0:
            return data, "无缺失值"
            
        method = config.get('missing_method', 'interpolate')
        
        if method == 'interpolate':
            data = data.interpolate(method='linear', limit_direction='both')
        elif method == 'spline':
            # 时间序列专用的样条插值
            order = config.get('spline_order', 3)
            for col in data.columns:
                if data[col].isnull().any():
                    # 获取非空值的索引和值
                    not_null = data[col].notnull()
                    if not_null.sum() > order:
                        x = data.index[not_null]
                        y = data[col][not_null]
                        # 创建样条插值器
                        f = interpolate.interp1d(x, y, kind='cubic', 
                                               fill_value='extrapolate')
                        # 插值所有缺失值
                        data[col] = data[col].fillna(pd.Series(f(data.index), index=data.index))
        elif method == 'forward_fill':
            data = data.fillna(method='ffill')
        elif method == 'backward_fill':
            data = data.fillna(method='bfill')
        elif method == 'mean':
            data = data.fillna(data.mean())
            
        return data, f"处理了 {missing_count} 个缺失值，方法: {method}"
    
    def _handle_outliers(self, data, config):
        """处理异常值"""
        method = config.get('outlier_method', 'iqr')
        threshold = config.get('outlier_threshold', 3.0)
        outlier_count = 0
        
        for col in data.columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col]))
                outliers = z_scores > threshold
                
            elif method == 'isolation':
                if len(data) > 10:  # 确保有足够的样本
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(data[col].values.reshape(-1, 1)) == -1
                else:
                    outliers = np.zeros(len(data), dtype=bool)
            
            # 记录异常值数量
            outlier_count += outliers.sum()
            
            # 处理异常值（用中位数替换）
            data.loc[outliers, col] = data[col].median()
            
        # 额外的截断处理
        if config.get('clip_outliers', False):
            percentiles = config.get('clip_percentiles', [1, 99])
            for col in data.columns:
                lower_p = np.percentile(data[col], percentiles[0])
                upper_p = np.percentile(data[col], percentiles[1])
                data[col] = data[col].clip(lower=lower_p, upper=upper_p)
            
        return data, f"处理了 {outlier_count} 个异常值，方法: {method}"
    
    def _time_series_specific_preprocessing(self, data, config):
        """时间序列专用预处理"""
        # 去趋势
        if config.get('detrend', False):
            detrend_method = config.get('detrend_method', 'linear')
            for col in data.columns:
                if detrend_method == 'linear':
                    data[col] = scipy_detrend(data[col], type='linear')
                elif detrend_method == 'constant':
                    data[col] = scipy_detrend(data[col], type='constant')
        
        # 应用滤波器
        if config.get('apply_filter', False):
            filter_type = config.get('filter_type', 'butterworth')
            
            if filter_type == 'butterworth':
                order = config.get('filter_order', 4)
                cutoff = config.get('filter_cutoff', 0.1)
                
                # 设计巴特沃思低通滤波器
                b, a = butter(order, cutoff, btype='low')
                
                for col in data.columns:
                    # 应用零相位滤波
                    data[col] = filtfilt(b, a, data[col])
                    
            elif filter_type == 'moving_average':
                window = config.get('smoothing_window', 3)
                data = data.rolling(window=window, center=True).mean()
                data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    
    def _comprehensive_preprocessing(self, data, config):
        """全面预处理的额外步骤"""
        # 检查平稳性
        if config.get('check_stationarity', False):
            try:
                from statsmodels.tsa.stattools import adfuller
                
                for col in data.columns:
                    # ADF检验
                    try:
                        result = adfuller(data[col].dropna())
                        is_stationary = result[1] <= 0.05  # p-value <= 0.05 表示平稳
                        
                        # 如果不平稳且需要处理
                        if not is_stationary and config.get('make_stationary', False):
                            max_diff = config.get('max_diff_order', 2)
                            
                            # 尝试差分
                            diff_data = data[col].copy()
                            for i in range(max_diff):
                                diff_data = diff_data.diff().dropna()
                                if len(diff_data) > 10:
                                    result = adfuller(diff_data)
                                    if result[1] <= 0.05:
                                        break
                            
                            # 如果差分后平稳，使用差分数据（但需要重新构造原长度）
                            if result[1] <= 0.05:
                                # 简单地用差分后的数据填充
                                data[col] = data[col].diff().fillna(0)
                                
                    except Exception as e:
                        print(f"平稳性检验失败 {col}: {e}")
            except ImportError:
                print("警告: statsmodels未安装，跳过平稳性检验")
        
        # 去除趋势
        if config.get('remove_trend', False):
            trend_method = config.get('trend_method', 'linear')
            
            for col in data.columns:
                if trend_method == 'linear':
                    # 线性去趋势
                    x = np.arange(len(data))
                    coeffs = np.polyfit(x, data[col], 1)
                    trend = np.polyval(coeffs, x)
                    data[col] = data[col] - trend
                elif trend_method == 'polynomial':
                    # 多项式去趋势
                    x = np.arange(len(data))
                    coeffs = np.polyfit(x, data[col], 2)
                    trend = np.polyval(coeffs, x)
                    data[col] = data[col] - trend
        
        # 平滑数据
        if config.get('smooth_data', False):
            window = config.get('smoothing_window', 3)
            data = data.rolling(window=window, center=True).mean()
            data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    
    def _normalize_data(self, data, config):
        """数据标准化"""
        method = config.get('normalization_method', 'zscore')
        
        normalized_data = data.copy()
        
        for col in data.columns:
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            # 拟合并转换数据
            col_data = data[col].values.reshape(-1, 1)
            normalized_data[col] = scaler.fit_transform(col_data).flatten()
            
            # 保存标准化器
            self.scalers[col] = scaler
        
        return normalized_data, f"标准化方法: {method}"
    
    def inverse_transform(self, data, columns=None):
        """反标准化数据"""
        if not self.scalers:
            return data
            
        if columns is None:
            columns = data.columns if hasattr(data, 'columns') else range(data.shape[1])
        
        denormalized_data = data.copy()
        
        for i, col in enumerate(columns):
            if col in self.scalers:
                if hasattr(data, 'iloc'):
                    col_data = data.iloc[:, i].values.reshape(-1, 1)
                    denormalized_data.iloc[:, i] = self.scalers[col].inverse_transform(col_data).flatten()
                else:
                    col_data = data[:, i].reshape(-1, 1)
                    denormalized_data[:, i] = self.scalers[col].inverse_transform(col_data).flatten()
        
        return denormalized_data


def preprocess_pfno_data(data, preprocessing_config):
    """
    PFNO数据预处理的便捷函数
    
    Args:
        data: pandas.DataFrame, 原始数据
        preprocessing_config: dict, 预处理配置
        
    Returns:
        processed_data: 预处理后的数据
        preprocessor: 预处理器实例（用于后续反变换）
        info: 预处理信息
    """
    method = preprocessing_config.get('method', 'standard')
    
    # 创建预处理器
    preprocessor = TimeSeriesPreprocessor(preprocessing_config)
    
    # 执行预处理
    processed_data, info = preprocessor.preprocess(data, method)
    
    return processed_data, preprocessor, info


# 预设配置示例
PRESET_PREPROCESSING_CONFIGS = {
    'minimal': {
        'handle_missing': True,
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 3.0,
        'missing_method': 'interpolate',
        'normalize': False
    },
    
    'standard': {
        'handle_missing': True,
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 3.0,
        'missing_method': 'interpolate',
        'normalize': True,
        'normalization_method': 'zscore',
        'clip_outliers': True,
        'clip_percentiles': [1, 99]
    },
    
    'comprehensive': {
        'handle_missing': True,
        'handle_outliers': True,
        'outlier_method': 'isolation',
        'outlier_threshold': 2.5,
        'missing_method': 'interpolate',
        'normalize': True,
        'normalization_method': 'robust',
        'clip_outliers': True,
        'clip_percentiles': [0.5, 99.5],
        'check_stationarity': True,
        'make_stationary': True,
        'max_diff_order': 2,
        'remove_trend': True,
        'trend_method': 'linear',
        'smooth_data': True,
        'smoothing_window': 3
    },
    
    'time_series_focused': {
        'handle_missing': True,
        'missing_method': 'spline',
        'spline_order': 3,
        'handle_outliers': True,
        'outlier_method': 'zscore',
        'outlier_threshold': 2.5,
        'normalize': True,
        'normalization_method': 'zscore',
        'detrend': True,
        'detrend_method': 'linear',
        'seasonal_decompose': False,
        'seasonal_period': None,
        'apply_filter': True,
        'filter_type': 'butterworth',
        'filter_order': 4,
        'filter_cutoff': 0.1,
        'interpolate_method': 'cubic',
        'handle_edge_effects': True
    }
} 