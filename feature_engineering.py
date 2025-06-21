import numpy as np

def create_sliding_windows(data, window_size, predict_steps=1):
    """
    创建滑动窗口数据集，支持多步预测

    参数:
    data: 输入数据 (1D数组)
    window_size: 窗口大小
    predict_steps: 预测步数

    返回:
    X: 输入序列 [n_samples, window_size]
    y: 目标值 [n_samples, predict_steps]
    """
    X, y = [], []
    for i in range(len(data) - window_size - predict_steps + 1):
        X.append(data[i:i + window_size])
        # 预测目标: 未来 predict_steps 个点
        y.append(data[i + window_size:i + window_size + predict_steps])
    return np.array(X), np.array(y)