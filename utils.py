import numpy as np


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """
    计算平均绝对百分比误差(MAPE)

    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    epsilon: 避免除以零的小值

    返回:
    MAPE值 (百分比)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    non_zero_mask = np.abs(y_true) > epsilon
    if np.sum(non_zero_mask) == 0:
        return np.nan

    # 只计算非零值的误差
    y_true_nonzero = y_true[non_zero_mask]
    y_pred_nonzero = y_pred[non_zero_mask]

    # 计算绝对百分比误差
    ape = np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)
    return np.mean(ape) * 100  # 返回百分比