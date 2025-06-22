import numpy as np


def create_sliding_windows(data, window_size, predict_steps=1, target_col='OT'):
    """
    创建滑动窗口数据集，支持多特征输入

    参数:
    data: 输入数据 (DataFrame)
    window_size: 窗口大小
    predict_steps: 预测步数
    target_col: 目标列名（默认为'OT'）

    返回:
    X: 输入序列 [n_samples, window_size, n_features]
    y: 目标值 [n_samples, predict_steps]（仅目标列）
    """
    X, y = [], []
    n_features = data.shape[1]

    # 获取目标列的索引
    target_idx = data.columns.get_loc(target_col)

    for i in range(len(data) - window_size - predict_steps + 1):
        # 输入窗口：包含所有特征
        X.append(data.iloc[i:i + window_size].values)

        # 目标：只取OT列的未来predict_steps个值
        y.append(data[target_col].iloc[i + window_size:i + window_size + predict_steps].values)

    return np.array(X), np.array(y)


def add_engineered_features(df):
    """
    添加工程特征增强模型输入
    """
    # 1. 时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month

    # 2. 统计特征
    for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']:
        df[f'{col}_diff1'] = df[col].diff()
        df[f'{col}_rolling_mean_24'] = df[col].rolling(24).mean()
        df[f'{col}_rolling_std_24'] = df[col].rolling(24).std()

    # 3. 交互特征
    df['total_useful_load'] = df['HUFL'] + df['MUFL'] + df['LUFL']
    df['total_useless_load'] = df['HULL'] + df['MULL'] + df['LULL']
    df['load_ratio'] = df['total_useful_load'] / (df['total_useless_load'] + 1e-6)

    # 处理缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df