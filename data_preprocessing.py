import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df  # 返回所有特征列


def handle_missing_values(df):
    # 对每列分别进行插值
    for col in df.columns:
        df[col] = df[col].interpolate(method='time')
    return df


def split_sequential(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def normalize_data(train, test):
    """
    标准化数据 (使用训练集的统计量)

    参数:
    train: 训练集DataFrame
    test: 测试集DataFrame

    返回:
    标准化后的训练集和测试集
    """
    train_mean = train.mean()
    train_std = train.std()

    # 避免除零错误
    train_std = train_std.replace(0, 1)

    train_norm = (train - train_mean) / train_std
    test_norm = (test - train_mean) / train_std

    return train_norm, test_norm, train_mean, train_std