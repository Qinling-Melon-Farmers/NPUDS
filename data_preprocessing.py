import pandas as pd

def load_data():
    df = pd.read_csv('./ETTdata/ETTh1.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df[['OT']]  # 仅使用OT列

def handle_missing_values(df):
    return df.interpolate(method='time')

def split_sequential(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]