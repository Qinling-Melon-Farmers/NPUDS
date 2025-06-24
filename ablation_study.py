import pandas as pd

from feature_engineering import create_sliding_windows
from model_building import train_model, build_lstm_model, predict


def window_size_ablation(train_data, test_data, sizes=[8, 16, 32, 64, 128], device='device'):
    results = {}

    # 确保使用OT列的值（NumPy数组）
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data['OT'].values
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data['OT'].values

    for ws in sizes:
        print(f"\n===== 测试窗口大小: {ws} =====")
        X_train, y_train = create_sliding_windows(train_data, ws)
        X_test, y_test = create_sliding_windows(test_data, ws)

        model = build_lstm_model(ws, device=device)
        model = train_model(model, X_train, y_train, device=device)

        # 评估
        y_pred = predict(model, X_test, device)
        mse = ((y_test - y_pred) ** 2).mean()
        results[ws] = mse
        print(f"窗口大小 {ws} 的测试MSE: {mse:.4f}")

    return results