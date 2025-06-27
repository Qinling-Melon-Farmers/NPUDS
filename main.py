# 在文件顶部设置后端
import matplotlib

from ablation_study import window_size_ablation

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体支持
try:
    # 尝试使用系统支持的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果中文字体不可用，改用英文
    print("警告: 中文字体不可用，将使用英文显示")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

import torch
from data_preprocessing import load_data, handle_missing_values, split_sequential
from feature_engineering import create_sliding_windows
from model_building import build_lstm_model, train_model, predict
from config import CONFIG
import numpy as np


def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 数据预处理
    df = load_data(CONFIG['data_path'])
    df = handle_missing_values(df)
    train, test = split_sequential(df)

    # 2. 特征工程
    window_size = CONFIG['default_window']

    # 创建滑动窗口 - 返回形状为 [n_samples, window_size]
    X_train, y_train = create_sliding_windows(train['OT'].values, window_size)
    X_test, y_test = create_sliding_windows(test['OT'].values, window_size)

    # 3. 模型训练
    print("\n===== 训练模型 =====")
    model = build_lstm_model(window_size, input_size=1, device=device)
    model = train_model(model, X_train, y_train, device=device)

    # 4. 预测
    print("\n===== 测试预测 =====")
    y_pred = predict(model, X_test, device)

    # 5. 评估
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"测试集MSE: {mse:.4f}")
    print(f"测试集RMSE: {np.sqrt(mse):.4f}")
    print(f"测试集R^2: {1 - mse / np.var(y_test):.4f}")
    print(f"测试集MAE: {np.mean(np.abs(y_test - y_pred)):.4f}")

    # 6.消融实验
    print("\n===== 消融实验 =====")
    window_sizes = [8, 16, 32, 64, 128]
    results = window_size_ablation(train, test, window_sizes, device)
    print(results)

    # 7. 可视化示例m
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:3500], label='真实值')
    plt.plot(y_pred[:3500], label='预测值')
    plt.legend()
    plt.title('前3500个样本预测对比')
    plt.savefig('prediction_comparison.png')

if __name__ == "__main__":
    main()