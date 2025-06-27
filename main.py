# 在文件顶部设置后端
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from ablation_study import window_size_ablation

# 设置中文字体支持
try:
    # 尝试使用系统支持的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果中文字体不可用，改用英文
    print("警告: 中文字体不可用，将使用英文显示")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

from data_preprocessing import load_data, handle_missing_values, split_sequential
from feature_engineering import create_sliding_windows
from model_building import build_lstm_model, train_model, predict
from config import CONFIG

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)


def calculate_da_and_metrics(last_values, actuals, predictions, scaler, threshold=0.01):
    """
    计算方向准确率(DA)和其他指标（在原始数据尺度上）

    参数:
        last_values: 每个窗口最后一个点的值
        actuals: 真实值（归一化空间）
        predictions: 预测值（归一化空间）
        scaler: 用于反归一化的scaler对象
        threshold: 显著变化的阈值

    返回:
        metrics: 包含所有指标的字典
        actuals_inv: 原始数据尺度的真实值
        predictions_inv: 原始数据尺度的预测值
    """
    # 将所有数组合并以便反归一化
    combined = np.concatenate([
        last_values.reshape(-1, 1),
        actuals.reshape(-1, 1),
        predictions.reshape(-1, 1)
    ])

    combined_inv = scaler.inverse_transform(combined)

    # 分离反归一化后的数据
    n_samples = len(last_values)
    last_values_inv = combined_inv[:n_samples].flatten()
    actuals_inv = combined_inv[n_samples:n_samples * 2].flatten()
    predictions_inv = combined_inv[n_samples * 2:].flatten()

    # 计算指标
    mse = np.mean((actuals_inv - predictions_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals_inv - predictions_inv))

    # 计算实际变化方向
    actual_change = actuals_inv - last_values_inv
    # 计算预测变化方向
    pred_change = predictions_inv - last_values_inv

    # 方向准确计算
    directional_correct = np.sign(actual_change) == np.sign(pred_change)

    # 过滤掉变化量非常小的情况
    significant_change = np.abs(actual_change) > threshold
    valid_samples = np.sum(significant_change)

    # 计算方向准确率
    if valid_samples > 0:
        da = np.mean(directional_correct[significant_change]) * 100
    else:
        da = 0.0

    metrics = {
        'DA': da,
        'Valid_DA_Samples': valid_samples,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

    return metrics, actuals_inv, predictions_inv, last_values_inv


def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 数据预处理
    df = load_data(CONFIG['data_path'])
    df = handle_missing_values(df)

    # 创建归一化器
    scaler = MinMaxScaler()
    # 仅对目标变量归一化
    df['OT'] = scaler.fit_transform(df['OT'].values.reshape(-1, 1))

    # 分割数据
    train, test = split_sequential(df)

    # 2. 特征工程
    window_size = CONFIG['default_window']

    # 创建滑动窗口
    X_train, y_train = create_sliding_windows(train['OT'].values, window_size)
    X_test, y_test = create_sliding_windows(test['OT'].values, window_size)

    # 获取窗口的最后一个值 (用于计算变化方向)
    test_last_values = X_test[:, -1]

    # 3. 模型训练
    print("\n===== 训练模型 =====")
    model = build_lstm_model(window_size, input_size=1, device=device)
    model = train_model(model, X_train, y_train, device=device)

    # 4. 预测
    print("\n===== 测试预测 =====")
    y_pred = predict(model, X_test, device)

    # 5. 在原始数据尺度上评估
    print("\n===== 评估结果 (原始数据尺度) =====")
    metrics, y_test_inv, y_pred_inv, test_last_inv = calculate_da_and_metrics(
        test_last_values, y_test, y_pred, scaler
    )

    print(f"测试集样本数: {len(y_test)}")
    print(f"有效DA样本数: {metrics['Valid_DA_Samples']} (绝对变化量 > 0.01)")
    print(f"方向准确率 (DA): {metrics['DA']:.2f}%")
    print(f"测试集MSE: {metrics['MSE']:.6f}")
    print(f"测试集RMSE: {metrics['RMSE']:.6f}")
    print(f"测试集MAE: {metrics['MAE']:.6f}")

    # 6.消融实验
    print("\n===== 消融实验 =====")
    window_sizes = [8, 16, 32, 64, 128]

    # 包装函数用于消融实验
    def ablation_model_func(X_train, y_train, X_test, device):
        model = build_lstm_model(X_train.shape[1], input_size=1, device=device)
        return train_model(model, X_train, y_train, device=device, verbose=0)

    results = window_size_ablation(train['OT'].values, test['OT'].values, window_sizes,
                                   ablation_model_func, scaler, device)
    print(results)

    # 7. 可视化示例
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[:3500], label='真实值')
    plt.plot(y_pred_inv[:3500], label='预测值')
    plt.legend()
    plt.title('前3500个样本预测对比 (原始数据尺度)')
    plt.savefig('prediction_comparison.png')

    # 添加方向准确率可视化
    plt.figure(figsize=(10, 6))
    plt.plot(test_last_inv[:100], label='最后一个已知值', marker='o')
    plt.plot(y_test_inv[:100], label='真实值', marker='x')
    plt.plot(y_pred_inv[:100], label='预测值', marker='^')

    # 标记方向正确的预测
    changes = y_test_inv[:100] - test_last_inv[:100]
    correct_directions = (np.sign(changes) == np.sign(y_pred_inv[:100] - test_last_inv[:100]))
    correct_indices = np.where(correct_directions)[0]

    for i in correct_indices:
        plt.annotate('✓', (i, max(y_test_inv[i], y_pred_inv[i])),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=12, color='green')

    plt.legend()
    plt.title('方向准确率示意图 (✓ 表示方向正确)')
    plt.savefig('directional_accuracy.png')


if __name__ == "__main__":
    main()