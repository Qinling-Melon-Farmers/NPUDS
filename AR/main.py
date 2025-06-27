import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import os


# 1. 数据加载与预处理
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 不存在")

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    df = df.dropna()

    # 提取目标变量并标准化
    ts = df['OT'].values
    mean_ot, std_ot = ts.mean(), ts.std()
    ts_normalized = (ts - mean_ot) / std_ot

    # 添加标准化后的列用于后续可视化
    df['OT_normalized'] = ts_normalized
    return ts_normalized, mean_ot, std_ot, df


# 2. 构建自回归数据集
def create_ar_dataset(data, window_size):
    X, y = [], []
    # 同时记录每个窗口的最后一个值（用于DA计算）
    last_values = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
        last_values.append(data[i + window_size - 1])  # 窗口的最后一个值

    return np.array(X), np.array(y), np.array(last_values)


# 3. 训练自回归模型
def train_ar_model(train_data, window_size):
    model = AutoReg(train_data, lags=window_size, old_names=False)
    return model.fit()


# 4. 模型预测
def ar_predict(model, history, steps, window_size):
    predictions = []
    current_history = history.copy()

    for _ in range(steps):
        # 使用最近的window_size个点预测下一个点
        yhat = model.predict(current_history[-window_size:], dynamic=False)[0]
        predictions.append(yhat)
        current_history = np.append(current_history, yhat)

    return predictions


# 5. 增强评估指标 (添加DA计算)
def calculate_direction_accuracy(last_values, y_true, y_pred, threshold=0.01):
    """
    计算方向准确率 (Directional Accuracy)

    参数:
        last_values: 每个样本的前一个真实值
        y_true: 当前时间步的真实值
        y_pred: 当前时间步的预测值
        threshold: 变化量的阈值，小于此值的变化被视为中性而不计入统计

    返回:
        da: 方向准确率(百分比)
        valid_samples: 有效样本数量
    """
    # 计算实际变化
    actual_changes = y_true - last_values
    # 计算预测变化
    pred_changes = y_pred - last_values

    # 确定方向是否正确
    directional_correct = np.sign(actual_changes) == np.sign(pred_changes)

    # 过滤掉变化量小于阈值的样本
    significant_change = np.abs(actual_changes) > threshold
    valid_samples = np.sum(significant_change)

    # 计算方向准确率
    if valid_samples > 0:
        da = np.mean(directional_correct[significant_change]) * 100
    else:
        da = 0.0

    return da, valid_samples


# 6. 消融实验：窗口大小影响
def ablation_study_window_sizes(data, window_sizes):
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for ws in window_sizes:
        print(f"\n进行窗口大小 {ws} 的消融实验...")
        X, y, last_values = create_ar_dataset(data, ws)
        mse_scores, rmse_scores, mae_scores, da_scores = [], [], [], []
        valid_da_samples = 0

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            last_values_test = last_values[test_index]

            # 训练模型
            model_fit = train_ar_model(y_train, ws)

            # 预测整个测试集
            y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

            # 计算各种指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # 计算方向准确率
            da, valid_samples = calculate_direction_accuracy(
                last_values_test, y_test, y_pred, threshold=0.01
            )

            valid_da_samples += valid_samples

            # 存储结果
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            da_scores.append(da)

            print(f"  折叠 {fold + 1}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, DA={da:.2f}%")

        results[ws] = {
            'avg_mse': np.mean(mse_scores),
            'avg_rmse': np.mean(rmse_scores),
            'avg_mae': np.mean(mae_scores),
            'avg_da': np.mean(da_scores),
            'mse_scores': mse_scores,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores,
            'da_scores': da_scores,
            'valid_da_samples': valid_da_samples
        }
        print(f"窗口 {ws} 平均: MSE={results[ws]['avg_mse']:.4f}, " +
              f"RMSE={results[ws]['avg_rmse']:.4f}, " +
              f"MAE={results[ws]['avg_mae']:.4f}, " +
              f"DA={results[ws]['avg_da']:.2f}%")

    return results


# 7. 异常值检测
def detect_anomalies(model, data, window_size, threshold=2.5):
    anomalies = []
    reconstruction_errors = []
    train_length = len(model.data.orig_endog)  # 训练数据长度

    for i in range(window_size, len(data)):
        # 准备输入数据
        start_idx = max(0, i - window_size)
        history = data[start_idx:i]

        # 预测当前点
        prediction = model.predict(history, dynamic=False)[0]
        true_value = data[i]

        # 计算重建误差
        error = abs(true_value - prediction)
        reconstruction_errors.append(error)

        # 计算动态阈值
        if len(reconstruction_errors) > 10:
            error_std = np.std(reconstruction_errors[:-1])
            if error > threshold * error_std:
                anomalies.append({
                    'index': i,
                    'true_value': true_value,
                    'prediction': prediction,
                    'error': error,
                    'threshold': threshold * error_std
                })

    return anomalies, reconstruction_errors


# 8. 增强可视化结果
def visualize_results(df, test_size, y_test, y_pred, anomalies, mean_ot, std_ot, results):
    plt.figure(figsize=(16, 18))

    # 原始数据
    plt.subplot(4, 2, 1)
    plt.plot(df.index, df['OT'], label='实际油温', color='blue', alpha=0.7)
    plt.title('变压器油温时间序列', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('油温(°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 预测对比
    plt.subplot(4, 2, 2)
    split_index = int(len(df) * (1 - test_size))
    test_dates = df.index[split_index:]

    # 获取方向准确率
    test_start_idx = split_index - (len(df) - len(y_test))
    last_values = df['OT_normalized'].iloc[test_start_idx - 1:test_start_idx + len(y_test) - 1].values
    da, valid_samples = calculate_direction_accuracy(last_values, y_test, y_pred)

    # 确保长度匹配
    min_len = min(len(y_test), len(y_pred), len(test_dates))
    y_test_actual = y_test[:min_len] * std_ot + mean_ot
    y_pred_actual = y_pred[:min_len] * std_ot + mean_ot

    plt.plot(test_dates[:min_len], y_test_actual, label='实际值', color='blue', alpha=0.7)
    plt.plot(test_dates[:min_len], y_pred_actual, label='预测值', color='red', alpha=0.7)

    # 在图上添加DA指标
    plt.text(0.02, 0.95, f'方向准确率 (DA): {da:.2f}%\n有效样本数: {valid_samples}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title('测试集预测效果', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('油温(°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 异常点标记
    plt.subplot(4, 2, 3)
    plt.plot(df.index, df['OT'], label='实际油温', color='blue', alpha=0.7)

    if anomalies:
        anomaly_indices = [a['index'] for a in anomalies]
        anomaly_dates = df.index[anomaly_indices]
        anomaly_values = df['OT'].iloc[anomaly_indices]
        plt.scatter(anomaly_dates, anomaly_values,
                    color='red', s=50, label='检测异常点', zorder=5)

    plt.title('异常值检测结果', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('油温(°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 消融实验结果可视化 - MSE, RMSE, MAE
    plt.subplot(4, 2, 4)
    window_sizes = list(results.keys())
    mse_scores = [results[ws]['avg_mse'] for ws in window_sizes]
    rmse_scores = [results[ws]['avg_rmse'] for ws in window_sizes]
    mae_scores = [results[ws]['avg_mae'] for ws in window_sizes]

    plt.plot(window_sizes, mse_scores, 'o-', label='MSE', color='blue')
    plt.plot(window_sizes, rmse_scores, 's-', label='RMSE', color='green')
    plt.plot(window_sizes, mae_scores, 'd-', label='MAE', color='purple')
    plt.xlabel('窗口大小', fontsize=12)
    plt.ylabel('评估指标值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('窗口大小消融实验结果 (MSE/RMSE/MAE)', fontsize=14)

    # 消融实验结果可视化 - DA
    plt.subplot(4, 2, 5)
    da_scores = [results[ws]['avg_da'] for ws in window_sizes]
    valid_samples = [results[ws]['valid_da_samples'] for ws in window_sizes]

    plt.plot(window_sizes, da_scores, 'o-', label='DA', color='red')
    plt.xlabel('窗口大小', fontsize=12)
    plt.ylabel('方向准确率(%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加右侧Y轴显示有效样本数
    ax2 = plt.gca().twinx()
    ax2.bar(window_sizes, valid_samples, alpha=0.3, color='orange', width=10)
    ax2.set_ylabel('有效样本总数', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('窗口大小消融实验结果 (方向准确率)', fontsize=14)

    # 预测误差分布
    plt.subplot(4, 2, 6)
    errors = y_test - y_pred
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title("测试集预测误差分布", fontsize=14)
    plt.xlabel("预测误差", fontsize=12)
    plt.ylabel("频数", fontsize=12)

    # 方向准确率可视化示例
    plt.subplot(4, 2, (7, 8))
    last_values_actual = last_values[:min_len] * std_ot + mean_ot
    actual_changes = y_test_actual - last_values_actual[:min_len]
    pred_changes = y_pred_actual - last_values_actual[:min_len]

    # 只取前50个样本
    n_show = min(50, min_len)

    # 计算方向正确性
    correct = np.sign(actual_changes[:n_show]) == np.sign(pred_changes[:n_show])

    plt.figure(figsize=(16, 18))
    plt.subplot(4, 2, (7, 8))
    plt.plot(range(n_show), actual_changes[:n_show], 'o-', label='实际变化')
    plt.plot(range(n_show), pred_changes[:n_show], 'x-', label='预测变化')

    # 标记方向正确的点
    for i in range(n_show):
        if correct[i]:
            plt.plot(i, actual_changes[i], 'go', markersize=8)
        else:
            plt.plot(i, actual_changes[i], 'ro', markersize=8)

    # 添加图例
    plt.scatter([], [], c='green', s=50, label='方向正确')
    plt.scatter([], [], c='red', s=50, label='方向错误')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('方向准确率示意图 (绿色=正确, 红色=错误)', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('变化量(°C)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('ar_model_results_enhanced.png', dpi=300)
    print("结果已保存为 ar_model_results_enhanced.png")
    plt.show()


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("变压器油温预测 - 自回归模型")
    print("=" * 60)

    # 配置参数
    DATA_PATH = './ETTdata/ETTm1.csv'
    WINDOW_SIZE = 96
    TEST_SIZE = 0.2
    PREDICTION_STEPS = 96
    DA_THRESHOLD = 0.01  # 方向准确率计算的阈值

    # 1. 数据加载与预处理
    print("\n步骤1: 数据加载与预处理...")
    try:
        ts_normalized, mean_ot, std_ot, df = load_and_preprocess(DATA_PATH)
        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"油温均值: {mean_ot:.2f}°C, 标准差: {std_ot:.2f}°C")
    except Exception as e:
        print(f"数据处理错误: {e}")
        exit(1)

    # 2. 创建数据集 (添加last_values返回)
    print("\n步骤2: 创建数据集...")
    X, y, last_values = create_ar_dataset(ts_normalized, WINDOW_SIZE)
    print(f"数据集创建完成，窗口大小: {WINDOW_SIZE}, 总样本数: {len(X)}")

    # 3. 划分训练集测试集
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    last_values_test = last_values[split_idx:]
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 4. 训练模型
    print("\n步骤3: 训练自回归模型...")
    model = train_ar_model(y_train, WINDOW_SIZE)
    print(f"模型训练完成，系数数量: {len(model.params)}")

    # 5. 测试集预测评估 - 使用索引预测
    print("\n步骤4: 测试集评估...")
    try:
        # 使用模型内部的索引进行预测
        start_idx = len(y_train)
        end_idx = start_idx + len(y_test) - 1
        y_pred = model.predict(start=start_idx, end=end_idx)
    except Exception as e:
        print(f"索引预测出错: {e}")
        print("改用逐步预测方法...")
        # 逐步预测作为备选方案
        y_pred = []
        for i in range(len(X_test)):
            pred = model.predict(X_test[i].reshape(1, -1))[0]
            y_pred.append(pred)
        y_pred = np.array(y_pred)

    # 计算各种评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # 计算方向准确率
    da, valid_samples = calculate_direction_accuracy(
        last_values_test, y_test, y_pred, DA_THRESHOLD
    )

    print(f"单步预测性能:")
    print(f"  MSE = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  DA = {da:.2f}% (有效样本数: {valid_samples})")

    # 6. 多步预测
    print(f"\n步骤5: 多步预测 ({PREDICTION_STEPS}步)...")
    history = ts_normalized[split_idx:split_idx + WINDOW_SIZE]
    multi_step_pred = ar_predict(model, history, PREDICTION_STEPS, WINDOW_SIZE)

    # 反标准化
    multi_step_pred_actual = np.array(multi_step_pred) * std_ot + mean_ot
    print(f"最后5个预测值:")
    for i, pred in enumerate(multi_step_pred_actual[-5:], 1):
        print(f"  未来第 {i * 15} 分钟: {pred:.2f}°C")

    # 7. 消融实验：窗口大小影响
    print("\n步骤6: 进行消融实验 (窗口大小影响)...")
    window_sizes = [24, 48, 96, 144, 192]
    ablation_results = ablation_study_window_sizes(ts_normalized, window_sizes)

    # 8. 异常值检测
    print("\n步骤7: 异常值检测...")
    anomalies, errors = detect_anomalies(model, ts_normalized, WINDOW_SIZE)
    print(f"检测到 {len(anomalies)} 个异常点")
    if anomalies:
        for i, anom in enumerate(anomalies[:5], 1):
            actual_temp = anom['true_value'] * std_ot + mean_ot
            pred_temp = anom['prediction'] * std_ot + mean_ot
            print(f"  异常点 {i}: 实际值={actual_temp:.2f}°C, 预测值={pred_temp:.2f}°C, 误差={anom['error']:.4f}")

    # 9. 可视化结果
    print("\n步骤8: 可视化结果...")
    visualize_results(df, TEST_SIZE, y_test, y_pred, anomalies, mean_ot, std_ot, ablation_results)

    print("\n程序执行完成!")