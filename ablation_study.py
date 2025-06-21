import pandas as pd
import numpy as np
import torch
from model_building import (
    build_multi_task_model,
    train_multi_task_model,
    predict_next_step,
    predict_long_term,
    reconstruct_sequence
)


def window_size_ablation(train_data, test_data, sizes=[8, 16, 32, 64, 128],
                         predict_steps=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    多任务模型的窗口大小消融实验

    参数:
    train_data: 训练数据 (DataFrame 或 Series)
    test_data: 测试数据 (DataFrame 或 Series)
    sizes: 要测试的窗口大小列表
    predict_steps: 预测步数 (默认为3)
    device: 计算设备

    返回:
    包含所有评估指标的结果字典
    """
    # 确保使用OT列的值（NumPy数组）
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data['OT'].values
    elif isinstance(train_data, pd.Series):
        train_data = train_data.values

    if isinstance(test_data, pd.DataFrame):
        test_data = test_data['OT'].values
    elif isinstance(test_data, pd.Series):
        test_data = test_data.values

    # 初始化结果存储
    results = {
        'window_size': [],
        'next_step_mse': [],
        'long_term_mse': [],
        'step1_mse': [],
        'step2_mse': [],
        'step3_mse': [],
        'full_recon_mse': [],
        'partial_recon_mse': []
    }

    for ws in sizes:
        print(f"\n===== 测试窗口大小: {ws} =====")

        # 创建滑动窗口数据集（支持多步预测）
        def create_windows(data, window_size):
            X, y = [], []
            for i in range(len(data) - window_size - predict_steps + 1):
                X.append(data[i:i + window_size])
                y.append(data[i + window_size:i + window_size + predict_steps])
            return np.array(X), np.array(y)

        X_train, y_train = create_windows(train_data, ws)
        X_test, y_test = create_windows(test_data, ws)

        # 构建并训练多任务模型
        model = build_multi_task_model(
            window_size=ws,
            input_size=1,
            predict_steps=predict_steps,
            reconstruction_mode='full',
            device=device
        )

        model = train_multi_task_model(
            model,
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            device=device
        )

        # 评估所有任务
        ## 1. 普通预测 (X_{n+1})
        y_pred_next = predict_next_step(model, X_test, device=device)
        next_step_mse = np.mean((y_test[:, 0] - y_pred_next) ** 2)

        ## 2. 长时延预测 (X_{n+k})
        long_term_preds = []
        for i in range(len(X_test)):
            preds = predict_long_term(model, X_test[i:i + 1], steps=predict_steps, device=device)
            long_term_preds.append(preds)

        long_term_preds = np.array(long_term_preds)
        long_term_mse = np.mean((y_test - long_term_preds) ** 2)

        ## 分步评估
        step1_mse = np.mean((y_test[:, 0] - long_term_preds[:, 0]) ** 2)
        step2_mse = np.mean((y_test[:, 1] - long_term_preds[:, 1]) ** 2)
        step3_mse = np.mean((y_test[:, 2] - long_term_preds[:, 2]) ** 2)

        ## 3. 重构任务
        full_recon = reconstruct_sequence(model, X_test, mode='full', device=device)
        full_recon_mse = np.mean((X_test - full_recon) ** 2)

        partial_recon = reconstruct_sequence(model, X_test, mode='partial', device=device)
        half_len = ws // 2
        partial_recon_mse = np.mean((X_test[:, half_len:] - partial_recon) ** 2)

        # 记录结果
        results['window_size'].append(ws)
        results['next_step_mse'].append(next_step_mse)
        results['long_term_mse'].append(long_term_mse)
        results['step1_mse'].append(step1_mse)
        results['step2_mse'].append(step2_mse)
        results['step3_mse'].append(step3_mse)
        results['full_recon_mse'].append(full_recon_mse)
        results['partial_recon_mse'].append(partial_recon_mse)

        # 打印当前结果
        print(f"普通预测 (X_{{n+1}}) MSE: {next_step_mse:.4f}")
        print(f"长时延预测 (平均) MSE: {long_term_mse:.4f}")
        print(f"  步长1 (X_{{n+1}}) MSE: {step1_mse:.4f}")
        print(f"  步长2 (X_{{n+2}}) MSE: {step2_mse:.4f}")
        print(f"  步长3 (X_{{n+3}}) MSE: {step3_mse:.4f}")
        print(f"完整重构 MSE: {full_recon_mse:.4f}")
        print(f"部分重构 MSE: {partial_recon_mse:.4f}")

    # 将结果转换为DataFrame便于分析
    results_df = pd.DataFrame(results)

    # 添加综合评分（权重可根据需要调整）
    results_df['overall_score'] = (
            0.4 * (1 - results_df['next_step_mse'] / results_df['next_step_mse'].max()) +
            0.3 * (1 - results_df['long_term_mse'] / results_df['long_term_mse'].max()) +
            0.2 * (1 - results_df['full_recon_mse'] / results_df['full_recon_mse'].max()) +
            0.1 * (1 - results_df['partial_recon_mse'] / results_df['partial_recon_mse'].max())
    )

    return results_df


def visualize_ablation_results(results_df):
    """
    可视化消融实验结果

    参数:
    results_df: 包含结果的DataFrame
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    # 1. 预测任务性能
    plt.subplot(2, 2, 1)
    plt.plot(results_df['window_size'], results_df['next_step_mse'], 'o-', label='普通预测(X_{n+1})')
    plt.plot(results_df['window_size'], results_df['step1_mse'], 's--', label='长时延-步长1')
    plt.plot(results_df['window_size'], results_df['step2_mse'], 'd--', label='长时延-步长2')
    plt.plot(results_df['window_size'], results_df['step3_mse'], '^--', label='长时延-步长3')
    plt.xlabel('窗口大小')
    plt.ylabel('MSE')
    plt.title('预测任务性能 vs 窗口大小')
    plt.legend()
    plt.grid(True)

    # 2. 重构任务性能
    plt.subplot(2, 2, 2)
    plt.plot(results_df['window_size'], results_df['full_recon_mse'], 'o-', label='完整重构')
    plt.plot(results_df['window_size'], results_df['partial_recon_mse'], 's-', label='部分重构')
    plt.xlabel('窗口大小')
    plt.ylabel('MSE')
    plt.title('重构任务性能 vs 窗口大小')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 对数尺度更好显示重构性能

    # 3. 综合评分
    plt.subplot(2, 2, 3)
    plt.plot(results_df['window_size'], results_df['overall_score'], 'o-r')
    plt.xlabel('窗口大小')
    plt.ylabel('综合评分')
    plt.title('综合性能评分 vs 窗口大小')
    plt.grid(True)

    # 4. 误差比例分析
    plt.subplot(2, 2, 4)
    # 计算长时延预测相对于单步预测的误差增长
    error_growth = results_df['step3_mse'] / results_df['next_step_mse']
    plt.plot(results_df['window_size'], error_growth, 's-g')
    plt.xlabel('窗口大小')
    plt.ylabel('步长3误差/步长1误差')
    plt.title('长时延误差增长 vs 窗口大小')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('window_size_ablation.png')
    print("消融实验结果图已保存为 'window_size_ablation.png'")

    return results_df