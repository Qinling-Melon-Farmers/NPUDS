# 在文件顶部设置后端
import matplotlib

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
import numpy as np
from data_preprocessing import load_data, handle_missing_values, split_sequential
from model_building import (
    build_multi_task_model,
    train_multi_task_model,
    predict_next_step,
    predict_long_term,
    reconstruct_sequence
)
from config import CONFIG


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


def visualize_results(X_test, y_test, y_pred_next, long_term_preds, full_recon, partial_recon, window_size):
    """
    更新后的可视化函数
    """
    plt.figure(figsize=(18, 15))

    # 样本索引
    sample_idx = 0
    predict_steps = len(long_term_preds[0])

    # 1. 普通预测 (X_{n+1})
    plt.subplot(3, 2, 1)
    plt.plot(y_test[sample_idx], 'o-', label='真实值')
    plt.plot([0], [y_pred_next[sample_idx]], 'ro', label='预测值')
    plt.title(f"普通预测 (X_{{n+1}})")
    plt.xlabel("预测步长")
    plt.ylabel("油温")
    plt.legend()

    # 2. 长时延预测 (X_{n+k})
    plt.subplot(3, 2, 2)
    plt.plot(range(1, predict_steps + 1), y_test[sample_idx], 'o-', label='真实值')
    plt.plot(range(1, predict_steps + 1), long_term_preds[sample_idx], 'o-', label='预测值')
    plt.title(f"长时延预测 ({predict_steps}步)")
    plt.xlabel("预测步长")
    plt.ylabel("油温")
    plt.legend()

    # 3. 重构整个窗口 (X_n)
    plt.subplot(3, 2, 3)
    plt.plot(X_test[sample_idx], label='原始序列')
    plt.plot(full_recon[sample_idx], label='重构序列')
    plt.title("整个窗口重构")
    plt.xlabel("时间步")
    plt.ylabel("油温")
    plt.legend()

    # 4. 重构部分序列 (X_{n/2} 到 X_n)
    plt.subplot(3, 2, 4)
    half_len = window_size // 2
    plt.plot(range(window_size), X_test[sample_idx], label='原始序列')
    plt.plot(range(half_len, window_size), partial_recon[sample_idx], label='重构部分')
    plt.title("部分序列重构")
    plt.xlabel("时间步")
    plt.ylabel("油温")
    plt.legend()

    # 5. 预测对比 (前100个样本的下一步预测)
    plt.subplot(3, 1, 3)
    plt.plot(y_test[:100, 0], label='真实值')
    plt.plot(y_pred_next[:100], label='预测值')
    plt.title("前100个样本的下一步预测对比")
    plt.xlabel("样本索引")
    plt.ylabel("油温")
    plt.legend()

    plt.tight_layout()
    plt.savefig('multi_task_results.png')
    print("多任务结果图已保存为 'multi_task_results.png'")


def main(run_ablation=False):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 数据预处理
    df = load_data(CONFIG['data_path'])
    df = handle_missing_values(df)
    train, test = split_sequential(df)

    # 2. 特征工程
    window_size = CONFIG['default_window']
    predict_steps = 3  # 预测未来3步

    # 创建滑动窗口 - 返回形状为 [n_samples, window_size] 和 [n_samples, predict_steps]
    X_train, y_train = create_sliding_windows(train['OT'].values, window_size, predict_steps)
    X_test, y_test = create_sliding_windows(test['OT'].values, window_size, predict_steps)

    # 3. 模型训练
    print("\n===== 训练多任务模型 =====")
    model = build_multi_task_model(
        window_size=window_size,
        input_size=1,
        predict_steps=predict_steps,
        reconstruction_mode='full',
        device=device
    )

    trained_model = train_multi_task_model(
        model,
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        device=device
    )

    # 4. 执行不同任务
    print("\n===== 执行预测和重构任务 =====")

    # 4.1 普通预测 (X_{n+1})
    y_pred_next = predict_next_step(trained_model, X_test, device=device)

    # 4.2 长时延预测 (X_{n+k})
    long_term_preds = []
    for i in range(len(X_test)):
        initial_seq = X_test[i:i + 1]  # 取一个样本
        preds = predict_long_term(trained_model, initial_seq, steps=predict_steps, device=device)
        long_term_preds.append(preds)

    # 4.3 重构整个窗口 (X_n)
    full_recon = reconstruct_sequence(trained_model, X_test, mode='full', device=device)

    # 4.4 重构部分序列 (X_{n/2} 到 X_n)
    partial_recon = reconstruct_sequence(trained_model, X_test, mode='partial', device=device)

    # 5. 评估
    print("\n===== 评估结果 =====")

    # 普通预测评估 - 只比较下一步预测
    mse_next = np.mean((y_test[:, 0] - y_pred_next) ** 2)
    print(f"普通预测 (X_{{n+1}}) MSE: {mse_next:.4f}")  # 修复这里：使用双大括号转义

    # 长时延预测评估 - 比较所有预测步
    long_term_preds_array = np.array(long_term_preds)  # 转换为数组 [n_samples, predict_steps]
    mse_long_term = np.mean((y_test - long_term_preds_array) ** 2)
    print(f"长时延预测 (平均MSE): {mse_long_term:.4f}")

    # 分步评估长时延预测
    for step in range(predict_steps):
        step_mse = np.mean((y_test[:, step] - long_term_preds_array[:, step]) ** 2)
        print(f"  步长 {step + 1} (X_{{n+{step + 1}}}) MSE: {step_mse:.4f}")  # 修复这里

    # 重构评估
    full_recon_error = np.mean((X_test - full_recon) ** 2)
    print(f"完整窗口重构 MSE: {full_recon_error:.4f}")

    # 部分重构评估 (只评估后半部分)
    half_len = window_size // 2
    partial_recon_error = np.mean((X_test[:, half_len:] - partial_recon) ** 2)
    print(f"部分序列重构 MSE: {partial_recon_error:.4f}")
    # 添加基线模型比较
    print("\n===== 基线模型比较 =====")
    from persistence_baseline import PersistenceModel

    # 初始化基线模型
    baseline_model = PersistenceModel(predict_steps=predict_steps)

    # 基线模型预测
    baseline_next = baseline_model.predict_single_step(X_test)
    baseline_multi = baseline_model.predict_multi_step(X_test)

    # 基线模型评估
    baseline_mse_next = np.mean((y_test[:, 0] - baseline_next) ** 2)
    baseline_mse_multi = np.mean((y_test - baseline_multi) ** 2)

    print(f"基线模型单步预测 MSE: {baseline_mse_next:.4f}")
    print(
        f"您的模型单步预测 MSE: {mse_next:.4f} (改进: {(baseline_mse_next - mse_next) / baseline_mse_next * 100:.2f}%)")

    print(f"基线模型多步预测 MSE: {baseline_mse_multi:.4f}")
    print(
        f"您的模型多步预测 MSE: {mse_long_term:.4f} (改进: {(baseline_mse_multi - mse_long_term) / baseline_mse_multi * 100:.2f}%)")

    # 可视化比较
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100, 0], label='真实值')
    plt.plot(y_pred_next[:100], label='您的模型')
    plt.plot(baseline_next[:100], label='基线模型', alpha=0.7)
    plt.title("单步预测对比")
    plt.xlabel("样本索引")
    plt.ylabel("油温")
    plt.legend()
    plt.savefig('model_vs_baseline.png')

    # 6. 可视化结果
    print("\n===== 可视化结果 =====")
    visualize_results(
        X_test,
        y_test,
        y_pred_next,
        long_term_preds,
        full_recon,
        partial_recon,
        window_size
    )

    # 7. 运行消融实验
    if run_ablation:
        print("\n===== 开始窗口大小消融实验 =====")
        from ablation_study import window_size_ablation, visualize_ablation_results

        # 运行消融实验
        ablation_results = window_size_ablation(
            train['OT'],
            test['OT'],
            sizes=[8, 16, 32, 64, 128],  # 测试的窗口大小
            predict_steps=predict_steps,
            device=device
        )

        # 可视化消融实验结果
        visualize_ablation_results(ablation_results)

        # 找到最佳窗口大小
        best_idx = ablation_results['overall_score'].idxmax()
        best_ws = ablation_results.loc[best_idx, 'window_size']
        print(f"\n最佳窗口大小: {best_ws} (综合评分: {ablation_results.loc[best_idx, 'overall_score']:.4f})")

        # 保存结果
        ablation_results.to_csv('window_size_ablation_results.csv', index=False)
        print("消融实验结果已保存为 'window_size_ablation_results.csv'")

        # 更新配置中的默认窗口大小
        CONFIG['default_window'] = best_ws
        print(f"已更新默认窗口大小为: {best_ws}")


if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='多任务时间序列模型')
    parser.add_argument('--ablation', action='store_true',
                        help='运行窗口大小消融实验')

    args = parser.parse_args()

    # 运行主函数，根据参数决定是否运行消融实验
    main(run_ablation=args.ablation)