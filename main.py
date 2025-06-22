# 在文件顶部设置后端
import matplotlib

from feature_engineering import create_sliding_windows, add_engineered_features
from persistence_baseline import PersistenceModel

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
from data_preprocessing import load_data, handle_missing_values, split_sequential, normalize_data
from model_building import (
    build_simplified_predictor,
    train_predict_model,
    predict_next_step,
    predict_long_term, SimplifiedPredictor
)
from config import CONFIG


def visualize_results(X_test, y_test, y_pred_next, long_term_preds, full_recon, partial_recon, window_size):
    """
    更新后的可视化函数 - 支持多通道重构
    """
    plt.figure(figsize=(18, 15))

    # 样本索引
    sample_idx = 0
    predict_steps = len(long_term_preds[0])
    n_features = X_test.shape[2]  # 特征数量

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


    # 3. 预测对比 (前100个样本的下一步预测)
    plt.subplot(3, 2, 6)
    plt.plot(y_test[:100, 0], label='真实值')
    plt.plot(y_pred_next[:100], label='预测值')
    plt.title("前100个样本的下一步预测对比")
    plt.xlabel("样本索引")
    plt.ylabel("油温")
    plt.legend()

    plt.tight_layout()
    plt.savefig('multi_task_results_multi_channel.png')
    print("多通道多任务结果图已保存为 'multi_task_results_multi_channel.png'")


def main(run_ablation=False):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 数据预处理
    df = load_data(CONFIG['data_path'])
    df = handle_missing_values(df)
    train, test = split_sequential(df)

    # 新增：数据标准化
    train_norm, test_norm, train_mean, train_std = normalize_data(train, test)

    # 保存标准化参数，用于后续结果反标准化
    CONFIG['train_mean'] = train_mean
    CONFIG['train_std'] = train_std

    # 2. 特征工程
    window_size = CONFIG['default_window']
    predict_steps = 3

    # 使用修改后的create_sliding_windows，指定目标列为'OT'
    X_train, y_train = create_sliding_windows(train_norm, window_size, predict_steps, target_col='OT')
    X_test, y_test = create_sliding_windows(test_norm, window_size, predict_steps, target_col='OT')

    print(f"输入数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"输出数据形状: y_train={y_train.shape}, y_test={y_test.shape}")  # 现在应该是 (13902, 3)
    # 添加工程特征
    train = add_engineered_features(train)
    test = add_engineered_features(test)

    # 重新标准化
    train_norm, test_norm, train_mean, train_std = normalize_data(train, test)

    # 特征工程
    X_train, y_train = create_sliding_windows(train_norm, window_size, predict_steps, target_col='OT')
    X_test, y_test = create_sliding_windows(test_norm, window_size, predict_steps, target_col='OT')

    print(f"输入数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"输出数据形状: y_train={y_train.shape}, y_test={y_test.shape}")

    # 3. 训练简化模型
    print("\n===== 训练简化预测模型 =====")
    model = SimplifiedPredictor(
        input_size=X_train.shape[2],  # 特征数量
        hidden_size=128,
        output_size=1,
        predict_steps=predict_steps
    ).to(device)

    trained_model = train_predict_model(
        model,
        X_train,
        y_train,
        epochs=100,  # 增加训练轮次
        batch_size=64,
        device=device
    )

    # 4. 执行不同任务
    print("\n===== 执行预测任务 =====")

    # 4.1 普通预测 (X_{n+1})
    y_pred_next = predict_next_step(trained_model, X_test, device=device)

    # 4.2 长时延预测 (X_{n+k})
    long_term_preds = []
    for i in range(len(X_test)):
        initial_seq = X_test[i:i + 1]  # 取一个样本
        preds = predict_long_term(trained_model, initial_seq, steps=predict_steps, device=device)
        long_term_preds.append(preds)

    # 新增：反标准化结果
    def denormalize_ot(values, mean, std):
        """反标准化OT值"""
        return values * std['OT'] + mean['OT']

    # 反标准化预测结果
    y_test_denorm = denormalize_ot(y_test, train_mean, train_std)
    y_pred_next_denorm = denormalize_ot(y_pred_next, train_mean, train_std)
    long_term_preds_denorm = [denormalize_ot(preds, train_mean, train_std) for preds in long_term_preds]

    # 5. 评估 (使用反标准化后的数据)
    print("\n===== 评估结果 (反标准化后) =====")

    # 普通预测评估
    mse_next = np.mean((y_test_denorm[:, 0] - y_pred_next_denorm) ** 2)
    print(f"您的模型单步预测 (X_{{n+1}}) MSE: {mse_next:.4f}")

    # 长时延预测评估
    long_term_preds_array = np.array(long_term_preds_denorm)
    mse_long_term = np.mean((y_test_denorm - long_term_preds_array) ** 2)
    print(f"您的模型长时延预测 (平均MSE): {mse_long_term:.4f}")

    # 添加基线模型比较
    print("\n===== 基线模型比较 =====")

    # 创建基线模型实例 - 指定目标列为'OT'
    baseline_model = PersistenceModel(predict_steps=predict_steps, target_col='OT')

    # 使用训练集确定目标列索引
    baseline_model.fit(train_norm)

    # 基线模型预测
    baseline_next = baseline_model.predict_single_step(X_test)
    baseline_multi = baseline_model.predict_multi_step(X_test)

    # 反标准化基线预测结果
    baseline_next_denorm = denormalize_ot(baseline_next, train_mean, train_std)
    baseline_multi_denorm = denormalize_ot(baseline_multi, train_mean, train_std)

    # 基线模型评估
    baseline_mse_next = np.mean((y_test_denorm[:, 0] - baseline_next_denorm) ** 2)
    baseline_mse_multi = np.mean((y_test_denorm - baseline_multi_denorm) ** 2)

    print(f"基线模型单步预测 MSE: {baseline_mse_next:.4f}")
    print(
        f"您的模型单步预测 MSE: {mse_next:.4f} (改进: {(baseline_mse_next - mse_next) / baseline_mse_next * 100:.2f}%)")

    print(f"基线模型多步预测 MSE: {baseline_mse_multi:.4f}")
    print(
        f"您的模型多步预测 MSE: {mse_long_term:.4f} (改进: {(baseline_mse_multi - mse_long_term) / baseline_mse_multi * 100:.2f}%)")

    # 可视化比较
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm[:100, 0], label='真实值')
    plt.plot(y_pred_next_denorm[:100], label='您的模型')
    plt.plot(baseline_next_denorm[:100], label='基线模型', alpha=0.7)
    plt.title("单步预测对比")
    plt.xlabel("样本索引")
    plt.ylabel("油温")
    plt.legend()
    plt.savefig('model_vs_baseline.png')

    # 6. 可视化结果
    print("\n===== 可视化结果 =====")
    visualize_results(
        X_test,  # 注意：可视化时使用标准化数据
        y_test_denorm,  # 但标签使用反标准化数据
        y_pred_next_denorm,
        long_term_preds_denorm,
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