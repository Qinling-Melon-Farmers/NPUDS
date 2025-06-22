import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PersistenceModel:
    """
    持久性模型 (上一步值预测下一步) - 支持多特征输入

    修改说明:
    1. 添加了target_col参数指定目标列
    2. 添加了target_idx参数存储目标列索引
    3. 所有预测方法现在专门针对目标列(OT)进行预测
    """

    def __init__(self, predict_steps=1, target_col='OT'):
        self.predict_steps = predict_steps
        self.target_col = target_col
        self.target_idx = None  # 将在fit方法中确定

    def fit(self, X, y=None):
        """
        确定目标列的索引

        参数:
        X: 输入数据 (DataFrame或特征矩阵)
        """
        if isinstance(X, pd.DataFrame):
            # 从DataFrame确定目标列索引
            self.target_idx = X.columns.get_loc(self.target_col)
        elif X.ndim == 2:
            # 对于二维数组，假设最后一列是目标列
            self.target_idx = -1
        else:
            raise ValueError("无法确定目标列索引，请提供DataFrame或二维数组")
        return self

    def predict_single_step(self, X):
        """
        单步预测: 使用序列的最后一步的目标特征值作为预测值

        参数:
        X: 输入序列 [n_samples, window_size, n_features] 或 [n_samples, window_size]

        返回:
        预测值 [n_samples]
        """
        if X.ndim == 3:
            # 取每个序列最后一个时间步的目标特征值
            return X[:, -1, self.target_idx]
        elif X.ndim == 2:
            # 对于二维输入，直接取最后一个值
            return X[:, -1]
        else:
            raise ValueError(f"不支持的输入维度: {X.ndim}")

    def predict_multi_step(self, X):
        """
        多步预测: 使用序列的最后一步的目标特征值作为所有预测步的值

        参数:
        X: 输入序列 [n_samples, window_size, n_features] 或 [n_samples, window_size]

        返回:
        预测值 [n_samples, predict_steps]
        """
        # 获取单步预测值
        last_vals = self.predict_single_step(X)
        return np.tile(last_vals[:, np.newaxis], (1, self.predict_steps))

    def predict_long_term(self, initial_seq, steps):
        """
        长时延预测: 使用初始序列的最后时间步的目标特征值作为所有预测步的值

        参数:
        initial_seq: 初始序列 [1, window_size, n_features] 或 [1, window_size]
        steps: 预测步数

        返回:
        预测序列 [steps]
        """
        if initial_seq.ndim == 3:
            last_val = initial_seq[0, -1, self.target_idx]
        elif initial_seq.ndim == 2:
            last_val = initial_seq[0, -1]
        else:
            raise ValueError(f"不支持的输入维度: {initial_seq.ndim}")

        return np.full(steps, last_val)


def baseline_experiment(train_data, test_data, window_size, predict_steps=3, target_col='OT'):
    """
    基线模型实验 - 支持多特征输入

    参数:
    train_data: 训练数据 (DataFrame或数组)
    test_data: 测试数据 (DataFrame或数组)
    window_size: 窗口大小
    predict_steps: 预测步数
    target_col: 目标列名 (仅当输入为DataFrame时使用)

    返回:
    包含评估指标和可视化结果的字典
    """

    # 创建滑动窗口 - 支持多特征输入
    def create_windows(data, window_size, predict_steps=1, target_col=None):
        X, y = [], []

        if isinstance(data, pd.DataFrame):
            # DataFrame处理
            target_values = data[target_col].values
            feature_values = data.values

            for i in range(len(data) - window_size - predict_steps + 1):
                X.append(feature_values[i:i + window_size])
                y.append(target_values[i + window_size:i + window_size + predict_steps])
        else:
            # 数组处理 - 假设最后一列是目标列
            for i in range(len(data) - window_size - predict_steps + 1):
                X.append(data[i:i + window_size, :])
                y.append(data[i + window_size:i + window_size + predict_steps, -1])

        return np.array(X), np.array(y)

    # 准备数据
    X_test, y_test = create_windows(test_data, window_size, predict_steps, target_col)

    # 初始化模型
    model = PersistenceModel(predict_steps=predict_steps, target_col=target_col)
    model.fit(train_data)  # 需要确定目标列索引

    # 单步预测
    y_pred_next = model.predict_single_step(X_test)

    # 多步预测
    y_pred_multi = model.predict_multi_step(X_test)

    # 长时延预测示例
    long_term_example = model.predict_long_term(X_test[0:1], steps=predict_steps)

    # 评估
    metrics = {
        'next_step_mse': mean_squared_error(y_test[:, 0], y_pred_next),
        'next_step_mae': mean_absolute_error(y_test[:, 0], y_pred_next),
        'next_step_r2': r2_score(y_test[:, 0], y_pred_next),
        'multi_step_mse': mean_squared_error(y_test, y_pred_multi),
        'multi_step_mae': mean_absolute_error(y_test, y_pred_multi),
        'multi_step_r2': r2_score(y_test, y_pred_multi)
    }

    # 可视化
    plt.figure(figsize=(15, 10))

    # 1. 单步预测对比
    plt.subplot(2, 2, 1)
    plt.plot(y_test[:100, 0], label='真实值')
    plt.plot(y_pred_next[:100], label='预测值')
    plt.title("单步预测对比 (前100个样本)")
    plt.xlabel("样本索引")
    plt.ylabel("值")
    plt.legend()

    # 2. 多步预测示例
    plt.subplot(2, 2, 2)
    sample_idx = 0
    plt.plot(range(1, predict_steps + 1), y_test[sample_idx], 'o-', label='真实值')
    plt.plot(range(1, predict_steps + 1), y_pred_multi[sample_idx], 'o-', label='预测值')
    plt.title(f"多步预测示例 ({predict_steps}步)")
    plt.xlabel("预测步长")
    plt.ylabel("值")
    plt.legend()

    # 3. 长时延预测示例
    plt.subplot(2, 2, 3)
    plt.plot(range(window_size), X_test[sample_idx], 'o-', label='输入序列')
    plt.plot(range(window_size, window_size + predict_steps), y_test[sample_idx], 'o-', label='真实值')
    plt.plot(range(window_size, window_size + predict_steps), long_term_example, 'x--', label='预测值')
    plt.title("长时延预测示例")
    plt.xlabel("时间步")
    plt.ylabel("值")
    plt.legend()

    # 4. 预测误差分布
    plt.subplot(2, 2, 4)
    errors = y_test[:, 0] - y_pred_next
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title("单步预测误差分布")
    plt.xlabel("预测误差")
    plt.ylabel("频数")

    plt.tight_layout()
    plt.savefig('baseline_results.png')

    return {
        'metrics': metrics,
        'y_true': y_test,
        'y_pred_next': y_pred_next,
        'y_pred_multi': y_pred_multi
    }


if __name__ == "__main__":
    # 示例使用 - 假设我们有时序数据
    # 生成示例数据
    np.random.seed(42)
    time = np.arange(0, 1000, 0.1)
    data = np.sin(time) + np.random.normal(0, 0.1, len(time))

    # 分割数据
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # 运行基线实验
    print("===== 运行持久性基线模型 =====")
    results = baseline_experiment(
        train_data,
        test_data,
        window_size=24,
        predict_steps=3
    )

    # 打印评估结果
    metrics = results['metrics']
    print("\n===== 评估结果 =====")
    print(f"单步预测 MSE: {metrics['next_step_mse']:.4f}")
    print(f"单步预测 MAE: {metrics['next_step_mae']:.4f}")
    print(f"单步预测 R²: {metrics['next_step_r2']:.4f}")
    print(f"多步预测 MSE: {metrics['multi_step_mse']:.4f}")
    print(f"多步预测 MAE: {metrics['multi_step_mae']:.4f}")
    print(f"多步预测 R²: {metrics['multi_step_r2']:.4f}")

    print("\n可视化结果已保存为 'baseline_results.png'")