import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import os
import warnings
from scipy.signal import correlate

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')


# ======================
# 数据处理模块 (支持原始序列和差分序列)
# ======================
class AutoRegressiveDataset(Dataset):
    def __init__(self, ot_series, window_size=24, pred_step=1, diff_mode=False):
        """
        自回归数据集
        :param ot_series: 油温时间序列 (1D数组)
        :param window_size: 历史窗口大小
        :param pred_step: 预测步长
        :param diff_mode: 是否使用差分模式
        """
        self.ot_series = ot_series
        self.window_size = window_size
        self.pred_step = pred_step
        self.diff_mode = diff_mode

        if diff_mode:
            # 差分模式: 使用一阶差分序列
            self.diff_series = np.diff(ot_series, n=1)
            self.base_values = ot_series[:-1]  # 用于还原的基础值
        else:
            self.diff_series = None
            self.base_values = None

    def __len__(self):
        if self.diff_mode:
            return len(self.diff_series) - self.window_size - self.pred_step + 1
        else:
            return len(self.ot_series) - self.window_size - self.pred_step + 1

    def __getitem__(self, idx):
        if self.diff_mode:
            # 差分模式
            # 输入: 历史窗口内的差分值
            x = self.diff_series[idx:idx + self.window_size]

            # 输出: 未来pred_step步的差分值
            y = self.diff_series[idx + self.window_size:idx + self.window_size + self.pred_step]

            # 基础值: 用于还原预测值
            base = self.base_values[idx + self.window_size]

            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                    torch.tensor(base, dtype=torch.float32))
        else:
            # 原始序列模式
            # 输入: 历史窗口内的油温值
            x = self.ot_series[idx:idx + self.window_size]

            # 输出: 未来pred_step步的油温值
            y = self.ot_series[idx + self.window_size:idx + self.window_size + self.pred_step]

            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32))


def load_and_preprocess_data(file_path):
    """
    加载并预处理数据，仅提取油温(OT)序列
    """
    # 加载数据
    df = pd.read_csv(file_path)

    # 数据清洗
    df.fillna(method='ffill', inplace=True)  # 前向填充缺失值
    df.drop_duplicates(inplace=True)

    # 仅提取油温序列
    ot_series = df['OT'].values.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler()
    scaled_ot = scaler.fit_transform(ot_series).flatten()

    return scaled_ot, scaler


def directional_accuracy(y_true, y_pred):
    """计算方向准确率"""
    # 确保数组长度一致
    assert len(y_true) == len(y_pred), "实际值和预测值长度必须相同"

    # 计算实际值的方向变化: 从第二个时间点开始
    true_dir = np.sign(y_true[1:] - y_true[:-1])

    # 计算预测值的方向变化: 从第二个时间点开始
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])

    return np.mean(true_dir == pred_dir)


def calculate_adjusted_da(actuals, predictions, phase_shift):
    """
    计算相位校正后的方向准确率(DA)
    :param actuals: 实际值数组
    :param predictions: 预测值数组
    :param phase_shift: 相位偏移值
    :return: 校正后的DA
    """
    # 应用相位校正
    adjusted_predictions = np.roll(predictions, -phase_shift)

    # 移除无效区域
    if phase_shift > 0:
        # 正偏移：预测超前，开头部分无效
        valid_actuals = actuals[phase_shift:]
        valid_adjusted = adjusted_predictions[phase_shift:]
    elif phase_shift < 0:
        # 负偏移：预测滞后，结尾部分无效
        valid_actuals = actuals[:phase_shift]
        valid_adjusted = adjusted_predictions[:phase_shift]
    else:
        # 无偏移
        valid_actuals = actuals
        valid_adjusted = adjusted_predictions

    # 计算校正后的DA
    return directional_accuracy(valid_actuals, valid_adjusted)


def calculate_adjusted_mape(actuals, predictions, phase_shift):
    """
    计算相位校正后的平均绝对百分比误差(MAPE)
    :param actuals: 实际值数组
    :param predictions: 预测值数组
    :param phase_shift: 相位偏移值
    :return: 校正后的MAPE
    """
    # 应用相位校正
    adjusted_predictions = np.roll(predictions, -phase_shift)

    # 移除无效区域
    if phase_shift > 0:
        # 正偏移：预测超前，开头部分无效
        valid_actuals = actuals[phase_shift:]
        valid_adjusted = adjusted_predictions[phase_shift:]
    elif phase_shift < 0:
        # 负偏移：预测滞后，结尾部分无效
        valid_actuals = actuals[:phase_shift]
        valid_adjusted = adjusted_predictions[:phase_shift]
    else:
        # 无偏移
        valid_actuals = actuals
        valid_adjusted = adjusted_predictions

    # 计算校正后的MAPE
    return np.mean(np.abs((valid_actuals - valid_adjusted) / valid_actuals)) * 100


# ======================
# LSTM模型 (自回归)
# ======================
class AutoRegressiveLSTM(nn.Module):
    """
    自回归LSTM模型，仅使用OT历史数据预测未来OT值
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_size=64, num_layers=2, dropout=0.1):
        super(AutoRegressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM处理
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出用于预测
        out = self.fc(out[:, -1, :])
        return out


# ======================
# 训练与评估函数
# ======================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, diff_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"{'训练轮数':^7} | {'训练损失':^12} | {'验证损失':^10} | {'学习率':^8}")
    print("-" * 45)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if diff_mode:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            # 增加维度: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            inputs = inputs.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if diff_mode:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')

        # 打印训练进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {current_lr:^8.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('MSE 损失')
    plt.title('LSTM训练与验证损失曲线')
    plt.legend()
    plt.savefig('lstm_loss_curve.png')
    plt.close()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    return model


def evaluate_model(model, test_loader, scaler, diff_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_actuals = []
    all_predictions = []

    if diff_mode:
        # 差分模式需要额外存储基础值
        all_base = []

    with torch.no_grad():
        for batch in test_loader:
            if diff_mode:
                inputs, targets, base = batch
                inputs, targets, base = inputs.to(device), targets.to(device), base.to(device)
                all_base.append(base.cpu().numpy())
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

            # 增加一个特征维度: (batch_size, window_size) -> (batch_size, window_size, 1)
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)  # 输出: (batch_size, pred_step)

            # 保存目标值和预测值
            all_actuals.append(targets.cpu().numpy())
            all_predictions.append(outputs.squeeze(-1).cpu().numpy())  # 如果pred_step=1，则squeeze后是1D

    # 合并所有批次
    all_actuals = np.concatenate(all_actuals, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    if diff_mode:
        all_base = np.concatenate(all_base, axis=0)
        # 还原预测值: T(t) = T(t-1) + ΔT_pred
        predictions_recon = all_base + all_predictions
        # 还原真实值: T(t) = T(t-1) + ΔT_true
        actuals_recon = all_base + all_actuals

        # 反归一化
        actuals = scaler.inverse_transform(actuals_recon.reshape(-1, 1)).flatten()
        predictions = scaler.inverse_transform(predictions_recon.reshape(-1, 1)).flatten()
    else:
        # 反归一化
        actuals = scaler.inverse_transform(all_actuals.reshape(-1, 1)).flatten()
        predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()

    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    # 计算MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # 计算时间序列R²（改进的持久性模型）
    def calculate_persistence_prediction(actuals):
        persistence_pred = np.zeros_like(actuals)
        persistence_pred[0] = actuals[0]
        persistence_pred[1:] = actuals[:-1]
        return persistence_pred

    persistence_pred = calculate_persistence_prediction(actuals)
    persistence_mse = mean_squared_error(actuals, persistence_pred)
    ts_r2 = 1 - (mse / persistence_mse) if persistence_mse > 0 else 0.0

    # 计算方向准确率
    da = directional_accuracy(actuals, predictions)

    # 计算相位偏移
    def calculate_phase_shift(actuals, predictions):
        # 使用零延迟互相关
        cross_corr = correlate(actuals - np.mean(actuals),
                               predictions - np.mean(predictions),
                               mode='same')

        # 找到最大相关性的位置
        center = len(actuals) // 2
        lag = np.argmax(cross_corr) - center

        # 调整方向: 正滞后表示预测落后于实际值
        return -lag, cross_corr

    phase_shift, cross_corr = calculate_phase_shift(actuals, predictions)
    # 计算校正后的DA
    adjusted_da = calculate_adjusted_da(actuals, predictions, phase_shift)
    # 计算校正后的MAPE
    adjusted_mape = calculate_adjusted_mape(actuals, predictions, phase_shift)

    # 打印评估指标
    print("\n" + "=" * 60)
    print(f"{'LSTM模型综合评估指标':^60}")
    print("=" * 60)
    print(f"均方误差 (MSE):       {mse:.4f}")
    print(f"均方根误差 (RMSE):    {rmse:.4f}")
    print(f"平均绝对误差 (MAE):    {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
    print(f"校正后MAPE (Adj-MAPE): {adjusted_mape:.4f}%")
    print(f"决定系数 (R²):       {r2:.4f}")
    print(f"时间序列R² (TS R²):   {ts_r2:.4f}")
    print(f"方向准确率 (DA):           {da:.4f}")
    print(f"校正后方向准确率 (Adj-DA):  {adjusted_da:.4f}")
    print(f"相位偏移 (步数):           {phase_shift}")

    # 绘制预测对比图（前200个点）
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:200], 'b-', label='实际油温')
    plt.plot(predictions[:200], 'r--', label='预测油温')
    plt.fill_between(range(200),
                     actuals[:200] - rmse,
                     actuals[:200] + rmse,
                     color='gray', alpha=0.3, label='±RMSE区间')
    plt.xlabel('时间点')
    plt.ylabel('油温值(℃)')
    plt.title('LSTM实际油温与预测油温对比')
    plt.legend()
    plt.savefig('lstm_prediction_comparison.png')
    plt.close()

    # 绘制残差图
    residuals = actuals - predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', label='零误差线')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('LSTM预测残差分析')
    plt.legend()
    plt.savefig('lstm_residual_plot.png')
    plt.close()

    # 新增指标对比图
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Adj-MAPE', 'R²', 'TS R²', 'DA', 'Adj-DA']
    values = [mse, rmse, mae, mape, adjusted_mape, r2, ts_r2, da, adjusted_da]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    plt.ylabel('指标值')
    plt.title('LSTM模型评估指标对比')
    plt.xticks(rotation=45)

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}' if height < 1 else f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('lstm_metrics_comparison.png')
    plt.close()

    return rmse


# ======================
# 消融实验 (包含差分处理)
# ======================
def run_ablation_study(ot_series, scaler):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 50)
    print(f"{'运行LSTM消融实验':^50}")
    print("=" * 50)

    # 实验1: 差分处理 vs 原始序列
    modes = [('原始序列', False), ('差分序列', True)]
    print("\n>> 实验1: 差分处理效果分析")

    for mode_name, diff_mode in modes:
        print(f"\n运行模式: {mode_name}")

        # 创建数据集
        dataset = AutoRegressiveDataset(ot_series, window_size=64, diff_mode=diff_mode)

        # 时间序列划分 (70%-15%-15%)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        # 创建数据加载器
        if diff_mode:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)
            test_loader = DataLoader(test_dataset, batch_size=64)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)
            test_loader = DataLoader(test_dataset, batch_size=64)

        # 初始化模型
        model = AutoRegressiveLSTM(
            input_dim=1,
            output_dim=1,
            hidden_size=64,
            num_layers=2
        ).to(device)

        # 训练模型
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(15):  # 适度训练轮数
            for batch in train_loader:
                if diff_mode:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # 评估模型
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if diff_mode:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'mode_{mode_name}'] = test_loss
        print(f"处理模式: {mode_name:10} | 测试损失: {test_loss:.6f}")

    # 实验2: 不同窗口大小 (在差分模式下)
    window_sizes = [8, 16, 32, 64, 128]
    print("\n>> 实验2: 窗口大小影响分析 (差分模式)")
    for ws in window_sizes:
        dataset = AutoRegressiveDataset(ot_series, window_size=ws, diff_mode=True)

        # 时间序列划分
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # 初始化模型
        model = AutoRegressiveLSTM(
            input_dim=1,
            output_dim=1,
            hidden_size=64,
            num_layers=2
        ).to(device)

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # 评估模型
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'window_{ws}'] = test_loss
        print(f"窗口大小: {ws:3d} | 测试损失: {test_loss:.6f}")

    # 实验3: 不同输出步长 (在差分模式下)
    pred_steps = [1, 3, 6, 12]  # 预测1步(15min), 3步(45min), 6步(1.5h), 12步(3h)
    print("\n>> 实验3: 预测步长影响分析 (差分模式)")
    for ps in pred_steps:
        dataset = AutoRegressiveDataset(ot_series, window_size=64, pred_step=ps, diff_mode=True)

        # 时间序列划分
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # 初始化模型 (输出维度改为预测步长)
        model = AutoRegressiveLSTM(
            input_dim=1,
            output_dim=ps,
            hidden_size=64,
            num_layers=2
        ).to(device)

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # 评估模型
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'pred_step_{ps}'] = test_loss
        print(f"预测步长: {ps:2d} | 测试损失: {test_loss:.6f}")

    # 保存结果
    print("\n>> LSTM消融实验结果汇总")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k:15}: {v:.6f}")

    # 可视化结果
    plt.figure(figsize=(15, 5))

    # 处理模式结果
    mode_results = {k: v for k, v in results.items() if 'mode' in k}
    plt.subplot(1, 3, 1)
    plt.bar([k.split('_')[1] for k in mode_results.keys()], list(mode_results.values()))
    plt.xlabel('处理模式')
    plt.ylabel('测试损失')
    plt.title('差分处理效果分析')

    # 窗口大小结果
    window_results = {k: v for k, v in results.items() if 'window' in k}
    plt.subplot(1, 3, 2)
    plt.plot([int(k.split('_')[1]) for k in window_results.keys()], list(window_results.values()), 'o-')
    plt.xlabel('窗口大小')
    plt.ylabel('测试损失')
    plt.title('窗口大小影响分析 (差分模式)')

    # 预测步长结果
    step_results = {k: v for k, v in results.items() if 'pred_step' in k}
    plt.subplot(1, 3, 3)
    plt.plot([int(k.split('_')[2]) for k in step_results.keys()], list(step_results.values()), 'o-')
    plt.xlabel('预测步长')
    plt.ylabel('测试损失')
    plt.title('预测步长影响分析 (差分模式)')

    plt.tight_layout()
    plt.savefig('lstm_ablation_results.png')
    plt.close()

    return results


# ======================
# 主函数 (使用原始序列)
# ======================
def main():
    # 参数设置
    DATA_PATH = r'C:\Users\ASUS\Desktop\数据科学的数学方法\大作业\Project1-LSTM\ETTdata\ETTm1.csv'
    WINDOW_SIZE = 64  # 使用64个历史点 (约16小时)
    PRED_STEP = 1  # 预测下一步
    BATCH_SIZE = 64
    EPOCHS = 50

    # 1. 数据加载与预处理 (仅提取油温)
    print("加载并预处理油温数据...")
    ot_series, scaler = load_and_preprocess_data(DATA_PATH)
    print(f"已加载 {len(ot_series)} 个油温样本")

    # 2. 创建自回归数据集 (原始序列模式)
    print("创建自回归数据集 (原始序列模式)...")
    dataset = AutoRegressiveDataset(ot_series, window_size=WINDOW_SIZE, pred_step=PRED_STEP, diff_mode=False)
    print(f"数据集大小: {len(dataset)} 个样本")

    # 3. 划分数据集 (时间序列方式)
    print("按时间顺序划分数据集...")
    indices = np.arange(len(dataset))
    train_idx = indices[:int(0.7 * len(dataset))]
    val_idx = indices[int(0.7 * len(dataset)):int(0.85 * len(dataset))]
    test_idx = indices[int(0.85 * len(dataset)):]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # 4. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)} | 测试样本: {len(test_dataset)}")

    # 5. 初始化自回归LSTM模型
    print("初始化自回归LSTM模型...")
    model = AutoRegressiveLSTM(
        input_dim=1,
        output_dim=PRED_STEP,
        hidden_size=64,
        num_layers=3
    )

    # 打印模型结构
    print("\n模型架构:")
    print(model)

    # 6. 训练模型
    print("\n训练模型...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, diff_mode=False)

    # 7. 评估模型
    print("\n评估模型...")
    test_rmse = evaluate_model(model, test_loader, scaler, diff_mode=False)

    # 8. 消融实验 (包含差分处理)
    print("\n执行消融实验 (包含差分处理)...")
    ablation_results = run_ablation_study(ot_series, scaler)

    # 9. 保存最终模型
    torch.save(model.state_dict(), 'auto_regressive_lstm.pth')
    print("\nLSTM模型已成功保存为 'auto_regressive_lstm.pth'")


if __name__ == "__main__":
    main()