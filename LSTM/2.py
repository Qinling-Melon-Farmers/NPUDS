import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import math
import os
import warnings
import torch.nn.functional as F
from sklearn.metrics import r2_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证可复现性
torch.manual_seed(3407)
np.random.seed(3407)
warnings.filterwarnings('ignore')


# ======================
# 自定义损失函数
# ======================
class WeightedMSELoss(nn.Module):
    """
    加权MSE损失函数，对极端值区域赋予更高权重
    """

    def __init__(self, low_threshold=0.2, high_threshold=0.8, low_weight=3.0, high_weight=3.0):
        super(WeightedMSELoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.low_weight = low_weight
        self.high_weight = high_weight

    def forward(self, y_pred, y_true):
        # 计算基础MSE
        base_loss = F.mse_loss(y_pred, y_true, reduction='none')

        # 创建权重矩阵
        weights = torch.ones_like(y_true)

        # 低值区域赋予高权重
        low_mask = (y_true < self.low_threshold)
        weights[low_mask] = self.low_weight

        # 高值区域赋予高权重
        high_mask = (y_true > self.high_threshold)
        weights[high_mask] = self.high_weight

        # 应用权重
        weighted_loss = base_loss * weights
        return torch.mean(weighted_loss)


# ======================
# 数据处理模块 (自回归专用)
# ======================
class AutoRegressiveDataset(Dataset):
    def __init__(self, ot_series, window_size=24, pred_step=1):
        """
        自回归数据集
        :param ot_series: 油温时间序列 (1D数组)
        :param window_size: 历史窗口大小
        :param pred_step: 预测步长
        """
        self.ot_series = ot_series
        self.window_size = window_size
        self.pred_step = pred_step

    def __len__(self):
        return len(self.ot_series) - self.window_size - self.pred_step + 1

    def __getitem__(self, idx):
        # 输入: 历史窗口内的油温值
        x = self.ot_series[idx:idx + self.window_size]

        # 输出: 未来pred_step步的油温值
        y = self.ot_series[idx + self.window_size:idx + self.window_size + self.pred_step]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_and_preprocess_data(file_path, apply_boxcox=True):
    """
    加载并预处理数据，仅提取油温(OT)序列
    :param apply_boxcox: 是否应用Box-Cox变换解决异方差性
    """
    # 加载数据
    df = pd.read_csv(file_path)

    # 数据清洗
    df.fillna(method='ffill', inplace=True)  # 前向填充缺失值
    df.drop_duplicates(inplace=True)

    # 仅提取油温序列
    ot_series = df['OT'].values.reshape(-1, 1)

    # 记录原始数据用于后续反变换
    original_ot = ot_series.copy()

    # Box-Cox变换解决异方差性
    boxcox_lambda = None
    if apply_boxcox:
        # 确保所有值为正
        min_val = np.min(ot_series)
        if min_val <= 0:
            offset = -min_val + 1e-5
            ot_series += offset

        # 应用Box-Cox变换
        ot_series, boxcox_lambda = stats.boxcox(ot_series.flatten())
        ot_series = ot_series.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler()
    scaled_ot = scaler.fit_transform(ot_series).flatten()

    return scaled_ot, scaler, boxcox_lambda, original_ot


# ======================
# 增强的LSTM模型 (带注意力机制)
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


class EnhancedAutoRegressiveLSTM(nn.Module):
    """
    增强的自回归LSTM模型，包含双向LSTM和注意力机制
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(EnhancedAutoRegressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),  # 双向LSTM输出为2*hidden_size
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态 (双向LSTM需要2倍隐藏状态)
        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM处理
        lstm_out, _ = self.lstm(x, (h0, c0))  # 输出形状: (batch_size, seq_len, 2*hidden_size)

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, 2*hidden_size)

        # 输出层
        out = self.fc(context)
        return out


# ======================
# 训练与评估函数
# ======================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, use_weighted_loss=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 选择损失函数
    if use_weighted_loss:
        criterion = WeightedMSELoss(low_threshold=0.2, high_threshold=0.8, low_weight=3.0, high_weight=3.0)
    else:
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)

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
            for inputs, targets in val_loader:
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
            torch.save(model.state_dict(), 'best_enhanced_lstm_model.pth')

        # 打印训练进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {current_lr:^8.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('MSE 损失')
    plt.title('增强LSTM训练与验证损失曲线')
    plt.legend()
    plt.savefig('enhanced_lstm_loss_curve.png')
    plt.close()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_enhanced_lstm_model.pth'))
    return model


def inverse_boxcox(value, lambda_val, offset=0):
    """
    反Box-Cox变换
    :param value: 变换后的值
    :param lambda_val: Box-Cox变换使用的lambda值
    :param offset: 应用Box-Cox前添加的偏移量
    :return: 原始值
    """
    if lambda_val == 0:
        return np.exp(value) - offset
    else:
        return (value * lambda_val + 1) ** (1 / lambda_val) - offset


def evaluate_model(model, test_loader, scaler, boxcox_lambda=None, original_min=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
            outputs = model(inputs)

            # 收集结果
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    # 转换为numpy数组
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)

    # 反归一化
    actuals = scaler.inverse_transform(actuals).flatten()
    predictions = scaler.inverse_transform(predictions).flatten()

    # 反Box-Cox变换
    if boxcox_lambda is not None:
        # 计算原始数据的最小值（用于偏移）
        if original_min is None:
            offset = 0
        else:
            offset = max(0, -original_min + 1e-5) if original_min <= 0 else 0

        actuals = inverse_boxcox(actuals, boxcox_lambda, offset)
        predictions = inverse_boxcox(predictions, boxcox_lambda, offset)

    # 计算评估指标
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # 计算R²分数
    r2 = r2_score(actuals, predictions)

    # 计算时间序列R²（基于持久性模型）
    persistence_pred = np.roll(actuals, 1)
    persistence_pred[0] = actuals[0]
    persistence_mse = np.mean((actuals - persistence_pred) ** 2)
    ts_r2 = 1 - (mse / persistence_mse)

    print("\n" + "=" * 70)
    print(f"{'增强LSTM模型综合评估指标':^70}")
    print("=" * 70)
    print(f"均方误差 (MSE):            {mse:.4f}")
    print(f"均方根误差 (RMSE):         {rmse:.4f}")
    print(f"平均绝对误差 (MAE):         {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE):    {mape:.2f}%")
    print(f"R²分数:                   {r2:.4f}")
    print(f"时间序列R²:               {ts_r2:.4f}")

    # 绘制预测对比图
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:200], 'b-', label='实际油温')
    plt.plot(predictions[:200], 'r--', label='预测油温')

    # 计算滚动RMSE
    rolling_rmse = np.zeros(200)
    for i in range(1, 201):
        if i < 10:
            rolling_rmse[i - 1] = np.sqrt(np.mean((actuals[:i] - predictions[:i]) ** 2))
        else:
            rolling_rmse[i - 1] = np.sqrt(np.mean((actuals[i - 10:i] - predictions[i - 10:i]) ** 2))

    plt.fill_between(range(200),
                     predictions[:200] - rolling_rmse,
                     predictions[:200] + rolling_rmse,
                     color='gray', alpha=0.3, label='±滚动RMSE区间')

    plt.xlabel('时间点')
    plt.ylabel('油温值(℃)')
    plt.title('增强LSTM实际油温与预测油温对比')
    plt.legend()
    plt.savefig('enhanced_lstm_prediction_comparison.png')
    plt.close()

    # 绘制残差图
    residuals = actuals - predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', label='零误差线')

    # 添加回归线
    z = np.polyfit(predictions, residuals, 1)
    p = np.poly1d(z)
    plt.plot(predictions, p(predictions), "r--", label='残差趋势线')

    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('增强LSTM预测残差分析')
    plt.legend()
    plt.savefig('enhanced_lstm_residual_plot.png')
    plt.close()

    # 新增指标对比图
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'TS R²']
    values = [mse, rmse, mae, mape, r2, ts_r2]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.ylabel('指标值')
    plt.title('增强LSTM模型评估指标对比')

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}' if abs(height) < 10 else f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('enhanced_lstm_metrics_comparison.png')
    plt.close()

    return rmse, mape, ts_r2


# ======================
# 消融实验
# ======================
def run_ablation_study(ot_series, scaler, boxcox_lambda=None, original_min=None):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 50)
    print(f"{'运行增强LSTM消融实验':^50}")
    print("=" * 50)

    # 实验1: 不同窗口大小
    window_sizes = [8, 16, 32, 64, 128]
    print("\n>> 实验1: 窗口大小影响分析")
    for ws in window_sizes:
        dataset = AutoRegressiveDataset(ot_series, window_size=ws)

        # 时间序列划分 (70%-15%-15%)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # 初始化模型
        model = EnhancedAutoRegressiveLSTM(
            input_dim=1,
            output_dim=1,
            hidden_size=128,
            num_layers=2
        ).to(device)

        # 快速训练 (仅用于消融实验)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = WeightedMSELoss()  # 使用加权损失

        model.train()
        for epoch in range(10):  # 减少训练轮数以加速
            for inputs, targets in train_loader:
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'window_{ws}'] = test_loss
        print(f"窗口大小: {ws:3d} | 测试损失: {test_loss:.6f}")

    # 实验2: 不同输出步长
    pred_steps = [1, 3, 6, 12]  # 预测1步(15min), 3步(45min), 6步(1.5h), 12步(3h)
    print("\n>> 实验2: 预测步长影响分析")
    for ps in pred_steps:
        dataset = AutoRegressiveDataset(ot_series, window_size=64, pred_step=ps)

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
        model = EnhancedAutoRegressiveLSTM(
            input_dim=1,
            output_dim=ps,
            hidden_size=128,
            num_layers=2
        ).to(device)

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = WeightedMSELoss()  # 使用加权损失

        model.train()
        for epoch in range(10):
            for inputs, targets in train_loader:
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'pred_step_{ps}'] = test_loss
        print(f"预测步长: {ps:2d} | 测试损失: {test_loss:.6f}")

    # 实验3: 不同模型组件的影响
    components = ['Base', '+Bidirectional', '+Attention', '+WeightedLoss', 'Full']
    print("\n>> 实验3: 模型组件影响分析")

    for comp in components:
        dataset = AutoRegressiveDataset(ot_series, window_size=64)

        # 时间序列划分
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # 根据组件创建模型
        if comp == 'Base':
            model = AutoRegressiveLSTM(
                input_dim=1,
                output_dim=1,
                hidden_size=64,
                num_layers=2
            ).to(device)
            criterion = nn.MSELoss()
        elif comp == '+Bidirectional':
            model = EnhancedAutoRegressiveLSTM(
                input_dim=1,
                output_dim=1,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            )
            # 移除注意力层
            model.attention = nn.Identity()
            model.to(device)
            criterion = nn.MSELoss()
        elif comp == '+Attention':
            model = EnhancedAutoRegressiveLSTM(
                input_dim=1,
                output_dim=1,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ).to(device)
            criterion = nn.MSELoss()
        elif comp == '+WeightedLoss':
            model = AutoRegressiveLSTM(
                input_dim=1,
                output_dim=1,
                hidden_size=64,
                num_layers=2
            ).to(device)
            criterion = WeightedMSELoss()
        else:  # Full
            model = EnhancedAutoRegressiveLSTM(
                input_dim=1,
                output_dim=1,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ).to(device)
            criterion = WeightedMSELoss()

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(10):
            for inputs, targets in train_loader:
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'component_{comp}'] = test_loss
        print(f"组件: {comp:15} | 测试损失: {test_loss:.6f}")

    # 保存结果
    print("\n>> 增强LSTM消融实验结果汇总")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:20}: {v:.6f}")

    # 可视化结果
    plt.figure(figsize=(15, 5))

    # 窗口大小结果
    plt.subplot(1, 3, 1)
    window_results = {k: v for k, v in results.items() if 'window' in k}
    plt.plot([int(k.split('_')[1]) for k in window_results.keys()], list(window_results.values()), 'o-')
    plt.xlabel('窗口大小')
    plt.ylabel('测试损失')
    plt.title('窗口大小影响分析')

    # 预测步长结果
    plt.subplot(1, 3, 2)
    step_results = {k: v for k, v in results.items() if 'pred_step' in k}
    plt.plot([int(k.split('_')[2]) for k in step_results.keys()], list(step_results.values()), 'o-')
    plt.xlabel('预测步长')
    plt.ylabel('测试损失')
    plt.title('预测步长影响分析')

    # 模型组件结果
    plt.subplot(1, 3, 3)
    comp_results = {k: v for k, v in results.items() if 'component' in k}
    comp_names = [k.split('_')[1] for k in comp_results.keys()]
    plt.bar(comp_names, list(comp_results.values()))
    plt.xlabel('模型组件')
    plt.ylabel('测试损失')
    plt.title('模型组件影响')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('enhanced_lstm_ablation_results.png')
    plt.close()

    return results


# ======================
# 极端值专用模型
# ======================
class ExtremeValueModel(nn.Module):
    """
    极端值专用模型，处理低值和高值区域的预测
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_size=64, num_layers=2):
        super(ExtremeValueModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_extreme_value_model(train_loader, val_loader, epochs=30, lr=0.001):
    """
    训练极端值专用模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExtremeValueModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f"极端值模型 Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.6f}")

    return model


# ======================
# 主函数
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
    ot_series, scaler, boxcox_lambda, original_ot = load_and_preprocess_data(DATA_PATH, apply_boxcox=True)
    print(f"已加载 {len(ot_series)} 个油温样本")

    # 计算原始数据的最小值（用于Box-Cox反变换）
    original_min = np.min(original_ot)

    # 2. 创建自回归数据集
    print("创建自回归数据集...")
    dataset = AutoRegressiveDataset(ot_series, window_size=WINDOW_SIZE, pred_step=PRED_STEP)
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

    # 5. 初始化增强的自回归LSTM模型
    print("初始化增强的自回归LSTM模型...")
    model = EnhancedAutoRegressiveLSTM(
        input_dim=1,
        output_dim=PRED_STEP,
        hidden_size=128,
        num_layers=3,
        dropout=0.2
    )

    # 打印模型结构
    print("\n模型架构:")
    print(model)

    # 6. 训练模型
    print("\n训练增强模型...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, use_weighted_loss=True)

    # 7. 评估模型
    print("\n评估增强模型...")
    test_rmse, test_mape, ts_r2 = evaluate_model(model, test_loader, scaler, boxcox_lambda, original_min)

    # 8. 创建极端值专用数据集
    print("\n创建极端值专用数据集...")
    # 识别极端值样本（低值和高值）
    low_threshold = np.percentile(ot_series, 10)  # 最低10%的值
    high_threshold = np.percentile(ot_series, 90)  # 最高10%的值

    extreme_indices = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        if target < low_threshold or target > high_threshold:
            extreme_indices.append(i)

    extreme_dataset = torch.utils.data.Subset(dataset, extreme_indices)
    print(f"极端值样本数量: {len(extreme_dataset)}")

    # 划分极端值数据集
    extreme_train_size = int(0.8 * len(extreme_dataset))
    extreme_val_size = len(extreme_dataset) - extreme_train_size
    extreme_train_dataset, extreme_val_dataset = random_split(
        extreme_dataset, [extreme_train_size, extreme_val_size])

    extreme_train_loader = DataLoader(extreme_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    extreme_val_loader = DataLoader(extreme_val_dataset, batch_size=BATCH_SIZE)

    # 9. 训练极端值专用模型
    print("\n训练极端值专用模型...")
    extreme_model = train_extreme_value_model(extreme_train_loader, extreme_val_loader, epochs=30)

    # 10. 集成预测
    def ensemble_predict(model, extreme_model, inputs, device):
        """
        集成预测函数：主模型预测 + 极端值修正
        """
        # 使用主模型进行预测
        inputs = inputs.unsqueeze(-1)
        main_pred = model(inputs)

        # 识别极端值预测
        with torch.no_grad():
            extreme_pred = extreme_model(inputs)

        # 如果主模型预测为极端值，则使用极端值模型的预测
        low_threshold = 0.2  # 归一化后的低阈值
        high_threshold = 0.8  # 归一化后的高阈值

        # 创建混合预测
        mixed_pred = main_pred.clone()
        extreme_mask = (main_pred < low_threshold) | (main_pred > high_threshold)
        mixed_pred[extreme_mask] = 0.7 * main_pred[extreme_mask] + 0.3 * extreme_pred[extreme_mask]

        return mixed_pred

    # 11. 评估集成模型
    print("\n评估集成模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    extreme_model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = ensemble_predict(model, extreme_model, inputs, device)

            # 收集结果
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    # 转换为numpy数组
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)

    # 反归一化
    actuals = scaler.inverse_transform(actuals).flatten()
    predictions = scaler.inverse_transform(predictions).flatten()

    # 反Box-Cox变换
    if boxcox_lambda is not None:
        offset = max(0, -original_min + 1e-5) if original_min <= 0 else 0
        actuals = inverse_boxcox(actuals, boxcox_lambda, offset)
        predictions = inverse_boxcox(predictions, boxcox_lambda, offset)

    # 计算评估指标
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # 计算时间序列R²
    persistence_pred = np.roll(actuals, 1)
    persistence_pred[0] = actuals[0]
    persistence_mse = np.mean((actuals - persistence_pred) ** 2)
    ts_r2 = 1 - (mse / persistence_mse)

    print("\n" + "=" * 70)
    print(f"{'集成模型综合评估指标':^70}")
    print("=" * 70)
    print(f"均方误差 (MSE):            {mse:.4f}")
    print(f"均方根误差 (RMSE):         {rmse:.4f}")
    print(f"平均绝对误差 (MAE):         {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE):    {mape:.2f}%")
    print(f"R²分数:                   {r2:.4f}")
    print(f"时间序列R²:               {ts_r2:.4f}")

    # 12. 消融实验
    print("\n执行消融实验...")
    ablation_results = run_ablation_study(ot_series, scaler, boxcox_lambda, original_min)

    # 13. 保存最终模型
    torch.save(model.state_dict(), 'enhanced_auto_regressive_lstm.pth')
    torch.save(extreme_model.state_dict(), 'extreme_value_model.pth')
    print("\n模型已成功保存: 'enhanced_auto_regressive_lstm.pth' 和 'extreme_value_model.pth'")


if __name__ == "__main__":
    main()