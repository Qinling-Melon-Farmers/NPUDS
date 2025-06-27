import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import correlate

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')


# ======================
# 数据处理模块
# ======================
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

    # 记录原始数据用于后续反变换
    original_ot = ot_series.copy()
    original_min = np.min(original_ot)

    # 归一化
    scaler = MinMaxScaler()
    scaled_ot = scaler.fit_transform(ot_series).flatten()

    return scaled_ot, scaler, original_ot


def create_diff_dataset(data, lookback=24):
    """创建差分数据集，打破时间滞后问题"""
    # 一阶差分：ΔT = T(t) - T(t-1)
    delta = np.diff(data, n=1)

    # 基础值数组 (用于还原预测值)
    base_values = data[:-1]  # t-1时刻的值

    # 构建序列样本
    X, y, base = [], [], []
    for i in range(lookback, len(delta)):
        # 创建二维格式 (seq_len, 1)
        X.append(delta[i - lookback:i].reshape(-1, 1).astype(np.float32))
        y.append(delta[i])
        base.append(base_values[i])  # 对应位置的基础值

    # 转换为数组
    X = np.array(X)
    y = np.array(y, dtype=np.float32)
    base = np.array(base, dtype=np.float32)

    return X, y, base


def directional_accuracy(y_true, y_pred):
    """计算方向准确率"""
    true_dir = np.sign(y_true[1:] - y_true[:-1])
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

# ======================
# 增强的LSTM模型 (带注意力机制)
# ======================
class EnhancedAutoRegressiveLSTM(nn.Module):
    """
    增强的自回归LSTM模型，包含注意力机制
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(EnhancedAutoRegressiveLSTM, self).__init__()
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

        # 注意力机制
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.attention_vector = nn.Linear(hidden_size, 1, bias=False)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # 确保输入为 float32 类型
        x = x.float()

        # 检查输入维度
        if x.dim() == 2:
            # 如果输入是二维 (batch_size, seq_len)，则添加特征维度
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            # 已经是三维 (batch_size, seq_len, features)，无需修改
            pass
        else:
            raise ValueError(f"LSTM: Expected input to be 2D or 3D, got {x.dim()}D instead")

        # 获取维度信息
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # 确保隐藏状态也是 float32
        h0 = h0.float()
        c0 = c0.float()

        # LSTM处理
        lstm_out, _ = self.lstm(x, (h0, c0))  # 输出形状: (batch_size, seq_len, hidden_size)

        # 注意力机制
        # 计算注意力权重
        u = torch.tanh(self.attention_linear(lstm_out))  # (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention_vector(u), dim=1)  # (batch_size, seq_len, 1)

        # 计算上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size)

        # 输出层
        out = self.fc(context)
        return out, attn_weights


# ======================
# 训练与评估函数
# ======================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
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
        for inputs, targets, _ in train_loader:  # 忽略基础值
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            # 确保输出和目标是标量
            outputs = outputs.squeeze()
            targets = targets.squeeze()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:  # 忽略基础值
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)

                # 确保输出和目标是标量
                outputs = outputs.squeeze()
                targets = targets.squeeze()

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



def calculate_phase_shift(actuals, predictions):
    """
    计算相位偏移 - 修复方向问题
    """
    # 计算互相关
    cross_corr = correlate(actuals - np.mean(actuals),
                           predictions - np.mean(predictions),
                           mode='full')

    # 找到最大相关位置
    lags = np.arange(-len(predictions) + 1, len(predictions))
    max_idx = np.argmax(cross_corr)
    lag = lags[max_idx]

    return lag, cross_corr, lags


def evaluate_model(model, test_loader, scaler, offset=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    actuals_diff, predictions_diff, actuals_base, predictions_recon = [], [], [], []
    all_attn_weights = []

    with torch.no_grad():
        for inputs, targets, base in test_loader:
            inputs = inputs.to(device)
            outputs, attn_weights = model(inputs)

            # 确保输出是标量
            outputs = outputs.squeeze()

            # 将结果转换为NumPy数组并展平
            targets_np = targets.cpu().numpy().flatten()
            outputs_np = outputs.cpu().numpy().flatten()
            base_np = base.cpu().numpy().flatten()

            # 收集结果
            actuals_diff.extend(targets_np)
            predictions_diff.extend(outputs_np)
            actuals_base.extend(base_np)

            # 还原预测值: T(t) = T(t-1) + ΔT_pred
            recon = base_np + outputs_np
            predictions_recon.extend(recon)

            # 收集注意力权重
            if attn_weights is not None:
                all_attn_weights.extend(attn_weights.detach().cpu().numpy())

    # 转换为numpy数组
    actuals_diff = np.array(actuals_diff)
    predictions_diff = np.array(predictions_diff)
    actuals_base = np.array(actuals_base)
    predictions_recon = np.array(predictions_recon)

    # 计算真实值: T(t) = T(t-1) + ΔT_true
    actuals_recon = actuals_base + actuals_diff

    # 反归一化
    actuals_recon = scaler.inverse_transform(actuals_recon.reshape(-1, 1)).flatten()
    predictions_recon = scaler.inverse_transform(predictions_recon.reshape(-1, 1)).flatten()


    # 计算评估指标
    mse = mean_squared_error(actuals_recon, predictions_recon)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_recon, predictions_recon)

    # 使用安全的MAPE计算
    def safe_mape(y_true, y_pred, epsilon=1e-5):
        abs_true = np.abs(y_true)
        safe_true = np.where(abs_true < epsilon, epsilon, abs_true)
        return np.mean(np.abs((y_true - y_pred) / safe_true)) * 100

    mape = safe_mape(actuals_recon, predictions_recon)

    # 计算R²分数
    r2 = r2_score(actuals_recon, predictions_recon)

    # 计算时间序列R²（改进的持久性模型）
    def calculate_persistence_prediction(actuals):
        persistence_pred = np.zeros_like(actuals)
        persistence_pred[0] = actuals[0]
        persistence_pred[1:] = actuals[:-1]
        return persistence_pred

    persistence_pred = calculate_persistence_prediction(actuals_recon)
    persistence_mse = mean_squared_error(actuals_recon, persistence_pred)
    ts_r2 = 1 - (mse / persistence_mse) if persistence_mse > 0 else 0.0

    # 计算方向准确率
    da = directional_accuracy(actuals_recon, predictions_recon)

    # 计算相位偏移（修复方向）
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

    phase_shift, cross_corr = calculate_phase_shift(actuals_recon, predictions_recon)
    # 计算校正后的DA
    adjusted_da = calculate_adjusted_da(actuals_recon, predictions_recon, phase_shift)

    # 在打印评估指标处添加
    print(f"方向准确率 (DA):           {da:.4f}")
    print(f"校正后方向准确率 (Adj-DA):  {adjusted_da:.4f}")
    print(f"相位偏移 (步数):           {phase_shift}")

    # 在指标对比图中添加
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'TS R²', 'DA', 'Adj-DA']
    values = [mse, rmse, mae, mape, r2, ts_r2, da, adjusted_da]
    print("\n" + "=" * 70)
    print(f"{'增强LSTM模型综合评估指标 (差分转换后)':^70}")
    print("=" * 70)
    print(f"均方误差 (MSE):            {mse:.4f}")
    print(f"均方根误差 (RMSE):         {rmse:.4f}")
    print(f"平均绝对误差 (MAE):         {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE):    {mape:.2f}%")
    print(f"R²分数:                   {r2:.4f}")
    print(f"时间序列R²:               {ts_r2:.4f}")
    print(f"方向准确率 (DA):           {da:.4f}")
    print(f"相位偏移 (步数):           {phase_shift}")

    # 绘制预测对比图
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_recon[:200], 'b-', label='实际油温')
    plt.plot(predictions_recon[:200], 'r--', label='预测油温')

    # 绘制相位偏移后的预测（修复方向）
    if phase_shift != 0:
        # 根据相位偏移方向调整
        if phase_shift > 0:
            # 正相位偏移：预测落后于实际值，需要向前移动预测
            adjusted_predictions = np.roll(predictions_recon, -phase_shift)
            adjusted_predictions[:phase_shift] = np.nan
        else:
            # 负相位偏移：预测超前于实际值，需要向后移动预测
            adjusted_predictions = np.roll(predictions_recon, -phase_shift)
            adjusted_predictions[-phase_shift:] = np.nan

        plt.plot(adjusted_predictions[:200], 'g-.', label=f'相位校正预测(偏移:{phase_shift}步)')

    # 计算滚动RMSE
    rolling_rmse = np.zeros(200)
    for i in range(1, 201):
        if i < 10:
            rolling_rmse[i - 1] = np.sqrt(np.mean((actuals_recon[:i] - predictions_recon[:i]) ** 2))
        else:
            rolling_rmse[i - 1] = np.sqrt(np.mean((actuals_recon[i - 10:i] - predictions_recon[i - 10:i]) ** 2))

    plt.fill_between(range(200),
                     predictions_recon[:200] - rolling_rmse,
                     predictions_recon[:200] + rolling_rmse,
                     color='gray', alpha=0.3, label='±滚动RMSE区间')

    plt.xlabel('时间点')
    plt.ylabel('油温值(℃)')
    plt.title('增强LSTM实际油温与预测油温对比 (差分转换)')
    plt.legend()
    plt.savefig('enhanced_lstm_prediction_comparison_diff.png')
    plt.close()

    # 绘制残差图
    residuals = actuals_recon - predictions_recon
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions_recon, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', label='零误差线')

    # 添加回归线
    z = np.polyfit(predictions_recon, residuals, 1)
    p = np.poly1d(z)
    plt.plot(predictions_recon, p(predictions_recon), "r--", label='残差趋势线')

    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('增强LSTM预测残差分析 (差分转换)')
    plt.legend()
    plt.savefig('enhanced_lstm_residual_plot_diff.png')
    plt.close()

    # 绘制注意力权重热力图
    if len(all_attn_weights) > 0:
        sample_attn = all_attn_weights[0].squeeze()  # 取第一个样本的注意力权重
        plt.figure(figsize=(10, 6))
        plt.imshow(sample_attn.reshape(1, -1), cmap='viridis', aspect='auto')
        plt.colorbar(label='注意力权重')
        plt.xlabel('时间步')
        plt.ylabel('样本')
        plt.title('注意力权重分布 (第一个样本)')
        plt.savefig('enhanced_lstm_attention_weights.png')
        plt.close()

    # 新增指标对比图
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'TS R²', 'DA']
    values = [mse, rmse, mae, mape, r2, ts_r2, da]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    plt.ylabel('指标值')
    plt.title('增强LSTM模型评估指标对比 (差分转换)')

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}' if abs(height) < 10 else f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('enhanced_lstm_metrics_comparison_diff.png')
    plt.close()

    return rmse, mape, ts_r2, da, phase_shift


# ======================
# 主函数
# ======================
def main():
    # 参数设置
    DATA_PATH = r'C:\Users\ASUS\Desktop\数据科学的数学方法\大作业\Project1-LSTM\ETTdata\ETTm1.csv'
    WINDOW_SIZE = 24  # 历史窗口大小
    PRED_STEP = 1  # 预测步长
    BATCH_SIZE = 64
    EPOCHS = 50

    # 1. 数据加载与预处理 (仅提取油温)
    print("加载并预处理油温数据...")
    scaled_ot, scaler,  original_ot = load_and_preprocess_data(DATA_PATH)
    print(f"已加载 {len(scaled_ot)} 个油温样本")

    # 2. 创建差分数据集
    print("创建差分数据集...")
    X_diff, y_diff, base_values = create_diff_dataset(scaled_ot, lookback=WINDOW_SIZE)
    print(f"差分数据集大小: {len(X_diff)} 个样本")

    # 3. 创建PyTorch数据集
    print("创建自回归数据集...")
    # 将差分序列和基础值数组组合成数据集
    dataset = []
    for i in range(len(X_diff)):
        # 输入数据是二维格式 (seq_len, 1)
        input_data = torch.tensor(X_diff[i], dtype=torch.float32)
        target = torch.tensor(y_diff[i], dtype=torch.float32)
        base = torch.tensor(base_values[i], dtype=torch.float32)
        dataset.append((input_data, target, base))

    # 4. 划分数据集 (时间序列方式)
    print("按时间顺序划分数据集...")
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)} | 测试样本: {len(test_dataset)}")

    # 5. 创建数据加载器 (使用自定义collate_fn)
    def collate_fn(batch):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        bases = [item[2] for item in batch]

        # 堆叠输入数据
        inputs = torch.stack(inputs)  # 自动添加批次维度
        targets = torch.stack(targets)
        bases = torch.stack(bases)

        return inputs, targets, bases

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 6. 初始化增强的自回归LSTM模型
    print("初始化增强的自回归LSTM模型...")
    model = EnhancedAutoRegressiveLSTM(
        input_dim=1,
        output_dim=PRED_STEP,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )

    # 打印模型结构
    print("\n模型架构:")
    print(model)

    # 7. 训练模型
    print("\n训练增强模型...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # 8. 评估模型
    print("\n评估增强模型...")
    test_rmse, test_mape, ts_r2, da, phase_shift = evaluate_model(
        model, test_loader, scaler)

    # 9. 保存最终模型
    torch.save(model.state_dict(), 'enhanced_auto_regressive_lstm_diff.pth')
    print("\n模型已成功保存: 'enhanced_auto_regressive_lstm_diff.pth'")


if __name__ == "__main__":
    main()