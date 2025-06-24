import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import math
import os
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)


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


# ======================
# Transformer模型 (自回归)
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码层
        :param d_model: 模型的特征维度
        :param max_len: 可支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项，用于缩放频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数列使用sin函数编码位置信息
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列使用cos函数编码位置信息
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整张量形状为 (1, max_len, d_model) 并注册为buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AutoRegressiveTransformer(nn.Module):
    """
    自回归Transformer模型，仅使用OT历史数据预测未来OT值
    """
    def __init__(self, input_dim=1, output_dim=1, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(AutoRegressiveTransformer, self).__init__()
        # 输入嵌入层 (将标量转换为向量)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.d_model = d_model

    def forward(self, src):
        # src形状: (seq_len, batch_size, input_dim)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Transformer处理
        output = self.transformer_encoder(src)

        # 只取最后一个时间步的输出用于预测
        output = self.fc(output[-1])
        return output


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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 增加维度: (batch_size, seq_len) -> (seq_len, batch_size, 1)
            inputs = inputs.unsqueeze(-1).permute(1, 0, 2)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(1)

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
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印训练进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {current_lr:^8.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('MSE 损失')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model


def evaluate_model(model, test_loader, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
            outputs = model(inputs)

            # 收集结果
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    # 反归一化
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)

    actuals = scaler.inverse_transform(actuals).flatten()
    predictions = scaler.inverse_transform(predictions).flatten()

    # 计算评估指标
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # MAPE计算

    print("\n" + "=" * 60)
    print(f"{'模型综合评估指标':^60}")
    print("=" * 60)
    print(f"均方误差 (MSE):       {mse:.4f}")
    print(f"均方根误差 (RMSE):    {rmse:.4f}")
    print(f"平均绝对误差 (MAE):    {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

    # 绘制预测对比图
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:200], 'b-', label='实际油温')
    plt.plot(predictions[:200], 'r--', label='预测油温')
    plt.fill_between(range(200),
                     actuals[:200] - rmse,
                     actuals[:200] + rmse,
                     color='gray', alpha=0.3, label='±RMSE区间')
    plt.xlabel('时间点')
    plt.ylabel('油温值(℃)')
    plt.title('实际油温与预测油温对比')
    plt.legend()
    plt.savefig('prediction_comparison.png')
    plt.close()

    # 绘制残差图
    residuals = actuals - predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', label='零误差线')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('预测残差分析')
    plt.legend()
    plt.savefig('residual_plot.png')
    plt.close()

    # 新增指标对比图
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
    values = [mse, rmse, mae, mape]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('指标值')
    plt.title('模型评估指标对比')

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}' if height < 1 else f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

    return rmse, mape


# ======================
# 消融实验
# ======================
def run_ablation_study(ot_series, scaler):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*50)
    print(f"{'运行消融实验':^50}")
    print("="*50)

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
        model = AutoRegressiveTransformer(
            input_dim=1,
            output_dim=1,
            d_model=64,
            nhead=4,
            num_layers=2
        ).to(device)

        # 训练模型
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 快速训练 (仅用于消融实验)
        model.train()
        for epoch in range(10):  # 减少训练轮数以加速
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
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
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(1)

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
        model = AutoRegressiveTransformer(
            input_dim=1,
            output_dim=ps,
            d_model=64,
            nhead=4,
            num_layers=2
        ).to(device)

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
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
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(1)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'pred_step_{ps}'] = test_loss
        print(f"预测步长: {ps:2d} | 测试损失: {test_loss:.6f}")

    # 实验3: 不同数据划分方式
    print("\n>> 实验3: 数据划分策略分析")
    dataset = AutoRegressiveDataset(ot_series, window_size=64)

    # 方式1: 随机划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset1, val_dataset1, test_dataset1 = random_split(
        dataset, [train_size, val_size, test_size])

    # 方式2: 时间序列划分
    indices = np.arange(len(dataset))
    train_idx = indices[:int(0.7 * len(dataset))]
    val_idx = indices[int(0.7 * len(dataset)):int(0.85 * len(dataset))]
    test_idx = indices[int(0.85 * len(dataset)):]

    train_dataset2 = torch.utils.data.Subset(dataset, train_idx)
    val_dataset2 = torch.utils.data.Subset(dataset, val_idx)
    test_dataset2 = torch.utils.data.Subset(dataset, test_idx)

    strategies = [('Random Split', train_dataset1, test_dataset1),
                  ('Temporal Split', train_dataset2, test_dataset2)]

    for name, train_set, test_set in strategies:
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64)

        # 初始化模型
        model = AutoRegressiveTransformer(
            input_dim=1,
            output_dim=1,
            d_model=64,
            nhead=4,
            num_layers=2
        ).to(device)

        # 快速训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
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
                inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(1)

        test_loss = test_loss / len(test_loader.dataset)
        results[f'split_{name.replace(" ", "_")}'] = test_loss
        print(f"{name:15} | Test Loss: {test_loss:.6f}")

    # 保存结果
    print("\n>> 消融实验结果汇总")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k:15}: {v:.6f}")

    # 可视化结果
    plt.figure(figsize=(12, 6))

    # 窗口大小结果
    window_results = {k: v for k, v in results.items() if 'window' in k}
    plt.subplot(1, 3, 1)
    plt.plot([int(k.split('_')[1]) for k in window_results.keys()], list(window_results.values()), 'o-')
    plt.xlabel('窗口大小')
    plt.ylabel('测试损失')
    plt.title('窗口大小影响分析')

    # 预测步长结果
    step_results = {k: v for k, v in results.items() if 'pred_step' in k}
    plt.subplot(1, 3, 2)
    plt.plot([int(k.split('_')[2]) for k in step_results.keys()], list(step_results.values()), 'o-')
    plt.xlabel('预测步长')
    plt.ylabel('测试损失')
    plt.title('预测步长影响分析')

    # 数据划分结果
    split_results = {k: v for k, v in results.items() if 'split' in k}
    plt.subplot(1, 3, 3)
    plt.bar([k.split('_')[1] for k in split_results.keys()], list(split_results.values()))
    plt.xlabel('划分策略')
    plt.ylabel('测试损失')
    plt.title('数据划分策略影响')

    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.close()

    return results


# ======================
# 主函数
# ======================
def main():
    # 参数设置
    DATA_PATH = './ETTdata/ETTm1.csv'
    WINDOW_SIZE = 64  # 使用64个历史点 (约16小时)
    PRED_STEP = 1  # 预测下一步
    BATCH_SIZE = 64
    EPOCHS = 50

    # 1. 数据加载与预处理 (仅提取油温)
    print("加载并预处理油温数据...")
    ot_series, scaler = load_and_preprocess_data(DATA_PATH)
    print(f"已加载 {len(ot_series)} 个油温样本")

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

    # 5. 初始化自回归Transformer模型
    print("初始化自回归Transformer模型...")
    model = AutoRegressiveTransformer(
        input_dim=1,
        output_dim=PRED_STEP,
        d_model=64,
        nhead=4,
        num_layers=3
    )

    # 打印模型结构
    print("\n模型架构:")
    print(model)

    # 6. 训练模型
    print("\n训练模型...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # 7. 评估模型
    print("\n评估模型...")
    test_rmse = evaluate_model(model, test_loader, scaler)

    # 8. 消融实验
    print("\n执行消融实验...")
    ablation_results = run_ablation_study(ot_series, scaler)

    # 9. 保存最终模型
    torch.save(model.state_dict(), 'auto_regressive_transformer.pth')
    print("\n模型已成功保存为 'auto_regressive_transformer.pth'")


if __name__ == "__main__":
    main()