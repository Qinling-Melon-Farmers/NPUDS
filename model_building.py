import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class SimplifiedPredictor(nn.Module):
    """
    简化的预测专用模型
    专注于油温预测，不包含重构任务
    """

    def __init__(self, input_size=7, hidden_size=128, output_size=1, predict_steps=1):
        super().__init__()
        self.input_size = input_size  # 确保这一行存在
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.predict_steps = predict_steps

        # 增强的编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,  # 增加层数
            batch_first=True,
            dropout=0.2
        )

        # 预测解码器
        self.predict_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, output_size * predict_steps),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.encoder(x)
        prediction = self.predict_decoder(hidden[-1])  # 取最后一层的隐藏状态
        return prediction.view(-1, self.predict_steps)


def build_simplified_predictor(input_size=7, hidden_size=128, output_size=1, predict_steps=1, device='cpu'):
    """
    构建简化版的预测专用模型

    参数:
    input_size: 输入特征维度 (默认为7)
    hidden_size: LSTM隐藏层大小
    output_size: 输出维度 (默认为1，即OT值)
    predict_steps: 预测步数 (默认为1)
    device: 计算设备 (默认为'cpu')

    返回:
    model: 构建好的 SimplifiedPredictor 模型
    """
    model = SimplifiedPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        predict_steps=predict_steps
    )
    return model.to(device)



def train_predict_model(model, X_train, y_train, epochs=100, batch_size=32, device='cpu'):
    """
    训练预测专用模型
    """
    # 转换为张量
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练循环
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_true = []

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            # 前向传播
            predictions = model(X_batch)

            # 计算损失
            loss = criterion(predictions, y_batch)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(predictions[:, 0].detach().cpu().numpy())
            all_true.extend(y_batch[:, 0].cpu().numpy())

        # 更新学习率
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        # 计算指标
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_predict_model.pth')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_predict_model.pth'))
    return model


def predict_next_step(model, X, device='cpu'):
    """
    普通预测 (X_{n+1}) - 简化模型版本

    参数:
    model: 训练好的模型
    X: 输入序列 [samples, window_size, features]
    device: 计算设备

    返回:
    预测结果 [samples] (只预测下一步)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        prediction = model(X_tensor)  # 简化模型直接返回预测值

        # 确保预测形状正确
        if prediction.dim() == 2:
            # 预测形状应为 [samples, predict_steps]
            return prediction[:, 0].cpu().numpy()
        else:
            raise ValueError(f"意外预测形状: {prediction.shape}")


def predict_long_term(model, initial_seq, steps=3, device='cpu'):
    """
    长时延预测 (X_{n+k}) - 简化模型版本

    参数:
    model: 训练好的模型
    initial_seq: 初始序列 [1, window_size, features]
    steps: 预测步数
    device: 计算设备

    返回:
    预测序列 [steps] (OT值)
    """
    model.eval()
    with torch.no_grad():
        # 确保初始序列是3D
        if len(initial_seq.shape) == 2:
            initial_seq = initial_seq.reshape(1, initial_seq.shape[1], -1)

        current_seq = torch.tensor(initial_seq, dtype=torch.float32).to(device)
        predictions = []

        for _ in range(steps):
            # 预测下一个点
            prediction = model(current_seq)  # 简化模型直接返回预测值
            next_val = prediction[0, 0].item()  # 取第一个样本的第一步预测

            # 创建新点 - 只有OT值，其他特征未知
            new_point = np.zeros((1, 1, model.input_size))
            new_point[0, 0, -1] = next_val  # 假设OT是最后一个特征

            # 更新序列: 移除第一个点，添加新预测
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                torch.tensor(new_point, dtype=torch.float32).to(device)
            ], dim=1)

            predictions.append(next_val)

    return np.array(predictions)
