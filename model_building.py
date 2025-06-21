import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1,
                 predict_steps=1, reconstruction_mode='full'):
        """
        多任务时间序列模型

        参数:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层大小
        output_size: 输出维度(预测任务)
        predict_steps: 预测步数 (1:普通预测, >1:长时延预测)
        reconstruction_mode: 重构模式 ('full', 'partial')
        """
        super(MultiTaskLSTMModel, self).__init__()
        self.predict_steps = predict_steps
        self.reconstruction_mode = reconstruction_mode

        # LSTM编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # 预测任务解码器
        self.predict_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * predict_steps),
        )

        # 重构任务解码器
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )# 输出与输入相同维度

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入序列 [batch_size, seq_len, input_size]

        返回:
        prediction: 预测结果 [batch_size, predict_steps]
        reconstruction: 重构结果 [batch_size, seq_len] 或 [batch_size, seq_len//2]
        """
        # 编码器
        lstm_out, (hidden, cell) = self.encoder(x)

        # 预测任务
        prediction = self.predict_decoder(hidden.squeeze(0))
        prediction = prediction.view(-1, self.predict_steps)

        # 重构任务
        if self.reconstruction_mode == 'full':
            # 重构整个序列
            reconstruction = self.recon_decoder(lstm_out).squeeze(-1)
        elif self.reconstruction_mode == 'partial':
            # 只重构序列的后半部分
            seq_len = lstm_out.size(1)
            half_len = seq_len // 2
            # 只取后半部分时间步的重构
            recon_part = self.recon_decoder(lstm_out[:, half_len:, :]).squeeze(-1)

            # 创建完整长度的重构输出，前半部分设为0（实际中不计算损失）
            full_recon = torch.zeros_like(x.squeeze(-1))
            full_recon[:, half_len:] = recon_part
            reconstruction = full_recon
        else:
            raise ValueError(f"未知重构模式: {self.reconstruction_mode}")

        return prediction, reconstruction


def build_multi_task_model(window_size, input_size=1, predict_steps=1,
                           reconstruction_mode='full', device='cpu'):
    """
    构建多任务模型

    参数:
    window_size: 输入窗口大小
    input_size: 输入特征维度
    predict_steps: 预测步数 (1:普通预测, >1:长时延预测)
    reconstruction_mode: 重构模式 ('full', 'partial')
    device: 计算设备
    """
    model = MultiTaskLSTMModel(
        input_size=input_size,
        predict_steps=predict_steps,
        reconstruction_mode=reconstruction_mode
    )
    return model.to(device)


def train_multi_task_model(model, X_train, y_train, epochs=50, batch_size=32, device='cpu'):
    """
    训练多任务模型

    参数:
    model: 模型实例
    X_train: 训练输入 [samples, window_size]
    y_train: 训练标签 (预测目标) [samples]
    epochs: 训练轮数
    batch_size: 批大小
    device: 计算设备

    返回:
    训练好的模型
    """
    # 确保数据是3D: [samples, sequence_length, features]
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # 转换为张量
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion_pred = nn.MSELoss()  # 预测任务损失
    criterion_recon = nn.MSELoss()  # 重构任务损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_pred_loss = 0
        total_recon_loss = 0
        all_preds = []
        all_true = []

        for X_batch, y_batch in loader:
            if len(X_batch.shape) == 2:
                X_batch = X_batch.unsqueeze(-1)  # [batch, window] -> [batch, window, 1]

            optimizer.zero_grad()

            # 前向传播
            prediction, reconstruction = model(X_batch)

            # 计算预测任务损失
            pred_loss = criterion_pred(prediction.squeeze(), y_batch)

            # 计算重构任务损失
            if model.reconstruction_mode == 'full':
                # 整个序列的重构损失
                recon_loss = criterion_recon(reconstruction, X_batch.squeeze(-1))
            else:
                # 只计算后半部分的重构损失
                seq_len = X_batch.size(1)
                half_len = seq_len // 2
                recon_loss = criterion_recon(
                    reconstruction[:, half_len:],
                    X_batch[:, half_len:, 0]
                )

            # 组合损失
            loss = pred_loss + 0.5 * recon_loss  # 加权组合

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_recon_loss += recon_loss.item()

            # 收集预测结果用于指标计算
            all_preds.extend(prediction.squeeze().cpu().detach().numpy())
            all_true.extend(y_batch.cpu().detach().numpy())

        # 计算指标
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)

        print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(loader):.4f}, '
              f'Pred Loss: {total_pred_loss / len(loader):.4f}, '
              f'Recon Loss: {total_recon_loss / len(loader):.4f}, '
              f'MAE: {mae:.4f}, R²: {r2:.4f}')

    return model


def predict_next_step(model, X, device='cpu'):
    """
    普通预测 (X_{n+1})

    参数:
    model: 训练好的模型
    X: 输入序列 [samples, window_size]
    device: 计算设备

    返回:
    预测结果 [samples] (只预测下一步)
    """
    model.eval()
    with torch.no_grad():
        # 确保输入是3D
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        prediction, _ = model(X_tensor)
        # 只取每个样本的第一步预测
        next_step_pred = prediction[:, 0].cpu().numpy()
    return next_step_pred


def predict_long_term(model, initial_seq, steps=3, device='cpu'):
    """
    长时延预测 (X_{n+k})

    参数:
    model: 训练好的模型
    initial_seq: 初始序列 [1, window_size]
    steps: 预测步数
    device: 计算设备

    返回:
    预测序列 [steps]
    """
    model.eval()
    with torch.no_grad():
        # 确保初始序列是3D
        if len(initial_seq.shape) == 2:
            initial_seq = initial_seq.reshape(1, initial_seq.shape[1], 1)

        current_seq = torch.tensor(initial_seq, dtype=torch.float32).to(device)
        predictions = []

        for _ in range(steps):
            # 预测下一个点
            prediction, _ = model(current_seq)
            next_val = prediction.squeeze().cpu().numpy()[0]
            predictions.append(next_val)

            # 更新序列: 移除第一个点，添加新预测
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                torch.tensor([[[next_val]]], dtype=torch.float32).to(device)
            ], dim=1)

    return np.array(predictions)


def reconstruct_sequence(model, X, mode='full', device='cpu'):
    """
    重构序列

    参数:
    model: 训练好的模型
    X: 输入序列 [samples, window_size]
    mode: 重构模式 ('full', 'partial')
    device: 计算设备

    返回:
    重构序列 [samples, window_size] 或 [samples, window_size//2]
    """
    model.eval()
    with torch.no_grad():
        # 确保输入是3D
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        _, reconstruction = model(X_tensor)

        # 根据模式返回不同部分
        if mode == 'full':
            return reconstruction.cpu().numpy()
        elif mode == 'partial':
            seq_len = X.shape[1]
            half_len = seq_len // 2
            return reconstruction[:, half_len:].cpu().numpy()
        else:
            raise ValueError(f"未知重构模式: {mode}")