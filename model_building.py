import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskLSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1,  # 修改input_size为8
                 predict_steps=1, reconstruction_mode='full'):
        super(MultiTaskLSTMModel, self).__init__()
        self.predict_steps = predict_steps
        self.reconstruction_mode = reconstruction_mode
        self.input_size = input_size  # 保存输入尺寸

        # LSTM编码器 - 输入尺寸改为8
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

        # 重构任务解码器 - 输出尺寸改为8
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),  # 输出所有特征
        )

    def forward(self, x):
        # 编码器
        lstm_out, (hidden, cell) = self.encoder(x)

        # 预测任务 - 只预测OT值
        prediction = self.predict_decoder(hidden.squeeze(0))
        prediction = prediction.view(-1, self.predict_steps)

        # 重构任务 - 重构所有特征
        if self.reconstruction_mode == 'full':
            reconstruction = self.recon_decoder(lstm_out)
        elif self.reconstruction_mode == 'partial':
            seq_len = lstm_out.size(1)
            half_len = seq_len // 2
            recon_part = self.recon_decoder(lstm_out[:, half_len:, :])

            full_recon = torch.zeros(
                x.size(0), seq_len, self.input_size,
                device=x.device, dtype=x.dtype
            )
            full_recon[:, half_len:] = recon_part
            reconstruction = full_recon
        else:
            raise ValueError(f"未知重构模式: {self.reconstruction_mode}")

        return prediction, reconstruction


def build_multi_task_model(window_size, input_size=8, predict_steps=1,  # 默认输入尺寸改为8
                           reconstruction_mode='full', device='cpu'):
    """
    构建多任务模型

    参数:
    window_size: 输入窗口大小
    input_size: 输入特征维度 (现在为8)
    predict_steps: 预测步数
    reconstruction_mode: 重构模式
    device: 计算设备
    """
    model = MultiTaskLSTMModel(
        input_size=input_size,
        predict_steps=predict_steps,
        reconstruction_mode=reconstruction_mode
    )
    return model.to(device)


def train_multi_task_model(model, X_train, y_train, epochs=50, batch_size=32, device='cpu'):
    # 数据已经是3D: [samples, sequence_length, features]
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
            optimizer.zero_grad()

            # 前向传播
            prediction, reconstruction = model(X_batch)

            # 计算预测任务损失 - y_batch形状为 [batch_size, predict_steps]
            pred_loss = criterion_pred(prediction, y_batch)  # 注意：prediction形状应为 [batch_size, predict_steps]

            # 计算重构任务损失 - 现在重构所有特征
            if model.reconstruction_mode == 'full':
                recon_loss = criterion_recon(reconstruction, X_batch)
            else:
                seq_len = X_batch.size(1)
                half_len = seq_len // 2
                recon_loss = criterion_recon(
                    reconstruction[:, half_len:],
                    X_batch[:, half_len:]
                )
            # 组合损失 - 动态调整权重
            # 前期侧重重构，后期侧重预测
            if epoch < epochs // 2:
                loss = 0.4 * pred_loss + 0.6 * recon_loss
            else:
                loss = 0.7 * pred_loss + 0.3 * recon_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_recon_loss += recon_loss.item()

            # 收集预测结果用于指标计算
            all_preds.extend(prediction[:, 0].cpu().detach().numpy())  # 只取第一步预测
            all_true.extend(y_batch[:, 0].cpu().detach().numpy())

        # 计算指标
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)

        print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(loader):.4f}, '
              f'Pred Loss: {total_pred_loss / len(loader):.4f}, '
              f'Recon Loss: {total_recon_loss / len(loader):.4f}, '
              f'MAE: {mae:.4f}, R²: {r2:.4f}')

    return model


def predict_next_step(model, X, device='device'):
    """
    普通预测 (X_{n+1})

    参数:
    model: 训练好的模型
    X: 输入序列 [samples, window_size, features]
    device: 计算设备

    返回:
    预测结果 [samples] (只预测下一步的OT值)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        prediction, _ = model(X_tensor)
        # 只取每个样本的第一步预测
        next_step_pred = prediction[:, 0].cpu().numpy()
    return next_step_pred


def predict_long_term(model, initial_seq, steps=3, device='device'):
    """
    长时延预测 (X_{n+k})

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
            prediction, _ = model(current_seq)
            next_val = prediction[0, 0].item()  # 取第一个样本的第一步预测

            # 创建新点 - 只有OT值，其他特征未知
            # 这里简单假设其他特征不变，但实际应用中可能需要更好的处理
            new_point = np.zeros((1, 1, model.input_size))
            new_point[0, 0, -1] = next_val  # 假设OT是最后一个特征

            # 更新序列: 移除第一个点，添加新预测
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                torch.tensor(new_point, dtype=torch.float32).to(device)
            ], dim=1)

            predictions.append(next_val)

    return np.array(predictions)


def reconstruct_sequence(model, X, mode='full', device='device'):
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