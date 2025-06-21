import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x的形状应为 [batch_size, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 取序列最后一个时间步
        return self.linear(last_out)


def build_lstm_model(window_size, input_size=1, device='device'):
    model = LSTMModel(input_size=input_size)
    return model.to(device)


def train_model(model, X_train, y_train, epochs=50, batch_size=32, device='device'):
    # 确保数据是3D: [samples, sequence_length, features]
    if len(X_train.shape) == 2:
        # 添加特征维度: [samples, window_size] -> [samples, window_size, 1]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # 转换为张量
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_true = []

        for X_batch, y_batch in loader:
            if len(X_batch.shape) == 2:
                X_batch = X_batch.unsqueeze(-1)  # [batch, window] -> [batch, window, 1]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().detach().numpy())
            all_true.extend(y_batch.cpu().detach().numpy())

        # 计算指标
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

    return model


def predict(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        # 确保输入是3D
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor)
    return preds.squeeze().cpu().numpy()