import numpy as np

def inject_anomalies(data, anomaly_type='peak'):
    if anomaly_type == 'peak':
        idx = np.random.randint(0, len(data), size=10)
        data[idx] += data.max() * 0.5
    elif anomaly_type == 'uniform':
        start = np.random.randint(0, len(data)-100)
        data[start:start+100] = np.mean(data)
    return data

def calculate_rmsle(y_true, y_pred):
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt(np.mean(log_diff**2))