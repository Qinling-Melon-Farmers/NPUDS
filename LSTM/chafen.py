import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_individual_series(series, title, ylabel, save_name):
    """
    绘制单个序列的独立图像
    :param series: 要绘制的时间序列
    :param title: 图像标题
    :param ylabel: y轴标签
    :param save_name: 保存文件名
    """
    plt.figure(figsize=(12, 6))  # 新建 figure，避免复用
    plt.plot(series, label=ylabel)
    plt.title(title)
    plt.xlabel('时间点')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_name}.png")
    plt.close()  # 关闭当前图像，防止阻塞后续绘图


def plot_oil_temperature_difference(file_path):
    try:
        # 1. 加载数据
        df = pd.read_csv(file_path)

        # 2. 数据清洗
        df.ffill(inplace=True)  # 使用推荐方法替换 fillna(method='ffill')
        df.drop_duplicates(inplace=True)

        # 3. 提取油温序列
        if 'OT' not in df.columns:
            raise ValueError("CSV文件中未找到'OT'列，请检查列名是否正确")

        ot_series = df['OT'].values.reshape(-1, 1)

        # 4. 归一化
        scaler = MinMaxScaler()
        scaled_ot = scaler.fit_transform(ot_series).flatten()

        # 5. 计算一阶差分
        diff_series = np.diff(scaled_ot, n=1)

        # 确保差分序列为正数用于 Box-Cox 变换
        diff_series_positive = diff_series - np.min(diff_series) + 1e-5
        diff_series_bc, lambda_val = boxcox(diff_series_positive)

        # 打印关键变量长度，确认是否为空
        print("scaled_ot length:", len(scaled_ot))
        print("diff_series length:", len(diff_series))
        print("diff_series_bc length:", len(diff_series_bc))

        # 6. 单独绘图
        plot_individual_series(scaled_ot, '归一化原始油温序列', '归一化油温值', 'scaled_oil_temperature')
        plot_individual_series(diff_series_bc, f'Box-Cox变换后的油温变化序列 (λ={lambda_val:.4f})', '变换后差分值', 'boxcox_oil_temperature')
        plot_individual_series(diff_series, '一阶差分后的油温变化序列', '差分值 ΔT', 'diff_oil_temperature')

        # 7. 计算并打印统计信息
        print("\n油温序列统计信息:")
        print(f"原始序列长度: {len(scaled_ot)}")
        print(f"差分序列长度: {len(diff_series)}")
        print(f"原始序列均值: {np.mean(scaled_ot):.4f}")
        print(f"差分序列均值: {np.mean(diff_series):.4f}")
        print(f"原始序列标准差: {np.std(scaled_ot):.4f}")
        print(f"差分序列标准差: {np.std(diff_series):.4f}")
        print(f"原始序列最小值: {np.min(scaled_ot):.4f}, 最大值: {np.max(scaled_ot):.4f}")
        print(f"差分序列最小值: {np.min(diff_series):.4f}, 最大值: {np.max(diff_series):.4f}")
        print(f"差分序列中绝对值大于0.1的比例: {np.sum(np.abs(diff_series) > 0.1) / len(diff_series):.2%}")
        print(f"Box-Cox变换参数 λ: {lambda_val:.4f}")
        print(f"变换后序列均值: {np.mean(diff_series_bc):.4f}")
        print(f"变换后序列标准差: {np.std(diff_series_bc):.4f}")

    except Exception as e:
        print("发生异常：", e)


if __name__ == "__main__":
    DATA_PATH = r'C:\Users\ASUS\Desktop\数据科学的数学方法\大作业\Project1-LSTM\ETTdata\ETTm1.csv'
    plot_oil_temperature_difference(DATA_PATH)