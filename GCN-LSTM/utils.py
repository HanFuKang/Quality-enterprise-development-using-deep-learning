import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def plot_real_vs_pred(true_values, predicted_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(true_values, label='GT', color='blue', marker='o', linestyle='-', linewidth=2)
    ax.plot(predicted_values, label='Pred', color='red', marker='x', linestyle='--', linewidth=2)

    ax.set_title('Real vs Pred', fontsize=16, pad=15)
    ax.set_xlabel('Sample', fontsize=14)
    ax.set_ylabel('Val', fontsize=14)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.legend(loc='best', fontsize=12)

    for i in range(len(true_values)):
        if i % 10 == 0:  # 每10个点添加一次注释
            try:
                ax.text(i, true_values[i], f'{true_values[i]:.2f}', ha='right', va='bottom', fontsize=9, color='blue')
                ax.text(i, predicted_values[i][0], f'{predicted_values[i][0]:.2f}', ha='left', va='top', fontsize=9, color='red')
            except :  # 如果预测值数组越界，则跳过该点
                ax.text(i, true_values[i], f'{true_values[i]:.2f}', ha='right', va='bottom', fontsize=9, color='blue')
                ax.text(i, predicted_values[i], f'{predicted_values[i]:.2f}', ha='left', va='top', fontsize=9, color='red')
    plt.tight_layout()
    plt.savefig('true_vs_predicted.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片



def BinsrPlot(true_values, predicted_values):
    # 计算误差
    errors = predicted_values - true_values

    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 创建图形对象
    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 6))
    # 绘制误差的直方图
    ax2.hist(errors, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Prediction Errors', fontsize=14)
    ax2.set_xlabel('Error Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)

    # 添加网格
    ax2.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    ax2.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # 美化图形
    plt.tight_layout()

    # 显示图形
    plt.show()


from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def ErrorPlot(true_values, predicted_values):
    # 计算相关系数和均方误差
    corr, _ = pearsonr(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(6, 6))
    # 绘制散点图
    ax.scatter(true_values, predicted_values, color='blue', alpha=0.6, label='Predictions',marker='o')
    # 绘制对角线
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # 设置对角线的起始位置
            np.max([ax.get_xlim(), ax.get_ylim()])]  # 设置对角线的结束位置
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0,color='pink',marker='*')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 添加标题和标签
    ax.set_title('Scatter Plot of True vs. Predicted Values', fontsize=16)
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)

    # 添加网格线
    ax.grid(True, which="both", ls="--", c='gray', alpha=0.5)

    # 添加统计信息
    stats_text = f'Pearson Correlation: {corr:.2f}\nMSE: {mse:.2f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # 添加图例
    ax.legend()

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()