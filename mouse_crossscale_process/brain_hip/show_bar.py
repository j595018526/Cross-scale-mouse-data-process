import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 数据
groups = ['Blank', 'Tumor-1', 'Tumor-2']


# # MRI indicators
# genes = ['T2 value', 'rADC', 'Mean', 'Energy']
# data = {
#     'T2 value': [215.7043, 158.6301, 97.2105],
#     'rADC': [75.492302, 16.720889, -4.029737],
#     'Mean': [3656.353027, 1615.501465, 1380.018066],
#     'Energy': [62728368128, 37502652416, 39852593152]
# }
# # 最精简方式：对每个指标归一化到[-1,1]
# for key in data:
#     arr = np.array(data[key])
#     data[key] = (2 * (arr - arr.min()) / (arr.max() - arr.min()) - 1).tolist()


# # # HE indicators
# genes = ['CPECs', 'Endothelial', 'Astrocyte_Resting', 'Microglia_Homeostatic']
# data = {
#     'CPECs': [0.33, -0.44, -0.2],
#     'Endothelial': [0.09, -1.02, -2.18],
#     'Astrocyte_Resting': [-0.7, -1.92, -4.6],
#     'Microglia_Homeostatic': [-0.72, -0.49, -1.59],
# }


# ST indicators（四种细胞类型分别对应相应基因）
# genes = ['Ttr', 'Atp1a1']  # CPECs
# data = {
#     'Ttr': [-0.03, 0.01, 0.03],
#     'Atp1a1': [0.68, 0.74, 1.29]
# }

# genes = ['Cldn5', 'H2-D1', 'Cdh5', 'Ctnnb1', 'Flt1', 'H2-K1', 'B2m', 'Pecam1',]  # Endothelial
# data = {
#     'Cldn5': [-0.01, 0.85, 0.62],
#     'H2-D1': [0.25, 0.99, 1.49],
#     'Cdh5': [0.67, 1.46, 1],
#     'Ctnnb1': [0.26, 0.66, 0.27],
#     'Flt1': [-0.15, -0.03, 0.1],
#     'H2-K1': [-0.07, 0.46, 0.9],
#     'B2m': [-0.02, 0.61, 0.6],
#     'Pecam1': [0.19, 0.45, 0.21]
#     }
#
# genes = ['Cdh4', 'Kcnj10', 'Grm3', 'Slc1a2', 'Fasn', 'Sorcs2', 'Gfap', 'Slco1c1']  # Astrocyte_Resting
# data = {
#     'Cdh4': [0.11, 0.43, 1.58],
#     'Kcnj10': [0.39, 0.58, 1.04],
#     'Grm3': [-0.59, -0.15, 0.96],
#     'Slc1a2': [-0.12, 0.08, 0.07],
#     'Fasn': [0.53, 0.63, 1.17],
#     'Sorcs2': [0.02, 0.35, 1.62],
#     'Gfap': [0.04, 0.07, 0.26],
#     'Slco1c1': [-0.13, 0.34, 0.53]
# }
#
genes = ['Smad3', 'Tgfbr2', 'Lrp1', 'Apoe', 'Spred2', 'Elmo1']  # Microglia_Homeostatic
data = {
    'Smad3': [0.13, 0.14, 0.68],
    'Tgfbr2': [0.11, 0.43, 1.21],
    'Lrp1': [0.32, 0.96, 0.79],
    'Apoe': [0.59, 1.36, 1.49],
    'Spred2': [0.31, 0.42, 1.48],
    'Elmo1': [-0.16, -0.2, -0.32]
}



# 对每个组（样本）的数据进行标准化
# 首先将数据转换为二维数组：行为基因，列为组
data_matrix = np.array([data[gene] for gene in genes])  # shape: (n_genes, n_groups)

# 对每个组（列）进行标准化
# 方法1：使用 MinMaxScaler 归一化到 [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized_by_group = scaler.fit_transform(data_matrix.T).T  # 转置后标准化再转置回来

# 方法2：使用 StandardScaler 标准化（均值为0，标准差为1）
# scaler = StandardScaler()
# data_normalized_by_group = scaler.fit_transform(data_matrix.T).T

# 将标准化后的数据存回字典
data_normalized = {}
for i, gene in enumerate(genes):
    data_normalized[gene] = data_normalized_by_group[i]

# 自动计算动态参数
num_genes = len(genes)
width = 0.25
x = np.arange(num_genes)

# 动态计算图形尺寸
fig_width = max(8, num_genes * 0.8)
fig, ax = plt.subplots(figsize=(fig_width, 6))

# 莫兰迪色系
colors = ['#6A9FB5', '#E8B86D', '#D4A05A']

# 确定使用哪个数据集进行绘图
# 如果想使用标准化后的数据，取消下面这行的注释
plot_data = data_normalized
# 如果想使用原始数据，使用下面这行
plot_data = data

# 为每个组画一组柱子
for i, group in enumerate(groups):
    values = [plot_data[gene][i] for gene in genes]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=group,
                  color=colors[i], edgecolor='white', linewidth=0.8)

    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        y_offset = abs(height) * 0.05 if height != 0 else 0.05
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2., height - y_offset,
                    f'{height:.2f}', ha='center', va='top', fontsize=8)



# 设置标签和标题
ax.set_ylabel('Cell: Microglia_Homeostatic', fontsize=18)
# 设置x轴刻度
ax.set_xticks(x)
ax.set_xticklabels(genes, fontsize=18, rotation=45, ha='right' if num_genes > 6 else 'center')

# 设置y轴范围
y_min = min(min(plot_data[gene]) for gene in genes)
y_max = max(max(plot_data[gene]) for gene in genes)
y_margin = (y_max - y_min) * 0.1
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# 其他设置
# ax.legend(loc='upper left', fontsize=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
