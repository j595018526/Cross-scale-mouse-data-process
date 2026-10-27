import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 配置参数
file_path = r'./brain_data/Box_features_m3.xlsx'
gene_name = 'Dlg4'  # 需要分析的基因列

# 读取数据
df = pd.read_excel(file_path, sheet_name='features')
mri_col = df['T2 value'].values
cell_col = df['All cells'].values
gene_col = df[gene_name].values
types = df['type'].values

# 颜色映射
color_map = {'hip': 'red', 'tha': 'yellow', 'white': 'green'}
colors = [color_map[t] for t in types]

# 计算Spearman相关系数
print("=" * 50)
print("Spearman Correlation Results:")
r1, p1 = spearmanr(mri_col, cell_col)
print(f"MRI vs cell: r = {r1:.2f}, p = {p1:.4f}")
r3, p3 = spearmanr(cell_col, gene_col)
print(f"cell vs {gene_name}: r = {r3:.2f}, p = {p3:.4f}")
r2, p2 = spearmanr(mri_col, gene_col)
print(f"MRI vs {gene_name}: r = {r2:.2f}, p = {p2:.4f}")

# 归一化（仅用于绘图，Spearman不需要归一化）
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

mri_norm = normalize(mri_col)
cell_norm = normalize(cell_col)
gene_norm = normalize(gene_col)

######################################## 绘制相关性散点图 #########################################
# 图1：T2 value vs All cells（独立绘制）
fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4.5))
ax1.scatter(mri_norm, cell_norm, c=colors, alpha=0.7, s=28, edgecolors='k', linewidth=0.5)
z = np.polyfit(mri_norm, cell_norm, 1)
ax1.plot(mri_norm, np.poly1d(z)(mri_norm), 'k-', linewidth=0.5, label='Trend line')
ax1.set_title(f'T2 value——All cells: r = {r1:.2f}')
ax1.set_xlabel('T2 value')
ax1.set_ylabel('All cells')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图2：T2 value vs gene 和 All cells vs gene（一行两列）
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
pairs = [(mri_norm, gene_norm, 'T2 value', gene_name, r2),
         (cell_norm, gene_norm, 'All cells', gene_name, r3)]

for ax, (x, y, xlabel, ylabel, r) in zip(axes2, pairs):
    ax.scatter(x, y, c=colors, alpha=0.7, s=28, edgecolors='k', linewidth=0.5)
    z = np.polyfit(x, y, 1)
    ax.plot(x, np.poly1d(z)(x), 'k-', linewidth=0.5, label='Trend line')
    ax.set_title(f'{xlabel}——{ylabel}: r = {r:.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

########################## 绘制空间趋势分布图 ##########################
# 图3：T2 value 和 All cells 的空间分布（一行两列）
boxes = np.arange(len(mri_norm))
x_smooth = np.linspace(boxes.min(), boxes.max(), 200)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.5))
spatial_data1 = [(mri_norm, 'T2 value', 'Spatial Variation of T2 value'),
                 (cell_norm, 'All cells', 'Spatial Variation of All cells')]

for ax, (data, ylabel, title) in zip(axes3, spatial_data1):
    ax.scatter(boxes, data, c=colors, alpha=0.7, s=28, edgecolors='k', linewidth=0.5)
    z = np.polyfit(boxes, data, 3)
    ax.plot(x_smooth, np.poly1d(z)(x_smooth), 'r-', linewidth=2, label='Polynomial fit')
    ax.set_xlabel('Region')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图4：Gene 空间分布（单独绘制）
fig4, ax4 = plt.subplots(1, 1, figsize=(5, 4.5))
ax4.scatter(boxes, gene_norm, c=colors, alpha=0.7, s=28, edgecolors='k', linewidth=0.5)
z = np.polyfit(boxes, gene_norm, 3)
ax4.plot(x_smooth, np.poly1d(z)(x_smooth), 'r-', linewidth=2, label='Polynomial fit')
ax4.set_xlabel('Region')
ax4.set_ylabel(f'Gene: {gene_name}')
ax4.set_title(f'Spatial Variation of {gene_name}')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
