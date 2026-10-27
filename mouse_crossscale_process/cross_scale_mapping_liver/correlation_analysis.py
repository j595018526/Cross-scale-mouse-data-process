import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 读取数据
file_path = r'D:\PythonCode\mouse_process\cross_scale_mapping\tumor_8_liver_features.xlsx'
mri_data = pd.read_excel(file_path, sheet_name='MRI')
cell_data = pd.read_excel(file_path, sheet_name='cell_count')
expression_data = pd.read_excel(file_path, sheet_name='total_expression')

# 提取列并归一化
cell_name = 'HSC_Activated_ECM'
gene_name = 'mm10___Runx1'

std_col = mri_data['std'].values
cell_col = cell_data[cell_name].values
gene_col = expression_data[gene_name].values
types = expression_data['type'].values

std_norm = (std_col - std_col.min()) / (std_col.max() - std_col.min())
cell_norm = (cell_col - cell_col.min()) / (cell_col.max() - cell_col.min())
gene_norm = (gene_col - gene_col.min()) / (gene_col.max() - gene_col.min())

# 颜色映射
color_map = {'tumor_edge': 'red', 'tumor_interior': 'yellow', 'normal': 'green'}
colors = [color_map[t] for t in types]

# 计算Spearman相关系数
print("=" * 50)
print(f"Spearman Correlation Results:")

r1, p1 = spearmanr(std_norm, cell_norm)
print(f"MRI std vs {cell_name}: r = {r1:.2f}, p = {p1:.3f}")

r2, p2 = spearmanr(std_norm, gene_norm)
print(f"MRI std vs {gene_name}: r = {r2:.2f}, p = {p2:.3f}")

r3, p3 = spearmanr(cell_norm, gene_norm)
print(f"{cell_name} vs {gene_name}: r = {r3:.2f}, p = {p3:.3f}")

# 绘制散点图（一行三列）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 定义绘图函数
def plot_scatter(ax, x, y, colors, xlabel, ylabel, title, r_value):
    ax.scatter(x, y, c=colors, alpha=0.9, s=28, edgecolors='k', linewidth=0.5)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "k", linewidth=0.5, label='Trend line')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (r = {r_value:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 子图1: MRI std vs cell
plot_scatter(axes[0], std_norm, cell_norm, colors, 'MRI index', f'{cell_name} cell count', f'MRI vs {cell_name}', r1)

# 子图2: cell vs gene
plot_scatter(axes[1], cell_norm, gene_norm, colors, f'{cell_name} cell count', f'{gene_name} gene expression', f'{cell_name} vs {gene_name}', r3)

# 子图3: MRI std vs gene
plot_scatter(axes[2], std_norm, gene_norm, colors, 'MRI index', f'{gene_name} gene expression', f'MRI vs {gene_name}', r2)

plt.tight_layout()
plt.show()

# ========== 绘制MRI、cell和gene随着box个数的变化情况 ==========
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

boxes = np.arange(len(std_norm))  # 创建box序号
x_smooth = np.linspace(boxes.min(), boxes.max(), 200)  # 用于平滑曲线的x轴

# 子图1：MRI std的空间分布及拟合曲线
ax1 = axes2[0]
ax1.scatter(boxes, std_norm, c=colors, s=28, alpha=0.7, edgecolors='k', linewidth=0.5)
# 多项式拟合曲线（3次多项式）
z_std = np.polyfit(boxes, std_norm, 3)
p_std = np.poly1d(z_std)
ax1.plot(x_smooth, p_std(x_smooth), "r-", linewidth=2, label='Polynomial fit')
ax1.set_xlabel('Region')
ax1.set_ylabel('MRI index')
ax1.set_title('Spatial Variation of MRI index')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：cell count的空间分布及拟合曲线
ax2 = axes2[1]
ax2.scatter(boxes, cell_norm, c=colors, s=28, alpha=0.7, edgecolors='k', linewidth=0.5)
# 多项式拟合曲线（3次多项式）
z_cell = np.polyfit(boxes, cell_norm, 3)
p_cell = np.poly1d(z_cell)
ax2.plot(x_smooth, p_cell(x_smooth), "r-", linewidth=2, label='Polynomial fit')
ax2.set_xlabel('Region')
ax2.set_ylabel(f'{cell_name} cell count')
ax2.set_title(f'Spatial Variation of {cell_name}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：gene expression的空间分布及拟合曲线
ax3 = axes2[2]
ax3.scatter(boxes, gene_norm, c=colors, s=28, alpha=0.7, edgecolors='k', linewidth=0.5)
# 多项式拟合曲线（3次多项式）
z_gene = np.polyfit(boxes, gene_norm, 3)
p_gene = np.poly1d(z_gene)
ax3.plot(x_smooth, p_gene(x_smooth), "r-", linewidth=2, label='Polynomial fit')
ax3.set_xlabel('Region')
ax3.set_ylabel(f'{gene_name} gene expression')
ax3.set_title(f'Spatial Variation of {gene_name}')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
